import logging
import inspect
import time
from collections import deque

import ray
import torch
from transformers import AutoConfig

from .payloads import (
    normalize_text_backward_outputs,
    normalize_vision_outputs,
    TextBackwardOutputs,
    VisionOutputs,
)
from .tensor_transfer import TensorTransferRequest, receive_tensor
from .trainer import Trainer
from .utils import get_physical_gpu_id

logger = logging.getLogger(__name__)


class MegatronBaseTrainer(Trainer):
    """Shared Megatron initialization helpers for Ray trainers."""

    def __init__(self, config, rank: int, **kwargs):
        super().__init__(config, rank, **kwargs)
        self.megatron_model = None
        self.megatron_model_meta = None
        self.megatron_args = None
        self.megatron_bridge = None
        self._megatron_initialized = False
        self.receiver_gpu_ids = None
        self.use_ipc = False
        self._weights_load_requested = False
        self._weights_loaded = False
        self._weights_load_path = None

    def _initialize_megatron(self):
        if self._megatron_initialized:
            return

        self._get_backend(component_name="megatron")
        self._get_device()

        from megatron.training import initialize_megatron
        from swift.megatron.argument import MegatronArguments
        from swift.megatron.model import get_megatron_model_meta
        from swift.megatron.utils import convert_hf_config

        model_name = self.config["model_name"]
        model_type = self.config["model_type"]
        engine_config = self.config.get("engine_config", {})

        megatron_model_meta = get_megatron_model_meta(model_type)
        if megatron_model_meta is None:
            raise ValueError(f"Megatron model_type '{model_type}' is not registered in ms-swift.")

        hf_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        megatron_kwargs = convert_hf_config(hf_config)
        megatron_overrides = {
            key.replace("megatron_", "", 1): value
            for key, value in engine_config.items()
            if key.startswith("megatron_") and value is not None
        }
        if megatron_overrides:
            logger.info(f"[r{self.rank}] Overriding Megatron args from engine_config: {megatron_overrides}")
            megatron_kwargs.update(megatron_overrides)

        tp_size = int(engine_config.get("tensor_parallel_size", 1))
        sp_size = int(engine_config.get("sequence_parallel_size", 1))
        pp_size = int(engine_config.get("pipeline_model_parallel_size", 1))
        ep_size = int(engine_config.get("expert_model_parallel_size", 1))
        num_experts = engine_config.get("num_experts", None)

        torch_dtype = self._get_torch_dtype(self.config["dtype"])
        convert_kwargs = {
            "use_cpu_initialization": bool(engine_config.get("use_cpu_initialization", True)),
            "no_save_optim": True,
            "no_save_rng": True,
            "no_load_optim": True,
            "no_load_rng": True,
            "finetune": True,
            "attention_backend": engine_config.get("attention_backend", "unfused"),
            "tensor_model_parallel_size": tp_size,
            "pipeline_model_parallel_size": pp_size,
            # sequence_parallel (boolean) enables SP within TP group for activations
            # Required for MoE + TP. Enable when TP > 1.
            "sequence_parallel": tp_size > 1,
            # context_parallel_size controls sequence splitting across devices
            "context_parallel_size": sp_size,
            "expert_model_parallel_size": ep_size,
        }
        if num_experts is not None:
            resolved_num_experts = int(num_experts)
            if "num_experts" in megatron_kwargs:
                if int(megatron_kwargs["num_experts"]) != resolved_num_experts:
                    logger.info(
                        f"[r{self.rank}] Overriding converted HF num_experts="
                        f"{megatron_kwargs['num_experts']} with engine_config value {resolved_num_experts}"
                    )
                megatron_kwargs["num_experts"] = resolved_num_experts
            else:
                convert_kwargs["num_experts"] = resolved_num_experts

        supported_args = set(inspect.signature(MegatronArguments).parameters)
        combined_kwargs = {**megatron_kwargs, **convert_kwargs}
        dropped_kwargs = {
            key: value for key, value in combined_kwargs.items() if key not in supported_args
        }
        if dropped_kwargs:
            logger.warning(
                f"[r{self.rank}] Dropping unsupported MegatronArguments kwargs: {sorted(dropped_kwargs)}"
            )
        megatron_kwargs = {key: value for key, value in megatron_kwargs.items() if key in supported_args}
        convert_kwargs = {key: value for key, value in convert_kwargs.items() if key in supported_args}

        megatron_args = MegatronArguments(
            model=model_name,
            model_type=model_type,
            torch_dtype=torch_dtype,
            **megatron_kwargs,
            **convert_kwargs,
        )
        extra_args = megatron_args.parse_to_megatron()
        initialize_megatron(extra_args_provider=megatron_model_meta.extra_args_provider, args_defaults=extra_args)

        self.megatron_model_meta = megatron_model_meta
        self.megatron_args = megatron_args
        self._megatron_initialized = True

    def _build_megatron_model(self):
        if self.megatron_model is not None:
            return

        self._initialize_megatron()
        self.megatron_model = self.megatron_model_meta.model_provider(pre_process=True, post_process=True)
        self.megatron_bridge = self.megatron_model_meta.bridge_cls()

        load_path = self._get_engine_config_value("bridge_load_path", self.config["model_name"])
        load_weights = self._get_engine_config_value("load_weights", True)
        self._weights_load_path = load_path
        self._weights_load_requested = bool(load_weights)
        self._weights_loaded = False
        if load_weights:
            logger.info(f"[r{self.rank}] Loading Megatron weights from {load_path}.")
            self.megatron_bridge.load_weights(self.megatron_model, load_path)
            self._weights_loaded = True
        else:
            logger.warning(f"[r{self.rank}] Skipping Megatron weight load (engine_config.load_weights=false).")

        # Ensure model parameters are on the active CUDA device for test forwards.
        device = self._get_device()
        self.megatron_model.to(device)
        if self.optimizer is None:
            self._build_optimizer(self.megatron_model.parameters())
            total_steps = int(self.config.get("num_epochs", 1)) * int(self.config.get("num_iterations", 1))
            self._build_scheduler(max(1, total_steps))

    def is_process_group_initialized(self):
        import torch.distributed as dist

        return dist.is_initialized()

    def set_receiver_info(self, receiver_gpu_ids: list[str], use_ipc: bool):
        """Set receiver GPU IDs and whether to use CUDA IPC."""
        self.receiver_gpu_ids = receiver_gpu_ids
        self.use_ipc = use_ipc
        logger.debug(
            f"[r{self.rank}] {self.__class__.__name__}: receiver_gpu_ids={receiver_gpu_ids}, use_ipc={use_ipc}"
        )

    def get_megatron_num_layers(self) -> int:
        """Expose Megatron num_layers for tests."""
        self._initialize_megatron()
        return int(getattr(self.megatron_args, "num_layers", 0))

    def get_weight_load_status(self) -> dict:
        """Expose weight loading status for tests/verification."""
        import os

        return {
            "requested": self._weights_load_requested,
            "loaded": self._weights_loaded,
            "path": self._weights_load_path,
            "path_exists": bool(self._weights_load_path and os.path.exists(self._weights_load_path)),
        }

    def get_runtime_metadata(self) -> dict:
        """Return placement, process-group, and Megatron runtime metadata."""
        import os
        import torch.distributed as dist

        initialized = dist.is_initialized()
        return {
            "rank": self.rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "physical_gpu_id": get_physical_gpu_id(),
            "process_group": {
                "initialized": initialized,
                "world_size": dist.get_world_size() if initialized else 1,
                "rank": dist.get_rank() if initialized else self.rank,
                "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            },
            "megatron": {
                "tp": int(getattr(self.megatron_args, "tensor_model_parallel_size", 1) or 1),
                "cp": int(getattr(self.megatron_args, "context_parallel_size", 1) or 1),
                "pp": int(getattr(self.megatron_args, "pipeline_model_parallel_size", 1) or 1),
                "ep": int(getattr(self.megatron_args, "expert_model_parallel_size", 1) or 1),
            },
            "optimizer": {
                "present": self.optimizer is not None,
                "step_count": self._optimizer_step_count,
                "scheduler_present": hasattr(self, "scheduler") and self.scheduler is not None,
            },
            "weights": self.get_weight_load_status(),
        }

    def get_effective_layer_counts(self) -> dict:
        """Return effective layer counts visible to the instantiated Megatron args."""
        self._initialize_megatron()
        return {
            "megatron_num_layers": int(getattr(self.megatron_args, "num_layers", 0) or 0),
            "vision_effective_layers": int(getattr(self.megatron_args, "vision_num_layers", 0) or 0) or None,
            "language_effective_layers": int(getattr(self.megatron_args, "num_layers", 0) or 0),
        }

    def get_optimizer_probe_snapshot(self, parameter_path: str | None = None) -> dict:
        """Snapshot one validated parameter for optimizer-update verification."""
        if parameter_path is None:
            raise ValueError("parameter_path is required; runtime probe selection is not allowed")
        name, param = self._resolve_probe_parameter(parameter_path)
        grad_norm = None
        has_nonzero_grad = False
        if param.grad is not None:
            grad_norm = float(param.grad.detach().float().norm().item())
            has_nonzero_grad = grad_norm > 0.0
        return {
            "parameter_path": parameter_path,
            "resolved_name": name,
            "param_l2_norm": float(param.detach().float().norm().item()),
            "grad_l2_norm": grad_norm,
            "has_nonzero_grad": has_nonzero_grad,
            "optimizer_step_count": self._optimizer_step_count,
        }

    def verify_optimizer_update(self, before: dict, *, atol: float = 1e-6) -> dict:
        """Compare the current probe parameter state against a previous snapshot."""
        after = self.get_optimizer_probe_snapshot(before.get("parameter_path"))
        norm_delta = abs(after["param_l2_norm"] - float(before["param_l2_norm"]))
        expected_step = int(before["optimizer_step_count"]) + 1
        actual_step = int(after["optimizer_step_count"])
        return {
            "before": before,
            "after": after,
            "param_norm_delta": norm_delta,
            "param_norm_changed": norm_delta > atol,
            "expected_optimizer_step_count": expected_step,
            "actual_optimizer_step_count": actual_step,
            "iteration_step_counter_advanced": actual_step == expected_step,
            "optimizer_update_verified": norm_delta > atol and actual_step == expected_step,
        }

    def _resolve_probe_parameter(self, parameter_path: str):
        if self.megatron_model is None:
            raise RuntimeError("Megatron model is not built")

        normalized = parameter_path
        for prefix in ("self.", "megatron_model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        named_params = dict(self.megatron_model.named_parameters())
        if normalized in named_params:
            return normalized, named_params[normalized]

        raise KeyError(f"Parameter path '{parameter_path}' was not found on megatron_model")


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=6)
class MegatronVisionTrainer(MegatronBaseTrainer):
    """Megatron-backed vision trainer using ms-swift Mcore-Bridge."""

    def __init__(self, config, rank: int, **kwargs):
        super().__init__(config, rank, **kwargs)
        self._pending_outputs: deque[torch.Tensor] = deque()
        self._dummy_batch = None

    def _build_dummy_vision_batch(self, device: torch.device):
        visual_module = self._get_visual_module()
        patch_embed = getattr(visual_module, "patch_embed", None)
        if patch_embed is None:
            raise RuntimeError("Megatron vision module missing patch_embed; cannot build dummy batch.")

        in_channels = int(getattr(patch_embed, "in_channels", 3))
        patch_size = int(getattr(patch_embed, "patch_size", 14))
        temporal_patch = int(getattr(patch_embed, "temporal_patch_size", 2))

        grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long, device=device)
        grid_t, grid_h, grid_w = (int(v) for v in grid_thw[0].tolist())

        expected_frames = temporal_patch * grid_t
        expected_h = patch_size * grid_h
        expected_w = patch_size * grid_w

        dtype = patch_embed.proj.weight.dtype if hasattr(patch_embed, "proj") else torch.float32
        pixel_values = torch.zeros(
            1,
            in_channels,
            expected_frames,
            expected_h,
            expected_w,
            device=device,
            dtype=dtype,
        )

        return {"pixel_values": pixel_values, "image_grid_thw": grid_thw}

    def _validate_dummy_vision_batch(self, batch):
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]
        if image_grid_thw.dim() != 2 or image_grid_thw.shape[-1] != 3:
            raise RuntimeError(f"Invalid image_grid_thw shape: {image_grid_thw.shape}")

        visual_module = self._get_visual_module()
        patch_embed = getattr(visual_module, "patch_embed", None)
        if patch_embed is None:
            raise RuntimeError("Megatron vision module missing patch_embed; cannot validate dummy batch.")

        in_channels = int(getattr(patch_embed, "in_channels", 3))
        patch_size = int(getattr(patch_embed, "patch_size", 14))
        temporal_patch = int(getattr(patch_embed, "temporal_patch_size", 2))
        grid_t, grid_h, grid_w = (int(v) for v in image_grid_thw[0].tolist())

        expected_frames = temporal_patch * grid_t
        expected_h = patch_size * grid_h
        expected_w = patch_size * grid_w

        expected_shape = (1, in_channels, expected_frames, expected_h, expected_w)
        if tuple(pixel_values.shape) != expected_shape:
            raise RuntimeError(
                f"Dummy pixel_values shape {tuple(pixel_values.shape)} does not match "
                f"expected {expected_shape} from grid_thw={image_grid_thw.tolist()} "
                f"and patch_embed (c={in_channels}, t={temporal_patch}, p={patch_size})."
            )

    def build_model(self):
        self._build_megatron_model()
        self.megatron_model.train()

    def initialize_trainer(self):
        device = self._get_device()
        self._dummy_batch = self._build_dummy_vision_batch(device)
        self._validate_dummy_vision_batch(self._dummy_batch)

    def _get_visual_module(self):
        if self.megatron_model is None or self.megatron_model.visual is None:
            raise RuntimeError("Megatron visual module is not available for this model.")
        if hasattr(self.megatron_model.visual, "visual"):
            return self.megatron_model.visual.visual
        return self.megatron_model.visual

    def _extract_vision_embeddings(self, outputs):
        """Normalize vision outputs to a single tensor for text input."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (list, tuple)):
            for item in outputs:
                if isinstance(item, torch.Tensor):
                    return item
            raise RuntimeError("Vision outputs list/tuple did not contain a tensor.")
        if isinstance(outputs, dict):
            for key in ("last_hidden_state", "hidden_states", "vision_embeddings", "embeddings"):
                value = outputs.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            for value in outputs.values():
                if isinstance(value, torch.Tensor):
                    return value
            raise RuntimeError("Vision outputs dict did not contain a tensor.")

        for attr in ("last_hidden_state", "vision_embeddings", "embeddings"):
            value = getattr(outputs, attr, None)
            if isinstance(value, torch.Tensor):
                return value
        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(hidden_states, torch.Tensor):
            return hidden_states
        if isinstance(hidden_states, (list, tuple)) and hidden_states:
            last_hidden = hidden_states[-1]
            if isinstance(last_hidden, torch.Tensor):
                return last_hidden

        raise RuntimeError(f"Unsupported vision outputs type: {type(outputs)}")

    def forward_step(self, iteration: int = -1):
        batch = self._dummy_batch
        pixel_values = batch["pixel_values"]
        image_grid_thw = batch["image_grid_thw"]

        autocast_context = self._get_autocast_context()
        visual_module = self._get_visual_module()
        with autocast_context:
            outputs = visual_module(hidden_states=pixel_values, grid_thw=image_grid_thw)
        embeddings = self._extract_vision_embeddings(outputs)

        self._pending_outputs.append(embeddings)
        return VisionOutputs(embeddings=embeddings, meta={"iteration": iteration})

    def _retrieve_gradient_tensor(self, vision_grad_ref):
        if vision_grad_ref is None:
            return None

        if isinstance(vision_grad_ref, ray.ObjectRef):
            vision_grad_data = ray.get(vision_grad_ref)
        else:
            vision_grad_data = vision_grad_ref

        if isinstance(vision_grad_data, (TextBackwardOutputs, dict)):
            normalized = normalize_text_backward_outputs(vision_grad_data)
            vision_grad_data = normalized.grad

        if vision_grad_data is None:
            return None

        if isinstance(vision_grad_data, dict) and "use_ipc" in vision_grad_data:
            transfer_request = TensorTransferRequest.from_dict(vision_grad_data)
            receiver_gpu_id = get_physical_gpu_id()
            vision_grad = receive_tensor(transfer_request, receiver_gpu_id)
        else:
            vision_grad = vision_grad_data

        return vision_grad

    def _apply_vision_backward(self, vision_grad: torch.Tensor | None):
        if vision_grad is None:
            raise ValueError(f"[r{self.rank}] No gradient provided for backward pass")

        if not self._pending_outputs:
            raise RuntimeError("No pending vision outputs for backward.")
        outputs = self._pending_outputs.popleft()

        if vision_grad.dim() == 2 and outputs.dim() == 3:
            if outputs.shape[0] == 1:
                vision_grad = vision_grad.unsqueeze(0)
            else:
                raise RuntimeError(
                    f"[r{self.rank}] Dimension mismatch: vision_grad is 2D {vision_grad.shape} "
                    f"but vision outputs batch_size={outputs.shape[0]} > 1"
                )
        elif vision_grad.dim() == 3 and outputs.dim() == 2:
            vision_grad = vision_grad.squeeze(0)

        outputs.backward(gradient=vision_grad, retain_graph=False)

    def backward_step(self, vision_grad_ref=None):
        if not self._pending_outputs:
            raise RuntimeError("No pending vision outputs for backward.")
        if vision_grad_ref is None:
            outputs = self._pending_outputs.popleft()
            outputs.sum().backward()
            return {"backward_time_ms": 0.0}

        vision_grad = self._retrieve_gradient_tensor(vision_grad_ref)
        self._apply_vision_backward(vision_grad)
        return {"backward_time_ms": 0.0}


@ray.remote(enable_tensor_transport=True, num_gpus=1, num_cpus=6)
class MegatronTextTrainer(MegatronBaseTrainer):
    """Megatron-backed text trainer using ms-swift Mcore-Bridge."""

    def __init__(self, config, rank: int, **kwargs):
        super().__init__(config, rank, **kwargs)
        self._pending_loss = None
        self._vision_embeddings = None

    def build_model(self):
        self._build_megatron_model()
        self.megatron_model.train()

    def initialize_trainer(self):
        device = self._get_device()
        seq_len = int(self.config.get("text_seq_len", 8))
        vocab_size = int(getattr(self.megatron_args, "padded_vocab_size", 32000))
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)
        labels = input_ids.clone()
        self._dummy_batch = {"input_ids": input_ids, "labels": labels}

    def forward_step(self, vision_payload=None, iteration: int = -1):
        from megatron.training.utils import get_ltor_masks_and_position_ids

        if iteration == -1 and isinstance(vision_payload, int):
            iteration = vision_payload
            vision_payload = None

        vision_embeddings = None
        if vision_payload is not None:
            # Unwrap list payloads (ActorGroup.execute_all returns a list)
            if isinstance(vision_payload, list):
                if len(vision_payload) != 1:
                    raise RuntimeError(
                        f"[r{self.rank}] Expected single vision payload, got list of {len(vision_payload)} items."
                    )
                vision_payload = vision_payload[0]
            # Handle ObjectRefs (may be inside the unwrapped payload)
            if isinstance(vision_payload, ray.ObjectRef):
                vision_payload = ray.get(vision_payload)
            # Handle nested lists (in case the unwrapped item is also a list)
            if isinstance(vision_payload, list):
                if len(vision_payload) != 1:
                    raise RuntimeError(
                        f"[r{self.rank}] Expected single vision payload, got nested list of {len(vision_payload)} items."
                    )
                vision_payload = vision_payload[0]

            normalized = normalize_vision_outputs(vision_payload)
            vision_iteration = normalized.meta.get("iteration")
            if vision_iteration is not None and vision_iteration != iteration:
                raise RuntimeError(
                    f"[r{self.rank}] Iteration mismatch! Vision iteration={vision_iteration}, "
                    f"text iteration={iteration}."
                )
            vision_embeddings_data = normalized.embeddings
            if isinstance(vision_embeddings_data, dict) and "use_ipc" in vision_embeddings_data:
                transfer_request = TensorTransferRequest.from_dict(vision_embeddings_data)
                receiver_gpu_id = get_physical_gpu_id()
                vision_embeddings = receive_tensor(transfer_request, receiver_gpu_id)
            else:
                vision_embeddings = vision_embeddings_data

            if vision_embeddings is not None:
                if not isinstance(vision_embeddings, torch.Tensor):
                    raise RuntimeError(
                        f"[r{self.rank}] Vision embeddings must be a tensor (got {type(vision_embeddings)})."
                    )
                vision_embeddings = vision_embeddings.detach().requires_grad_(True)

        batch = self._dummy_batch
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            input_ids,
            eod_token=0,
            pad_token=0,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            pad_mask_loss=False,
        )
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).repeat(3, 1, 1)
            # Apply context parallelism slicing for ALL models (including mRoPE).
            # ms-swift uses split_cp_inputs for proper load balancing with causal attention.
            context_parallel_size = int(getattr(self.megatron_args, "context_parallel_size", 1))
            if context_parallel_size > 1:
                from megatron.core import parallel_state

                cp_size = parallel_state.get_context_parallel_world_size()
                cp_rank = parallel_state.get_context_parallel_rank()
                seq_len = position_ids.shape[-1]
                # Use interleaved CP slicing pattern matching ms-swift's split_cp_inputs:
                # Split sequence into 2*cp_size chunks, each rank gets chunk i and chunk (2*cp_size-i-1)
                # This balances load for causal attention (early tokens attend to fewer, late to more).
                if seq_len % (2 * cp_size) != 0:
                    raise RuntimeError(
                        f"Sequence length {seq_len} not divisible by 2*context_parallel_size ({2 * cp_size})."
                    )
                chunk_size = seq_len // (2 * cp_size)
                indices = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=position_ids.device)

                # Slice position_ids: [3, bs, seq_len] -> [3, bs, 2*chunk_size]
                position_ids = position_ids.view(3, position_ids.shape[1], 2 * cp_size, chunk_size)
                position_ids = position_ids.index_select(2, indices)
                position_ids = position_ids.view(3, position_ids.shape[1], -1)

                # Slice labels: [bs, seq_len] -> [bs, 2*chunk_size]
                labels = labels.view(labels.shape[0], 2 * cp_size, chunk_size)
                labels = labels.index_select(1, indices)
                labels = labels.view(labels.shape[0], -1)

                # Slice loss_mask: [bs, seq_len] -> [bs, 2*chunk_size]
                loss_mask = loss_mask.view(loss_mask.shape[0], 2 * cp_size, chunk_size)
                loss_mask = loss_mask.index_select(1, indices)
                loss_mask = loss_mask.view(loss_mask.shape[0], -1)
        loss = self.megatron_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        if loss.numel() > 1:
            loss = loss.mean()
        if vision_embeddings is not None:
            loss = loss + (vision_embeddings.sum() * 0.0)

        self._pending_loss = loss
        self._vision_embeddings = vision_embeddings
        return {"loss": float(loss.detach().cpu()), "iteration": iteration}

    def backward_step(self):
        profile_time = self.config.get("profile_time", False)
        if profile_time:
            backward_start = time.perf_counter()

        if self._pending_loss is None:
            raise RuntimeError("No pending loss for backward.")
        self._pending_loss.backward()
        grad_payload = None
        if self._vision_embeddings is not None:
            if self._vision_embeddings.grad is None:
                raise RuntimeError("Vision embeddings gradient is None after backward.")
            grad_payload = self._vision_embeddings.grad

        self._pending_loss = None
        self._vision_embeddings = None

        backward_time_ms = 0.0
        if profile_time:
            torch.cuda.synchronize()
            backward_time_ms = (time.perf_counter() - backward_start) * 1000

        return TextBackwardOutputs(grad=grad_payload, meta={"backward_time_ms": backward_time_ms})
