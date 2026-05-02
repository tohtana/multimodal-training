"""BridgeTrainer: learnable MLP projection between pipeline stages (native engine).

Sits between VisionTrainer and TextTrainer in a three-stage pipeline:
  vision → bridge → text

The bridge receives VisionOutputs from the upstream stage, projects through a
two-layer MLP (GELU activation), and emits VisionOutputs so the downstream
TextTrainer can consume it unchanged.  On backward, it receives
TextBackwardOutputs from downstream, backpropagates through the MLP, and emits
TextBackwardOutputs for the upstream VisionTrainer.
"""

import logging
import os
from collections import deque
from dataclasses import replace

import ray
import torch
import torch.nn as nn

from .payloads import TextBackwardOutputs, VisionOutputs, normalize_text_backward_outputs, normalize_vision_outputs
from .qwen3_vision_contract import (
    apply_qwen3_vision_backward,
    coerce_qwen3_vision_gradients,
    collect_qwen3_vision_gradients,
    QWEN3_VISUAL_PAYLOAD_META_KEY,
    Qwen3VisionGradients,
    Qwen3VisionPayload,
    qwen3_payload_metadata,
)
from .trainer import Trainer

logger = logging.getLogger(__name__)


class BridgeProjection(nn.Module):
    """Two-layer MLP with GELU activation for bridging vision and text dimensions."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = (input_dim + output_dim) // 2
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


@ray.remote(num_gpus=1, num_cpus=6)
class BridgeTrainer(Trainer):
    """Learnable bridge between pipeline stages (native engine, no parallelism)."""

    def __init__(self, config, rank: int, **kwargs):
        super().__init__(config, rank, **kwargs)
        self.model = None
        self._pending_inputs: deque[torch.Tensor] = deque()
        self._pending_outputs: deque[torch.Tensor] = deque()
        logger.debug(f"[r{self.rank}] BridgeTrainer initialized")

    @staticmethod
    def _make_bridge_input_tensor(tensor: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        tensor = tensor.to(device=device, dtype=dtype)
        if not tensor.requires_grad:
            tensor = tensor.detach().requires_grad_(True)
        tensor.retain_grad()
        return tensor

    def _make_qwen3_bridge_input_payload(
        self,
        payload: Qwen3VisionPayload,
        *,
        primary_embeddings: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Qwen3VisionPayload:
        primary = self._make_bridge_input_tensor(primary_embeddings, device=device, dtype=dtype)
        deepstack = tuple(
            self._make_bridge_input_tensor(tensor, device=device, dtype=dtype)
            for tensor in payload.deepstack_visual_embeds
        )
        return replace(payload, primary_embeddings=primary, deepstack_visual_embeds=deepstack)

    def _project_qwen3_payload(self, payload: Qwen3VisionPayload) -> Qwen3VisionPayload:
        projected_primary = self.model(payload.primary_embeddings)
        projected_deepstack = tuple(self.model(tensor) for tensor in payload.deepstack_visual_embeds)
        return replace(payload, primary_embeddings=projected_primary, deepstack_visual_embeds=projected_deepstack)

    @staticmethod
    def _move_qwen3_gradients_to_payload(
        gradients: Qwen3VisionGradients,
        payload: Qwen3VisionPayload,
    ) -> Qwen3VisionGradients:
        primary_grad = gradients.primary_grad.to(
            device=payload.primary_embeddings.device,
            dtype=payload.primary_embeddings.dtype,
        )
        deepstack_grads = tuple(
            grad.to(device=tensor.device, dtype=tensor.dtype)
            for grad, tensor in zip(gradients.deepstack_grads, payload.deepstack_visual_embeds)
        )
        return Qwen3VisionGradients(primary_grad=primary_grad, deepstack_grads=deepstack_grads)

    def build_model(self):
        device = self._get_device()
        dtype_str = self.config.get("dtype", "bfloat16")
        torch_dtype = self._get_torch_dtype(dtype_str)

        input_dim = self.config.get("bridge_input_dim", 5120)
        output_dim = self.config.get("bridge_output_dim", 5120)
        hidden_dim = self.config.get("bridge_hidden_dim", None)

        self.model = BridgeProjection(input_dim, output_dim, hidden_dim)
        self.model.to(device=device, dtype=torch_dtype)
        self.model.train()

        # Build optimizer and scheduler using base class helpers
        params = list(self.model.parameters())
        self._build_optimizer(params)

        num_training_steps = self.config.get("num_iterations", 1) * self.config.get("num_epochs", 1)
        if num_training_steps > 0:
            self._build_scheduler(num_training_steps)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(
            f"[r{self.rank}] BridgeProjection built: "
            f"input_dim={input_dim}, output_dim={output_dim}, "
            f"hidden_dim={hidden_dim or (input_dim + output_dim) // 2}, "
            f"params={total_params:,}, dtype={torch_dtype}"
        )

    def forward_step(self, upstream_ref, iteration: int = -1):
        """Project upstream VisionOutputs through MLP, return VisionOutputs.

        Args:
            upstream_ref: Ray ObjectRef or payload from the upstream stage.
            iteration: Current training iteration.

        Returns:
            VisionOutputs with projected embeddings.
        """
        # Resolve upstream payload
        if isinstance(upstream_ref, ray.ObjectRef):
            upstream_data = ray.get(upstream_ref)
        else:
            upstream_data = upstream_ref

        vision_payload = normalize_vision_outputs(upstream_data)
        embeddings = vision_payload.embeddings
        qwen3_payload = vision_payload.meta.get(QWEN3_VISUAL_PAYLOAD_META_KEY)
        if qwen3_payload is not None and not isinstance(qwen3_payload, Qwen3VisionPayload):
            raise RuntimeError(f"Invalid Qwen3 visual payload type: {type(qwen3_payload)}")

        # Move to device if needed
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        if isinstance(qwen3_payload, Qwen3VisionPayload):
            if isinstance(embeddings, torch.Tensor):
                primary_embeddings = embeddings
            else:
                primary_embeddings = torch.as_tensor(embeddings)
            input_payload = self._make_qwen3_bridge_input_payload(
                qwen3_payload,
                primary_embeddings=primary_embeddings,
                device=device,
                dtype=dtype,
            )

            autocast_ctx = self._get_autocast_context()
            with autocast_ctx:
                projected_payload = self._project_qwen3_payload(input_payload)

            self._pending_inputs.append(input_payload)
            self._pending_outputs.append(projected_payload)

            meta = dict(vision_payload.meta)
            meta["iteration"] = iteration
            meta.update(qwen3_payload_metadata(projected_payload))
            meta["qwen3_bridge_projected_structured"] = True

            return VisionOutputs(
                embeddings=projected_payload.primary_embeddings,
                attention_mask=vision_payload.attention_mask,
                meta=meta,
            )

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.to(device=device, dtype=dtype)
        else:
            embeddings = torch.as_tensor(embeddings, device=device, dtype=dtype)

        # Ensure grad tracking for backward
        if not embeddings.requires_grad:
            embeddings = embeddings.detach().requires_grad_(True)

        # Forward through MLP with optional autocast
        autocast_ctx = self._get_autocast_context()
        with autocast_ctx:
            projected = self.model(embeddings)

        # Store for backward
        self._pending_inputs.append(embeddings)
        self._pending_outputs.append(projected)

        meta = dict(vision_payload.meta)
        meta["iteration"] = iteration

        return VisionOutputs(embeddings=projected, attention_mask=vision_payload.attention_mask, meta=meta)

    def backward_step(self, downstream_grad_ref):
        """Backpropagate gradient through MLP, return TextBackwardOutputs.

        Args:
            downstream_grad_ref: Ray ObjectRef or TextBackwardOutputs from downstream.

        Returns:
            TextBackwardOutputs with gradient for the upstream stage.
        """
        # Resolve downstream gradient
        if isinstance(downstream_grad_ref, ray.ObjectRef):
            grad_data = ray.get(downstream_grad_ref)
        else:
            grad_data = downstream_grad_ref

        grad_payload = normalize_text_backward_outputs(grad_data)
        grad_tensor = grad_payload.grad

        if not self._pending_outputs or not self._pending_inputs:
            raise RuntimeError(
                f"[r{self.rank}] No pending bridge outputs/inputs for backward. "
                "Ensure forward_step was called before backward_step."
            )

        projected = self._pending_outputs.popleft()
        input_embeddings = self._pending_inputs.popleft()

        if isinstance(projected, Qwen3VisionPayload):
            if isinstance(grad_tensor, torch.Tensor):
                if projected.deepstack_visual_embeds:
                    raise RuntimeError(
                        "Bridge received only a primary gradient tensor for a structured Qwen3 payload with "
                        "deepstack tensors."
                    )
                grad_tensor = Qwen3VisionGradients(primary_grad=grad_tensor)
            structured_grad = coerce_qwen3_vision_gradients(grad_tensor)
            structured_grad = self._move_qwen3_gradients_to_payload(structured_grad, projected)
            apply_qwen3_vision_backward(projected, structured_grad)
            upstream_grad = collect_qwen3_vision_gradients(input_embeddings)
            return TextBackwardOutputs(grad=upstream_grad, meta=dict(grad_payload.meta))

        # Move grad to same device as output
        device = projected.device
        if isinstance(grad_tensor, torch.Tensor):
            grad_tensor = grad_tensor.to(device=device, dtype=projected.dtype)

        # Match dimensions
        if grad_tensor.dim() == 2 and projected.dim() == 3:
            if projected.shape[0] == 1:
                grad_tensor = grad_tensor.unsqueeze(0)
        elif grad_tensor.dim() == 3 and projected.dim() == 2:
            grad_tensor = grad_tensor.squeeze(0)

        # Backward through the MLP
        projected.backward(gradient=grad_tensor, retain_graph=False)

        # Extract gradient for upstream
        upstream_grad = input_embeddings.grad
        if upstream_grad is not None:
            upstream_grad = upstream_grad.detach().clone()
            # Squeeze batch dim to match what vision trainer expects
            if upstream_grad.dim() == 3 and upstream_grad.shape[0] == 1:
                upstream_grad = upstream_grad.squeeze(0)

        return TextBackwardOutputs(grad=upstream_grad, meta=dict(grad_payload.meta))

    # ── Abstract method stubs required by Trainer ──

    def _load_model_config(self, model_name):
        return None

    def _get_transformer_layers(self, model):
        return []

    def _get_tensor_parallel_mapping(self):
        return {}

    def save_checkpoint(self, checkpoint_dir: str, epoch: int):
        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}", "bridge")
        os.makedirs(epoch_dir, exist_ok=True)
        path = os.path.join(epoch_dir, f"rank_{self.rank}.pt")
        torch.save({"model": self.model.state_dict()}, path)
        logger.debug(f"[r{self.rank}] Saved bridge checkpoint to {path}")
        return path

    def load_checkpoint(self, checkpoint_dir: str, epoch: int):
        path = os.path.join(checkpoint_dir, f"epoch_{epoch}", "bridge", f"rank_{self.rank}.pt")
        if not os.path.exists(path):
            logger.warning(f"[r{self.rank}] Bridge checkpoint not found: {path}")
            return False
        data = torch.load(path, map_location="cpu")
        self.model.load_state_dict(data["model"])
        logger.debug(f"[r{self.rank}] Loaded bridge checkpoint from {path}")
        return True
