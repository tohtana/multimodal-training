"""MultiStageActor: a Ray actor hosting multiple pipeline stage trainers (design §6.2).

Each actor in an ActorGroup holds one or more StageTrainer instances (one per
stage placed on this resource set). The actor dispatches forward/backward/optimizer
calls to the named stage trainer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import ray
import torch
import torch.nn as nn

from ..ray.payloads import StageGradients, StageOutputs
from ..ray.shared_buffer import SharedActivationBuffer, SharedEventPool

logger = logging.getLogger(__name__)


class MultiStageActor:
    """A Ray actor that hosts one or more pipeline stage trainers.

    Unlike RayActor, this does not set up torch.distributed — the pipeline
    framework handles placement and transport separately.

    Model construction happens via build_model_from_state_dict(), which receives:
    - A model_cls and constructor kwargs (must be serializable)
    - An initial state_dict (transferred via Ray ObjectRef)
    - Optimizer and loss configuration
    """

    def __init__(self, config: dict, rank: int):
        self.rank = rank
        self.config = config
        self._trainers: dict[str, StageTrainer] = {}
        self._last_forward_outputs: dict[str, StageOutputs] = {}
        self._last_backward_grads: dict[str, StageGradients | None] = {}

    def build_model_from_state_dict(
        self,
        stage_name: str,
        model_cls: type,
        model_kwargs: dict,
        state_dict: dict | None = None,
        is_terminal: bool = False,
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_cls: type | None = None,
        loss_kwargs: dict | None = None,
        engine: str = "native",
        ds_config: dict | None = None,
    ) -> bool:
        """Build a stage trainer from a model class + state dict.

        All arguments must be serializable (no lambdas or closures).
        The model is constructed as model_cls(**model_kwargs), then state_dict is loaded.

        Args:
            stage_name: Name of the stage.
            model_cls: The nn.Module class (e.g., nn.Linear).
            model_kwargs: Constructor kwargs (e.g., {"in_features": 32, "out_features": 64}).
            state_dict: Optional initial state dict.
            is_terminal: Whether this is a terminal (loss-computing) stage.
            optimizer_cls: Optimizer class (e.g., torch.optim.Adam).
            optimizer_kwargs: Optimizer kwargs excluding params (e.g., {"lr": 0.01}).
            loss_cls: Loss function class for terminal stages (e.g., nn.CrossEntropyLoss).
            loss_kwargs: Loss function kwargs.
            engine: "native" or "deepspeed".
            ds_config: DeepSpeed config dict (required when engine="deepspeed").

        Returns:
            True on success.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Construct model
        model = model_cls(**model_kwargs).to(device)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Construct loss
        loss_fn = None
        if loss_cls is not None and is_terminal:
            loss_fn = loss_cls(**(loss_kwargs or {}))

        if engine == "deepspeed":
            from ..pipeline.ds_trainer import DeepSpeedStageTrainer

            if ds_config is None:
                raise ValueError(f"Stage '{stage_name}': engine='deepspeed' requires ds_config")
            trainer = DeepSpeedStageTrainer(
                stage_name=stage_name,
                model=model,
                ds_config=ds_config,
                loss_fn=loss_fn,
                is_terminal=is_terminal,
                device=device,
            )
        else:
            from ..pipeline.native_trainer import StageTrainer

            # Construct optimizer (native only — DeepSpeed manages its own)
            optimizer = None
            if optimizer_cls is not None:
                opt_kwargs = optimizer_kwargs or {}
                optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

            trainer = StageTrainer(
                stage_name=stage_name,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                is_terminal=is_terminal,
                device=device,
            )

        self._trainers[stage_name] = trainer
        logger.info(f"[r{self.rank}] Built {engine} model for stage '{stage_name}' on {device}")
        return True

    def forward_step(
        self,
        stage_name: str,
        inputs: StageOutputs | None = None,
        labels: Any = None,
    ) -> StageOutputs:
        """Run forward for a named stage.

        For cross-GPU transport, call with .options(tensor_transport="nccl").remote()
        so the returned StageOutputs tensor is delivered via RDT/NCCL.
        """
        trainer = self._get_trainer(stage_name)
        result = trainer.forward_step(inputs, labels=labels)
        self._last_forward_outputs[stage_name] = result
        return result

    def backward_step(
        self,
        stage_name: str,
        downstream_grad: StageGradients | dict | None = None,
    ) -> StageGradients | None:
        """Run backward for a named stage.

        For cross-GPU transport, call with .options(tensor_transport="nccl").remote()
        so the returned StageGradients tensor is delivered via RDT/NCCL.

        Accepts either StageGradients or an IPC metadata dict (from T1 transport).
        IPC dicts are auto-detected and reconstructed before use.
        """
        if isinstance(downstream_grad, dict) and downstream_grad.get("__ipc__"):
            from ..ray.tensor_transfer import reconstruct_tensor_from_ipc
            from ..ray.utils import get_physical_gpu_id

            tensor = reconstruct_tensor_from_ipc(
                downstream_grad["ipc_handle"],
                get_physical_gpu_id(),
                downstream_grad["gpu_id"],
                downstream_grad.get("event_handle"),
            )
            downstream_grad = StageGradients(grad=tensor, meta=downstream_grad.get("meta"))

        trainer = self._get_trainer(stage_name)
        result = trainer.backward_step(downstream_grad)
        self._last_backward_grads[stage_name] = result
        return result

    def optimizer_step(
        self,
        stage_name: str,
        global_grad_norm: float | None = None,
        max_norm: float = 1.0,
    ) -> None:
        """Step optimizer for a named stage."""
        trainer = self._get_trainer(stage_name)
        trainer.optimizer_step(global_grad_norm=global_grad_norm, max_norm=max_norm)

    def compute_grad_norm_sq(self, stage_name: str) -> float:
        """Compute sum-of-squares gradient norm for a named stage."""
        trainer = self._get_trainer(stage_name)
        return trainer.compute_grad_norm_sq()

    def zero_grad(self, stage_name: str) -> None:
        """Zero gradients for a named stage."""
        trainer = self._get_trainer(stage_name)
        trainer.zero_grad()

    def get_last_loss(self, stage_name: str) -> float | None:
        """Get the loss from the last forward of a terminal stage.

        Returns the scalar loss value (safe to ray.get without CUDA deserialization).
        """
        trainer = self._get_trainer(stage_name)
        return getattr(trainer, "_last_loss_value", None)

    def get_last_forward_meta(self, stage_name: str) -> dict:
        """Get metadata from the last forward (scalars only, no CUDA tensors)."""
        trainer = self._get_trainer(stage_name)
        return getattr(trainer, "_last_forward_meta", {})

    def run_full_backward(self, stage_names_reversed: list[str], stage_successors: dict[str, list[str]]) -> bool:
        """Run backward for all stages within this actor, passing gradients locally.

        Args:
            stage_names_reversed: Stage names in reverse topological order.
            stage_successors: Map from stage name to its successor stage names.

        Returns:
            True when all backward ops complete (safe scalar for ray.get).
        """
        grad_map: dict[str, "StageGradients | None"] = {}

        for name in stage_names_reversed:
            trainer = self._trainers[name]
            succs = stage_successors.get(name, [])

            if not succs:
                # Terminal stage
                upstream_grad = trainer.backward_step(downstream_grad=None)
            else:
                # Get gradient from successor
                succ_name = succs[0]
                downstream_grad = grad_map.get(succ_name)
                upstream_grad = trainer.backward_step(downstream_grad=downstream_grad)

            grad_map[name] = upstream_grad

        return True

    def sum_gradients_ipc(self, grads_or_ipc: list) -> StageGradients:
        """Sum gradient payloads that may include IPC-wrapped dicts.

        Reconstructs IPC handles before summing. Used when gradient sources
        include both T1 (IPC) and T2 (RDT) actors.
        """
        from ..ray.tensor_transfer import reconstruct_tensor_from_ipc
        from ..ray.utils import get_physical_gpu_id

        my_gpu = get_physical_gpu_id()
        resolved = []
        for g in grads_or_ipc:
            if isinstance(g, dict) and g.get("__ipc__"):
                tensor = reconstruct_tensor_from_ipc(g["ipc_handle"], my_gpu, g["gpu_id"], g.get("event_handle"))
                resolved.append(StageGradients(grad=tensor, meta=g.get("meta")))
            else:
                resolved.append(g)
        return self.sum_gradients(resolved)

    def sum_gradients(self, grads: list[StageGradients]) -> StageGradients:
        """Sum multiple gradient payloads into one (used for M:N backward aggregation).

        When a source stage maps to multiple destination actors, the backward pass
        produces one gradient per destination actor. This method aggregates them
        by summing on-device before passing to the source stage's backward_step.
        """
        if len(grads) == 1:
            return grads[0]
        summed = grads[0].grad.clone()
        for g in grads[1:]:
            summed = summed + g.grad
        return StageGradients(grad=summed, meta=grads[0].meta)

    def accumulate_gradient(
        self,
        current: StageGradients,
        new_grad: StageGradients,
    ) -> StageGradients:
        """Add a gradient to an accumulator (for sequential M:N aggregation).

        Unlike sum_gradients (which takes a list), this takes two top-level
        arguments that Ray resolves individually — avoiding issues with
        unresolved ObjectRefs in lists on tensor-transport-enabled actors.
        """
        summed = current.grad + new_grad.grad
        return StageGradients(grad=summed, meta=current.meta)

    def get_physical_gpu_id(self) -> str:
        """Get the physical GPU UUID for this actor's device."""
        from ..ray.utils import get_physical_gpu_id

        return get_physical_gpu_id()

    def create_ipc_for_output(self, stage_name: str) -> dict | None:
        """Create CUDA IPC handle for the last forward output of a stage.

        Returns a dict of IPC metadata (no CUDA tensors — safe for object store).
        Must be called on the same actor after forward_step.
        Detaches the tensor (autograd graphs do not cross process boundaries).
        """
        result = self._last_forward_outputs.get(stage_name)
        if result is None or result.activations is None or not result.activations.is_cuda:
            return None
        from ..ray.tensor_transfer import create_ipc_handle

        handle, gpu_id, event_handle = create_ipc_handle(result.activations.detach())
        return {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }

    def forward_from_ipc(self, stage_name: str, ipc_data: dict, labels: Any = None) -> StageOutputs:
        """Reconstruct tensor from IPC metadata and run forward.

        Used by T1 (same-GPU, different-process) transport receivers.
        """
        from ..ray.tensor_transfer import reconstruct_tensor_from_ipc
        from ..ray.utils import get_physical_gpu_id

        tensor = reconstruct_tensor_from_ipc(
            ipc_data["ipc_handle"],
            get_physical_gpu_id(),
            ipc_data["gpu_id"],
            ipc_data.get("event_handle"),
        )
        inputs = StageOutputs(activations=tensor, meta=ipc_data.get("meta"))
        return self.forward_step(stage_name, inputs, labels)

    def create_ipc_for_grad(self, stage_name: str) -> dict | None:
        """Create CUDA IPC handle for the last backward gradient of a stage.

        Returns a dict of IPC metadata (no CUDA tensors — safe for object store).
        Must be called on the same actor after backward_step.
        """
        result = self._last_backward_grads.get(stage_name)
        if result is None or result.grad is None or not result.grad.is_cuda:
            return None
        from ..ray.tensor_transfer import create_ipc_handle

        handle, gpu_id, event_handle = create_ipc_handle(result.grad)
        return {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }

    def backward_from_ipc(self, stage_name: str, ipc_data: dict) -> StageGradients | None:
        """Reconstruct gradient from IPC metadata and run backward.

        Used by T1 (same-GPU, different-process) transport receivers.
        """
        from ..ray.tensor_transfer import reconstruct_tensor_from_ipc
        from ..ray.utils import get_physical_gpu_id

        tensor = reconstruct_tensor_from_ipc(
            ipc_data["ipc_handle"],
            get_physical_gpu_id(),
            ipc_data["gpu_id"],
            ipc_data.get("event_handle"),
        )
        grad = StageGradients(grad=tensor, meta=ipc_data.get("meta"))
        return self.backward_step(stage_name, grad)

    # ── Shared buffer methods (T19: zero-overhead inter-process tensor transfer) ──

    def infer_activation_spec(
        self,
        stage_name: str,
        sample_inputs: StageOutputs,
        sample_labels: Any = None,
    ) -> dict:
        """Infer output shape/dtype without mutating trainer runtime queues/state.

        Uses trainer.model(...) under torch.no_grad(); must NOT call forward_step/backward_step.
        """
        trainer = self._get_trainer(stage_name)
        x = sample_inputs.activations.to(trainer.device)
        with torch.no_grad():
            out = trainer.model(x)
        return {
            "shape": tuple(out.shape),
            "dtype": out.dtype,
            "payload_mode": "unknown",
            "required_meta_keys": [],
        }

    def setup_shared_buffers(
        self,
        buffer_id: str,
        num_slots: int,
        shape: tuple,
        dtype: torch.dtype,
    ) -> dict:
        """Create shared ring buffer + event pool. Returns IPC handles dict.

        Called once on the producer actor at pipeline setup time.
        buffer_id convention: "{src}->{dst}:{fwd|bwd}:r{src_rank}->r{dst_rank}"
        """
        device = torch.device("cuda:0")
        buffer = SharedActivationBuffer(num_slots, shape, dtype, device)
        events = SharedEventPool(num_slots, device)

        if not hasattr(self, "_shared_local_buffers"):
            self._shared_local_buffers = {}
            self._shared_local_events = {}
        self._shared_local_buffers[buffer_id] = buffer
        self._shared_local_events[buffer_id] = events

        return {
            "buffer_id": buffer_id,
            "buffer_ipc": buffer.export_ipc_handles(),
            "event_ipc": events.export_ipc_handles(),
        }

    def open_shared_buffers(self, ipc_data: dict) -> bool:
        """Open shared ring buffer + event pool from IPC handles.

        Called once on the consumer actor at pipeline setup time.
        """
        device = torch.device("cuda:0")
        buffer_id = ipc_data["buffer_id"]

        if not hasattr(self, "_shared_remote_buffers"):
            self._shared_remote_buffers = {}
            self._shared_remote_events = {}
        self._shared_remote_buffers[buffer_id] = SharedActivationBuffer.open_from_ipc(ipc_data["buffer_ipc"], device)
        self._shared_remote_events[buffer_id] = SharedEventPool.open_from_ipc(ipc_data["event_ipc"], device)
        return True

    def clear_shared_buffers(self, buffer_ids: list[str] | None = None) -> bool:
        """Best-effort cleanup used when setup fails part-way. Never raises."""
        for attr in (
            "_shared_local_buffers",
            "_shared_local_events",
            "_shared_remote_buffers",
            "_shared_remote_events",
        ):
            store = getattr(self, attr, None)
            if not isinstance(store, dict):
                continue
            if buffer_ids is None:
                store.clear()
            else:
                for key in buffer_ids:
                    store.pop(key, None)
        return True

    def forward_to_buffer(
        self,
        stage_name: str,
        buffer_id: str,
        slot: int,
        inputs=None,
        labels=None,
    ) -> bool:
        """Run forward, then write activations to shared buffer slot and record event.

        v1 payload contract: activations-only. Raises if attention_mask or meta present.
        """
        result = self.forward_step(stage_name, inputs, labels)
        if result.attention_mask is not None or bool(result.meta):
            raise RuntimeError(
                f"{stage_name} emitted attention_mask/meta; shared_buffer v1 requires activations-only payload"
            )
        self._shared_local_buffers[buffer_id].write(slot, result.activations.detach())
        self._shared_local_events[buffer_id].record(slot)
        return True

    def forward_from_buffer(self, stage_name: str, buffer_id: str, slot: int, labels=None) -> bool:
        """Wait on shared buffer event, read activations, run forward."""
        self._shared_remote_events[buffer_id].wait(slot)
        tensor = self._shared_remote_buffers[buffer_id].read(slot)
        inputs = StageOutputs(activations=tensor, attention_mask=None, meta={})
        self.forward_step(stage_name, inputs, labels)
        return True

    def backward_to_buffer(self, stage_name: str, buffer_id: str, slot: int, downstream_grad=None) -> bool:
        """Run backward, write upstream gradient to shared buffer slot, record event."""
        result = self.backward_step(stage_name, downstream_grad)
        if result is not None and result.grad is not None:
            if bool(result.meta):
                raise RuntimeError(
                    f"{stage_name} emitted gradient meta; shared_buffer v1 requires tensor-only StageGradients"
                )
            self._shared_local_buffers[buffer_id].write(slot, result.grad)
            self._shared_local_events[buffer_id].record(slot)
        return True

    def backward_from_buffer(self, stage_name: str, buffer_id: str, slot: int) -> bool:
        """Wait on shared buffer event, read gradient, run backward."""
        self._shared_remote_events[buffer_id].wait(slot)
        grad_tensor = self._shared_remote_buffers[buffer_id].read(slot)
        downstream_grad = StageGradients(grad=grad_tensor, meta={})
        self.backward_step(stage_name, downstream_grad)
        return True

    def _get_trainer(self, stage_name: str) -> "StageTrainer":
        if stage_name not in self._trainers:
            raise ValueError(f"[r{self.rank}] Stage '{stage_name}' not built. Call build_model first.")
        return self._trainers[stage_name]
