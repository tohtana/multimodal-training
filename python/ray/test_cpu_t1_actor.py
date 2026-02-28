"""CPU-friendly T1 actor used by pipeline correctness tests."""

from __future__ import annotations

import os
from typing import Any

from .multi_stage_actor import MultiStageActor
from .payloads import StageGradients, StageOutputs


class CpuT1IpcActor(MultiStageActor):
    """T1 IPC shim that avoids CUDA IPC handles for CPU-only regression tests."""

    def get_physical_gpu_id(self) -> str:
        # Deterministic fake ID so parent/child rank-matched actors become T1 pairs.
        return f"FAKE-GPU-{self.rank}"

    def create_ipc_for_output(self, stage_name: str, microbatch_id: int | None = None) -> dict:
        if microbatch_id is None:
            raise ValueError(f"T1 create_ipc_for_output requires microbatch_id (stage='{stage_name}')")
        key = (stage_name, microbatch_id)
        result = self._t1_forward_ipc_cache.get(key)
        if result is None:
            raise KeyError(f"T1 IPC cache miss for forward output: stage='{stage_name}', microbatch={microbatch_id}")
        return {
            "__ipc__": True,
            "tensor": result.activations.detach().clone(),
            "meta": result.meta,
            "producer_microbatch_id": microbatch_id,
        }

    def forward_from_ipc(
        self,
        stage_name: str,
        ipc_data: dict,
        labels: Any = None,
        microbatch_id: int | None = None,
    ) -> StageOutputs:
        if microbatch_id is None:
            raise ValueError(f"T1 forward_from_ipc requires microbatch_id (stage='{stage_name}')")
        producer_mb = ipc_data.get("producer_microbatch_id")
        if producer_mb != microbatch_id:
            raise RuntimeError(
                f"T1 forward mapping mismatch: stage='{stage_name}', "
                f"producer_microbatch={producer_mb}, microbatch={microbatch_id}"
            )
        forced_fail = os.environ.get("MM_TEST_T1_FAIL_MB")
        if forced_fail is not None and int(forced_fail) == microbatch_id:
            raise RuntimeError(f"Injected T1 forward failure: stage='{stage_name}', microbatch={microbatch_id}")
        inputs = StageOutputs(activations=ipc_data["tensor"], meta=ipc_data.get("meta"))
        return self.forward_step(stage_name, inputs, labels, microbatch_id=microbatch_id)

    def create_ipc_for_grad(self, stage_name: str, microbatch_id: int | None = None) -> dict:
        if microbatch_id is None:
            raise ValueError(f"T1 create_ipc_for_grad requires microbatch_id (stage='{stage_name}')")
        key = (stage_name, microbatch_id)
        result = self._t1_backward_ipc_cache.get(key)
        if result is None:
            raise KeyError(f"T1 IPC cache miss for backward grad: stage='{stage_name}', microbatch={microbatch_id}")
        return {
            "__ipc__": True,
            "grad": result.grad.detach().clone() if result is not None and result.grad is not None else None,
            "meta": result.meta if result is not None else None,
            "producer_microbatch_id": microbatch_id,
        }

    def backward_step(
        self,
        stage_name: str,
        downstream_grad: StageGradients | dict | None = None,
        microbatch_id: int | None = None,
    ) -> StageGradients | None:
        # For CPU tests, resolve our fake IPC grad payload directly instead of CUDA IPC reconstruction.
        if isinstance(downstream_grad, dict) and downstream_grad.get("__ipc__") and "grad" in downstream_grad:
            producer_mb = downstream_grad.get("producer_microbatch_id")
            if producer_mb is not None and microbatch_id is not None and producer_mb != microbatch_id:
                raise RuntimeError(
                    f"T1 backward mapping mismatch: stage='{stage_name}', "
                    f"producer_microbatch={producer_mb}, microbatch={microbatch_id}"
                )
            downstream_grad = StageGradients(grad=downstream_grad.get("grad"), meta=downstream_grad.get("meta"))
        return super().backward_step(stage_name, downstream_grad, microbatch_id=microbatch_id)

    def backward_from_ipc(
        self,
        stage_name: str,
        ipc_data: dict,
        microbatch_id: int | None = None,
    ) -> StageGradients | None:
        if microbatch_id is None:
            raise ValueError(f"T1 backward_from_ipc requires microbatch_id (stage='{stage_name}')")
        producer_mb = ipc_data.get("producer_microbatch_id")
        if producer_mb != microbatch_id:
            raise RuntimeError(
                f"T1 backward mapping mismatch: stage='{stage_name}', "
                f"producer_microbatch={producer_mb}, microbatch={microbatch_id}"
            )
        grad_tensor = ipc_data.get("grad")
        grad = StageGradients(grad=grad_tensor, meta=ipc_data.get("meta"))
        return self.backward_step(stage_name, grad, microbatch_id=microbatch_id)

    def optimizer_step(
        self,
        stage_name: str,
        global_grad_norm: float | None = None,
        max_norm: float = 1.0,
    ) -> None:
        injected_stage = os.environ.get("MM_TEST_FAIL_OPT_STAGE")
        if injected_stage == stage_name:
            raise RuntimeError(f"Injected optimizer failure: stage='{stage_name}', microbatch=0")
        super().optimizer_step(stage_name, global_grad_norm=global_grad_norm, max_norm=max_norm)
