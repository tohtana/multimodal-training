"""Benchmark: batched microbatch processing — one Ray call per actor per phase.

Instead of 32 Ray calls (4 per microbatch × 8 microbatches), batch all
microbatches into one call per actor:

  Phase 1: attn.forward_batch(8)  → list of 8 IPC handles
  Phase 2: moe.forward_backward_batch(8 handles) → list of 8 grad IPC handles
  Phase 3: attn.backward_batch(8 grad handles)

Total: 3 Ray calls for the schedule + 5 post-schedule = 8 calls.
"""

from __future__ import annotations

import time

import ray
import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
    generate_dummy_batch,
)
from examples.attn_moe_overlap.step6_mps_overlap import (
    BATCH_SIZE,
    BASE_LR,
    DTYPE,
    NUM_CLASSES,
    NUM_MICROBATCHES,
    SEQ_LEN,
    TIMED_ITERS,
    WARMUP_ITERS,
    BenchmarkMultiStageActor,
    _make_pipeline,
    _make_single_gpu_resources,
    _make_specs,
)
from python.pipeline.placement import PlacementManager
from python.ray.payloads import StageGradients, StageOutputs


class BatchedActor(BenchmarkMultiStageActor):
    """Actor that processes all microbatches in batched calls."""

    def forward_batch_ipc(self, stage_name, num_microbatches, dummy_inputs=None, labels_list=None):
        """Forward all microbatches, return list of IPC handles.

        Uses preloaded data for source stages.
        """
        from python.ray.tensor_transfer import create_ipc_handle

        ipc_handles = []
        for mb_i in range(num_microbatches):
            # forward_step uses preloaded data for the source stage
            super().forward_step(
                stage_name,
                StageOutputs(activations=dummy_inputs) if dummy_inputs is not None else None,
                labels_list[mb_i] if labels_list is not None else None,
            )
            result = self._last_forward_outputs.get(stage_name)
            handle, gpu_id, event_handle = create_ipc_handle(result.activations.detach())
            ipc_handles.append({
                "__ipc__": True,
                "ipc_handle": handle,
                "gpu_id": gpu_id,
                "event_handle": event_handle,
                "meta": result.meta,
            })
        return ipc_handles

    def forward_backward_batch_ipc(self, stage_name, fwd_ipc_handles, labels_list):
        """Forward + backward all microbatches from IPC handles.

        For terminal stage: forward all, then backward all (grad accumulation).
        Returns list of upstream grad IPC handles.
        """
        from python.ray.tensor_transfer import create_ipc_handle, reconstruct_tensor_from_ipc
        from python.ray.utils import get_physical_gpu_id

        my_gpu = get_physical_gpu_id()
        num_mb = len(fwd_ipc_handles)

        # Forward all microbatches
        for mb_i in range(num_mb):
            ipc = fwd_ipc_handles[mb_i]
            tensor = reconstruct_tensor_from_ipc(
                ipc["ipc_handle"], my_gpu, ipc["gpu_id"], ipc.get("event_handle")
            )
            inputs = StageOutputs(activations=tensor, meta=ipc.get("meta"))
            super().forward_step(stage_name, inputs, labels_list[mb_i])

        # Backward all microbatches (in reverse for proper grad accumulation)
        grad_ipc_handles = []
        for mb_i in range(num_mb):
            super().backward_step(stage_name, None)  # terminal stage
            result = self._last_backward_grads.get(stage_name)
            handle, gpu_id, event_handle = create_ipc_handle(result.grad)
            grad_ipc_handles.append({
                "__ipc__": True,
                "ipc_handle": handle,
                "gpu_id": gpu_id,
                "event_handle": event_handle,
                "meta": result.meta,
            })
        return grad_ipc_handles

    def backward_batch_from_ipc(self, stage_name, grad_ipc_handles):
        """Backward all microbatches from grad IPC handles. Returns True (void)."""
        from python.ray.tensor_transfer import reconstruct_tensor_from_ipc
        from python.ray.utils import get_physical_gpu_id

        my_gpu = get_physical_gpu_id()
        for ipc in grad_ipc_handles:
            tensor = reconstruct_tensor_from_ipc(
                ipc["ipc_handle"], my_gpu, ipc["gpu_id"], ipc.get("event_handle")
            )
            grad = StageGradients(grad=tensor, meta=ipc.get("meta"))
            super().backward_step(stage_name, grad)
        return True


def main():
    import numpy as np

    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)
    dummy_mb = torch.zeros(1, 1, 1, dtype=DTYPE)
    mb_labels_list = list(labels.chunk(NUM_MICROBATCHES, dim=0))

    ray.init()
    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=BatchedActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        # Warmup
        for _ in range(WARMUP_ITERS):
            # Phase 1: attn forward all microbatches → IPC handles
            fwd_handles_ref = attn_actor.forward_batch_ipc.remote(
                "attn", NUM_MICROBATCHES, dummy_mb, mb_labels_list
            )
            # Phase 2: moe forward+backward all → grad IPC handles
            grad_handles_ref = moe_actor.forward_backward_batch_ipc.remote(
                "moe", fwd_handles_ref, mb_labels_list
            )
            # Phase 3: attn backward all
            ray.get(attn_actor.backward_batch_from_ipc.remote("attn", grad_handles_ref))
            # Post-schedule
            ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

        # Timed
        losses = []
        times_ms = []
        for step in range(TIMED_ITERS):
            t0 = time.perf_counter()

            # 3 Ray calls for the schedule (fire-and-forget with dependency chain)
            fwd_handles_ref = attn_actor.forward_batch_ipc.remote(
                "attn", NUM_MICROBATCHES, dummy_mb, mb_labels_list
            )
            grad_handles_ref = moe_actor.forward_backward_batch_ipc.remote(
                "moe", fwd_handles_ref, mb_labels_list
            )
            attn_actor.backward_batch_from_ipc.remote("attn", grad_handles_ref)

            # Post-schedule: 5 Ray calls
            loss = ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            losses.append(loss)
            times_ms.append(elapsed_ms)
            print(f"  [batched] Step {step:3d} | Loss: {loss:.6f} | Time: {elapsed_ms:.2f} ms")

        arr = np.array(times_ms)
        print(f"\n{'=' * 60}")
        print(f"BATCHED RESULTS (3 schedule calls + 5 post-schedule = 8 total)")
        print(f"{'=' * 60}")
        print(f"  Mean:           {arr.mean():.2f} +/- {arr.std():.2f} ms/iter")
        print(f"  Min:            {arr.min():.2f} ms")
        print(f"  Ray calls/iter: 8  (was 53)")
        print(f"  Single-process: ~202 ms (batch=1 x8, warm GPU)")
        print(f"  Overhead:       {arr.mean() - 202:+.0f} ms ({arr.mean()/202:.1f}x)")
        print(f"  Final loss:     {losses[-1]:.6f}")

        manager.shutdown()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
