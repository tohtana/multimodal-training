"""Measure a SINGLE isolated iteration with fire-forget (no cross-iteration pipelining).

This reveals the true per-iteration latency including scheduling gaps.
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
    BenchmarkMultiStageActor,
    _make_pipeline,
    _make_single_gpu_resources,
    _make_specs,
)
from python.pipeline.placement import PlacementManager
from python.ray.payloads import StageGradients, StageOutputs


class FusedVoidActor(BenchmarkMultiStageActor):
    """Fused IPC + void returns."""

    def forward_step_ipc(self, stage_name, inputs=None, labels=None):
        super().forward_step(stage_name, inputs, labels)
        result = self._last_forward_outputs.get(stage_name)
        if result is None or result.activations is None or not result.activations.is_cuda:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.activations.detach())
        return {"__ipc__": True, "ipc_handle": handle, "gpu_id": gpu_id,
                "event_handle": event_handle, "meta": result.meta}

    def backward_step_ipc(self, stage_name, downstream_grad=None):
        super().backward_step(stage_name, downstream_grad)
        result = self._last_backward_grads.get(stage_name)
        if result is None or result.grad is None or not result.grad.is_cuda:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.grad)
        return {"__ipc__": True, "ipc_handle": handle, "gpu_id": gpu_id,
                "event_handle": event_handle, "meta": result.meta}

    def backward_from_ipc(self, stage_name, ipc_data):
        from python.ray.tensor_transfer import reconstruct_tensor_from_ipc
        from python.ray.utils import get_physical_gpu_id
        tensor = reconstruct_tensor_from_ipc(
            ipc_data["ipc_handle"], get_physical_gpu_id(),
            ipc_data["gpu_id"], ipc_data.get("event_handle"))
        grad = StageGradients(grad=tensor, meta=ipc_data.get("meta"))
        super().backward_step(stage_name, grad)
        return True

    def noop(self):
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

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=FusedVoidActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        N = 30

        # Warmup
        for _ in range(5):
            for mb_i in range(NUM_MICROBATCHES):
                ipc_ref = attn_actor.forward_step_ipc.remote("attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i])
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)
            ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

        # === Test 1: Back-to-back fire-forget (cross-iteration pipelining) ===
        t0 = time.perf_counter()
        for _ in range(N):
            for mb_i in range(NUM_MICROBATCHES):
                ipc_ref = attn_actor.forward_step_ipc.remote("attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i])
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)
        ray.get(attn_actor.noop.remote())  # drain
        t1 = time.perf_counter()
        pipelined_ms = (t1 - t0) / N * 1000

        # === Test 2: Single-iteration with barrier (no cross-iteration pipelining) ===
        times = []
        for _ in range(N):
            t0 = time.perf_counter()
            for mb_i in range(NUM_MICROBATCHES):
                ipc_ref = attn_actor.forward_step_ipc.remote("attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i])
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)
            ray.get(attn_actor.noop.remote())  # barrier: drain this iteration
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
        isolated_ms = np.mean(times)
        isolated_std = np.std(times)

        # === Test 3: Isolated + post-schedule calls (like real runner) ===
        times_full = []
        for _ in range(N):
            t0 = time.perf_counter()
            for mb_i in range(NUM_MICROBATCHES):
                ipc_ref = attn_actor.forward_step_ipc.remote("attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i])
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)
            # Post-schedule (same as real runner)
            ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))
            t1 = time.perf_counter()
            times_full.append((t1 - t0) * 1000)
        full_ms = np.mean(times_full)
        full_std = np.std(times_full)

        # === Test 4: Measure post-schedule overhead alone ===
        # Run one iteration to populate state, then measure just post-schedule
        for mb_i in range(NUM_MICROBATCHES):
            ipc_ref = attn_actor.forward_step_ipc.remote("attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i])
            moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
            grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
            attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)
        ray.get(attn_actor.noop.remote())  # drain schedule

        post_times = []
        for _ in range(N):
            t0 = time.perf_counter()
            ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))
            t1 = time.perf_counter()
            post_times.append((t1 - t0) * 1000)
        post_ms = np.mean(post_times)

        print("\n" + "=" * 65)
        print("SINGLE-ITERATION LATENCY (fused IPC, 4 calls/mb)")
        print("=" * 65)
        print(f"  Back-to-back (cross-iter pipeline): {pipelined_ms:8.2f} ms/iter")
        print(f"  Isolated (barrier per iter):        {isolated_ms:8.2f} +/- {isolated_std:.1f} ms/iter")
        print(f"  Isolated + post-schedule:           {full_ms:8.2f} +/- {full_std:.1f} ms/iter")
        print(f"  Post-schedule only (5 ray.get):     {post_ms:8.2f} ms")
        print(f"  Schedule-only (full - post):        {full_ms - post_ms:8.2f} ms")
        print(f"")
        print(f"  Single-process baseline (A):        ~452 ms (batch=1 x8 grad accum)")
        print(f"  Pipeline overhead:                  {full_ms - 452:+.0f} ms ({full_ms/452:.1f}x)")
        print(f"  Cross-iter pipeline effect:         {isolated_ms - pipelined_ms:+.1f} ms")

        manager.shutdown()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
