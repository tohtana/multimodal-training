"""Benchmark: fused forward+IPC and backward+IPC to reduce Ray call count.

Current T1 per-microbatch (6 calls):
  attn.forward_step → attn.create_ipc_for_output → moe.forward_from_ipc
  → moe.backward_step → moe.create_ipc_for_grad → attn.backward_step

Fused T1 per-microbatch (4 calls):
  attn.forward_step_ipc → moe.forward_from_ipc
  → moe.backward_step_ipc → attn.backward_from_ipc
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
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import OneFOneBScheduler
from python.ray.payloads import StageGradients, StageOutputs


class FusedIPCActor(BenchmarkMultiStageActor):
    """Actor that fuses forward/backward with IPC handle creation.

    Returns only the IPC handle dict (~1 KB) instead of the full tensor,
    eliminating both the extra Ray call and the return-value serialization.
    """

    def forward_step_ipc(self, stage_name, inputs=None, labels=None):
        """Forward + create IPC handle in one call. Returns IPC dict."""
        super().forward_step(stage_name, inputs, labels)
        # Create IPC handle from the just-computed output
        result = self._last_forward_outputs.get(stage_name)
        if result is None or result.activations is None or not result.activations.is_cuda:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.activations.detach())
        return {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }

    def backward_step_ipc(self, stage_name, downstream_grad=None):
        """Backward + create IPC handle for upstream grad in one call."""
        super().backward_step(stage_name, downstream_grad)
        # Create IPC handle from the just-computed gradient
        result = self._last_backward_grads.get(stage_name)
        if result is None or result.grad is None or not result.grad.is_cuda:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.grad)
        return {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }

    def backward_from_ipc(self, stage_name, ipc_data):
        """Reconstruct grad from IPC and run backward. Return True (void)."""
        from python.ray.tensor_transfer import reconstruct_tensor_from_ipc
        from python.ray.utils import get_physical_gpu_id
        tensor = reconstruct_tensor_from_ipc(
            ipc_data["ipc_handle"],
            get_physical_gpu_id(),
            ipc_data["gpu_id"],
            ipc_data.get("event_handle"),
        )
        grad = StageGradients(grad=tensor, meta=ipc_data.get("meta"))
        super().backward_step(stage_name, grad)
        return True  # void return — no tensor serialization


def run_fused_variant(name, config, attn_sd, moe_sd, data, labels):
    """Run 8-microbatch iteration using fused IPC calls (4 calls/mb)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=FusedIPCActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        dummy_mb = torch.zeros(1, 1, 1, dtype=DTYPE)
        mb_labels_list = list(labels.chunk(NUM_MICROBATCHES, dim=0))

        # Warmup
        for _ in range(WARMUP_ITERS):
            for mb_i in range(NUM_MICROBATCHES):
                # 4 calls per microbatch (fused IPC)
                ipc_ref = attn_actor.forward_step_ipc.remote(
                    "attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i]
                )
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)

            # Post-schedule: loss, grad_norm, optimizer
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

            for mb_i in range(NUM_MICROBATCHES):
                ipc_ref = attn_actor.forward_step_ipc.remote(
                    "attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i]
                )
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                grad_ipc_ref = moe_actor.backward_step_ipc.remote("moe", None)
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)

            loss = ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            losses.append(loss)
            times_ms.append(elapsed_ms)
            print(f"  [{name}] Step {step:3d} | Loss: {loss:.6f} | Time: {elapsed_ms:.2f} ms")

        manager.shutdown()
        return times_ms, losses
    finally:
        if ray.is_initialized():
            ray.shutdown()


def run_void_variant(name, config, attn_sd, moe_sd, data, labels):
    """Run 8-microbatch iteration using void returns but separate IPC (6 calls/mb)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=FusedIPCActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        dummy_mb = torch.zeros(1, 1, 1, dtype=DTYPE)
        mb_labels_list = list(labels.chunk(NUM_MICROBATCHES, dim=0))

        # Warmup
        for _ in range(WARMUP_ITERS):
            for mb_i in range(NUM_MICROBATCHES):
                # 6 calls per microbatch (separate IPC, void returns)
                attn_actor.forward_step_ipc.remote(
                    "attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i]
                )  # returns IPC dict (small) but we don't use this ref
                ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                moe_actor.backward_step_ipc.remote("moe", None)
                grad_ipc_ref = moe_actor.create_ipc_for_grad.remote("moe")
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)

            ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

        losses = []
        times_ms = []
        for step in range(TIMED_ITERS):
            t0 = time.perf_counter()

            for mb_i in range(NUM_MICROBATCHES):
                attn_actor.forward_step_ipc.remote(
                    "attn", StageOutputs(activations=dummy_mb), mb_labels_list[mb_i]
                )
                ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
                moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels_list[mb_i])
                moe_actor.backward_step_ipc.remote("moe", None)
                grad_ipc_ref = moe_actor.create_ipc_for_grad.remote("moe")
                attn_actor.backward_from_ipc.remote("attn", grad_ipc_ref)

            loss = ray.get(moe_actor.get_last_loss.remote("moe"))
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
            ray.get(attn_actor.optimizer_step.remote("attn"))
            ray.get(moe_actor.optimizer_step.remote("moe"))

            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            losses.append(loss)
            times_ms.append(elapsed_ms)
            print(f"  [{name}] Step {step:3d} | Loss: {loss:.6f} | Time: {elapsed_ms:.2f} ms")

        manager.shutdown()
        return times_ms, losses
    finally:
        if ray.is_initialized():
            ray.shutdown()


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

    # --- Fused IPC (4 calls/mb) ---
    print("\n=== Fused IPC (4 calls/microbatch) ===")
    times_fused, losses_fused = run_fused_variant(
        "fused", config, attn_sd, moe_sd, data, labels,
    )

    # --- Separate IPC (6 calls/mb) ---
    print("\n=== Separate IPC (6 calls/microbatch) ===")
    times_separate, losses_separate = run_void_variant(
        "separate", config, attn_sd, moe_sd, data, labels,
    )

    # --- Report ---
    arr_fused = np.array(times_fused)
    arr_separate = np.array(times_separate)

    print("\n" + "=" * 60)
    print("E2E RESULTS")
    print("=" * 60)
    print(f"  Fused (4 calls/mb, 32+5=37 total):     {arr_fused.mean():.2f} +/- {arr_fused.std():.2f} ms")
    print(f"  Separate (6 calls/mb, 48+5=53 total):   {arr_separate.mean():.2f} +/- {arr_separate.std():.2f} ms")
    print(f"  Speedup:                                 {arr_separate.mean() / arr_fused.mean():.2f}x")
    print(f"  Calls eliminated per iter:               {53 - 37} ({(53-37)/53*100:.0f}%)")
    print(f"\n  Single-process baseline (A):             ~452 ms (batch=1 x8 grad accum)")
    print(f"  Fused vs A:                              {arr_fused.mean() / 452:.1f}x")

    print(f"\n  Fused final loss:    {losses_fused[-1]:.6f}")
    print(f"  Separate final loss: {losses_separate[-1]:.6f}")
    print(f"  Loss gap:            {abs(losses_fused[-1] - losses_separate[-1]):.6f}")


if __name__ == "__main__":
    main()
