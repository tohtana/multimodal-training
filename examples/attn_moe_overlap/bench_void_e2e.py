"""E2E: Variant B with void returns vs normal returns.

Uses the real pipeline runner with a VoidReturnActor that skips
return-value serialization for forward_step/backward_step.
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
from python.ray.payloads import StageOutputs


class VoidReturnActor(BenchmarkMultiStageActor):
    """BenchmarkMultiStageActor that returns scalars instead of tensors.

    Internal state (_last_forward_outputs, _last_backward_grads) is still
    updated by the parent. IPC methods read from internal state, so they
    still work. The pipeline runner stores the returned ObjectRefs but
    never ray.get()s them for T1 transport.
    """

    def forward_step(self, stage_name, inputs=None, labels=None):
        super().forward_step(stage_name, inputs, labels)
        return True

    def backward_step(self, stage_name, downstream_grad=None):
        super().backward_step(stage_name, downstream_grad)
        return True


def run_variant(name, actor_cls, config, attn_sd, moe_sd, data, labels):
    """Run a full variant B-style benchmark."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init()

    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=actor_cls)
        plan = manager.plan()
        manager.build_models(plan)

        # Pre-load data on source actor
        attn_group = plan.stage_to_actor_group["attn"]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in attn_group.actors
        ])
        dummy_data = torch.zeros(BATCH_SIZE, 1, 1, dtype=DTYPE)

        runner = RayPipelineRunner(pipeline, plan, scheduler=OneFOneBScheduler())

        losses = []
        times_ms = []
        total_iters = WARMUP_ITERS + TIMED_ITERS
        for step in range(total_iters):
            is_timed = step >= WARMUP_ITERS
            t0 = time.perf_counter()
            result = runner.run_iteration(
                data=dummy_data, labels=labels, iteration=step, num_microbatches=NUM_MICROBATCHES,
            )
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000
            loss = result["loss"]
            losses.append(loss)
            if is_timed:
                times_ms.append(elapsed_ms)

            label = "warmup" if not is_timed else "timed"
            print(f"  [{name}] Step {step:3d} ({label}) | Loss: {loss:.6f} | Time: {elapsed_ms:.2f} ms")

        runner.shutdown()
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

    # --- Normal (returns tensors) ---
    print("\n=== Variant B: Normal (returns full tensors) ===")
    times_normal, losses_normal = run_variant(
        "normal", BenchmarkMultiStageActor, config, attn_sd, moe_sd, data, labels,
    )

    # --- Void (returns True) ---
    print("\n=== Variant B: Void (returns True, no serialization) ===")
    times_void, losses_void = run_variant(
        "void", VoidReturnActor, config, attn_sd, moe_sd, data, labels,
    )

    # --- Report ---
    arr_normal = np.array(times_normal)
    arr_void = np.array(times_void)

    print("\n" + "=" * 60)
    print("E2E RESULTS")
    print("=" * 60)
    print(f"  Normal:  {arr_normal.mean():.2f} +/- {arr_normal.std():.2f} ms/iter")
    print(f"  Void:    {arr_void.mean():.2f} +/- {arr_void.std():.2f} ms/iter")
    print(f"  Speedup: {arr_normal.mean() / arr_void.mean():.1f}x")
    print(f"\n  Single-process baseline (A): ~202 ms (batch=1 x8, warm GPU)")
    print(f"  Void vs A overhead: {arr_void.mean() - 202:+.0f} ms ({arr_void.mean() / 202:.1f}x)")

    print(f"\n  Normal final loss: {losses_normal[-1]:.6f}")
    print(f"  Void final loss:   {losses_void[-1]:.6f}")
    print(f"  Loss gap:          {abs(losses_normal[-1] - losses_void[-1]):.6f}")


if __name__ == "__main__":
    main()
