"""Pinpoint: is return-value serialization the bottleneck?

Compares identical compute with two return modes:
  - Normal: return the full StageOutputs/StageGradients tensor (33.6 MB)
  - Void: do the same compute, but return True (tiny scalar)
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
from python.ray.payloads import StageOutputs


class InstrumentedActor(BenchmarkMultiStageActor):
    """Actor with void variants of forward/backward that skip return serialization."""

    def forward_step_void(self, stage_name, inputs=None, labels=None):
        """Run forward but return True instead of the full tensor."""
        super().forward_step(stage_name, inputs, labels)
        return True

    def backward_step_void(self, stage_name, downstream_grad=None):
        """Run backward but return True instead of the full gradient."""
        super().backward_step(stage_name, downstream_grad)
        return True

    def create_ipc_and_forward_void(self, stage_name, ipc_data, labels=None):
        """forward_from_ipc but return True."""
        super().forward_from_ipc(stage_name, ipc_data, labels)
        return True

    def noop(self):
        return True

    def noop_cuda_sync(self):
        torch.cuda.synchronize()
        return True


def main():
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)

    # Tiny dummy matching actual pipeline
    dummy_data = torch.zeros(BATCH_SIZE, 1, 1, dtype=DTYPE)
    dummy_mb = dummy_data.chunk(NUM_MICROBATCHES, dim=0)[0]  # (1, 1, 1)
    mb_labels = labels.chunk(NUM_MICROBATCHES, dim=0)[0]

    N = 30

    ray.init()
    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=InstrumentedActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        # Warmup (run full chain a few times)
        for _ in range(5):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
            ray.get(moe_actor.forward_from_ipc.remote("moe", ipc, mb_labels))
            ray.get(moe_actor.backward_step.remote("moe", None))
            ipc_g = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
            ray.get(attn_actor.backward_step.remote("attn", ipc_g))

        results = {}

        # === Baseline: Ray overhead ===
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.noop.remote())
        t1 = time.perf_counter()
        results["noop"] = (t1 - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.noop_cuda_sync.remote())
        t1 = time.perf_counter()
        results["noop + cuda.sync"] = (t1 - t0) / N * 1000

        # === FORWARD: attn (source stage) ===
        # Normal: returns StageOutputs with (1, 8192, 2048) bf16 = 33.6 MB
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
        t1 = time.perf_counter()
        results["attn.forward (returns 33.6MB)"] = (t1 - t0) / N * 1000

        # Void: same compute, returns True
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step_void.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
        t1 = time.perf_counter()
        results["attn.forward_void (returns True)"] = (t1 - t0) / N * 1000

        # === IPC: create handle ===
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        t1 = time.perf_counter()
        results["attn.create_ipc_for_output"] = (t1 - t0) / N * 1000

        # === FORWARD: moe (from IPC) ===
        ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))

        # Normal: returns StageOutputs with (1, 10) bf16 = 20 bytes (tiny — moe output is logits)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.forward_from_ipc.remote("moe", ipc, mb_labels))
        t1 = time.perf_counter()
        results["moe.forward_from_ipc (returns logits)"] = (t1 - t0) / N * 1000

        # Void: same compute, returns True
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.create_ipc_and_forward_void.remote("moe", ipc, mb_labels))
        t1 = time.perf_counter()
        results["moe.forward_void (returns True)"] = (t1 - t0) / N * 1000

        # === BACKWARD: moe (terminal, no downstream grad) ===
        # Normal: returns StageGradients with grad (1, 8192, 2048) bf16 = 33.6 MB
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.backward_step.remote("moe", None))
        t1 = time.perf_counter()
        results["moe.backward (returns 33.6MB grad)"] = (t1 - t0) / N * 1000

        # Void: same compute, returns True
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.backward_step_void.remote("moe", None))
        t1 = time.perf_counter()
        results["moe.backward_void (returns True)"] = (t1 - t0) / N * 1000

        # === IPC: create grad handle ===
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
        t1 = time.perf_counter()
        results["moe.create_ipc_for_grad"] = (t1 - t0) / N * 1000

        # === BACKWARD: attn (non-terminal, receives IPC grad) ===
        ipc_g = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))

        # Normal: returns StageGradients with grad (1, 8192, 2048) bf16 = 33.6 MB
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.backward_step.remote("attn", ipc_g))
        t1 = time.perf_counter()
        results["attn.backward (returns 33.6MB grad)"] = (t1 - t0) / N * 1000

        # Void: same compute, returns True
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.backward_step_void.remote("attn", ipc_g))
        t1 = time.perf_counter()
        results["attn.backward_void (returns True)"] = (t1 - t0) / N * 1000

        # === Report ===
        print("\n" + "=" * 75)
        print("PER-OPERATION BREAKDOWN (30 iterations, serial ray.get)")
        print("=" * 75)
        for name, ms in results.items():
            print(f"  {name:50s}: {ms:8.2f} ms")

        # Compute serialization cost
        print("\n" + "=" * 75)
        print("RETURN-VALUE SERIALIZATION COST (normal - void)")
        print("=" * 75)
        pairs = [
            ("attn.forward", "attn.forward (returns 33.6MB)", "attn.forward_void (returns True)"),
            ("moe.forward", "moe.forward_from_ipc (returns logits)", "moe.forward_void (returns True)"),
            ("moe.backward", "moe.backward (returns 33.6MB grad)", "moe.backward_void (returns True)"),
            ("attn.backward", "attn.backward (returns 33.6MB grad)", "attn.backward_void (returns True)"),
        ]
        total_ser = 0
        for label, normal_key, void_key in pairs:
            diff = results[normal_key] - results[void_key]
            total_ser += diff
            print(f"  {label:20s}: {diff:+8.2f} ms  ({results[normal_key]:.1f} - {results[void_key]:.1f})")

        print(f"\n  Total serialization per microbatch: {total_ser:.1f} ms")
        print(f"  Total serialization per iteration (8 mb): {total_ser * 8:.0f} ms")

        # Compute per-microbatch total
        per_mb_normal = (
            results["attn.forward (returns 33.6MB)"]
            + results["attn.create_ipc_for_output"]
            + results["moe.forward_from_ipc (returns logits)"]
            + results["moe.backward (returns 33.6MB grad)"]
            + results["moe.create_ipc_for_grad"]
            + results["attn.backward (returns 33.6MB grad)"]
        )
        per_mb_void = (
            results["attn.forward_void (returns True)"]
            + results["attn.create_ipc_for_output"]
            + results["moe.forward_void (returns True)"]
            + results["moe.backward_void (returns True)"]
            + results["moe.create_ipc_for_grad"]
            + results["attn.backward_void (returns True)"]
        )
        print(f"\n" + "=" * 75)
        print("ESTIMATED ITERATION TIME (8 microbatches)")
        print("=" * 75)
        print(f"  Per-microbatch (normal, 6 calls): {per_mb_normal:.1f} ms")
        print(f"  Per-microbatch (void, 6 calls):   {per_mb_void:.1f} ms")
        print(f"  8-microbatch (normal):            {per_mb_normal * 8:.0f} ms")
        print(f"  8-microbatch (void):              {per_mb_void * 8:.0f} ms")
        print(f"  Single-process baseline (A):      ~202 ms (batch=1 x8, warm GPU)")
        print(f"\n  Ray base overhead: {results['noop']:.1f} ms/call × 48 calls = {results['noop'] * 48:.0f} ms")

        manager.shutdown()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
