"""Measure IPC handle and per-op overhead in isolation."""

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


def main():
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)

    ray.init()
    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=BenchmarkMultiStageActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_group = plan.stage_to_actor_group["attn"]
        moe_group = plan.stage_to_actor_group["moe"]
        ray.get([a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES) for a in attn_group.actors])

        attn_actor = attn_group.actors[0]
        moe_actor = moe_group.actors[0]

        # Use microbatch-sized data
        mb_data = data.chunk(NUM_MICROBATCHES, dim=0)[0]
        mb_labels = labels.chunk(NUM_MICROBATCHES, dim=0)[0]

        # Warmup
        ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels))
        ipc_data = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_data, mb_labels))
        ray.get(moe_actor.backward_step.remote("moe", None))
        ipc_grad = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
        ray.get(attn_actor.backward_step.remote("attn", ipc_grad))

        N = 30
        results = {}

        # 1. attn.forward_step (source stage with preloaded data)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels))
        t1 = time.perf_counter()
        results["attn.forward_step"] = (t1 - t0) / N * 1000

        # 2. attn.create_ipc_for_output
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        t1 = time.perf_counter()
        results["attn.create_ipc_for_output"] = (t1 - t0) / N * 1000

        # 3. moe.forward_from_ipc (reusing same handle)
        ipc_data = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_data, mb_labels))
        t1 = time.perf_counter()
        results["moe.forward_from_ipc (cached handle)"] = (t1 - t0) / N * 1000

        # 4. moe.forward_from_ipc (fresh handle each time)
        t0 = time.perf_counter()
        for _ in range(N):
            ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
            ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels))
        t1 = time.perf_counter()
        results["create_ipc + forward_from_ipc (chain)"] = (t1 - t0) / N * 1000

        # 5. moe.backward_step (terminal, no downstream grad)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.backward_step.remote("moe", None))
        t1 = time.perf_counter()
        results["moe.backward_step (terminal)"] = (t1 - t0) / N * 1000

        # 6. moe.create_ipc_for_grad
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
        t1 = time.perf_counter()
        results["moe.create_ipc_for_grad"] = (t1 - t0) / N * 1000

        # 7. attn.backward_step (with IPC grad, cached handle)
        ipc_grad = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.backward_step.remote("attn", ipc_grad))
        t1 = time.perf_counter()
        results["attn.backward_step (cached IPC grad)"] = (t1 - t0) / N * 1000

        # 8. Full microbatch chain (6 serial calls)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels))
            ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
            ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels))
            ray.get(moe_actor.backward_step.remote("moe", None))
            ipc_grad_ref = moe_actor.create_ipc_for_grad.remote("moe")
            ray.get(attn_actor.backward_step.remote("attn", ipc_grad_ref))
        t1 = time.perf_counter()
        results["full microbatch chain (6 calls)"] = (t1 - t0) / N * 1000

        # 9. Full microbatch chain fire-and-forget (let Ray handle deps)
        t0 = time.perf_counter()
        for _ in range(N):
            _f_attn = attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels)
            _ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
            _f_moe = moe_actor.forward_from_ipc.remote("moe", _ipc_ref, mb_labels)
            _b_moe = moe_actor.backward_step.remote("moe", None)
            _ipc_grad_ref = moe_actor.create_ipc_for_grad.remote("moe")
            _b_attn = attn_actor.backward_step.remote("attn", _ipc_grad_ref)
        # Wait only for the final ref of the last iteration
        ray.get(_b_attn)
        t1 = time.perf_counter()
        results["full chain fire-forget (no mid-get)"] = (t1 - t0) / N * 1000

        # 10. get_last_loss
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.get_last_loss.remote("moe"))
        t1 = time.perf_counter()
        results["moe.get_last_loss"] = (t1 - t0) / N * 1000

        # 11. compute_grad_norm_sq
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.compute_grad_norm_sq.remote("attn"))
        t1 = time.perf_counter()
        results["attn.compute_grad_norm_sq"] = (t1 - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.compute_grad_norm_sq.remote("moe"))
        t1 = time.perf_counter()
        results["moe.compute_grad_norm_sq"] = (t1 - t0) / N * 1000

        # 12. optimizer_step
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.optimizer_step.remote("attn"))
        t1 = time.perf_counter()
        results["attn.optimizer_step"] = (t1 - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.optimizer_step.remote("moe"))
        t1 = time.perf_counter()
        results["moe.optimizer_step"] = (t1 - t0) / N * 1000

        # Report
        print("\n=== Per-Operation Overhead (30 iterations each) ===")
        for name, ms in results.items():
            print(f"  {name:45s}: {ms:8.2f} ms/call")

        # Extrapolate full iteration
        print("\n=== Extrapolated Full Iteration (8 microbatches) ===")
        chain_ms = results["full microbatch chain (6 calls)"]
        chain_ff_ms = results["full chain fire-forget (no mid-get)"]
        loss_ms = results["moe.get_last_loss"]
        norm_ms = results["attn.compute_grad_norm_sq"] + results["moe.compute_grad_norm_sq"]
        opt_ms = results["attn.optimizer_step"] + results["moe.optimizer_step"]

        total_serial = chain_ms * 8 + loss_ms + norm_ms + opt_ms
        total_ff = chain_ff_ms * 8 + loss_ms + norm_ms + opt_ms
        print(f"  8 × microbatch chain (serial get):  {chain_ms * 8:.1f} ms")
        print(f"  8 × microbatch chain (fire-forget):  {chain_ff_ms * 8:.1f} ms")
        print(f"  loss + grad_norm + optimizer:        {loss_ms + norm_ms + opt_ms:.1f} ms")
        print(f"  Predicted total (serial):            {total_serial:.1f} ms")
        print(f"  Predicted total (fire-forget):       {total_ff:.1f} ms")
        print(f"  Variant A (single-process) baseline: ~202 ms (batch=1 x8, warm GPU)")

        manager.shutdown()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
