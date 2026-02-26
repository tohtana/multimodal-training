"""Detailed breakdown of Variant B void overhead.

Isolates:
1. Pure CUDA compute (no Ray) per microbatch
2. Ray scheduling gap (dependency resolution latency)
3. IPC handle creation vs reuse
4. Per-microbatch chain: isolated serial vs pipelined fire-forget
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
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import OneFOneBScheduler
from python.ray.payloads import StageGradients, StageOutputs


class InstrumentedVoidActor(BenchmarkMultiStageActor):
    """Void-return actor with additional measurement methods."""

    def forward_step(self, stage_name, inputs=None, labels=None):
        super().forward_step(stage_name, inputs, labels)
        return True

    def backward_step(self, stage_name, downstream_grad=None):
        super().backward_step(stage_name, downstream_grad)
        return True

    def forward_step_void_no_ipc(self, stage_name, inputs=None, labels=None):
        """Forward with preloaded data, no IPC, return True."""
        super().forward_step(stage_name, inputs, labels)
        return True

    def forward_from_ipc_void(self, stage_name, ipc_data, labels=None):
        """Reconstruct from IPC + forward, return True."""
        super().forward_from_ipc(stage_name, ipc_data, labels)
        return True

    def backward_with_ipc_void(self, stage_name, ipc_data):
        """Reconstruct grad from IPC + backward, return True."""
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
        return True

    def noop(self):
        return True

    def cuda_sync(self):
        torch.cuda.synchronize()
        return True

    # --- IPC reuse test ---
    def create_and_cache_ipc_for_output(self, stage_name):
        """Create IPC handle and cache it for reuse."""
        result = self._last_forward_outputs.get(stage_name)
        if result is None or result.activations is None:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.activations.detach())
        self._cached_ipc_output = {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }
        return self._cached_ipc_output

    def get_cached_ipc_for_output(self):
        """Return previously cached IPC handle (no new cudaIpcGetMemHandle)."""
        return getattr(self, "_cached_ipc_output", None)

    def create_and_cache_ipc_for_grad(self, stage_name):
        """Create IPC handle for grad and cache it."""
        result = self._last_backward_grads.get(stage_name)
        if result is None or result.grad is None:
            return None
        from python.ray.tensor_transfer import create_ipc_handle
        handle, gpu_id, event_handle = create_ipc_handle(result.grad)
        self._cached_ipc_grad = {
            "__ipc__": True,
            "ipc_handle": handle,
            "gpu_id": gpu_id,
            "event_handle": event_handle,
            "meta": result.meta,
        }
        return self._cached_ipc_grad

    def get_cached_ipc_for_grad(self):
        """Return previously cached IPC grad handle."""
        return getattr(self, "_cached_ipc_grad", None)


def measure_pure_cuda_compute():
    """Measure forward/backward without Ray (pure CUDA on driver)."""
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_stage = Qwen3AttentionStage(config, dtype=DTYPE).cuda()
    moe_stage = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE).cuda()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        list(attn_stage.parameters()) + list(moe_stage.parameters()),
        lr=BASE_LR, foreach=False,
    )

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cuda", DTYPE)
    # Microbatch
    mb_data = data.chunk(NUM_MICROBATCHES, dim=0)[0]
    mb_labels = labels.chunk(NUM_MICROBATCHES, dim=0)[0]

    # Warmup
    for _ in range(5):
        h = attn_stage(mb_data)
        logits = moe_stage(h)
        loss = loss_fn(logits, mb_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    N = 50

    # Per-stage forward timing
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        h = attn_stage(mb_data)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    attn_fwd_ms = (t1 - t0) / N * 1000

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        logits = moe_stage(h.detach())
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    moe_fwd_ms = (t1 - t0) / N * 1000

    # Full microbatch (fwd + bwd + opt)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        h = attn_stage(mb_data)
        logits = moe_stage(h)
        loss = loss_fn(logits, mb_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    full_mb_ms = (t1 - t0) / N * 1000

    # Full iteration (8 microbatches with grad accumulation)
    all_mbs_data = list(data.chunk(NUM_MICROBATCHES, dim=0))
    all_mbs_labels = list(labels.chunk(NUM_MICROBATCHES, dim=0))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N):
        for mb_i in range(NUM_MICROBATCHES):
            h = attn_stage(all_mbs_data[mb_i])
            logits = moe_stage(h)
            loss = loss_fn(logits, all_mbs_labels[mb_i])
            (loss / NUM_MICROBATCHES).backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    full_iter_ms = (t1 - t0) / N * 1000

    print("\n=== Pure CUDA Compute (no Ray, no IPC) ===")
    print(f"  attn forward (batch=1):      {attn_fwd_ms:.2f} ms")
    print(f"  moe forward (batch=1):       {moe_fwd_ms:.2f} ms")
    print(f"  full microbatch (fwd+bwd):   {full_mb_ms:.2f} ms")
    print(f"  full iteration (8 mb):       {full_iter_ms:.2f} ms")
    print(f"  variant A (batch=1 x8, warm):~202 ms")
    return full_iter_ms


def measure_ray_scheduling_gap():
    """Measure the gap between dependent Ray tasks (scheduling latency)."""
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)
    dummy_mb = torch.zeros(1, 1, 1, dtype=DTYPE)
    mb_labels = labels.chunk(NUM_MICROBATCHES, dim=0)[0]

    ray.init()
    try:
        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_sd, moe_sd)

        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=InstrumentedVoidActor)
        plan = manager.plan()
        manager.build_models(plan)

        attn_actor = plan.stage_to_actor_group["attn"].actors[0]
        moe_actor = plan.stage_to_actor_group["moe"].actors[0]
        ray.get([
            a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES)
            for a in plan.stage_to_actor_group["attn"].actors
        ])

        # Warmup
        for _ in range(5):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
            ray.get(moe_actor.forward_from_ipc_void.remote("moe", ipc, mb_labels))
            ray.get(moe_actor.backward_step.remote("moe", None))
            ipc_g = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
            ray.get(attn_actor.backward_with_ipc_void.remote("attn", ipc_g))

        N = 30
        results = {}

        # --- A) Noop baseline ---
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.noop.remote())
        t1 = time.perf_counter()
        results["noop (serial)"] = (t1 - t0) / N * 1000

        # Chain of 6 noops across 2 actors (no dependency)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.noop.remote())
            ray.get(attn_actor.noop.remote())
            ray.get(moe_actor.noop.remote())
            ray.get(moe_actor.noop.remote())
            ray.get(moe_actor.noop.remote())
            ray.get(attn_actor.noop.remote())
        t1 = time.perf_counter()
        results["6 noops serial (3 attn + 3 moe)"] = (t1 - t0) / N * 1000

        # Chain of 6 noops with fire-and-forget (data deps via ObjectRef)
        t0 = time.perf_counter()
        for _ in range(N):
            r1 = attn_actor.noop.remote()
            r2 = attn_actor.noop.remote()       # depends on r1 (same actor)
            r3 = moe_actor.noop.remote()         # independent
            r4 = moe_actor.noop.remote()         # depends on r3 (same actor)
            r5 = moe_actor.noop.remote()         # depends on r4 (same actor)
            r6 = attn_actor.noop.remote()        # depends on r2 (same actor)
        ray.get(r6)
        t1 = time.perf_counter()
        results["6 noops fire-forget (last get)"] = (t1 - t0) / N * 1000

        # --- B) Full microbatch chain: serial ray.get between each call ---
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
            ray.get(moe_actor.forward_from_ipc_void.remote("moe", ipc, mb_labels))
            ray.get(moe_actor.backward_step.remote("moe", None))
            ipc_g = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
            ray.get(attn_actor.backward_with_ipc_void.remote("attn", ipc_g))
        t1 = time.perf_counter()
        results["microbatch chain (serial get)"] = (t1 - t0) / N * 1000

        # --- C) Full microbatch chain: fire-and-forget (like pipeline runner) ---
        t0 = time.perf_counter()
        for _ in range(N):
            _f_attn = attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels)
            _ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
            _f_moe = moe_actor.forward_from_ipc_void.remote("moe", _ipc_ref, mb_labels)
            _b_moe = moe_actor.backward_step.remote("moe", None)
            _ipc_g_ref = moe_actor.create_ipc_for_grad.remote("moe")
            _b_attn = attn_actor.backward_with_ipc_void.remote("attn", _ipc_g_ref)
        ray.get(_b_attn)
        t1 = time.perf_counter()
        results["microbatch chain (fire-forget)"] = (t1 - t0) / N * 1000

        # --- D) 8 microbatch chains: serial ---
        t0 = time.perf_counter()
        for _ in range(N):
            for _ in range(8):
                ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
                ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
                ray.get(moe_actor.forward_from_ipc_void.remote("moe", ipc, mb_labels))
                ray.get(moe_actor.backward_step.remote("moe", None))
                ipc_g = ray.get(moe_actor.create_ipc_for_grad.remote("moe"))
                ray.get(attn_actor.backward_with_ipc_void.remote("attn", ipc_g))
        t1 = time.perf_counter()
        results["8 mb chains (serial get)"] = (t1 - t0) / N * 1000

        # --- E) 8 microbatch chains: fire-forget each mb, serial across mbs ---
        t0 = time.perf_counter()
        for _ in range(N):
            for _ in range(8):
                _f_attn = attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels)
                _ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
                _f_moe = moe_actor.forward_from_ipc_void.remote("moe", _ipc_ref, mb_labels)
                _b_moe = moe_actor.backward_step.remote("moe", None)
                _ipc_g_ref = moe_actor.create_ipc_for_grad.remote("moe")
                _b_attn = attn_actor.backward_with_ipc_void.remote("attn", _ipc_g_ref)
            ray.get(_b_attn)
        t1 = time.perf_counter()
        results["8 mb chains (fire-forget, 1 final get)"] = (t1 - t0) / N * 1000

        # --- F) IPC handle reuse test ---
        # Fresh handle each time
        t0 = time.perf_counter()
        for _ in range(N):
            ipc = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        t1 = time.perf_counter()
        results["create_ipc_for_output (fresh)"] = (t1 - t0) / N * 1000

        # Cached handle (first call caches, subsequent calls return cached)
        ray.get(attn_actor.create_and_cache_ipc_for_output.remote("attn"))
        t0 = time.perf_counter()
        for _ in range(N):
            ipc = ray.get(attn_actor.get_cached_ipc_for_output.remote())
        t1 = time.perf_counter()
        results["get_cached_ipc (reuse)"] = (t1 - t0) / N * 1000

        # Is the handle still valid for reconstruction?
        cached_ipc = ray.get(attn_actor.get_cached_ipc_for_output.remote())
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(moe_actor.forward_from_ipc_void.remote("moe", cached_ipc, mb_labels))
        t1 = time.perf_counter()
        results["forward_from_ipc (cached handle)"] = (t1 - t0) / N * 1000

        # forward_from_ipc with fresh handle for comparison
        t0 = time.perf_counter()
        for _ in range(N):
            ipc_ref = attn_actor.create_ipc_for_output.remote("attn")
            ray.get(moe_actor.forward_from_ipc_void.remote("moe", ipc_ref, mb_labels))
        t1 = time.perf_counter()
        results["forward_from_ipc (fresh handle chain)"] = (t1 - t0) / N * 1000

        # Same for grad (must run forward+backward first to populate state)
        ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
        ipc_tmp = ray.get(attn_actor.create_ipc_for_output.remote("attn"))
        ray.get(moe_actor.forward_from_ipc_void.remote("moe", ipc_tmp, mb_labels))
        ray.get(moe_actor.backward_step.remote("moe", None))
        ray.get(moe_actor.create_and_cache_ipc_for_grad.remote("moe"))
        cached_ipc_g = ray.get(moe_actor.get_cached_ipc_for_grad.remote())

        t0 = time.perf_counter()
        for _ in range(N):
            # Re-run forward each time so backward has activations
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ray.get(attn_actor.backward_with_ipc_void.remote("attn", cached_ipc_g))
        t1 = time.perf_counter()
        results["backward_ipc (cached grad handle)"] = (t1 - t0) / N * 1000

        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ipc_g_ref = moe_actor.create_ipc_for_grad.remote("moe")
            ray.get(attn_actor.backward_with_ipc_void.remote("attn", ipc_g_ref))
        t1 = time.perf_counter()
        results["backward_ipc (fresh grad handle chain)"] = (t1 - t0) / N * 1000

        # --- G) Full chain with cached IPC handles ---
        # Run a full chain to populate both actors' state, then cache handles
        ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
        ray.get(attn_actor.create_and_cache_ipc_for_output.remote("attn"))
        cached_fwd_ipc = ray.get(attn_actor.get_cached_ipc_for_output.remote())
        ray.get(moe_actor.forward_from_ipc_void.remote("moe", cached_fwd_ipc, mb_labels))
        ray.get(moe_actor.backward_step.remote("moe", None))
        ray.get(moe_actor.create_and_cache_ipc_for_grad.remote("moe"))
        cached_bwd_ipc = ray.get(moe_actor.get_cached_ipc_for_grad.remote())

        # Microbatch with cached handles (skip create_ipc calls entirely)
        t0 = time.perf_counter()
        for _ in range(N):
            ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=dummy_mb), mb_labels))
            ray.get(moe_actor.forward_from_ipc_void.remote("moe", cached_fwd_ipc, mb_labels))
            ray.get(moe_actor.backward_step.remote("moe", None))
            ray.get(attn_actor.backward_with_ipc_void.remote("attn", cached_bwd_ipc))
        t1 = time.perf_counter()
        results["microbatch (cached IPC, 4 calls)"] = (t1 - t0) / N * 1000

        # --- Report ---
        print("\n=== Ray Scheduling & IPC Breakdown ===")
        for name, ms in results.items():
            print(f"  {name:50s}: {ms:8.2f} ms")

        # Summary
        print("\n=== Summary ===")
        noop_6 = results["6 noops serial (3 attn + 3 moe)"]
        chain_serial = results["microbatch chain (serial get)"]
        chain_ff = results["microbatch chain (fire-forget)"]
        chain_cached = results["microbatch (cached IPC, 4 calls)"]
        iter_serial = results["8 mb chains (serial get)"]
        iter_ff = results["8 mb chains (fire-forget, 1 final get)"]

        print(f"  Per-microbatch:")
        print(f"    6 noops (pure scheduling):   {noop_6:.1f} ms")
        print(f"    6 calls serial (IPC chain):  {chain_serial:.1f} ms")
        print(f"    4 calls cached IPC:          {chain_cached:.1f} ms")
        print(f"    6 calls fire-forget:         {chain_ff:.1f} ms")
        print(f"  Per-iteration (8 mb):")
        print(f"    Serial get:                  {iter_serial:.0f} ms")
        print(f"    Fire-forget:                 {iter_ff:.0f} ms")
        print(f"    Single-process baseline:     ~202 ms (batch=1 x8, warm GPU)")

        ipc_fresh = results["create_ipc_for_output (fresh)"]
        ipc_cached = results["get_cached_ipc (reuse)"]
        print(f"\n  IPC handle overhead:")
        print(f"    Fresh create:  {ipc_fresh:.2f} ms")
        print(f"    Cached reuse:  {ipc_cached:.2f} ms")
        print(f"    Savings:       {ipc_fresh - ipc_cached:.2f} ms/call")

        manager.shutdown()
    finally:
        ray.shutdown()


def main():
    cuda_compute_ms = measure_pure_cuda_compute()
    measure_ray_scheduling_gap()


if __name__ == "__main__":
    main()
