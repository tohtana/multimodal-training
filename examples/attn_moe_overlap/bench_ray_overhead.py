"""Benchmark Ray remote call overhead for Variant B pipeline.

Measures:
1. Per-step dispatch + execution time in the pipeline schedule
2. Post-schedule overhead (loss, grad_norm, optimizer)
3. IPC handle creation/reconstruction cost
4. Bare Ray noop / small-tensor call overhead
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
    BASE_LR,
    BATCH_SIZE,
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
from python.ray.payloads import StageOutputs


def measure_bare_ray_overhead():
    """Measure bare Ray remote call overhead with no-op and small tasks."""

    @ray.remote(num_gpus=0.5)
    class NoopActor:
        def noop(self):
            return True

        def echo_small(self, x):
            return x

        def echo_tensor(self, t):
            return t.shape

        def gpu_noop(self):
            torch.cuda.synchronize()
            return True

    actor = NoopActor.remote()
    ray.get(actor.noop.remote())  # warmup

    N = 50
    # Noop
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(actor.noop.remote())
    t1 = time.perf_counter()
    noop_ms = (t1 - t0) / N * 1000

    # Small python object
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(actor.echo_small.remote(42))
    t1 = time.perf_counter()
    echo_small_ms = (t1 - t0) / N * 1000

    # Small CPU tensor (1KB)
    small_t = torch.zeros(128)
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(actor.echo_tensor.remote(small_t))
    t1 = time.perf_counter()
    echo_1kb_ms = (t1 - t0) / N * 1000

    # Medium CPU tensor (1MB)
    med_t = torch.zeros(256 * 1024)
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(actor.echo_tensor.remote(med_t))
    t1 = time.perf_counter()
    echo_1mb_ms = (t1 - t0) / N * 1000

    # GPU synchronize noop
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(actor.gpu_noop.remote())
    t1 = time.perf_counter()
    gpu_noop_ms = (t1 - t0) / N * 1000

    # Fire-and-forget chain (no intermediate ray.get)
    t0 = time.perf_counter()
    refs = [actor.noop.remote() for _ in range(N)]
    ray.get(refs)
    t1 = time.perf_counter()
    batch_noop_ms = (t1 - t0) / N * 1000

    print("\n=== Bare Ray Call Overhead ===")
    print(f"  noop (serial ray.get):        {noop_ms:.2f} ms/call")
    print(f"  echo(42):                     {echo_small_ms:.2f} ms/call")
    print(f"  echo(1KB tensor):             {echo_1kb_ms:.2f} ms/call")
    print(f"  echo(1MB tensor):             {echo_1mb_ms:.2f} ms/call")
    print(f"  gpu_noop (cuda.sync):         {gpu_noop_ms:.2f} ms/call")
    print(f"  noop (batch {N}, one get):    {batch_noop_ms:.2f} ms/call")

    ray.kill(actor)


def measure_pipeline_breakdown():
    """Measure per-phase timing in a Variant B iteration."""
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)

    resource_sets, placements = _make_single_gpu_resources()
    pipeline = _make_pipeline(resource_sets, placements)
    specs = _make_specs(config, attn_sd, moe_sd)

    manager = PlacementManager(pipeline, model_specs=specs, actor_cls=BenchmarkMultiStageActor)
    plan = manager.plan()
    manager.build_models(plan)

    # Preload data
    attn_group = plan.stage_to_actor_group["attn"]
    ray.get([a.preload_data.remote("attn", data, labels, NUM_MICROBATCHES) for a in attn_group.actors])
    dummy_data = torch.zeros(BATCH_SIZE, 1, 1, dtype=DTYPE)

    scheduler = OneFOneBScheduler()
    runner = RayPipelineRunner(pipeline, plan, scheduler=scheduler)

    # Warmup
    for i in range(3):
        runner.run_iteration(data=dummy_data, labels=labels, iteration=i, num_microbatches=NUM_MICROBATCHES)

    # Instrumented run: manually replicate _run_iteration_mixed with timing
    from python.pipeline.dag import PipelineDAG

    dag = PipelineDAG(pipeline)
    topo_order = dag.topological_sort()
    schedule = scheduler.generate_schedule(topo_order, NUM_MICROBATCHES)
    router = runner.router

    mb_data, mb_labels = RayPipelineRunner._split_batch(dummy_data, labels, NUM_MICROBATCHES)

    stage_output_refs = {}
    stage_grad_refs = {}
    step_times = []

    t_iter_start = time.perf_counter()

    # Schedule dispatch loop
    t_dispatch_start = time.perf_counter()
    for step_idx, step in enumerate(schedule):
        group = plan.stage_to_actor_group[step.stage_name]
        mb_id = step.microbatch_id

        t_step = time.perf_counter()

        if step.op.value == "forward":
            preds = dag.predecessors(step.stage_name)
            if not preds:
                refs = [
                    actor.forward_step.remote(
                        step.stage_name, StageOutputs(activations=mb_data[mb_id]), mb_labels[mb_id], mb_id
                    )
                    for actor in group.actors
                ]
            else:
                pred_name = preds[0]
                pred_group = plan.stage_to_actor_group[pred_name]
                routing = router.get_routing(pred_name, step.stage_name)
                refs = []
                for dst_rank, actor in enumerate(group.actors):
                    src_rank = routing.dst_to_src[dst_rank]
                    transport = router.get_actor_pair_transport(pred_name, step.stage_name, src_rank, dst_rank)
                    if transport == "t1":
                        ipc_ref = pred_group.actors[src_rank].create_ipc_for_output.remote(pred_name, mb_id)
                        ref = actor.forward_from_ipc.remote(step.stage_name, ipc_ref, mb_labels[mb_id], mb_id)
                    else:
                        pred_refs = stage_output_refs[pred_name][mb_id]
                        ref = actor.forward_step.remote(step.stage_name, pred_refs[src_rank], mb_labels[mb_id], mb_id)
                    refs.append(ref)
            stage_output_refs.setdefault(step.stage_name, {})[mb_id] = refs

        else:  # BACKWARD
            succs = dag.successors(step.stage_name)
            if not succs:
                refs = [actor.backward_step.remote(step.stage_name, None, mb_id) for actor in group.actors]
            else:
                succ_name = succs[0]
                succ_group = plan.stage_to_actor_group[succ_name]
                routing = router.get_routing(step.stage_name, succ_name)
                refs = []
                for src_rank, actor in enumerate(group.actors):
                    dst_ranks = routing.src_to_dst[src_rank]
                    # Gather grads with transport
                    if not stage_grad_refs.get(succ_name, {}).get(mb_id) or not dst_ranks:
                        grad_ref = None
                    elif len(dst_ranks) == 1:
                        dst_rank = dst_ranks[0]
                        transport = router.get_actor_pair_transport(step.stage_name, succ_name, src_rank, dst_rank)
                        if transport == "t1":
                            grad_ref = succ_group.actors[dst_rank].create_ipc_for_grad.remote(succ_name, mb_id)
                        else:
                            grad_ref = stage_grad_refs[succ_name][mb_id][dst_rank]
                    else:
                        grad_ref = None  # simplification
                    ref = actor.backward_step.remote(step.stage_name, grad_ref, mb_id)
                    refs.append(ref)
            stage_grad_refs.setdefault(step.stage_name, {})[mb_id] = refs

        t_step_end = time.perf_counter()
        step_times.append((step_idx, step.op.value, step.stage_name, mb_id, (t_step_end - t_step) * 1000))

    t_dispatch_end = time.perf_counter()

    # Post-schedule: loss
    t_loss_start = time.perf_counter()
    loss_value = None
    for stage_cfg in pipeline.stages:
        if stage_cfg.is_terminal:
            group = plan.stage_to_actor_group[stage_cfg.name]
            loss_value = ray.get(group.actors[0].get_last_loss.remote(stage_cfg.name))
    t_loss_end = time.perf_counter()

    # Post-schedule: grad norm
    t_norm_start = time.perf_counter()
    import math

    total_norm_sq = 0.0
    for name in topo_order:
        group = plan.stage_to_actor_group[name]
        norm_sq = ray.get(group.actors[0].compute_grad_norm_sq.remote(name))
        total_norm_sq += norm_sq
    global_grad_norm = math.sqrt(total_norm_sq)
    t_norm_end = time.perf_counter()

    # Post-schedule: optimizer
    t_opt_start = time.perf_counter()
    opt_refs = []
    for name in topo_order:
        group = plan.stage_to_actor_group[name]
        for actor in group.actors:
            ref = actor.optimizer_step.remote(name)
            opt_refs.append(ref)
    ray.get(opt_refs)
    t_opt_end = time.perf_counter()

    t_iter_end = time.perf_counter()

    # Report
    dispatch_ms = (t_dispatch_end - t_dispatch_start) * 1000
    loss_ms = (t_loss_end - t_loss_start) * 1000
    norm_ms = (t_norm_end - t_norm_start) * 1000
    opt_ms = (t_opt_end - t_opt_start) * 1000
    total_ms = (t_iter_end - t_iter_start) * 1000

    print("\n=== Pipeline Iteration Breakdown ===")
    print(f"  Total iteration:              {total_ms:.2f} ms")
    print(f"  Schedule dispatch (driver):   {dispatch_ms:.2f} ms  ({len(schedule)} steps)")
    print(f"  get_last_loss:                {loss_ms:.2f} ms")
    print(f"  compute_grad_norm (×{len(topo_order)}):      {norm_ms:.2f} ms")
    print(f"  optimizer_step (×{len(topo_order)}):         {opt_ms:.2f} ms")
    print(f"  Sum of phases:                {dispatch_ms + loss_ms + norm_ms + opt_ms:.2f} ms")

    # Per-step dispatch timing
    print(f"\n--- Per-Step Driver Dispatch Time ---")
    for idx, op, stage, mb, ms in step_times[:8]:
        print(f"  Step {idx:3d}: {op:8s} {stage:5s} mb={mb} → {ms:.3f} ms")
    print(f"  ... ({len(step_times)} total steps)")
    total_dispatch = sum(ms for _, _, _, _, ms in step_times)
    print(f"  Total step dispatch: {total_dispatch:.2f} ms")
    print(f"  Mean per step: {total_dispatch / len(step_times):.3f} ms")

    # Now measure: how long does ray.get on the last ref take?
    # This tells us how long the actual execution takes (dispatch is non-blocking)
    print(f"\n--- Blocking Execution Time ---")
    print(f"  The get_last_loss call ({loss_ms:.2f} ms) is the first ray.get() after dispatch.")
    print(f"  It blocks until all prior tasks complete (due to actor task ordering).")
    print(f"  So {loss_ms:.2f} ms includes the ENTIRE schedule execution time.")

    runner.shutdown()
    manager.shutdown()


def measure_ipc_overhead():
    """Measure IPC handle creation and reconstruction overhead."""
    config = create_qwen3_config()

    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_sd = attn_model.state_dict()
    moe_sd = moe_model.state_dict()

    torch.manual_seed(42)
    data, labels = generate_dummy_batch(config, BATCH_SIZE, SEQ_LEN, NUM_CLASSES, "cpu", DTYPE)

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

    # Run one forward to populate last_forward_outputs
    mb_data = data.chunk(NUM_MICROBATCHES, dim=0)[0]
    mb_labels = labels.chunk(NUM_MICROBATCHES, dim=0)[0]
    ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels, 0))

    N = 50

    # Measure create_ipc_for_output
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(attn_actor.create_ipc_for_output.remote("attn", 0))
    t1 = time.perf_counter()
    create_ipc_ms = (t1 - t0) / N * 1000

    # Measure forward_from_ipc (create + reconstruct + forward)
    t0 = time.perf_counter()
    for _ in range(N):
        ipc_ref = attn_actor.create_ipc_for_output.remote("attn", 0)
        ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_ref, mb_labels, 0))
    t1 = time.perf_counter()
    full_ipc_fwd_ms = (t1 - t0) / N * 1000

    # Measure forward_step directly (no IPC, just object store)
    # First get a StageOutputs ref from attn
    t0 = time.perf_counter()
    for _ in range(N):
        fwd_ref = attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels, 0)
        ray.get(moe_actor.forward_step.remote("moe", fwd_ref, mb_labels, 0))
    t1 = time.perf_counter()
    direct_fwd_ms = (t1 - t0) / N * 1000

    # Measure just attn forward
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(attn_actor.forward_step.remote("attn", StageOutputs(activations=mb_data), mb_labels, 0))
    t1 = time.perf_counter()
    attn_fwd_ms = (t1 - t0) / N * 1000

    # Measure just moe forward (using IPC handle from attn)
    ipc_data = ray.get(attn_actor.create_ipc_for_output.remote("attn", 0))
    t0 = time.perf_counter()
    for _ in range(N):
        ray.get(moe_actor.forward_from_ipc.remote("moe", ipc_data, mb_labels, 0))
    t1 = time.perf_counter()
    moe_fwd_ipc_ms = (t1 - t0) / N * 1000

    print("\n=== IPC & Forward Overhead ===")
    print(f"  create_ipc_for_output:        {create_ipc_ms:.2f} ms/call")
    print(f"  IPC path (create+fwd_ipc):    {full_ipc_fwd_ms:.2f} ms/call")
    print(f"  ObjectRef path (fwd+fwd):     {direct_fwd_ms:.2f} ms/call")
    print(f"  attn forward_step only:       {attn_fwd_ms:.2f} ms/call")
    print(f"  moe forward_from_ipc only:    {moe_fwd_ipc_ms:.2f} ms/call")

    runner_shutdown = None  # no runner created here
    manager.shutdown()


def main():
    ray.init()
    try:
        measure_bare_ray_overhead()
    finally:
        ray.shutdown()

    ray.init()
    try:
        measure_pipeline_breakdown()
    finally:
        ray.shutdown()

    ray.init()
    try:
        measure_ipc_overhead()
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
