"""T1 microbatch correctness and cache-lifecycle regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import ray
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import PipelineIterationError, RayPipelineRunner
from python.pipeline.scheduler import GPipeScheduler
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage
from python.ray.multi_stage_actor import MultiStageActor
from python.ray.payloads import StageOutputs
from python.ray.test_cpu_t1_actor import CpuT1IpcActor

INPUT_DIM = 8
HIDDEN_DIM = 16
OUTPUT_DIM = 4
BATCH_SIZE = 16
SEED = 7
LR = 0.01


def _build_pipeline() -> Pipeline:
    return Pipeline(
        stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
        edges=[EdgeConfig(src="a", dst="b")],
        resource_sets=[
            ResourceSet(name="parent", num_gpus=1, device_ids=(0,)),
            ResourceSet(name="child", num_gpus=1, device_ids=(0,), subset_of="parent"),
        ],
        placements=[
            Placement(stage_name="a", resource_set="child"),
            Placement(stage_name="b", resource_set="parent"),
        ],
    )


def _build_model_specs() -> list[StageModelSpec]:
    torch.manual_seed(SEED)
    sd_a = nn.Linear(INPUT_DIM, HIDDEN_DIM).state_dict()

    torch.manual_seed(SEED + 1)
    sd_b = nn.Linear(HIDDEN_DIM, OUTPUT_DIM).state_dict()

    return [
        StageModelSpec(
            stage_name="a",
            model_cls=nn.Linear,
            model_kwargs={"in_features": INPUT_DIM, "out_features": HIDDEN_DIM},
            state_dict=sd_a,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
        ),
        StageModelSpec(
            stage_name="b",
            model_cls=nn.Linear,
            model_kwargs={"in_features": HIDDEN_DIM, "out_features": OUTPUT_DIM},
            state_dict=sd_b,
            is_terminal=True,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
            loss_cls=nn.CrossEntropyLoss,
        ),
    ]


def _generate_batch(seed_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cpu").manual_seed(SEED + seed_offset)
    data = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen, device="cpu")
    labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), generator=gen, device="cpu")
    return data, labels


def _all_actor_handles(plan) -> list:
    actors = []
    for group in plan.resource_set_to_actor_group.values():
        actors.extend(group.actors)
    return actors


def _count_t1_pairs(runner: RayPipelineRunner, src_stage: str, dst_stage: str) -> int:
    routing = runner.router.get_routing(src_stage, dst_stage)
    count = 0
    for src_rank, dst_ranks in routing.src_to_dst.items():
        for dst_rank in dst_ranks:
            if runner.router.get_actor_pair_transport(src_stage, dst_stage, src_rank, dst_rank) == "t1":
                count += 1
    return count


@pytest.fixture(scope="module")
def ray_context():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


def _build_runner(failure_injection: dict | None = None) -> tuple[PlacementManager, object, RayPipelineRunner]:
    pipeline = _build_pipeline()
    specs = _build_model_specs()
    manager = PlacementManager(pipeline, model_specs=specs, actor_cls=CpuT1IpcActor)
    plan = manager.plan()
    manager.build_models(plan)
    runner = RayPipelineRunner(pipeline, plan, scheduler=GPipeScheduler(), failure_injection=failure_injection)
    return manager, plan, runner


@pytest.mark.cpu_only
def test_t1_gpipe_microbatch_mapping_and_cache_bounds(ray_context, monkeypatch):
    monkeypatch.delenv("MM_TEST_T1_FAIL_MB", raising=False)
    manager, plan, runner = _build_runner()
    try:
        assert runner.router.has_t1_edges(), "Expected at least one T1 edge for overlap placement"
        data, labels = _generate_batch()
        num_microbatches = 16
        result = runner.run_iteration(data=data, labels=labels, iteration=0, num_microbatches=num_microbatches)
        assert result["loss"] is not None
        assert torch.isfinite(torch.tensor(result["loss"]))

        t1_pair_count = _count_t1_pairs(runner, "a", "b")
        assert t1_pair_count > 0
        cleanup_stats = runner.get_last_t1_cache_cleanup_stats()
        assert cleanup_stats, "Expected cleanup stats from T1 cache clear path"
        for stats in cleanup_stats:
            assert stats["forward_peak_entries"] <= t1_pair_count * num_microbatches
            assert stats["backward_peak_entries"] <= t1_pair_count * num_microbatches

        actor_stats = ray.get([a.get_t1_ipc_cache_stats.remote() for a in _all_actor_handles(plan)])
        for stats in actor_stats:
            assert stats["forward_current_entries"] == 0
            assert stats["backward_current_entries"] == 0
    finally:
        runner.shutdown()
        manager.shutdown()


@pytest.mark.cpu_only
def test_t1_cache_cleared_after_iteration_failure(ray_context):
    manager, plan, runner = _build_runner(
        failure_injection={"stage_name": "b", "op": "forward", "iteration": 0, "microbatch_id": 2}
    )
    try:
        data, labels = _generate_batch(seed_offset=1)
        with pytest.raises(PipelineIterationError):
            runner.run_iteration(data=data, labels=labels, iteration=0, num_microbatches=4)

        actor_stats = ray.get([a.get_t1_ipc_cache_stats.remote() for a in _all_actor_handles(plan)])
        for stats in actor_stats:
            assert stats["forward_current_entries"] == 0
            assert stats["backward_current_entries"] == 0
    finally:
        runner.shutdown()
        manager.shutdown()


@pytest.mark.cpu_only
def test_t1_contract_miss_and_missing_microbatch_errors():
    actor = MultiStageActor({}, rank=0)

    with pytest.raises(KeyError, match=r"stage='attn', microbatch=3"):
        actor.create_ipc_for_output("attn", 3)
    with pytest.raises(KeyError, match=r"stage='moe', microbatch=1"):
        actor.create_ipc_for_grad("moe", 1)

    with pytest.raises(ValueError, match="requires microbatch_id"):
        actor.create_ipc_for_output("attn", None)
    with pytest.raises(ValueError, match="requires microbatch_id"):
        actor.create_ipc_for_grad("moe", None)
    with pytest.raises(ValueError, match="requires microbatch_id"):
        actor.forward_from_ipc("moe", {"ipc_handle": "x", "gpu_id": "g"}, labels=None, microbatch_id=None)
    with pytest.raises(ValueError, match="requires microbatch_id"):
        actor.backward_from_ipc("attn", {"ipc_handle": "x", "gpu_id": "g"}, microbatch_id=None)


@pytest.mark.cpu_only
def test_t1_repeated_reads_keep_cache_entries():
    actor = CpuT1IpcActor({}, rank=0)
    payload = StageOutputs(activations=torch.randn(2, 2), meta={"k": "v"})
    actor._t1_forward_ipc_cache[("a", 0)] = payload

    first = actor.create_ipc_for_output("a", 0)
    second = actor.create_ipc_for_output("a", 0)

    assert first["producer_microbatch_id"] == 0
    assert second["producer_microbatch_id"] == 0
    assert torch.equal(first["tensor"], second["tensor"])
    assert ("a", 0) in actor._t1_forward_ipc_cache
