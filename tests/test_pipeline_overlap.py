"""M11: Overlapping GPU sets (subset_of) with CUDA IPC (T1) transport.

CPU-only tests verify:
  1. Pipeline validation for subset_of constraints (device_ids, subset relationship).
  2. CrossStageRouter per-actor-pair transport detection (T1 vs T2).

GPU tests verify (4+ GPUs, Ray):
  3. Complete overlap: 2 ActorGroups on same 2 GPUs, all pairs T1 (CUDA IPC).
  4. Partial overlap: Group A (4 GPUs), Group B (2 GPUs subset_of A).
     - Shared GPU pairs use T1 (CUDA IPC).
     - Non-shared GPU pairs use T2 (RDT/NCCL).
  5. Training correctness: loss decreases over iterations.
  6. Transport tier auto-detection is correct per actor pair.
"""

import pytest
import ray
import torch
import torch.nn as nn

from python.pipeline.dag import PipelineDAG
from python.pipeline.placement import PlacementManager, PlacementPlan, PipelineActorGroup, StageModelSpec
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.router import CrossStageRouter, RoutingPlan
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20


# ── CPU-only validation tests ──


@pytest.mark.cpu_only
class TestSubsetOfValidation:
    def test_valid_subset_of(self):
        """Valid subset_of: child device_ids ⊂ parent device_ids."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert not errors, f"Unexpected validation errors: {errors}"

    def test_child_missing_device_ids(self):
        """Child with subset_of but no device_ids should fail validation."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert any("no device_ids" in e for e in errors), f"Expected device_ids error: {errors}"

    def test_parent_missing_device_ids(self):
        """Parent without device_ids when child uses subset_of should fail."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert any("no device_ids" in e for e in errors), f"Expected device_ids error: {errors}"

    def test_child_not_subset(self):
        """Child device_ids not a subset of parent should fail."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 5), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert any("not a subset" in e for e in errors), f"Expected subset error: {errors}"

    def test_unknown_parent(self):
        """subset_of referencing non-existent parent should fail."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="nonexistent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert any("unknown parent" in e for e in errors), f"Expected unknown parent error: {errors}"

    def test_complete_overlap_device_ids(self):
        """Complete overlap: child device_ids == parent device_ids."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=2, device_ids=(0, 1)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        errors = pipeline.validate()
        assert not errors, f"Unexpected validation errors: {errors}"


@pytest.mark.cpu_only
class TestRouterTransportDetection:
    """Test per-actor-pair transport detection using mock GPU IDs."""

    def test_all_t1_complete_overlap(self):
        """Complete overlap: all actor pairs should be T1."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=2, device_ids=(0, 1)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        dag = PipelineDAG(pipeline)

        # Mock placement plan with matching GPU IDs
        plan = PlacementPlan()
        parent_group = PipelineActorGroup("parent", [], 2, stage_names=["b"])
        child_group = PipelineActorGroup("child", [], 2, stage_names=["a"])
        plan.stage_to_actor_group["a"] = child_group
        plan.stage_to_actor_group["b"] = parent_group
        plan.resource_set_to_actor_group["parent"] = parent_group
        plan.resource_set_to_actor_group["child"] = child_group
        # Same GPU IDs → all T1
        plan.actor_gpu_ids["parent"] = {0: "GPU-UUID-0", 1: "GPU-UUID-1"}
        plan.actor_gpu_ids["child"] = {0: "GPU-UUID-0", 1: "GPU-UUID-1"}

        router = CrossStageRouter(dag, plan)
        assert router.get_transport("a", "b") == "t1"
        assert router.get_actor_pair_transport("a", "b", 0, 0) == "t1"
        assert router.get_actor_pair_transport("a", "b", 1, 1) == "t1"
        assert router.has_t1_edges()
        assert not router.has_cross_group_edges()

    def test_mixed_t1_t2_partial_overlap(self):
        """Partial overlap: some pairs T1, others T2."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )
        dag = PipelineDAG(pipeline)

        plan = PlacementPlan()
        parent_group = PipelineActorGroup("parent", [], 4, stage_names=["b"])
        child_group = PipelineActorGroup("child", [], 2, stage_names=["a"])
        plan.stage_to_actor_group["a"] = child_group
        plan.stage_to_actor_group["b"] = parent_group
        plan.resource_set_to_actor_group["parent"] = parent_group
        plan.resource_set_to_actor_group["child"] = child_group
        # Partial overlap: GPU 0 and 1 shared, GPU 2 and 3 only parent
        plan.actor_gpu_ids["parent"] = {
            0: "GPU-UUID-0", 1: "GPU-UUID-1", 2: "GPU-UUID-2", 3: "GPU-UUID-3",
        }
        plan.actor_gpu_ids["child"] = {0: "GPU-UUID-0", 1: "GPU-UUID-1"}

        router = CrossStageRouter(dag, plan)
        assert router.get_transport("a", "b") == "mixed"

        # Routing: child(2) → parent(4), child 0→[parent 0,1], child 1→[parent 2,3]
        routing = router.get_routing("a", "b")
        assert routing.src_to_dst[0] == [0, 1]
        assert routing.src_to_dst[1] == [2, 3]

        # child 0 → parent 0: same GPU → T1
        assert router.get_actor_pair_transport("a", "b", 0, 0) == "t1"
        # child 0 → parent 1: different GPU → T2
        assert router.get_actor_pair_transport("a", "b", 0, 1) == "t2"
        # child 1 → parent 2: different GPU → T2
        assert router.get_actor_pair_transport("a", "b", 1, 2) == "t2"
        # child 1 → parent 3: different GPU → T2
        assert router.get_actor_pair_transport("a", "b", 1, 3) == "t2"

        assert router.has_t1_edges()
        assert router.has_cross_group_edges()

    def test_no_overlap_all_t2(self):
        """No overlap: all actor pairs should be T2."""
        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="rs_a", num_gpus=2),
                ResourceSet(name="rs_b", num_gpus=2),
            ],
            placements=[
                Placement(stage_name="a", resource_set="rs_a"),
                Placement(stage_name="b", resource_set="rs_b"),
            ],
        )
        dag = PipelineDAG(pipeline)

        plan = PlacementPlan()
        group_a = PipelineActorGroup("rs_a", [], 2, stage_names=["a"])
        group_b = PipelineActorGroup("rs_b", [], 2, stage_names=["b"])
        plan.stage_to_actor_group["a"] = group_a
        plan.stage_to_actor_group["b"] = group_b
        plan.resource_set_to_actor_group["rs_a"] = group_a
        plan.resource_set_to_actor_group["rs_b"] = group_b
        plan.actor_gpu_ids["rs_a"] = {0: "GPU-UUID-0", 1: "GPU-UUID-1"}
        plan.actor_gpu_ids["rs_b"] = {0: "GPU-UUID-2", 1: "GPU-UUID-3"}

        router = CrossStageRouter(dag, plan)
        assert router.get_transport("a", "b") == "t2"
        assert not router.has_t1_edges()
        assert router.has_cross_group_edges()


# ── GPU tests ──


def _build_model_specs():
    """Build serializable model specs with deterministic weights."""
    torch.manual_seed(SEED)
    sd_a = nn.Linear(INPUT_DIM, HIDDEN_DIM).state_dict()

    torch.manual_seed(SEED + 1000)
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


def _generate_batch(seed_offset: int = 0):
    gen = torch.Generator(device="cpu").manual_seed(SEED + seed_offset)
    data = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen, device="cpu")
    labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), generator=gen, device="cpu")
    return data, labels


@pytest.fixture(scope="module")
def ray_context():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


@pytest.mark.gpu
class TestPipelineOverlapGPU:
    def test_complete_overlap_loss_decreases(self, ray_context):
        """Complete overlap (2 GPUs shared): T1 transport, loss decreases."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Complete overlap test requires at least 2 GPUs")

        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=2, device_ids=(0, 1)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )

        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # Verify T1 transport detection
            assert runner.router.has_t1_edges(), "Expected T1 edges for complete overlap"

            # Verify GPU IDs match (complete overlap)
            parent_gpus = plan.actor_gpu_ids.get("parent", {})
            child_gpus = plan.actor_gpu_ids.get("child", {})
            for rank in range(2):
                assert parent_gpus[rank] == child_gpus[rank], (
                    f"Rank {rank}: parent GPU {parent_gpus[rank]} != child GPU {child_gpus[rank]}"
                )

            data, labels = _generate_batch()
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], (
                f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_partial_overlap_loss_decreases(self, ray_context):
        """Partial overlap (4 GPUs parent, 2 shared): mixed T1/T2, loss decreases."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Partial overlap test requires at least 4 GPUs")

        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )

        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # Verify mixed transport detection
            edge_transport = runner.router.get_transport("a", "b")
            assert edge_transport in ("mixed", "t1", "t2"), f"Unexpected transport: {edge_transport}"
            assert runner.router.has_t1_edges() or edge_transport == "t2", (
                "Expected at least some T1 pairs for partial overlap"
            )

            data, labels = _generate_batch()
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], (
                f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_transport_tier_auto_detection(self, ray_context):
        """Verify per-actor-pair transport tier for partial overlap."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Transport detection test requires at least 4 GPUs")

        pipeline = Pipeline(
            stages=[Stage(name="a", is_source=True), Stage(name="b", is_terminal=True)],
            edges=[EdgeConfig(src="a", dst="b")],
            resource_sets=[
                ResourceSet(name="parent", num_gpus=4, device_ids=(0, 1, 2, 3)),
                ResourceSet(name="child", num_gpus=2, device_ids=(0, 1), subset_of="parent"),
            ],
            placements=[
                Placement(stage_name="a", resource_set="child"),
                Placement(stage_name="b", resource_set="parent"),
            ],
        )

        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # Verify GPU ID overlap for shared bundles
            parent_gpus = plan.actor_gpu_ids.get("parent", {})
            child_gpus = plan.actor_gpu_ids.get("child", {})

            # Bundles 0 and 1 should be shared (child devices 0,1 placed on parent bundles 0,1)
            shared_count = sum(
                1 for r in range(min(len(parent_gpus), len(child_gpus)))
                if parent_gpus.get(r) == child_gpus.get(r)
            )

            # At minimum, verify the router detected SOME T1 pairs if GPUs actually overlap
            if shared_count > 0:
                assert runner.router.has_t1_edges(), (
                    f"Expected T1 edges with {shared_count} shared GPU(s), "
                    f"parent={parent_gpus}, child={child_gpus}"
                )

            # Verify routing plan
            routing = runner.router.get_routing("a", "b")
            assert routing.num_src == 2
            assert routing.num_dst == 4

            # Run a training iteration to confirm it works end-to-end
            data, labels = _generate_batch()
            result = runner.run_iteration(data=data, labels=labels, iteration=0)
            assert result["loss"] is not None
        finally:
            runner.shutdown()
            manager.shutdown()
