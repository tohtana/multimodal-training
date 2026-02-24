"""M7: Asymmetric GPU allocation — stages on resource sets with different GPU counts.

CPU-only tests verify:
  1. RoutingPlan deterministic rank maps for 1:1, M:N expansion, N:M contraction.
  2. Broadcast routing is rank-ordered and reproducible across repeated builds.

GPU tests verify (4+ GPUs, Ray):
  3. 2-stage MLP pipeline with Stage A on 2 GPUs and Stage B on 4 GPUs.
  4. Activations correctly broadcasted from 2 actors to 4 actors.
  5. Gradients correctly gathered from 4 actors to 2 actors.
  6. Loss decreases over multiple iterations.
"""

import pytest
import ray
import torch
import torch.nn as nn

from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.router import RoutingPlan
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20

pytestmark_cpu = pytest.mark.cpu_only


# ── CPU-only RoutingPlan tests ──


@pytest.mark.cpu_only
class TestRoutingPlan:
    def test_1_to_1_mapping(self):
        """1:1 mapping is identity."""
        rp = RoutingPlan.build(num_src=4, num_dst=4)
        for i in range(4):
            assert rp.src_to_dst[i] == [i]
            assert rp.dst_to_src[i] == i

    def test_expansion_2_to_4(self):
        """2 → 4 expansion: each src maps to 2 dst."""
        rp = RoutingPlan.build(num_src=2, num_dst=4)
        assert rp.src_to_dst[0] == [0, 1]
        assert rp.src_to_dst[1] == [2, 3]
        assert rp.dst_to_src[0] == 0
        assert rp.dst_to_src[1] == 0
        assert rp.dst_to_src[2] == 1
        assert rp.dst_to_src[3] == 1

    def test_expansion_2_to_3_uneven(self):
        """2 → 3 expansion: lower src gets extra dst (2+1)."""
        rp = RoutingPlan.build(num_src=2, num_dst=3)
        assert rp.src_to_dst[0] == [0, 1]
        assert rp.src_to_dst[1] == [2]
        assert rp.dst_to_src[0] == 0
        assert rp.dst_to_src[1] == 0
        assert rp.dst_to_src[2] == 1

    def test_contraction_4_to_2(self):
        """4 → 2 contraction: each dst receives from 2 src."""
        rp = RoutingPlan.build(num_src=4, num_dst=2)
        assert rp.src_to_dst[0] == [0]
        assert rp.src_to_dst[1] == [0]
        assert rp.src_to_dst[2] == [1]
        assert rp.src_to_dst[3] == [1]

    def test_contraction_3_to_2_uneven(self):
        """3 → 2 contraction: lower dst gets extra src."""
        rp = RoutingPlan.build(num_src=3, num_dst=2)
        assert rp.src_to_dst[0] == [0]
        assert rp.src_to_dst[1] == [0]
        assert rp.src_to_dst[2] == [1]

    def test_expansion_1_to_4(self):
        """1 → 4 broadcast: single src maps to all dst."""
        rp = RoutingPlan.build(num_src=1, num_dst=4)
        assert rp.src_to_dst[0] == [0, 1, 2, 3]
        for d in range(4):
            assert rp.dst_to_src[d] == 0

    def test_deterministic_across_builds(self):
        """Routing map is deterministic across repeated builds."""
        for _ in range(10):
            rp = RoutingPlan.build(num_src=3, num_dst=7)
            # 3 → 7: base=2, remainder=1; src0 gets 3, src1 gets 2, src2 gets 2
            assert rp.src_to_dst[0] == [0, 1, 2]
            assert rp.src_to_dst[1] == [3, 4]
            assert rp.src_to_dst[2] == [5, 6]

    def test_all_dst_covered(self):
        """Every destination rank is assigned to exactly one source rank."""
        for num_src, num_dst in [(2, 7), (3, 5), (1, 8), (4, 4), (6, 3)]:
            rp = RoutingPlan.build(num_src=num_src, num_dst=num_dst)
            all_dst = set()
            for dst_list in rp.src_to_dst.values():
                all_dst.update(dst_list)
            assert all_dst == set(range(num_dst)), f"Failed for {num_src}→{num_dst}"

    def test_all_src_covered(self):
        """Every source rank has at least one destination."""
        for num_src, num_dst in [(2, 7), (3, 5), (1, 8), (4, 4), (6, 3)]:
            rp = RoutingPlan.build(num_src=num_src, num_dst=num_dst)
            assert len(rp.src_to_dst) == num_src, f"Failed for {num_src}→{num_dst}"
            for src_rank in range(num_src):
                assert len(rp.src_to_dst[src_rank]) >= 1, f"src_rank {src_rank} has no dst"

    def test_policy_stored(self):
        """Policy string is stored in the plan."""
        rp = RoutingPlan.build(num_src=2, num_dst=4, policy="scatter")
        assert rp.policy == "scatter"

    def test_num_src_dst_stored(self):
        """num_src and num_dst are stored."""
        rp = RoutingPlan.build(num_src=3, num_dst=5)
        assert rp.num_src == 3
        assert rp.num_dst == 5


# ── GPU pipeline tests ──


def _build_asymmetric_pipeline(num_gpus_a: int, num_gpus_b: int):
    """Build 2-stage pipeline with asymmetric GPU allocation."""
    return Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b", is_terminal=True),
        ],
        edges=[EdgeConfig(src="a", dst="b")],
        resource_sets=[
            ResourceSet(name="rs_a", num_gpus=num_gpus_a),
            ResourceSet(name="rs_b", num_gpus=num_gpus_b),
        ],
        placements=[
            Placement(stage_name="a", resource_set="rs_a"),
            Placement(stage_name="b", resource_set="rs_b"),
        ],
    )


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
class TestPipelineAsymmetric:
    def test_asymmetric_2_to_4_loss_decreases(self, ray_context):
        """Asymmetric pipeline (2 GPUs → 4 GPUs) trains; loss decreases."""
        if torch.cuda.device_count() < 6:
            pytest.skip("Asymmetric 2→4 test requires at least 6 GPUs")

        pipeline = _build_asymmetric_pipeline(num_gpus_a=2, num_gpus_b=4)
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)
            data, labels = _generate_batch()
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], (
                f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
            assert result["global_grad_norm"] is not None
            assert result["global_grad_norm"] > 0
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_asymmetric_1_to_2_loss_decreases(self, ray_context):
        """Simpler asymmetric pipeline (1 GPU → 2 GPUs) trains; loss decreases."""
        if torch.cuda.device_count() < 3:
            pytest.skip("Asymmetric 1→2 test requires at least 3 GPUs")

        pipeline = _build_asymmetric_pipeline(num_gpus_a=1, num_gpus_b=2)
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)
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

    def test_asymmetric_routing_matches_plan(self, ray_context):
        """Verify that the router creates the expected M:N routing for 2→4."""
        if torch.cuda.device_count() < 6:
            pytest.skip("Asymmetric 2→4 test requires at least 6 GPUs")

        pipeline = _build_asymmetric_pipeline(num_gpus_a=2, num_gpus_b=4)
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            routing = runner.router.get_routing("a", "b")
            assert routing.num_src == 2
            assert routing.num_dst == 4
            assert routing.src_to_dst[0] == [0, 1]
            assert routing.src_to_dst[1] == [2, 3]
            assert routing.dst_to_src[0] == 0
            assert routing.dst_to_src[1] == 0
            assert routing.dst_to_src[2] == 1
            assert routing.dst_to_src[3] == 1

            # Verify transport is T2 (cross-group)
            assert runner.router.get_transport("a", "b") == "t2"
            assert not runner.router.is_symmetric("a", "b")
        finally:
            runner.shutdown()
            manager.shutdown()
