"""M9: Three-stage pipeline with mixed T0/T2 transport.

Vision → Bridge → Decoder where:
- Vision + Bridge share the same resource set (T0, zero-copy)
- Bridge → Decoder crosses resource sets (T2, RDT/NCCL)

Verifies:
  1. vision→bridge is T0 (same ActorGroup).
  2. bridge→decoder is T2 (cross-ActorGroup via NCCL).
  3. Loss decreases over multiple iterations.
  4. Grad norm is positive and consistent.
"""

import pytest
import ray
import torch
import torch.nn as nn

from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

# Dimensions
INPUT_DIM = 32
BRIDGE_DIM = 48
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20


def _build_three_stage_pipeline():
    """Build vision → bridge → decoder pipeline.

    Vision + Bridge on encoder_gpus (2 GPUs, T0 between them).
    Decoder on decoder_gpus (2 GPUs, T2 from bridge).
    """
    return Pipeline(
        stages=[
            Stage(name="vision", is_source=True),
            Stage(name="bridge"),
            Stage(name="decoder", is_terminal=True),
        ],
        edges=[
            EdgeConfig(src="vision", dst="bridge"),
            EdgeConfig(src="bridge", dst="decoder"),
        ],
        resource_sets=[
            ResourceSet(name="encoder_gpus", num_gpus=2),
            ResourceSet(name="decoder_gpus", num_gpus=2),
        ],
        placements=[
            Placement(stage_name="vision", resource_set="encoder_gpus"),
            Placement(stage_name="bridge", resource_set="encoder_gpus"),
            Placement(stage_name="decoder", resource_set="decoder_gpus"),
        ],
    )


def _build_model_specs():
    torch.manual_seed(SEED)
    sd_vision = nn.Linear(INPUT_DIM, BRIDGE_DIM).state_dict()

    torch.manual_seed(SEED + 1000)
    sd_bridge = nn.Linear(BRIDGE_DIM, HIDDEN_DIM).state_dict()

    torch.manual_seed(SEED + 2000)
    sd_decoder = nn.Linear(HIDDEN_DIM, OUTPUT_DIM).state_dict()

    return [
        StageModelSpec(
            stage_name="vision",
            model_cls=nn.Linear,
            model_kwargs={"in_features": INPUT_DIM, "out_features": BRIDGE_DIM},
            state_dict=sd_vision,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
        ),
        StageModelSpec(
            stage_name="bridge",
            model_cls=nn.Linear,
            model_kwargs={"in_features": BRIDGE_DIM, "out_features": HIDDEN_DIM},
            state_dict=sd_bridge,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
        ),
        StageModelSpec(
            stage_name="decoder",
            model_cls=nn.Linear,
            model_kwargs={"in_features": HIDDEN_DIM, "out_features": OUTPUT_DIM},
            state_dict=sd_decoder,
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
class TestPipelineThreeStage:
    def test_transport_tiers(self, ray_context):
        """Verify vision→bridge is T0, bridge→decoder is T2."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Three-stage test requires at least 4 GPUs")

        pipeline = _build_three_stage_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # vision and bridge share encoder_gpus → T0
            assert runner.router.get_transport("vision", "bridge") == "t0"
            assert runner.router.is_same_group("vision", "bridge")

            # bridge and decoder are on different resource sets → T2
            assert runner.router.get_transport("bridge", "decoder") == "t2"
            assert not runner.router.is_same_group("bridge", "decoder")

            # Both edges are 1:1 (symmetric)
            assert runner.router.is_symmetric("vision", "bridge")
            assert runner.router.is_symmetric("bridge", "decoder")
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_loss_decreases(self, ray_context):
        """Three-stage pipeline trains for multiple iterations; loss decreases."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Three-stage test requires at least 4 GPUs")

        pipeline = _build_three_stage_pipeline()
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

    def test_grad_norm_across_stages(self, ray_context):
        """Global grad norm aggregates all 3 stages correctly."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Three-stage test requires at least 4 GPUs")

        pipeline = _build_three_stage_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)
            data, labels = _generate_batch()

            result = runner.run_iteration(data=data, labels=labels, iteration=0)
            grad_norm = result["global_grad_norm"]

            # Grad norm must be > 0 (3 stages all have gradients)
            assert grad_norm > 0, "Global grad norm should be positive"
            # Verify it's a reasonable value (not NaN or inf)
            assert grad_norm == grad_norm, "Global grad norm is NaN"
            assert grad_norm < float("inf"), "Global grad norm is inf"
        finally:
            runner.shutdown()
            manager.shutdown()
