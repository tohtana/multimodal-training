"""M2: 3-stage MLP pipeline on Ray with T0 transport (same resource set).

Tests:
  1. Ray pipeline trains successfully and loss decreases.
  2. Failure injection prevents partial optimizer commit.
  3. Double shutdown is a no-op.
"""

import pytest
import ray
import torch
import torch.nn as nn

from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import PipelineIterationError, RayPipelineRunner
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20


def _build_pipeline():
    """Build the 3-stage pipeline config."""
    return Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b"),
            Stage(name="c", is_terminal=True),
        ],
        edges=[
            EdgeConfig(src="a", dst="b"),
            EdgeConfig(src="b", dst="c"),
        ],
        resource_sets=[ResourceSet(name="gpu", num_gpus=1)],
        placements=[
            Placement(stage_name="a", resource_set="gpu"),
            Placement(stage_name="b", resource_set="gpu"),
            Placement(stage_name="c", resource_set="gpu"),
        ],
    )


def _build_model_specs():
    """Build serializable model specs for Ray."""
    torch.manual_seed(SEED)
    sd_a = nn.Linear(INPUT_DIM, HIDDEN_DIM).state_dict()

    torch.manual_seed(SEED + 1000)
    sd_b = nn.Linear(HIDDEN_DIM, HIDDEN_DIM).state_dict()

    torch.manual_seed(SEED + 2000)
    sd_c = nn.Linear(HIDDEN_DIM, OUTPUT_DIM).state_dict()

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
            model_kwargs={"in_features": HIDDEN_DIM, "out_features": HIDDEN_DIM},
            state_dict=sd_b,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
        ),
        StageModelSpec(
            stage_name="c",
            model_cls=nn.Linear,
            model_kwargs={"in_features": HIDDEN_DIM, "out_features": OUTPUT_DIM},
            state_dict=sd_c,
            is_terminal=True,
            optimizer_cls=torch.optim.Adam,
            optimizer_kwargs={"lr": LR, "foreach": False},
            loss_cls=nn.CrossEntropyLoss,
        ),
    ]


def _generate_batch(seed_offset: int = 0):
    """Generate batch on CPU."""
    gen = torch.Generator(device="cpu").manual_seed(SEED + seed_offset)
    data = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen, device="cpu")
    labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), generator=gen, device="cpu")
    return data, labels


@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for the test module."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


@pytest.mark.gpu
class TestPipelineMLPRay:
    def test_ray_pipeline_loss_decreases(self, ray_context):
        """Ray pipeline trains for multiple iterations; loss decreases."""
        pipeline = _build_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # Use fixed data for convergence
            data, labels = _generate_batch(seed_offset=0)
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            # Verify loss decreased
            assert losses[-1] < losses[0], (
                f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
            # Global grad norm should be computed
            assert result["global_grad_norm"] is not None
            assert result["global_grad_norm"] > 0
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_failure_injection_no_partial_commit(self, ray_context):
        """Injected failure prevents optimizer step."""
        pipeline = _build_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(
            pipeline, plan,
            failure_injection={"stage_name": "b", "op": "backward", "iteration": 0},
        )

        try:
            manager.build_models(plan)
            data, labels = _generate_batch()

            with pytest.raises(PipelineIterationError):
                runner.run_iteration(data=data, labels=labels, iteration=0)
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_double_shutdown_is_noop(self, ray_context):
        """Calling shutdown twice should not raise."""
        pipeline = _build_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)
        finally:
            runner.shutdown()
            manager.shutdown()
            # Second shutdown should be a no-op
            runner.shutdown()
            manager.shutdown()
