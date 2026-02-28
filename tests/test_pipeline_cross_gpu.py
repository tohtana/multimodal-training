"""M3: 2-stage MLP pipeline with cross-GPU transport (T2 — RDT/NCCL).

Stage A on GPU 0, Stage B (terminal) on GPU 1, each in its own resource set.
Verifies:
  1. Forward activations arrive on GPU 1 via RDT; backward gradients arrive on GPU 0.
  2. Loss decreases over multiple iterations (training works end-to-end across GPUs).
  3. Loss matches single-GPU T0 Ray runner (same model, same data, same seed).
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
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20


def _build_cross_gpu_pipeline():
    """Build 2-stage pipeline with separate resource sets (1 GPU each)."""
    return Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b", is_terminal=True),
        ],
        edges=[
            EdgeConfig(src="a", dst="b"),
        ],
        resource_sets=[
            ResourceSet(name="gpu0", num_gpus=1),
            ResourceSet(name="gpu1", num_gpus=1),
        ],
        placements=[
            Placement(stage_name="a", resource_set="gpu0"),
            Placement(stage_name="b", resource_set="gpu1"),
        ],
    )


def _build_same_gpu_pipeline():
    """Build 2-stage pipeline with a single resource set (T0, same GPU)."""
    return Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b", is_terminal=True),
        ],
        edges=[
            EdgeConfig(src="a", dst="b"),
        ],
        resource_sets=[ResourceSet(name="gpu", num_gpus=1)],
        placements=[
            Placement(stage_name="a", resource_set="gpu"),
            Placement(stage_name="b", resource_set="gpu"),
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
class TestPipelineCrossGPU:
    def test_cross_gpu_loss_decreases(self, ray_context):
        """Cross-GPU pipeline trains for multiple iterations; loss decreases."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Cross-GPU test requires at least 2 GPUs")

        pipeline = _build_cross_gpu_pipeline()
        specs = _build_model_specs()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            data, labels = _generate_batch(seed_offset=0)
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            assert result["global_grad_norm"] is not None
            assert result["global_grad_norm"] > 0
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_cross_gpu_matches_t0(self, ray_context):
        """Cross-GPU (T2) pipeline produces same loss as same-GPU (T0) Ray runner."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Cross-GPU test requires at least 2 GPUs")

        data, labels = _generate_batch(seed_offset=0)

        # Run T0 pipeline (same GPU, all stages on one resource set)
        t0_pipeline = _build_same_gpu_pipeline()
        t0_specs = _build_model_specs()
        t0_manager = PlacementManager(t0_pipeline, model_specs=t0_specs)
        t0_plan = t0_manager.plan()
        t0_runner = RayPipelineRunner(t0_pipeline, t0_plan)

        try:
            t0_manager.build_models(t0_plan)
            t0_losses = []
            for i in range(5):
                result = t0_runner.run_iteration(data=data, labels=labels, iteration=i)
                t0_losses.append(result["loss"])
        finally:
            t0_runner.shutdown()
            t0_manager.shutdown()

        # Run T2 pipeline (cross-GPU, stages on different resource sets)
        t2_pipeline = _build_cross_gpu_pipeline()
        t2_specs = _build_model_specs()
        t2_manager = PlacementManager(t2_pipeline, model_specs=t2_specs)
        t2_plan = t2_manager.plan()
        t2_runner = RayPipelineRunner(t2_pipeline, t2_plan)

        try:
            t2_manager.build_models(t2_plan)
            t2_losses = []
            for i in range(5):
                result = t2_runner.run_iteration(data=data, labels=labels, iteration=i)
                t2_losses.append(result["loss"])
        finally:
            t2_runner.shutdown()
            t2_manager.shutdown()

        # Compare first iteration loss (same model weights, same data)
        assert (
            abs(t2_losses[0] - t0_losses[0]) < 1e-4
        ), f"First iteration loss mismatch: t2={t2_losses[0]:.6f}, t0={t0_losses[0]:.6f}"
        # Compare last iteration loss (within tolerance for FP32 across GPUs)
        assert (
            abs(t2_losses[-1] - t0_losses[-1]) < 1e-3
        ), f"Loss mismatch after 5 iters: t2={t2_losses[-1]:.6f}, t0={t0_losses[-1]:.6f}"
