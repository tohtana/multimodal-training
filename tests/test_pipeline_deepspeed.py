"""M4: 2-stage MLP pipeline with mixed native/DeepSpeed engines.

Stage A uses native PyTorch, Stage B uses DeepSpeed (ZeRO-0, fp32).
Verifies:
  1. DeepSpeed engine wraps Stage B's model correctly.
  2. Loss decreases over iterations.
  3. Loss matches all-native pipeline from M2 (same model, same data).
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


def _build_pipeline():
    """Build 2-stage pipeline (all stages on same resource set)."""
    return Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b", is_terminal=True),
        ],
        edges=[EdgeConfig(src="a", dst="b")],
        resource_sets=[ResourceSet(name="gpu", num_gpus=1)],
        placements=[
            Placement(stage_name="a", resource_set="gpu"),
            Placement(stage_name="b", resource_set="gpu"),
        ],
    )


def _ds_config_zero0(batch_size: int, lr: float) -> dict:
    """Minimal DeepSpeed config for ZeRO-0 fp32."""
    return {
        "train_batch_size": batch_size,
        "train_micro_batch_size_per_gpu": batch_size,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "fp16": {"enabled": False},
        "bf16": {"enabled": False},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": lr,
                "weight_decay": 0.0,
                "torch_adam": True,
                "adam_w_mode": False,
            },
        },
        "zero_allow_untested_optimizer": True,
    }


def _build_native_specs():
    """Build all-native model specs."""
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


def _build_mixed_specs():
    """Build mixed specs: Stage A native, Stage B DeepSpeed ZeRO-0."""
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
            engine="native",
        ),
        StageModelSpec(
            stage_name="b",
            model_cls=nn.Linear,
            model_kwargs={"in_features": HIDDEN_DIM, "out_features": OUTPUT_DIM},
            state_dict=sd_b,
            is_terminal=True,
            loss_cls=nn.CrossEntropyLoss,
            engine="deepspeed",
            ds_config=_ds_config_zero0(BATCH_SIZE, LR),
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
class TestPipelineDeepSpeed:
    def test_deepspeed_loss_decreases(self, ray_context):
        """Mixed native+DeepSpeed pipeline trains and loss decreases."""
        pipeline = _build_pipeline()
        specs = _build_mixed_specs()
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

    def test_deepspeed_matches_native(self, ray_context):
        """DeepSpeed ZeRO-0 fp32 produces same loss as native PyTorch."""
        pipeline = _build_pipeline()
        data, labels = _generate_batch(seed_offset=0)

        # Run all-native pipeline
        native_specs = _build_native_specs()
        native_manager = PlacementManager(pipeline, model_specs=native_specs)
        native_plan = native_manager.plan()
        native_runner = RayPipelineRunner(pipeline, native_plan)

        try:
            native_manager.build_models(native_plan)
            native_losses = []
            for i in range(5):
                result = native_runner.run_iteration(data=data, labels=labels, iteration=i)
                native_losses.append(result["loss"])
        finally:
            native_runner.shutdown()
            native_manager.shutdown()

        # Run mixed native+DeepSpeed pipeline
        mixed_specs = _build_mixed_specs()
        mixed_manager = PlacementManager(pipeline, model_specs=mixed_specs)
        mixed_plan = mixed_manager.plan()
        mixed_runner = RayPipelineRunner(pipeline, mixed_plan)

        try:
            mixed_manager.build_models(mixed_plan)
            mixed_losses = []
            for i in range(5):
                result = mixed_runner.run_iteration(data=data, labels=labels, iteration=i)
                mixed_losses.append(result["loss"])
        finally:
            mixed_runner.shutdown()
            mixed_manager.shutdown()

        # First iteration must match exactly (same weights, same data, fp32)
        assert (
            abs(mixed_losses[0] - native_losses[0]) < 1e-5
        ), f"First iteration loss mismatch: ds={mixed_losses[0]:.6f}, native={native_losses[0]:.6f}"
        # After 5 iterations (FP32, small tolerance for optimizer impl differences)
        assert (
            abs(mixed_losses[-1] - native_losses[-1]) < 1e-3
        ), f"Loss mismatch after 5 iters: ds={mixed_losses[-1]:.6f}, native={native_losses[-1]:.6f}"
