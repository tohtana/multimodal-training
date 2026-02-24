"""M10: 1F1B microbatch schedule tests.

CPU tests verify:
  1. OneFOneBScheduler produces valid schedules (all ops present, dependencies met).
  2. 1F1B has fewer pipeline bubbles than sequential (warmup is shorter).
  3. Schedule is valid for various (N_stages, M_microbatches) combinations.

GPU tests verify:
  4. 3-stage MLP pipeline with 4 microbatches: 1F1B matches sequential schedule loss.
  5. Loss decreases over iterations with gradient accumulation.
"""

import pytest
import ray
import torch
import torch.nn as nn

from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import OneFOneBScheduler, OpType, SequentialScheduler
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 48
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
LR = 0.01


# ── CPU-only scheduler tests ──


def _validate_schedule(stage_order, num_microbatches, steps):
    """Validate that a schedule respects all data dependencies."""
    N = len(stage_order)
    M = num_microbatches
    stage_idx = {name: i for i, name in enumerate(stage_order)}

    fwd_done = set()  # (stage_idx, mb)
    bwd_done = set()

    for step in steps:
        s = stage_idx[step.stage_name]
        mb = step.microbatch_id

        if step.op == OpType.FORWARD:
            # Predecessor must have forwarded this mb
            if s > 0:
                assert (s - 1, mb) in fwd_done, (
                    f"F({step.stage_name}, mb{mb}): predecessor not done"
                )
            fwd_done.add((s, mb))
        else:
            # This stage must have forwarded this mb
            assert (s, mb) in fwd_done, (
                f"B({step.stage_name}, mb{mb}): forward not done"
            )
            # Successor must have done backward for this mb (unless last stage)
            if s < N - 1:
                assert (s + 1, mb) in bwd_done, (
                    f"B({step.stage_name}, mb{mb}): successor backward not done"
                )
            bwd_done.add((s, mb))

    # All ops must be present
    assert len(fwd_done) == N * M, f"Expected {N*M} forwards, got {len(fwd_done)}"
    assert len(bwd_done) == N * M, f"Expected {N*M} backwards, got {len(bwd_done)}"


@pytest.mark.cpu_only
class TestOneFOneBScheduler:
    def test_basic_3_stages_4_microbatches(self):
        """3 stages, 4 microbatches: schedule is valid."""
        sched = OneFOneBScheduler()
        stages = ["a", "b", "c"]
        steps = sched.generate_schedule(stages, 4)
        _validate_schedule(stages, 4, steps)
        assert len(steps) == 3 * 4 * 2  # 3 stages × 4 mb × (F+B)

    def test_2_stages_2_microbatches(self):
        stages = ["a", "b"]
        steps = OneFOneBScheduler().generate_schedule(stages, 2)
        _validate_schedule(stages, 2, steps)

    def test_single_stage_4_microbatches(self):
        stages = ["a"]
        steps = OneFOneBScheduler().generate_schedule(stages, 4)
        _validate_schedule(stages, 4, steps)
        # Single stage: F0,B0,F1,B1,F2,B2,F3,B3
        assert len(steps) == 8

    def test_4_stages_8_microbatches(self):
        stages = ["s0", "s1", "s2", "s3"]
        steps = OneFOneBScheduler().generate_schedule(stages, 8)
        _validate_schedule(stages, 8, steps)

    def test_stages_equal_microbatches(self):
        """N stages, N microbatches (minimal case for full 1F1B)."""
        stages = ["a", "b", "c"]
        steps = OneFOneBScheduler().generate_schedule(stages, 3)
        _validate_schedule(stages, 3, steps)

    def test_more_stages_than_microbatches(self):
        """5 stages, 2 microbatches: still valid."""
        stages = ["a", "b", "c", "d", "e"]
        steps = OneFOneBScheduler().generate_schedule(stages, 2)
        _validate_schedule(stages, 2, steps)

    def test_single_microbatch_matches_sequential(self):
        """With 1 microbatch, 1F1B should produce the same ops as sequential."""
        stages = ["a", "b", "c"]
        seq_steps = SequentialScheduler().generate_schedule(stages, 1)
        ofob_steps = OneFOneBScheduler().generate_schedule(stages, 1)
        # Both should have the same operations (possibly different order)
        seq_ops = {(s.op, s.stage_name, s.microbatch_id) for s in seq_steps}
        ofob_ops = {(s.op, s.stage_name, s.microbatch_id) for s in ofob_steps}
        assert seq_ops == ofob_ops

    def test_interleaving_occurs(self):
        """1F1B interleaves forward/backward across microbatches (unlike sequential)."""
        stages = ["a", "b", "c"]
        steps = OneFOneBScheduler().generate_schedule(stages, 4)
        # In sequential: all 4 mb0 ops come before any mb1 op
        # In 1F1B: backward of early mb happens before all forwards of late mb
        ops_before_mb3_fwd = []
        for step in steps:
            if step.op == OpType.FORWARD and step.microbatch_id == 3:
                break
            ops_before_mb3_fwd.append(step)
        # At least one backward should happen before stage0's forward of mb3
        has_backward = any(s.op == OpType.BACKWARD for s in ops_before_mb3_fwd)
        assert has_backward, "1F1B should interleave backward before late-mb forward"

    def test_empty_microbatches(self):
        steps = OneFOneBScheduler().generate_schedule(["a", "b"], 0)
        assert steps == []


# ── GPU pipeline tests ──


def _build_three_stage_pipeline_cross_gpu():
    """3-stage pipeline on 2 separate resource sets."""
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
        resource_sets=[
            ResourceSet(name="gpu0", num_gpus=1),
            ResourceSet(name="gpu1", num_gpus=1),
        ],
        placements=[
            Placement(stage_name="a", resource_set="gpu0"),
            Placement(stage_name="b", resource_set="gpu0"),
            Placement(stage_name="c", resource_set="gpu1"),
        ],
    )


def _build_model_specs():
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
class TestPipeline1F1B:
    def test_1f1b_loss_decreases(self, ray_context):
        """3-stage pipeline with 1F1B schedule (4 microbatches) trains correctly."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires at least 2 GPUs")

        pipeline = _build_three_stage_pipeline_cross_gpu()
        specs = _build_model_specs()
        scheduler = OneFOneBScheduler()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan, scheduler=scheduler)

        try:
            manager.build_models(plan)
            data, labels = _generate_batch()
            losses = []

            for i in range(10):
                result = runner.run_iteration(
                    data=data, labels=labels, iteration=i, num_microbatches=4,
                )
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], (
                f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
            )
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_1f1b_matches_sequential(self, ray_context):
        """1F1B and sequential produce the same loss (same gradient accumulation)."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires at least 2 GPUs")

        pipeline = _build_three_stage_pipeline_cross_gpu()
        data, labels = _generate_batch()
        num_mb = 4
        num_iters = 3

        # Run with sequential scheduler
        seq_specs = _build_model_specs()
        seq_manager = PlacementManager(pipeline, model_specs=seq_specs)
        seq_plan = seq_manager.plan()
        seq_runner = RayPipelineRunner(pipeline, seq_plan, scheduler=SequentialScheduler())
        try:
            seq_manager.build_models(seq_plan)
            seq_losses = []
            for i in range(num_iters):
                result = seq_runner.run_iteration(
                    data=data, labels=labels, iteration=i, num_microbatches=num_mb,
                )
                seq_losses.append(result["loss"])
        finally:
            seq_runner.shutdown()
            seq_manager.shutdown()

        # Run with 1F1B scheduler
        ofob_specs = _build_model_specs()
        ofob_manager = PlacementManager(pipeline, model_specs=ofob_specs)
        ofob_plan = ofob_manager.plan()
        ofob_runner = RayPipelineRunner(pipeline, ofob_plan, scheduler=OneFOneBScheduler())
        try:
            ofob_manager.build_models(ofob_plan)
            ofob_losses = []
            for i in range(num_iters):
                result = ofob_runner.run_iteration(
                    data=data, labels=labels, iteration=i, num_microbatches=num_mb,
                )
                ofob_losses.append(result["loss"])
        finally:
            ofob_runner.shutdown()
            ofob_manager.shutdown()

        # First iteration loss must match exactly (same weights, same data, fp32)
        assert abs(ofob_losses[0] - seq_losses[0]) < 1e-5, (
            f"First iter loss mismatch: 1f1b={ofob_losses[0]:.6f}, seq={seq_losses[0]:.6f}"
        )
        # After iterations: within tolerance (same gradient accumulation)
        assert abs(ofob_losses[-1] - seq_losses[-1]) < 1e-3, (
            f"Final loss mismatch: 1f1b={ofob_losses[-1]:.6f}, seq={seq_losses[-1]:.6f}"
        )
