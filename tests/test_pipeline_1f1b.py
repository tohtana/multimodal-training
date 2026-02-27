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
from python.pipeline.scheduler import (
    DEFAULT_GPIPE_MAX_MICROBATCHES,
    GPipeScheduler,
    OneFOneBScheduler,
    OpType,
    PipelineScheduler,
    SequentialScheduler,
    get_scheduler,
    validate_scheduler_request,
)
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
                assert (s - 1, mb) in fwd_done, f"F({step.stage_name}, mb{mb}): predecessor not done"
            fwd_done.add((s, mb))
        else:
            # This stage must have forwarded this mb
            assert (s, mb) in fwd_done, f"B({step.stage_name}, mb{mb}): forward not done"
            # Successor must have done backward for this mb (unless last stage)
            if s < N - 1:
                assert (s + 1, mb) in bwd_done, f"B({step.stage_name}, mb{mb}): successor backward not done"
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


@pytest.mark.cpu_only
class TestGPipeScheduler:
    def test_basic_2_stages_4_microbatches(self):
        sched = GPipeScheduler()
        stages = ["a", "b"]
        steps = sched.generate_schedule(stages, 4)
        _validate_schedule(stages, 4, steps)
        assert len(steps) == 2 * 2 * 4  # 2 stages × 4 mb × (F+B)

    def test_all_forwards_before_backwards(self):
        """GPipe guarantee: every forward precedes every backward."""
        stages = ["a", "b", "c"]
        steps = GPipeScheduler().generate_schedule(stages, 4)
        last_fwd_idx = max(i for i, s in enumerate(steps) if s.op == OpType.FORWARD)
        first_bwd_idx = min(i for i, s in enumerate(steps) if s.op == OpType.BACKWARD)
        assert last_fwd_idx < first_bwd_idx

    def test_stage_first_forward_order(self):
        """Forwards are emitted stage-by-stage (all mb for stage 0, then stage 1, ...)."""
        stages = ["a", "b"]
        steps = GPipeScheduler().generate_schedule(stages, 4)
        fwd_steps = [s for s in steps if s.op == OpType.FORWARD]
        # First 4 should be stage "a", next 4 should be stage "b"
        assert all(s.stage_name == "a" for s in fwd_steps[:4])
        assert all(s.stage_name == "b" for s in fwd_steps[4:])

    def test_backward_fifo_order(self):
        """Backward mb order is FIFO (0,1,2,...) per stage, matching StageTrainer activation pop."""
        stages = ["a", "b"]
        steps = GPipeScheduler().generate_schedule(stages, 4)
        for stage_name in stages:
            bwd_mbs = [s.microbatch_id for s in steps if s.op == OpType.BACKWARD and s.stage_name == stage_name]
            assert bwd_mbs == [0, 1, 2, 3], f"Stage {stage_name} backward not FIFO: {bwd_mbs}"

    def test_backward_reverse_stage_order(self):
        """Backward stages are in reverse topological order."""
        stages = ["a", "b", "c"]
        steps = GPipeScheduler().generate_schedule(stages, 4)
        bwd_steps = [s for s in steps if s.op == OpType.BACKWARD]
        # First 4 backward steps should be stage "c", then "b", then "a"
        assert all(s.stage_name == "c" for s in bwd_steps[:4])
        assert all(s.stage_name == "b" for s in bwd_steps[4:8])
        assert all(s.stage_name == "a" for s in bwd_steps[8:])

    def test_3_stages_8_microbatches(self):
        stages = ["s0", "s1", "s2"]
        steps = GPipeScheduler().generate_schedule(stages, 8)
        _validate_schedule(stages, 8, steps)

    def test_single_microbatch_same_ops_as_sequential(self):
        stages = ["a", "b", "c"]
        gpipe = GPipeScheduler().generate_schedule(stages, 1)
        seq = SequentialScheduler().generate_schedule(stages, 1)
        gpipe_ops = {(s.op, s.stage_name, s.microbatch_id) for s in gpipe}
        seq_ops = {(s.op, s.stage_name, s.microbatch_id) for s in seq}
        assert gpipe_ops == seq_ops

    def test_empty_microbatches(self):
        steps = GPipeScheduler().generate_schedule(["a", "b"], 0)
        assert steps == []


@pytest.mark.cpu_only
class TestGetScheduler:
    def test_known_schedulers(self):
        assert isinstance(get_scheduler("sequential"), PipelineScheduler)
        assert isinstance(get_scheduler("1f1b"), PipelineScheduler)
        assert isinstance(get_scheduler("gpipe"), PipelineScheduler)
        assert type(get_scheduler("sequential")) is SequentialScheduler
        assert type(get_scheduler("1f1b")) is OneFOneBScheduler
        assert type(get_scheduler("gpipe")) is GPipeScheduler

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            get_scheduler("nonexistent")


@pytest.mark.cpu_only
class TestValidateSchedulerRequest:
    def test_known_scheduler_and_microbatch_limits(self):
        validate_scheduler_request("1f1b", num_microbatches=128)
        validate_scheduler_request("sequential", num_microbatches=128)
        validate_scheduler_request(
            "gpipe",
            num_microbatches=DEFAULT_GPIPE_MAX_MICROBATCHES,
            gpipe_max_microbatches=DEFAULT_GPIPE_MAX_MICROBATCHES,
        )

    def test_gpipe_over_limit_raises(self):
        with pytest.raises(ValueError, match="may OOM"):
            validate_scheduler_request("gpipe", num_microbatches=9, gpipe_max_microbatches=8)

    def test_gpipe_at_limit_passes(self):
        # Exactly at limit should pass
        validate_scheduler_request("gpipe", num_microbatches=8, gpipe_max_microbatches=8)

    def test_unknown_scheduler_raises(self):
        with pytest.raises(ValueError, match="Unknown scheduler"):
            validate_scheduler_request("unknown", num_microbatches=1)

    def test_invalid_gpipe_max_raises(self):
        with pytest.raises(ValueError, match="gpipe_max_microbatches must be >= 1"):
            validate_scheduler_request("gpipe", num_microbatches=1, gpipe_max_microbatches=0)


@pytest.mark.cpu_only
class TestStep6SchedulerPlumbing:
    def test_run_variant_b_forwards_scheduler(self, monkeypatch):
        import examples.attn_moe_overlap.step6_mps_overlap as step6

        captured = {}

        def _fake_run_pipeline_variant(
            name,
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            actor_cls=step6.BenchmarkMultiStageActor,
            scheduler_name="1f1b",
            gpipe_max_microbatches=DEFAULT_GPIPE_MAX_MICROBATCHES,
        ):
            captured["scheduler_name"] = scheduler_name
            captured["gpipe_max_microbatches"] = gpipe_max_microbatches
            return [1.0], [0.5], object()

        monkeypatch.setattr(step6, "_run_pipeline_variant", _fake_run_pipeline_variant)
        monkeypatch.setattr(step6.ray, "is_initialized", lambda: False)
        monkeypatch.setattr(step6.ray, "init", lambda *args, **kwargs: None)
        monkeypatch.setattr(step6.ray, "shutdown", lambda *args, **kwargs: None)

        out = step6.run_variant_b(
            config=None,
            attn_state_dict={},
            moe_state_dict={},
            data=None,
            labels=None,
            attn_fwd_flops=100,
            moe_fwd_flops=200,
            peak_tflops=None,
            scheduler_name="gpipe",
            gpipe_max_microbatches=4,
        )
        assert out["status"] == "ok"
        assert captured == {"scheduler_name": "gpipe", "gpipe_max_microbatches": 4}

    def test_run_pipeline_variant_guard_fails_before_setup(self, monkeypatch):
        import examples.attn_moe_overlap.step6_mps_overlap as step6

        touched = {"setup_called": False}

        original_make = step6._make_single_gpu_resources

        def _fail_if_called():
            touched["setup_called"] = True
            return original_make()

        monkeypatch.setattr(step6, "_make_single_gpu_resources", _fail_if_called)

        with pytest.raises(ValueError, match="may OOM"):
            step6._run_pipeline_variant(
                "B",
                None,
                None,
                None,
                None,
                None,
                scheduler_name="gpipe",
                gpipe_max_microbatches=1,
            )

        assert touched["setup_called"] is False


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
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=4,
                )
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
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
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=num_mb,
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
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=num_mb,
                )
                ofob_losses.append(result["loss"])
        finally:
            ofob_runner.shutdown()
            ofob_manager.shutdown()

        # First iteration loss must match exactly (same weights, same data, fp32)
        assert (
            abs(ofob_losses[0] - seq_losses[0]) < 1e-5
        ), f"First iter loss mismatch: 1f1b={ofob_losses[0]:.6f}, seq={seq_losses[0]:.6f}"
        # After iterations: within tolerance (same gradient accumulation)
        assert (
            abs(ofob_losses[-1] - seq_losses[-1]) < 1e-3
        ), f"Final loss mismatch: 1f1b={ofob_losses[-1]:.6f}, seq={seq_losses[-1]:.6f}"


@pytest.mark.gpu
class TestPipelineGPipe:
    def test_gpipe_loss_decreases(self, ray_context):
        """3-stage pipeline with GPipe schedule trains correctly."""
        if torch.cuda.device_count() < 2:
            pytest.skip("Requires at least 2 GPUs")

        pipeline = _build_three_stage_pipeline_cross_gpu()
        specs = _build_model_specs()
        scheduler = GPipeScheduler()
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan, scheduler=scheduler)

        try:
            manager.build_models(plan)
            data, labels = _generate_batch()
            losses = []

            for i in range(10):
                result = runner.run_iteration(
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=4,
                )
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_gpipe_matches_sequential(self, ray_context):
        """GPipe and sequential produce the same loss (same gradient accumulation)."""
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
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=num_mb,
                )
                seq_losses.append(result["loss"])
        finally:
            seq_runner.shutdown()
            seq_manager.shutdown()

        # Run with GPipe scheduler
        gpipe_specs = _build_model_specs()
        gpipe_manager = PlacementManager(pipeline, model_specs=gpipe_specs)
        gpipe_plan = gpipe_manager.plan()
        gpipe_runner = RayPipelineRunner(pipeline, gpipe_plan, scheduler=GPipeScheduler())
        try:
            gpipe_manager.build_models(gpipe_plan)
            gpipe_losses = []
            for i in range(num_iters):
                result = gpipe_runner.run_iteration(
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=num_mb,
                )
                gpipe_losses.append(result["loss"])
        finally:
            gpipe_runner.shutdown()
            gpipe_manager.shutdown()

        # First iteration loss must match exactly (same weights, same data, fp32)
        assert (
            abs(gpipe_losses[0] - seq_losses[0]) < 1e-5
        ), f"First iter loss mismatch: gpipe={gpipe_losses[0]:.6f}, seq={seq_losses[0]:.6f}"
        # After iterations: within tolerance (same gradient accumulation)
        assert (
            abs(gpipe_losses[-1] - seq_losses[-1]) < 1e-3
        ), f"Final loss mismatch: gpipe={gpipe_losses[-1]:.6f}, seq={seq_losses[-1]:.6f}"
