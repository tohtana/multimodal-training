"""Step 6: MPS compute overlap — 3-way comparison on a single GPU.

Runs three variants from identical random init:
  A. Single-process baseline (step3-style): both stages in one process
  B. subset_of, no MPS: two processes on same GPU, CUDA serializes kernels
  C. subset_of + MPS: two processes on same GPU, MPS enables kernel concurrency

Each variant runs 30 warmup + 10 timed iterations. Reports timing, per-stage MFU,
correctness (convergence + B-C loss gap), and optionally PyTorch Profiler traces.

Pipeline variants pre-load data on the source actor's GPU to avoid
CPU→object-store→actor serialization overhead dominating timing.

Requires at least 1 CUDA GPU with MPS support (Volta+).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import ray
import torch
import torch.nn as nn

from examples.attn_moe_overlap.flops_utils import (
    compute_attn_forward_flops,
    compute_mfu,
    compute_moe_forward_flops,
    get_gpu_peak_tflops,
)
from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
    generate_dummy_batch,
)
from examples.attn_moe_overlap.mps_utils import MPSContext, check_mps_support, get_gpu_arch
from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import (
    DEFAULT_GPIPE_MAX_MICROBATCHES,
    get_scheduler,
    validate_scheduler_request,
)
from python.pipeline.stage import EdgeConfig, ParallelismType, Pipeline, Placement, ResourceSet, Stage
from python.ray.multi_stage_actor import MultiStageActor
from python.ray.payloads import StageOutputs

logger = logging.getLogger(__name__)

WARMUP_ITERS = 30
TIMED_ITERS = 10
NUM_MICROBATCHES = 8
BATCH_SIZE = 8
SEQ_LEN = 8192
NUM_CLASSES = 10
BASE_LR = 1e-3
DTYPE = torch.bfloat16

# IPC error keywords for classifying MPS+IPC incompatibility
_IPC_ERROR_KEYWORDS = (
    "cudaIpc",
    "create_ipc_for_output",
    "forward_from_ipc",
    "create_ipc_for_grad",
    "backward_from_ipc",
    "IPC",
    "cudaErrorNotSupported",
)


def _make_result(
    status: str,
    times_ms: list[float] | None = None,
    losses: list[float] | None = None,
    stage_metrics: dict | None = None,
    error: str | None = None,
) -> dict:
    """Build a standardized variant result dict."""
    return {
        "status": status,
        "times_ms": times_ms if status == "ok" else [],
        "losses": losses if status == "ok" else [],
        "warmup_iters": WARMUP_ITERS,
        "timed_iters": TIMED_ITERS,
        "stage_metrics": stage_metrics if status == "ok" else {},
        "error": error,
    }


def _is_ipc_error(error_str: str) -> bool:
    """Check if an error string indicates MPS+CUDA IPC incompatibility."""
    return any(kw in error_str for kw in _IPC_ERROR_KEYWORDS)


# ---------------------------------------------------------------------------
# BenchmarkMultiStageActor — pre-loads data on GPU for source stages
# ---------------------------------------------------------------------------


class BenchmarkMultiStageActor(MultiStageActor):
    """MultiStageActor that pre-loads source stage data on GPU.

    Eliminates CPU→object-store→actor serialization overhead by caching
    microbatches on GPU. The pipeline runner still sends data for the source
    stage, but the actor ignores it and uses the cached GPU microbatch.
    """

    def preload_data(self, stage_name: str, data, labels, num_microbatches: int) -> bool:
        """Pre-load and split data on GPU for a source stage.

        Args:
            stage_name: The source stage that will use cached data.
            data: Full batch tensor (CPU). Split into microbatches and moved to GPU.
            labels: Full labels tensor (CPU). Split into microbatches and moved to GPU.
            num_microbatches: Number of microbatches to split into.
        """
        device = torch.device("cuda:0")
        if isinstance(data, torch.Tensor):
            self._preloaded_data = list(data.to(device).chunk(num_microbatches, dim=0))
        else:
            self._preloaded_data = [data] * num_microbatches
        if isinstance(labels, torch.Tensor):
            self._preloaded_labels = list(labels.to(device).chunk(num_microbatches, dim=0))
        else:
            self._preloaded_labels = [labels] * num_microbatches
        self._preloaded_stage = stage_name
        self._preload_counter = 0
        return True

    def forward_step(self, stage_name, inputs=None, labels=None):
        # For the preloaded source stage, use cached GPU data instead of the
        # serialized CPU input from the pipeline runner.
        if hasattr(self, "_preloaded_stage") and self._preloaded_stage == stage_name:
            idx = self._preload_counter % len(self._preloaded_data)
            self._preload_counter += 1
            inputs = StageOutputs(activations=self._preloaded_data[idx])
            labels = self._preloaded_labels[idx]
        return super().forward_step(stage_name, inputs, labels)


class VoidReturnActor(BenchmarkMultiStageActor):
    """BenchmarkMultiStageActor that returns scalars instead of tensors.

    Eliminates return-value serialization overhead. Internal state
    (_last_forward_outputs, _last_backward_grads) is still updated by the
    parent. T1 transport (IPC) reads from internal state, not return values.
    """

    def forward_step(self, stage_name, inputs=None, labels=None):
        super().forward_step(stage_name, inputs, labels)
        return True

    def backward_step(self, stage_name, downstream_grad=None):
        super().backward_step(stage_name, downstream_grad)
        return True


class SharedBufferActor(VoidReturnActor):
    """VoidReturnActor with shared ring buffer support.

    Inherits GPU data preloading and void returns from VoidReturnActor.
    Shared buffer methods (forward_to_buffer, forward_from_buffer, etc.)
    are inherited from MultiStageActor.
    """

    pass


# ---------------------------------------------------------------------------
# Variant A: Single-process baseline
# ---------------------------------------------------------------------------


def run_variant_a(
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    attn_fwd_flops: int,
    moe_fwd_flops: int,
    peak_tflops: float | None,
) -> dict:
    """Run single-process baseline with microbatch gradient accumulation.

    Processes NUM_MICROBATCHES microbatches of batch_size=1 per iteration,
    accumulating gradients before a single optimizer step. This matches
    the Ray pipeline's execution pattern for fair comparison.
    """
    print(f"\n=== Variant A: Single-process baseline ({NUM_MICROBATCHES} microbatches, grad accum) ===")
    device = "cuda:0"

    attn_stage = Qwen3AttentionStage(config, dtype=DTYPE).to(device)
    moe_stage = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE).to(device)
    attn_stage.load_state_dict(attn_state_dict)
    moe_stage.load_state_dict(moe_state_dict)

    all_params = list(attn_stage.parameters()) + list(moe_stage.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=BASE_LR, foreach=False)
    loss_fn = nn.CrossEntropyLoss()

    x = data.to(device)
    y = labels.to(device)
    # Chunk into microbatches (batch_size=1 each)
    x_mbs = list(x.chunk(NUM_MICROBATCHES, dim=0))
    y_mbs = list(y.chunk(NUM_MICROBATCHES, dim=0))

    losses = []
    times_ms = []
    attn_times_ms = []
    moe_times_ms = []

    total_iters = WARMUP_ITERS + TIMED_ITERS
    for step in range(total_iters):
        is_timed = step >= WARMUP_ITERS

        start_ev = torch.cuda.Event(enable_timing=True)
        end_ev = torch.cuda.Event(enable_timing=True)
        attn_start = torch.cuda.Event(enable_timing=True)
        attn_end = torch.cuda.Event(enable_timing=True)
        moe_start = torch.cuda.Event(enable_timing=True)
        moe_end = torch.cuda.Event(enable_timing=True)

        start_ev.record()
        optimizer.zero_grad()

        total_loss = 0.0
        for mb_i in range(NUM_MICROBATCHES):
            attn_start.record()
            h = attn_stage(x_mbs[mb_i])
            attn_end.record()

            moe_start.record()
            logits = moe_stage(h)
            moe_end.record()

            loss = loss_fn(logits, y_mbs[mb_i])
            (loss / NUM_MICROBATCHES).backward()
            total_loss += loss.item()

        optimizer.step()

        end_ev.record()
        torch.cuda.synchronize()

        avg_loss = total_loss / NUM_MICROBATCHES
        elapsed = start_ev.elapsed_time(end_ev)
        attn_elapsed = attn_start.elapsed_time(attn_end)
        moe_elapsed = moe_start.elapsed_time(moe_end)

        losses.append(avg_loss)
        if is_timed:
            times_ms.append(elapsed)
            attn_times_ms.append(attn_elapsed)
            moe_times_ms.append(moe_elapsed)

        label = "warmup" if not is_timed else "timed"
        print(f"  [A] Step {step:3d} ({label}) | Loss: {avg_loss:.6f} | Time: {elapsed:.2f} ms")

    # Compute per-stage metrics
    attn_train_flops = 3 * attn_fwd_flops
    moe_train_flops = 3 * moe_fwd_flops
    attn_mean_ms = sum(attn_times_ms) / len(attn_times_ms)
    moe_mean_ms = sum(moe_times_ms) / len(moe_times_ms)
    attn_achieved = attn_train_flops / (attn_mean_ms / 1000.0)
    moe_achieved = moe_train_flops / (moe_mean_ms / 1000.0)

    stage_metrics = {
        "attn": {
            "forward_flops": attn_fwd_flops,
            "train_flops": attn_train_flops,
            "timed_mean_ms": attn_mean_ms,
            "achieved_flops_per_sec": attn_achieved,
            "mfu": compute_mfu(attn_achieved, peak_tflops),
        },
        "moe": {
            "forward_flops": moe_fwd_flops,
            "train_flops": moe_train_flops,
            "timed_mean_ms": moe_mean_ms,
            "achieved_flops_per_sec": moe_achieved,
            "mfu": compute_mfu(moe_achieved, peak_tflops),
        },
    }

    return _make_result("ok", times_ms, losses, stage_metrics)


# ---------------------------------------------------------------------------
# Pipeline helpers (shared by B and C)
# ---------------------------------------------------------------------------


def _make_single_gpu_resources():
    """Return resource_sets and placements for single-GPU subset_of."""
    resource_sets = [
        ResourceSet(name="rs_parent", num_gpus=1, device_ids=(0,)),
        ResourceSet(name="rs_child", num_gpus=1, device_ids=(0,), subset_of="rs_parent"),
    ]
    placements = [
        Placement(stage_name="attn", resource_set="rs_parent"),
        Placement(stage_name="moe", resource_set="rs_child"),
    ]
    return resource_sets, placements


def _make_pipeline(resource_sets, placements):
    """Create a 2-stage attention+MoE pipeline."""
    pipeline = Pipeline(
        stages=[
            Stage(name="attn", is_source=True, parallelism=ParallelismType.NONE),
            Stage(name="moe", is_terminal=True, parallelism=ParallelismType.NONE),
        ],
        edges=[EdgeConfig(src="attn", dst="moe")],
        resource_sets=resource_sets,
        placements=placements,
        num_microbatches=NUM_MICROBATCHES,
    )
    errors = pipeline.validate()
    assert not errors, f"Pipeline validation errors: {errors}"
    return pipeline


def _make_specs(config, attn_state_dict, moe_state_dict):
    """Create StageModelSpec list for PlacementManager."""
    lr = BASE_LR / NUM_MICROBATCHES
    return [
        StageModelSpec(
            stage_name="attn",
            model_cls=Qwen3AttentionStage,
            model_kwargs={"config": config, "dtype": DTYPE},
            state_dict=attn_state_dict,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": lr, "foreach": False},
        ),
        StageModelSpec(
            stage_name="moe",
            model_cls=Qwen3MoEStageWithHead,
            model_kwargs={"config": config, "num_classes": NUM_CLASSES, "dtype": DTYPE},
            state_dict=moe_state_dict,
            is_terminal=True,
            loss_cls=nn.CrossEntropyLoss,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": lr, "foreach": False},
        ),
    ]


def _run_pipeline_variant(
    name: str,
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    actor_cls=VoidReturnActor,
    scheduler_name: str = "1f1b",
    gpipe_max_microbatches: int = DEFAULT_GPIPE_MAX_MICROBATCHES,
    transport: str = "ipc",
) -> tuple[list[float], list[float], object]:
    """Run a pipeline variant and return (times_ms, losses, plan).

    Pre-loads data on the source actor's GPU so timing reflects compute + IPC,
    not CPU→object-store serialization. A tiny dummy tensor is sent through the
    pipeline for the source stage; the actor overrides it with cached GPU data.

    Args:
        transport: "ipc" (per-microbatch IPC) or "shared_buffer" (pre-shared ring buffer).

    Caller is responsible for ray.init/shutdown and MPS lifecycle.
    """
    # Validate scheduler before any pipeline setup (fail-fast)
    validate_scheduler_request(scheduler_name, NUM_MICROBATCHES, gpipe_max_microbatches=gpipe_max_microbatches)

    resource_sets, placements = _make_single_gpu_resources()
    pipeline = _make_pipeline(resource_sets, placements)
    specs = _make_specs(config, attn_state_dict, moe_state_dict)

    runner = None
    manager = None
    try:
        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=actor_cls)
        plan = manager.plan()
        manager.build_models(plan)

        # Pre-load full batch on the source (attn) actor's GPU.
        # The actor splits it into microbatches and caches them on GPU.
        attn_group = plan.stage_to_actor_group["attn"]
        preload_refs = [
            actor.preload_data.remote("attn", data, labels, NUM_MICROBATCHES) for actor in attn_group.actors
        ]
        ray.get(preload_refs)

        # Tiny dummy for the pipeline runner's source stage input.
        # The BenchmarkMultiStageActor ignores this and uses cached GPU data.
        dummy_data = torch.zeros(BATCH_SIZE, 1, 1, dtype=DTYPE)

        runner = RayPipelineRunner(pipeline, plan, scheduler=get_scheduler(scheduler_name))

        # Shared buffer setup (T19)
        if transport == "shared_buffer":
            # Derive activation shape from static config (step6 path: known fixed shapes)
            batch_chunk = BATCH_SIZE // NUM_MICROBATCHES
            activation_shape = (batch_chunk, SEQ_LEN, config.hidden_size)
            activation_specs = {
                ("attn", "moe"): {
                    "shape": activation_shape,
                    "dtype": DTYPE,
                    "payload_mode": "activations_only",
                    "required_meta_keys": [],
                },
            }
            ok, reason = runner.enable_shared_buffer_overlap(
                num_microbatches=NUM_MICROBATCHES,
                activation_specs=activation_specs,
                scheduler_name=scheduler_name,
            )
            if not ok:
                print(f"  [{name}] shared_buffer setup failed: {reason}; falling back to IPC")

        losses = []
        times_ms = []
        total_iters = WARMUP_ITERS + TIMED_ITERS
        for step in range(total_iters):
            is_timed = step >= WARMUP_ITERS
            t0 = time.perf_counter()
            result = runner.run_iteration(
                data=dummy_data, labels=labels, iteration=step, num_microbatches=NUM_MICROBATCHES
            )
            t1 = time.perf_counter()

            assert result["loss"] is not None, f"Loss is None at step {step}"
            losses.append(result["loss"])
            elapsed_ms = (t1 - t0) * 1000.0
            if is_timed:
                times_ms.append(elapsed_ms)

            label = "warmup" if not is_timed else "timed"
            print(f"  [{name}] Step {step:3d} ({label}) | Loss: {result['loss']:.6f} | Time: {elapsed_ms:.2f} ms")

        return times_ms, losses, plan
    finally:
        if runner is not None:
            runner.shutdown()
        if manager is not None:
            manager.shutdown()


# ---------------------------------------------------------------------------
# Variant B: subset_of, no MPS
# ---------------------------------------------------------------------------


def run_variant_b(
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    attn_fwd_flops: int,
    moe_fwd_flops: int,
    peak_tflops: float | None,
    scheduler_name: str = "1f1b",
    gpipe_max_microbatches: int = DEFAULT_GPIPE_MAX_MICROBATCHES,
) -> dict:
    """Run pipeline variant without MPS."""
    print("\n=== Variant B: subset_of, no MPS ===")

    # Clean session boundary
    if ray.is_initialized():
        ray.shutdown()
    # Remove MPS env vars from driver to ensure clean state
    for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
        os.environ.pop(key, None)

    try:
        ray.init()
        times_ms, losses, _plan = _run_pipeline_variant(
            "B",
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            scheduler_name=scheduler_name,
            gpipe_max_microbatches=gpipe_max_microbatches,
        )

        # Build stage metrics (wall-clock based for pipeline variants)
        mean_ms = sum(times_ms) / len(times_ms)
        attn_train_flops = 3 * attn_fwd_flops
        moe_train_flops = 3 * moe_fwd_flops
        total_train_flops = attn_train_flops + moe_train_flops
        total_achieved = total_train_flops / (mean_ms / 1000.0)

        stage_metrics = {
            "attn": {
                "forward_flops": attn_fwd_flops,
                "train_flops": attn_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (attn_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (attn_train_flops / total_train_flops), peak_tflops),
            },
            "moe": {
                "forward_flops": moe_fwd_flops,
                "train_flops": moe_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (moe_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (moe_train_flops / total_train_flops), peak_tflops),
            },
        }

        return _make_result("ok", times_ms, losses, stage_metrics)
    except Exception as e:
        print(f"  [B] FAILED: {e}")
        return _make_result("failed", error=str(e))
    finally:
        if ray.is_initialized():
            ray.shutdown()


# ---------------------------------------------------------------------------
# Variant C: subset_of + MPS
# ---------------------------------------------------------------------------


def run_variant_c(
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    attn_fwd_flops: int,
    moe_fwd_flops: int,
    peak_tflops: float | None,
    scheduler_name: str = "1f1b",
    gpipe_max_microbatches: int = DEFAULT_GPIPE_MAX_MICROBATCHES,
) -> dict:
    """Run pipeline variant with MPS enabled."""
    print("\n=== Variant C: subset_of + MPS ===")

    if not check_mps_support():
        return _make_result("skipped_unsupported", error="MPS not supported on this machine")

    # Clean session boundary
    if ray.is_initialized():
        ray.shutdown()

    mps_ctx = MPSContext(gpu_id=0)
    try:
        mps_ctx.__enter__()

        # Start Ray with MPS env vars
        ray.init(runtime_env={"env_vars": mps_ctx.get_env_vars()})

        # IPC smoke test (1 iteration, uses raw data — not timing-sensitive)
        print("  [C] Running IPC smoke test under MPS...")
        smoke_runner = None
        smoke_manager = None
        try:
            resource_sets, placements = _make_single_gpu_resources()
            pipeline = _make_pipeline(resource_sets, placements)
            specs = _make_specs(config, attn_state_dict, moe_state_dict)
            smoke_manager = PlacementManager(pipeline, model_specs=specs, actor_cls=VoidReturnActor)
            smoke_plan = smoke_manager.plan()
            smoke_manager.build_models(smoke_plan)
            smoke_runner = RayPipelineRunner(pipeline, smoke_plan, scheduler=get_scheduler(scheduler_name))
            smoke_result = smoke_runner.run_iteration(
                data=data, labels=labels, iteration=0, num_microbatches=NUM_MICROBATCHES
            )
            assert smoke_result["loss"] is not None, "Smoke test loss is None"
            print(f"  [C] Smoke test passed (loss={smoke_result['loss']:.6f})")
        except Exception as e:
            error_str = str(e)
            if _is_ipc_error(error_str):
                print(f"  [C] MPS+IPC incompatible: {error_str}")
                return _make_result("skipped_unsupported", error=f"MPS+CUDA IPC incompatible: {error_str}")
            else:
                print(f"  [C] Smoke test failed (non-IPC error): {error_str}")
                return _make_result("failed", error=f"Smoke test failed: {error_str}")
        finally:
            if smoke_runner is not None:
                smoke_runner.shutdown()
            if smoke_manager is not None:
                smoke_manager.shutdown()

        # Need fresh Ray session after smoke test
        if ray.is_initialized():
            ray.shutdown()
        ray.init(runtime_env={"env_vars": mps_ctx.get_env_vars()})

        # Timed run
        times_ms, losses, _plan = _run_pipeline_variant(
            "C",
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            scheduler_name=scheduler_name,
            gpipe_max_microbatches=gpipe_max_microbatches,
        )

        # Build stage metrics
        mean_ms = sum(times_ms) / len(times_ms)
        attn_train_flops = 3 * attn_fwd_flops
        moe_train_flops = 3 * moe_fwd_flops
        total_train_flops = attn_train_flops + moe_train_flops
        total_achieved = total_train_flops / (mean_ms / 1000.0)

        stage_metrics = {
            "attn": {
                "forward_flops": attn_fwd_flops,
                "train_flops": attn_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (attn_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (attn_train_flops / total_train_flops), peak_tflops),
            },
            "moe": {
                "forward_flops": moe_fwd_flops,
                "train_flops": moe_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (moe_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (moe_train_flops / total_train_flops), peak_tflops),
            },
        }

        return _make_result("ok", times_ms, losses, stage_metrics)
    except Exception as e:
        print(f"  [C] FAILED: {e}")
        return _make_result("failed", error=str(e))
    finally:
        if ray.is_initialized():
            ray.shutdown()
        mps_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Variant D: subset_of + MPS + shared buffer
# ---------------------------------------------------------------------------


def run_variant_d(
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    attn_fwd_flops: int,
    moe_fwd_flops: int,
    peak_tflops: float | None,
    scheduler_name: str = "gpipe",
    gpipe_max_microbatches: int = DEFAULT_GPIPE_MAX_MICROBATCHES,
) -> dict:
    """Run pipeline variant with MPS + shared ring buffers for true overlap."""
    print("\n=== Variant D: subset_of + MPS + shared_buffer ===")

    if not check_mps_support():
        return _make_result("skipped_unsupported", error="MPS not supported on this machine")

    if scheduler_name != "gpipe":
        return _make_result("skipped_unsupported", error=f"shared_buffer v1 requires gpipe, got {scheduler_name}")

    # Clean session boundary
    if ray.is_initialized():
        ray.shutdown()

    mps_ctx = MPSContext(gpu_id=0)
    try:
        mps_ctx.__enter__()
        ray.init(runtime_env={"env_vars": mps_ctx.get_env_vars()})

        times_ms, losses, _plan = _run_pipeline_variant(
            "D",
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            actor_cls=SharedBufferActor,
            scheduler_name=scheduler_name,
            gpipe_max_microbatches=gpipe_max_microbatches,
            transport="shared_buffer",
        )

        # Build stage metrics
        mean_ms = sum(times_ms) / len(times_ms)
        attn_train_flops = 3 * attn_fwd_flops
        moe_train_flops = 3 * moe_fwd_flops
        total_train_flops = attn_train_flops + moe_train_flops
        total_achieved = total_train_flops / (mean_ms / 1000.0)

        stage_metrics = {
            "attn": {
                "forward_flops": attn_fwd_flops,
                "train_flops": attn_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (attn_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (attn_train_flops / total_train_flops), peak_tflops),
            },
            "moe": {
                "forward_flops": moe_fwd_flops,
                "train_flops": moe_train_flops,
                "timed_mean_ms": mean_ms,
                "achieved_flops_per_sec": total_achieved * (moe_train_flops / total_train_flops),
                "mfu": compute_mfu(total_achieved * (moe_train_flops / total_train_flops), peak_tflops),
            },
        }

        return _make_result("ok", times_ms, losses, stage_metrics)
    except Exception as e:
        print(f"  [D] FAILED: {e}")
        return _make_result("failed", error=str(e))
    finally:
        if ray.is_initialized():
            ray.shutdown()
        mps_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# ProfiledMultiStageActor
# ---------------------------------------------------------------------------


class ProfiledMultiStageActor(VoidReturnActor):
    """BenchmarkMultiStageActor subclass with PyTorch Profiler integration."""

    def enable_profiling(self, trace_dir: str):
        """Enable per-actor PyTorch Profiler."""
        os.makedirs(trace_dir, exist_ok=True)
        self._profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        )
        self._profiler.__enter__()
        self._trace_dir = trace_dir

    def flush_profiling(self):
        """Stop profiler and export trace."""
        if hasattr(self, "_profiler") and self._profiler is not None:
            self._profiler.__exit__(None, None, None)
            self._profiler.export_chrome_trace(os.path.join(self._trace_dir, "trace.json"))
            self._profiler = None

    def forward_step(self, stage_name, inputs=None, labels=None):
        if getattr(self, "_profiler", None) is None:
            return super().forward_step(stage_name, inputs, labels)
        with torch.profiler.record_function(f"{stage_name}.forward"):
            return super().forward_step(stage_name, inputs, labels)

    def backward_step(self, stage_name, downstream_grad=None):
        if getattr(self, "_profiler", None) is None:
            return super().backward_step(stage_name, downstream_grad)
        with torch.profiler.record_function(f"{stage_name}.backward"):
            out = super().backward_step(stage_name, downstream_grad)
        self._profiler.step()
        return out


# ---------------------------------------------------------------------------
# Profiling run
# ---------------------------------------------------------------------------


def run_profiling(
    variant: str,
    trace_dir: str,
    config,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    scheduler_name: str = "1f1b",
    gpipe_max_microbatches: int = DEFAULT_GPIPE_MAX_MICROBATCHES,
) -> dict:
    """Run a profiling pass for the specified variant.

    Args:
        variant: "no_mps" or "mps"
    """
    print(f"\n=== Profiling: {variant} ===")
    profiling_warmup = 5
    profiling_iters = 5
    variant_trace_dir = os.path.join(trace_dir, variant)

    mps_ctx = None
    try:
        if variant == "mps":
            if not check_mps_support():
                print(f"  [prof/{variant}] MPS not supported, skipping")
                return _make_result("skipped_unsupported", error="MPS not supported")
            mps_ctx = MPSContext(gpu_id=0)
            mps_ctx.__enter__()
            ray.init(runtime_env={"env_vars": mps_ctx.get_env_vars()})
        else:
            for key in ("CUDA_MPS_PIPE_DIRECTORY", "CUDA_MPS_LOG_DIRECTORY", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"):
                os.environ.pop(key, None)
            ray.init()

        resource_sets, placements = _make_single_gpu_resources()
        pipeline = _make_pipeline(resource_sets, placements)
        specs = _make_specs(config, attn_state_dict, moe_state_dict)

        runner = None
        manager = PlacementManager(pipeline, model_specs=specs, actor_cls=ProfiledMultiStageActor)
        try:
            plan = manager.plan()
            manager.build_models(plan)

            # Pre-load data on source actor
            attn_group = plan.stage_to_actor_group["attn"]
            moe_group = plan.stage_to_actor_group["moe"]
            preload_refs = [
                actor.preload_data.remote("attn", data, labels, NUM_MICROBATCHES) for actor in attn_group.actors
            ]
            ray.get(preload_refs)
            dummy_data = torch.zeros(BATCH_SIZE, 1, 1, dtype=DTYPE)

            runner = RayPipelineRunner(pipeline, plan, scheduler=get_scheduler(scheduler_name))

            # Enable profiling on actors
            attn_trace = os.path.join(variant_trace_dir, "stage_attn")
            moe_trace = os.path.join(variant_trace_dir, "stage_moe")

            enable_refs = []
            for actor in attn_group.actors:
                enable_refs.append(actor.enable_profiling.remote(attn_trace))
            for actor in moe_group.actors:
                enable_refs.append(actor.enable_profiling.remote(moe_trace))
            ray.get(enable_refs)

            try:
                total = profiling_warmup + profiling_iters
                for step in range(total):
                    result = runner.run_iteration(
                        data=dummy_data, labels=labels, iteration=step, num_microbatches=NUM_MICROBATCHES
                    )
                    label = "warmup" if step < profiling_warmup else "profiled"
                    print(f"  [prof/{variant}] Step {step:3d} ({label}) | Loss: {result['loss']:.6f}")
            finally:
                flush_refs = []
                for actor in attn_group.actors:
                    flush_refs.append(actor.flush_profiling.remote())
                for actor in moe_group.actors:
                    flush_refs.append(actor.flush_profiling.remote())
                ray.get(flush_refs)

            print(f"  [prof/{variant}] Traces saved to {variant_trace_dir}/")
            return _make_result("ok")
        finally:
            if runner is not None:
                runner.shutdown()
            manager.shutdown()

    except Exception as e:
        error_str = str(e)
        if variant == "mps" and _is_ipc_error(error_str):
            print(f"  [prof/{variant}] MPS+IPC incompatible, skipping profiling: {error_str}")
            return _make_result("skipped_unsupported", error=f"MPS+IPC incompatible: {error_str}")
        print(f"  [prof/{variant}] FAILED: {error_str}")
        return _make_result("failed", error=error_str)
    finally:
        if ray.is_initialized():
            ray.shutdown()
        if mps_ctx is not None:
            mps_ctx.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_report(
    results_a: dict,
    results_b: dict,
    results_c: dict,
    config,
    peak_tflops: float | None,
    results_d: dict | None = None,
):
    """Print the comparison report."""
    import numpy as np

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    arch = get_gpu_arch()
    peak_str = f"{peak_tflops:.1f}" if peak_tflops is not None else "N/A"

    print("\n" + "=" * 60)
    print("MPS Overlap Results")
    print("=" * 60)
    print(f"\nGPU: {gpu_name} | Arch: {arch} | bf16 Peak: {peak_str} TFLOP/s")
    print(
        f"Config: batch={BATCH_SIZE}, seq={SEQ_LEN}, hidden={config.hidden_size}, "
        f"heads={config.num_attention_heads}, kv_heads={config.num_key_value_heads}, "
        f"experts={config.num_experts}, k={config.num_experts_per_tok}"
    )

    # --- Timing ---
    all_variants = [
        ("A. Single-process", results_a),
        ("B. subset_of (no MPS)", results_b),
        ("C. subset_of + MPS (IPC)", results_c),
    ]
    if results_d is not None:
        all_variants.append(("D. subset_of + MPS (shared_buf)", results_d))

    print(f"\n--- Timing (mean +/- std over {TIMED_ITERS} iterations) ---")
    for label, r in all_variants:
        if r["status"] == "ok":
            arr = np.array(r["times_ms"])
            print(f"{label:40s}: {arr.mean():.2f} +/- {arr.std():.2f} ms/iter")
        else:
            print(f"{label:40s}: {r['status']} ({r.get('error', 'N/A')})")

    # Speedups
    if results_b["status"] == "ok" and results_c["status"] == "ok":
        b_mean = np.mean(results_b["times_ms"])
        c_mean = np.mean(results_c["times_ms"])
        print(f"{'Speedup C vs B':40s}: {b_mean / c_mean:.2f}x")

    if results_d is not None and results_d["status"] == "ok":
        d_mean = np.mean(results_d["times_ms"])
        if results_a["status"] == "ok":
            a_mean = np.mean(results_a["times_ms"])
            print(f"{'Speedup D vs A':40s}: {a_mean / d_mean:.2f}x")
        if results_c["status"] == "ok":
            c_mean = np.mean(results_c["times_ms"])
            print(f"{'Speedup D vs C':40s}: {c_mean / d_mean:.2f}x")
        if results_b["status"] == "ok":
            b_mean = np.mean(results_b["times_ms"])
            print(f"{'Speedup D vs B':40s}: {b_mean / d_mean:.2f}x")

    if results_a["status"] == "ok" and results_c["status"] == "ok":
        a_mean = np.mean(results_a["times_ms"])
        c_mean = np.mean(results_c["times_ms"])
        print(f"{'Speedup C vs A':40s}: {a_mean / c_mean:.2f}x")

    # --- Variant Status ---
    print("\n--- Variant Status ---")
    status_variants = [("A", results_a), ("B", results_b), ("C", results_c)]
    if results_d is not None:
        status_variants.append(("D", results_d))
    for label, r in status_variants:
        status_str = r["status"]
        if r.get("error"):
            status_str += f" (reason: {r['error']})"
        print(f"{label}: {status_str}")

    # --- Per-Stage MFU ---
    print("\n--- Per-Stage MFU (pipeline variants) ---")
    mfu_variants = [("B/no_mps", results_b), ("C/mps+ipc", results_c)]
    if results_d is not None:
        mfu_variants.append(("D/mps+sbuf", results_d))
    for label, r in mfu_variants:
        if r["status"] == "ok" and r.get("stage_metrics"):
            for stage_name in ("attn", "moe"):
                sm = r["stage_metrics"].get(stage_name, {})
                achieved = sm.get("achieved_flops_per_sec", 0)
                mfu_val = sm.get("mfu")
                mfu_str = f"{mfu_val * 100:.1f}%" if mfu_val is not None else "N/A"
                print(f"{label} {stage_name:10s}: {achieved / 1e9:.1f} GFLOP/s (MFU: {mfu_str})")
        else:
            status = r["status"]
            print(f"{label} {'':10s}: SKIPPED ({status})")

    # --- Correctness ---
    print("\n--- Correctness ---")
    correctness_variants = [("A", results_a), ("B", results_b), ("C", results_c)]
    if results_d is not None:
        correctness_variants.append(("D", results_d))
    for label, r in correctness_variants:
        if r["status"] == "ok":
            all_losses = r["losses"]
            first_5 = sum(all_losses[:5]) / 5
            min_loss = min(all_losses)
            converged = min_loss < first_5 * 0.5
            symbol = "Y" if converged else "N"
            print(f"{label} final loss: {all_losses[-1]:.6f} min: {min_loss:.6f} (learned: {symbol})")
        else:
            print(f"{label} final loss: SKIPPED ({r['status']})")

    if results_b["status"] == "ok" and results_c["status"] == "ok":
        b_final = results_b["losses"][-1]
        c_final = results_c["losses"][-1]
        gap = abs(c_final - b_final)
        print(f"B-C gap: {gap:.6f} (tolerance: 0.10)")
    elif results_c["status"] != "ok":
        print(f"B-C gap: N/A (C {results_c['status']})")

    if results_d is not None and results_d["status"] == "ok" and results_b["status"] == "ok":
        b_final = results_b["losses"][-1]
        d_final = results_d["losses"][-1]
        gap = abs(d_final - b_final)
        print(f"B-D gap: {gap:.6f} (tolerance: 0.10)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="MPS compute overlap benchmark")
    parser.add_argument(
        "--trace-dir", type=str, default="/mnt/local_storage/mps_traces", help="Directory for profiling traces"
    )
    parser.add_argument("--skip-profiling", action="store_true", help="Skip profiling pass")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="1f1b",
        choices=["1f1b", "gpipe", "sequential"],
        help="Pipeline schedule strategy (default: 1f1b)",
    )
    parser.add_argument(
        "--gpipe-max-microbatches",
        type=int,
        default=DEFAULT_GPIPE_MAX_MICROBATCHES,
        help="Safety cap for GPipe microbatches (default: %(default)s)",
    )
    parser.add_argument(
        "--transport",
        type=str,
        default="ipc",
        choices=["ipc", "shared_buffer"],
        help="T1 transport mechanism (default: ipc; shared_buffer adds variant D with pre-shared ring buffers)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available")
        sys.exit(1)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"MPS supported: {check_mps_support()}")
    print(f"GPU arch: {get_gpu_arch()}")

    config = create_qwen3_config()
    peak_tflops = get_gpu_peak_tflops(DTYPE)

    # Compute FLOPs
    attn_fwd_flops = compute_attn_forward_flops(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
    )
    moe_fwd_flops = compute_moe_forward_flops(
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        hidden_size=config.hidden_size,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_intermediate_size=config.moe_intermediate_size,
        num_classes=NUM_CLASSES,
    )
    print(f"Attention forward FLOPs: {attn_fwd_flops:,}")
    print(f"MoE forward FLOPs: {moe_fwd_flops:,}")

    # Create initial models with deterministic seed
    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=DTYPE)
    moe_model = Qwen3MoEStageWithHead(config, NUM_CLASSES, dtype=DTYPE)
    attn_state_dict = attn_model.state_dict()
    moe_state_dict = moe_model.state_dict()

    # Fixed batch (on CPU, bf16)
    torch.manual_seed(42)
    data, labels = generate_dummy_batch(
        config, batch_size=BATCH_SIZE, seq_len=SEQ_LEN, num_classes=NUM_CLASSES, device="cpu", dtype=DTYPE
    )

    # --- Run variants ---
    results_a = run_variant_a(
        config,
        attn_state_dict,
        moe_state_dict,
        data,
        labels,
        attn_fwd_flops,
        moe_fwd_flops,
        peak_tflops,
    )

    results_b = run_variant_b(
        config,
        attn_state_dict,
        moe_state_dict,
        data,
        labels,
        attn_fwd_flops,
        moe_fwd_flops,
        peak_tflops,
        scheduler_name=args.scheduler,
        gpipe_max_microbatches=args.gpipe_max_microbatches,
    )

    results_c = run_variant_c(
        config,
        attn_state_dict,
        moe_state_dict,
        data,
        labels,
        attn_fwd_flops,
        moe_fwd_flops,
        peak_tflops,
        scheduler_name=args.scheduler,
        gpipe_max_microbatches=args.gpipe_max_microbatches,
    )

    results_d = None
    if args.transport == "shared_buffer":
        results_d = run_variant_d(
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            attn_fwd_flops,
            moe_fwd_flops,
            peak_tflops,
            scheduler_name=args.scheduler,
            gpipe_max_microbatches=args.gpipe_max_microbatches,
        )

    # --- Profiling (separate pass) ---
    if not args.skip_profiling:
        run_profiling(
            "no_mps",
            args.trace_dir,
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            scheduler_name=args.scheduler,
            gpipe_max_microbatches=args.gpipe_max_microbatches,
        )
        run_profiling(
            "mps",
            args.trace_dir,
            config,
            attn_state_dict,
            moe_state_dict,
            data,
            labels,
            scheduler_name=args.scheduler,
            gpipe_max_microbatches=args.gpipe_max_microbatches,
        )

    # --- Report ---
    print_report(results_a, results_b, results_c, config, peak_tflops, results_d=results_d)

    # --- Convergence validation ---
    failed = False

    for label, r in [("A", results_a), ("B", results_b)]:
        if r["status"] != "ok":
            print(f"\nFAILED: Variant {label} status={r['status']}")
            failed = True
        else:
            all_losses = r["losses"]
            first_5 = sum(all_losses[:5]) / 5
            min_loss = min(all_losses)
            if min_loss >= first_5 * 0.5:
                print(f"\nFAILED: Variant {label} not learning: first_5={first_5:.6f}, min={min_loss:.6f}")
                failed = True

    if results_c["status"] == "ok":
        c_losses = results_c["losses"]
        c_first_5 = sum(c_losses[:5]) / 5
        c_min = min(c_losses)
        if c_min >= c_first_5 * 0.5:
            print(f"\nFAILED: Variant C not learning: first_5={c_first_5:.6f}, min={c_min:.6f}")
            failed = True
        if results_b["status"] == "ok":
            gap = abs(c_losses[-1] - results_b["losses"][-1])
            if gap > 0.10:
                print(f"\nFAILED: B-C loss gap {gap:.6f} exceeds tolerance 0.10")
                failed = True
    elif results_c["status"] == "failed":
        print(f"\nFAILED: Variant C status=failed ({results_c.get('error', 'N/A')})")
        failed = True

    if results_d is not None:
        if results_d["status"] == "ok":
            d_losses = results_d["losses"]
            d_first_5 = sum(d_losses[:5]) / 5
            d_min = min(d_losses)
            if d_min >= d_first_5 * 0.5:
                print(f"\nFAILED: Variant D not learning: first_5={d_first_5:.6f}, min={d_min:.6f}")
                failed = True
            if results_b["status"] == "ok":
                gap = abs(d_losses[-1] - results_b["losses"][-1])
                if gap > 0.10:
                    print(f"\nFAILED: B-D loss gap {gap:.6f} exceeds tolerance 0.10")
                    failed = True
        elif results_d["status"] == "failed":
            print(f"\nFAILED: Variant D status=failed ({results_d.get('error', 'N/A')})")
            failed = True

    if failed:
        sys.exit(1)
    else:
        print("\nPASSED: All required variants converged")


if __name__ == "__main__":
    main()
