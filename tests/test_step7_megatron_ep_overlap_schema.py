"""CPU-only tests for step7 Megatron EP overlap schema utilities."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.megatron_overlap_schema import (
    CASE_SCHEMA_VERSION,
    MATRIX_SCHEMA_VERSION,
    REQUIRED_STATUS_KEYS,
    build_config_fingerprint,
    build_case_id,
    build_case_payload,
    build_invalid_environment_payload,
    build_matrix_summary,
    build_runtime_metadata,
    build_torch_compile_metadata,
    compute_speedup,
    evaluate_stage_diff,
    normalize_moe_routing_mode,
    parse_batch_sizes,
    parse_nccl_tuples,
    parse_runtime_backends,
    should_retry,
    should_skip_existing,
    validate_case_payload,
    validate_matrix_summary,
)
from examples.attn_moe_overlap.megatron_layer_runtime import (
    RuntimeConfig,
    TorchCompileFailure,
    _build_equal_token_routing_state,
    _install_torch_compile_safe_moe_cpu_handoff,
    _prepare_stage_callable,
    classify_exception,
    describe_attention_runtime,
    _resolve_profiler_schedule,
    _run_iteration_schedule,
)
from examples.attn_moe_overlap.step7_megatron_ep_overlap import (
    _aggregate_stage_results,
    _build_run_config,
    _build_case_rerun_command,
    _collapse_timed_window_s,
    _ensure_output_dir_identity_matches,
    _normalize_launch_failure,
    _run_case_attempt,
    _run_torch_profiler_capture,
    _select_torch_profiler_cases,
)

pytestmark = [pytest.mark.cpu_only]


def _sample_topology() -> dict:
    return {
        "attn_dp_size": 2,
        "moe_ep_size": 4,
        "attn_gpu_ids": [0, 1],
        "moe_gpu_ids": [0, 1, 2, 3],
        "seed": 1234,
    }


def _sample_nccl() -> dict:
    return {
        "socket_nthreads": 4,
        "max_nchannels": 16,
        "max_ctas": 32,
        "tuple": "4,16,32",
        "env_applied": True,
    }


def _sample_nccl_off() -> dict:
    return {
        "socket_nthreads": None,
        "max_nchannels": None,
        "max_ctas": None,
        "tuple": "off",
        "env_applied": False,
    }


def _sample_case_payload(
    case_id: str,
    mode: str,
    status: str = "ok",
    *,
    seq_len: int = 512,
    batch_size: int = 1,
    runtime_backend: str = "mps_only",
    green_ctx_attn_sms: int | None = None,
    green_ctx_moe_sms: int | None = None,
) -> dict:
    return build_case_payload(
        case_id=case_id,
        status=status,
        mode=mode,
        seq_len=seq_len,
        batch_size=batch_size,
        runtime_backend=runtime_backend,
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
        moe_routing_mode="normal",
        runtime=build_runtime_metadata(
            runtime_backend=runtime_backend,
            green_ctx_attn_sms=green_ctx_attn_sms,
            green_ctx_moe_sms=green_ctx_moe_sms,
            granted_sms_by_role={"attn": [green_ctx_attn_sms], "moe": [green_ctx_moe_sms]},
            device_total_sms_by_role={"attn": [132], "moe": [132]},
        ),
        timing_ms={"total": 10.0, "timed_wall": 8.0, "attn": 4.0, "moe": 6.0},
        overlap_ms=1.5,
        finite={"all_finite": True, "first_nonfinite": None},
        stage_signatures={
            "attn": {"sum": 1.0, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
            "moe": {"sum": 2.0, "mean": 0.2, "std": 0.3, "max_abs": 2.5},
        },
        baseline_diff={
            "baseline_case_id": None,
            "all_within_tolerance": None,
            "stages": {"attn": None, "moe": None},
        },
        error={"code": None, "message": None, "traceback": None},
        attempt_count=1,
        retry_trigger="none",
        artifact_path=f"/tmp/{case_id}.json",
    )


def test_collapse_timed_window_spans_earliest_start_to_latest_end():
    window = _collapse_timed_window_s(
        [
            {"start_s": 10.25, "end_s": 10.75},
            {"start_s": 10.0, "end_s": 11.0},
            {"start_s": None, "end_s": None},
        ]
    )
    assert window["start_s"] == pytest.approx(10.0)
    assert window["end_s"] == pytest.approx(11.0)
    assert window["duration_ms"] == pytest.approx(1000.0)


def test_select_torch_profiler_cases_returns_backend_reruns_for_selected_pairs():
    mps_only_serial = _sample_case_payload("mps-only-serial", "serial", seq_len=1024, batch_size=1)
    mps_only_overlap = _sample_case_payload("mps-only-overlap", "overlap", seq_len=1024, batch_size=1)
    mps_only_overlap["timing_ms"]["timed_wall"] = 10.0
    mps_green_ctx_serial = _sample_case_payload(
        "green-serial",
        "serial",
        seq_len=1024,
        batch_size=1,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    mps_green_ctx_overlap = _sample_case_payload(
        "green-overlap",
        "overlap",
        seq_len=1024,
        batch_size=1,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    mps_green_ctx_overlap["timing_ms"]["timed_wall"] = 7.0

    selected = _select_torch_profiler_cases(
        [mps_only_serial, mps_only_overlap, mps_green_ctx_serial, mps_green_ctx_overlap],
        selection="representative",
    )

    assert [row["runtime_backend"] for row in selected] == ["mps_only", "mps_green_ctx"]
    assert [row["case_id"] for row in selected] == ["mps-only-overlap", "green-overlap"]
    assert all(row["mode"] == "overlap" for row in selected)


def test_select_torch_profiler_cases_supports_all_successful_selection():
    serial = _sample_case_payload("case-serial", "serial", seq_len=1024, batch_size=2)
    overlap = _sample_case_payload("case-overlap", "overlap", seq_len=1024, batch_size=2)
    failed = _sample_case_payload("case-failed", "overlap", status="runtime_error", seq_len=1024, batch_size=2)

    selected = _select_torch_profiler_cases([failed, overlap, serial], selection="all-successful")

    assert [row["case_id"] for row in selected] == ["case-overlap", "case-serial"]


def test_parse_batch_sizes_requires_positive_integers():
    assert parse_batch_sizes("1, 2,4") == [1, 2, 4]
    with pytest.raises(ValueError):
        parse_batch_sizes("1,0")


def test_parse_runtime_backends_dedupes_and_validates_values():
    assert parse_runtime_backends("mps_only,mps_green_ctx,mps_only") == ["mps_only", "mps_green_ctx"]
    with pytest.raises(ValueError):
        parse_runtime_backends("mps_only,unknown")


def test_normalize_moe_routing_mode_validates_values():
    assert normalize_moe_routing_mode("equal_tokens") == "equal_tokens"
    assert normalize_moe_routing_mode("NORMAL") == "normal"
    with pytest.raises(ValueError):
        normalize_moe_routing_mode("weird")


def test_build_runtime_metadata_omits_green_ctx_request_when_backend_disabled():
    runtime = build_runtime_metadata(
        runtime_backend="mps_only",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    assert runtime["green_ctx_enabled"] is False
    assert runtime["requested_sms_by_role"] == {"attn": None, "moe": None}


def test_resolve_profiler_schedule_defaults_wait_to_warmup():
    wait_iters, active_iters = _resolve_profiler_schedule(
        warmup_iters=100,
        timed_iters=100,
        profiler_wait_iters=None,
        profiler_active_timed_iters=5,
    )
    assert wait_iters == 100
    assert active_iters == 5


def test_resolve_profiler_schedule_supports_independent_wait_iters():
    wait_iters, active_iters = _resolve_profiler_schedule(
        warmup_iters=100,
        timed_iters=100,
        profiler_wait_iters=12,
        profiler_active_timed_iters=7,
    )
    assert wait_iters == 12
    assert active_iters == 7


def test_run_iteration_schedule_serial_lockstep_steps_once_and_skips_inactive_phase():
    barrier_calls: list[tuple[str, int]] = []
    forward_calls: list[tuple[str, int, bool]] = []
    profiler_steps: list[int] = []
    clock = iter(range(100, 200))

    windows = _run_iteration_schedule(
        execution_schedule="serial_lockstep",
        stage_role="attn",
        total_iters=3,
        warmup_iters=1,
        barrier_wait=lambda phase, iter_idx: barrier_calls.append((phase, iter_idx)),
        run_forward=lambda phase, iter_idx, is_timed: forward_calls.append((phase, iter_idx, is_timed)),
        profiler_step=lambda iter_idx: profiler_steps.append(iter_idx),
        now=lambda: float(next(clock)),
    )

    assert barrier_calls == [
        ("serial_start", 0),
        ("serial_between", 0),
        ("serial_end", 0),
        ("serial_start", 1),
        ("serial_between", 1),
        ("serial_end", 1),
        ("serial_start", 2),
        ("serial_between", 2),
        ("serial_end", 2),
    ]
    assert forward_calls == [
        ("attn", 0, False),
        ("attn", 1, True),
        ("attn", 2, True),
    ]
    assert profiler_steps == [0, 1, 2]
    assert windows["stage_timed_window_s"]["duration_ms"] == pytest.approx(4000.0)
    assert windows["schedule_timed_window_s"]["duration_ms"] == pytest.approx(6000.0)


def test_run_case_attempt_serial_uses_joint_launch_and_launch_timed_window(monkeypatch):
    calls: list[dict[str, object]] = []

    def _fake_launch_workers(*, stage_specs, common_config, timeout_s, mps_env):
        del timeout_s, mps_env
        calls.append({"stage_specs": stage_specs, "common_config": common_config})
        return {
            "status": "ok",
            "schedule_timed_window_s": {"start_s": 10.0, "end_s": 11.0, "duration_ms": 1000.0},
            "results": [
                {
                    "role": "attn",
                    "rank": 0,
                    "status": "ok",
                    "failure_origin": False,
                    "attention_backend": "auto",
                    "attention_impl": {
                        "requested_backend": "auto",
                        "config_attention_backend": "auto",
                        "transformer_impl": "transformer_engine",
                        "layer_class": "TransformerLayer",
                        "self_attention_class": "SelfAttention",
                        "core_attention_class": "TEDotProductAttention",
                        "nvte_backend_flags": {"flash": "1", "fused": "1", "unfused": "1"},
                    },
                    "moe_grouped_gemm": True,
                    "moe_token_dispatcher_type": "alltoall",
                    "overlap_moe_expert_parallel_comm": True,
                    "timing_ms": {"cuda": 1.25, "step_total": 1.5, "timed_wall": 400.0},
                    "timed_window_s": {"start_s": 10.1, "end_s": 10.5, "duration_ms": 400.0},
                    "schedule_timed_window_s": {"start_s": 10.0, "end_s": 11.0, "duration_ms": 1000.0},
                    "enqueue_windows": [(1.0, 1.1)],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 1.0, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
                    "moe_routing_mode": "normal",
                    "tokens_per_expert": None,
                    "local_tokens_per_expert": None,
                },
                {
                    "role": "moe",
                    "rank": 0,
                    "status": "ok",
                    "failure_origin": False,
                    "attention_backend": "auto",
                    "attention_impl": {
                        "requested_backend": "auto",
                        "config_attention_backend": "auto",
                        "transformer_impl": "transformer_engine",
                        "layer_class": "TransformerLayer",
                        "self_attention_class": "SelfAttention",
                        "core_attention_class": "TEDotProductAttention",
                        "nvte_backend_flags": {"flash": "1", "fused": "1", "unfused": "1"},
                    },
                    "moe_grouped_gemm": True,
                    "moe_token_dispatcher_type": "alltoall",
                    "overlap_moe_expert_parallel_comm": True,
                    "timing_ms": {"cuda": 2.5, "step_total": 3.0, "timed_wall": 700.0},
                    "timed_window_s": {"start_s": 10.2, "end_s": 10.9, "duration_ms": 700.0},
                    "schedule_timed_window_s": {"start_s": 10.0, "end_s": 11.0, "duration_ms": 1000.0},
                    "enqueue_windows": [(1.2, 1.4)],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0, "mean": 0.2, "std": 0.3, "max_abs": 2.5},
                    "moe_routing_mode": "normal",
                    "tokens_per_expert": None,
                    "local_tokens_per_expert": None,
                },
            ],
        }

    monkeypatch.setattr(
        "examples.attn_moe_overlap.step7_megatron_ep_overlap._launch_workers",
        _fake_launch_workers,
    )

    result = _run_case_attempt(
        mode="serial",
        common_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            "runtime_backend": "mps_only",
            "green_ctx_attn_sms": None,
            "green_ctx_moe_sms": None,
            "dtype": "bf16",
            "seq_len": 1024,
            "batch_size": 1,
            "seed": 1234,
            "warmup_iters": 1,
            "timed_iters": 3,
            "moe_ep_size": 4,
            "num_experts": None,
            "moe_routing_mode": "normal",
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": True,
            "attention_backend": "auto",
            "nccl_tuple": (4, 16, 32),
            "profiler_trace_root": None,
            "profiler_wait_iters": None,
            "profiler_active_timed_iters": 2,
        },
        attn_gpu_ids=[0],
        moe_gpu_ids=[0],
        timeout_s=180.0,
        mps_env={},
    )

    assert len(calls) == 1
    assert calls[0]["common_config"]["execution_schedule"] == "serial_lockstep"
    assert {spec["role"] for spec in calls[0]["stage_specs"]} == {"attn", "moe"}
    assert result["status"] == "ok"
    assert result["attention_backend"] == {"requested": "auto", "attn": "auto", "moe": "auto"}
    assert result["attention_impl"]["attn"]["core_attention_class"] == "TEDotProductAttention"
    assert result["attention_impl"]["attn"]["transformer_impl"] == "transformer_engine"
    assert result["moe_runtime"] == {
        "grouped_gemm": {"requested": True, "attn": True, "moe": True},
        "token_dispatcher_type": {"requested": "alltoall", "attn": "alltoall", "moe": "alltoall"},
        "overlap_expert_parallel_comm": {"requested": True, "attn": True, "moe": True},
    }
    assert result["timing_ms"]["timed_wall"] == pytest.approx(1000.0)
    assert result["timing_ms"]["attn"] == pytest.approx(1.25)
    assert result["timing_ms"]["moe"] == pytest.approx(2.5)
    assert result["overlap_ms"] == pytest.approx(0.0)
    assert result["runtime"]["green_ctx_enabled"] is False
    assert result["tokens_per_expert"] is None


def test_normalize_launch_failure_prefers_root_cause_worker_error():
    normalized = _normalize_launch_failure(
        results=[
            {
                "role": "attn",
                "rank": 0,
                "status": "oom",
                "failure_origin": True,
                "error": {"code": "oom", "message": "CUDA out of memory", "traceback": None},
            },
            {
                "role": "moe",
                "rank": 0,
                "status": "runtime_error",
                "failure_origin": False,
                "error": {"code": "runtime_error", "message": "Schedule aborted", "traceback": None},
            },
        ],
        expected=4,
        timed_out=True,
        exitcodes={"123": 1},
    )

    assert normalized["status"] == "oom"
    assert normalized["error"]["message"] == "CUDA out of memory"


def test_run_case_attempt_reduces_equal_tokens_counts(monkeypatch):
    def _fake_launch_workers(*, stage_specs, common_config, timeout_s, mps_env):
        del stage_specs, common_config, timeout_s, mps_env
        return {
            "status": "ok",
            "schedule_timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
            "results": [
                {
                    "role": "attn",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 1.0, "step_total": 1.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 1.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": None,
                    "local_tokens_per_expert": None,
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [2, 2, 0, 0],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 1,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [0, 0, 1, 1],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
            ],
        }

    monkeypatch.setattr(
        "examples.attn_moe_overlap.step7_megatron_ep_overlap._launch_workers",
        _fake_launch_workers,
    )

    result = _run_case_attempt(
        mode="overlap",
        common_config={
            "model_name": "repo/model",
            "model_type": "dummy",
            "runtime_backend": "mps_only",
            "green_ctx_attn_sms": None,
            "green_ctx_moe_sms": None,
            "dtype": "bf16",
            "seq_len": 8,
            "batch_size": 1,
            "seed": 1234,
            "warmup_iters": 1,
            "timed_iters": 2,
            "moe_ep_size": 2,
            "num_experts": 4,
            "moe_routing_mode": "equal_tokens",
            "attn_mps_active_thread_pct": None,
            "moe_grouped_gemm": False,
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": False,
            "attention_backend": "auto",
            "nccl_tuple": None,
        },
        attn_gpu_ids=[0],
        moe_gpu_ids=[0, 1],
        timeout_s=10.0,
        mps_env={},
    )

    assert result["tokens_per_expert"] == [2, 2, 1, 1]


def test_run_case_attempt_rejects_mismatched_equal_tokens_vectors(monkeypatch):
    def _fake_launch_workers(*, stage_specs, common_config, timeout_s, mps_env):
        del stage_specs, common_config, timeout_s, mps_env
        return {
            "status": "ok",
            "schedule_timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
            "results": [
                {
                    "role": "attn",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 1.0, "step_total": 1.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 1.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": None,
                    "local_tokens_per_expert": None,
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [2, 2, 0, 0],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 1,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [0, 0, 1],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
            ],
        }

    monkeypatch.setattr(
        "examples.attn_moe_overlap.step7_megatron_ep_overlap._launch_workers",
        _fake_launch_workers,
    )

    with pytest.raises(RuntimeError, match="identical lengths"):
        _run_case_attempt(
            mode="overlap",
            common_config={
                "model_name": "repo/model",
                "model_type": "dummy",
                "runtime_backend": "mps_only",
                "green_ctx_attn_sms": None,
                "green_ctx_moe_sms": None,
                "dtype": "bf16",
                "seq_len": 8,
                "batch_size": 1,
                "seed": 1234,
                "warmup_iters": 1,
                "timed_iters": 2,
                "moe_ep_size": 2,
                "num_experts": 4,
                "moe_routing_mode": "equal_tokens",
                "attn_mps_active_thread_pct": None,
                "moe_grouped_gemm": False,
                "moe_token_dispatcher_type": "alltoall",
                "overlap_moe_expert_parallel_comm": False,
                "attention_backend": "auto",
                "nccl_tuple": None,
            },
            attn_gpu_ids=[0],
            moe_gpu_ids=[0, 1],
            timeout_s=10.0,
            mps_env={},
        )


def test_run_case_attempt_equal_tokens_tolerates_non_finite_outputs(monkeypatch):
    def _fake_launch_workers(*, stage_specs, common_config, timeout_s, mps_env):
        del stage_specs, common_config, timeout_s, mps_env
        return {
            "status": "ok",
            "schedule_timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
            "results": [
                {
                    "role": "attn",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 1.0, "step_total": 1.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 1.0},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": None,
                    "local_tokens_per_expert": None,
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 0,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {
                        "all_finite": False,
                        "first_nonfinite": {"module": "moe_layer", "phase": "forward", "tensor": "output", "iter": 0},
                    },
                    "output_signature": {"sum": float("nan")},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [2, 2, 0, 0],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
                {
                    "role": "moe",
                    "rank": 1,
                    "status": "ok",
                    "timing_ms": {"cuda": 2.0, "step_total": 2.0},
                    "timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
                    "enqueue_windows": [],
                    "finite": {
                        "all_finite": False,
                        "first_nonfinite": {"module": "moe_layer", "phase": "forward", "tensor": "output", "iter": 0},
                    },
                    "output_signature": {"sum": float("nan")},
                    "moe_routing_mode": "equal_tokens",
                    "tokens_per_expert": [2, 2, 1, 1],
                    "local_tokens_per_expert": [0, 0, 1, 1],
                    "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": 132},
                },
            ],
        }

    monkeypatch.setattr(
        "examples.attn_moe_overlap.step7_megatron_ep_overlap._launch_workers",
        _fake_launch_workers,
    )

    result = _run_case_attempt(
        mode="serial",
        common_config={
            "model_name": "repo/model",
            "model_type": "dummy",
            "runtime_backend": "mps_only",
            "green_ctx_attn_sms": None,
            "green_ctx_moe_sms": None,
            "dtype": "bf16",
            "seq_len": 8,
            "batch_size": 1,
            "seed": 1234,
            "warmup_iters": 1,
            "timed_iters": 2,
            "moe_ep_size": 2,
            "num_experts": 4,
            "moe_routing_mode": "equal_tokens",
            "moe_grouped_gemm": False,
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": False,
            "attention_backend": "auto",
            "nccl_tuple": None,
        },
        attn_gpu_ids=[0],
        moe_gpu_ids=[0, 1],
        timeout_s=10.0,
        mps_env={},
    )

    assert result["status"] == "ok"
    assert result["finite"]["all_finite"] is False
    assert result["tokens_per_expert"] == [2, 2, 1, 1]


def test_run_torch_profiler_capture_uses_script_rerun_command(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    class _Completed:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, capture_output, text):
        del capture_output, text
        calls.append(list(cmd))
        trace_dir = Path(cmd[cmd.index("--torch-profiler-trace-dir") + 1])
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "worker0.pt.trace.json").write_text("{}")
        return _Completed()

    monkeypatch.setattr("examples.attn_moe_overlap.step7_megatron_ep_overlap.subprocess.run", _fake_run)

    args = SimpleNamespace(
        capture_torch_profiler="on",
        torch_profiler_selection="all-successful",
        output_dir=str(tmp_path / "out"),
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        attn_dp_size=4,
        moe_ep_size=4,
        seed=1234,
        batch_size=1,
        warmup_iters=1,
        timed_iters=3,
        worker_timeout_s=180.0,
        attention_backend="fused",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        torch_profiler_wait_iters=11,
        torch_profiler_active_iters=2,
    )
    run_config = {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "model_type": "qwen3_moe",
        "topology": {"attn_dp_size": 4, "moe_ep_size": 4, "attn_gpu_ids": [0, 1, 2, 3], "moe_gpu_ids": [0, 1, 2, 3]},
        "warmup_iters": 1,
        "timed_iters": 3,
        "worker_timeout_s": 180.0,
        "num_experts": None,
        "moe_routing_mode": "equal_tokens",
        "mps_active_thread_pct": None,
        "attn_mps_active_thread_pct": 80,
        "attention_backend": "fused",
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": True,
        "torch_compile": "off",
    }
    serial = _sample_case_payload("case-serial", "serial", status="ok", seq_len=2048, batch_size=4)
    overlap = _sample_case_payload("case-overlap", "overlap", status="ok", seq_len=2048, batch_size=4)
    serial["moe_routing_mode"] = "equal_tokens"
    overlap["moe_routing_mode"] = "equal_tokens"
    green_serial = _sample_case_payload(
        "green-serial",
        "serial",
        status="ok",
        seq_len=2048,
        batch_size=4,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    green_serial["moe_routing_mode"] = "equal_tokens"
    green_overlap = _sample_case_payload(
        "green-overlap",
        "overlap",
        status="ok",
        seq_len=2048,
        batch_size=4,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    green_overlap["moe_routing_mode"] = "equal_tokens"

    status = _run_torch_profiler_capture(
        args=args,
        run_config=run_config,
        cases=[serial, overlap, green_serial, green_overlap],
    )

    assert status == "ok"
    assert len(calls) == 4
    assert calls[0][0] == sys.executable
    assert calls[0][1] == str((REPO_ROOT / "examples/attn_moe_overlap/step7_megatron_ep_overlap.py").resolve())
    assert "-m" not in calls[0]
    assert calls[0][calls[0].index("--attention-backend") + 1] == "fused"
    assert calls[0][calls[0].index("--moe-token-dispatcher-type") + 1] == "alltoall"
    assert calls[0][calls[0].index("--moe-routing-mode") + 1] == "equal_tokens"
    assert "--moe-grouped-gemm" in calls[0]
    assert "--overlap-moe-expert-parallel-comm" in calls[0]
    assert calls[0][calls[0].index("--batch-size") + 1] == "4"
    assert calls[0][calls[0].index("--torch-profiler-wait-iters") + 1] == "11"
    assert calls[0][calls[0].index("--attn-mps-active-thread-pct") + 1] == "80"
    assert sorted(cmd[cmd.index("--runtime-backends") + 1] for cmd in calls) == [
        "mps_green_ctx",
        "mps_green_ctx",
        "mps_only",
        "mps_only",
    ]
    green_cmd = next(cmd for cmd in calls if cmd[cmd.index("--runtime-backends") + 1] == "mps_green_ctx")
    assert green_cmd[green_cmd.index("--green-ctx-attn-sms") + 1] == "64"
    assert green_cmd[green_cmd.index("--green-ctx-moe-sms") + 1] == "64"

    trace_index = json.loads((tmp_path / "out" / "torch_profiler" / "trace_index.json").read_text())
    assert trace_index["status"] == "ok"
    assert trace_index["selection"] == "all-successful"
    assert all(entry["trace_files"] == [f"{entry['trace_dir']}/worker0.pt.trace.json"] for entry in trace_index["entries"])
    assert all(entry["profiler_status"] == "ok" for entry in trace_index["entries"])


def test_case_id_is_deterministic_and_sensitive_to_batch_size_and_nccl_tuple():
    kwargs = {
        "mode": "serial",
        "seq_len": 512,
        "batch_size": 1,
        "runtime_backend": "mps_only",
        "green_ctx_attn_sms": None,
        "green_ctx_moe_sms": None,
        "moe_routing_mode": "normal",
        "dtype": "bf16",
        "seed": 1234,
        "attn_dp_size": 2,
        "moe_ep_size": 4,
        "attn_gpu_ids": [0, 1],
        "moe_gpu_ids": [0, 1, 2, 3],
    }
    case_id_a = build_case_id(**kwargs, nccl_tuple=(4, 16, 32))
    case_id_b = build_case_id(**kwargs, nccl_tuple=(4, 16, 32))
    case_id_c = build_case_id(**kwargs, nccl_tuple=(8, 16, 32))
    case_id_d = build_case_id(**{**kwargs, "batch_size": 4}, nccl_tuple=(4, 16, 32))
    case_id_e = build_case_id(
        **{**kwargs, "runtime_backend": "mps_green_ctx", "green_ctx_attn_sms": 64, "green_ctx_moe_sms": 64},
        nccl_tuple=(4, 16, 32),
    )
    case_id_f = build_case_id(**{**kwargs, "moe_routing_mode": "equal_tokens"}, nccl_tuple=(4, 16, 32))
    assert case_id_a == case_id_b
    assert case_id_a != case_id_c
    assert case_id_a != case_id_d
    assert case_id_a != case_id_e
    assert case_id_a != case_id_f


def test_case_id_supports_disabled_nccl_tuning():
    case_id = build_case_id(
        mode="serial",
        seq_len=512,
        batch_size=1,
        runtime_backend="mps_only",
        green_ctx_attn_sms=None,
        green_ctx_moe_sms=None,
        moe_routing_mode="normal",
        dtype="bf16",
        seed=1234,
        attn_dp_size=2,
        moe_ep_size=4,
        attn_gpu_ids=[0, 1],
        moe_gpu_ids=[0, 1, 2, 3],
        nccl_tuple=None,
    )
    assert "nccl-off" in case_id


def test_case_payload_validation_requires_contract_keys():
    payload = _sample_case_payload("case-a", "serial")
    assert validate_case_payload(payload) == []
    del payload["timing_ms"]
    errors = validate_case_payload(payload)
    assert any("timing_ms" in error for error in errors)


def test_validate_case_payload_rejects_missing_equal_tokens_metadata():
    payload = build_case_payload(
        case_id="eq-case",
        status="ok",
        mode="overlap",
        seq_len=512,
        batch_size=1,
        runtime_backend="mps_only",
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
        moe_routing_mode="equal_tokens",
        timing_ms={"total": 1.0, "timed_wall": 0.9, "attn": 0.4, "moe": 0.5},
        overlap_ms=0.0,
        finite={"all_finite": True, "first_nonfinite": None},
        stage_signatures={"attn": None, "moe": None},
        error={"code": None, "message": None, "traceback": None},
    )

    errors = validate_case_payload(payload)
    assert any("tokens_per_expert" in error for error in errors)


def test_validate_case_payload_allows_missing_equal_tokens_metadata_for_failed_cases():
    payload = build_case_payload(
        case_id="eq-failed",
        status="invalid_environment",
        mode="overlap",
        seq_len=512,
        batch_size=1,
        runtime_backend="mps_only",
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
        moe_routing_mode="equal_tokens",
        timing_ms={"total": None, "timed_wall": None, "attn": None, "moe": None},
        overlap_ms=0.0,
        finite={"all_finite": False, "first_nonfinite": None},
        stage_signatures={"attn": None, "moe": None},
        error={"code": "invalid_environment", "message": "boom", "traceback": None},
    )

    assert validate_case_payload(payload) == []


def test_build_equal_token_routing_state_balances_tokens():
    hidden_states = torch.zeros((7, 1, 8), dtype=torch.bfloat16)
    state = _build_equal_token_routing_state(
        hidden_states=hidden_states,
        num_experts=4,
        top_k=2,
        local_expert_indices=[2, 3],
    )

    assert state.tokens_per_expert == [3, 4, 4, 3]
    assert state.local_tokens_per_expert == [0, 0, 4, 3]
    assert max(state.tokens_per_expert) - min(state.tokens_per_expert) == 1
    assert state.routing_map.dtype == torch.bool
    assert torch.allclose(state.probs.sum(dim=1), torch.ones(7, dtype=hidden_states.dtype))
    assert torch.all(state.routing_map.sum(dim=1) == 2)


def test_speedup_and_stage_diff_tolerance_mapping():
    assert compute_speedup(20.0, 10.0) == pytest.approx(2.0)
    diff = evaluate_stage_diff(
        stage_name="attn",
        dtype="fp32",
        test_signature={"sum": 1.000001, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
        ref_signature={"sum": 1.0, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
    )
    assert diff["within_tolerance"] is True


def test_stage_diff_marks_numerical_mismatch_for_large_relative_error():
    diff = evaluate_stage_diff(
        stage_name="moe",
        dtype="fp32",
        test_signature={"sum": 2.0, "mean": 0.2, "std": 1.2, "max_abs": 2.5},
        ref_signature={"sum": 2.0, "mean": 0.2, "std": 0.2, "max_abs": 2.5},
    )
    assert diff["within_tolerance"] is False
    assert diff["max_abs_diff"] is not None
    assert diff["max_rel_diff"] is not None


def test_retry_policy_allows_single_retry_only_for_oom_timeout():
    assert should_retry("oom", 1) is True
    assert should_retry("timeout", 1) is True
    assert should_retry("runtime_error", 1) is False
    assert should_retry("oom", 2) is False


def test_runtime_config_threads_attention_and_moe_overrides_into_engine_config():
    runtime_config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="attn",
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=True,
        dtype="bf16",
        seq_len=1024,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=1,
        num_experts=None,
    )

    from examples.attn_moe_overlap.megatron_layer_runtime import MegatronSingleLayerRuntime

    trainer_config = MegatronSingleLayerRuntime(runtime_config)._build_trainer_config()
    assert trainer_config["engine_config"]["attention_backend"] == "auto"
    assert trainer_config["engine_config"]["megatron_moe_grouped_gemm"] is True
    assert trainer_config["engine_config"]["megatron_moe_token_dispatcher_type"] == "alltoall"
    assert trainer_config["engine_config"]["megatron_overlap_moe_expert_parallel_comm"] is True


def test_runtime_config_omits_false_boolean_moe_overrides():
    runtime_config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="attn",
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        dtype="bf16",
        seq_len=1024,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=1,
        num_experts=None,
    )

    from examples.attn_moe_overlap.megatron_layer_runtime import MegatronSingleLayerRuntime

    trainer_config = MegatronSingleLayerRuntime(runtime_config)._build_trainer_config()
    assert trainer_config["engine_config"]["megatron_moe_grouped_gemm"] is None
    assert trainer_config["engine_config"]["megatron_moe_token_dispatcher_type"] == "alltoall"
    assert trainer_config["engine_config"]["megatron_overlap_moe_expert_parallel_comm"] is None


def test_runtime_config_equal_tokens_preserves_model_topk():
    runtime_config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="moe",
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        dtype="bf16",
        seq_len=1024,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=2,
        num_experts=8,
        moe_routing_mode="equal_tokens",
    )

    from examples.attn_moe_overlap.megatron_layer_runtime import MegatronSingleLayerRuntime

    trainer_config = MegatronSingleLayerRuntime(runtime_config)._build_trainer_config()
    assert "megatron_moe_router_topk" not in trainer_config["engine_config"]
    assert "megatron_moe_router_pre_softmax" not in trainer_config["engine_config"]


def test_invalid_environment_payload_contract():
    payload = build_invalid_environment_payload(
        case_id="case-invalid",
        mode="serial",
        seq_len=512,
        batch_size=2,
        runtime_backend="mps_green_ctx",
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
        runtime=build_runtime_metadata(
            runtime_backend="mps_green_ctx",
            green_ctx_attn_sms=64,
            green_ctx_moe_sms=64,
        ),
        message="CUDA is not available",
    )
    assert payload["status"] == "invalid_environment"
    assert payload["error"]["code"] == "invalid_environment"
    assert payload["batch_size"] == 2
    assert payload["runtime_backend"] == "mps_green_ctx"


def test_parse_nccl_tuples_normalization_and_mixed_arg_rejection():
    assert parse_nccl_tuples(
        nccl_tuples=None,
        nccl_socket_nthreads=None,
        nccl_max_nchannels=None,
        nccl_max_ctas=None,
    ) == [None]
    tuples = parse_nccl_tuples(
        nccl_tuples="4,16,32;8,32,64;4,16,32",
        nccl_socket_nthreads=None,
        nccl_max_nchannels=None,
        nccl_max_ctas=None,
    )
    assert tuples == [(4, 16, 32), (8, 32, 64)]
    assert parse_nccl_tuples(
        nccl_tuples="off",
        nccl_socket_nthreads=None,
        nccl_max_nchannels=None,
        nccl_max_ctas=None,
    ) == [None]
    with pytest.raises(ValueError):
        parse_nccl_tuples(
            nccl_tuples="4,16,32",
            nccl_socket_nthreads=4,
            nccl_max_nchannels=None,
            nccl_max_ctas=None,
        )
    with pytest.raises(ValueError):
        parse_nccl_tuples(
            nccl_tuples="off;4,16,32",
            nccl_socket_nthreads=None,
            nccl_max_nchannels=None,
            nccl_max_ctas=None,
        )


def test_describe_attention_runtime_reports_layer_and_nvte_flags(monkeypatch):
    monkeypatch.setenv("NVTE_FLASH_ATTN", "1")
    monkeypatch.setenv("NVTE_FUSED_ATTN", "1")
    monkeypatch.setenv("NVTE_UNFUSED_ATTN", "1")

    class _Backend:
        name = "auto"

    class _Config:
        attention_backend = _Backend()
        transformer_impl = "transformer_engine"

    class _CoreAttention:
        pass

    class _SelfAttention:
        core_attention = _CoreAttention()

    class _Layer:
        config = _Config()
        self_attention = _SelfAttention()

    info = describe_attention_runtime(_Layer(), requested_backend="auto")
    assert info["requested_backend"] == "auto"
    assert info["config_attention_backend"] == "auto"
    assert info["transformer_impl"] == "transformer_engine"
    assert info["core_attention_class"] == "_CoreAttention"
    assert info["nvte_backend_flags"] == {"flash": "1", "fused": "1", "unfused": "1"}


def test_matrix_summary_contract_and_status_count_invariant():
    serial = _sample_case_payload("case-serial", "serial", status="ok", seq_len=1024, batch_size=2)
    overlap = _sample_case_payload("case-overlap", "overlap", status="oom", seq_len=1024, batch_size=2)
    overlap["attempt_count"] = 2
    identity_fields = {
        "seq_lens": [1024],
        "batch_sizes": [2],
        "dtypes": ["bf16"],
        "nccl_tuples": ["4,16,32"],
        "runtime_backends": ["mps_only"],
        "green_ctx_sms": {"attn": None, "moe": None},
        "device_sm_signature": {"0": 132, "1": 132},
        "moe_routing_mode": "normal",
        "torch_compile": "off",
    }
    summary = build_matrix_summary(
        run_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            **identity_fields,
            "config_fingerprint": build_config_fingerprint(identity_fields),
            "batch_size": 2,
        },
        cases=[serial, overlap],
        total_points=2,
    )
    assert validate_matrix_summary(summary) == []
    assert summary["schema_version"] == MATRIX_SCHEMA_VERSION
    by_status = summary["counts"]["by_status"]
    assert by_status["ok"] == 1
    assert by_status["oom"] == 1
    assert sum(by_status.values()) == summary["counts"]["completed_cases"]
    for key in REQUIRED_STATUS_KEYS:
        assert key in by_status
    assert summary["counts"]["comparison_points"] == 1
    assert summary["counts"]["backend_pair_points"] == 0
    comparison_row = summary["comparison_rows"][0]
    assert comparison_row["seq_len"] == 1024
    assert comparison_row["batch_size"] == 2
    assert comparison_row["runtime_backend"] == "mps_only"
    assert comparison_row["serial_status"] == "ok"
    assert comparison_row["overlap_status"] == "oom"
    assert comparison_row["overlap_total_ms"] is None
    assert comparison_row["timed_speedup_vs_serial"] is None


def test_matrix_summary_persists_backend_pairs_for_same_code_comparisons():
    mps_only_serial = _sample_case_payload("mps-only-serial", "serial", seq_len=1024, batch_size=2)
    mps_only_overlap = _sample_case_payload("mps-only-overlap", "overlap", seq_len=1024, batch_size=2)
    mps_only_overlap["timing_ms"]["timed_wall"] = 9.0
    mps_only_overlap["timing_ms"]["total"] = 11.0
    mps_only_overlap["overlap"]["timed_speedup_vs_serial"] = 1.11

    green_serial = _sample_case_payload(
        "green-serial",
        "serial",
        seq_len=1024,
        batch_size=2,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    green_overlap = _sample_case_payload(
        "green-overlap",
        "overlap",
        seq_len=1024,
        batch_size=2,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    green_overlap["timing_ms"]["timed_wall"] = 7.0
    green_overlap["timing_ms"]["total"] = 9.0
    green_overlap["overlap"]["timed_speedup_vs_serial"] = 1.43

    identity_fields = {
        "seq_lens": [1024],
        "batch_sizes": [2],
        "dtypes": ["bf16"],
        "nccl_tuples": ["4,16,32"],
        "runtime_backends": ["mps_only", "mps_green_ctx"],
        "green_ctx_sms": {"attn": 64, "moe": 64},
        "device_sm_signature": {"0": 132, "1": 132},
        "torch_compile": "off",
    }
    summary = build_matrix_summary(
        run_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            **identity_fields,
            "config_fingerprint": build_config_fingerprint(identity_fields),
        },
        cases=[mps_only_serial, mps_only_overlap, green_serial, green_overlap],
        total_points=4,
    )

    backend_pairs = summary["backend_pair_rows"]
    assert summary["counts"]["backend_pair_points"] == 1
    assert backend_pairs[0]["pair_status"] == "ok"
    assert backend_pairs[0]["mps_only_overlap_case_id"] == "mps-only-overlap"
    assert backend_pairs[0]["mps_green_ctx_overlap_case_id"] == "green-overlap"
    assert backend_pairs[0]["mps_only_serial_timed_speedup_vs_serial"] == pytest.approx(1.0)
    assert backend_pairs[0]["overlap_timed_speedup_mps_green_ctx_vs_mps_only"] == pytest.approx(9.0 / 7.0)
    assert backend_pairs[0]["delta_overlap_timed_wall_ms"] == pytest.approx(2.0)
    assert backend_pairs[0]["delta_timed_speedup_vs_serial"] == pytest.approx(0.32, abs=1e-6)


def test_matrix_summary_marks_backend_pair_as_both_failed_when_both_backends_fail():
    mps_only_serial = _sample_case_payload("mps-only-serial", "serial", seq_len=1024, batch_size=2)
    mps_only_overlap = _sample_case_payload("mps-only-overlap", "overlap", status="runtime_error", seq_len=1024, batch_size=2)
    green_serial = _sample_case_payload(
        "green-serial",
        "serial",
        seq_len=1024,
        batch_size=2,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )
    green_overlap = _sample_case_payload(
        "green-overlap",
        "overlap",
        status="runtime_error",
        seq_len=1024,
        batch_size=2,
        runtime_backend="mps_green_ctx",
        green_ctx_attn_sms=64,
        green_ctx_moe_sms=64,
    )

    identity_fields = {
        "seq_lens": [1024],
        "batch_sizes": [2],
        "dtypes": ["bf16"],
        "nccl_tuples": ["4,16,32"],
        "runtime_backends": ["mps_only", "mps_green_ctx"],
        "green_ctx_sms": {"attn": 64, "moe": 64},
        "device_sm_signature": {"0": 132, "1": 132},
        "torch_compile": "off",
    }
    summary = build_matrix_summary(
        run_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            **identity_fields,
            "config_fingerprint": build_config_fingerprint(identity_fields),
        },
        cases=[mps_only_serial, mps_only_overlap, green_serial, green_overlap],
        total_points=4,
    )

    backend_pairs = summary["backend_pair_rows"]
    assert summary["counts"]["backend_pair_points"] == 1
    assert backend_pairs[0]["pair_status"] == "both_failed"
    assert backend_pairs[0]["mps_only_overlap_status"] == "runtime_error"
    assert backend_pairs[0]["mps_green_ctx_overlap_status"] == "runtime_error"


def test_ensure_output_dir_identity_matches_rejects_config_mismatch(tmp_path):
    output_dir = tmp_path / "matrix"
    output_dir.mkdir()
    summary_path = output_dir / "matrix_summary.json"
    existing_identity = {
        "seq_lens": [1024],
        "batch_sizes": [1],
        "dtypes": ["bf16"],
        "nccl_tuples": ["4,16,32"],
        "runtime_backends": ["mps_only"],
        "green_ctx_sms": {"attn": None, "moe": None},
        "device_sm_signature": {"0": 132},
        "capture_torch_profiler": "off",
        "torch_profiler_selection": None,
        "torch_profiler_wait_iters": None,
        "torch_profiler_active_iters": None,
        "moe_routing_mode": "normal",
        "torch_compile": "off",
    }
    summary_path.write_text(
        json.dumps(
            {
                "run_config": {
                    **existing_identity,
                    "config_fingerprint": build_config_fingerprint(existing_identity),
                }
            }
        )
    )
    with pytest.raises(RuntimeError):
        _ensure_output_dir_identity_matches(
            output_dir,
            {
                **existing_identity,
                "runtime_backends": ["mps_only", "mps_green_ctx"],
                "config_fingerprint": "different",
            },
        )


def test_build_run_config_persists_identity_fields_and_fingerprint():
    args = SimpleNamespace(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        single_mode=None,
        warmup_iters=1,
        timed_iters=2,
        capture_nsys="off",
        attention_backend="auto",
        moe_token_dispatcher_type="alltoall",
        moe_grouped_gemm=True,
        overlap_moe_expert_parallel_comm=False,
        capture_torch_profiler="off",
        torch_profiler_selection="representative",
        torch_profiler_wait_iters=None,
        torch_profiler_active_iters=2,
        worker_timeout_s=180.0,
        num_experts=None,
        moe_routing_mode="normal",
        torch_compile="on",
        mps_active_thread_pct=None,
        attn_mps_active_thread_pct=None,
    )
    run_config = _build_run_config(
        args=args,
        topology=_sample_topology(),
        seq_lens=[1024],
        batch_sizes=[1, 2],
        dtypes=["bf16"],
        nccl_tuples=[(4, 16, 32)],
        runtime_backends=["mps_only", "mps_green_ctx"],
        green_ctx_sms={"attn": 64, "moe": 64},
        device_sm_signature={"0": 132, "1": 132},
        nsys_status="off",
        torch_profiler_status="off",
    )
    assert run_config["runtime_backends"] == ["mps_only", "mps_green_ctx"]
    assert run_config["green_ctx_sms"] == {"attn": 64, "moe": 64}
    assert run_config["device_sm_signature"] == {"0": 132, "1": 132}
    assert run_config["torch_profiler_selection"] is None
    assert run_config["moe_routing_mode"] == "normal"
    assert run_config["torch_compile"] == "on"
    assert isinstance(run_config["config_fingerprint"], str)


def test_resume_skip_behavior_honors_rerun_flag():
    payload = _sample_case_payload("case-existing", "serial", status="runtime_error")
    assert should_skip_existing(payload, rerun_existing=False) is True
    assert should_skip_existing(payload, rerun_existing=True) is False


def test_legacy_case_payload_is_not_reused_for_resume_or_comparison():
    payload = _sample_case_payload("case-legacy", "serial", status="ok")
    payload["schema_version"] = "megatron_ep_overlap.case.v1"
    del payload["batch_size"]
    assert should_skip_existing(payload, rerun_existing=False) is False


def test_case_payload_uses_current_schema():
    payload = _sample_case_payload("case-current", "serial", status="ok")
    assert payload["schema_version"] == CASE_SCHEMA_VERSION


def test_run_torch_profiler_capture_returns_off_without_successful_pair(tmp_path):
    args = SimpleNamespace(
        capture_torch_profiler="on",
        torch_profiler_selection="all-successful",
        output_dir=str(tmp_path / "out"),
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        attn_dp_size=4,
        moe_ep_size=4,
        seed=1234,
        batch_size=1,
        warmup_iters=1,
        timed_iters=3,
        worker_timeout_s=180.0,
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        torch_profiler_wait_iters=11,
        torch_profiler_active_iters=2,
    )
    run_config = {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "model_type": "qwen3_moe",
        "topology": {"attn_dp_size": 4, "moe_ep_size": 4, "attn_gpu_ids": [0, 1, 2, 3], "moe_gpu_ids": [0, 1, 2, 3]},
        "warmup_iters": 1,
        "timed_iters": 3,
        "worker_timeout_s": 180.0,
        "num_experts": None,
        "mps_active_thread_pct": None,
        "attn_mps_active_thread_pct": None,
        "attention_backend": "auto",
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": False,
        "torch_compile": "off",
    }
    status = _run_torch_profiler_capture(
        args=args,
        run_config=run_config,
        cases=[_sample_case_payload("case-bad", "overlap", status="runtime_error", batch_size=4)],
    )
    assert status == "off"
    assert not (tmp_path / "out" / "torch_profiler" / "trace_index.json").exists()


def _sample_worker_result(
    role: str,
    rank: int,
    *,
    status: str = "ok",
    torch_compile_requested: str = "off",
    torch_compile_status: str = "eager",
    error_code: str | None = None,
) -> dict[str, object]:
    resolved_error_code = error_code
    if resolved_error_code is None and status != "ok":
        resolved_error_code = status
    return {
        "role": role,
        "rank": rank,
        "status": status,
        "failure_origin": status != "ok",
        "attention_backend": "auto",
        "attention_impl": {"requested_backend": "auto"},
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": True,
        "timing_ms": {"cuda": 1.0, "step_total": 1.1, "timed_wall": 5.0},
        "timed_window_s": {"start_s": 1.0 + rank, "end_s": 1.5 + rank, "duration_ms": 500.0},
        "schedule_timed_window_s": {"start_s": 1.0, "end_s": 2.0, "duration_ms": 1000.0},
        "enqueue_windows": [(1.0, 1.1)],
        "finite": {"all_finite": True, "first_nonfinite": None},
        "output_signature": {"sum": 1.0, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
        "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": None},
        "moe_routing_mode": "normal",
        "tokens_per_expert": None,
        "local_tokens_per_expert": None,
        "torch_compile": {
            "requested": torch_compile_requested,
            "status": torch_compile_status,
        },
        "error": {
            "code": resolved_error_code,
            "message": None if resolved_error_code is None else str(resolved_error_code),
            "traceback": None,
        },
    }


def test_build_case_rerun_command_preserves_torch_compile_flag(tmp_path):
    case_payload = _sample_case_payload("case-compile", "overlap", status="ok")
    case_payload["torch_compile"] = build_torch_compile_metadata(
        requested="on",
        by_role={"attn": {"status": "compiled"}, "moe": {"status": "compiled"}},
    )
    run_config = {
        "model_name": "Qwen/Qwen3-30B-A3B",
        "model_type": "qwen3_moe",
        "topology": {"attn_dp_size": 2, "moe_ep_size": 4, "attn_gpu_ids": [0, 1], "moe_gpu_ids": [0, 1, 2, 3]},
        "worker_timeout_s": 180.0,
        "num_experts": None,
        "mps_active_thread_pct": None,
        "attn_mps_active_thread_pct": None,
        "attention_backend": "auto",
        "moe_grouped_gemm": True,
        "moe_token_dispatcher_type": "alltoall",
        "overlap_moe_expert_parallel_comm": False,
        "torch_compile": "on",
    }

    command = _build_case_rerun_command(
        run_config=run_config,
        case_payload=case_payload,
        output_dir=tmp_path / "rerun",
        capture_nsys="off",
        capture_torch_profiler="off",
        warmup_iters=100,
        timed_iters=100,
    )

    assert command[command.index("--torch-compile") + 1] == "on"


def test_validate_case_payload_rejects_missing_torch_compile_metadata():
    payload = _sample_case_payload("case-missing-compile", "serial", status="ok")
    del payload["torch_compile"]

    errors = validate_case_payload(payload)

    assert "missing key: torch_compile" in errors


def test_validate_case_payload_rejects_invalid_torch_compile_role_status():
    payload = _sample_case_payload("case-invalid-compile", "serial", status="ok")
    payload["torch_compile"] = {
        "requested": "on",
        "by_role": {"attn": {"status": "weird"}, "moe": {"status": "compiled"}},
    }

    errors = validate_case_payload(payload)

    assert any("torch_compile.by_role.attn.status" in error for error in errors)


def test_validate_matrix_summary_rejects_missing_torch_compile_run_config():
    identity_fields = {
        "seq_lens": [1024],
        "batch_sizes": [1],
        "dtypes": ["bf16"],
        "nccl_tuples": ["4,16,32"],
        "runtime_backends": ["mps_only"],
        "green_ctx_sms": {"attn": None, "moe": None},
        "device_sm_signature": {"0": 132},
        "moe_routing_mode": "normal",
        "torch_compile": "off",
    }
    summary = build_matrix_summary(
        run_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            **identity_fields,
            "config_fingerprint": build_config_fingerprint(identity_fields),
            "batch_size": 1,
        },
        cases=[_sample_case_payload("case-summary", "serial", status="ok", batch_size=1)],
        total_points=1,
    )
    del summary["run_config"]["torch_compile"]

    errors = validate_matrix_summary(summary)

    assert any("run_config.torch_compile" in error for error in errors)


@pytest.mark.parametrize(("stage_role", "expected_bias"), [("attn", 1.0), ("moe", 2.0)])
def test_prepare_stage_callable_compiles_requested_role(stage_role, expected_bias):
    class FakeLayer:
        def __init__(self):
            self.config = SimpleNamespace(moe_router_topk=1)
            self.mlp = SimpleNamespace(
                router=SimpleNamespace(forward=lambda input_tensor: input_tensor),
                token_dispatcher=SimpleNamespace(local_expert_indices=[0]),
            )

        def _forward_attention(self, hidden_states, attention_mask):
            del attention_mask
            return hidden_states + 1.0, None

        def _forward_mlp(self, hidden_states, inference_context=None):
            del inference_context
            return hidden_states + 2.0

    compile_calls: list[object] = []

    def _fake_compile(fn):
        compile_calls.append(fn)

        def _wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return _wrapped

    hidden_states = torch.ones((2, 1, 3), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role=stage_role,
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        dtype="bf16",
        seq_len=2,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=1,
        num_experts=1,
        torch_compile_enabled=True,
    )

    prepared = _prepare_stage_callable(
        config=config,
        layer=FakeLayer(),
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        compile_fn=_fake_compile,
    )
    output_tensor = prepared.run()

    assert torch.allclose(output_tensor, hidden_states + expected_bias)
    assert prepared.compile_payload == {"requested": "on", "status": "compiled"}
    assert len(compile_calls) == 1
    assert prepared.consume_equal_token_routing_state() is None


def test_install_torch_compile_safe_moe_cpu_handoff_wraps_and_restores(monkeypatch):
    module_globals: dict[str, object] = {}
    exec(
        """
def maybe_move_tensor_to_cpu(tensor, as_numpy=False, record_stream=False):
    return tensor, as_numpy, record_stream

class FakeTokenDispatcher:
    def dispatch_preprocess(self, hidden_states, routing_map, probs):
        return hidden_states, probs

    def token_dispatch(self, hidden_states, probs):
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        return hidden_states, probs

    def combine_preprocess(self, hidden_states):
        return hidden_states

    def token_combine(self, hidden_states):
        return hidden_states

    def combine_postprocess(self, hidden_states):
        return hidden_states

    def _maybe_dtoh_and_synchronize(self, point, tokens_per_expert=None):
        del point
        return maybe_move_tensor_to_cpu(tokens_per_expert, as_numpy=True, record_stream=True)
""",
        module_globals,
    )
    fake_dispatcher = module_globals["FakeTokenDispatcher"]()
    dispatcher_cls = module_globals["FakeTokenDispatcher"]
    original = module_globals["maybe_move_tensor_to_cpu"]
    wrapped_by_name: dict[str, object] = {}

    def _fake_disable(fn):
        def _wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        wrapped_by_name[fn.__name__] = _wrapped
        return _wrapped

    monkeypatch.setattr(torch._dynamo, "disable", _fake_disable)

    restore = _install_torch_compile_safe_moe_cpu_handoff(fake_dispatcher)

    assert module_globals["maybe_move_tensor_to_cpu"] is wrapped_by_name["maybe_move_tensor_to_cpu"]
    for method_name in [
        "dispatch_preprocess",
        "token_dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "token_combine",
        "combine_postprocess",
    ]:
        assert getattr(fake_dispatcher, method_name).__func__ is wrapped_by_name[method_name]

    restore()

    assert module_globals["maybe_move_tensor_to_cpu"] is original
    for method_name in [
        "dispatch_preprocess",
        "token_dispatch",
        "dispatch_postprocess",
        "combine_preprocess",
        "token_combine",
        "combine_postprocess",
    ]:
        assert getattr(fake_dispatcher, method_name).__func__ is dispatcher_cls.__dict__[method_name]


def test_prepare_stage_callable_compiles_equal_tokens_moe_with_router_override():
    class FakeLayer:
        def __init__(self):
            self.config = SimpleNamespace(moe_router_topk=1)
            self.mlp = SimpleNamespace(
                router=SimpleNamespace(forward=lambda input_tensor: (_ for _ in ()).throw(AssertionError("router override missing"))),
                token_dispatcher=SimpleNamespace(local_expert_indices=[0, 1]),
            )

        def _forward_mlp(self, hidden_states, inference_context=None):
            del inference_context
            probs, routing_map = self.mlp.router.forward(hidden_states)
            assert routing_map.dtype == torch.bool
            return hidden_states + probs.sum()

    compile_calls: list[object] = []

    def _fake_compile(fn):
        compile_calls.append(fn)

        def _wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return _wrapped

    hidden_states = torch.ones((2, 1, 3), dtype=torch.float32)
    attention_mask = torch.zeros((1, 1, 2, 2), dtype=torch.bool)
    layer = FakeLayer()
    original_forward = layer.mlp.router.forward
    config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="moe",
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        dtype="bf16",
        seq_len=2,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=1,
        num_experts=2,
        moe_routing_mode="equal_tokens",
        torch_compile_enabled=True,
    )

    prepared = _prepare_stage_callable(
        config=config,
        layer=layer,
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        compile_fn=_fake_compile,
    )

    output_tensor = prepared.run()
    state = prepared.consume_equal_token_routing_state()

    assert torch.allclose(output_tensor, hidden_states + 2.0)
    assert compile_calls
    assert layer.mlp.router.forward is original_forward
    assert state is not None
    assert state.tokens_per_expert == [1, 1]
    assert state.local_tokens_per_expert == [1, 1]


def test_prepare_stage_callable_compile_failure_classifies_as_torch_compile_failed():
    class FakeLayer:
        def __init__(self):
            self.config = SimpleNamespace(moe_router_topk=1)
            self.mlp = SimpleNamespace(
                router=SimpleNamespace(forward=lambda input_tensor: input_tensor),
                token_dispatcher=SimpleNamespace(local_expert_indices=[0]),
            )

        def _forward_attention(self, hidden_states, attention_mask):
            del attention_mask
            return hidden_states, None

    def _raise_compile(_fn):
        raise RuntimeError("compile blew up")

    config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="attn",
        runtime_backend="mps_only",
        attention_backend="auto",
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        overlap_moe_expert_parallel_comm=False,
        dtype="bf16",
        seq_len=2,
        batch_size=1,
        seed=1234,
        expert_model_parallel_size=1,
        num_experts=None,
        torch_compile_enabled=True,
    )

    with pytest.raises(TorchCompileFailure) as exc_info:
        _prepare_stage_callable(
            config=config,
            layer=FakeLayer(),
            hidden_states=torch.ones((2, 1, 3), dtype=torch.float32),
            attention_mask=torch.zeros((1, 1, 2, 2), dtype=torch.bool),
            compile_fn=_raise_compile,
        )

    status, error = classify_exception(exc_info.value)

    assert status == "runtime_error"
    assert error["code"] == "torch_compile_failed"


def test_aggregate_stage_results_marks_compiled_when_all_workers_compile():
    results = [
        _sample_worker_result("attn", 0, torch_compile_requested="on", torch_compile_status="compiled"),
        _sample_worker_result("attn", 1, torch_compile_requested="on", torch_compile_status="compiled"),
    ]

    stage = _aggregate_stage_results(
        "attn",
        results,
        expected_world_size=2,
        torch_compile_requested="on",
    )

    assert stage["status"] == "ok"
    assert stage["torch_compile"] == {"requested": "on", "status": "compiled"}


def test_aggregate_stage_results_propagates_compile_failures():
    results = [
        _sample_worker_result("attn", 0, torch_compile_requested="on", torch_compile_status="compiled"),
        _sample_worker_result(
            "attn",
            1,
            status="runtime_error",
            torch_compile_requested="on",
            torch_compile_status="compile_failed",
            error_code="torch_compile_failed",
        ),
    ]

    stage = _aggregate_stage_results(
        "attn",
        results,
        expected_world_size=2,
        torch_compile_requested="on",
    )

    assert stage["status"] == "runtime_error"
    assert stage["error"]["code"] == "torch_compile_failed"
    assert stage["torch_compile"] == {"requested": "on", "status": "compile_failed"}


def test_aggregate_stage_results_rejects_mixed_successful_compile_statuses():
    results = [
        _sample_worker_result("attn", 0, torch_compile_requested="on", torch_compile_status="compiled"),
        _sample_worker_result("attn", 1, torch_compile_requested="on", torch_compile_status="eager"),
    ]

    stage = _aggregate_stage_results(
        "attn",
        results,
        expected_world_size=2,
        torch_compile_requested="on",
    )

    assert stage["status"] == "runtime_error"
    assert stage["error"]["code"] == "torch_compile_status_mismatch"
    assert stage["torch_compile"] == {"requested": "on", "status": "eager"}


def test_run_case_attempt_prefers_torch_compile_failed_error(monkeypatch):
    def _fake_launch_workers(*, stage_specs, common_config, timeout_s, mps_env):
        del stage_specs, common_config, timeout_s, mps_env
        return {
            "status": "ok",
            "schedule_timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "results": [
                {
                    **_sample_worker_result("attn", 0, status="runtime_error", error_code="runtime_error"),
                    "error": {"code": "runtime_error", "message": "Schedule aborted", "traceback": None},
                },
                _sample_worker_result(
                    "moe",
                    0,
                    status="runtime_error",
                    torch_compile_requested="on",
                    torch_compile_status="compile_failed",
                    error_code="torch_compile_failed",
                ),
            ],
        }

    monkeypatch.setattr(
        "examples.attn_moe_overlap.step7_megatron_ep_overlap._launch_workers",
        _fake_launch_workers,
    )

    result = _run_case_attempt(
        mode="serial",
        common_config={
            "model_name": "Qwen/Qwen3-30B-A3B",
            "model_type": "qwen3_moe",
            "runtime_backend": "mps_only",
            "green_ctx_attn_sms": None,
            "green_ctx_moe_sms": None,
            "dtype": "bf16",
            "seq_len": 16384,
            "batch_size": 1,
            "seed": 1234,
            "warmup_iters": 100,
            "timed_iters": 100,
            "moe_ep_size": 1,
            "num_experts": 128,
            "moe_routing_mode": "equal_tokens",
            "attn_mps_active_thread_pct": None,
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "overlap_moe_expert_parallel_comm": False,
            "attention_backend": "auto",
            "nccl_tuple": None,
            "torch_compile": "on",
        },
        attn_gpu_ids=[0],
        moe_gpu_ids=[0],
        timeout_s=10.0,
        mps_env={},
    )

    assert result["status"] == "runtime_error"
    assert result["error"]["code"] == "torch_compile_failed"
