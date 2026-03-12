"""CPU-only tests for step7 Megatron EP overlap schema utilities."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.megatron_overlap_schema import (
    REQUIRED_STATUS_KEYS,
    build_case_id,
    build_case_payload,
    build_invalid_environment_payload,
    build_matrix_summary,
    compute_speedup,
    evaluate_stage_diff,
    parse_nccl_tuples,
    should_retry,
    should_skip_existing,
    validate_case_payload,
    validate_matrix_summary,
)
from examples.attn_moe_overlap.megatron_layer_runtime import (
    RuntimeConfig,
    _resolve_profiler_schedule,
    _run_iteration_schedule,
)
from examples.attn_moe_overlap.step7_megatron_ep_overlap import (
    _collapse_timed_window_s,
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
    }


def _sample_case_payload(case_id: str, mode: str, status: str = "ok") -> dict:
    return build_case_payload(
        case_id=case_id,
        status=status,
        mode=mode,
        seq_len=512,
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
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


def test_select_torch_profiler_cases_prefers_passing_serial_and_overlap():
    serial = _sample_case_payload("case-serial", "serial", status="ok")
    overlap = _sample_case_payload("case-overlap", "overlap", status="ok")
    failed_overlap = _sample_case_payload("case-overlap-bad", "overlap", status="runtime_error")
    selected = _select_torch_profiler_cases([failed_overlap, serial, overlap])
    assert [row["case_id"] for row in selected] == ["case-serial", "case-overlap"]


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
                    "timing_ms": {"cuda": 1.25, "step_total": 1.5, "timed_wall": 400.0},
                    "timed_window_s": {"start_s": 10.1, "end_s": 10.5, "duration_ms": 400.0},
                    "schedule_timed_window_s": {"start_s": 10.0, "end_s": 11.0, "duration_ms": 1000.0},
                    "enqueue_windows": [(1.0, 1.1)],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 1.0, "mean": 0.1, "std": 0.2, "max_abs": 1.5},
                },
                {
                    "role": "moe",
                    "rank": 0,
                    "status": "ok",
                    "failure_origin": False,
                    "attention_backend": "auto",
                    "timing_ms": {"cuda": 2.5, "step_total": 3.0, "timed_wall": 700.0},
                    "timed_window_s": {"start_s": 10.2, "end_s": 10.9, "duration_ms": 700.0},
                    "schedule_timed_window_s": {"start_s": 10.0, "end_s": 11.0, "duration_ms": 1000.0},
                    "enqueue_windows": [(1.2, 1.4)],
                    "finite": {"all_finite": True, "first_nonfinite": None},
                    "output_signature": {"sum": 2.0, "mean": 0.2, "std": 0.3, "max_abs": 2.5},
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
            "dtype": "bf16",
            "seq_len": 1024,
            "batch_size": 1,
            "seed": 1234,
            "warmup_iters": 1,
            "timed_iters": 3,
            "moe_ep_size": 4,
            "num_experts": None,
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
    assert result["timing_ms"]["timed_wall"] == pytest.approx(1000.0)
    assert result["timing_ms"]["attn"] == pytest.approx(1.25)
    assert result["timing_ms"]["moe"] == pytest.approx(2.5)
    assert result["overlap_ms"] == pytest.approx(0.0)


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
        torch_profiler_wait_iters=11,
        torch_profiler_active_iters=2,
    )
    serial = _sample_case_payload("case-serial", "serial", status="ok")
    overlap = _sample_case_payload("case-overlap", "overlap", status="ok")

    status = _run_torch_profiler_capture(
        args=args,
        cases=[serial, overlap],
        attn_gpu_ids=[0, 1, 2, 3],
        moe_gpu_ids=[0, 1, 2, 3],
    )

    assert status == "ok"
    assert len(calls) == 2
    assert calls[0][0] == sys.executable
    assert calls[0][1] == str((REPO_ROOT / "examples/attn_moe_overlap/step7_megatron_ep_overlap.py").resolve())
    assert "-m" not in calls[0]
    assert calls[0][calls[0].index("--attention-backend") + 1] == "fused"
    assert calls[0][calls[0].index("--torch-profiler-wait-iters") + 1] == "11"

    trace_index = json.loads((tmp_path / "out" / "torch_profiler" / "trace_index.json").read_text())
    assert trace_index["status"] == "ok"
    assert all(entry["trace_files"] == [f"{entry['trace_dir']}/worker0.pt.trace.json"] for entry in trace_index["entries"])


def test_case_id_is_deterministic_and_sensitive_to_nccl_tuple():
    kwargs = {
        "mode": "serial",
        "seq_len": 512,
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
    assert case_id_a == case_id_b
    assert case_id_a != case_id_c


def test_case_payload_validation_requires_contract_keys():
    payload = _sample_case_payload("case-a", "serial")
    assert validate_case_payload(payload) == []
    del payload["timing_ms"]
    errors = validate_case_payload(payload)
    assert any("timing_ms" in error for error in errors)


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


def test_runtime_config_uses_auto_attention_backend_by_default():
    runtime_config = RuntimeConfig(
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        stage_role="attn",
        attention_backend="auto",
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


def test_invalid_environment_payload_contract():
    payload = build_invalid_environment_payload(
        case_id="case-invalid",
        mode="serial",
        seq_len=512,
        dtype="bf16",
        seed=1234,
        topology=_sample_topology(),
        nccl_env=_sample_nccl(),
        message="CUDA is not available",
    )
    assert payload["status"] == "invalid_environment"
    assert payload["error"]["code"] == "invalid_environment"


def test_parse_nccl_tuples_normalization_and_mixed_arg_rejection():
    tuples = parse_nccl_tuples(
        nccl_tuples="4,16,32;8,32,64;4,16,32",
        nccl_socket_nthreads=None,
        nccl_max_nchannels=None,
        nccl_max_ctas=None,
    )
    assert tuples == [(4, 16, 32), (8, 32, 64)]
    with pytest.raises(ValueError):
        parse_nccl_tuples(
            nccl_tuples="4,16,32",
            nccl_socket_nthreads=4,
            nccl_max_nchannels=None,
            nccl_max_ctas=None,
        )


def test_matrix_summary_contract_and_status_count_invariant():
    serial = _sample_case_payload("case-serial", "serial", status="ok")
    overlap = _sample_case_payload("case-overlap", "overlap", status="oom")
    overlap["attempt_count"] = 2
    summary = build_matrix_summary(
        run_config={"model_name": "Qwen/Qwen3-30B-A3B", "model_type": "qwen3_moe"},
        cases=[serial, overlap],
        total_points=2,
    )
    assert validate_matrix_summary(summary) == []
    by_status = summary["counts"]["by_status"]
    assert by_status["ok"] == 1
    assert by_status["oom"] == 1
    assert sum(by_status.values()) == summary["counts"]["completed_cases"]
    for key in REQUIRED_STATUS_KEYS:
        assert key in by_status


def test_resume_skip_behavior_honors_rerun_flag():
    payload = _sample_case_payload("case-existing", "serial", status="runtime_error")
    assert should_skip_existing(payload, rerun_existing=False) is True
    assert should_skip_existing(payload, rerun_existing=True) is False
