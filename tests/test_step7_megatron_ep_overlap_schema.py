"""CPU-only tests for step7 Megatron EP overlap schema utilities."""

import sys
from pathlib import Path

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
from examples.attn_moe_overlap.step7_megatron_ep_overlap import _collapse_timed_window_s

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
