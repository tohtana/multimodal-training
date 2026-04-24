"""CPU-only tests for the Step-8 module GPU sweep schema helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.module_gpu_sweep_schema import (
    CASE_SCHEMA_VERSION,
    SUMMARY_SCHEMA_VERSION,
    VALID_STATUS_KEYS,
    build_case_id,
    build_case_payload,
    build_module_summary,
    build_status_counts,
    normalize_dtype_name,
    normalize_stage_role,
    parse_batch_sizes,
    parse_dtypes,
    parse_gpu_ids,
    parse_seq_lens,
    relative_case_path,
    validate_case_payload,
    validate_module_summary,
    write_case_json,
)

pytestmark = [pytest.mark.cpu_only]


def _sample_case_payload() -> dict:
    return build_case_payload(
        case_id=build_case_id(
            stage_role="attn",
            seq_len=1024,
            batch_size=1,
            dtype="bf16",
            seed=1234,
            world_size=8,
        ),
        attempt_id="20260423T000000Z",
        module="attn",
        gpu_type="H100",
        seq_len=1024,
        batch_size=1,
        dtype="bf16",
        warmup_iters=100,
        timed_iters=100,
        seed=1234,
        gpu_ids=list(range(8)),
        world_size=8,
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        status="ok",
        status_reason=None,
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_routing_mode="equal_tokens",
        num_experts=128,
        forward_pass_ms=12.5,
        module_timing_ms=11.75,
        tokens_per_iter=1024,
        tokens_per_second=81920.0,
        timing_ms={
            "timed_wall_total": 1250.0,
            "timed_wall_per_iter": 12.5,
            "module_cuda": 11.75,
            "module_step_total": 12.0,
        },
        runtime={
            "detected_gpu_name": "NVIDIA H100 80GB HBM3",
            "device_names_by_rank": ["NVIDIA H100 80GB HBM3"] * 8,
            "device_capabilities_by_rank": [[9, 0]] * 8,
            "runtime_by_rank": [{}] * 8,
        },
        error={"code": None, "message": None, "traceback": None},
        pricing_status="ok",
        per_gpu_hourly_cost_usd=2.69,
        node_hourly_cost_usd=21.52,
        cost_per_token_usd=7.291666e-08,
        runpod_model="H100 SXM",
        superproject_commit="abc123",
        multimodal_training_commit="def456",
        artifact_path=relative_case_path(
            build_case_id(
                stage_role="attn",
                seq_len=1024,
                batch_size=1,
                dtype="bf16",
                seed=1234,
                world_size=8,
            )
        ),
    )


def test_parse_helpers_validate_and_dedupe_values():
    assert parse_seq_lens("1024,2048") == [1024, 2048]
    assert parse_batch_sizes("1,2") == [1, 2]
    assert parse_gpu_ids("0,1,2,3,4,5,6,7") == list(range(8))
    assert parse_dtypes("bf16,bfloat16,fp16") == ["bf16", "fp16"]

    with pytest.raises(ValueError):
        parse_seq_lens("0")
    with pytest.raises(ValueError):
        parse_gpu_ids("0,0")


def test_normalizers_accept_expected_values():
    assert normalize_stage_role("ATTN") == "attn"
    assert normalize_stage_role("moe") == "moe"
    assert normalize_dtype_name("bfloat16") == "bf16"
    assert normalize_dtype_name("FP16") == "fp16"


def test_build_case_payload_validates_against_schema():
    payload = _sample_case_payload()
    assert payload["schema_version"] == CASE_SCHEMA_VERSION
    assert validate_case_payload(payload) == []

    broken = dict(payload)
    broken["artifact_path"] = "/tmp/abs.json"
    errors = validate_case_payload(broken)
    assert any("artifact_path must be output-dir-relative" in error for error in errors)


def test_write_case_json_materializes_relative_artifact_path(tmp_path: Path):
    payload = _sample_case_payload()
    written = write_case_json(tmp_path, payload)
    materialized = json.loads(written.read_text(encoding="utf-8"))

    assert written == tmp_path / "cases" / f"{payload['case_id']}.json"
    assert materialized["artifact_path"] == relative_case_path(payload["case_id"])
    assert validate_case_payload(materialized) == []


def test_build_module_summary_requires_relative_case_paths():
    case = _sample_case_payload()
    summary = build_module_summary(
        stage_role="attn",
        attempt_id="20260423T000000Z",
        command=["python", "step8_module_gpu_sweep.py"],
        gpu_type="H100",
        detected_gpu_name="NVIDIA H100 80GB HBM3",
        superproject_commit="abc123",
        multimodal_training_commit="def456",
        case_paths=[relative_case_path(case["case_id"])],
        status_counts=build_status_counts([case]),
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        gpu_ids=list(range(8)),
        seq_lens=[1024],
        batch_sizes=[1],
        dtypes=["bf16"],
        warmup_iters=100,
        timed_iters=100,
        attention_backend="auto",
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        moe_routing_mode="equal_tokens",
        num_experts=128,
    )

    assert summary["schema_version"] == SUMMARY_SCHEMA_VERSION
    assert validate_module_summary(summary) == []

    broken = dict(summary)
    broken["case_paths"] = ["/tmp/abs.json"]
    errors = validate_module_summary(broken)
    assert any("case_paths must be output-dir-relative" in error for error in errors)


def test_build_status_counts_covers_all_known_statuses():
    counts = build_status_counts(
        [
            _sample_case_payload(),
            {
                **_sample_case_payload(),
                "case_id": "other",
                "status": "timeout",
                "artifact_path": relative_case_path("other"),
            },
        ]
    )
    assert set(VALID_STATUS_KEYS).issubset(counts)
    assert counts["ok"] == 1
    assert counts["timeout"] == 1
