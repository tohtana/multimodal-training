"""CPU-only tests for the Attention/MoE composite scheduler prototype."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.composite_scheduler import (  # noqa: E402
    BlockSpec,
    ShapeConfig,
    default_block_specs,
    probe_runtime_capabilities,
    run_composite_schedule,
    validate_schedule,
)

pytestmark = [pytest.mark.cpu_only]


def _tiny_shape() -> ShapeConfig:
    return ShapeConfig(batch=1, seq_len=8, hidden=16, num_experts=2, top_k=1)


def _error_codes(blocks: list[BlockSpec]) -> set[str]:
    return {error.code for error in validate_schedule(blocks)}


def test_default_schedule_validates() -> None:
    assert validate_schedule(default_block_specs()) == []


def test_validation_rejects_duplicate_block_names() -> None:
    errors = validate_schedule(
        [
            BlockSpec(name="attention", kind="attention"),
            BlockSpec(name="attention", kind="moe"),
        ]
    )
    assert {error.code for error in errors} == {"duplicate_block"}
    assert errors[0].to_json() == {
        "code": "duplicate_block",
        "message": "Block name 'attention' appears 2 times",
        "block": "attention",
        "dependency": None,
    }


def test_validation_rejects_unknown_dependency() -> None:
    assert _error_codes([BlockSpec(name="attention", kind="attention", depends_on=("missing",))]) == {
        "unknown_dependency"
    }


def test_validation_rejects_self_dependency() -> None:
    assert _error_codes([BlockSpec(name="attention", kind="attention", depends_on=("attention",))]) == {
        "self_dependency"
    }


def test_validation_rejects_two_block_cycle() -> None:
    errors = validate_schedule(
        [
            BlockSpec(name="attention", kind="attention", depends_on=("moe",)),
            BlockSpec(name="moe", kind="moe", depends_on=("attention",)),
        ]
    )
    assert {error.code for error in errors} == {"cycle"}


def test_validation_rejects_three_block_cycle() -> None:
    errors = validate_schedule(
        [
            BlockSpec(name="attention", kind="attention", depends_on=("router",)),
            BlockSpec(name="router", kind="moe", depends_on=("moe",)),
            BlockSpec(name="moe", kind="moe", depends_on=("attention",)),
        ]
    )
    assert {error.code for error in errors} == {"cycle"}


def test_cpu_serial_json_shape() -> None:
    payload = run_composite_schedule(
        schedule="serial",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        seed=123,
    )

    assert set(payload) == {
        "schema_version",
        "schedule",
        "device",
        "shape",
        "iterations",
        "capabilities",
        "blocks",
        "summary",
    }
    assert payload["schema_version"] == "v1"
    assert payload["schedule"] == "serial"
    assert payload["device"]["resolved"] == "cpu"
    assert payload["shape"] == {
        "batch": 1,
        "seq_len": 8,
        "hidden": 16,
        "num_experts": 2,
        "top_k": 1,
    }
    assert payload["iterations"] == {"warmup": 0, "timed": 1}
    assert len(payload["blocks"]) == 2
    assert {block["name"] for block in payload["blocks"]} == {"attention", "moe"}
    assert {block["stream_role"] for block in payload["blocks"]} == {"default"}
    assert all(block["finite"] for block in payload["blocks"])
    assert all(math.isfinite(block["checksum"]) for block in payload["blocks"])
    assert payload["summary"]["schedule_valid"] is True
    assert payload["summary"]["all_finite"] is True
    assert math.isfinite(payload["summary"]["combined_checksum"])
    assert payload["summary"]["peak_allocated_bytes"] is None
    assert payload["summary"]["peak_reserved_bytes"] is None
    assert payload["summary"]["validation_errors"] == []


def test_cpu_stream_fallback_uses_cpu_stream_role() -> None:
    payload = run_composite_schedule(
        schedule="stream",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        seed=123,
    )
    assert payload["schema_version"] == "v1"
    assert payload["device"]["resolved"] == "cpu"
    assert {block["stream_role"] for block in payload["blocks"]} == {"cpu"}


def test_cpu_serial_and_stream_checksums_match() -> None:
    serial = run_composite_schedule(
        schedule="serial",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        seed=456,
    )
    stream = run_composite_schedule(
        schedule="stream",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        seed=456,
    )

    serial_blocks = {block["name"]: block for block in serial["blocks"]}
    stream_blocks = {block["name"]: block for block in stream["blocks"]}
    assert serial["summary"]["combined_checksum"] == stream["summary"]["combined_checksum"]
    for name, serial_block in serial_blocks.items():
        assert serial_block["checksum"] == stream_blocks[name]["checksum"]


def test_invalid_schedule_returns_stable_json_shape() -> None:
    payload = run_composite_schedule(
        schedule="serial",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        blocks=[
            BlockSpec(name="attention", kind="attention"),
            BlockSpec(name="attention", kind="moe"),
        ],
    )

    assert payload["schema_version"] == "v1"
    assert payload["blocks"] == []
    assert payload["summary"]["schedule_valid"] is False
    assert payload["summary"]["all_finite"] is None
    assert payload["summary"]["combined_checksum"] is None
    assert payload["summary"]["peak_allocated_bytes"] is None
    assert payload["summary"]["peak_reserved_bytes"] is None
    assert payload["shape"] == _tiny_shape().to_json()
    assert payload["iterations"] == {"warmup": 0, "timed": 1}
    assert payload["capabilities"]["cuda"]["requested"] == "cpu"
    [error] = payload["summary"]["validation_errors"]
    assert set(error) == {"code", "message", "block", "dependency"}
    assert error["code"] == "duplicate_block"


def test_capability_probe_is_structured_without_gpu_requirement() -> None:
    capabilities = probe_runtime_capabilities("cpu")
    assert set(capabilities) == {"torch", "cuda", "green_context"}
    assert set(capabilities["torch"]) == {"version", "cuda"}
    assert {
        "available",
        "requested",
        "used",
        "device_count",
        "device_index",
        "device_name",
        "device_capability",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    } <= set(capabilities["cuda"])
    assert capabilities["cuda"]["requested"] == "cpu"
    assert capabilities["cuda"]["used"] is False
    assert isinstance(capabilities["cuda"]["available"], bool)
    assert isinstance(capabilities["green_context"]["visible"], bool)
    assert isinstance(capabilities["green_context"]["usable"], bool)
    assert "reason" in capabilities["green_context"]


def test_cli_emits_parseable_cpu_json() -> None:
    cmd = [
        sys.executable,
        "-m",
        "examples.attn_moe_overlap.composite_scheduler",
        "--schedule",
        "serial",
        "--device",
        "cpu",
        "--batch",
        "1",
        "--seq-len",
        "8",
        "--hidden",
        "16",
        "--num-experts",
        "2",
        "--top-k",
        "1",
        "--warmup-iters",
        "0",
        "--timed-iters",
        "1",
    ]
    completed = subprocess.run(cmd, cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    assert payload["schema_version"] == "v1"
    assert payload["summary"]["schedule_valid"] is True
    assert len(payload["blocks"]) == 2
