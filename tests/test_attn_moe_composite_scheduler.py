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
    blocks = default_block_specs()
    assert validate_schedule(blocks) == []
    assert len(blocks) == 8
    assert sum(block.group == "attention" for block in blocks) == 3
    assert sum(block.group == "moe" for block in blocks) == 5
    assert sum(block.group == "attention" for block in blocks) <= 10
    assert sum(block.group == "moe" for block in blocks) <= 10
    assert [block.name.split("_", maxsplit=1)[0] for block in blocks] == [
        "A0",
        "A1",
        "A2",
        "M0",
        "M1",
        "M2",
        "M3",
        "M4",
    ]


def test_default_schedule_dependencies_are_intended_dag() -> None:
    blocks = {block.name: block for block in default_block_specs()}
    assert blocks["A1_attention_probs"].depends_on == ("A0_attention_scores",)
    assert blocks["A2_attention_output"].depends_on == ("A1_attention_probs",)
    assert blocks["M1_topk_dispatch"].depends_on == ("M0_router_probs",)
    assert blocks["M3_expert_output"].depends_on == ("M2_expert_hidden",)
    assert blocks["M4_moe_output"].depends_on == ("M1_topk_dispatch", "M3_expert_output")


def test_validation_rejects_duplicate_block_names() -> None:
    errors = validate_schedule(
        [
            BlockSpec(name="A0_attention_scores", group="attention", kind="attention_scores"),
            BlockSpec(name="A0_attention_scores", group="attention", kind="attention_scores"),
        ]
    )
    assert {error.code for error in errors} == {"duplicate_block"}
    assert errors[0].to_json() == {
        "code": "duplicate_block",
        "message": "Block name 'A0_attention_scores' appears 2 times",
        "block": "A0_attention_scores",
        "dependency": None,
    }


def test_validation_rejects_unknown_dependency() -> None:
    blocks = [
        BlockSpec(
            name="A0_attention_scores",
            group="attention",
            kind="attention_scores",
            depends_on=("missing",),
        )
    ]
    assert _error_codes(blocks) == {"unknown_dependency"}


def test_validation_rejects_self_dependency() -> None:
    blocks = [
        BlockSpec(
            name="A0_attention_scores",
            group="attention",
            kind="attention_scores",
            depends_on=("A0_attention_scores",),
        )
    ]
    assert _error_codes(blocks) == {"self_dependency"}


def test_validation_rejects_two_block_cycle() -> None:
    errors = validate_schedule(
        [
            BlockSpec(
                name="A0_attention_scores",
                group="attention",
                kind="attention_scores",
                depends_on=("M0_router_probs",),
            ),
            BlockSpec(
                name="M0_router_probs",
                group="moe",
                kind="router_probs",
                depends_on=("A0_attention_scores",),
            ),
        ]
    )
    assert {error.code for error in errors} == {"cycle"}


def test_validation_rejects_three_block_cycle() -> None:
    errors = validate_schedule(
        [
            BlockSpec(
                name="A0_attention_scores",
                group="attention",
                kind="attention_scores",
                depends_on=("M1_topk_dispatch",),
            ),
            BlockSpec(
                name="M1_topk_dispatch",
                group="moe",
                kind="topk_dispatch",
                depends_on=("M2_expert_hidden",),
            ),
            BlockSpec(
                name="M2_expert_hidden",
                group="moe",
                kind="expert_hidden",
                depends_on=("A0_attention_scores",),
            ),
        ]
    )
    assert {error.code for error in errors} == {"cycle"}


def test_validation_rejects_unsupported_kind_and_group() -> None:
    errors = validate_schedule(
        [
            BlockSpec(name="bad_kind", group="attention", kind="not_a_block"),
            BlockSpec(name="bad_group", group="bad", kind="attention_scores"),
        ]
    )
    assert {"unsupported_block_kind", "unsupported_block_group"} <= {error.code for error in errors}


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
    assert len(payload["blocks"]) == 8
    assert {block["group"] for block in payload["blocks"]} == {"attention", "moe"}
    assert all({"inputs", "outputs", "group", "kind"} <= set(block) for block in payload["blocks"])
    assert {block["stream_role"] for block in payload["blocks"]} == {"default"}
    assert all(block["finite"] for block in payload["blocks"])
    assert all(math.isfinite(block["checksum"]) for block in payload["blocks"])
    assert payload["summary"]["schedule_valid"] is True
    assert payload["summary"]["all_finite"] is True
    assert math.isfinite(payload["summary"]["combined_checksum"])
    assert set(payload["summary"]["final_checksums"]) == {"attention_output", "moe_output"}
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
    assert len(payload["blocks"]) == 8


def test_value_lifetimes_include_intermediates_and_outputs() -> None:
    payload = run_composite_schedule(
        schedule="serial",
        device="cpu",
        shape=_tiny_shape(),
        warmup_iters=0,
        timed_iters=1,
        seed=123,
    )
    lifetimes = {value["name"]: value for value in payload["summary"]["value_lifetimes"]}

    assert lifetimes["attention_scores"]["produced_by"] == "A0_attention_scores"
    assert lifetimes["attention_scores"]["consumed_by"] == ["A1_attention_probs"]
    assert lifetimes["dispatch"]["produced_by"] == "M1_topk_dispatch"
    assert lifetimes["dispatch"]["consumed_by"] == ["M4_moe_output"]
    assert lifetimes["attention_output"]["produced_by"] == "A2_attention_output"
    assert lifetimes["attention_output"]["consumed_by"] == []
    assert lifetimes["moe_output"]["produced_by"] == "M4_moe_output"
    assert lifetimes["moe_output"]["consumed_by"] == []
    assert lifetimes["attention_input"]["persistent"] is True
    assert lifetimes["router"]["input"] is True
    assert lifetimes["attention_scores"]["shape"] == [1, 8, 8]
    assert lifetimes["dispatch"]["dtype"] == "torch.float32"


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
    assert serial["summary"]["final_checksums"] == stream["summary"]["final_checksums"]
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
            BlockSpec(name="A0_attention_scores", group="attention", kind="attention_scores"),
            BlockSpec(name="A0_attention_scores", group="attention", kind="attention_scores"),
        ],
    )

    assert payload["schema_version"] == "v1"
    assert payload["blocks"] == []
    assert payload["summary"]["schedule_valid"] is False
    assert payload["summary"]["all_finite"] is None
    assert payload["summary"]["combined_checksum"] is None
    assert payload["summary"]["final_checksums"] is None
    assert payload["summary"]["value_lifetimes"] == []
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
    assert len(payload["blocks"]) == 8
