"""CPU-only tests for the Step-8 attention/MoE memory sweep helpers."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.step8_attn_moe_memory_sweep import (  # noqa: E402
    BenchmarkConfig,
    bytes_to_units,
    canonical_matrix,
    expected_row_keys,
    parse_int_csv,
    parse_modes,
    parse_modules,
    render_results_markdown,
    write_csv,
    write_json,
)

pytestmark = [pytest.mark.cpu_only]


def _config() -> BenchmarkConfig:
    return BenchmarkConfig(
        modules=("attention", "moe"),
        modes=("forward", "forward_backward"),
        seq_lens=(1024, 2048, 4096, 8192, 16384, 32768),
        batch_sizes=(1, 2),
        dtype_name="bf16",
        device="cuda",
        warmup_iters=1,
        timed_iters=1,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        num_experts=8,
        top_k=2,
        seed=1234,
    )


def _metadata() -> dict:
    config = _config()
    return {
        "command": "python step8_attn_moe_memory_sweep.py --modules all",
        "commit_sha": "abc123",
        "multimodal_training_commit_sha": "def456",
        "branch": "test",
        "gpu_type": "H100",
        "gpu_name": "NVIDIA H100 80GB HBM3",
        "gpu_count": 8,
        "cuda_version": "12.8",
        "cuda_driver_version": "580.126.09",
        "pytorch_version": "2.x",
        "python_version": "3.11",
        "config": {
            "warmup_iters": config.warmup_iters,
            "timed_iters": config.timed_iters,
            "dtype": config.dtype_name,
            "hidden_size": config.hidden_size,
            "num_heads": config.num_heads,
            "intermediate_size": config.intermediate_size,
            "num_experts": config.num_experts,
            "top_k": config.top_k,
        },
    }


def _row(module: str, mode: str, seq_len: int, batch_size: int, status: str = "ok") -> dict:
    return {
        "module": module,
        "mode": mode,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "status": status,
        "status_reason": "" if status == "ok" else "synthetic failure",
        "elapsed_ms_mean": 1.25,
        "tokens_per_second": float(seq_len * batch_size) / 0.00125,
        "max_memory_allocated_bytes": 1073741824,
        "max_memory_allocated_mib": 1024.0,
        "max_memory_allocated_gib": 1.0,
        "max_memory_reserved_bytes": 2147483648,
        "max_memory_reserved_mib": 2048.0,
        "max_memory_reserved_gib": 2.0,
    }


def test_parse_helpers():
    assert parse_int_csv("1024,2048", field_name="seq-lens") == (1024, 2048)
    assert parse_modules("all") == ("attention", "moe")
    assert parse_modules("attn,moe,attention") == ("attention", "moe")
    assert parse_modes("forward+backward,fwd") == ("forward_backward", "forward")

    with pytest.raises(ValueError):
        parse_int_csv("0", field_name="batch-sizes")
    with pytest.raises(ValueError):
        parse_modules("overlap")


def test_canonical_matrix_resolves_issue_grid_as_12_cells():
    config = _config()
    assert len(canonical_matrix(config.seq_lens, config.batch_sizes)) == 12
    assert len(expected_row_keys(config)) == 48
    assert ("attention", "forward", 32768, 2) in expected_row_keys(config)
    assert ("moe", "forward_backward", 32768, 2) in expected_row_keys(config)


def test_bytes_to_units_reports_raw_mib_and_gib():
    converted = bytes_to_units(1073741824)
    assert converted["bytes"] == 1073741824
    assert converted["mib"] == 1024.0
    assert converted["gib"] == 1.0


def test_render_results_markdown_separates_modules_modes_and_notes_12_cells():
    rows = [
        _row(module, mode, seq_len, batch_size)
        for module, mode, seq_len, batch_size in expected_row_keys(_config())
    ]
    text = render_results_markdown(
        metadata=_metadata(),
        rows=rows,
        csv_path=Path("todo/docs/slug/verification/results.csv"),
        json_path=Path("todo/docs/slug/verification/results.json"),
    )

    assert "6 sequence lengths x 2 batch sizes = 12 cells" in text
    assert "## Attention forward" in text
    assert "## Attention forward+backward" in text
    assert "## Moe forward" in text
    assert "## Moe forward+backward" in text
    assert "All cells completed with `status=ok`." in text


def test_write_machine_readable_artifacts(tmp_path: Path):
    rows = [
        _row("attention", "forward", 1024, 1),
        _row("moe", "forward_backward", 32768, 2, status="oom"),
    ]
    csv_path = tmp_path / "rows.csv"
    json_path = tmp_path / "rows.json"

    write_csv(csv_path, rows)
    write_json(json_path, _metadata(), rows)

    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["max_memory_allocated_bytes"] == "1073741824"
    assert csv_rows[1]["status"] == "oom"

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["status_counts"] == {"ok": 1, "oom": 1}
    assert payload["rows"][1]["status_reason"] == "synthetic failure"
