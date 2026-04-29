"""CPU-only tests for the Attention/MoE composite scheduler sweep helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.composite_scheduler_sweep import (  # noqa: E402
    DEFAULT_NCU_METRICS,
    MEMORY_FIELDNAMES,
    SweepConfig,
    _add_forward_profile_to_row,
    build_ncu_command,
    build_ncu_preflight_command,
    canonical_matrix,
    case_id,
    parse_command_prefix,
    parse_int_csv,
    parse_metric_csv,
    summarize_ncu_csv,
)

pytestmark = [pytest.mark.cpu_only]


def _sweep_config(tmp_path: Path) -> SweepConfig:
    return SweepConfig(
        batch_sizes=(1, 2, 4),
        seq_lens=(1024, 2048),
        schedule="stream",
        device="cuda",
        hidden=1024,
        num_experts=8,
        top_k=2,
        warmup_iters=0,
        timed_iters=1,
        seed=0,
        profile_forward_blocks=True,
        ncu_metrics=DEFAULT_NCU_METRICS,
        ncu_timeout_sec=30,
        ncu_path="ncu",
        ncu_prefix=(),
        cuda_visible_devices="0",
        output_dir=tmp_path,
        command="python composite_scheduler_sweep.py",
    )


def test_parse_int_csv_rejects_empty_and_non_positive_values() -> None:
    assert parse_int_csv("1, 2,4", field_name="batch_sizes") == (1, 2, 4)
    with pytest.raises(ValueError, match="must not be empty"):
        parse_int_csv(" ,, ", field_name="batch_sizes")
    with pytest.raises(ValueError, match="positive"):
        parse_int_csv("1,0", field_name="batch_sizes")


def test_parse_metric_csv_rejects_empty_values() -> None:
    assert parse_metric_csv("a,b, c") == ("a", "b", "c")
    with pytest.raises(ValueError, match="must not be empty"):
        parse_metric_csv(" ,, ")


def test_canonical_matrix_is_batch_major() -> None:
    assert canonical_matrix(seq_lens=(1024, 2048), batch_sizes=(1, 2)) == [
        (1024, 1),
        (2048, 1),
        (1024, 2),
        (2048, 2),
    ]
    assert case_id(batch_size=4, seq_len=32768) == "batch4_seq32768"


def test_build_ncu_command_profiles_child_sweep_case(tmp_path: Path) -> None:
    config = _sweep_config(tmp_path)
    command = build_ncu_command(
        config=config,
        batch_size=2,
        seq_len=4096,
        raw_csv_path=tmp_path / "raw.csv",
        child_json_path=tmp_path / "child.json",
    )

    assert command[0] == "ncu"
    assert "--nvtx" in command
    assert "--print-nvtx-rename" in command
    assert "--ncu-profile-child" in command
    assert "--metrics" in command
    assert ",".join(DEFAULT_NCU_METRICS) in command
    assert command[command.index("--batch") + 1] == "2"
    assert command[command.index("--seq-len") + 1] == "4096"
    assert command[command.index("--child-json-output") + 1] == str(tmp_path / "child.json")


def test_ncu_prefix_is_tokenized_and_applied_to_profile_and_preflight(tmp_path: Path) -> None:
    config = SweepConfig(
        batch_sizes=(1,),
        seq_lens=(1024,),
        schedule="stream",
        device="cuda",
        hidden=1024,
        num_experts=8,
        top_k=2,
        warmup_iters=0,
        timed_iters=1,
        seed=0,
        profile_forward_blocks=True,
        ncu_metrics=DEFAULT_NCU_METRICS,
        ncu_timeout_sec=30,
        ncu_path="/usr/local/cuda/bin/ncu",
        ncu_prefix=parse_command_prefix("sudo -n -E"),
        cuda_visible_devices="0",
        output_dir=tmp_path,
        command="python composite_scheduler_sweep.py",
    )

    profile_command = build_ncu_command(
        config=config,
        batch_size=1,
        seq_len=1024,
        raw_csv_path=tmp_path / "raw.csv",
        child_json_path=tmp_path / "child.json",
    )
    preflight_command = build_ncu_preflight_command(config=config)

    assert profile_command[:4] == ["sudo", "-n", "-E", "/usr/local/cuda/bin/ncu"]
    assert preflight_command[:4] == ["sudo", "-n", "-E", "/usr/local/cuda/bin/ncu"]
    assert "--query-metrics-mode" in preflight_command


def test_forward_profile_fields_include_wall_time_and_memory_values() -> None:
    assert "forward_block_A0_attention_scores_wall_ms" in MEMORY_FIELDNAMES
    assert "forward_block_M4_moe_output_wall_ms" in MEMORY_FIELDNAMES

    row = {field: None for field in MEMORY_FIELDNAMES}
    payload = {
        "summary": {"wall_clock_ms": 12.5},
        "blocks": [
            {
                "name": "A0_attention_scores",
                "profile": {
                    "wall_ms": 1.25,
                    "peak_allocated_bytes": 2 * 1024**2,
                    "peak_reserved_bytes": 3 * 1024**2,
                },
            },
            {
                "name": "M4_moe_output",
                "profile": {
                    "wall_ms": 2.5,
                    "peak_allocated_bytes": 4 * 1024**2,
                    "peak_reserved_bytes": 5 * 1024**2,
                },
            },
        ],
    }

    _add_forward_profile_to_row(row, payload)

    assert row["forward_profile_wall_clock_ms"] == 12.5
    assert row["forward_block_A0_attention_scores_wall_ms"] == 1.25
    assert row["forward_block_A0_attention_scores_peak_allocated_bytes"] == 2 * 1024**2
    assert row["forward_block_A0_attention_scores_peak_allocated_mib"] == 2.0
    assert row["forward_block_A0_attention_scores_peak_reserved_bytes"] == 3 * 1024**2
    assert row["forward_block_M4_moe_output_wall_ms"] == 2.5
    assert row["forward_block_M4_moe_output_peak_allocated_bytes"] == 4 * 1024**2
    assert row["forward_block_M4_moe_output_peak_allocated_mib"] == 4.0
    assert row["forward_block_M4_moe_output_peak_reserved_bytes"] == 5 * 1024**2


def test_summarize_ncu_csv_groups_metrics_by_nvtx_renamed_block(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process",
                '"ID","Kernel Name","Metric Name","Metric Value"',
                '"0","attn_moe_block::A0_attention_scores","gpu__time_duration.sum","100"',
                (
                    '"0","attn_moe_block::A0_attention_scores",'
                    '"sm__throughput.avg.pct_of_peak_sustained_elapsed","75.0"'
                ),
                (
                    '"0","attn_moe_block::A0_attention_scores",'
                    '"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed","50.0"'
                ),
                '"1","attn_moe_block::M2_expert_hidden","gpu__time_duration.sum","200"',
                ('"1","attn_moe_block::M2_expert_hidden",' '"sm__throughput.avg.pct_of_peak_sustained_elapsed","80.0"'),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries, parse_error = summarize_ncu_csv(csv_path)

    assert parse_error is None
    by_block = {row["block"]: row for row in summaries}
    assert by_block["A0_attention_scores"]["metric_count"] == 3
    assert by_block["A0_attention_scores"]["gpu_time_duration_ns_sum"] == 100.0
    assert by_block["A0_attention_scores"]["sm_throughput_pct_mean"] == 75.0
    assert by_block["A0_attention_scores"]["sm_cycles_active_pct_mean"] == 50.0
    assert by_block["M2_expert_hidden"]["metric_count"] == 2


def test_summarize_ncu_csv_reads_metric_name_average_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu_average.csv"
    csv_path.write_text(
        "\n".join(
            [
                '"ID","Kernel Name","Metric Name","Invocations","Average","Minimum","Maximum"',
                '"0","attn_moe_block::A1_attention_probs","gpu__time_duration.sum","3","10","9","11"',
                (
                    '"0","attn_moe_block::A1_attention_probs",'
                    '"sm__throughput.avg.pct_of_peak_sustained_elapsed","3","72.5","70","75"'
                ),
                (
                    '"0","attn_moe_block::A1_attention_probs",'
                    '"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed","3","61.5","60","63"'
                ),
                '"1","attn_moe_block::M4_moe_output","gpu__time_duration.sum","2","20","19","21"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries, parse_error = summarize_ncu_csv(csv_path)

    assert parse_error is None
    by_block = {row["block"]: row for row in summaries}
    assert by_block["A1_attention_probs"]["gpu_time_duration_ns_sum"] == 30.0
    assert by_block["A1_attention_probs"]["sm_throughput_pct_mean"] == 72.5
    assert by_block["A1_attention_probs"]["sm_cycles_active_pct_mean"] == 61.5
    assert by_block["M4_moe_output"]["gpu_time_duration_ns_sum"] == 40.0


def test_summarize_ncu_csv_reads_wide_metric_columns_and_skips_units(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu_wide.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected to process",
                (
                    '"ID","Kernel Name","gpu__time_duration.sum",'
                    '"sm__throughput.avg.pct_of_peak_sustained_elapsed",'
                    '"smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"'
                ),
                '"","","nsecond","%","%"',
                '"0","attn_moe_block::A2_attention_output","120","70.0","55.0"',
                '"1","attn_moe_block::M3_expert_output","240","82.0","65.0"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summaries, parse_error = summarize_ncu_csv(csv_path)

    assert parse_error is None
    by_block = {row["block"]: row for row in summaries}
    assert set(by_block) == {"A2_attention_output", "M3_expert_output"}
    assert by_block["A2_attention_output"]["metric_count"] == 3
    assert by_block["A2_attention_output"]["gpu_time_duration_ns_sum"] == 120.0
    assert by_block["A2_attention_output"]["sm_throughput_pct_mean"] == 70.0
    assert by_block["A2_attention_output"]["sm_cycles_active_pct_mean"] == 55.0
    assert by_block["M3_expert_output"]["gpu_time_duration_ns_sum"] == 240.0


def test_summarize_ncu_csv_reports_missing_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text("==ERROR== ERR_NVGPUCTRPERM\n", encoding="utf-8")

    summaries, parse_error = summarize_ncu_csv(csv_path)

    assert summaries == []
    assert parse_error == "ncu_csv_header_missing"
