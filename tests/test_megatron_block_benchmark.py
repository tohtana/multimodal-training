"""CPU-only tests for the Megatron block benchmark helpers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap import megatron_block_benchmark as benchmark  # noqa: E402
from examples.attn_moe_overlap.megatron_block_benchmark import (  # noqa: E402
    BLOCK_NAMES,
    CaseDescriptor,
    build_cases,
    build_ncu_child_env,
    build_ncu_preflight_command,
    build_ncu_profile_command,
    parse_command_prefix,
    render_pivot_table,
    summarize_ncu_csv,
)
from examples.attn_moe_overlap.megatron_layer_runtime import (  # noqa: E402
    MegatronSingleLayerRuntime,
    RuntimeConfig,
)

pytestmark = [pytest.mark.cpu_only]


def test_build_cases_preserves_requested_matrix_and_blocks() -> None:
    cases = build_cases(seq_lens=[1024, 2048], batch_sizes=[1, 2, 4])

    assert len(cases) == 12
    assert cases[0] == CaseDescriptor(stage_role="attn", seq_len=1024, batch_size=1)
    assert cases[1] == CaseDescriptor(stage_role="moe", seq_len=1024, batch_size=1)
    assert cases[0].blocks == benchmark.BLOCKS_BY_STAGE_ROLE["attn"]
    assert cases[1].blocks == benchmark.BLOCKS_BY_STAGE_ROLE["moe"]
    assert set(cases[0].blocks + cases[1].blocks) == set(BLOCK_NAMES)


def test_ncu_preflight_command_and_prefix_are_noninteractive() -> None:
    prefix = parse_command_prefix("sudo -n -E")
    command = build_ncu_preflight_command(
        "/usr/local/cuda/bin/ncu",
        prefix,
        ["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
    )

    assert command[:4] == ["sudo", "-n", "-E", "/usr/local/cuda/bin/ncu"]
    assert "--query-metrics-mode" in command


def test_ncu_profile_child_command_includes_required_output_dir(tmp_path: Path) -> None:
    args = SimpleNamespace(
        ncu_path="/usr/local/cuda/bin/ncu",
        _ncu_metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        profile_warmup_iters=5,
        profile_iters=1,
        worker_timeout_s=60.0,
        model_name="Qwen/Qwen3-30B-A3B",
        model_type="qwen3_moe",
        dtype="bf16",
        attention_backend="auto",
        moe_token_dispatcher_type="alltoall",
        moe_routing_mode="equal_tokens",
        num_experts=128,
        output_dir=tmp_path / "verification",
        ncu_raw_dir=tmp_path / "ncu_raw",
    )
    child_json_path = tmp_path / "child.json"
    command = build_ncu_profile_command(
        args=args,
        descriptor=CaseDescriptor(stage_role="attn", seq_len=1024, batch_size=1),
        gpu_ids=[0],
        prefix=("sudo", "-n", "-E"),
        raw_csv_path=tmp_path / "raw.csv",
        child_json_path=child_json_path,
    )

    env_index = command.index("env")
    output_dir_index = command.index("--output-dir")
    child_json_index = command.index("--child-json-output")
    assert command[env_index + 1] == f"RAY_TRAIN_LOG_FILE={args.ncu_raw_dir / 'ray_train_ncu.log'}"
    assert command[output_dir_index + 1] == str(args.output_dir)
    assert command[child_json_index + 1] == str(child_json_path)
    assert output_dir_index < child_json_index


def test_ncu_child_env_redirects_ray_train_log(tmp_path: Path) -> None:
    args = SimpleNamespace(ncu_raw_dir=tmp_path / "ncu_raw")

    env = build_ncu_child_env(args)

    assert env["RAY_TRAIN_LOG_FILE"] == str(args.ncu_raw_dir / "ray_train_ncu.log")


def test_gpu_worker_launch_uses_spawn_and_rank_local_devices() -> None:
    context = benchmark._multiprocessing_context()
    env, selected_device = benchmark._worker_cuda_env(
        gpu_ids=[0, 2, 5],
        rank=1,
        master_addr="127.0.0.1",
        master_port=29599,
    )

    assert context.get_start_method() == "spawn"
    assert env["CUDA_VISIBLE_DEVICES"] == "0,2,5"
    assert env["LOCAL_RANK"] == "1"
    assert env["RANK"] == "1"
    assert env["WORLD_SIZE"] == "3"
    assert env["MASTER_PORT"] == "29599"
    assert selected_device == 1


def test_megatron_runtime_uses_local_rank_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCAL_RANK", "3")

    runtime = MegatronSingleLayerRuntime(
        RuntimeConfig(
            model_name="Qwen/Qwen3-30B-A3B",
            model_type="qwen3_moe",
            stage_role="moe",
            runtime_backend="mps_only",
            attention_backend="auto",
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            overlap_moe_expert_parallel_comm=False,
            dtype="bf16",
            seq_len=1024,
            batch_size=1,
            seed=1234,
            expert_model_parallel_size=8,
            num_experts=128,
            moe_routing_mode="equal_tokens",
        )
    )

    assert str(runtime.device) == "cuda:3"


def test_resolve_ncu_prefix_falls_back_when_permission_probe_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_probe(command: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
        del timeout_s
        if command[0] == "ncu":
            return subprocess.CompletedProcess(command, 0, "", "ERR_NVGPUCTRPERM")
        return subprocess.CompletedProcess(command, 0, "metrics ok", "")

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        del kwargs
        assert command == ["sudo", "-n", "true"]
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(benchmark, "_run_ncu_probe", fake_probe)
    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)
    args = SimpleNamespace(
        _ncu_metrics=["sm__throughput.avg.pct_of_peak_sustained_elapsed"],
        ncu_timeout_sec=5,
        ncu_path="ncu",
        ncu_prefix="",
    )

    prefix, error = benchmark.resolve_ncu_prefix(args, tmp_path / "ncu_probe.log")

    assert prefix == ("sudo", "-n", "-E")
    assert error is None


def test_summarize_ncu_csv_groups_by_megatron_nvtx_block(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected",
                '"ID","Kernel Name","Metric Name","Metric Value"',
                '"1","attn_moe_block::A0_qkv/foo","sm__throughput.avg.pct_of_peak_sustained_elapsed","75.0"',
                '"1","","smsp__cycles_active.avg.pct_of_peak_sustained_elapsed","55.5"',
                '"1","","gpu__time_duration.sum","1000"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    summaries, error = summarize_ncu_csv(csv_path)
    assert error is None
    assert summaries == [
        {
            "block": "A0_qkv",
            "metric_count": 3,
            "gpu_time_duration_ns_sum": 1000.0,
            "sm_throughput_pct_mean": 75.0,
            "sm_active_pct_mean": 55.5,
        }
    ]


def test_summarize_ncu_csv_prefers_range_column_for_library_kernels(tmp_path: Path) -> None:
    csv_path = tmp_path / "ncu.csv"
    csv_path.write_text(
        "\n".join(
            [
                "==PROF== Connected",
                '"Domain","Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg","Kernel Name","Invocations","Metric Name","Average"',
                '"<default domain>"," ""attn_moe_block::A3_output_projection:none:none:none:none:none:none""  ""nvte_cublas_gemm_v2:none:none:none:none:none:none"" ","nvte_cublas_gemm_v2/nvjet_tst","1","device__attribute_architecture","384"',
                '"<default domain>"," ""attn_moe_block::A3_output_projection:none:none:none:none:none:none""  ""nvte_cublas_gemm_v2:none:none:none:none:none:none"" ","nvte_cublas_gemm_v2/nvjet_tst","1","sm__throughput.avg.pct_of_peak_sustained_elapsed","42.25"',
                '"<default domain>"," ""attn_moe_block::A3_output_projection:none:none:none:none:none:none""  ""nvte_cublas_gemm_v2:none:none:none:none:none:none"" ","nvte_cublas_gemm_v2/nvjet_tst","1","smsp__cycles_active.avg.pct_of_peak_sustained_elapsed","61.5"',
                '"<default domain>"," ""attn_moe_block::A3_output_projection:none:none:none:none:none:none""  ""nvte_cublas_gemm_v2:none:none:none:none:none:none"" ","nvte_cublas_gemm_v2/nvjet_tst","2","gpu__time_duration.sum","1000"',
                "",
            ]
        ),
        encoding="utf-8",
    )

    summaries, error = summarize_ncu_csv(csv_path)

    assert error is None
    assert summaries == [
        {
            "block": "A3_output_projection",
            "metric_count": 3,
            "gpu_time_duration_ns_sum": 2000.0,
            "sm_throughput_pct_mean": 42.25,
            "sm_active_pct_mean": 61.5,
        }
    ]


def test_render_pivot_table_keeps_failed_cells_visible() -> None:
    text = render_pivot_table(
        title="Per-Block SM Throughput %",
        rows=[
            {
                "batch_size": 1,
                "seq_len": 1024,
                "block": "A0_qkv",
                "status": "ok",
                "sm_throughput_pct_mean": 80.125,
            },
            {
                "batch_size": 1,
                "seq_len": 1024,
                "block": "M0_router",
                "status": "oom",
                "sm_throughput_pct_mean": None,
            },
        ],
        field="sm_throughput_pct_mean",
        batch_sizes=[1],
        seq_lens=[1024],
        digits=2,
    )

    assert "| 1 | 1024 | 80.12 | not_run | not_run | not_run | oom |" in text
