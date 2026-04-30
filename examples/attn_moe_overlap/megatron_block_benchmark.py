"""Measure real Megatron attention/MoE stage blocks for Qwen3-30B-A3B.

The workload in this script is deliberately limited to
``MegatronSingleLayerRuntime`` calls.  It does not import or execute the
synthetic composite scheduler prototypes.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import platform
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch


def _early_bootstrap_local_pythonpath() -> list[str]:
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "multimodal-training",
        project_root / "Megatron-LM",
        project_root / "ms-swift",
    ]
    added_paths: list[str] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate_text = str(candidate)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
            added_paths.append(candidate_text)

    if added_paths:
        existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = ":".join([*added_paths, *([existing] if existing else [])])
    return added_paths


_early_bootstrap_local_pythonpath()

try:
    from examples.attn_moe_overlap.module_gpu_sweep_schema import (
        normalize_dtype_name,
        normalize_stage_role,
        parse_batch_sizes,
        parse_gpu_ids,
        parse_seq_lens,
    )
except ModuleNotFoundError:
    from module_gpu_sweep_schema import (  # type: ignore[no-redef]
        normalize_dtype_name,
        normalize_stage_role,
        parse_batch_sizes,
        parse_gpu_ids,
        parse_seq_lens,
    )

from megatron.core.transformer.profiling import (  # noqa: E402
    ATTENTION_BLOCK_NAMES,
    MOE_BLOCK_NAMES,
    NVTX_RANGE_PREFIX,
    get_profile_block_names,
)

SCHEMA_VERSION = "megatron_block_benchmark.v2"
DEFAULT_BATCH_SIZES = (1, 2, 4)
DEFAULT_SEQ_LENS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
)
NCU_DURATION_METRIC = "gpu__time_duration.sum"
NCU_SM_THROUGHPUT_METRIC = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
NCU_SM_ACTIVE_METRIC = "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed"
QWEN3_30B_A3B_BASELINE = {
    "model_name": "Qwen/Qwen3-30B-A3B",
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "norm_topk_prob": True,
    "decoder_sparse_step": 1,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "dtype": "bf16",
    "transformer_impl": "transformer_engine",
    "attention_backend": "auto",
    "moe_grouped_gemm": True,
    "moe_routing_mode": "equal_tokens",
    "moe_token_dispatcher_type": "alltoall",
    "seed": 1234,
}
STAGE_ROLES = ("attn", "moe")
BLOCKS_BY_STAGE_ROLE = {
    "attn": tuple(ATTENTION_BLOCK_NAMES),
    "moe": tuple(MOE_BLOCK_NAMES),
}
BLOCK_NAMES = tuple(get_profile_block_names())

MEASUREMENT_FIELDNAMES = (
    "batch_size",
    "seq_len",
    "block",
    "stage_role",
    "status",
    "status_reason",
    "warmup_iters",
    "timed_iters",
    "world_size",
    "gpu_ids",
    "forward_wall_ms",
    "forward_cuda_ms",
    "total_wall_ms",
    "total_cuda_ms",
    "peak_allocated_bytes",
    "peak_allocated_gib",
    "peak_reserved_bytes",
    "peak_reserved_gib",
    "allocated_before_bytes",
    "reserved_before_bytes",
    "allocated_after_bytes",
    "reserved_after_bytes",
    "tokens_per_iter",
    "tokens_per_second",
    "device_names",
    "error_type",
    "error_message",
    "started_at_utc",
    "completed_at_utc",
)
UTILIZATION_FIELDNAMES = (
    "batch_size",
    "seq_len",
    "block",
    "stage_role",
    "status",
    "status_reason",
    "warmup_iters",
    "profile_iters",
    "world_size",
    "gpu_ids",
    "sm_throughput_pct_mean",
    "sm_active_pct_mean",
    "gpu_time_duration_ns_sum",
    "metric_count",
    "profiler",
    "ncu_raw_csv",
    "child_json",
    "command",
    "error_type",
    "error_message",
    "started_at_utc",
    "completed_at_utc",
)


@dataclass(frozen=True)
class CaseDescriptor:
    stage_role: str
    seq_len: int
    batch_size: int

    @property
    def blocks(self) -> tuple[str, ...]:
        return BLOCKS_BY_STAGE_ROLE[self.stage_role]

    @property
    def case_id(self) -> str:
        return "__".join(
            (
                f"stage-{self.stage_role}",
                f"seq-{self.seq_len}",
                f"batch-{self.batch_size}",
            )
        )


def _bootstrap_local_pythonpath() -> list[str]:
    return _early_bootstrap_local_pythonpath()


def now_tag() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def parse_command_prefix(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        if not raw.strip():
            return ()
        return tuple(shlex.split(raw))
    return tuple(str(part) for part in raw)


def parse_metric_csv(raw: str) -> tuple[str, ...]:
    metrics = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not metrics:
        raise ValueError("ncu metrics must not be empty")
    return metrics


def bytes_to_gib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0**3)


def _repo_commit(repo_root: Path, subpath: str | None = None) -> str | None:
    target = repo_root if subpath is None else repo_root / subpath
    try:
        result = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def build_cases(seq_lens: Sequence[int], batch_sizes: Sequence[int]) -> list[CaseDescriptor]:
    return [
        CaseDescriptor(stage_role=role, seq_len=int(seq_len), batch_size=int(batch_size))
        for batch_size in batch_sizes
        for seq_len in seq_lens
        for role in STAGE_ROLES
    ]


def _worker_result_path(worker_result_dir: Path, rank: int) -> Path:
    return worker_result_dir / f"rank{rank}.json"


def _worker_cuda_env(
    *,
    gpu_ids: Sequence[int],
    rank: int,
    master_addr: str,
    master_port: int,
) -> tuple[dict[str, str], int]:
    """Return a spawn-safe CUDA/distributed environment for one worker rank."""

    if not gpu_ids:
        raise ValueError("gpu_ids must not be empty")
    if rank < 0 or rank >= len(gpu_ids):
        raise ValueError(f"rank {rank} is outside gpu_ids size {len(gpu_ids)}")
    visible_devices = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    local_rank = int(rank)
    return (
        {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "LOCAL_RANK": str(local_rank),
            "RANK": str(rank),
            "WORLD_SIZE": str(len(gpu_ids)),
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": str(master_port),
            "NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
        },
        local_rank,
    )


def _multiprocessing_context() -> mp.context.BaseContext:
    return mp.get_context("spawn")


def _load_worker_results(worker_result_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(worker_result_dir.glob("rank*.json"))
    ]


def _worker_main(
    *,
    stage_role: str,
    rank: int,
    world_size: int,
    gpu_ids: Sequence[int],
    local_rank: int,
    master_addr: str,
    master_port: int,
    model_name: str,
    model_type: str,
    dtype: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    warmup_iters: int,
    timed_iters: int,
    attention_backend: str,
    moe_grouped_gemm: bool,
    moe_token_dispatcher_type: str,
    moe_routing_mode: str,
    num_experts: int,
    include_backward: bool,
    enable_cuda_profiler: bool,
    worker_result_dir: str,
) -> None:
    try:
        from examples.attn_moe_overlap.megatron_layer_runtime import (
            MegatronSingleLayerRuntime,
            RuntimeConfig,
            classify_exception,
            cleanup_distributed_state,
        )
    except ModuleNotFoundError:
        from megatron_layer_runtime import (  # type: ignore[no-redef]
            MegatronSingleLayerRuntime,
            RuntimeConfig,
            classify_exception,
            cleanup_distributed_state,
        )

    status = "ok"
    failure_origin = False
    payload: dict[str, Any] = {}
    runtime: Any | None = None
    device_name: str | None = None
    device_capability: list[int] | None = None
    try:
        _bootstrap_local_pythonpath()
        worker_env, selected_device = _worker_cuda_env(
            gpu_ids=gpu_ids,
            rank=rank,
            master_addr=master_addr,
            master_port=master_port,
        )
        if int(world_size) != int(worker_env["WORLD_SIZE"]):
            raise RuntimeError(
                f"world_size mismatch: launcher={world_size} env={worker_env['WORLD_SIZE']}"
            )
        if int(local_rank) != int(selected_device):
            raise RuntimeError(f"local_rank mismatch: launcher={local_rank} env={selected_device}")
        os.environ.update(worker_env)
        for env_key in (
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "NCCL_SOCKET_NTHREADS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_MAX_CTAS",
        ):
            os.environ.pop(env_key, None)

        torch.cuda.set_device(selected_device)
        device_name = str(torch.cuda.get_device_name(selected_device))
        device_capability = [
            int(value) for value in torch.cuda.get_device_capability(selected_device)
        ]
        runtime = MegatronSingleLayerRuntime(
            RuntimeConfig(
                model_name=model_name,
                model_type=model_type,
                stage_role=stage_role,
                runtime_backend="mps_only",
                attention_backend=attention_backend,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_token_dispatcher_type=moe_token_dispatcher_type,
                overlap_moe_expert_parallel_comm=False,
                dtype=dtype,
                seq_len=seq_len,
                batch_size=batch_size,
                seed=seed,
                expert_model_parallel_size=world_size if stage_role == "moe" else 1,
                stage_label=stage_role,
                num_experts=num_experts,
                moe_routing_mode=moe_routing_mode,
            )
        )
        runtime.initialize()
        layer_mapping = runtime.describe_layer_mapping()
        measurement = runtime.run_stage_measurement(
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            include_backward=include_backward,
            enable_cuda_profiler=enable_cuda_profiler,
        )
        status = str(measurement.get("status", "runtime_error"))
        payload = {
            "status": status,
            "device_name": device_name,
            "device_capability": device_capability,
            "layer_mapping": layer_mapping,
            **measurement,
        }
    except BaseException as exc:  # noqa: BLE001 - every matrix cell must become a status row.
        status, error = classify_exception(exc)
        failure_origin = True
        payload = {
            "status": status,
            "device_name": device_name,
            "device_capability": device_capability,
            "error": error,
            "traceback": getattr(error, "traceback", None),
        }
    finally:
        if runtime is not None:
            try:
                runtime.cleanup()
            except Exception:
                pass
        cleanup_distributed_state()
        result = {
            "rank": int(rank),
            "status": status,
            "failure_origin": failure_origin,
            **payload,
        }
        path = _worker_result_path(Path(worker_result_dir), rank)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _select_root_cause(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    priorities = (
        lambda item: item.get("failure_origin") is True and item.get("status") == "oom",
        lambda item: item.get("failure_origin") is True and item.get("status") == "runtime_error",
        lambda item: item.get("status") == "oom",
        lambda item: item.get("status") == "timeout",
        lambda item: item.get("status") == "runtime_error",
    )
    for predicate in priorities:
        for result in results:
            if predicate(result):
                return result
    return None


def _launch_workers(
    *,
    descriptor: CaseDescriptor,
    gpu_ids: list[int],
    timeout_s: float,
    common_config: dict[str, Any],
    include_backward: bool,
    enable_cuda_profiler: bool,
) -> dict[str, Any]:
    _bootstrap_local_pythonpath()
    worker_result_dir = Path(tempfile.mkdtemp(prefix="megatron_block_results_"))
    processes: list[mp.Process] = []
    master_port = _find_free_port()
    expected = len(gpu_ids)
    try:
        context = _multiprocessing_context()
        for rank, _gpu_id in enumerate(gpu_ids):
            process = context.Process(
                target=_worker_main,
                kwargs={
                    "stage_role": descriptor.stage_role,
                    "rank": rank,
                    "world_size": expected,
                    "gpu_ids": tuple(gpu_ids),
                    "local_rank": rank,
                    "master_addr": "127.0.0.1",
                    "master_port": master_port,
                    "seq_len": descriptor.seq_len,
                    "batch_size": descriptor.batch_size,
                    "include_backward": include_backward,
                    "enable_cuda_profiler": enable_cuda_profiler,
                    "worker_result_dir": str(worker_result_dir),
                    **common_config,
                },
            )
            process.start()
            processes.append(process)

        deadline = time.time() + float(timeout_s)
        while time.time() < deadline:
            results = _load_worker_results(worker_result_dir)
            if len(results) >= expected:
                break
            if processes and all(not process.is_alive() for process in processes):
                break
            time.sleep(0.2)

        results = _load_worker_results(worker_result_dir)
        timed_out = len(results) < expected and any(process.is_alive() for process in processes)
        if timed_out:
            for process in processes:
                if process.is_alive():
                    process.terminate()

        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()

        results = _load_worker_results(worker_result_dir)
        if len(results) < expected:
            exitcodes = {str(process.pid): process.exitcode for process in processes}
            return {
                "status": "timeout" if timed_out else "runtime_error",
                "status_reason": f"missing worker results ({len(results)}/{expected}); exitcodes={exitcodes}",
                "results": results,
            }
        failing = _select_root_cause(results)
        if failing is not None:
            error = failing.get("error") or {}
            return {
                "status": str(failing.get("status") or "runtime_error"),
                "status_reason": error.get("message") or str(failing.get("status")),
                "results": results,
            }
        return {"status": "ok", "status_reason": None, "results": results}
    finally:
        for path in worker_result_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        worker_result_dir.rmdir()


def _max_nested(results: list[dict[str, Any]], *keys: str) -> Any:
    values: list[float] = []
    for result in results:
        current: Any = result
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            values.append(float(current))
    return max(values) if values else None


def _first_nested(results: list[dict[str, Any]], *keys: str) -> Any:
    for result in results:
        current: Any = result
        for key in keys:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            return current
    return None


def _block_stat_values(results: list[dict[str, Any]], block: str, key: str) -> list[float]:
    values: list[float] = []
    for result in results:
        block_stats = (result.get("block_stats") or {}).get(block) or {}
        value = block_stats.get(key)
        if value is not None:
            values.append(float(value))
    return values


def _max_block_stat(results: list[dict[str, Any]], block: str, key: str) -> float | None:
    values = _block_stat_values(results, block, key)
    return max(values) if values else None


def _observed_block_count(results: list[dict[str, Any]], block: str) -> int:
    observed = 0
    for result in results:
        block_stats = (result.get("block_stats") or {}).get(block) or {}
        if int(block_stats.get("count") or 0) > 0:
            observed += 1
    return observed


def aggregate_measurement_rows(
    *,
    descriptor: CaseDescriptor,
    launch_result: dict[str, Any],
    gpu_ids: list[int],
    warmup_iters: int,
    timed_iters: int,
    started_at_utc: str,
    completed_at_utc: str,
) -> list[dict[str, Any]]:
    results = launch_result.get("results") or []
    status = str(launch_result.get("status") or "runtime_error")
    reason = launch_result.get("status_reason")
    memory_after = [((result.get("memory") or {}).get("after") or {}) for result in results]
    peak_allocated = max((item.get("max_allocated") for item in memory_after if item.get("max_allocated") is not None), default=None)
    peak_reserved = max((item.get("max_reserved") for item in memory_after if item.get("max_reserved") is not None), default=None)
    allocated_after = max((item.get("allocated") for item in memory_after if item.get("allocated") is not None), default=None)
    reserved_after = max((item.get("reserved") for item in memory_after if item.get("reserved") is not None), default=None)
    memory_before = [((result.get("memory") or {}).get("before") or {}) for result in results]
    allocated_before = max((item.get("allocated") for item in memory_before if item.get("allocated") is not None), default=None)
    reserved_before = max((item.get("reserved") for item in memory_before if item.get("reserved") is not None), default=None)
    forward_wall = _max_nested(results, "timing_ms", "forward_wall_mean")
    tokens_per_iter = int(descriptor.seq_len * descriptor.batch_size)
    failing = _select_root_cause(results)
    error = (failing or {}).get("error") or {}
    device_names = [result.get("device_name") for result in results if result.get("device_name")]
    rows: list[dict[str, Any]] = []
    for block in descriptor.blocks:
        observed_count = _observed_block_count(results, block)
        block_status = status
        block_reason = None if status == "ok" else reason
        block_forward_wall = None
        block_forward_cuda = None
        block_peak_allocated = None
        block_peak_reserved = None
        block_tokens_per_second = None
        if status == "ok":
            if observed_count == 0:
                block_status = "not_observed"
                block_reason = "instrumented_block_not_observed"
            else:
                block_forward_wall = _max_block_stat(results, block, "forward_wall_ms_mean")
                block_forward_cuda = _max_block_stat(results, block, "forward_cuda_ms_mean")
                block_peak_allocated = _max_block_stat(results, block, "peak_allocated_bytes")
                block_peak_reserved = _max_block_stat(results, block, "peak_reserved_bytes")
                if block_forward_wall is not None and block_forward_wall > 0:
                    block_tokens_per_second = tokens_per_iter / (block_forward_wall / 1000.0)
        rows.append(
            {
                "batch_size": descriptor.batch_size,
                "seq_len": descriptor.seq_len,
                "block": block,
                "stage_role": descriptor.stage_role,
                "status": block_status,
                "status_reason": block_reason,
                "warmup_iters": int(warmup_iters),
                "timed_iters": int(timed_iters),
                "world_size": len(gpu_ids),
                "gpu_ids": ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "forward_wall_ms": block_forward_wall,
                "forward_cuda_ms": block_forward_cuda,
                "total_wall_ms": _max_nested(results, "timing_ms", "total_wall_mean"),
                "total_cuda_ms": _max_nested(results, "timing_ms", "total_cuda_mean"),
                "peak_allocated_bytes": block_peak_allocated,
                "peak_allocated_gib": bytes_to_gib(
                    None if block_peak_allocated is None else int(block_peak_allocated)
                ),
                "peak_reserved_bytes": block_peak_reserved,
                "peak_reserved_gib": bytes_to_gib(
                    None if block_peak_reserved is None else int(block_peak_reserved)
                ),
                "allocated_before_bytes": allocated_before,
                "reserved_before_bytes": reserved_before,
                "allocated_after_bytes": allocated_after,
                "reserved_after_bytes": reserved_after,
                "tokens_per_iter": tokens_per_iter,
                "tokens_per_second": block_tokens_per_second,
                "device_names": ";".join(device_names),
                "observed_rank_count": observed_count,
                "stage_forward_wall_ms": forward_wall,
                "stage_peak_allocated_bytes": peak_allocated,
                "stage_peak_reserved_bytes": peak_reserved,
                "error_type": error.get("code"),
                "error_message": error.get("message"),
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*fieldnames, *extras], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in [*fieldnames, *extras]})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build_metadata(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    return {
        "schema_version": SCHEMA_VERSION,
        "command": format_command(sys.argv),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "superproject_commit": _repo_commit(project_root),
        "multimodal_training_commit": _repo_commit(project_root, "multimodal-training"),
        "megatron_lm_commit": _repo_commit(project_root, "Megatron-LM"),
        "baseline": dict(QWEN3_30B_A3B_BASELINE),
        "nvtx_prefix": NVTX_RANGE_PREFIX,
        "block_names": list(BLOCK_NAMES),
        "blocks_by_stage_role": {
            stage_role: list(blocks) for stage_role, blocks in BLOCKS_BY_STAGE_ROLE.items()
        },
        "matrix": {
            "batch_sizes": list(args._batch_sizes),
            "seq_lens": list(args._seq_lens),
        },
        "gpu_ids": {
            "attn": list(args._attn_gpu_ids),
            "moe": list(args._moe_gpu_ids),
        },
        "iterations": {
            "measurement_warmup": int(args.warmup_iters),
            "measurement_timed": int(args.timed_iters),
            "profile_warmup": int(args.profile_warmup_iters),
            "profile_iters": int(args.profile_iters),
        },
        "ncu_subset": {
            "batch_sizes": list(args._ncu_batch_sizes),
            "seq_lens": list(args._ncu_seq_lens),
        },
    }


def _payload_base(metadata: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "metadata": metadata,
        "rows": list(rows),
        "status_counts": build_status_counts(rows),
    }


def build_status_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def run_measurement_sweep(args: argparse.Namespace, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "megatron_block_measurement.csv"
    json_path = output_dir / "megatron_block_measurement.json"
    mapping_path = output_dir / "megatron_layer_mapping.json"
    mapping_by_role: dict[str, Any] = {}
    common_config = _common_runtime_config(args)
    for descriptor in build_cases(args._seq_lens, args._batch_sizes):
        started_at = now_utc_iso()
        gpu_ids = args._attn_gpu_ids if descriptor.stage_role == "attn" else args._moe_gpu_ids
        print(f"[measure] {descriptor.case_id} warmup={args.warmup_iters} timed={args.timed_iters}", flush=True)
        launch_result = _launch_workers(
            descriptor=descriptor,
            gpu_ids=gpu_ids,
            timeout_s=float(args.worker_timeout_s),
            common_config=common_config,
            include_backward=True,
            enable_cuda_profiler=False,
        )
        completed_at = now_utc_iso()
        rows.extend(
            aggregate_measurement_rows(
                descriptor=descriptor,
                launch_result=launch_result,
                gpu_ids=gpu_ids,
                warmup_iters=int(args.warmup_iters),
                timed_iters=int(args.timed_iters),
                started_at_utc=started_at,
                completed_at_utc=completed_at,
            )
        )
        mapping = _first_nested(launch_result.get("results") or [], "layer_mapping")
        if mapping is not None:
            mapping_by_role[descriptor.stage_role] = mapping
            write_json(
                mapping_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "baseline": QWEN3_30B_A3B_BASELINE,
                    "mapping_by_role": mapping_by_role,
                    "note": (
                        "Fine block ranges are emitted from Megatron-LM internals. "
                        "The bridge benchmark only invokes MegatronSingleLayerRuntime stages."
                    ),
                },
            )
        write_csv(csv_path, rows, MEASUREMENT_FIELDNAMES)
        write_json(json_path, _payload_base(metadata, rows))
    return rows


def _common_runtime_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "dtype": args.dtype,
        "seed": int(args.seed),
        "warmup_iters": int(args.warmup_iters),
        "timed_iters": int(args.timed_iters),
        "attention_backend": args.attention_backend,
        "moe_grouped_gemm": bool(args.moe_grouped_gemm),
        "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
        "moe_routing_mode": args.moe_routing_mode,
        "num_experts": int(args.num_experts),
    }


def _float_or_none(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _csv_columns(line: str) -> list[str]:
    try:
        return next(csv.reader([line]))
    except csv.Error:
        return []


def _find_ncu_csv_header(lines: Sequence[str], metrics: Sequence[str]) -> int | None:
    for index, line in enumerate(lines):
        columns = set(_csv_columns(line))
        if "Metric Name" in columns and ({"Metric Value", "Average"} & columns):
            return index
        if any(metric in columns for metric in metrics) and ({"Kernel Name", "Name"} & columns):
            return index
    return None


def _ncu_kernel_name(row: dict[str, Any]) -> str:
    return str(row.get("Kernel Name") or row.get("Name") or "")


def _infer_block_from_text(text: str) -> str:
    if NVTX_RANGE_PREFIX in text:
        tail = text.split(NVTX_RANGE_PREFIX, maxsplit=1)[1]
        return tail.split(":", maxsplit=1)[0].split("/", maxsplit=1)[0].strip() or "unattributed"
    for block in BLOCK_NAMES:
        if block in text:
            return block
    return "unattributed"


def _ncu_range_text(row: dict[str, Any]) -> str:
    return " ".join(str(value or "") for key, value in row.items() if key.startswith("Range"))


def _infer_block_from_ncu_row(row: dict[str, Any], last_block: str) -> str:
    # Nsight Compute can keep the NVTX range in the Range column while leaving
    # the kernel name as the library kernel, for example TransformerEngine GEMMs.
    for source_text in (_ncu_range_text(row), _ncu_kernel_name(row)):
        block = _infer_block_from_text(source_text)
        if block != "unattributed":
            return block
    return last_block or "unattributed"


def _long_ncu_metric_value(row: dict[str, Any], metric_name: str) -> float | None:
    value = _float_or_none(row.get("Metric Value"))
    if value is not None:
        return value
    average = _float_or_none(row.get("Average"))
    if average is None:
        return None
    if metric_name == NCU_DURATION_METRIC:
        invocations = _float_or_none(row.get("Invocations"))
        if invocations is not None:
            return average * invocations
    return average


def summarize_ncu_csv(csv_path: Path, metrics: Sequence[str] = DEFAULT_NCU_METRICS) -> tuple[list[dict[str, Any]], str | None]:
    if not csv_path.exists():
        return [], "ncu_csv_missing"
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = _find_ncu_csv_header(lines, metrics)
    if header_index is None:
        return [], "ncu_csv_header_missing"
    grouped: dict[str, dict[str, list[float]]] = {}
    metric_rows = 0
    last_block = ""
    requested_metrics = set(metrics)
    reader = csv.DictReader(lines[header_index:])
    fieldnames = set(reader.fieldnames or [])
    wide_metrics = tuple(metric for metric in metrics if metric in fieldnames)
    for row in reader:
        block = _infer_block_from_ncu_row(row, last_block)
        if block != "unattributed":
            last_block = block
        metric_name = row.get("Metric Name")
        if metric_name is not None:
            metric_name = str(metric_name)
            if metric_name not in requested_metrics:
                continue
            value = _long_ncu_metric_value(row, metric_name)
            if value is None:
                continue
            grouped.setdefault(block, {}).setdefault(metric_name, []).append(value)
            metric_rows += 1
            continue
        for metric in wide_metrics:
            value = _float_or_none(row.get(metric))
            if value is None:
                continue
            grouped.setdefault(block, {}).setdefault(metric, []).append(value)
            metric_rows += 1
    if metric_rows == 0:
        return [], "ncu_csv_no_metric_rows"
    summaries: list[dict[str, Any]] = []
    for block, block_metrics in sorted(grouped.items()):
        duration = block_metrics.get(NCU_DURATION_METRIC, [])
        throughput = block_metrics.get(NCU_SM_THROUGHPUT_METRIC, [])
        active = block_metrics.get(NCU_SM_ACTIVE_METRIC, [])
        summaries.append(
            {
                "block": block,
                "metric_count": sum(len(values) for values in block_metrics.values()),
                "gpu_time_duration_ns_sum": sum(duration) if duration else None,
                "sm_throughput_pct_mean": sum(throughput) / len(throughput) if throughput else None,
                "sm_active_pct_mean": sum(active) / len(active) if active else None,
            }
        )
    return summaries, None


def _run_ncu_probe(command: list[str], timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout_s, check=False)


def _permission_denied(output: str) -> bool:
    lower = output.lower()
    return "err_nvgpuctrperm" in lower or "permission" in lower and "profil" in lower


def build_ncu_preflight_command(ncu_path: str, prefix: Sequence[str], metrics: Sequence[str]) -> list[str]:
    return [
        *prefix,
        ncu_path,
        "--query-metrics-mode",
        "all",
        "--metrics",
        ",".join(metrics),
    ]


def resolve_ncu_prefix(args: argparse.Namespace, log_path: Path) -> tuple[tuple[str, ...] | None, str | None]:
    metrics = args._ncu_metrics
    timeout_s = int(args.ncu_timeout_sec)
    attempts: list[str] = []
    ordinary = build_ncu_preflight_command(args.ncu_path, (), metrics)
    ordinary_result = _run_ncu_probe(ordinary, timeout_s)
    attempts.append(_format_probe_log("ordinary", ordinary, ordinary_result))
    ordinary_output = ordinary_result.stdout + ordinary_result.stderr
    permission_denied = _permission_denied(ordinary_output)
    if ordinary_result.returncode == 0 and not permission_denied:
        log_path.write_text("\n\n".join(attempts), encoding="utf-8")
        return (), None
    if not permission_denied:
        user_prefix = parse_command_prefix(args.ncu_prefix)
        if user_prefix:
            user_cmd = build_ncu_preflight_command(args.ncu_path, user_prefix, metrics)
            user_result = _run_ncu_probe(user_cmd, timeout_s)
            attempts.append(_format_probe_log("user-prefix", user_cmd, user_result))
            user_output = user_result.stdout + user_result.stderr
            if user_result.returncode == 0 and not _permission_denied(user_output):
                log_path.write_text("\n\n".join(attempts), encoding="utf-8")
                return user_prefix, None
            permission_denied = _permission_denied(user_output)
        if not permission_denied:
            log_path.write_text("\n\n".join(attempts), encoding="utf-8")
            return None, "ncu_preflight_failed"

    sudo_check = subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True, check=False)
    attempts.append(_format_probe_log("sudo-check", ["sudo", "-n", "true"], sudo_check))
    if sudo_check.returncode != 0:
        log_path.write_text("\n\n".join(attempts), encoding="utf-8")
        return None, "ncu_permission_denied"
    sudo_prefix = ("sudo", "-n", "-E")
    sudo_cmd = build_ncu_preflight_command(args.ncu_path, sudo_prefix, metrics)
    sudo_result = _run_ncu_probe(sudo_cmd, timeout_s)
    attempts.append(_format_probe_log("sudo-ncu", sudo_cmd, sudo_result))
    log_path.write_text("\n\n".join(attempts), encoding="utf-8")
    sudo_output = sudo_result.stdout + sudo_result.stderr
    if sudo_result.returncode == 0 and not _permission_denied(sudo_output):
        return sudo_prefix, None
    if _permission_denied(sudo_output):
        return None, "ncu_permission_denied_after_sudo"
    return None, "ncu_preflight_failed_after_sudo"


def _format_probe_log(label: str, command: Sequence[str], result: subprocess.CompletedProcess[str]) -> str:
    return "\n".join(
        [
            f"## {label}",
            "$ " + format_command(command),
            f"exit_code={result.returncode}",
            "--- stdout ---",
            result.stdout[-4000:],
            "--- stderr ---",
            result.stderr[-4000:],
        ]
    )


def build_ncu_profile_command(
    *,
    args: argparse.Namespace,
    descriptor: CaseDescriptor,
    gpu_ids: list[int],
    prefix: Sequence[str],
    raw_csv_path: Path,
    child_json_path: Path,
) -> list[str]:
    child_env_prefix = ["env", f"RAY_TRAIN_LOG_FILE={build_ncu_child_env(args)['RAY_TRAIN_LOG_FILE']}"]
    return [
        *prefix,
        *child_env_prefix,
        args.ncu_path,
        "--target-processes",
        "all",
        "--profile-from-start",
        "off",
        "--nvtx",
        "--print-nvtx-rename",
        "kernel",
        "--print-summary",
        "per-nvtx",
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--metrics",
        ",".join(args._ncu_metrics),
        "--log-file",
        str(raw_csv_path),
        "--force-overwrite",
        sys.executable,
        str(Path(__file__).resolve()),
        "--profile-child",
        "--stage-role",
        descriptor.stage_role,
        "--gpu-ids",
        ",".join(str(gpu_id) for gpu_id in gpu_ids),
        "--seq-lens",
        str(descriptor.seq_len),
        "--batch-sizes",
        str(descriptor.batch_size),
        "--warmup-iters",
        str(args.profile_warmup_iters),
        "--timed-iters",
        str(args.profile_iters),
        "--worker-timeout-s",
        str(args.worker_timeout_s),
        "--model-name",
        args.model_name,
        "--model-type",
        args.model_type,
        "--dtype",
        args.dtype,
        "--attention-backend",
        args.attention_backend,
        "--moe-token-dispatcher-type",
        args.moe_token_dispatcher_type,
        "--moe-routing-mode",
        args.moe_routing_mode,
        "--num-experts",
        str(args.num_experts),
        "--output-dir",
        str(args.output_dir),
        "--child-json-output",
        str(child_json_path),
    ]


def build_ncu_child_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("RAY_TRAIN_LOG_FILE", str(Path(args.ncu_raw_dir) / "ray_train_ncu.log"))
    return env


def _util_failure_rows(
    *,
    descriptor: CaseDescriptor,
    gpu_ids: list[int],
    status: str,
    status_reason: str,
    command: str | None,
    error_type: str | None,
    error_message: str | None,
    raw_csv_path: Path | None,
    child_json_path: Path | None,
    args: argparse.Namespace,
    started_at_utc: str,
    completed_at_utc: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for block in descriptor.blocks:
        rows.append(
            {
                "batch_size": descriptor.batch_size,
                "seq_len": descriptor.seq_len,
                "block": block,
                "stage_role": descriptor.stage_role,
                "status": status,
                "status_reason": status_reason,
                "warmup_iters": int(args.profile_warmup_iters),
                "profile_iters": int(args.profile_iters),
                "world_size": len(gpu_ids),
                "gpu_ids": ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "sm_throughput_pct_mean": None,
                "sm_active_pct_mean": None,
                "gpu_time_duration_ns_sum": None,
                "metric_count": None,
                "profiler": "ncu",
                "ncu_raw_csv": None if raw_csv_path is None else str(raw_csv_path),
                "child_json": None if child_json_path is None else str(child_json_path),
                "command": command,
                "error_type": error_type,
                "error_message": error_message,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
            }
        )
    return rows


def run_ncu_case(
    *,
    args: argparse.Namespace,
    descriptor: CaseDescriptor,
    gpu_ids: list[int],
    prefix: Sequence[str],
) -> list[dict[str, Any]]:
    started_at = now_utc_iso()
    raw_dir = Path(args.ncu_raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = raw_dir / f"{descriptor.case_id}.csv"
    child_json_path = raw_dir / f"{descriptor.case_id}.child.json"
    command = build_ncu_profile_command(
        args=args,
        descriptor=descriptor,
        gpu_ids=gpu_ids,
        prefix=prefix,
        raw_csv_path=raw_csv_path,
        child_json_path=child_json_path,
    )
    command_text = format_command(command)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=int(args.ncu_timeout_sec),
            check=False,
            env=build_ncu_child_env(args),
        )
        completed_at = now_utc_iso()
    except subprocess.TimeoutExpired as exc:
        return _util_failure_rows(
            descriptor=descriptor,
            gpu_ids=gpu_ids,
            status="timeout",
            status_reason="ncu_timeout",
            command=command_text,
            error_type="TimeoutExpired",
            error_message=str(exc),
            raw_csv_path=raw_csv_path,
            child_json_path=child_json_path,
            args=args,
            started_at_utc=started_at,
            completed_at_utc=now_utc_iso(),
        )

    child_payload = _load_json_if_exists(child_json_path)
    child_status = str((child_payload or {}).get("status") or "")
    child_reason = (child_payload or {}).get("status_reason")
    if child_status and child_status != "ok":
        return _util_failure_rows(
            descriptor=descriptor,
            gpu_ids=gpu_ids,
            status=child_status,
            status_reason=str(child_reason or child_status),
            command=command_text,
            error_type=((child_payload or {}).get("error") or {}).get("code"),
            error_message=((child_payload or {}).get("error") or {}).get("message"),
            raw_csv_path=raw_csv_path,
            child_json_path=child_json_path,
            args=args,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
        )

    if completed.returncode != 0:
        message = "\n".join((completed.stderr or completed.stdout or "").splitlines()[:8])
        return _util_failure_rows(
            descriptor=descriptor,
            gpu_ids=gpu_ids,
            status="profiler_failed",
            status_reason="ncu_failed",
            command=command_text,
            error_type="ncu_failed",
            error_message=message,
            raw_csv_path=raw_csv_path,
            child_json_path=child_json_path,
            args=args,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
        )

    summaries, parse_error = summarize_ncu_csv(raw_csv_path, args._ncu_metrics)
    if parse_error is not None:
        return _util_failure_rows(
            descriptor=descriptor,
            gpu_ids=gpu_ids,
            status="parse_failed",
            status_reason=parse_error,
            command=command_text,
            error_type=parse_error,
            error_message=parse_error,
            raw_csv_path=raw_csv_path,
            child_json_path=child_json_path,
            args=args,
            started_at_utc=started_at,
            completed_at_utc=completed_at,
        )
    summaries_by_block = {str(item.get("block")): item for item in summaries}
    rows: list[dict[str, Any]] = []
    for block in descriptor.blocks:
        summary = summaries_by_block.get(block)
        if summary is None:
            rows.append(
                {
                    "batch_size": descriptor.batch_size,
                    "seq_len": descriptor.seq_len,
                    "block": block,
                    "stage_role": descriptor.stage_role,
                    "status": "not_observed",
                    "status_reason": f"ncu_block_missing:{block}",
                    "warmup_iters": int(args.profile_warmup_iters),
                    "profile_iters": int(args.profile_iters),
                    "world_size": len(gpu_ids),
                    "gpu_ids": ",".join(str(gpu_id) for gpu_id in gpu_ids),
                    "sm_throughput_pct_mean": None,
                    "sm_active_pct_mean": None,
                    "gpu_time_duration_ns_sum": None,
                    "metric_count": None,
                    "profiler": "ncu",
                    "ncu_raw_csv": str(raw_csv_path),
                    "child_json": str(child_json_path),
                    "command": command_text,
                    "error_type": "ncu_block_missing",
                    "error_message": f"available blocks: {[item.get('block') for item in summaries]}",
                    "started_at_utc": started_at,
                    "completed_at_utc": completed_at,
                }
            )
            continue
        rows.append(
            {
                "batch_size": descriptor.batch_size,
                "seq_len": descriptor.seq_len,
                "block": block,
                "stage_role": descriptor.stage_role,
                "status": "ok",
                "status_reason": None,
                "warmup_iters": int(args.profile_warmup_iters),
                "profile_iters": int(args.profile_iters),
                "world_size": len(gpu_ids),
                "gpu_ids": ",".join(str(gpu_id) for gpu_id in gpu_ids),
                "sm_throughput_pct_mean": summary.get("sm_throughput_pct_mean"),
                "sm_active_pct_mean": summary.get("sm_active_pct_mean"),
                "gpu_time_duration_ns_sum": summary.get("gpu_time_duration_ns_sum"),
                "metric_count": summary.get("metric_count"),
                "profiler": "ncu",
                "ncu_raw_csv": str(raw_csv_path),
                "child_json": str(child_json_path),
                "command": command_text,
                "error_type": None,
                "error_message": None,
                "started_at_utc": started_at,
                "completed_at_utc": completed_at,
            }
        )
    return rows


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def run_utilization_sweep(args: argparse.Namespace, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "megatron_block_ncu_utilization.csv"
    json_path = output_dir / "megatron_block_ncu_utilization.json"
    probe_log = output_dir / "ncu_permission_probe.log"
    prefix, preflight_error = resolve_ncu_prefix(args, probe_log)
    cases = build_cases(args._seq_lens, args._batch_sizes)
    selected_ncu_shapes = {
        (int(batch_size), int(seq_len))
        for batch_size in args._ncu_batch_sizes
        for seq_len in args._ncu_seq_lens
    }
    if prefix is None:
        started_at = now_utc_iso()
        completed_at = now_utc_iso()
        error_message = probe_log.read_text(encoding="utf-8", errors="replace")[-2000:] if probe_log.exists() else ""
        for descriptor in cases:
            gpu_ids = args._attn_gpu_ids if descriptor.stage_role == "attn" else args._moe_gpu_ids
            rows.extend(
                _util_failure_rows(
                    descriptor=descriptor,
                    gpu_ids=gpu_ids,
                    status="profiler_failed",
                    status_reason=preflight_error or "ncu_preflight_failed",
                    command=None,
                    error_type=preflight_error,
                    error_message=error_message,
                    raw_csv_path=None,
                    child_json_path=None,
                    args=args,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                )
            )
        write_csv(csv_path, rows, UTILIZATION_FIELDNAMES)
        payload = _payload_base(metadata, rows)
        payload["preflight_log"] = str(probe_log)
        write_json(json_path, payload)
        return rows

    for descriptor in cases:
        gpu_ids = args._attn_gpu_ids if descriptor.stage_role == "attn" else args._moe_gpu_ids
        if (descriptor.batch_size, descriptor.seq_len) not in selected_ncu_shapes:
            now = now_utc_iso()
            rows.extend(
                _util_failure_rows(
                    descriptor=descriptor,
                    gpu_ids=gpu_ids,
                    status="not_run",
                    status_reason="ncu_subset_not_requested",
                    command=None,
                    error_type=None,
                    error_message=None,
                    raw_csv_path=None,
                    child_json_path=None,
                    args=args,
                    started_at_utc=now,
                    completed_at_utc=now,
                )
            )
            write_csv(csv_path, rows, UTILIZATION_FIELDNAMES)
            payload = _payload_base(metadata, rows)
            payload["preflight_log"] = str(probe_log)
            payload["ncu_prefix"] = list(prefix)
            payload["ncu_metrics"] = list(args._ncu_metrics)
            write_json(json_path, payload)
            continue
        print(
            f"[ncu] {descriptor.case_id} warmup={args.profile_warmup_iters} profile_iters={args.profile_iters}",
            flush=True,
        )
        rows.extend(run_ncu_case(args=args, descriptor=descriptor, gpu_ids=gpu_ids, prefix=prefix))
        write_csv(csv_path, rows, UTILIZATION_FIELDNAMES)
        payload = _payload_base(metadata, rows)
        payload["preflight_log"] = str(probe_log)
        payload["ncu_prefix"] = list(prefix)
        payload["ncu_metrics"] = list(args._ncu_metrics)
        write_json(json_path, payload)
    return rows


def run_profile_child(args: argparse.Namespace) -> int:
    descriptor = CaseDescriptor(
        stage_role=normalize_stage_role(args.stage_role),
        seq_len=args._seq_lens[0],
        batch_size=args._batch_sizes[0],
    )
    gpu_ids = args._attn_gpu_ids if descriptor.stage_role == "attn" else args._moe_gpu_ids
    started_at = now_utc_iso()
    launch_result = _launch_workers(
        descriptor=descriptor,
        gpu_ids=gpu_ids,
        timeout_s=float(args.worker_timeout_s),
        common_config=_common_runtime_config(args),
        include_backward=False,
        enable_cuda_profiler=True,
    )
    completed_at = now_utc_iso()
    rows = aggregate_measurement_rows(
        descriptor=descriptor,
        launch_result=launch_result,
        gpu_ids=gpu_ids,
        warmup_iters=int(args.warmup_iters),
        timed_iters=int(args.timed_iters),
        started_at_utc=started_at,
        completed_at_utc=completed_at,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": launch_result.get("status") or "runtime_error",
        "status_reason": launch_result.get("status_reason"),
        "rows": rows,
    }
    if args.child_json_output:
        write_json(Path(args.child_json_output), payload)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _row_lookup(rows: Sequence[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    return {
        (int(row["batch_size"]), int(row["seq_len"]), str(row["block"])): row
        for row in rows
    }


def _format_table_value(row: dict[str, Any] | None, field: str, digits: int = 2) -> str:
    if row is None:
        return "not_run"
    status = str(row.get("status") or "unknown")
    if status != "ok":
        return status
    value = row.get(field)
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def render_pivot_table(
    *,
    title: str,
    rows: Sequence[dict[str, Any]],
    field: str,
    batch_sizes: Sequence[int],
    seq_lens: Sequence[int],
    digits: int = 2,
) -> str:
    lookup = _row_lookup(rows)
    lines = [
        f"### {title}",
        "",
        "| batch_size | seq_len | " + " | ".join(BLOCK_NAMES) + " |",
        "| ---: | ---: | " + " | ".join("---:" for _ in BLOCK_NAMES) + " |",
    ]
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            cells = [
                _format_table_value(lookup.get((int(batch_size), int(seq_len), block)), field, digits)
                for block in BLOCK_NAMES
            ]
            lines.append(f"| {batch_size} | {seq_len} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def render_results_markdown(
    *,
    metadata: dict[str, Any],
    measurement_rows: Sequence[dict[str, Any]],
    utilization_rows: Sequence[dict[str, Any]],
) -> str:
    baseline = metadata["baseline"]
    batch_sizes = metadata["matrix"]["batch_sizes"]
    seq_lens = metadata["matrix"]["seq_lens"]
    lines = [
        "# Results: attn-moe-composite-scheduler-prototype",
        "",
        "## Correction Status",
        "",
        "The prior synthetic/placeholder composite-scheduler sweep is superseded. "
        "This rerun uses `MegatronSingleLayerRuntime` with fine ranges instrumented "
        "inside the actual Megatron `TransformerLayer` attention and MoE implementation.",
        "",
        "No synthetic Qwen-shaped SDPA or MLP blocks are used for the measured workload.",
        "",
        f"NVTX/record_function prefix: `{metadata.get('nvtx_prefix')}`.",
        "",
        "## Baseline",
        "",
        f"- Model: `{baseline['model_name']}` (`{baseline['model_type']}`)",
        (
            f"- Shape: hidden_size=`{baseline['hidden_size']}`, "
            f"num_attention_heads=`{baseline['num_attention_heads']}`, "
            f"num_key_value_heads=`{baseline['num_key_value_heads']}`, "
            f"head_dim=`{baseline['head_dim']}`, intermediate_size=`{baseline['intermediate_size']}`, "
            f"moe_intermediate_size=`{baseline['moe_intermediate_size']}`"
        ),
        (
            f"- MoE: num_experts=`{baseline['num_experts']}`, "
            f"num_experts_per_tok=`{baseline['num_experts_per_tok']}`, "
            f"norm_topk_prob=`{baseline['norm_topk_prob']}`, "
            f"decoder_sparse_step=`{baseline['decoder_sparse_step']}`"
        ),
        (
            f"- Runtime: dtype=`{baseline['dtype']}`, "
            f"transformer_impl=`{baseline['transformer_impl']}`, "
            f"attention_backend=`{baseline['attention_backend']}`, "
            f"moe_grouped_gemm=`{baseline['moe_grouped_gemm']}`, "
            f"moe_routing_mode=`{baseline['moe_routing_mode']}`, "
            f"moe_token_dispatcher_type=`{baseline['moe_token_dispatcher_type']}`, seed=`{baseline['seed']}`"
        ),
        (
            f"- Positioning: max_position_embeddings=`{baseline['max_position_embeddings']}`, "
            f"rope_theta=`{baseline['rope_theta']}`"
        ),
        "",
        "## Artifacts",
        "",
        "- Measurement CSV/JSON: `verification/megatron_block_measurement.csv`, "
        "`verification/megatron_block_measurement.json`",
        "- NCU CSV/JSON: `verification/megatron_block_ncu_utilization.csv`, "
        "`verification/megatron_block_ncu_utilization.json`",
        "- Layer mapping: `verification/megatron_layer_mapping.json`",
        "- NCU probe: `verification/ncu_permission_probe.log`",
        "- Rerun commands/log tails: `verification/megatron_rerun_commands.txt`, "
        "`verification/megatron_measurement_full_tail.log`, `verification/megatron_ncu_full_tail.log`",
        "",
        "## Tables",
        "",
        render_pivot_table(
            title="Per-Block Peak Allocated Memory (GiB)",
            rows=measurement_rows,
            field="peak_allocated_gib",
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            digits=2,
        ),
        render_pivot_table(
            title="Per-Block Forward Wall Time (ms)",
            rows=measurement_rows,
            field="forward_wall_ms",
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            digits=2,
        ),
        render_pivot_table(
            title="Per-Block SM Throughput %",
            rows=utilization_rows,
            field="sm_throughput_pct_mean",
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            digits=2,
        ),
        render_pivot_table(
            title="Per-Block SM Active %",
            rows=utilization_rows,
            field="sm_active_pct_mean",
            batch_sizes=batch_sizes,
            seq_lens=seq_lens,
            digits=2,
        ),
        "## Status Counts",
        "",
        f"- Measurement: `{build_status_counts(measurement_rows)}`",
        f"- Utilization: `{build_status_counts(utilization_rows)}`",
        "",
        "## Commands",
        "",
        "The exact rerun command list is also saved in `verification/megatron_rerun_commands.txt`.",
        "",
        "```bash",
        metadata.get("command", ""),
        "```",
        "",
        "## Notes",
        "",
        "- Block columns are fine Megatron ranges: `A0_qkv`, `A1_rotary`, "
        "`A2_core_attention`, `A3_output_projection`, `M0_router`, "
        "`M1_dispatch_preprocess`, `M2_token_dispatch`, `M3_dispatch_postprocess`, "
        "`M4_experts`, `M5_combine_preprocess`, `M6_token_combine`.",
        "- `A2_core_attention` is expected to be "
        "`megatron.core.extensions.transformer_engine.TEDotProductAttention`, wrapping "
        "TransformerEngine `te.pytorch.DotProductAttention`; the observed class/module is "
        "recorded in `verification/megatron_layer_mapping.json`.",
        "- Forward+backward is used for memory measurement. The forward wall-time column "
        "is measured inside the same timed iterations before the scalar-loss backward pass.",
        "",
    ]
    return "\n".join(lines)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("rows") or [])


def write_results_markdown(args: argparse.Namespace, metadata: dict[str, Any]) -> None:
    output_dir = Path(args.output_dir)
    measurement_rows = load_rows(output_dir / "megatron_block_measurement.json")
    utilization_rows = load_rows(output_dir / "megatron_block_ncu_utilization.json")
    path = Path(args.results_md)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_results_markdown(
            metadata=metadata,
            measurement_rows=measurement_rows,
            utilization_rows=utilization_rows,
        ),
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--seq-lens", default=",".join(str(value) for value in DEFAULT_SEQ_LENS))
    parser.add_argument("--attn-gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--moe-gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--stage-role", choices=STAGE_ROLES, default="attn", help=argparse.SUPPRESS)
    parser.add_argument("--gpu-ids", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model-name", default=QWEN3_30B_A3B_BASELINE["model_name"])
    parser.add_argument("--model-type", default=QWEN3_30B_A3B_BASELINE["model_type"])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-grouped-gemm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-token-dispatcher-type", default="alltoall")
    parser.add_argument("--moe-routing-mode", choices=["normal", "equal_tokens"], default="equal_tokens")
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--profile-warmup-iters", type=int, default=100)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--worker-timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--results-md", type=Path, default=None)
    parser.add_argument("--skip-measurement", action="store_true")
    parser.add_argument("--skip-utilization", action="store_true")
    parser.add_argument("--ncu-path", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--ncu-prefix", default="")
    parser.add_argument("--ncu-metrics", default=",".join(DEFAULT_NCU_METRICS))
    parser.add_argument("--ncu-timeout-sec", type=int, default=900)
    parser.add_argument("--ncu-raw-dir", type=Path, default=Path("/mnt/local_storage/attn-moe-composite-scheduler-prototype/ncu_raw"))
    parser.add_argument("--ncu-batch-sizes", default=None)
    parser.add_argument("--ncu-seq-lens", default=None)
    parser.add_argument("--profile-child", action="store_true")
    parser.add_argument("--child-json-output", type=Path, default=None)
    return parser.parse_args()


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.dtype = normalize_dtype_name(args.dtype)
    args._seq_lens = parse_seq_lens(args.seq_lens)
    args._batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.gpu_ids:
        parsed = parse_gpu_ids(args.gpu_ids)
        if args.stage_role == "attn":
            args.attn_gpu_ids = args.gpu_ids
        else:
            args.moe_gpu_ids = args.gpu_ids
    else:
        parsed = None
    args._attn_gpu_ids = parsed if args.gpu_ids and args.stage_role == "attn" else parse_gpu_ids(args.attn_gpu_ids)
    args._moe_gpu_ids = parsed if args.gpu_ids and args.stage_role == "moe" else parse_gpu_ids(args.moe_gpu_ids)
    args._ncu_metrics = parse_metric_csv(args.ncu_metrics)
    args._ncu_batch_sizes = parse_batch_sizes(args.ncu_batch_sizes or args.batch_sizes)
    args._ncu_seq_lens = parse_seq_lens(args.ncu_seq_lens or args.seq_lens)
    if args.results_md is None:
        args.results_md = args.output_dir.parent / "results.md" if args.output_dir.name == "verification" else args.output_dir / "results.md"
    if args.num_experts <= 0:
        raise ValueError("--num-experts must be positive")
    if args.num_experts % len(args._moe_gpu_ids) != 0:
        raise ValueError("--num-experts must be divisible by moe world size")
    if args.timed_iters <= 0 or args.profile_iters <= 0:
        raise ValueError("--timed-iters and --profile-iters must be positive")
    return args


def main() -> int:
    mp.set_start_method("spawn", force=True)
    args = _normalize_args(_parse_args())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(args)
    if args.profile_child:
        return run_profile_child(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Megatron block benchmarking")
    if not args.skip_measurement:
        measurement_rows = run_measurement_sweep(args, metadata)
        write_json(args.output_dir / "megatron_block_measurement.json", _payload_base(metadata, measurement_rows))
    if not args.skip_utilization:
        utilization_rows = run_utilization_sweep(args, metadata)
        payload = _payload_base(metadata, utilization_rows)
        payload["ncu_metrics"] = list(args._ncu_metrics)
        write_json(args.output_dir / "megatron_block_ncu_utilization.json", payload)
    write_results_markdown(args, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
