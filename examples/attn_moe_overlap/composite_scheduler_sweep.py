"""Sweep the Attention/MoE composite scheduler for memory and Nsight utilization."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.attn_moe_overlap.composite_scheduler import (  # noqa: E402
    FINAL_OUTPUTS,
    PROFILE_LABEL_PREFIX,
    ShapeConfig,
    _build_workload,
    _final_checksums,
    _memory_peaks,
    _nvtx_range,
    _resolve_device,
    _run_serial,
    _run_stream_cuda,
    _topological_waves,
    default_block_specs,
    probe_runtime_capabilities,
    run_composite_schedule,
)

DEFAULT_BATCH_SIZES = (1, 2, 4)
DEFAULT_SEQ_LENS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_HIDDEN = 1024
DEFAULT_NUM_EXPERTS = 8
DEFAULT_TOP_K = 2
DEFAULT_NCU_METRICS = (
    "gpu__time_duration.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
)
BLOCK_NAMES = tuple(block.name for block in default_block_specs())
BLOCK_FIELD_PREFIXES = tuple(f"forward_block_{name}" for name in BLOCK_NAMES)

MEMORY_BASE_FIELDNAMES = (
    "batch_size",
    "seq_len",
    "status",
    "status_reason",
    "schedule",
    "device",
    "hidden",
    "num_experts",
    "top_k",
    "warmup_iters",
    "timed_iters",
    "seed",
    "forward_backward_wall_ms",
    "forward_wall_ms_total",
    "backward_wall_ms_total",
    "loss",
    "tokens_per_iter",
    "tokens_per_second",
    "forward_backward_peak_allocated_bytes",
    "forward_backward_peak_allocated_mib",
    "forward_backward_peak_reserved_bytes",
    "forward_backward_peak_reserved_mib",
    "memory_allocated_before_bytes",
    "memory_reserved_before_bytes",
    "memory_allocated_after_bytes",
    "memory_reserved_after_bytes",
    "forward_profile_status",
    "forward_profile_status_reason",
    "forward_profile_wall_clock_ms",
    "attention_output_checksum",
    "moe_output_checksum",
    "cuda_available",
    "gpu_name",
    "torch_version",
    "torch_cuda_version",
    "python_version",
    "command",
    "started_at_utc",
    "completed_at_utc",
    "error_type",
    "error_message",
)
MEMORY_FIELDNAMES = MEMORY_BASE_FIELDNAMES + tuple(
    field
    for prefix in BLOCK_FIELD_PREFIXES
    for field in (
        f"{prefix}_peak_allocated_bytes",
        f"{prefix}_peak_allocated_mib",
        f"{prefix}_peak_reserved_bytes",
        f"{prefix}_peak_reserved_mib",
    )
)

UTILIZATION_FIELDNAMES = (
    "batch_size",
    "seq_len",
    "block",
    "status",
    "status_reason",
    "profiler",
    "metric_count",
    "kernel_metric_rows",
    "gpu_time_duration_ns_sum",
    "sm_throughput_pct_mean",
    "sm_cycles_active_pct_mean",
    "ncu_raw_csv",
    "child_json",
    "command",
    "started_at_utc",
    "completed_at_utc",
    "error_type",
    "error_message",
)


@dataclass(frozen=True)
class SweepConfig:
    batch_sizes: tuple[int, ...]
    seq_lens: tuple[int, ...]
    schedule: str
    device: str
    hidden: int
    num_experts: int
    top_k: int
    warmup_iters: int
    timed_iters: int
    seed: int
    profile_forward_blocks: bool
    ncu_metrics: tuple[str, ...]
    ncu_timeout_sec: int
    ncu_path: str
    cuda_visible_devices: str
    output_dir: Path
    command: str


class RunLogger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{now_utc_iso()}] {message}"
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_csv(raw: str, *, field_name: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must contain comma-separated integers, got {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(values)


def parse_metric_csv(raw: str) -> tuple[str, ...]:
    values = tuple(metric.strip() for metric in raw.split(",") if metric.strip())
    if not values:
        raise ValueError("ncu_metrics must not be empty")
    return values


def canonical_matrix(seq_lens: Sequence[int], batch_sizes: Sequence[int]) -> list[tuple[int, int]]:
    return [(int(seq_len), int(batch_size)) for batch_size in batch_sizes for seq_len in seq_lens]


def case_id(*, batch_size: int, seq_len: int) -> str:
    return f"batch{batch_size}_seq{seq_len}"


def bytes_to_mib(value: int | None) -> float | None:
    if value is None:
        return None
    return float(value) / (1024.0**2)


def cuda_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "allocated": None,
            "reserved": None,
            "max_allocated": None,
            "max_reserved": None,
        }
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def _cleanup_cuda_after_error(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


def _set_initial_requires_grad(state: Any) -> None:
    for tensor in state.values.values():
        if tensor.is_floating_point():
            tensor.requires_grad_(True)


def _loss_from_final_outputs(state: Any) -> torch.Tensor:
    losses = [state.values[name].float().mean() for name in FINAL_OUTPUTS if name in state.values]
    if not losses:
        raise RuntimeError("No final outputs were produced by the composite scheduler")
    loss = losses[0]
    for term in losses[1:]:
        loss = loss + term
    return loss


def _run_forward_once(
    *,
    schedule: str,
    blocks: Sequence[Any],
    waves: Sequence[Sequence[Any]],
    state: Any,
    shape: ShapeConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float]:
    if schedule == "stream" and device.type == "cuda":
        return _run_stream_cuda(
            blocks=blocks,
            waves=waves,
            state=state,
            shape=shape,
            device=device,
            warmup_iters=0,
            timed_iters=1,
        )
    return _run_serial(
        blocks=blocks,
        waves=waves,
        state=state,
        shape=shape,
        device=device,
        warmup_iters=0,
        timed_iters=1,
        stream_role="cpu" if schedule == "stream" and device.type == "cpu" else "default",
    )


def _run_forward_backward_once(
    *,
    schedule: str,
    blocks: Sequence[Any],
    waves: Sequence[Sequence[Any]],
    shape: ShapeConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    state = _build_workload(shape, device, seed)
    _set_initial_requires_grad(state)
    block_records, forward_wall_ms = _run_forward_once(
        schedule=schedule,
        blocks=blocks,
        waves=waves,
        state=state,
        shape=shape,
        device=device,
    )
    loss = _loss_from_final_outputs(state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_start = time.perf_counter()
    with _nvtx_range("attn_moe_backward", enabled=device.type == "cuda"):
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    backward_wall_ms = (time.perf_counter() - backward_start) * 1000.0
    return {
        "state": state,
        "block_records": block_records,
        "forward_wall_ms": float(forward_wall_ms),
        "backward_wall_ms": float(backward_wall_ms),
        "loss": float(loss.detach().float().item()),
        "final_checksums": _final_checksums(state),
    }


def _empty_memory_row(
    *,
    batch_size: int,
    seq_len: int,
    config: SweepConfig,
    started_at_utc: str,
) -> dict[str, Any]:
    row = {field: None for field in MEMORY_FIELDNAMES}
    row.update(
        {
            "batch_size": int(batch_size),
            "seq_len": int(seq_len),
            "schedule": config.schedule,
            "device": config.device,
            "hidden": int(config.hidden),
            "num_experts": int(config.num_experts),
            "top_k": int(config.top_k),
            "warmup_iters": int(config.warmup_iters),
            "timed_iters": int(config.timed_iters),
            "seed": int(config.seed),
            "tokens_per_iter": int(batch_size) * int(seq_len),
            "command": config.command,
            "started_at_utc": started_at_utc,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
        }
    )
    return row


def _add_memory_value(row: dict[str, Any], prefix: str, value: int | None) -> None:
    row[f"{prefix}_bytes"] = None if value is None else int(value)
    row[f"{prefix}_mib"] = bytes_to_mib(value)


def _add_forward_profile_to_row(row: dict[str, Any], payload: dict[str, Any]) -> None:
    row["forward_profile_wall_clock_ms"] = payload["summary"]["wall_clock_ms"]
    for block in payload["blocks"]:
        profile = block.get("profile") or {}
        prefix = f"forward_block_{block['name']}"
        allocated = profile.get("peak_allocated_bytes")
        reserved = profile.get("peak_reserved_bytes")
        _add_memory_value(row, f"{prefix}_peak_allocated", allocated)
        _add_memory_value(row, f"{prefix}_peak_reserved", reserved)


def _run_forward_profile(
    *,
    row: dict[str, Any],
    config: SweepConfig,
    shape: ShapeConfig,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> None:
    if not config.profile_forward_blocks:
        row["forward_profile_status"] = "skipped"
        row["forward_profile_status_reason"] = "disabled"
        return
    if device.type != "cuda":
        row["forward_profile_status"] = "skipped"
        row["forward_profile_status_reason"] = "cuda_required"
        return
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        payload = run_composite_schedule(
            schedule=config.schedule,
            device=config.device,
            shape=shape,
            warmup_iters=config.warmup_iters,
            timed_iters=1,
            seed=config.seed + 100_000 + int(batch_size) * 1000 + int(seq_len),
            profile=True,
            profile_trace_output=None,
        )
        row["forward_profile_status"] = "ok"
        row["forward_profile_status_reason"] = None
        _add_forward_profile_to_row(row, payload)
    except BaseException as exc:  # noqa: BLE001 - rows should record failures and continue
        row["forward_profile_status"] = "oom" if _is_oom(exc) else "error"
        row["forward_profile_status_reason"] = type(exc).__name__
        if _is_oom(exc):
            _cleanup_cuda_after_error(device)


def run_memory_case(
    *,
    config: SweepConfig,
    batch_size: int,
    seq_len: int,
) -> dict[str, Any]:
    started_at_utc = now_utc_iso()
    row = _empty_memory_row(batch_size=batch_size, seq_len=seq_len, config=config, started_at_utc=started_at_utc)
    shape = ShapeConfig(
        batch=int(batch_size),
        seq_len=int(seq_len),
        hidden=int(config.hidden),
        num_experts=int(config.num_experts),
        top_k=int(config.top_k),
    )
    device = _resolve_device(config.device)
    capabilities = probe_runtime_capabilities(config.device)
    cuda_info = capabilities["cuda"]
    row["cuda_available"] = cuda_info["available"]
    row["gpu_name"] = cuda_info["device_name"]
    blocks = default_block_specs()
    waves, cycle_blocks = _topological_waves(blocks)
    if cycle_blocks:
        raise RuntimeError(f"Unexpected cycle in default block specs: {cycle_blocks}")

    last_result: dict[str, Any] | None = None
    forward_wall_ms_total = 0.0
    backward_wall_ms_total = 0.0
    try:
        if device.type == "cuda":
            torch.cuda.set_device(device)
        for warmup_idx in range(config.warmup_iters):
            warmup_result = _run_forward_backward_once(
                schedule=config.schedule,
                blocks=blocks,
                waves=waves,
                shape=shape,
                device=device,
                seed=config.seed + warmup_idx,
            )
            del warmup_result
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()
        before = cuda_memory_snapshot(device)
        wall_start = time.perf_counter()
        for timed_idx in range(config.timed_iters):
            last_result = _run_forward_backward_once(
                schedule=config.schedule,
                blocks=blocks,
                waves=waves,
                shape=shape,
                device=device,
                seed=config.seed + 10_000 + timed_idx,
            )
            forward_wall_ms_total += float(last_result["forward_wall_ms"])
            backward_wall_ms_total += float(last_result["backward_wall_ms"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        after = cuda_memory_snapshot(device)
        peak_allocated, peak_reserved = _memory_peaks(device)
        row.update(
            {
                "status": "ok",
                "status_reason": None,
                "forward_backward_wall_ms": float(wall_ms),
                "forward_wall_ms_total": forward_wall_ms_total,
                "backward_wall_ms_total": backward_wall_ms_total,
                "loss": None if last_result is None else last_result["loss"],
                "tokens_per_second": (
                    (int(batch_size) * int(seq_len) * int(config.timed_iters)) / (wall_ms / 1000.0)
                    if wall_ms > 0.0
                    else None
                ),
                "forward_backward_peak_allocated_bytes": peak_allocated,
                "forward_backward_peak_allocated_mib": bytes_to_mib(peak_allocated),
                "forward_backward_peak_reserved_bytes": peak_reserved,
                "forward_backward_peak_reserved_mib": bytes_to_mib(peak_reserved),
                "memory_allocated_before_bytes": before["allocated"],
                "memory_reserved_before_bytes": before["reserved"],
                "memory_allocated_after_bytes": after["allocated"],
                "memory_reserved_after_bytes": after["reserved"],
            }
        )
        if last_result is not None:
            final_checksums = last_result["final_checksums"]
            row["attention_output_checksum"] = final_checksums.get("attention_output")
            row["moe_output_checksum"] = final_checksums.get("moe_output")
        del last_result
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _run_forward_profile(
            row=row,
            config=config,
            shape=shape,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
    except BaseException as exc:  # noqa: BLE001 - one failed config must not abort the matrix
        row["status"] = "oom" if _is_oom(exc) else "error"
        row["status_reason"] = type(exc).__name__
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc).splitlines()[0][:500]
        if _is_oom(exc):
            _cleanup_cuda_after_error(device)
        else:
            gc.collect()
        row["forward_profile_status"] = "skipped"
        row["forward_profile_status_reason"] = f"forward_backward_{row['status']}"
        row["traceback"] = traceback.format_exc(limit=5)
    finally:
        row["completed_at_utc"] = now_utc_iso()
        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def _float_or_none(raw: Any) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().replace(",", "")
    if not text or text.lower() in {"nan", "n/a"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _find_ncu_csv_header(lines: Sequence[str]) -> int | None:
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            return index
    return None


def _infer_block_from_kernel_name(kernel_name: str) -> str:
    if PROFILE_LABEL_PREFIX in kernel_name:
        tail = kernel_name.split(PROFILE_LABEL_PREFIX, maxsplit=1)[1]
        return tail.split(":", maxsplit=1)[0].split("/", maxsplit=1)[0].strip() or "unattributed"
    for block_name in BLOCK_NAMES:
        if block_name in kernel_name:
            return block_name
    return "unattributed"


def summarize_ncu_csv(csv_path: Path) -> tuple[list[dict[str, Any]], str | None]:
    if not csv_path.exists():
        return [], "ncu_csv_missing"
    lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_index = _find_ncu_csv_header(lines)
    if header_index is None:
        return [], "ncu_csv_header_missing"

    grouped: dict[str, dict[str, list[float]]] = {}
    metric_rows = 0
    reader = csv.DictReader(lines[header_index:])
    for csv_row in reader:
        metric_name = csv_row.get("Metric Name")
        metric_value = _float_or_none(csv_row.get("Metric Value"))
        if metric_name is None or metric_value is None:
            continue
        kernel_name = csv_row.get("Kernel Name") or csv_row.get("Name") or ""
        block = _infer_block_from_kernel_name(kernel_name)
        grouped.setdefault(block, {}).setdefault(metric_name, []).append(metric_value)
        metric_rows += 1

    if metric_rows == 0:
        return [], "ncu_csv_no_metric_rows"

    summaries: list[dict[str, Any]] = []
    for block, metrics in sorted(grouped.items()):
        duration_values = metrics.get("gpu__time_duration.sum", [])
        sm_throughput_values = metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", [])
        sm_active_values = metrics.get("smsp__cycles_active.avg.pct_of_peak_sustained_elapsed", [])
        summaries.append(
            {
                "block": block,
                "metric_count": sum(len(values) for values in metrics.values()),
                "kernel_metric_rows": sum(len(values) for values in metrics.values()),
                "gpu_time_duration_ns_sum": sum(duration_values) if duration_values else None,
                "sm_throughput_pct_mean": (
                    sum(sm_throughput_values) / len(sm_throughput_values) if sm_throughput_values else None
                ),
                "sm_cycles_active_pct_mean": (
                    sum(sm_active_values) / len(sm_active_values) if sm_active_values else None
                ),
            }
        )
    return summaries, None


def build_ncu_command(
    *,
    config: SweepConfig,
    batch_size: int,
    seq_len: int,
    raw_csv_path: Path,
    child_json_path: Path,
) -> list[str]:
    script_path = Path(__file__).resolve()
    return [
        config.ncu_path,
        "--target-processes",
        "all",
        "--nvtx",
        "--print-nvtx-rename",
        "kernel",
        "--print-summary",
        "per-nvtx",
        "--csv",
        "--page",
        "raw",
        "--print-fp",
        "--print-units",
        "base",
        "--metrics",
        ",".join(config.ncu_metrics),
        "--log-file",
        str(raw_csv_path),
        "--force-overwrite",
        sys.executable,
        str(script_path),
        "--ncu-profile-child",
        "--schedule",
        config.schedule,
        "--device",
        config.device,
        "--batch",
        str(batch_size),
        "--seq-len",
        str(seq_len),
        "--hidden",
        str(config.hidden),
        "--num-experts",
        str(config.num_experts),
        "--top-k",
        str(config.top_k),
        "--warmup-iters",
        str(config.warmup_iters),
        "--timed-iters",
        "1",
        "--seed",
        str(config.seed),
        "--child-json-output",
        str(child_json_path),
    ]


def _util_failure_rows_for_matrix(
    *,
    config: SweepConfig,
    status_reason: str,
    error_message: str,
    command: str,
    started_at_utc: str,
    completed_at_utc: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq_len, batch_size in canonical_matrix(config.seq_lens, config.batch_sizes):
        row = {field: None for field in UTILIZATION_FIELDNAMES}
        row.update(
            {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "block": "all",
                "status": "profiler_failed",
                "status_reason": status_reason,
                "profiler": "ncu",
                "command": command,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "error_type": status_reason,
                "error_message": error_message,
            }
        )
        rows.append(row)
    return rows


def run_ncu_preflight(*, config: SweepConfig, log_path: Path) -> tuple[bool, str | None, str, str]:
    command = [
        config.ncu_path,
        "--query-metrics-mode",
        "all",
        "--metrics",
        ",".join(config.ncu_metrics),
    ]
    started_at_utc = now_utc_iso()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=config.ncu_timeout_sec, check=False)
    completed_at_utc = now_utc_iso()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(shlex.quote(part) for part in command),
                f"exit_code={completed.returncode}",
                "--- stdout ---",
                completed.stdout,
                "--- stderr ---",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    output = completed.stdout + completed.stderr
    if "ERR_NVGPUCTRPERM" in output or "permission" in output.lower():
        return False, "ncu_permission_denied", started_at_utc, completed_at_utc
    if completed.returncode == 0:
        return True, None, started_at_utc, completed_at_utc
    return False, "ncu_preflight_failed", started_at_utc, completed_at_utc


def run_ncu_case(
    *,
    config: SweepConfig,
    batch_size: int,
    seq_len: int,
) -> list[dict[str, Any]]:
    started_at_utc = now_utc_iso()
    raw_dir = config.output_dir / "ncu_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_csv_path = raw_dir / f"{case_id(batch_size=batch_size, seq_len=seq_len)}.csv"
    child_json_path = raw_dir / f"{case_id(batch_size=batch_size, seq_len=seq_len)}_child.json"
    command = build_ncu_command(
        config=config,
        batch_size=batch_size,
        seq_len=seq_len,
        raw_csv_path=raw_csv_path,
        child_json_path=child_json_path,
    )
    env = os.environ.copy()
    if config.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = config.cuda_visible_devices
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=config.ncu_timeout_sec,
            check=False,
            env=env,
        )
        completed_at_utc = now_utc_iso()
    except subprocess.TimeoutExpired as exc:
        completed_at_utc = now_utc_iso()
        row = {field: None for field in UTILIZATION_FIELDNAMES}
        row.update(
            {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "block": "all",
                "status": "timeout",
                "status_reason": "ncu_timeout",
                "profiler": "ncu",
                "ncu_raw_csv": str(raw_csv_path),
                "child_json": str(child_json_path),
                "command": " ".join(shlex.quote(part) for part in command),
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "error_type": "TimeoutExpired",
                "error_message": str(exc),
            }
        )
        return [row]

    command_text = " ".join(shlex.quote(part) for part in command)
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").splitlines()
        row = {field: None for field in UTILIZATION_FIELDNAMES}
        row.update(
            {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "block": "all",
                "status": "profiler_failed",
                "status_reason": "ncu_failed",
                "profiler": "ncu",
                "ncu_raw_csv": str(raw_csv_path),
                "child_json": str(child_json_path),
                "command": command_text,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "error_type": "ncu_failed",
                "error_message": "\n".join(message[:6])[:500],
            }
        )
        return [row]

    summaries, parse_error = summarize_ncu_csv(raw_csv_path)
    if parse_error is not None:
        row = {field: None for field in UTILIZATION_FIELDNAMES}
        row.update(
            {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "block": "all",
                "status": "parse_failed",
                "status_reason": parse_error,
                "profiler": "ncu",
                "ncu_raw_csv": str(raw_csv_path),
                "child_json": str(child_json_path),
                "command": command_text,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                "error_type": parse_error,
                "error_message": parse_error,
            }
        )
        return [row]

    rows: list[dict[str, Any]] = []
    for summary in summaries:
        row = {field: None for field in UTILIZATION_FIELDNAMES}
        row.update(
            {
                "batch_size": int(batch_size),
                "seq_len": int(seq_len),
                "block": summary["block"],
                "status": "ok",
                "status_reason": None,
                "profiler": "ncu",
                "ncu_raw_csv": str(raw_csv_path),
                "child_json": str(child_json_path),
                "command": command_text,
                "started_at_utc": started_at_utc,
                "completed_at_utc": completed_at_utc,
                **summary,
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _base_payload(*, config: SweepConfig) -> dict[str, Any]:
    return {
        "schema_version": "attn_moe_composite_scheduler_sweep.v1",
        "command": config.command,
        "matrix": {
            "batch_sizes": list(config.batch_sizes),
            "seq_lens": list(config.seq_lens),
        },
        "shape": {
            "hidden": int(config.hidden),
            "num_experts": int(config.num_experts),
            "top_k": int(config.top_k),
        },
        "schedule": config.schedule,
        "device": config.device,
        "iterations": {
            "warmup": int(config.warmup_iters),
            "timed": int(config.timed_iters),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_visible_devices": config.cuda_visible_devices,
        },
    }


def run_memory_sweep(*, config: SweepConfig, logger: RunLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = config.output_dir / "composite_scheduler_memory_sweep.csv"
    json_path = config.output_dir / "composite_scheduler_memory_sweep.json"
    for seq_len, batch_size in canonical_matrix(config.seq_lens, config.batch_sizes):
        logger.log(f"memory start {case_id(batch_size=batch_size, seq_len=seq_len)}")
        row = run_memory_case(config=config, batch_size=batch_size, seq_len=seq_len)
        rows.append(row)
        write_csv(csv_path, rows, MEMORY_FIELDNAMES)
        payload = _base_payload(config=config)
        payload["rows"] = rows
        payload["note"] = (
            "forward_backward_* memory fields execute forward, scalar loss, and backward. "
            "forward_block_* peak fields are forward-only block profiler peaks; backward block attribution is not "
            "reported because autograd does not preserve this scheduler's forward block boundaries."
        )
        write_json(json_path, payload)
        logger.log(
            "memory complete "
            f"{case_id(batch_size=batch_size, seq_len=seq_len)} status={row['status']} "
            f"peak_allocated={row['forward_backward_peak_allocated_bytes']}"
        )
    return rows


def run_utilization_sweep(*, config: SweepConfig, logger: RunLogger) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = config.output_dir / "composite_scheduler_ncu_utilization.csv"
    json_path = config.output_dir / "composite_scheduler_ncu_utilization.json"
    preflight_log = config.output_dir / "ncu_permission_probe.log"
    logger.log("ncu preflight start")
    preflight_ok, reason, started_at_utc, completed_at_utc = run_ncu_preflight(config=config, log_path=preflight_log)
    if not preflight_ok:
        command = f"{config.ncu_path} --query-metrics-mode all --metrics {','.join(config.ncu_metrics)}"
        error_message = preflight_log.read_text(encoding="utf-8", errors="replace")[:1000]
        rows = _util_failure_rows_for_matrix(
            config=config,
            status_reason=reason or "ncu_preflight_failed",
            error_message=error_message,
            command=command,
            started_at_utc=started_at_utc,
            completed_at_utc=completed_at_utc,
        )
        write_csv(csv_path, rows, UTILIZATION_FIELDNAMES)
        payload = _base_payload(config=config)
        payload["profiler"] = "ncu"
        payload["metrics"] = list(config.ncu_metrics)
        payload["preflight_log"] = str(preflight_log)
        payload["rows"] = rows
        write_json(json_path, payload)
        logger.log(f"ncu preflight failed reason={reason}; wrote failed rows for full matrix")
        return rows

    for seq_len, batch_size in canonical_matrix(config.seq_lens, config.batch_sizes):
        logger.log(f"ncu start {case_id(batch_size=batch_size, seq_len=seq_len)}")
        rows.extend(run_ncu_case(config=config, batch_size=batch_size, seq_len=seq_len))
        write_csv(csv_path, rows, UTILIZATION_FIELDNAMES)
        payload = _base_payload(config=config)
        payload["profiler"] = "ncu"
        payload["metrics"] = list(config.ncu_metrics)
        payload["preflight_log"] = str(preflight_log)
        payload["rows"] = rows
        write_json(json_path, payload)
        logger.log(f"ncu complete {case_id(batch_size=batch_size, seq_len=seq_len)}")
    return rows


def run_ncu_profile_child(args: argparse.Namespace) -> None:
    shape = ShapeConfig(
        batch=args.batch,
        seq_len=args.seq_len,
        hidden=args.hidden,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    label = f"attn_moe_config_batch{args.batch}_seq{args.seq_len}"
    try:
        with _nvtx_range(label, enabled=args.device == "cuda"):
            payload = run_composite_schedule(
                schedule=args.schedule,
                device=args.device,
                shape=shape,
                warmup_iters=args.warmup_iters,
                timed_iters=args.timed_iters,
                seed=args.seed,
                profile=False,
            )
        text = json.dumps(payload, indent=2, sort_keys=True)
        if args.child_json_output:
            Path(args.child_json_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.child_json_output).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
    except BaseException as exc:  # noqa: BLE001 - child failures should become profiler rows
        payload = {
            "status": "oom" if _is_oom(exc) else "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        if args.child_json_output:
            Path(args.child_json_output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.child_json_output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--seq-lens", default=",".join(str(value) for value in DEFAULT_SEQ_LENS))
    parser.add_argument("--schedule", choices=("serial", "stream"), default="stream")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--warmup-iters", type=int, default=0)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, required=False, default=Path("/tmp/attn-moe-composite-scheduler-prototype")
    )
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--run-memory", action="store_true", default=True)
    parser.add_argument("--skip-memory", action="store_true")
    parser.add_argument("--run-utilization", action="store_true", default=True)
    parser.add_argument("--skip-utilization", action="store_true")
    parser.add_argument("--profile-forward-blocks", action="store_true", default=True)
    parser.add_argument("--no-profile-forward-blocks", action="store_true")
    parser.add_argument("--ncu-path", default="ncu")
    parser.add_argument("--ncu-metrics", default=",".join(DEFAULT_NCU_METRICS))
    parser.add_argument("--ncu-timeout-sec", type=int, default=300)
    parser.add_argument("--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    parser.add_argument("--ncu-profile-child", action="store_true")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--child-json-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.ncu_profile_child:
        run_ncu_profile_child(args)
        return

    command = " ".join(shlex.quote(part) for part in sys.argv)
    config = SweepConfig(
        batch_sizes=parse_int_csv(args.batch_sizes, field_name="batch_sizes"),
        seq_lens=parse_int_csv(args.seq_lens, field_name="seq_lens"),
        schedule=args.schedule,
        device=args.device,
        hidden=int(args.hidden),
        num_experts=int(args.num_experts),
        top_k=int(args.top_k),
        warmup_iters=int(args.warmup_iters),
        timed_iters=int(args.timed_iters),
        seed=int(args.seed),
        profile_forward_blocks=bool(args.profile_forward_blocks and not args.no_profile_forward_blocks),
        ncu_metrics=parse_metric_csv(args.ncu_metrics),
        ncu_timeout_sec=int(args.ncu_timeout_sec),
        ncu_path=args.ncu_path,
        cuda_visible_devices=args.cuda_visible_devices,
        output_dir=args.output_dir,
        command=command,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.log_path or config.output_dir / "composite_scheduler_sweep.log")
    logger.log(f"sweep command: {command}")
    if not args.skip_memory and args.run_memory:
        run_memory_sweep(config=config, logger=logger)
    if not args.skip_utilization and args.run_utilization:
        run_utilization_sweep(config=config, logger=logger)


if __name__ == "__main__":
    main()
