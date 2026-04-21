"""Step 7: Megatron DP-attention + EP-MoE overlap harness with MPS."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

TORCH_PROFILER_TRACE_INDEX_SCHEMA_VERSION = "megatron_ep_overlap.torch_profiler.v2"


def _early_bootstrap_local_pythonpath() -> list[str]:
    """Ensure sibling repo roots are importable during spawn-time module re-import."""
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
        merged_parts = [*added_paths]
        if existing:
            merged_parts.append(existing)
        os.environ["PYTHONPATH"] = ":".join(merged_parts)
    return added_paths


# Spawn workers re-import this module before calling main(); bootstrap early so the
# imports below resolve in child processes as well.
_early_bootstrap_local_pythonpath()

try:
    from examples.attn_moe_overlap.green_context_utils import (
        get_device_total_sms,
        green_context_supported,
    )
    from examples.attn_moe_overlap.megatron_overlap_schema import (
        build_config_fingerprint,
        build_case_id,
        build_case_payload,
        build_invalid_environment_payload,
        build_matrix_summary,
        build_runtime_metadata,
        canonical_nccl_tuple,
        case_output_path,
        compute_host_enqueue_overlap_ms,
        compute_speedup,
        evaluate_stage_diff,
        is_terminal_status,
        load_case_payload,
        normalize_moe_routing_mode,
        normalize_torch_compile_requested,
        normalize_dtype_name,
        parse_batch_sizes,
        parse_dtypes,
        parse_gpu_ids,
        parse_nccl_tuples,
        parse_runtime_backends,
        parse_seq_lens,
        PROFILER_STATUS_KEYS,
        should_retry,
        should_skip_existing,
        tolerance_for_dtype,
        TORCH_PROFILER_SELECTIONS,
        validate_case_payload,
        write_case_json,
        write_json_atomic,
        write_matrix_summary,
        write_matrix_summary_markdown,
    )
except ModuleNotFoundError:
    from green_context_utils import (  # type: ignore[no-redef]
        get_device_total_sms,
        green_context_supported,
    )
    from megatron_overlap_schema import (  # type: ignore[no-redef]
        build_config_fingerprint,
        build_case_id,
        build_case_payload,
        build_invalid_environment_payload,
        build_matrix_summary,
        build_runtime_metadata,
        canonical_nccl_tuple,
        case_output_path,
        compute_host_enqueue_overlap_ms,
        compute_speedup,
        evaluate_stage_diff,
        is_terminal_status,
        load_case_payload,
        normalize_moe_routing_mode,
        normalize_torch_compile_requested,
        normalize_dtype_name,
        parse_batch_sizes,
        parse_dtypes,
        parse_gpu_ids,
        parse_nccl_tuples,
        parse_runtime_backends,
        parse_seq_lens,
        PROFILER_STATUS_KEYS,
        should_retry,
        should_skip_existing,
        tolerance_for_dtype,
        TORCH_PROFILER_SELECTIONS,
        validate_case_payload,
        write_case_json,
        write_json_atomic,
        write_matrix_summary,
        write_matrix_summary_markdown,
    )


def _bootstrap_local_pythonpath() -> list[str]:
    """Add sibling repo roots (Megatron-LM/ms-swift) when available."""
    return _early_bootstrap_local_pythonpath()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _worker_result_path(worker_result_dir: Path, role: str, rank: int) -> Path:
    return worker_result_dir / f"{role}_rank{rank}.json"


def _load_worker_results(worker_result_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(worker_result_dir.glob("*.json")):
        results.append(json.loads(path.read_text()))
    return results


def _classify_error_message(message: str) -> str:
    lower = message.lower()
    if "out of memory" in lower:
        return "oom"
    return "runtime_error"


def _check_mps_support_for_devices(device_ids: list[int]) -> list[str]:
    errors: list[str] = []
    if shutil.which("nvidia-cuda-mps-control") is None:
        errors.append("nvidia-cuda-mps-control not found in PATH")
        return errors
    for gpu_id in device_ids:
        major, minor = torch.cuda.get_device_capability(gpu_id)
        if major < 7:
            errors.append(
                f"gpu-id {gpu_id} has compute capability {major}.{minor}; MPS requires >= 7.0"
            )
    return errors


class MultiGpuMPSContext:
    """Context manager for one MPS daemon serving a selected GPU set."""

    def __init__(
        self,
        gpu_ids: list[int],
        pipe_dir: str | None = None,
        log_dir: str | None = None,
        active_thread_pct: int | None = None,
    ):
        self.gpu_ids = gpu_ids
        self.pipe_dir = pipe_dir or f"/tmp/mm-step7-mps-pipe-{os.getuid()}-{os.getpid()}"
        self.log_dir = log_dir or f"/tmp/mm-step7-mps-log-{os.getuid()}-{os.getpid()}"
        self.active_thread_pct = active_thread_pct
        self._saved_env: dict[str, str | None] = {}
        self._started = False

    def __enter__(self):
        if shutil.which("nvidia-cuda-mps-control") is None:
            raise RuntimeError("nvidia-cuda-mps-control not found in PATH")
        os.makedirs(self.pipe_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        env_updates = {
            "CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in self.gpu_ids),
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
        }
        if self.active_thread_pct is not None:
            env_updates["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.active_thread_pct)

        for key, value in env_updates.items():
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = value

        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self._restore_env()
            raise RuntimeError(f"Failed to start MPS daemon: {result.stderr.strip()}")

        control_socket = Path(self.pipe_dir) / "control"
        for _ in range(30):
            if control_socket.exists():
                self._started = True
                break
            time.sleep(0.1)
        if not self._started:
            self._restore_env()
            raise RuntimeError("MPS daemon started but control socket did not appear")
        return self

    def __exit__(self, *_exc) -> None:
        if self._started:
            env = os.environ.copy()
            env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
            try:
                subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input="quit\n",
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )
            except Exception:
                pass
            time.sleep(0.5)
        for directory in (self.pipe_dir, self.log_dir):
            try:
                shutil.rmtree(directory, ignore_errors=True)
            except Exception:
                pass
        self._restore_env()

    def _restore_env(self) -> None:
        for key, old_value in self._saved_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        self._saved_env = {}

    def env_vars(self) -> dict[str, str]:
        env = {
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
        }
        if self.active_thread_pct is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.active_thread_pct)
        return env


@dataclass(frozen=True)
class CaseDescriptor:
    mode: str
    seq_len: int
    batch_size: int
    runtime_backend: str
    green_ctx_attn_sms: int | None
    green_ctx_moe_sms: int | None
    dtype: str
    nccl_tuple: tuple[int, int, int] | None
    moe_routing_mode: str
    case_id: str
    baseline_key: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megatron single-layer attention/MoE overlap matrix")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--attn-gpu-ids", type=str, required=True)
    parser.add_argument("--moe-gpu-ids", type=str, required=True)
    parser.add_argument("--attn-dp-size", type=int, required=True)
    parser.add_argument("--moe-ep-size", type=int, required=True)
    parser.add_argument("--seq-lens", type=str, default="512,2048,8192")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--dtypes", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--batch-sizes", type=str, default=None)
    parser.add_argument("--runtime-backends", type=str, default="mps_only")
    parser.add_argument("--green-ctx-attn-sms", type=int, default=None)
    parser.add_argument("--green-ctx-moe-sms", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=2)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--nccl-tuples", type=str, default=None)
    parser.add_argument("--nccl-socket-nthreads", type=int, default=None)
    parser.add_argument("--nccl-max-nchannels", type=int, default=None)
    parser.add_argument("--nccl-max-ctas", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--moe-routing-mode", choices=["normal", "equal_tokens"], default="normal")
    parser.add_argument(
        "--moe-token-dispatcher-type",
        choices=["allgather", "alltoall", "flex"],
        default="allgather",
        help="Megatron MoE token dispatcher type. Megatron recommends `alltoall` when expert parallelism is used.",
    )
    parser.add_argument(
        "--moe-grouped-gemm",
        action="store_true",
        help="Enable Megatron grouped GEMM for local expert MLPs when supported.",
    )
    parser.add_argument(
        "--overlap-moe-expert-parallel-comm",
        action="store_true",
        help="Enable Megatron overlap of expert-parallel communication with MoE execution when supported.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "fused", "flash", "unfused"],
        default="auto",
        help="Megatron attention backend for the Step-7 single-layer runtime. "
        "Defaults to Megatron's recommended backend selection (`auto`).",
    )
    parser.add_argument("--capture-nsys", choices=["on", "off"], default="off")
    parser.add_argument("--nsys-bin", type=str, default="nsys")
    parser.add_argument("--capture-torch-profiler", choices=["on", "off"], default="off")
    parser.add_argument("--torch-compile", choices=["on", "off"], default="off")
    parser.add_argument(
        "--torch-profiler-selection",
        choices=TORCH_PROFILER_SELECTIONS,
        default="representative",
    )
    parser.add_argument(
        "--torch-profiler-wait-iters",
        type=int,
        default=None,
        help="Iterations to skip before profiler capture starts; defaults to warmup-iters when unset.",
    )
    parser.add_argument("--torch-profiler-active-iters", type=int, default=5)
    parser.add_argument("--torch-profiler-recovery", action="store_true")
    parser.add_argument("--torch-profiler-trace-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mode", choices=["serial", "overlap"], default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    parser.add_argument("--mps-active-thread-pct", type=int, default=None)
    parser.add_argument(
        "--attn-mps-active-thread-pct",
        type=int,
        default=None,
        help="Override CUDA_MPS_ACTIVE_THREAD_PERCENTAGE for attention workers only.",
    )
    return parser.parse_args()


def _resolve_batch_sizes(args: argparse.Namespace) -> list[int]:
    single_batch_sizes = parse_batch_sizes(str(args.batch_size))
    if args.batch_sizes is None:
        return single_batch_sizes

    batch_sizes = parse_batch_sizes(args.batch_sizes)
    if batch_sizes != single_batch_sizes and args.batch_size != 1:
        raise ValueError(
            "--batch-size and --batch-sizes must resolve to the same values when both are provided"
        )
    return batch_sizes


def _resolve_runtime_backends(args: argparse.Namespace) -> list[str]:
    return parse_runtime_backends(args.runtime_backends)


def _resolved_green_ctx_sms(args: argparse.Namespace) -> dict[str, int | None]:
    return {
        "attn": None if args.green_ctx_attn_sms is None else int(args.green_ctx_attn_sms),
        "moe": None if args.green_ctx_moe_sms is None else int(args.green_ctx_moe_sms),
    }


def _effective_torch_profiler_wait_iters(args: argparse.Namespace) -> int | None:
    if args.capture_torch_profiler != "on":
        return None
    return int(args.warmup_iters if args.torch_profiler_wait_iters is None else args.torch_profiler_wait_iters)


def _profiler_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    if args.capture_torch_profiler != "on":
        return {
            "capture_requested": False,
            "selection": None,
            "wait_iters": None,
            "active_iters": None,
        }
    return {
        "capture_requested": True,
        "selection": args.torch_profiler_selection,
        "wait_iters": _effective_torch_profiler_wait_iters(args),
        "active_iters": int(args.torch_profiler_active_iters),
    }


def _device_sm_signature(device_total_sms: dict[int, int]) -> dict[str, int]:
    return {str(device_id): int(device_total_sms[device_id]) for device_id in sorted(device_total_sms)}


def _run_config_identity_fields(
    *,
    seq_lens: list[int],
    batch_sizes: list[int],
    dtypes: list[str],
    nccl_tuples: list[tuple[int, int, int] | None],
    runtime_backends: list[str],
    green_ctx_sms: dict[str, int | None],
    device_sm_signature: dict[str, int],
    capture_torch_profiler: str,
    torch_profiler_selection: str | None,
    torch_profiler_wait_iters: int | None,
    torch_profiler_active_iters: int | None,
    moe_routing_mode: str,
    torch_compile: str,
) -> dict[str, Any]:
    return {
        "seq_lens": [int(value) for value in seq_lens],
        "batch_sizes": [int(value) for value in batch_sizes],
        "dtypes": [normalize_dtype_name(value) for value in dtypes],
        "nccl_tuples": [canonical_nccl_tuple(tpl) for tpl in nccl_tuples],
        "runtime_backends": list(runtime_backends),
        "green_ctx_sms": {
            "attn": green_ctx_sms.get("attn"),
            "moe": green_ctx_sms.get("moe"),
        },
        "device_sm_signature": dict(device_sm_signature),
        "capture_torch_profiler": capture_torch_profiler,
        "torch_profiler_selection": torch_profiler_selection,
        "torch_profiler_wait_iters": torch_profiler_wait_iters,
        "torch_profiler_active_iters": torch_profiler_active_iters,
        "moe_routing_mode": normalize_moe_routing_mode(moe_routing_mode),
        "torch_compile": normalize_torch_compile_requested(torch_compile),
    }


def _build_run_config(
    *,
    args: argparse.Namespace,
    topology: dict[str, Any],
    seq_lens: list[int],
    batch_sizes: list[int],
    dtypes: list[str],
    nccl_tuples: list[tuple[int, int, int] | None],
    runtime_backends: list[str],
    green_ctx_sms: dict[str, int | None],
    device_sm_signature: dict[str, int],
    nsys_status: str | None = None,
    torch_profiler_status: str | None = None,
) -> dict[str, Any]:
    profiler_config = _profiler_config_from_args(args)
    identity_fields = _run_config_identity_fields(
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        dtypes=dtypes,
        nccl_tuples=nccl_tuples,
        runtime_backends=runtime_backends,
        green_ctx_sms=green_ctx_sms,
        device_sm_signature=device_sm_signature,
        capture_torch_profiler=args.capture_torch_profiler,
        torch_profiler_selection=profiler_config["selection"],
        torch_profiler_wait_iters=profiler_config["wait_iters"],
        torch_profiler_active_iters=profiler_config["active_iters"],
        moe_routing_mode=args.moe_routing_mode,
        torch_compile=args.torch_compile,
    )
    run_config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "topology": topology,
        **identity_fields,
        "config_fingerprint": build_config_fingerprint(identity_fields),
        "modes": [args.single_mode] if args.single_mode is not None else ["serial", "overlap"],
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "worker_timeout_s": args.worker_timeout_s,
        "num_experts": args.num_experts,
        "moe_routing_mode": normalize_moe_routing_mode(args.moe_routing_mode),
        "mps_active_thread_pct": args.mps_active_thread_pct,
        "attn_mps_active_thread_pct": args.attn_mps_active_thread_pct,
        "capture_nsys": args.capture_nsys,
        "nsys_status": nsys_status,
        "attention_backend": args.attention_backend,
        "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
        "moe_grouped_gemm": args.moe_grouped_gemm,
        "overlap_moe_expert_parallel_comm": args.overlap_moe_expert_parallel_comm,
        "capture_torch_profiler": args.capture_torch_profiler,
        "torch_profiler_selection": profiler_config["selection"],
        "torch_profiler_wait_iters": profiler_config["wait_iters"],
        "torch_profiler_active_iters": profiler_config["active_iters"],
        "torch_profiler_status": torch_profiler_status,
    }
    if len(batch_sizes) == 1:
        run_config["batch_size"] = batch_sizes[0]
    return run_config


def _ensure_output_dir_identity_matches(output_dir: Path, run_config: dict[str, Any]) -> None:
    summary_path = output_dir / "matrix_summary.json"
    cases_dir = output_dir / "cases"
    if not summary_path.exists():
        if cases_dir.exists() and any(cases_dir.glob("*.json")):
            raise RuntimeError(
                f"{output_dir} contains case artifacts but no matrix_summary.json; use a fresh output dir"
            )
        return

    existing_summary = json.loads(summary_path.read_text())
    existing_run_config = existing_summary.get("run_config") or {}
    expected_fields = {
        key: run_config.get(key)
        for key in (
            "seq_lens",
            "batch_sizes",
            "dtypes",
            "nccl_tuples",
            "runtime_backends",
            "green_ctx_sms",
            "device_sm_signature",
            "capture_torch_profiler",
            "torch_profiler_selection",
            "torch_profiler_wait_iters",
            "torch_profiler_active_iters",
            "moe_routing_mode",
            "torch_compile",
            "config_fingerprint",
        )
    }
    actual_fields = {
        key: existing_run_config.get(key)
        for key in (
            "seq_lens",
            "batch_sizes",
            "dtypes",
            "nccl_tuples",
            "runtime_backends",
            "green_ctx_sms",
            "device_sm_signature",
            "capture_torch_profiler",
            "torch_profiler_selection",
            "torch_profiler_wait_iters",
            "torch_profiler_active_iters",
            "moe_routing_mode",
            "torch_compile",
            "config_fingerprint",
        )
    }
    if actual_fields != expected_fields:
        raise RuntimeError(
            f"Output-dir config mismatch for {output_dir}: expected {expected_fields}, found {actual_fields}"
        )


def _validate_topology(
    *,
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
    attn_dp_size: int,
    moe_ep_size: int,
    cuda_device_count: int,
) -> list[str]:
    errors: list[str] = []
    if len(attn_gpu_ids) != attn_dp_size:
        errors.append(f"len(attn_gpu_ids) ({len(attn_gpu_ids)}) must equal attn_dp_size ({attn_dp_size})")
    if len(moe_gpu_ids) != moe_ep_size:
        errors.append(f"len(moe_gpu_ids) ({len(moe_gpu_ids)}) must equal moe_ep_size ({moe_ep_size})")
    if not set(attn_gpu_ids).issubset(set(moe_gpu_ids)):
        errors.append("attn_gpu_ids must be a subset of moe_gpu_ids")
    for gpu_id in sorted(set(attn_gpu_ids + moe_gpu_ids)):
        if gpu_id < 0 or gpu_id >= cuda_device_count:
            errors.append(f"gpu-id {gpu_id} out of range [0, {cuda_device_count - 1}]")
    return errors


def _preflight_errors(
    *,
    model_name: str,
    model_type: str,
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
    attn_dp_size: int,
    moe_ep_size: int,
    runtime_backends: list[str],
    green_ctx_sms: dict[str, int | None],
) -> tuple[list[str], dict[int, int]]:
    errors: list[str] = []
    device_total_sms: dict[int, int] = {}
    if not model_name.strip():
        errors.append("model-name must be non-empty")
    if not model_type.strip():
        errors.append("model-type must be non-empty")

    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        return errors, device_total_sms
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 2:
        errors.append(f"Need at least 2 GPUs, found {cuda_device_count}")
        return errors, device_total_sms

    errors.extend(
        _validate_topology(
            attn_gpu_ids=attn_gpu_ids,
            moe_gpu_ids=moe_gpu_ids,
            attn_dp_size=attn_dp_size,
            moe_ep_size=moe_ep_size,
            cuda_device_count=cuda_device_count,
        )
    )

    try:
        import ray  # noqa: F401
    except Exception as exc:
        errors.append(f"Failed to import ray: {exc}")
    try:
        import megatron  # noqa: F401
    except Exception as exc:
        errors.append(f"Failed to import megatron: {exc}")
    try:
        import swift.megatron  # noqa: F401
    except Exception as exc:
        errors.append(f"Failed to import swift.megatron: {exc}")

    unique_gpu_ids = sorted(set(attn_gpu_ids + moe_gpu_ids))
    errors.extend(_check_mps_support_for_devices(unique_gpu_ids))

    for gpu_id in unique_gpu_ids:
        properties = torch.cuda.get_device_properties(gpu_id)
        device_total_sms[gpu_id] = int(properties.multi_processor_count)

    if "mps_green_ctx" in runtime_backends:
        supported, support_error = green_context_supported()
        if not supported:
            errors.append(support_error or "CUDA Green Context support is unavailable")

        for role in ("attn", "moe"):
            requested = green_ctx_sms.get(role)
            if requested is None:
                errors.append(f"--green-ctx-{role}-sms is required when mps_green_ctx is selected")
            elif requested <= 0:
                errors.append(f"--green-ctx-{role}-sms must be > 0, got {requested}")

        if supported:
            for gpu_id in unique_gpu_ids:
                try:
                    device_total_sms[gpu_id] = int(get_device_total_sms(gpu_id))
                except Exception as exc:
                    errors.append(f"Green Context total-SM query failed for gpu-id {gpu_id}: {exc}")

        unique_total_sms = sorted(set(device_total_sms.values()))
        if len(unique_total_sms) > 1:
            errors.append(
                "Heterogeneous GPU total-SM topology is unsupported for mps_green_ctx: "
                f"{_device_sm_signature(device_total_sms)}"
            )
        for role, gpu_ids in (("attn", attn_gpu_ids), ("moe", moe_gpu_ids)):
            requested = green_ctx_sms.get(role)
            if requested is None:
                continue
            for gpu_id in gpu_ids:
                total_sms = device_total_sms.get(gpu_id)
                if total_sms is not None and requested > total_sms:
                    errors.append(
                        f"--green-ctx-{role}-sms ({requested}) exceeds gpu-id {gpu_id} total SMs ({total_sms})"
                    )
    return errors, device_total_sms


def _baseline_key(
    seq_len: int,
    batch_size: int,
    runtime_backend: str,
    green_ctx_attn_sms: int | None,
    green_ctx_moe_sms: int | None,
    moe_routing_mode: str,
    dtype: str,
    nccl_tuple: tuple[int, int, int] | None,
) -> str:
    green_ctx_fragment = (
        f"gc={int(green_ctx_attn_sms or 0)},{int(green_ctx_moe_sms or 0)}"
        if runtime_backend == "mps_green_ctx"
        else "gc=off"
    )
    return (
        f"seq={seq_len}|batch={batch_size}|backend={runtime_backend}|{green_ctx_fragment}|"
        f"routing={normalize_moe_routing_mode(moe_routing_mode)}|"
        f"dtype={normalize_dtype_name(dtype)}|"
        f"nccl={canonical_nccl_tuple(nccl_tuple)}"
    )


def _build_case_descriptors(
    *,
    modes: list[str],
    seq_lens: list[int],
    batch_sizes: list[int],
    runtime_backends: list[str],
    green_ctx_sms: dict[str, int | None],
    dtypes: list[str],
    nccl_tuples: list[tuple[int, int, int] | None],
    seed: int,
    attn_dp_size: int,
    moe_ep_size: int,
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
    moe_routing_mode: str,
) -> list[CaseDescriptor]:
    cases: list[CaseDescriptor] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for runtime_backend in runtime_backends:
                for dtype in dtypes:
                    for nccl_tuple in nccl_tuples:
                        for mode in modes:
                            baseline_key = _baseline_key(
                                seq_len,
                                batch_size,
                                runtime_backend,
                                green_ctx_sms.get("attn"),
                                green_ctx_sms.get("moe"),
                                moe_routing_mode,
                                dtype,
                                nccl_tuple,
                            )
                            case_id = build_case_id(
                                mode=mode,
                                seq_len=seq_len,
                                batch_size=batch_size,
                                runtime_backend=runtime_backend,
                                green_ctx_attn_sms=green_ctx_sms.get("attn"),
                                green_ctx_moe_sms=green_ctx_sms.get("moe"),
                                dtype=dtype,
                                seed=seed,
                                attn_dp_size=attn_dp_size,
                                moe_ep_size=moe_ep_size,
                                attn_gpu_ids=attn_gpu_ids,
                                moe_gpu_ids=moe_gpu_ids,
                                nccl_tuple=nccl_tuple,
                                moe_routing_mode=moe_routing_mode,
                            )
                            cases.append(
                                CaseDescriptor(
                                    mode=mode,
                                    seq_len=seq_len,
                                    batch_size=batch_size,
                                    runtime_backend=runtime_backend,
                                    green_ctx_attn_sms=green_ctx_sms.get("attn"),
                                    green_ctx_moe_sms=green_ctx_sms.get("moe"),
                                    dtype=dtype,
                                    nccl_tuple=nccl_tuple,
                                    moe_routing_mode=moe_routing_mode,
                                    case_id=case_id,
                                    baseline_key=baseline_key,
                                )
                            )
    return cases


def _null_timed_window() -> dict[str, float | None]:
    return {"start_s": None, "end_s": None, "duration_ms": None}


def _nccl_meta(nccl_tuple: tuple[int, int, int] | None) -> dict[str, Any]:
    return {
        "socket_nthreads": None if nccl_tuple is None else nccl_tuple[0],
        "max_nchannels": None if nccl_tuple is None else nccl_tuple[1],
        "max_ctas": None if nccl_tuple is None else nccl_tuple[2],
        "tuple": canonical_nccl_tuple(nccl_tuple),
        "env_applied": nccl_tuple is not None,
    }


def _empty_stage_result(
    *,
    role: str,
    status: str,
    error: dict[str, Any] | None,
    runtime: dict[str, Any] | None = None,
    torch_compile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error": error
        or {
            "code": status,
            "message": f"No results for role={role}",
            "traceback": None,
        },
        "timing_ms": None,
        "timed_window_s": _null_timed_window(),
        "enqueue_windows": [],
        "output_signature": None,
        "finite": {"all_finite": False, "first_nonfinite": None},
        "moe_grouped_gemm": None,
        "moe_token_dispatcher_type": None,
        "overlap_moe_expert_parallel_comm": None,
        "attention_impl": None,
        "torch_compile": torch_compile
        or {
            "requested": "off",
            "status": "eager",
        },
        "runtime": runtime
        or {
            "requested_sms_by_rank": [],
            "granted_sms_by_rank": [],
            "device_total_sms_by_rank": [],
        },
    }


def _aggregate_role_torch_compile(
    role_results: list[dict[str, Any]],
    *,
    torch_compile_requested: str,
    aggregated_error_code: str | None = None,
) -> tuple[dict[str, str], bool]:
    requested = normalize_torch_compile_requested(torch_compile_requested)
    successful_statuses: set[str] = set()
    any_compile_failed = aggregated_error_code == "torch_compile_failed"

    for result in role_results:
        worker_payload = result.get("torch_compile") or {}
        worker_status = worker_payload.get("status")
        if worker_status not in {"eager", "compiled", "compile_failed"}:
            worker_error_code = (result.get("error") or {}).get("code")
            if worker_error_code == "torch_compile_failed":
                worker_status = "compile_failed"
            elif result.get("status") == "ok" and requested == "on":
                worker_status = "compiled"
            else:
                worker_status = "eager"
        worker_status = str(worker_status)
        if worker_status == "compile_failed":
            any_compile_failed = True
        if result.get("status") == "ok":
            successful_statuses.add(worker_status)

    if any_compile_failed:
        status = "compile_failed"
    elif successful_statuses == {"compiled"} and successful_statuses:
        status = "compiled"
    else:
        status = "eager"

    return {"requested": requested, "status": status}, len(successful_statuses) > 1


def _select_root_cause_worker_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    priorities = [
        lambda item: item.get("failure_origin") is True and item.get("status") == "oom",
        lambda item: item.get("failure_origin") is True and item.get("status") == "runtime_error",
        lambda item: item.get("status") == "oom",
        lambda item: item.get("status") == "runtime_error",
        lambda item: item.get("status") == "timeout",
    ]
    for predicate in priorities:
        for result in results:
            if predicate(result):
                return result
    return None


def _normalize_launch_failure(
    *,
    results: list[dict[str, Any]],
    expected: int,
    timed_out: bool,
    exitcodes: dict[str, int | None],
) -> dict[str, Any]:
    root_cause = _select_root_cause_worker_result(results)
    if root_cause is not None:
        status = str(root_cause.get("status") or "runtime_error")
        error = root_cause.get("error") or {
            "code": status,
            "message": f"Worker {root_cause.get('role')} rank {root_cause.get('rank')} failed",
            "traceback": None,
        }
        return {"status": status, "error": error}

    status = "timeout" if timed_out else "runtime_error"
    return {
        "status": status,
        "error": {
            "code": status,
            "message": f"Missing worker results ({len(results)}/{expected}); exitcodes={exitcodes}",
            "traceback": None,
        },
    }


def _worker_main(
    *,
    role: str,
    rank: int,
    world_size: int,
    gpu_id: int,
    master_addr: str,
    master_port: int,
    model_name: str,
    model_type: str,
    runtime_backend: str,
    green_ctx_attn_sms: int | None,
    green_ctx_moe_sms: int | None,
    dtype: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    warmup_iters: int,
    timed_iters: int,
    moe_ep_size: int,
    num_experts: int | None,
    moe_grouped_gemm: bool,
    moe_token_dispatcher_type: str,
    overlap_moe_expert_parallel_comm: bool,
    attention_backend: str,
    nccl_tuple: tuple[int, int, int] | None,
    mps_env: dict[str, str],
    attn_mps_active_thread_pct: int | None,
    execution_schedule: str,
    iteration_barrier: Any | None,
    abort_event: Any | None,
    barrier_timeout_s: float,
    profiler_trace_root: str | None,
    profiler_wait_iters: int | None,
    profiler_active_timed_iters: int | None,
    worker_result_dir: str,
    moe_routing_mode: str,
    torch_compile_enabled: bool,
) -> None:
    try:
        from examples.attn_moe_overlap.megatron_layer_runtime import (
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            ScheduleAborted,
            cleanup_distributed_state,
            classify_exception,
        )
    except ModuleNotFoundError:
        from megatron_layer_runtime import (  # type: ignore[no-redef]
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            ScheduleAborted,
            cleanup_distributed_state,
            classify_exception,
        )

    status = "ok"
    payload: dict[str, Any] = {}
    failure_origin = False
    runtime: Any | None = None
    try:
        _bootstrap_local_pythonpath()
        os.environ.update(mps_env)
        if role == "attn":
            if attn_mps_active_thread_pct is None:
                os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
            else:
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(attn_mps_active_thread_pct)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        for env_key in ("NCCL_SOCKET_NTHREADS", "NCCL_MAX_NCHANNELS", "NCCL_MAX_CTAS"):
            os.environ.pop(env_key, None)
        if nccl_tuple is not None:
            os.environ["NCCL_SOCKET_NTHREADS"] = str(nccl_tuple[0])
            os.environ["NCCL_MAX_NCHANNELS"] = str(nccl_tuple[1])
            os.environ["NCCL_MAX_CTAS"] = str(nccl_tuple[2])
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")

        torch.cuda.set_device(0)
        runtime = MegatronSingleLayerRuntime(
            RuntimeConfig(
                model_name=model_name,
                model_type=model_type,
                stage_role=role,
                runtime_backend=runtime_backend,
                attention_backend=attention_backend,
                moe_grouped_gemm=moe_grouped_gemm,
                moe_token_dispatcher_type=moe_token_dispatcher_type,
                overlap_moe_expert_parallel_comm=overlap_moe_expert_parallel_comm,
                dtype=dtype,
                seq_len=seq_len,
                batch_size=batch_size,
                seed=seed,
                expert_model_parallel_size=moe_ep_size if role == "moe" else 1,
                green_ctx_attn_sms=green_ctx_attn_sms,
                green_ctx_moe_sms=green_ctx_moe_sms,
                num_experts=num_experts,
                moe_routing_mode=moe_routing_mode,
                torch_compile_enabled=bool(torch_compile_enabled),
            )
        )
        payload = runtime.run_stage(
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            execution_schedule=execution_schedule,
            iteration_barrier=iteration_barrier,
            abort_event=abort_event,
            barrier_timeout_s=barrier_timeout_s,
            profiler_trace_dir=str(Path(profiler_trace_root)) if profiler_trace_root is not None else None,
            profiler_worker_name=f"{role}_rank{rank}_gpu{gpu_id}",
            profiler_wait_iters=profiler_wait_iters,
            profiler_active_timed_iters=profiler_active_timed_iters,
        )
        status = payload.get("status", "runtime_error")
    except Exception as exc:
        status, error = classify_exception(exc)
        failure_origin = not isinstance(exc, ScheduleAborted)
        if abort_event is not None:
            abort_event.set()
        if iteration_barrier is not None:
            try:
                iteration_barrier.abort()
            except Exception:
                pass
        torch_compile_payload = (
            runtime.torch_compile_payload()
            if runtime is not None
            else {
                "requested": "on" if torch_compile_enabled else "off",
                "status": "compile_failed" if torch_compile_enabled else "eager",
            }
        )
        payload = {
            "status": status,
            "timing_ms": {"cuda": None, "step_total": None, "timed_wall": None},
            "timed_window_s": _null_timed_window(),
            "schedule_timed_window_s": _null_timed_window(),
            "enqueue_windows": [],
            "finite": {"all_finite": False, "first_nonfinite": None},
            "output_signature": None,
            "torch_compile": torch_compile_payload,
            "error": error,
        }
    finally:
        if runtime is not None:
            try:
                runtime.cleanup()
            except Exception:
                pass
        cleanup_distributed_state()
        write_json_atomic(
            _worker_result_path(Path(worker_result_dir), role, rank),
            {
                "role": role,
                "rank": rank,
                "status": status,
                "failure_origin": failure_origin,
                "attention_backend": attention_backend,
                "moe_routing_mode": moe_routing_mode,
                "moe_grouped_gemm": moe_grouped_gemm,
                "moe_token_dispatcher_type": moe_token_dispatcher_type,
                "overlap_moe_expert_parallel_comm": overlap_moe_expert_parallel_comm,
                **payload,
            },
        )


def _launch_workers(
    *,
    stage_specs: list[dict[str, Any]],
    common_config: dict[str, Any],
    timeout_s: float,
    mps_env: dict[str, str],
) -> dict[str, Any]:
    # Pre-spawn sys.path is what child processes inherit; restore local roots in case
    # imports done earlier (e.g., swift.megatron) rewrote path ordering.
    _bootstrap_local_pythonpath()

    processes: list[mp.Process] = []
    worker_result_dir = Path(tempfile.mkdtemp(prefix="step7_worker_results_"))
    expected = 0
    for spec in stage_specs:
        expected += len(spec["gpu_ids"])
    iteration_barrier = mp.Barrier(expected) if expected > 0 else None
    abort_event = mp.Event()
    try:
        for spec in stage_specs:
            for rank, gpu_id in enumerate(spec["gpu_ids"]):
                process = mp.Process(
                    target=_worker_main,
                    kwargs={
                        "role": spec["role"],
                        "rank": rank,
                        "world_size": spec["world_size"],
                        "gpu_id": gpu_id,
                        "master_addr": "127.0.0.1",
                        "master_port": spec["master_port"],
                        "model_name": common_config["model_name"],
                        "model_type": common_config["model_type"],
                        "runtime_backend": common_config["runtime_backend"],
                        "green_ctx_attn_sms": common_config["green_ctx_attn_sms"],
                        "green_ctx_moe_sms": common_config["green_ctx_moe_sms"],
                        "dtype": common_config["dtype"],
                        "seq_len": common_config["seq_len"],
                        "batch_size": common_config["batch_size"],
                        "seed": common_config["seed"],
                        "warmup_iters": common_config["warmup_iters"],
                        "timed_iters": common_config["timed_iters"],
                        "moe_ep_size": common_config["moe_ep_size"],
                        "num_experts": common_config["num_experts"],
                        "moe_grouped_gemm": common_config["moe_grouped_gemm"],
                        "moe_token_dispatcher_type": common_config["moe_token_dispatcher_type"],
                        "overlap_moe_expert_parallel_comm": common_config[
                            "overlap_moe_expert_parallel_comm"
                        ],
                        "attention_backend": common_config["attention_backend"],
                        "nccl_tuple": common_config["nccl_tuple"],
                        "mps_env": mps_env,
                        "attn_mps_active_thread_pct": common_config["attn_mps_active_thread_pct"],
                        "execution_schedule": common_config["execution_schedule"],
                        "iteration_barrier": iteration_barrier,
                        "abort_event": abort_event,
                        "barrier_timeout_s": timeout_s,
                        "profiler_trace_root": common_config.get("profiler_trace_root"),
                        "profiler_wait_iters": common_config.get("profiler_wait_iters"),
                        "profiler_active_timed_iters": common_config.get("profiler_active_timed_iters"),
                        "worker_result_dir": str(worker_result_dir),
                        "moe_routing_mode": common_config["moe_routing_mode"],
                        "torch_compile_enabled": common_config.get("torch_compile_enabled", False),
                    },
                )
                process.start()
                processes.append(process)

        deadline = time.time() + timeout_s
        while time.time() < deadline:
            results = _load_worker_results(worker_result_dir)
            if len(results) >= expected:
                break
            if processes and all(not process.is_alive() for process in processes):
                break
            time.sleep(0.1)

        results = _load_worker_results(worker_result_dir)
        timed_out = len(results) < expected and any(process.is_alive() for process in processes)
        if timed_out:
            abort_event.set()
            if iteration_barrier is not None:
                try:
                    iteration_barrier.abort()
                except Exception:
                    pass
            for process in processes:
                if process.is_alive():
                    process.terminate()

        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()

        results = _load_worker_results(worker_result_dir)
        schedule_timed_window = _collapse_timed_window_s(
            [item.get("schedule_timed_window_s") for item in results]
        )
        if len(results) < expected:
            exitcodes = {str(process.pid): process.exitcode for process in processes}
            normalized = _normalize_launch_failure(
                results=results,
                expected=expected,
                timed_out=timed_out,
                exitcodes=exitcodes,
            )
            return {
                "status": normalized["status"],
                "error": normalized["error"],
                "results": results,
                "schedule_timed_window_s": schedule_timed_window,
            }
        return {
            "status": "ok",
            "results": results,
            "schedule_timed_window_s": schedule_timed_window,
        }
    finally:
        shutil.rmtree(worker_result_dir, ignore_errors=True)


def _aggregate_stage_results(
    role: str,
    results: list[dict[str, Any]],
    *,
    expected_world_size: int | None = None,
    fallback_status: str | None = None,
    fallback_error: dict[str, Any] | None = None,
    torch_compile_requested: str = "off",
) -> dict[str, Any]:
    role_results = sorted((item for item in results if item.get("role") == role), key=lambda item: item["rank"])
    role_runtime = {
        "requested_sms_by_rank": [((item.get("runtime") or {}).get("requested_sms")) for item in role_results],
        "granted_sms_by_rank": [((item.get("runtime") or {}).get("granted_sms")) for item in role_results],
        "device_total_sms_by_rank": [((item.get("runtime") or {}).get("device_total_sms")) for item in role_results],
    }
    if not role_results:
        role_torch_compile, _ = _aggregate_role_torch_compile(
            role_results,
            torch_compile_requested=torch_compile_requested,
            aggregated_error_code=(fallback_error or {}).get("code"),
        )
        return _empty_stage_result(
            role=role,
            status=fallback_status or "runtime_error",
            error=fallback_error,
            torch_compile=role_torch_compile,
        )

    if expected_world_size is not None and len(role_results) < int(expected_world_size):
        failing = _select_root_cause_worker_result(role_results)
        if failing is not None:
            status = str(failing.get("status") or "runtime_error")
            error = failing.get("error") or {
                "code": status,
                "message": f"{role} rank {failing.get('rank')} failed",
                "traceback": None,
            }
            role_torch_compile, _ = _aggregate_role_torch_compile(
                role_results,
                torch_compile_requested=torch_compile_requested,
                aggregated_error_code=error.get("code"),
            )
            return _empty_stage_result(
                role=role,
                status=status,
                error=error,
                runtime=role_runtime,
                torch_compile=role_torch_compile,
            )
        role_torch_compile, _ = _aggregate_role_torch_compile(
            role_results,
            torch_compile_requested=torch_compile_requested,
            aggregated_error_code=(fallback_error or {}).get("code"),
        )
        return _empty_stage_result(
            role=role,
            status=fallback_status or "runtime_error",
            error=fallback_error,
            runtime=role_runtime,
            torch_compile=role_torch_compile,
        )

    failing = _select_root_cause_worker_result(role_results)
    if failing is not None:
        status = str(failing.get("status") or "runtime_error")
        error = failing.get("error") or {
            "code": status,
            "message": f"{role} rank {failing.get('rank')} failed",
            "traceback": None,
        }
        role_torch_compile, _ = _aggregate_role_torch_compile(
            role_results,
            torch_compile_requested=torch_compile_requested,
            aggregated_error_code=error.get("code"),
        )
        return _empty_stage_result(
            role=role,
            status=status,
            error=error,
            runtime=role_runtime,
            torch_compile=role_torch_compile,
        )

    role_torch_compile, mixed_successful_compile_statuses = _aggregate_role_torch_compile(
        role_results,
        torch_compile_requested=torch_compile_requested,
    )
    if mixed_successful_compile_statuses:
        return _empty_stage_result(
            role=role,
            status="runtime_error",
            error={
                "code": "torch_compile_status_mismatch",
                "message": f"{role} workers reported mixed torch compile statuses",
                "traceback": None,
            },
            runtime=role_runtime,
            torch_compile=role_torch_compile,
        )

    timed_window = _collapse_timed_window_s([item.get("timed_window_s") for item in role_results])
    rank0 = role_results[0]
    first_nonfinite = None
    all_finite = True
    for result in role_results:
        finite = result.get("finite", {})
        if not finite.get("all_finite", False):
            all_finite = False
            if first_nonfinite is None:
                first_nonfinite = finite.get("first_nonfinite")

    return {
        "status": "ok",
        "error": {"code": None, "message": None, "traceback": None},
        "attention_backend": rank0.get("attention_backend"),
        "attention_impl": rank0.get("attention_impl"),
        "torch_compile": role_torch_compile,
        "moe_routing_mode": rank0.get("moe_routing_mode", "normal"),
        "moe_grouped_gemm": rank0.get("moe_grouped_gemm"),
        "moe_token_dispatcher_type": rank0.get("moe_token_dispatcher_type"),
        "overlap_moe_expert_parallel_comm": rank0.get("overlap_moe_expert_parallel_comm"),
        "timing_ms": {
            "cuda": (rank0.get("timing_ms") or {}).get("cuda"),
            "step_total": (rank0.get("timing_ms") or {}).get("step_total"),
            "timed_wall": timed_window["duration_ms"],
        },
        "timed_window_s": timed_window,
        "enqueue_windows": rank0.get("enqueue_windows", []),
        "output_signature": rank0.get("output_signature"),
        "finite": {"all_finite": all_finite, "first_nonfinite": first_nonfinite},
        "runtime": role_runtime,
        "tokens_per_expert": rank0.get("tokens_per_expert"),
        "local_tokens_per_expert_by_rank": [
            item.get("local_tokens_per_expert") for item in role_results if item.get("local_tokens_per_expert") is not None
        ],
    }


def _stage_specs(attn_gpu_ids: list[int], moe_gpu_ids: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    attn_port = _find_free_port()
    moe_port = _find_free_port()
    while moe_port == attn_port:
        moe_port = _find_free_port()
    attn_spec = {
        "role": "attn",
        "gpu_ids": attn_gpu_ids,
        "world_size": len(attn_gpu_ids),
        "master_port": attn_port,
    }
    moe_spec = {
        "role": "moe",
        "gpu_ids": moe_gpu_ids,
        "world_size": len(moe_gpu_ids),
        "master_port": moe_port,
    }
    return attn_spec, moe_spec


def _collapse_timed_window_s(windows: list[dict[str, Any] | None]) -> dict[str, float | None]:
    starts: list[float] = []
    ends: list[float] = []
    for window in windows:
        if not isinstance(window, dict):
            continue
        start_s = window.get("start_s")
        end_s = window.get("end_s")
        if start_s is None or end_s is None:
            continue
        starts.append(float(start_s))
        ends.append(float(end_s))
    if not starts or not ends:
        return {"start_s": None, "end_s": None, "duration_ms": None}
    start_s = min(starts)
    end_s = max(ends)
    return {
        "start_s": start_s,
        "end_s": end_s,
        "duration_ms": max(0.0, (end_s - start_s) * 1000.0),
    }


def _run_case_attempt(
    *,
    mode: str,
    common_config: dict[str, Any],
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
    timeout_s: float,
    mps_env: dict[str, str],
) -> dict[str, Any]:
    start_s = time.perf_counter()
    execution_schedule = "serial_lockstep" if mode == "serial" else "overlap"
    attn_spec, moe_spec = _stage_specs(attn_gpu_ids, moe_gpu_ids)
    launch_result = _launch_workers(
        stage_specs=[attn_spec, moe_spec],
        common_config={**common_config, "execution_schedule": execution_schedule},
        timeout_s=timeout_s,
        mps_env=mps_env,
    )
    results = launch_result.get("results", [])
    fallback_status = launch_result.get("status")
    fallback_error = launch_result.get("error")
    if fallback_status == "ok":
        fallback_status = None
        fallback_error = None

    requested_torch_compile = normalize_torch_compile_requested(str(common_config.get("torch_compile", "off")))
    attn_stage = _aggregate_stage_results(
        "attn",
        results,
        expected_world_size=attn_spec["world_size"],
        fallback_status=fallback_status,
        fallback_error=fallback_error,
        torch_compile_requested=requested_torch_compile,
    )
    moe_stage = _aggregate_stage_results(
        "moe",
        results,
        expected_world_size=moe_spec["world_size"],
        fallback_status=fallback_status,
        fallback_error=fallback_error,
        torch_compile_requested=requested_torch_compile,
    )
    tokens_per_expert: list[int] | None = None
    if common_config["moe_routing_mode"] == "equal_tokens" and moe_stage["status"] == "ok":
        local_vectors = moe_stage.get("local_tokens_per_expert_by_rank") or []
        if not local_vectors:
            raise RuntimeError("equal_tokens expected per-rank local_tokens_per_expert vectors")
        lengths = {len(vector) for vector in local_vectors}
        if len(lengths) != 1:
            raise RuntimeError("equal_tokens local_tokens_per_expert vectors must have identical lengths")
        tokens_per_expert = [0 for _ in range(lengths.pop())]
        for vector in local_vectors:
            for index, value in enumerate(vector):
                tokens_per_expert[index] += int(value)

    case_timed_wall_ms = (launch_result.get("schedule_timed_window_s") or {}).get("duration_ms")
    host_overlap_ms = 0.0
    if mode == "overlap":
        host_overlap_ms = compute_host_enqueue_overlap_ms(
            attn_stage.get("enqueue_windows", []),
            moe_stage.get("enqueue_windows", []),
        )

    wall_ms = (time.perf_counter() - start_s) * 1000.0
    stage_statuses = [attn_stage["status"], moe_stage["status"]]
    if any(status == "oom" for status in stage_statuses):
        status = "oom"
        error = attn_stage["error"] if attn_stage["status"] == "oom" else moe_stage["error"]
    elif any(status == "timeout" for status in stage_statuses):
        status = "timeout"
        error = attn_stage["error"] if attn_stage["status"] == "timeout" else moe_stage["error"]
    elif any(status != "ok" for status in stage_statuses):
        status = "runtime_error"
        stage_errors = [attn_stage.get("error") or {}, moe_stage.get("error") or {}]
        compile_error = next((item for item in stage_errors if item.get("code") == "torch_compile_failed"), None)
        if compile_error is not None:
            error = compile_error
        else:
            error = attn_stage["error"] if attn_stage["status"] != "ok" else moe_stage["error"]
    else:
        status = "ok"
        error = {"code": None, "message": None, "traceback": None}

    first_nonfinite = (
        attn_stage["finite"]["first_nonfinite"]
        if attn_stage["finite"]["first_nonfinite"] is not None
        else moe_stage["finite"]["first_nonfinite"]
    )
    all_finite = bool(attn_stage["finite"]["all_finite"] and moe_stage["finite"]["all_finite"])
    if (
        status == "ok"
        and not all_finite
        and common_config["moe_routing_mode"] != "equal_tokens"
    ):
        status = "runtime_error"
        error = {
            "code": "non_finite",
            "message": "Detected non-finite outputs",
            "traceback": None,
        }

    return {
        "status": status,
        "error": error,
        "runtime_backend": common_config["runtime_backend"],
        "runtime": build_runtime_metadata(
            runtime_backend=common_config["runtime_backend"],
            green_ctx_attn_sms=common_config["green_ctx_attn_sms"],
            green_ctx_moe_sms=common_config["green_ctx_moe_sms"],
            granted_sms_by_role={
                "attn": (attn_stage.get("runtime") or {}).get("granted_sms_by_rank"),
                "moe": (moe_stage.get("runtime") or {}).get("granted_sms_by_rank"),
            },
            device_total_sms_by_role={
                "attn": (attn_stage.get("runtime") or {}).get("device_total_sms_by_rank"),
                "moe": (moe_stage.get("runtime") or {}).get("device_total_sms_by_rank"),
            },
        ),
        "attention_backend": {
            "requested": common_config["attention_backend"],
            "attn": attn_stage.get("attention_backend"),
            "moe": moe_stage.get("attention_backend"),
        },
        "attention_impl": {
            "attn": attn_stage.get("attention_impl"),
            "moe": moe_stage.get("attention_impl"),
        },
        "moe_runtime": {
            "grouped_gemm": {
                "requested": common_config["moe_grouped_gemm"],
                "attn": attn_stage.get("moe_grouped_gemm"),
                "moe": moe_stage.get("moe_grouped_gemm"),
            },
            "token_dispatcher_type": {
                "requested": common_config["moe_token_dispatcher_type"],
                "attn": attn_stage.get("moe_token_dispatcher_type"),
                "moe": moe_stage.get("moe_token_dispatcher_type"),
            },
            "overlap_expert_parallel_comm": {
                "requested": common_config["overlap_moe_expert_parallel_comm"],
                "attn": attn_stage.get("overlap_moe_expert_parallel_comm"),
                "moe": moe_stage.get("overlap_moe_expert_parallel_comm"),
            },
        },
        "moe_routing_mode": common_config["moe_routing_mode"],
        "torch_compile": {
            "requested": requested_torch_compile,
            "by_role": {
                "attn": {"status": str((attn_stage.get("torch_compile") or {}).get("status", "eager"))},
                "moe": {"status": str((moe_stage.get("torch_compile") or {}).get("status", "eager"))},
            },
        },
        "timing_ms": {
            "total": wall_ms,
            "timed_wall": case_timed_wall_ms,
            "attn": attn_stage["timing_ms"]["cuda"] if attn_stage["timing_ms"] else None,
            "moe": moe_stage["timing_ms"]["cuda"] if moe_stage["timing_ms"] else None,
        },
        "overlap_ms": host_overlap_ms,
        "finite": {"all_finite": all_finite, "first_nonfinite": first_nonfinite},
        "stage_signatures": {
            "attn": attn_stage["output_signature"],
            "moe": moe_stage["output_signature"],
        },
        "tokens_per_expert": tokens_per_expert,
    }


def _apply_baseline_diff(overlap_payload: dict[str, Any], serial_payload: dict[str, Any] | None) -> dict[str, Any]:
    if serial_payload is None:
        return overlap_payload
    dtype = overlap_payload["dtype"]
    attn_diff = evaluate_stage_diff(
        stage_name="attn",
        dtype=dtype,
        test_signature=overlap_payload["stage_signatures"].get("attn"),
        ref_signature=serial_payload["stage_signatures"].get("attn"),
    )
    moe_diff = evaluate_stage_diff(
        stage_name="moe",
        dtype=dtype,
        test_signature=overlap_payload["stage_signatures"].get("moe"),
        ref_signature=serial_payload["stage_signatures"].get("moe"),
    )
    all_within = bool(attn_diff["within_tolerance"] and moe_diff["within_tolerance"])
    overlap_payload["baseline_diff"] = {
        "baseline_case_id": serial_payload["case_id"],
        "all_within_tolerance": all_within,
        "stages": {
            "attn": attn_diff,
            "moe": moe_diff,
        },
    }
    overlap_payload["overlap"]["speedup_vs_serial"] = compute_speedup(
        serial_payload["timing_ms"]["total"],
        overlap_payload["timing_ms"]["total"],
    )
    overlap_payload["overlap"]["timed_speedup_vs_serial"] = compute_speedup(
        serial_payload["timing_ms"].get("timed_wall"),
        overlap_payload["timing_ms"].get("timed_wall"),
    )
    if overlap_payload["status"] == "ok" and not all_within:
        overlap_payload["status"] = "numerical_mismatch"
        overlap_payload["error"] = {
            "code": "numerical_mismatch",
            "message": (
                "Baseline tolerance exceeded: "
                f"attn(max_abs={attn_diff['max_abs_diff']}, max_rel={attn_diff['max_rel_diff']}), "
                f"moe(max_abs={moe_diff['max_abs_diff']}, max_rel={moe_diff['max_rel_diff']})"
            ),
            "traceback": None,
        }
    return overlap_payload


def _invalid_env_matrix(
    *,
    cases: list[CaseDescriptor],
    output_dir: str,
    strict_schema: bool,
    error_message: str,
    topology: dict[str, Any],
    profiler: dict[str, Any],
    torch_compile_requested: str,
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for descriptor in cases:
        payload = build_invalid_environment_payload(
            case_id=descriptor.case_id,
            mode=descriptor.mode,
            seq_len=descriptor.seq_len,
            batch_size=descriptor.batch_size,
            runtime_backend=descriptor.runtime_backend,
            dtype=descriptor.dtype,
            seed=topology["seed"],
            topology=topology,
            nccl_env=_nccl_meta(descriptor.nccl_tuple),
            runtime=build_runtime_metadata(
                runtime_backend=descriptor.runtime_backend,
                green_ctx_attn_sms=descriptor.green_ctx_attn_sms,
                green_ctx_moe_sms=descriptor.green_ctx_moe_sms,
            ),
            message=error_message,
            moe_routing_mode=descriptor.moe_routing_mode,
            profiler=profiler,
            torch_compile={
                "requested": normalize_torch_compile_requested(torch_compile_requested),
                "by_role": {"attn": {"status": "eager"}, "moe": {"status": "eager"}},
            },
        )
        write_case_json(output_dir=output_dir, payload=payload, strict_schema=strict_schema)
        payloads.append(payload)
    return payloads


def _load_case_payloads(output_dir: str | Path) -> list[dict[str, Any]]:
    return [load_case_payload(path) for path in sorted((Path(output_dir) / "cases").glob("*.json"))]


def _load_matrix_summary(output_dir: str | Path) -> dict[str, Any]:
    summary_path = Path(output_dir) / "matrix_summary.json"
    if not summary_path.exists():
        raise RuntimeError(f"{summary_path} does not exist")
    return json.loads(summary_path.read_text())


def _pair_metric(pair: dict[str, Any]) -> float | None:
    value = pair.get("delta_overlap_timed_wall_ms")
    if value is not None:
        return float(value)
    value = pair.get("delta_overlap_total_ms")
    if value is not None:
        return float(value)
    return None


def _infer_selection_run_config(cases: list[dict[str, Any]]) -> dict[str, Any]:
    runtime_backends = sorted({str(case.get("runtime_backend") or "") for case in cases if case.get("runtime_backend")})
    green_ctx_case = next((case for case in cases if case.get("runtime_backend") == "mps_green_ctx"), None)
    runtime = (green_ctx_case or {}).get("runtime") or {}
    return {
        "runtime_backends": runtime_backends,
        "green_ctx_sms": dict(runtime.get("requested_sms_by_role") or {"attn": None, "moe": None}),
    }


def _select_profiler_backend_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = build_matrix_summary(
        run_config=_infer_selection_run_config(cases),
        cases=cases,
        total_points=len(cases),
    )
    pair_rows = list(summary.get("backend_pair_rows") or [])
    if not pair_rows:
        return []

    successful_pairs = [pair for pair in pair_rows if pair.get("pair_status") == "ok" and _pair_metric(pair) is not None]
    selected: list[dict[str, Any]] = []

    if successful_pairs:
        positive_pairs = [pair for pair in successful_pairs if float(_pair_metric(pair) or 0.0) > 0.0]
        if positive_pairs:
            selected.append(max(positive_pairs, key=lambda pair: (float(_pair_metric(pair) or 0.0), pair["pair_id"])))

        selected.append(min(successful_pairs, key=lambda pair: (abs(float(_pair_metric(pair) or 0.0)), pair["pair_id"])))

        negative_pairs = [pair for pair in successful_pairs if float(_pair_metric(pair) or 0.0) < 0.0]
        if negative_pairs:
            selected.append(min(negative_pairs, key=lambda pair: (float(_pair_metric(pair) or 0.0), pair["pair_id"])))
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in selected:
        pair_id = str(pair.get("pair_id") or "")
        if pair_id in seen:
            continue
        seen.add(pair_id)
        deduped.append(pair)
    return deduped


def _requested_attention_backend(case_payload: dict[str, Any], run_config: dict[str, Any]) -> str:
    requested = (case_payload.get("attention_backend") or {}).get("requested")
    fallback = run_config.get("attention_backend")
    return str(fallback if requested is None else requested)


def _requested_moe_runtime(case_payload: dict[str, Any], run_config: dict[str, Any]) -> tuple[bool, str, bool]:
    moe_runtime = case_payload.get("moe_runtime") or {}
    grouped_gemm = (moe_runtime.get("grouped_gemm") or {}).get("requested")
    token_dispatcher = (moe_runtime.get("token_dispatcher_type") or {}).get("requested")
    overlap_comm = (moe_runtime.get("overlap_expert_parallel_comm") or {}).get("requested")
    return (
        bool(run_config.get("moe_grouped_gemm") if grouped_gemm is None else grouped_gemm),
        str(run_config.get("moe_token_dispatcher_type") if token_dispatcher is None else token_dispatcher),
        bool(run_config.get("overlap_moe_expert_parallel_comm") if overlap_comm is None else overlap_comm),
    )


def _requested_moe_routing_mode(case_payload: dict[str, Any], run_config: dict[str, Any]) -> str:
    requested = case_payload.get("moe_routing_mode")
    fallback = run_config.get("moe_routing_mode", "normal")
    return normalize_moe_routing_mode(str(fallback if requested is None else requested))


def _requested_green_ctx_sms(case_payload: dict[str, Any]) -> dict[str, int | None]:
    runtime = case_payload.get("runtime") or {}
    return dict(runtime.get("requested_sms_by_role") or {"attn": None, "moe": None})


def _case_topology(case_payload: dict[str, Any], run_config: dict[str, Any]) -> dict[str, Any]:
    topology = dict(run_config.get("topology") or {})
    topology.update(case_payload.get("topology") or {})
    return {
        "attn_dp_size": int(topology["attn_dp_size"]),
        "moe_ep_size": int(topology["moe_ep_size"]),
        "attn_gpu_ids": [int(value) for value in topology["attn_gpu_ids"]],
        "moe_gpu_ids": [int(value) for value in topology["moe_gpu_ids"]],
    }


def _build_case_rerun_command(
    *,
    run_config: dict[str, Any],
    case_payload: dict[str, Any],
    output_dir: Path,
    capture_nsys: str,
    capture_torch_profiler: str,
    warmup_iters: int,
    timed_iters: int,
    profiler_trace_dir: Path | None = None,
    profiler_wait_iters: int | None = None,
    profiler_active_iters: int | None = None,
) -> list[str]:
    topology = _case_topology(case_payload, run_config)
    grouped_gemm, token_dispatcher, overlap_comm = _requested_moe_runtime(case_payload, run_config)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--model-name",
        str(run_config["model_name"]),
        "--model-type",
        str(run_config["model_type"]),
        "--attn-gpu-ids",
        ",".join(str(gpu_id) for gpu_id in topology["attn_gpu_ids"]),
        "--moe-gpu-ids",
        ",".join(str(gpu_id) for gpu_id in topology["moe_gpu_ids"]),
        "--attn-dp-size",
        str(topology["attn_dp_size"]),
        "--moe-ep-size",
        str(topology["moe_ep_size"]),
        "--seq-lens",
        str(case_payload["seq_len"]),
        "--dtypes",
        str(case_payload["dtype"]),
        "--runtime-backends",
        str(case_payload["runtime_backend"]),
        "--single-mode",
        str(case_payload["mode"]),
        "--seed",
        str(case_payload["seed"]),
        "--batch-size",
        str(case_payload["batch_size"]),
        "--warmup-iters",
        str(warmup_iters),
        "--timed-iters",
        str(timed_iters),
        "--worker-timeout-s",
        str(run_config.get("worker_timeout_s", 600.0)),
        "--output-dir",
        str(output_dir),
        "--capture-nsys",
        capture_nsys,
        "--capture-torch-profiler",
        capture_torch_profiler,
        "--attention-backend",
        _requested_attention_backend(case_payload, run_config),
        "--moe-token-dispatcher-type",
        token_dispatcher,
        "--moe-routing-mode",
        _requested_moe_routing_mode(case_payload, run_config),
        "--torch-compile",
        str(run_config.get("torch_compile", "off")),
        "--rerun-existing",
    ]
    if run_config.get("num_experts") is not None:
        cmd.extend(["--num-experts", str(run_config["num_experts"])])
    if run_config.get("mps_active_thread_pct") is not None:
        cmd.extend(["--mps-active-thread-pct", str(run_config["mps_active_thread_pct"])])
    if run_config.get("attn_mps_active_thread_pct") is not None:
        cmd.extend(["--attn-mps-active-thread-pct", str(run_config["attn_mps_active_thread_pct"])])
    _append_nccl_tuple_args(cmd, str((case_payload.get("nccl") or {}).get("tuple") or "off"))
    if grouped_gemm:
        cmd.append("--moe-grouped-gemm")
    if overlap_comm:
        cmd.append("--overlap-moe-expert-parallel-comm")
    if case_payload.get("runtime_backend") == "mps_green_ctx":
        green_ctx_sms = _requested_green_ctx_sms(case_payload)
        if green_ctx_sms.get("attn") is not None:
            cmd.extend(["--green-ctx-attn-sms", str(green_ctx_sms["attn"])])
        if green_ctx_sms.get("moe") is not None:
            cmd.extend(["--green-ctx-moe-sms", str(green_ctx_sms["moe"])])
    if profiler_trace_dir is not None:
        cmd.extend(["--torch-profiler-trace-dir", str(profiler_trace_dir)])
        if profiler_wait_iters is not None:
            cmd.extend(["--torch-profiler-wait-iters", str(profiler_wait_iters)])
        if profiler_active_iters is not None:
            cmd.extend(["--torch-profiler-active-iters", str(profiler_active_iters)])
    return cmd


def _load_rerun_case_payload(case_id: str, rerun_output_dir: Path) -> dict[str, Any] | None:
    case_path = rerun_output_dir / "cases" / f"{case_id}.json"
    if case_path.exists():
        return load_case_payload(case_path)
    candidates = sorted((rerun_output_dir / "cases").glob("*.json"))
    if len(candidates) == 1:
        return load_case_payload(candidates[0])
    return None


def _normalize_torch_profiler_status(
    *,
    rerun_payload: dict[str, Any] | None,
    returncode: int,
    trace_files: list[str],
) -> str:
    if rerun_payload is not None:
        status = str(rerun_payload.get("status") or "")
        if status in PROFILER_STATUS_KEYS and status != "ok":
            return status
    if returncode == 0 and trace_files:
        return "ok"
    if rerun_payload is not None:
        status = str(rerun_payload.get("status") or "")
        if status in PROFILER_STATUS_KEYS:
            return status
    return "torch_profiler_capture_failed"


def _select_torch_profiler_cases(cases: list[dict[str, Any]], selection: str) -> list[dict[str, Any]]:
    cases_by_id = {str(case.get("case_id")): case for case in cases if case.get("case_id")}
    if selection == "all-successful":
        return sorted(
            (case for case in cases_by_id.values() if case.get("status") == "ok"),
            key=lambda case: str(case.get("case_id") or ""),
        )

    selected_pairs = _select_profiler_backend_pairs(cases)
    if not selected_pairs:
        return []

    reruns: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for pair in selected_pairs:
        for case_id_key in ("mps_only_overlap_case_id", "mps_green_ctx_overlap_case_id"):
            case_id = pair.get(case_id_key)
            if case_id is None:
                continue
            case_id_text = str(case_id)
            if case_id_text in seen_case_ids:
                continue
            case_payload = cases_by_id.get(case_id_text)
            if case_payload is None or case_payload.get("status") != "ok":
                continue
            seen_case_ids.add(case_id_text)
            reruns.append(case_payload)
    return reruns


def _append_nccl_tuple_args(cmd: list[str], nccl_tuple_token: str) -> None:
    cmd.extend(["--nccl-tuples", nccl_tuple_token])


def _run_nsys_capture(
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    if args.capture_nsys != "on":
        return "off"
    if shutil.which(args.nsys_bin) is None and not Path(args.nsys_bin).exists():
        return "nsys_capture_failed"

    selected = _select_torch_profiler_cases(cases, selection="representative")
    if not selected:
        return "off"
    nsys_dir = Path(args.output_dir) / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for case_payload in selected:
        case_id = str(case_payload["case_id"])
        case_dir = nsys_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        out_prefix = case_dir / "trace"
        rerun_output_dir = case_dir / "rerun_output"
        cmd = [
            args.nsys_bin,
            "profile",
            "--force-overwrite=true",
            "-o",
            str(out_prefix),
            *_build_case_rerun_command(
                run_config=run_config,
                case_payload=case_payload,
                output_dir=rerun_output_dir,
                capture_nsys="off",
                capture_torch_profiler="off",
                warmup_iters=1,
                timed_iters=1,
            ),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        entries.append(
            {
                "runtime_backend": case_payload["runtime_backend"],
                "case_id": case_id,
                "command": cmd,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "output_prefix": str(out_prefix),
            }
        )
        if completed.returncode != 0:
            write_json_atomic(nsys_dir / "trace_index.json", {"status": "nsys_capture_failed", "entries": entries})
            return "nsys_capture_failed"
    write_json_atomic(nsys_dir / "trace_index.json", {"status": "ok", "entries": entries})
    return "ok"
def _run_torch_profiler_capture(
    *,
    args: argparse.Namespace,
    run_config: dict[str, Any],
    cases: list[dict[str, Any]],
) -> str:
    if args.capture_torch_profiler != "on":
        return "off"

    selected = _select_torch_profiler_cases(cases, selection=args.torch_profiler_selection)
    if not selected:
        return "off"

    profiler_root = Path(args.output_dir) / "torch_profiler"
    profiler_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    effective_wait_iters = _effective_torch_profiler_wait_iters(args)
    for case_payload in selected:
        case_id = str(case_payload["case_id"])
        trace_dir = profiler_root / case_id
        rerun_output_dir = profiler_root / "reruns" / case_id
        cmd = _build_case_rerun_command(
            run_config=run_config,
            case_payload=case_payload,
            output_dir=rerun_output_dir,
            capture_nsys="off",
            capture_torch_profiler="off",
            warmup_iters=int(run_config.get("warmup_iters", args.warmup_iters)),
            timed_iters=int(run_config.get("timed_iters", args.timed_iters)),
            profiler_trace_dir=trace_dir,
            profiler_wait_iters=effective_wait_iters,
            profiler_active_iters=int(args.torch_profiler_active_iters),
        )
        completed = subprocess.run(cmd, capture_output=True, text=True)
        trace_files = [str(path) for path in sorted(trace_dir.rglob("*.pt.trace.json"))]
        rerun_payload = _load_rerun_case_payload(case_id, rerun_output_dir)
        profiler_status = _normalize_torch_profiler_status(
            rerun_payload=rerun_payload,
            returncode=completed.returncode,
            trace_files=trace_files,
        )
        entries.append(
            {
                "case_id": case_id,
                "mode": case_payload["mode"],
                "runtime_backend": case_payload["runtime_backend"],
                "seq_len": case_payload["seq_len"],
                "batch_size": case_payload["batch_size"],
                "dtype": case_payload["dtype"],
                "nccl_tuple": (case_payload.get("nccl") or {}).get("tuple"),
                "profiler_status": profiler_status,
                "command": cmd,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "trace_dir": str(trace_dir),
                "trace_files": trace_files,
                "rerun_output_dir": str(rerun_output_dir),
            }
        )

    overall_status = "ok" if all(entry["profiler_status"] == "ok" for entry in entries) else "partial_failure"
    write_json_atomic(
        profiler_root / "trace_index.json",
        {
            "schema_version": TORCH_PROFILER_TRACE_INDEX_SCHEMA_VERSION,
            "status": overall_status,
            "selection": args.torch_profiler_selection,
            "wait_iters": effective_wait_iters,
            "active_iters": int(args.torch_profiler_active_iters),
            "viewer": {
                "type": "tensorboard",
                "logdir": str(profiler_root),
            },
            "entries": entries,
        },
    )
    return "ok" if overall_status == "ok" else "torch_profiler_capture_failed"


def main() -> int:
    args = _parse_args()
    mp.set_start_method("spawn", force=True)
    _bootstrap_local_pythonpath()
    args.moe_routing_mode = normalize_moe_routing_mode(args.moe_routing_mode)

    attn_gpu_ids = parse_gpu_ids(args.attn_gpu_ids, field_name="attn-gpu-ids")
    moe_gpu_ids = parse_gpu_ids(args.moe_gpu_ids, field_name="moe-gpu-ids")
    seq_lens = parse_seq_lens(args.seq_lens)
    batch_sizes = _resolve_batch_sizes(args)
    runtime_backends = _resolve_runtime_backends(args)
    green_ctx_sms = _resolved_green_ctx_sms(args)
    dtype_raw = args.dtypes if args.dtypes is not None else args.dtype
    dtypes = parse_dtypes(dtype_raw)
    nccl_tuples = parse_nccl_tuples(
        nccl_tuples=args.nccl_tuples,
        nccl_socket_nthreads=args.nccl_socket_nthreads,
        nccl_max_nchannels=args.nccl_max_nchannels,
        nccl_max_ctas=args.nccl_max_ctas,
    )
    modes = [args.single_mode] if args.single_mode is not None else ["serial", "overlap"]
    output_dir = Path(args.output_dir)
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)

    topology = {
        "attn_dp_size": int(args.attn_dp_size),
        "moe_ep_size": int(args.moe_ep_size),
        "attn_gpu_ids": attn_gpu_ids,
        "moe_gpu_ids": moe_gpu_ids,
        "seed": int(args.seed),
    }

    preflight_errors, device_total_sms = _preflight_errors(
        model_name=args.model_name,
        model_type=args.model_type,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
        attn_dp_size=args.attn_dp_size,
        moe_ep_size=args.moe_ep_size,
        runtime_backends=runtime_backends,
        green_ctx_sms=green_ctx_sms,
    )
    if args.moe_routing_mode == "equal_tokens":
        if args.num_experts is None or int(args.num_experts) <= 0:
            preflight_errors.append("--moe-routing-mode equal_tokens requires --num-experts > 0")
        elif int(args.num_experts) % int(args.moe_ep_size) != 0:
            preflight_errors.append("--num-experts must be divisible by --moe-ep-size for equal_tokens")
    device_sm_signature = _device_sm_signature(device_total_sms)
    cases = _build_case_descriptors(
        modes=modes,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        runtime_backends=runtime_backends,
        green_ctx_sms=green_ctx_sms,
        dtypes=dtypes,
        nccl_tuples=nccl_tuples,
        seed=args.seed,
        attn_dp_size=args.attn_dp_size,
        moe_ep_size=args.moe_ep_size,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
        moe_routing_mode=args.moe_routing_mode,
    )
    total_points = len(cases)
    profiler_config = _profiler_config_from_args(args)

    all_case_payloads: list[dict[str, Any]] = []
    base_run_config = _build_run_config(
        args=args,
        topology=topology,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        dtypes=dtypes,
        nccl_tuples=nccl_tuples,
        runtime_backends=runtime_backends,
        green_ctx_sms=green_ctx_sms,
        device_sm_signature=device_sm_signature,
        nsys_status="off" if args.capture_nsys == "off" else None,
        torch_profiler_status="off" if args.capture_torch_profiler == "off" else None,
    )

    try:
        _ensure_output_dir_identity_matches(output_dir, base_run_config)
    except RuntimeError as exc:
        preflight_errors.append(str(exc))

    if args.torch_profiler_recovery:
        if args.capture_torch_profiler != "on":
            preflight_errors.append("--torch-profiler-recovery requires --capture-torch-profiler on")
        if args.torch_profiler_selection != "all-successful":
            preflight_errors.append(
                "--torch-profiler-recovery requires --torch-profiler-selection all-successful"
            )
        if preflight_errors:
            print("; ".join(preflight_errors), file=sys.stderr)
            return 1
        try:
            existing_summary = _load_matrix_summary(output_dir)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        persisted_case_payloads = _load_case_payloads(output_dir)
        if not persisted_case_payloads:
            print(f"{output_dir}/cases does not contain any case payloads", file=sys.stderr)
            return 1
        existing_run_config = dict(existing_summary.get("run_config") or {})
        torch_profiler_status = _run_torch_profiler_capture(
            args=args,
            run_config=existing_run_config,
            cases=persisted_case_payloads,
        )
        existing_run_config["capture_torch_profiler"] = args.capture_torch_profiler
        existing_run_config["torch_profiler_selection"] = profiler_config["selection"]
        existing_run_config["torch_profiler_wait_iters"] = profiler_config["wait_iters"]
        existing_run_config["torch_profiler_active_iters"] = profiler_config["active_iters"]
        existing_run_config["torch_profiler_status"] = torch_profiler_status
        summary = build_matrix_summary(
            run_config=existing_run_config,
            cases=persisted_case_payloads,
            total_points=int((existing_summary.get("counts") or {}).get("total_points", len(persisted_case_payloads))),
        )
        write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
        write_matrix_summary_markdown(args.output_dir, summary)
        return 0 if torch_profiler_status == "ok" else 1

    if preflight_errors:
        all_case_payloads = _invalid_env_matrix(
            cases=cases,
            output_dir=args.output_dir,
            strict_schema=args.strict_schema,
            error_message="; ".join(preflight_errors),
            topology=topology,
            profiler=profiler_config,
            torch_compile_requested=args.torch_compile,
        )
        summary = build_matrix_summary(
            run_config=base_run_config,
            cases=all_case_payloads,
            total_points=total_points,
        )
        write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
        write_matrix_summary_markdown(args.output_dir, summary)
        return 1

    serial_baselines: dict[str, dict[str, Any]] = {}
    try:
        with MultiGpuMPSContext(
            gpu_ids=sorted(set(moe_gpu_ids)),
            active_thread_pct=args.mps_active_thread_pct,
        ) as mps_ctx:
            mps_env = mps_ctx.env_vars()
            for descriptor in cases:
                existing_path = case_output_path(args.output_dir, descriptor.case_id)
                if existing_path.exists():
                    existing_payload = load_case_payload(existing_path)
                    if should_skip_existing(existing_payload, args.rerun_existing):
                        all_case_payloads.append(existing_payload)
                        if descriptor.mode == "serial":
                            serial_baselines[descriptor.baseline_key] = existing_payload
                        continue

                nccl_tuple = descriptor.nccl_tuple
                nccl_meta = _nccl_meta(nccl_tuple)
                common_config = {
                    "model_name": args.model_name,
                    "model_type": args.model_type,
                    "runtime_backend": descriptor.runtime_backend,
                    "green_ctx_attn_sms": descriptor.green_ctx_attn_sms,
                    "green_ctx_moe_sms": descriptor.green_ctx_moe_sms,
                    "dtype": descriptor.dtype,
                    "seq_len": descriptor.seq_len,
                    "batch_size": descriptor.batch_size,
                    "seed": args.seed,
                    "warmup_iters": args.warmup_iters,
                    "timed_iters": args.timed_iters,
                    "moe_ep_size": args.moe_ep_size,
                    "num_experts": args.num_experts,
                    "moe_routing_mode": args.moe_routing_mode,
                    "attn_mps_active_thread_pct": args.attn_mps_active_thread_pct,
                    "moe_grouped_gemm": args.moe_grouped_gemm,
                    "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
                    "overlap_moe_expert_parallel_comm": args.overlap_moe_expert_parallel_comm,
                    "attention_backend": args.attention_backend,
                    "nccl_tuple": nccl_tuple,
                    "torch_compile": args.torch_compile,
                    "torch_compile_enabled": args.torch_compile == "on",
                    "profiler_trace_root": args.torch_profiler_trace_dir,
                    "profiler_wait_iters": args.torch_profiler_wait_iters,
                    "profiler_active_timed_iters": args.torch_profiler_active_iters,
                }

                attempt_count = 1
                retry_trigger = "none"
                while True:
                    attempt_result = _run_case_attempt(
                        mode=descriptor.mode,
                        common_config=common_config,
                        attn_gpu_ids=attn_gpu_ids,
                        moe_gpu_ids=moe_gpu_ids,
                        timeout_s=args.worker_timeout_s,
                        mps_env=mps_env,
                    )
                    payload = build_case_payload(
                        case_id=descriptor.case_id,
                        status=attempt_result["status"],
                        mode=descriptor.mode,
                        seq_len=descriptor.seq_len,
                        batch_size=descriptor.batch_size,
                        runtime_backend=descriptor.runtime_backend,
                        dtype=descriptor.dtype,
                        seed=args.seed,
                        topology=topology,
                        nccl_env=nccl_meta,
                        moe_routing_mode=descriptor.moe_routing_mode,
                        runtime=attempt_result["runtime"],
                        timing_ms=attempt_result["timing_ms"],
                        overlap_ms=attempt_result["overlap_ms"],
                        finite=attempt_result["finite"],
                        stage_signatures=attempt_result["stage_signatures"],
                        error=attempt_result["error"],
                        profiler=profiler_config,
                        torch_compile=attempt_result["torch_compile"],
                        attempt_count=attempt_count,
                        retry_trigger=retry_trigger,
                        tokens_per_expert=attempt_result["tokens_per_expert"],
                        tokens_per_expert_min=(
                            None
                            if attempt_result["tokens_per_expert"] is None
                            else min(attempt_result["tokens_per_expert"])
                        ),
                        tokens_per_expert_max=(
                            None
                            if attempt_result["tokens_per_expert"] is None
                            else max(attempt_result["tokens_per_expert"])
                        ),
                        tokens_per_expert_spread=(
                            None
                            if attempt_result["tokens_per_expert"] is None
                            else max(attempt_result["tokens_per_expert"])
                            - min(attempt_result["tokens_per_expert"])
                        ),
                    )
                    payload["attention_backend"] = attempt_result["attention_backend"]
                    payload["attention_impl"] = attempt_result["attention_impl"]
                    payload["moe_runtime"] = attempt_result["moe_runtime"]

                    if should_retry(payload["status"], attempt_count):
                        retry_trigger = payload["status"]
                        attempt_count += 1
                        continue
                    break

                if descriptor.mode == "serial":
                    serial_baselines[descriptor.baseline_key] = payload
                else:
                    payload = _apply_baseline_diff(payload, serial_baselines.get(descriptor.baseline_key))
                    tol = tolerance_for_dtype(payload["dtype"])
                    if payload["baseline_diff"]["all_within_tolerance"] is None and payload["status"] == "ok":
                        payload["baseline_diff"]["stages"] = {
                            "attn": {
                                "stage": "attn",
                                "max_abs_diff": None,
                                "max_rel_diff": None,
                                "eps": tol["eps"],
                                "tolerance": {
                                    "max_abs_diff": tol["max_abs_diff"],
                                    "max_rel_diff": tol["max_rel_diff"],
                                },
                                "within_tolerance": False,
                            },
                            "moe": {
                                "stage": "moe",
                                "max_abs_diff": None,
                                "max_rel_diff": None,
                                "eps": tol["eps"],
                                "tolerance": {
                                    "max_abs_diff": tol["max_abs_diff"],
                                    "max_rel_diff": tol["max_rel_diff"],
                                },
                                "within_tolerance": False,
                            },
                        }

                if args.strict_schema:
                    schema_errors = validate_case_payload(payload)
                    if schema_errors:
                        payload["status"] = "runtime_error"
                        payload["error"] = {
                            "code": "schema_validation_error",
                            "message": "; ".join(schema_errors),
                            "traceback": None,
                        }

                write_case_json(output_dir=args.output_dir, payload=payload, strict_schema=args.strict_schema)
                all_case_payloads.append(payload)
    except Exception as exc:
        error_payloads = _invalid_env_matrix(
            cases=[case for case in cases if case.case_id not in {p["case_id"] for p in all_case_payloads}],
            output_dir=args.output_dir,
            strict_schema=args.strict_schema,
            error_message=f"MPS startup or matrix run failed: {exc}",
            topology=topology,
            profiler=profiler_config,
            torch_compile_requested=args.torch_compile,
        )
        all_case_payloads.extend(error_payloads)
        summary = build_matrix_summary(
            run_config=base_run_config,
            cases=all_case_payloads,
            total_points=total_points,
        )
        write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
        write_matrix_summary_markdown(args.output_dir, summary)
        return 1

    persisted_case_payloads = _load_case_payloads(output_dir)
    nsys_status = _run_nsys_capture(
        args=args,
        run_config=base_run_config,
        cases=persisted_case_payloads,
    )
    torch_profiler_status = _run_torch_profiler_capture(
        args=args,
        run_config=base_run_config,
        cases=persisted_case_payloads,
    )
    run_config = _build_run_config(
        args=args,
        topology=topology,
        seq_lens=seq_lens,
        batch_sizes=batch_sizes,
        dtypes=dtypes,
        nccl_tuples=nccl_tuples,
        runtime_backends=runtime_backends,
        green_ctx_sms=green_ctx_sms,
        device_sm_signature=device_sm_signature,
        nsys_status=nsys_status,
        torch_profiler_status=torch_profiler_status,
    )
    summary = build_matrix_summary(run_config=run_config, cases=persisted_case_payloads, total_points=total_points)
    write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
    write_matrix_summary_markdown(args.output_dir, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
