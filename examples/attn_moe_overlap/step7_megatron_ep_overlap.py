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
    from examples.attn_moe_overlap.megatron_overlap_schema import (
        build_case_id,
        build_case_payload,
        build_invalid_environment_payload,
        build_matrix_summary,
        canonical_nccl_tuple,
        case_output_path,
        compute_host_enqueue_overlap_ms,
        compute_speedup,
        evaluate_stage_diff,
        is_terminal_status,
        normalize_dtype_name,
        parse_dtypes,
        parse_gpu_ids,
        parse_nccl_tuples,
        parse_seq_lens,
        should_retry,
        should_skip_existing,
        tolerance_for_dtype,
        validate_case_payload,
        write_case_json,
        write_json_atomic,
        write_matrix_summary,
        write_matrix_summary_markdown,
        load_case_payload,
    )
except ModuleNotFoundError:
    from megatron_overlap_schema import (  # type: ignore[no-redef]
        build_case_id,
        build_case_payload,
        build_invalid_environment_payload,
        build_matrix_summary,
        canonical_nccl_tuple,
        case_output_path,
        compute_host_enqueue_overlap_ms,
        compute_speedup,
        evaluate_stage_diff,
        is_terminal_status,
        normalize_dtype_name,
        parse_dtypes,
        parse_gpu_ids,
        parse_nccl_tuples,
        parse_seq_lens,
        should_retry,
        should_skip_existing,
        tolerance_for_dtype,
        validate_case_payload,
        write_case_json,
        write_json_atomic,
        write_matrix_summary,
        write_matrix_summary_markdown,
        load_case_payload,
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
    dtype: str
    nccl_tuple: tuple[int, int, int]
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
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=2)
    parser.add_argument("--worker-timeout-s", type=float, default=600.0)
    parser.add_argument("--nccl-tuples", type=str, default=None)
    parser.add_argument("--nccl-socket-nthreads", type=int, default=None)
    parser.add_argument("--nccl-max-nchannels", type=int, default=None)
    parser.add_argument("--nccl-max-ctas", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument("--capture-nsys", choices=["on", "off"], default="off")
    parser.add_argument("--nsys-bin", type=str, default="nsys")
    parser.add_argument("--capture-torch-profiler", choices=["on", "off"], default="off")
    parser.add_argument("--torch-profiler-active-iters", type=int, default=5)
    parser.add_argument("--torch-profiler-trace-dir", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--single-mode", choices=["serial", "overlap"], default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    parser.add_argument("--mps-active-thread-pct", type=int, default=None)
    return parser.parse_args()


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
) -> list[str]:
    errors: list[str] = []
    if not model_name.strip():
        errors.append("model-name must be non-empty")
    if not model_type.strip():
        errors.append("model-type must be non-empty")

    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
        return errors
    cuda_device_count = torch.cuda.device_count()
    if cuda_device_count < 2:
        errors.append(f"Need at least 2 GPUs, found {cuda_device_count}")
        return errors

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

    errors.extend(_check_mps_support_for_devices(sorted(set(attn_gpu_ids + moe_gpu_ids))))
    return errors


def _baseline_key(seq_len: int, dtype: str, nccl_tuple: tuple[int, int, int]) -> str:
    return f"seq={seq_len}|dtype={normalize_dtype_name(dtype)}|nccl={canonical_nccl_tuple(nccl_tuple)}"


def _build_case_descriptors(
    *,
    modes: list[str],
    seq_lens: list[int],
    dtypes: list[str],
    nccl_tuples: list[tuple[int, int, int]],
    seed: int,
    attn_dp_size: int,
    moe_ep_size: int,
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
) -> list[CaseDescriptor]:
    cases: list[CaseDescriptor] = []
    for seq_len in seq_lens:
        for dtype in dtypes:
            for nccl_tuple in nccl_tuples:
                for mode in modes:
                    baseline_key = _baseline_key(seq_len, dtype, nccl_tuple)
                    case_id = build_case_id(
                        mode=mode,
                        seq_len=seq_len,
                        dtype=dtype,
                        seed=seed,
                        attn_dp_size=attn_dp_size,
                        moe_ep_size=moe_ep_size,
                        attn_gpu_ids=attn_gpu_ids,
                        moe_gpu_ids=moe_gpu_ids,
                        nccl_tuple=nccl_tuple,
                    )
                    cases.append(
                        CaseDescriptor(
                            mode=mode,
                            seq_len=seq_len,
                            dtype=dtype,
                            nccl_tuple=nccl_tuple,
                            case_id=case_id,
                            baseline_key=baseline_key,
                        )
                    )
    return cases


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
    dtype: str,
    seq_len: int,
    batch_size: int,
    seed: int,
    warmup_iters: int,
    timed_iters: int,
    moe_ep_size: int,
    num_experts: int | None,
    nccl_tuple: tuple[int, int, int],
    mps_env: dict[str, str],
    iteration_barrier: Any | None,
    profiler_trace_root: str | None,
    profiler_active_timed_iters: int | None,
    worker_result_dir: str,
) -> None:
    try:
        from examples.attn_moe_overlap.megatron_layer_runtime import (
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            cleanup_distributed_state,
            classify_exception,
        )
    except ModuleNotFoundError:
        from megatron_layer_runtime import (  # type: ignore[no-redef]
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            cleanup_distributed_state,
            classify_exception,
        )

    status = "ok"
    payload: dict[str, Any] = {}
    try:
        _bootstrap_local_pythonpath()
        os.environ.update(mps_env)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
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
                dtype=dtype,
                seq_len=seq_len,
                batch_size=batch_size,
                seed=seed,
                expert_model_parallel_size=moe_ep_size if role == "moe" else 1,
                num_experts=num_experts,
            )
        )
        payload = runtime.run_stage(
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            iteration_barrier=iteration_barrier,
            profiler_trace_dir=str(Path(profiler_trace_root)) if profiler_trace_root is not None else None,
            profiler_worker_name=f"{role}_rank{rank}_gpu{gpu_id}",
            profiler_active_timed_iters=profiler_active_timed_iters,
        )
        status = payload.get("status", "runtime_error")
    except Exception as exc:
        status, error = classify_exception(exc)
        payload = {
            "status": status,
            "timing_ms": {"cuda": None, "step_total": None, "timed_wall": None},
            "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "enqueue_windows": [],
            "finite": {"all_finite": False, "first_nonfinite": None},
            "output_signature": None,
            "error": error,
        }
    finally:
        cleanup_distributed_state()
        write_json_atomic(
            _worker_result_path(Path(worker_result_dir), role, rank),
            {
                "role": role,
                "rank": rank,
                "status": status,
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
                        "dtype": common_config["dtype"],
                        "seq_len": common_config["seq_len"],
                        "batch_size": common_config["batch_size"],
                        "seed": common_config["seed"],
                        "warmup_iters": common_config["warmup_iters"],
                        "timed_iters": common_config["timed_iters"],
                        "moe_ep_size": common_config["moe_ep_size"],
                        "num_experts": common_config["num_experts"],
                        "nccl_tuple": common_config["nccl_tuple"],
                        "mps_env": mps_env,
                        "iteration_barrier": iteration_barrier,
                        "profiler_trace_root": common_config.get("profiler_trace_root"),
                        "profiler_active_timed_iters": common_config.get("profiler_active_timed_iters"),
                        "worker_result_dir": str(worker_result_dir),
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
            for process in processes:
                if process.is_alive():
                    process.terminate()

        for process in processes:
            process.join(timeout=3)
            if process.is_alive():
                process.kill()

        results = _load_worker_results(worker_result_dir)
        if len(results) < expected:
            exitcodes = {str(process.pid): process.exitcode for process in processes}
            status = "timeout" if timed_out else "runtime_error"
            return {
                "status": status,
                "error": {
                    "code": status,
                    "message": (
                        f"Missing worker results ({len(results)}/{expected}); exitcodes={exitcodes}"
                    ),
                    "traceback": None,
                },
                "results": results,
            }
        return {"status": "ok", "results": results}
    finally:
        shutil.rmtree(worker_result_dir, ignore_errors=True)


def _aggregate_stage_results(role: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    role_results = sorted((item for item in results if item.get("role") == role), key=lambda item: item["rank"])
    if not role_results:
        return {
            "status": "runtime_error",
            "error": {"code": "runtime_error", "message": f"No results for role={role}", "traceback": None},
            "timing_ms": None,
            "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "enqueue_windows": [],
            "output_signature": None,
            "finite": {"all_finite": False, "first_nonfinite": None},
        }

    failing = next((item for item in role_results if item.get("status") != "ok"), None)
    if failing is not None:
        status = failing.get("status", "runtime_error")
        error = failing.get("error") or {
            "code": status,
            "message": f"{role} rank {failing.get('rank')} failed",
            "traceback": None,
        }
        return {
            "status": status,
            "error": error,
            "timing_ms": None,
            "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "enqueue_windows": [],
            "output_signature": None,
            "finite": {"all_finite": False, "first_nonfinite": None},
        }

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
        "timing_ms": {
            "cuda": (rank0.get("timing_ms") or {}).get("cuda"),
            "step_total": (rank0.get("timing_ms") or {}).get("step_total"),
            "timed_wall": timed_window["duration_ms"],
        },
        "timed_window_s": timed_window,
        "enqueue_windows": rank0.get("enqueue_windows", []),
        "output_signature": rank0.get("output_signature"),
        "finite": {"all_finite": all_finite, "first_nonfinite": first_nonfinite},
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
    case_timed_wall_ms: float | None = None
    if mode == "serial":
        attn_spec, moe_spec = _stage_specs(attn_gpu_ids, moe_gpu_ids)
        attn_launch = _launch_workers(
            stage_specs=[attn_spec],
            common_config=common_config,
            timeout_s=timeout_s,
            mps_env=mps_env,
        )
        if attn_launch["status"] != "ok":
            attn_stage = {
                "status": "timeout",
                "error": attn_launch["error"],
                "timing_ms": None,
                "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
                "enqueue_windows": [],
                "output_signature": None,
                "finite": {"all_finite": False, "first_nonfinite": None},
            }
            moe_stage = {
                "status": "runtime_error",
                "error": {
                    "code": "runtime_error",
                    "message": "Skipped moe stage because attn stage timed out",
                    "traceback": None,
                },
                "timing_ms": None,
                "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
                "enqueue_windows": [],
                "output_signature": None,
                "finite": {"all_finite": False, "first_nonfinite": None},
            }
        else:
            attn_stage = _aggregate_stage_results("attn", attn_launch["results"])
            moe_launch = _launch_workers(
                stage_specs=[moe_spec],
                common_config=common_config,
                timeout_s=timeout_s,
                mps_env=mps_env,
            )
            if moe_launch["status"] != "ok":
                moe_stage = {
                    "status": "timeout",
                    "error": moe_launch["error"],
                    "timing_ms": None,
                    "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
                    "enqueue_windows": [],
                    "output_signature": None,
                    "finite": {"all_finite": False, "first_nonfinite": None},
                }
            else:
                moe_stage = _aggregate_stage_results("moe", moe_launch["results"])
        host_overlap_ms = 0.0
        attn_timed_wall_ms = (attn_stage.get("timing_ms") or {}).get("timed_wall")
        moe_timed_wall_ms = (moe_stage.get("timing_ms") or {}).get("timed_wall")
        if attn_timed_wall_ms is not None and moe_timed_wall_ms is not None:
            case_timed_wall_ms = float(attn_timed_wall_ms) + float(moe_timed_wall_ms)
    else:
        attn_spec, moe_spec = _stage_specs(attn_gpu_ids, moe_gpu_ids)
        overlap_launch = _launch_workers(
            stage_specs=[attn_spec, moe_spec],
            common_config=common_config,
            timeout_s=timeout_s,
            mps_env=mps_env,
        )
        if overlap_launch["status"] != "ok":
            error = overlap_launch["error"]
            attn_stage = {
                "status": "timeout",
                "error": error,
                "timing_ms": None,
                "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
                "enqueue_windows": [],
                "output_signature": None,
                "finite": {"all_finite": False, "first_nonfinite": None},
            }
            moe_stage = {
                "status": "timeout",
                "error": error,
                "timing_ms": None,
                "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
                "enqueue_windows": [],
                "output_signature": None,
                "finite": {"all_finite": False, "first_nonfinite": None},
            }
            host_overlap_ms = 0.0
        else:
            attn_stage = _aggregate_stage_results("attn", overlap_launch["results"])
            moe_stage = _aggregate_stage_results("moe", overlap_launch["results"])
            host_overlap_ms = compute_host_enqueue_overlap_ms(
                attn_stage.get("enqueue_windows", []),
                moe_stage.get("enqueue_windows", []),
            )
            case_timed_window = _collapse_timed_window_s(
                [attn_stage.get("timed_window_s"), moe_stage.get("timed_window_s")]
            )
            case_timed_wall_ms = case_timed_window["duration_ms"]

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
    if status == "ok" and not all_finite:
        status = "runtime_error"
        error = {
            "code": "non_finite",
            "message": "Detected non-finite outputs",
            "traceback": None,
        }

    return {
        "status": status,
        "error": error,
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
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for descriptor in cases:
        nccl_tuple = descriptor.nccl_tuple
        payload = build_invalid_environment_payload(
            case_id=descriptor.case_id,
            mode=descriptor.mode,
            seq_len=descriptor.seq_len,
            dtype=descriptor.dtype,
            seed=topology["seed"],
            topology=topology,
            nccl_env={
                "socket_nthreads": nccl_tuple[0],
                "max_nchannels": nccl_tuple[1],
                "max_ctas": nccl_tuple[2],
                "tuple": canonical_nccl_tuple(nccl_tuple),
            },
            message=error_message,
        )
        write_case_json(output_dir=output_dir, payload=payload, strict_schema=strict_schema)
        payloads.append(payload)
    return payloads


def _run_nsys_capture(
    *,
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
) -> str:
    if args.capture_nsys != "on":
        return "off"
    if shutil.which(args.nsys_bin) is None and not Path(args.nsys_bin).exists():
        return "nsys_capture_failed"

    serial_case = next((row for row in cases if row.get("mode") == "serial" and row.get("status") == "ok"), None)
    overlap_case = next((row for row in cases if row.get("mode") == "overlap" and row.get("status") == "ok"), None)
    if serial_case is None or overlap_case is None:
        return "off"

    selected = [serial_case, overlap_case]
    nsys_dir = Path(args.output_dir) / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for row in selected:
        out_prefix = nsys_dir / row["case_id"]
        cmd = [
            args.nsys_bin,
            "profile",
            "--force-overwrite=true",
            "-o",
            str(out_prefix),
            sys.executable,
            "-m",
            "examples.attn_moe_overlap.step7_megatron_ep_overlap",
            "--model-name",
            args.model_name,
            "--model-type",
            args.model_type,
            "--attn-gpu-ids",
            ",".join(str(gpu_id) for gpu_id in attn_gpu_ids),
            "--moe-gpu-ids",
            ",".join(str(gpu_id) for gpu_id in moe_gpu_ids),
            "--attn-dp-size",
            str(args.attn_dp_size),
            "--moe-ep-size",
            str(args.moe_ep_size),
            "--seq-lens",
            str(row["seq_len"]),
            "--dtypes",
            str(row["dtype"]),
            "--single-mode",
            str(row["mode"]),
            "--nccl-tuples",
            str(row["nccl"]["tuple"]),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--warmup-iters",
            "1",
            "--timed-iters",
            "1",
            "--worker-timeout-s",
            str(args.worker_timeout_s),
            "--output-dir",
            str(nsys_dir / f"rerun_{row['case_id']}"),
            "--capture-nsys",
            "off",
            "--rerun-existing",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        entries.append(
            {
                "case_id": row["case_id"],
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


def _select_torch_profiler_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    serial_case = next((row for row in cases if row.get("mode") == "serial" and row.get("status") == "ok"), None)
    overlap_case = next((row for row in cases if row.get("mode") == "overlap" and row.get("status") == "ok"), None)
    if serial_case is not None:
        selected.append(serial_case)
    if overlap_case is not None:
        selected.append(overlap_case)
    return selected


def _run_torch_profiler_capture(
    *,
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    attn_gpu_ids: list[int],
    moe_gpu_ids: list[int],
) -> str:
    if args.capture_torch_profiler != "on":
        return "off"

    selected = _select_torch_profiler_cases(cases)
    if not selected:
        return "off"

    profiler_root = Path(args.output_dir) / "torch_profiler"
    profiler_root.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    step7_script = str(Path(__file__).resolve())
    for row in selected:
        trace_dir = profiler_root / row["case_id"]
        rerun_output_dir = profiler_root / f"rerun_{row['case_id']}"
        cmd = [
            sys.executable,
            step7_script,
            "--model-name",
            args.model_name,
            "--model-type",
            args.model_type,
            "--attn-gpu-ids",
            ",".join(str(gpu_id) for gpu_id in attn_gpu_ids),
            "--moe-gpu-ids",
            ",".join(str(gpu_id) for gpu_id in moe_gpu_ids),
            "--attn-dp-size",
            str(args.attn_dp_size),
            "--moe-ep-size",
            str(args.moe_ep_size),
            "--seq-lens",
            str(row["seq_len"]),
            "--dtypes",
            str(row["dtype"]),
            "--single-mode",
            str(row["mode"]),
            "--nccl-tuples",
            str(row["nccl"]["tuple"]),
            "--seed",
            str(args.seed),
            "--batch-size",
            str(args.batch_size),
            "--warmup-iters",
            str(args.warmup_iters),
            "--timed-iters",
            str(args.timed_iters),
            "--worker-timeout-s",
            str(args.worker_timeout_s),
            "--output-dir",
            str(rerun_output_dir),
            "--capture-nsys",
            "off",
            "--capture-torch-profiler",
            "off",
            "--torch-profiler-trace-dir",
            str(trace_dir),
            "--torch-profiler-active-iters",
            str(args.torch_profiler_active_iters),
            "--rerun-existing",
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        trace_files = [str(path) for path in sorted(trace_dir.rglob("*.pt.trace.json"))]
        entries.append(
            {
                "case_id": row["case_id"],
                "mode": row["mode"],
                "command": cmd,
                "returncode": completed.returncode,
                "stdout_tail": completed.stdout[-2000:],
                "stderr_tail": completed.stderr[-2000:],
                "trace_dir": str(trace_dir),
                "trace_files": trace_files,
                "tensorboard_logdir": str(trace_dir),
                "rerun_output_dir": str(rerun_output_dir),
            }
        )
        if completed.returncode != 0:
            write_json_atomic(
                profiler_root / "trace_index.json",
                {
                    "status": "torch_profiler_capture_failed",
                    "active_timed_iters": int(args.torch_profiler_active_iters),
                    "entries": entries,
                },
            )
            return "torch_profiler_capture_failed"

    write_json_atomic(
        profiler_root / "trace_index.json",
        {
            "status": "ok",
            "active_timed_iters": int(args.torch_profiler_active_iters),
            "viewer": {
                "type": "tensorboard",
                "logdir": str(profiler_root),
            },
            "entries": entries,
        },
    )
    return "ok"


def main() -> int:
    args = _parse_args()
    mp.set_start_method("spawn", force=True)
    _bootstrap_local_pythonpath()

    attn_gpu_ids = parse_gpu_ids(args.attn_gpu_ids, field_name="attn-gpu-ids")
    moe_gpu_ids = parse_gpu_ids(args.moe_gpu_ids, field_name="moe-gpu-ids")
    seq_lens = parse_seq_lens(args.seq_lens)
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

    cases = _build_case_descriptors(
        modes=modes,
        seq_lens=seq_lens,
        dtypes=dtypes,
        nccl_tuples=nccl_tuples,
        seed=args.seed,
        attn_dp_size=args.attn_dp_size,
        moe_ep_size=args.moe_ep_size,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
    )
    total_points = len(cases)

    preflight_errors = _preflight_errors(
        model_name=args.model_name,
        model_type=args.model_type,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
        attn_dp_size=args.attn_dp_size,
        moe_ep_size=args.moe_ep_size,
    )
    all_case_payloads: list[dict[str, Any]] = []
    if preflight_errors:
        all_case_payloads = _invalid_env_matrix(
            cases=cases,
            output_dir=args.output_dir,
            strict_schema=args.strict_schema,
            error_message="; ".join(preflight_errors),
            topology=topology,
        )
        summary = build_matrix_summary(
            run_config={
                "model_name": args.model_name,
                "model_type": args.model_type,
                "topology": topology,
                "seq_lens": seq_lens,
                "dtypes": dtypes,
                "nccl_tuples": [canonical_nccl_tuple(tpl) for tpl in nccl_tuples],
                "capture_nsys": args.capture_nsys,
            },
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
                nccl_meta = {
                    "socket_nthreads": nccl_tuple[0],
                    "max_nchannels": nccl_tuple[1],
                    "max_ctas": nccl_tuple[2],
                    "tuple": canonical_nccl_tuple(nccl_tuple),
                }
                common_config = {
                    "model_name": args.model_name,
                    "model_type": args.model_type,
                    "dtype": descriptor.dtype,
                    "seq_len": descriptor.seq_len,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "warmup_iters": args.warmup_iters,
                    "timed_iters": args.timed_iters,
                    "moe_ep_size": args.moe_ep_size,
                    "num_experts": args.num_experts,
                    "nccl_tuple": nccl_tuple,
                    "profiler_trace_root": args.torch_profiler_trace_dir,
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
                        dtype=descriptor.dtype,
                        seed=args.seed,
                        topology=topology,
                        nccl_env=nccl_meta,
                        timing_ms=attempt_result["timing_ms"],
                        overlap_ms=attempt_result["overlap_ms"],
                        finite=attempt_result["finite"],
                        stage_signatures=attempt_result["stage_signatures"],
                        error=attempt_result["error"],
                        attempt_count=attempt_count,
                        retry_trigger=retry_trigger,
                    )

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
        )
        all_case_payloads.extend(error_payloads)
        summary = build_matrix_summary(
            run_config={
                "model_name": args.model_name,
                "model_type": args.model_type,
                "topology": topology,
                "seq_lens": seq_lens,
                "dtypes": dtypes,
                "nccl_tuples": [canonical_nccl_tuple(tpl) for tpl in nccl_tuples],
                "capture_nsys": args.capture_nsys,
            },
            cases=all_case_payloads,
            total_points=total_points,
        )
        write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
        write_matrix_summary_markdown(args.output_dir, summary)
        return 1

    nsys_status = _run_nsys_capture(
        args=args,
        cases=all_case_payloads,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
    )
    torch_profiler_status = _run_torch_profiler_capture(
        args=args,
        cases=all_case_payloads,
        attn_gpu_ids=attn_gpu_ids,
        moe_gpu_ids=moe_gpu_ids,
    )
    if nsys_status == "nsys_capture_failed":
        failed_case = build_case_payload(
            case_id="nsys_capture_failed",
            status="nsys_capture_failed",
            mode="overlap",
            seq_len=seq_lens[0],
            dtype=dtypes[0],
            seed=args.seed,
            topology=topology,
            nccl_env={
                "socket_nthreads": nccl_tuples[0][0],
                "max_nchannels": nccl_tuples[0][1],
                "max_ctas": nccl_tuples[0][2],
                "tuple": canonical_nccl_tuple(nccl_tuples[0]),
            },
            error={
                "code": "nsys_capture_failed",
                "message": "Nsight Systems capture failed for representative cases",
                "traceback": None,
            },
            attempt_count=1,
            retry_trigger="none",
        )
        write_case_json(output_dir=args.output_dir, payload=failed_case, strict_schema=args.strict_schema)
        all_case_payloads.append(failed_case)

    run_config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "topology": topology,
        "seq_lens": seq_lens,
        "dtypes": dtypes,
        "nccl_tuples": [canonical_nccl_tuple(tpl) for tpl in nccl_tuples],
        "modes": modes,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "batch_size": args.batch_size,
        "capture_nsys": args.capture_nsys,
        "nsys_status": nsys_status,
        "capture_torch_profiler": args.capture_torch_profiler,
        "torch_profiler_active_iters": args.torch_profiler_active_iters,
        "torch_profiler_status": torch_profiler_status,
    }
    summary = build_matrix_summary(run_config=run_config, cases=all_case_payloads, total_points=total_points)
    write_matrix_summary(output_dir=args.output_dir, summary=summary, strict_schema=args.strict_schema)
    write_matrix_summary_markdown(args.output_dir, summary)

    if nsys_status == "nsys_capture_failed":
        return 1
    if torch_profiler_status == "torch_profiler_capture_failed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
