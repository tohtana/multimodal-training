"""Step 8: isolated attention/MoE forward sweep across 8-GPU hardware targets."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
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


_early_bootstrap_local_pythonpath()

try:
    from examples.attn_moe_overlap.module_gpu_sweep_schema import (
        build_case_id,
        build_case_payload,
        build_config_fingerprint,
        build_module_summary,
        build_status_counts,
        normalize_dtype_name,
        normalize_stage_role,
        parse_batch_sizes,
        parse_dtypes,
        parse_gpu_ids,
        parse_seq_lens,
        relative_case_path,
        VALID_STATUS_KEYS,
        write_case_json,
        write_module_summary,
    )
except ModuleNotFoundError:
    from module_gpu_sweep_schema import (  # type: ignore[no-redef]
        build_case_id,
        build_case_payload,
        build_config_fingerprint,
        build_module_summary,
        build_status_counts,
        normalize_dtype_name,
        normalize_stage_role,
        parse_batch_sizes,
        parse_dtypes,
        parse_gpu_ids,
        parse_seq_lens,
        relative_case_path,
        VALID_STATUS_KEYS,
        write_case_json,
        write_module_summary,
    )


def _bootstrap_local_pythonpath() -> list[str]:
    return _early_bootstrap_local_pythonpath()


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _worker_result_path(worker_result_dir: Path, rank: int) -> Path:
    return worker_result_dir / f"rank{rank}.json"


def _load_worker_results(worker_result_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for path in sorted(worker_result_dir.glob("*.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


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
            "message": f"Worker rank {root_cause.get('rank')} failed",
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


def _attempt_id_default() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


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


def _gpu_detected_name(gpu_id: int) -> str:
    return str(torch.cuda.get_device_name(gpu_id))


def _gpu_capability(gpu_id: int) -> tuple[int, int]:
    capability = torch.cuda.get_device_capability(gpu_id)
    return int(capability[0]), int(capability[1])


def _dtype_support_error(gpu_ids: list[int], dtypes: list[str]) -> str | None:
    normalized_dtypes = [normalize_dtype_name(dtype) for dtype in dtypes]
    if "bf16" not in normalized_dtypes:
        return None
    unsupported = [gpu_id for gpu_id in gpu_ids if _gpu_capability(gpu_id)[0] < 8]
    if not unsupported:
        return None
    unsupported_text = ",".join(str(value) for value in unsupported)
    return f"bf16 requires compute capability >= 8.0; unsupported gpu_ids={unsupported_text}"


@dataclass(frozen=True)
class CaseDescriptor:
    seq_len: int
    batch_size: int
    dtype: str
    case_id: str


def _build_case_descriptors(
    *,
    stage_role: str,
    seq_lens: list[int],
    batch_sizes: list[int],
    dtypes: list[str],
    seed: int,
    world_size: int,
) -> list[CaseDescriptor]:
    cases: list[CaseDescriptor] = []
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            for dtype in dtypes:
                cases.append(
                    CaseDescriptor(
                        seq_len=seq_len,
                        batch_size=batch_size,
                        dtype=dtype,
                        case_id=build_case_id(
                            stage_role=stage_role,
                            seq_len=seq_len,
                            batch_size=batch_size,
                            dtype=dtype,
                            seed=seed,
                            world_size=world_size,
                        ),
                    )
                )
    return cases


def _worker_main(
    *,
    stage_role: str,
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
    attention_backend: str,
    moe_grouped_gemm: bool,
    moe_token_dispatcher_type: str,
    moe_routing_mode: str,
    num_experts: int | None,
    worker_result_dir: str,
) -> None:
    try:
        from examples.attn_moe_overlap.megatron_layer_runtime import (
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            classify_exception,
            cleanup_distributed_state,
        )
    except ModuleNotFoundError:
        from megatron_layer_runtime import (  # type: ignore[no-redef]
            RuntimeConfig,
            MegatronSingleLayerRuntime,
            classify_exception,
            cleanup_distributed_state,
        )

    status = "ok"
    failure_origin = False
    device_name: str | None = None
    device_capability: list[int] | None = None
    payload: dict[str, Any] = {}
    try:
        _bootstrap_local_pythonpath()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["LOCAL_RANK"] = "0"
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
        for env_key in (
            "CUDA_MPS_PIPE_DIRECTORY",
            "CUDA_MPS_LOG_DIRECTORY",
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            "NCCL_SOCKET_NTHREADS",
            "NCCL_MAX_NCHANNELS",
            "NCCL_MAX_CTAS",
        ):
            os.environ.pop(env_key, None)

        torch.cuda.set_device(0)
        device_name = str(torch.cuda.get_device_name(0))
        device_capability = [int(value) for value in torch.cuda.get_device_capability(0)]

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
                num_experts=num_experts,
                moe_routing_mode=moe_routing_mode,
            )
        )
        payload = runtime.run_stage(
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            execution_schedule="overlap",
        )
        status = payload.get("status", "runtime_error")
    except Exception as exc:
        status, error = classify_exception(exc)
        failure_origin = True
        payload = {
            "status": status,
            "attention_backend": attention_backend,
            "attention_impl": None,
            "moe_grouped_gemm": moe_grouped_gemm,
            "moe_token_dispatcher_type": moe_token_dispatcher_type,
            "runtime": {"requested_sms": None, "granted_sms": None, "device_total_sms": None},
            "timing_ms": {"cuda": None, "step_total": None, "timed_wall": None},
            "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "schedule_timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
            "enqueue_windows": [],
            "finite": {"all_finite": False, "first_nonfinite": None},
            "output_signature": None,
            "tokens_per_expert": None,
            "local_tokens_per_expert": None,
            "error": error,
        }
    finally:
        cleanup_distributed_state()
        result_payload = {
            "rank": rank,
            "status": status,
            "failure_origin": failure_origin,
            "device_name": device_name,
            "device_capability": device_capability,
            **payload,
        }
        result_path = _worker_result_path(Path(worker_result_dir), rank)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _launch_workers(
    *,
    stage_role: str,
    gpu_ids: list[int],
    common_config: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    _bootstrap_local_pythonpath()
    processes: list[mp.Process] = []
    worker_result_dir = Path(tempfile.mkdtemp(prefix="step8_worker_results_"))
    expected = len(gpu_ids)
    master_port = _find_free_port()
    try:
        for rank, gpu_id in enumerate(gpu_ids):
            process = mp.Process(
                target=_worker_main,
                kwargs={
                    "stage_role": stage_role,
                    "rank": rank,
                    "world_size": expected,
                    "gpu_id": gpu_id,
                    "master_addr": "127.0.0.1",
                    "master_port": master_port,
                    "worker_result_dir": str(worker_result_dir),
                    **common_config,
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
            }
        return {"status": "ok", "error": None, "results": results}
    finally:
        for path in worker_result_dir.glob("*.json"):
            path.unlink(missing_ok=True)
        worker_result_dir.rmdir()


def _empty_stage_result(
    *,
    stage_role: str,
    status: str,
    error: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error": error or {"code": status, "message": f"No results for stage_role={stage_role}", "traceback": None},
        "attention_backend": None,
        "attention_impl": None,
        "timing_ms": {"cuda": None, "step_total": None, "timed_wall_total": None},
        "timed_window_s": {"start_s": None, "end_s": None, "duration_ms": None},
        "runtime": {
            "detected_gpu_name": None,
            "device_names_by_rank": [],
            "device_capabilities_by_rank": [],
            "runtime_by_rank": [],
        },
        "finite": {"all_finite": False, "first_nonfinite": None},
        "tokens_per_expert": None,
    }


def _aggregate_stage_results(
    stage_role: str,
    results: list[dict[str, Any]],
    *,
    expected_world_size: int,
    fallback_status: str | None = None,
    fallback_error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    role_results = sorted(results, key=lambda item: int(item.get("rank", 0)))
    runtime = {
        "detected_gpu_name": None if not role_results else role_results[0].get("device_name"),
        "device_names_by_rank": [item.get("device_name") for item in role_results],
        "device_capabilities_by_rank": [item.get("device_capability") for item in role_results],
        "runtime_by_rank": [item.get("runtime") for item in role_results],
    }
    if not role_results:
        return _empty_stage_result(stage_role=stage_role, status=fallback_status or "runtime_error", error=fallback_error)

    if len(role_results) < int(expected_world_size):
        failing = _select_root_cause_worker_result(role_results)
        if failing is not None:
            status = str(failing.get("status") or "runtime_error")
            error = failing.get("error") or {
                "code": status,
                "message": f"rank {failing.get('rank')} failed",
                "traceback": None,
            }
            return _empty_stage_result(stage_role=stage_role, status=status, error=error)
        return _empty_stage_result(stage_role=stage_role, status=fallback_status or "runtime_error", error=fallback_error)

    failing = _select_root_cause_worker_result(role_results)
    if failing is not None:
        status = str(failing.get("status") or "runtime_error")
        error = failing.get("error") or {
            "code": status,
            "message": f"rank {failing.get('rank')} failed",
            "traceback": None,
        }
        return _empty_stage_result(stage_role=stage_role, status=status, error=error)

    timed_window = _collapse_timed_window_s([item.get("timed_window_s") for item in role_results])
    first_nonfinite = None
    all_finite = True
    for result in role_results:
        finite = result.get("finite") or {}
        if not finite.get("all_finite", False):
            all_finite = False
            if first_nonfinite is None:
                first_nonfinite = finite.get("first_nonfinite")

    rank0 = role_results[0]
    tokens_per_expert: list[int] | None = None
    if stage_role == "moe":
        local_vectors = [
            item.get("local_tokens_per_expert")
            for item in role_results
            if item.get("local_tokens_per_expert") is not None
        ]
        if local_vectors:
            lengths = {len(vector) for vector in local_vectors}
            if len(lengths) == 1:
                tokens_per_expert = [0 for _ in range(lengths.pop())]
                for vector in local_vectors:
                    for index, value in enumerate(vector):
                        tokens_per_expert[index] += int(value)

    return {
        "status": "ok",
        "error": {"code": None, "message": None, "traceback": None},
        "attention_backend": rank0.get("attention_backend"),
        "attention_impl": rank0.get("attention_impl"),
        "timing_ms": {
            "cuda": (rank0.get("timing_ms") or {}).get("cuda"),
            "step_total": (rank0.get("timing_ms") or {}).get("step_total"),
            "timed_wall_total": timed_window.get("duration_ms"),
        },
        "timed_window_s": timed_window,
        "runtime": runtime,
        "finite": {"all_finite": all_finite, "first_nonfinite": first_nonfinite},
        "tokens_per_expert": tokens_per_expert,
    }


def _run_case_attempt(
    *,
    descriptor: CaseDescriptor,
    stage_role: str,
    gpu_ids: list[int],
    timeout_s: float,
    common_config: dict[str, Any],
    timed_iters: int,
    moe_routing_mode: str,
) -> dict[str, Any]:
    launch_result = _launch_workers(
        stage_role=stage_role,
        gpu_ids=gpu_ids,
        common_config={
            **common_config,
            "dtype": descriptor.dtype,
            "seq_len": descriptor.seq_len,
            "batch_size": descriptor.batch_size,
        },
        timeout_s=timeout_s,
    )
    fallback_status = launch_result.get("status")
    fallback_error = launch_result.get("error")
    if fallback_status == "ok":
        fallback_status = None
        fallback_error = None
    stage_result = _aggregate_stage_results(
        stage_role,
        launch_result.get("results", []),
        expected_world_size=len(gpu_ids),
        fallback_status=fallback_status,
        fallback_error=fallback_error,
    )

    status = str(stage_result.get("status") or "runtime_error")
    error = stage_result.get("error") or {"code": status, "message": status, "traceback": None}
    all_finite = bool((stage_result.get("finite") or {}).get("all_finite", False))
    first_nonfinite = (stage_result.get("finite") or {}).get("first_nonfinite")
    if status == "ok" and not all_finite and moe_routing_mode != "equal_tokens":
        status = "runtime_error"
        error = {
            "code": "non_finite",
            "message": "Detected non-finite outputs",
            "traceback": None,
        }

    timed_wall_total = (stage_result.get("timing_ms") or {}).get("timed_wall_total")
    forward_pass_ms = None
    if status == "ok" and timed_wall_total is not None and timed_iters > 0:
        forward_pass_ms = float(timed_wall_total) / float(timed_iters)
    module_timing_ms = None
    if status == "ok" and (stage_result.get("timing_ms") or {}).get("cuda") is not None:
        module_timing_ms = float(stage_result["timing_ms"]["cuda"])

    tokens_per_iter = int(descriptor.seq_len) * int(descriptor.batch_size)
    tokens_per_second = None
    if status == "ok" and forward_pass_ms is not None and forward_pass_ms > 0:
        tokens_per_second = tokens_per_iter / (forward_pass_ms / 1000.0)

    status_reason = error.get("message") if error else None
    if status == "ok":
        status_reason = None

    return {
        "status": status,
        "status_reason": status_reason,
        "error": error,
        "runtime": {
            **(stage_result.get("runtime") or {}),
            "attention_impl": stage_result.get("attention_impl"),
            "first_nonfinite": first_nonfinite,
        },
        "timing_ms": {
            "timed_wall_total": timed_wall_total,
            "timed_wall_per_iter": forward_pass_ms,
            "module_cuda": module_timing_ms,
            "module_step_total": (stage_result.get("timing_ms") or {}).get("step_total"),
        },
        "forward_pass_ms": forward_pass_ms,
        "module_timing_ms": module_timing_ms,
        "tokens_per_iter": tokens_per_iter,
        "tokens_per_second": tokens_per_second,
        "tokens_per_expert": stage_result.get("tokens_per_expert"),
    }


def _write_handled_preflight_matrix(
    *,
    status: str,
    message: str,
    output_dir: Path,
    cases: list[CaseDescriptor],
    args: argparse.Namespace,
    superproject_commit: str | None,
    multimodal_training_commit: str | None,
    detected_gpu_name: str | None,
) -> int:
    payloads: list[dict[str, Any]] = []
    for descriptor in cases:
        payload = build_case_payload(
            case_id=descriptor.case_id,
            attempt_id=args.attempt_id,
            module=args.stage_role,
            gpu_type=args.gpu_type,
            seq_len=descriptor.seq_len,
            batch_size=descriptor.batch_size,
            dtype=descriptor.dtype,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            seed=args.seed,
            gpu_ids=args._parsed_gpu_ids,
            world_size=len(args._parsed_gpu_ids),
            model_name=args.model_name,
            model_type=args.model_type,
            status=status,
            status_reason=message,
            attention_backend=args.attention_backend,
            moe_grouped_gemm=args.moe_grouped_gemm,
            moe_token_dispatcher_type=args.moe_token_dispatcher_type,
            moe_routing_mode=args.moe_routing_mode,
            num_experts=args.num_experts,
            tokens_per_iter=int(descriptor.seq_len) * int(descriptor.batch_size),
            pricing_status=None,
            superproject_commit=superproject_commit,
            multimodal_training_commit=multimodal_training_commit,
        )
        write_case_json(output_dir, payload)
        payloads.append(payload)

    summary = build_module_summary(
        stage_role=args.stage_role,
        attempt_id=args.attempt_id,
        command=sys.argv,
        gpu_type=args.gpu_type,
        detected_gpu_name=detected_gpu_name,
        superproject_commit=superproject_commit,
        multimodal_training_commit=multimodal_training_commit,
        case_paths=[relative_case_path(payload["case_id"]) for payload in payloads],
        status_counts=build_status_counts(payloads),
        model_name=args.model_name,
        model_type=args.model_type,
        gpu_ids=args._parsed_gpu_ids,
        seq_lens=args._parsed_seq_lens,
        batch_sizes=args._parsed_batch_sizes,
        dtypes=args._parsed_dtypes,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        attention_backend=args.attention_backend,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        moe_routing_mode=args.moe_routing_mode,
        num_experts=args.num_experts,
    )
    write_module_summary(output_dir, summary)
    return 0 if status in {"unsupported_dtype", "unavailable_hardware"} else 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Step-8 isolated attention/MoE sweep.")
    parser.add_argument("--stage-role", required=True, choices=["attn", "moe"])
    parser.add_argument("--gpu-type", required=True, help="Display GPU type label for the produced artifacts.")
    parser.add_argument("--gpu-ids", required=True, help="Comma-separated GPU ids. The benchmark contract expects 8.")
    parser.add_argument("--seq-lens", default="1024,2048,4096,8192,16384,32768")
    parser.add_argument("--batch-sizes", default="1,2")
    parser.add_argument("--dtypes", default="bf16")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--timed-iters", type=int, default=100)
    parser.add_argument("--worker-timeout-s", type=float, default=1800.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--attempt-id", default=_attempt_id_default())
    parser.add_argument("--model-name", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--model-type", default="qwen3_moe")
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument("--moe-grouped-gemm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-token-dispatcher-type", default="alltoall")
    parser.add_argument("--moe-routing-mode", default="equal_tokens", choices=["normal", "equal_tokens"])
    parser.add_argument("--num-experts", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mp.set_start_method("spawn", force=True)
    _bootstrap_local_pythonpath()

    args.stage_role = normalize_stage_role(args.stage_role)
    args._parsed_gpu_ids = parse_gpu_ids(args.gpu_ids)
    args._parsed_seq_lens = parse_seq_lens(args.seq_lens)
    args._parsed_batch_sizes = parse_batch_sizes(args.batch_sizes)
    args._parsed_dtypes = parse_dtypes(args.dtypes)

    if len(args._parsed_gpu_ids) != 8:
        print("Step-8 requires exactly 8 gpu-ids for the canonical benchmark contract", file=sys.stderr)
        return 1
    if not torch.cuda.is_available():
        print("CUDA is required for step8_module_gpu_sweep.py", file=sys.stderr)
        return 1
    if args.stage_role == "moe":
        if args.num_experts is None or int(args.num_experts) <= 0:
            print("--num-experts is required when --stage-role=moe", file=sys.stderr)
            return 1
        if int(args.num_experts) % len(args._parsed_gpu_ids) != 0:
            print("--num-experts must be divisible by the 8-GPU world size", file=sys.stderr)
            return 1

    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    (output_dir / "cases").mkdir(parents=True, exist_ok=True)

    superproject_commit = _repo_commit(project_root)
    multimodal_training_commit = _repo_commit(project_root, "multimodal-training")
    detected_gpu_name = _gpu_detected_name(args._parsed_gpu_ids[0])

    cases = _build_case_descriptors(
        stage_role=args.stage_role,
        seq_lens=args._parsed_seq_lens,
        batch_sizes=args._parsed_batch_sizes,
        dtypes=args._parsed_dtypes,
        seed=args.seed,
        world_size=len(args._parsed_gpu_ids),
    )

    dtype_error = _dtype_support_error(args._parsed_gpu_ids, args._parsed_dtypes)
    if dtype_error is not None:
        return _write_handled_preflight_matrix(
            status="unsupported_dtype",
            message=dtype_error,
            output_dir=output_dir,
            cases=cases,
            args=args,
            superproject_commit=superproject_commit,
            multimodal_training_commit=multimodal_training_commit,
            detected_gpu_name=detected_gpu_name,
        )

    common_config = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "seed": args.seed,
        "warmup_iters": args.warmup_iters,
        "timed_iters": args.timed_iters,
        "attention_backend": args.attention_backend,
        "moe_grouped_gemm": args.moe_grouped_gemm,
        "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
        "moe_routing_mode": args.moe_routing_mode,
        "num_experts": args.num_experts,
    }

    payloads: list[dict[str, Any]] = []
    for descriptor in cases:
        attempt = _run_case_attempt(
            descriptor=descriptor,
            stage_role=args.stage_role,
            gpu_ids=args._parsed_gpu_ids,
            timeout_s=float(args.worker_timeout_s),
            common_config=common_config,
            timed_iters=int(args.timed_iters),
            moe_routing_mode=str(args.moe_routing_mode),
        )
        payload = build_case_payload(
            case_id=descriptor.case_id,
            attempt_id=args.attempt_id,
            module=args.stage_role,
            gpu_type=args.gpu_type,
            seq_len=descriptor.seq_len,
            batch_size=descriptor.batch_size,
            dtype=descriptor.dtype,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            seed=args.seed,
            gpu_ids=args._parsed_gpu_ids,
            world_size=len(args._parsed_gpu_ids),
            model_name=args.model_name,
            model_type=args.model_type,
            status=attempt["status"],
            status_reason=attempt["status_reason"],
            attention_backend=args.attention_backend,
            moe_grouped_gemm=args.moe_grouped_gemm,
            moe_token_dispatcher_type=args.moe_token_dispatcher_type,
            moe_routing_mode=args.moe_routing_mode,
            num_experts=args.num_experts,
            forward_pass_ms=attempt["forward_pass_ms"],
            module_timing_ms=attempt["module_timing_ms"],
            tokens_per_iter=attempt["tokens_per_iter"],
            tokens_per_second=attempt["tokens_per_second"],
            timing_ms=attempt["timing_ms"],
            runtime=attempt["runtime"],
            error=attempt["error"],
            superproject_commit=superproject_commit,
            multimodal_training_commit=multimodal_training_commit,
            tokens_per_expert=attempt["tokens_per_expert"],
        )
        write_case_json(output_dir, payload)
        payloads.append(payload)

    summary = build_module_summary(
        stage_role=args.stage_role,
        attempt_id=args.attempt_id,
        command=sys.argv,
        gpu_type=args.gpu_type,
        detected_gpu_name=detected_gpu_name,
        superproject_commit=superproject_commit,
        multimodal_training_commit=multimodal_training_commit,
        case_paths=[relative_case_path(payload["case_id"]) for payload in payloads],
        status_counts=build_status_counts(payloads),
        model_name=args.model_name,
        model_type=args.model_type,
        gpu_ids=args._parsed_gpu_ids,
        seq_lens=args._parsed_seq_lens,
        batch_sizes=args._parsed_batch_sizes,
        dtypes=args._parsed_dtypes,
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        attention_backend=args.attention_backend,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_token_dispatcher_type=args.moe_token_dispatcher_type,
        moe_routing_mode=args.moe_routing_mode,
        num_experts=args.num_experts,
        config_fingerprint=build_config_fingerprint(
            {
                "stage_role": args.stage_role,
                "gpu_type": args.gpu_type,
                "gpu_ids": args._parsed_gpu_ids,
                "seq_lens": args._parsed_seq_lens,
                "batch_sizes": args._parsed_batch_sizes,
                "dtypes": args._parsed_dtypes,
                "seed": args.seed,
                "warmup_iters": args.warmup_iters,
                "timed_iters": args.timed_iters,
                "model_name": args.model_name,
                "model_type": args.model_type,
                "attention_backend": args.attention_backend,
                "moe_grouped_gemm": args.moe_grouped_gemm,
                "moe_token_dispatcher_type": args.moe_token_dispatcher_type,
                "moe_routing_mode": args.moe_routing_mode,
                "num_experts": args.num_experts,
            }
        ),
    )
    write_module_summary(output_dir, summary)

    return 0 if all(payload["status"] == "ok" for payload in payloads) else 1


if __name__ == "__main__":
    raise SystemExit(main())
