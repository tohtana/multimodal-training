"""Two-process overlap repro with optional MPS and no CUDA IPC."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import queue
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
)
from examples.attn_moe_overlap.mps_utils import MPSContext
from examples.attn_moe_overlap.repro_overlap_common import (
    build_case_id,
    build_case_payload,
    dtype_from_name,
    ensure_dirs,
    first_nonfinite_in_tensors,
    load_baseline_artifact,
    make_error_payload,
    preflight_checks,
    save_baseline_artifact,
    seed_everything,
    set_mps_thread_pct,
    signature_diff,
    tensor_signature,
    tolerance_for_dtype,
    write_case_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-process overlap NaN repro (no IPC/pipeline)")
    parser.add_argument("--mode", choices=["serial", "parallel"], required=True)
    parser.add_argument("--use-mps", choices=["on", "off"], default="off")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--attn-thread-pct", type=int, default=None)
    parser.add_argument("--moe-thread-pct", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--min-host-overlap-ms", type=float, default=0.25)
    parser.add_argument("--worker-timeout-s", type=float, default=120.0)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--baseline-json", type=str, default=None)
    parser.add_argument("--strict-schema", dest="strict_schema", action="store_true", default=True)
    parser.add_argument("--no-strict-schema", dest="strict_schema", action="store_false")
    return parser.parse_args()


def _validate_thread_pct(name: str, value: int | None) -> None:
    if value is None:
        return
    if value < 1 or value > 100:
        raise ValueError(f"{name} must be in [1, 100] when provided")


def _baseline_metadata(args: argparse.Namespace, use_mps: bool) -> dict[str, Any]:
    return {
        "experiment": "two_process",
        "dtype": args.dtype,
        "seq_len": int(args.seq_len),
        "seed": int(args.seed),
        "gpu_id": int(args.gpu_id),
        "batch_size": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "use_mps": bool(use_mps),
        "attn_thread_pct": args.attn_thread_pct,
        "moe_thread_pct": args.moe_thread_pct,
    }


def _metadata_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    for key, value in expected.items():
        if actual.get(key) != value:
            return False
    return True


def _worker_payload_path(output_dir: str, case_id: str) -> Path:
    return Path(output_dir) / "worker_payloads" / f"{case_id}.pt"


def _build_worker_payload(args: argparse.Namespace, case_id: str) -> Path:
    payload = {
        "dtype": args.dtype,
        "seq_len": int(args.seq_len),
        "batch_size": int(args.batch_size),
        "num_classes": int(args.num_classes),
        "attn_weight_seed": int(args.seed + 11),
        "moe_weight_seed": int(args.seed + 12),
        "attn_input_seed": int(args.seed + 21),
        "moe_input_seed": int(args.seed + 22),
        "labels_seed": int(args.seed + 23),
        "iter_seed_base": int(args.seed + 1000),
    }
    path = _worker_payload_path(args.output_dir, case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)
    return path


def _build_model_and_inputs(
    *,
    role: str,
    payload: dict[str, Any],
    gpu_id: int,
) -> tuple[nn.Module, torch.Tensor, torch.Tensor | None]:
    dtype_name = payload["dtype"]
    dtype = dtype_from_name(dtype_name)

    config = create_qwen3_config(
        attn_implementation="sdpa",
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
    )

    if role == "attn":
        seed_everything(int(payload["attn_weight_seed"]))
        model = Qwen3AttentionStage(config, dtype=dtype)
        input_seed = int(payload["attn_input_seed"])
        label_seed = None
    elif role == "moe":
        seed_everything(int(payload["moe_weight_seed"]))
        model = Qwen3MoEStageWithHead(config, int(payload["num_classes"]), dtype=dtype)
        input_seed = int(payload["moe_input_seed"])
        label_seed = int(payload["labels_seed"])
    else:
        raise ValueError(f"Unsupported role: {role}")

    model.eval()
    device = torch.device(f"cuda:{gpu_id}")
    model = model.to(device)

    seed_everything(input_seed)
    hidden_states = torch.randn(
        int(payload["batch_size"]),
        int(payload["seq_len"]),
        config.hidden_size,
        dtype=torch.float32,
    ).to(dtype)
    hidden_states = hidden_states.to(device)

    labels = None
    if label_seed is not None:
        seed_everything(label_seed)
        labels = torch.randint(
            0,
            int(payload["num_classes"]),
            (int(payload["batch_size"]),),
            dtype=torch.int64,
        ).to(device)

    return model, hidden_states, labels


def _collect_grad_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor | None]]:
    named: list[tuple[str, torch.Tensor | None]] = []
    for name, param in model.named_parameters():
        named.append((f"{name}.grad", param.grad))
        if len(named) >= 8:
            break
    return named


def _grad_checksum(model: nn.Module) -> float:
    checksum = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        checksum += float(param.grad.detach().float().abs().sum().item())
    return checksum


def _worker_main(
    *,
    role: str,
    payload_path: str,
    gpu_id: int,
    warmup_iters: int,
    timed_iters: int,
    thread_pct: int | None,
    barrier: mp.Barrier | None,
    result_queue: mp.Queue,
) -> None:
    try:
        set_mps_thread_pct(thread_pct)
        torch.cuda.set_device(gpu_id)

        payload = torch.load(payload_path, map_location="cpu")
        model, hidden_states, labels = _build_model_and_inputs(role=role, payload=payload, gpu_id=gpu_id)
        loss_fn = nn.CrossEntropyLoss()

        first_nonfinite: dict[str, Any] | None = None
        all_finite = True

        # `cuda` is intentionally forward-only for table comparability with same-process harness.
        fwd_cuda_ms_values: list[float] = []
        total_cuda_ms_values: list[float] = []
        step_ms_values: list[float] = []
        timed_phase_enqueue_start: float | None = None
        timed_phase_enqueue_end: float | None = None
        timed_enqueue_windows: list[tuple[float, float]] = []

        last_output: torch.Tensor | None = None
        last_loss: torch.Tensor | None = None

        total_iters = warmup_iters + timed_iters
        for iter_idx in range(total_iters):
            if barrier is not None:
                barrier.wait(timeout=30.0)

            seed_everything(int(payload["iter_seed_base"]) + iter_idx)
            model.zero_grad(set_to_none=True)

            total_start_event = torch.cuda.Event(enable_timing=True)
            total_end_event = torch.cuda.Event(enable_timing=True)
            fwd_start_event = torch.cuda.Event(enable_timing=True)
            fwd_end_event = torch.cuda.Event(enable_timing=True)

            iter_start = time.perf_counter()
            total_start_event.record()

            enqueue_start = time.perf_counter()
            fwd_start_event.record()
            output = model(hidden_states)
            fwd_end_event.record()
            enqueue_end = time.perf_counter()
            if role == "attn":
                loss = output.float().mean()
            else:
                assert labels is not None
                loss = loss_fn(output.float(), labels)
            loss.backward()

            total_end_event.record()
            torch.cuda.synchronize(gpu_id)
            iter_end = time.perf_counter()

            if iter_idx >= warmup_iters:
                step_ms_values.append((iter_end - iter_start) * 1000.0)
                fwd_cuda_ms_values.append(float(fwd_start_event.elapsed_time(fwd_end_event)))
                total_cuda_ms_values.append(float(total_start_event.elapsed_time(total_end_event)))
                if timed_phase_enqueue_start is None:
                    timed_phase_enqueue_start = enqueue_start
                timed_phase_enqueue_end = enqueue_end
                timed_enqueue_windows.append((enqueue_start, enqueue_end))

            if first_nonfinite is None:
                first_nonfinite = first_nonfinite_in_tensors(
                    iter_idx=iter_idx,
                    module=f"{role}_worker",
                    phase="forward",
                    named_tensors=[("output", output), ("loss", loss)],
                )
            if first_nonfinite is None:
                first_nonfinite = first_nonfinite_in_tensors(
                    iter_idx=iter_idx,
                    module=f"{role}_worker",
                    phase="backward",
                    named_tensors=_collect_grad_tensors(model),
                )
            if first_nonfinite is not None:
                all_finite = False

            last_output = output.detach()
            last_loss = loss.detach()

        result_queue.put(
            {
                "role": role,
                "status": "ok",
                "finite": {
                    "all_finite": bool(all_finite),
                    "first_nonfinite": first_nonfinite,
                },
                "timing_ms": {
                    "step_total": float(sum(step_ms_values) / len(step_ms_values)) if step_ms_values else None,
                    "cuda": float(sum(fwd_cuda_ms_values) / len(fwd_cuda_ms_values)) if fwd_cuda_ms_values else None,
                    "cuda_forward": float(sum(fwd_cuda_ms_values) / len(fwd_cuda_ms_values))
                    if fwd_cuda_ms_values
                    else None,
                    "cuda_total": float(sum(total_cuda_ms_values) / len(total_cuda_ms_values))
                    if total_cuda_ms_values
                    else None,
                },
                "enqueue_start": timed_phase_enqueue_start,
                "enqueue_end": timed_phase_enqueue_end,
                "timed_enqueue_windows": timed_enqueue_windows,
                "output_signature": tensor_signature(last_output),
                "loss_value": float(last_loss.item()) if last_loss is not None else None,
                "grad_checksum": float(_grad_checksum(model)),
                "error": {
                    "code": None,
                    "message": None,
                    "traceback": None,
                },
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "role": role,
                "status": "runtime_error",
                "finite": {
                    "all_finite": False,
                    "first_nonfinite": None,
                },
                "timing_ms": {
                    "step_total": None,
                    "cuda": None,
                },
                "enqueue_start": None,
                "enqueue_end": None,
                "timed_enqueue_windows": [],
                "output_signature": None,
                "loss_value": None,
                "grad_checksum": None,
                "error": {
                    "code": "runtime_error",
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            }
        )


def _launch_single_worker(
    *,
    role: str,
    payload_path: str,
    gpu_id: int,
    warmup_iters: int,
    timed_iters: int,
    thread_pct: int | None,
    worker_timeout_s: float,
) -> tuple[dict[str, Any], int]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker_main,
        kwargs={
            "role": role,
            "payload_path": payload_path,
            "gpu_id": gpu_id,
            "warmup_iters": warmup_iters,
            "timed_iters": timed_iters,
            "thread_pct": thread_pct,
            "barrier": None,
            "result_queue": result_queue,
        },
    )
    proc.start()

    result: dict[str, Any] | None = None
    try:
        result = result_queue.get(timeout=worker_timeout_s)
    except queue.Empty:
        result = {
            "role": role,
            "status": "runtime_error",
            "finite": {"all_finite": False, "first_nonfinite": None},
            "timing_ms": {"step_total": None, "cuda": None},
            "enqueue_start": None,
            "enqueue_end": None,
            "timed_enqueue_windows": [],
            "output_signature": None,
            "loss_value": None,
            "grad_checksum": None,
            "error": {
                "code": "worker_timeout",
                "message": f"Worker {role} timed out after {worker_timeout_s}s",
                "traceback": None,
            },
        }

    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)

    exit_code = 0 if result.get("status") == "ok" and proc.exitcode == 0 else 2
    return result, exit_code


def _launch_parallel_workers(
    *,
    payload_path: str,
    gpu_id: int,
    warmup_iters: int,
    timed_iters: int,
    attn_thread_pct: int | None,
    moe_thread_pct: int | None,
    worker_timeout_s: float,
) -> tuple[dict[str, Any], dict[str, Any], int]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    barrier = ctx.Barrier(2)

    workers = {
        "attn": ctx.Process(
            target=_worker_main,
            kwargs={
                "role": "attn",
                "payload_path": payload_path,
                "gpu_id": gpu_id,
                "warmup_iters": warmup_iters,
                "timed_iters": timed_iters,
                "thread_pct": attn_thread_pct,
                "barrier": barrier,
                "result_queue": result_queue,
            },
        ),
        "moe": ctx.Process(
            target=_worker_main,
            kwargs={
                "role": "moe",
                "payload_path": payload_path,
                "gpu_id": gpu_id,
                "warmup_iters": warmup_iters,
                "timed_iters": timed_iters,
                "thread_pct": moe_thread_pct,
                "barrier": barrier,
                "result_queue": result_queue,
            },
        ),
    }

    for proc in workers.values():
        proc.start()

    results: dict[str, dict[str, Any]] = {}
    deadline = time.time() + worker_timeout_s
    while len(results) < 2 and time.time() < deadline:
        timeout = max(0.1, deadline - time.time())
        try:
            result = result_queue.get(timeout=timeout)
            role = result.get("role")
            if role in ("attn", "moe"):
                results[role] = result
        except queue.Empty:
            break

    for role, proc in workers.items():
        proc.join(timeout=1.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=3.0)

    for role in ("attn", "moe"):
        if role not in results:
            results[role] = {
                "role": role,
                "status": "runtime_error",
                "finite": {"all_finite": False, "first_nonfinite": None},
                "timing_ms": {"step_total": None, "cuda": None},
                "enqueue_start": None,
                "enqueue_end": None,
                "timed_enqueue_windows": [],
                "output_signature": None,
                "loss_value": None,
                "grad_checksum": None,
                "error": {
                    "code": "worker_timeout",
                    "message": f"Worker {role} failed to report within {worker_timeout_s}s",
                    "traceback": None,
                },
            }

    exit_code = 0
    for role, proc in workers.items():
        if proc.exitcode != 0:
            exit_code = 2
        if results[role].get("status") != "ok":
            exit_code = 2

    return results["attn"], results["moe"], exit_code


def _select_first_nonfinite(attn: dict[str, Any], moe: dict[str, Any]) -> dict[str, Any] | None:
    candidates = []
    attn_nf = attn.get("finite", {}).get("first_nonfinite")
    moe_nf = moe.get("finite", {}).get("first_nonfinite")
    if attn_nf is not None:
        candidates.append(attn_nf)
    if moe_nf is not None:
        candidates.append(moe_nf)
    if not candidates:
        return None
    candidates.sort(key=lambda item: int(item.get("iter", 10**9)))
    return candidates[0]


def _run_case(args: argparse.Namespace, case_id: str, use_mps: bool) -> tuple[dict[str, Any], int]:
    ensure_dirs(args.output_dir)
    payload_path = _build_worker_payload(args, case_id)

    baseline_case_id = build_case_id(
        experiment="two_process",
        mode="serial",
        use_mps=use_mps,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
        attn_thread_pct=args.attn_thread_pct,
        moe_thread_pct=args.moe_thread_pct,
    )

    run_start = time.perf_counter()

    if args.mode == "serial":
        attn_result, attn_code = _launch_single_worker(
            role="attn",
            payload_path=str(payload_path),
            gpu_id=args.gpu_id,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            thread_pct=args.attn_thread_pct,
            worker_timeout_s=args.worker_timeout_s,
        )
        moe_result, moe_code = _launch_single_worker(
            role="moe",
            payload_path=str(payload_path),
            gpu_id=args.gpu_id,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            thread_pct=args.moe_thread_pct,
            worker_timeout_s=args.worker_timeout_s,
        )
        worker_code = 0 if attn_code == 0 and moe_code == 0 else 2
    else:
        attn_result, moe_result, worker_code = _launch_parallel_workers(
            payload_path=str(payload_path),
            gpu_id=args.gpu_id,
            warmup_iters=args.warmup_iters,
            timed_iters=args.timed_iters,
            attn_thread_pct=args.attn_thread_pct,
            moe_thread_pct=args.moe_thread_pct,
            worker_timeout_s=args.worker_timeout_s,
        )

    run_end = time.perf_counter()

    first_nonfinite = _select_first_nonfinite(attn_result, moe_result)
    all_finite = bool(attn_result.get("finite", {}).get("all_finite", False)) and bool(
        moe_result.get("finite", {}).get("all_finite", False)
    )

    status = "ok"
    exit_code = 0
    if worker_code != 0:
        status = "worker_failed"
        exit_code = 2

    overlap_required = args.mode == "parallel"
    overlap_ms = 0.0
    overlap_valid = True
    if overlap_required:
        attn_windows = attn_result.get("timed_enqueue_windows", []) or []
        moe_windows = moe_result.get("timed_enqueue_windows", []) or []
        overlaps_ms: list[float] = []
        for (attn_start, attn_end), (moe_start, moe_end) in zip(attn_windows, moe_windows):
            overlap_sec = max(0.0, min(float(attn_end), float(moe_end)) - max(float(attn_start), float(moe_start)))
            overlaps_ms.append(overlap_sec * 1000.0)

        if not overlaps_ms:
            overlap_valid = False
        else:
            overlap_ms = float(sum(overlaps_ms) / len(overlaps_ms))
            overlap_valid = overlap_ms >= args.min_host_overlap_ms
        if not overlap_valid and status == "ok":
            status = "invalid_overlap"
            exit_code = 2

    overlap = {
        "required": overlap_required,
        "host_enqueue_overlap_ms": float(overlap_ms),
        "min_required_host_overlap_ms": float(args.min_host_overlap_ms if overlap_required else 0.0),
        "valid": bool(overlap_valid),
    }

    atol, rtol = tolerance_for_dtype(args.dtype)
    numeric_diff = {
        "baseline_case_id": baseline_case_id if args.mode == "parallel" else None,
        "baseline_metadata_match": args.mode == "serial",
        "max_abs_diff": None,
        "relative_error": None,
        "atol": float(atol),
        "rtol": float(rtol),
    }

    baseline_meta = _baseline_metadata(args, use_mps)

    if args.mode == "serial":
        baseline_artifact = {
            "metadata": baseline_meta,
            "workers": {
                "attn": {
                    "output_signature": attn_result.get("output_signature"),
                    "loss_value": attn_result.get("loss_value"),
                    "grad_checksum": attn_result.get("grad_checksum"),
                },
                "moe": {
                    "output_signature": moe_result.get("output_signature"),
                    "loss_value": moe_result.get("loss_value"),
                    "grad_checksum": moe_result.get("grad_checksum"),
                },
            },
        }
        save_baseline_artifact(
            output_dir=args.output_dir,
            baseline_case_id=baseline_case_id,
            artifact=baseline_artifact,
            baseline_override_path=args.baseline_json,
        )
    else:
        try:
            _baseline_path, baseline_artifact = load_baseline_artifact(
                output_dir=args.output_dir,
                baseline_case_id=baseline_case_id,
                baseline_override_path=args.baseline_json,
            )
            metadata_ok = _metadata_match(baseline_meta, baseline_artifact.get("metadata", {}))
            numeric_diff["baseline_metadata_match"] = bool(metadata_ok)
            numeric_diff["baseline_case_id"] = baseline_case_id
            if not metadata_ok:
                if status == "ok":
                    status = "invalid_baseline"
                    exit_code = 2
            else:
                max_abs = 0.0
                max_rel = 0.0
                has_diff = False
                for role, current in (("attn", attn_result), ("moe", moe_result)):
                    baseline_worker = baseline_artifact.get("workers", {}).get(role, {})
                    c_sig = current.get("output_signature")
                    b_sig = baseline_worker.get("output_signature")
                    sig_abs, sig_rel = signature_diff(c_sig, b_sig)
                    if sig_abs is not None and sig_rel is not None:
                        max_abs = max(max_abs, sig_abs)
                        max_rel = max(max_rel, sig_rel)
                        has_diff = True

                    c_loss = current.get("loss_value")
                    b_loss = baseline_worker.get("loss_value")
                    if c_loss is not None and b_loss is not None:
                        loss_abs = abs(float(c_loss) - float(b_loss))
                        loss_rel = loss_abs / max(abs(float(b_loss)), 1e-12)
                        max_abs = max(max_abs, loss_abs)
                        max_rel = max(max_rel, loss_rel)
                        has_diff = True

                    c_grad = current.get("grad_checksum")
                    b_grad = baseline_worker.get("grad_checksum")
                    if c_grad is not None and b_grad is not None:
                        grad_abs = abs(float(c_grad) - float(b_grad))
                        grad_rel = grad_abs / max(abs(float(b_grad)), 1e-12)
                        max_abs = max(max_abs, grad_abs)
                        max_rel = max(max_rel, grad_rel)
                        has_diff = True

                if has_diff:
                    numeric_diff["max_abs_diff"] = float(max_abs)
                    numeric_diff["relative_error"] = float(max_rel)
        except Exception as exc:
            if status == "ok":
                status = "invalid_baseline"
                exit_code = 2
            return (
                build_case_payload(
                    case_id=case_id,
                    experiment="two_process",
                    mode=args.mode,
                    dtype=args.dtype,
                    seq_len=args.seq_len,
                    seed=args.seed,
                    gpu_id=args.gpu_id,
                    use_mps=use_mps,
                    status=status,
                    finite={"all_finite": all_finite, "first_nonfinite": first_nonfinite},
                    numeric_diff=numeric_diff,
                    timing_ms={
                        "step_total": (run_end - run_start) * 1000.0,
                        "attn_cuda": attn_result.get("timing_ms", {}).get("cuda"),
                        "moe_cuda": moe_result.get("timing_ms", {}).get("cuda"),
                    },
                    overlap=overlap,
                    error={
                        "code": "invalid_baseline",
                        "message": str(exc),
                        "traceback": None,
                    },
                    metadata={
                        "worker_payload_path": str(payload_path),
                        "workers": {
                            "attn": attn_result,
                            "moe": moe_result,
                        },
                        "warmup_iters": args.warmup_iters,
                        "timed_iters": args.timed_iters,
                        "worker_timeout_s": args.worker_timeout_s,
                    },
                ),
                exit_code,
            )

    payload = build_case_payload(
        case_id=case_id,
        experiment="two_process",
        mode=args.mode,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
        use_mps=use_mps,
        status=status,
        finite={"all_finite": all_finite, "first_nonfinite": first_nonfinite},
        numeric_diff=numeric_diff,
        timing_ms={
            "step_total": (run_end - run_start) * 1000.0,
            "attn_cuda": attn_result.get("timing_ms", {}).get("cuda"),
            "moe_cuda": moe_result.get("timing_ms", {}).get("cuda"),
        },
        overlap=overlap,
        error={
            "code": None if status == "ok" else status,
            "message": None if status == "ok" else "Worker execution failed" if status == "worker_failed" else None,
            "traceback": None,
        },
        metadata={
            "worker_payload_path": str(payload_path),
            "workers": {
                "attn": attn_result,
                "moe": moe_result,
            },
            "warmup_iters": args.warmup_iters,
            "timed_iters": args.timed_iters,
            "worker_timeout_s": args.worker_timeout_s,
            "attn_thread_pct": args.attn_thread_pct,
            "moe_thread_pct": args.moe_thread_pct,
            "baseline_metadata": baseline_meta,
        },
    )
    return payload, exit_code


def main() -> int:
    args = _parse_args()
    _validate_thread_pct("attn-thread-pct", args.attn_thread_pct)
    _validate_thread_pct("moe-thread-pct", args.moe_thread_pct)

    use_mps = args.use_mps == "on"
    case_id = build_case_id(
        experiment="two_process",
        mode=args.mode,
        use_mps=use_mps,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        gpu_id=args.gpu_id,
        attn_thread_pct=args.attn_thread_pct,
        moe_thread_pct=args.moe_thread_pct,
    )

    ok, preflight_payload = preflight_checks(
        experiment="two_process",
        mode=args.mode,
        gpu_id=args.gpu_id,
        dtype=args.dtype,
        seq_len=args.seq_len,
        seed=args.seed,
        use_mps=use_mps,
        output_dir=args.output_dir,
        strict_schema=args.strict_schema,
        attn_thread_pct=args.attn_thread_pct,
        moe_thread_pct=args.moe_thread_pct,
    )
    if not ok:
        ensure_dirs(args.output_dir)
        write_case_json(
            preflight_payload,
            output_dir=args.output_dir,
            json_output=args.json_output,
            strict_schema=args.strict_schema,
        )
        return 2

    try:
        if use_mps:
            with MPSContext(gpu_id=args.gpu_id):
                payload, code = _run_case(args, case_id=case_id, use_mps=True)
        else:
            for key in (
                "CUDA_MPS_PIPE_DIRECTORY",
                "CUDA_MPS_LOG_DIRECTORY",
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE",
            ):
                os.environ.pop(key, None)
            payload, code = _run_case(args, case_id=case_id, use_mps=False)
    except Exception as exc:
        payload = make_error_payload(
            case_id=case_id,
            experiment="two_process",
            mode=args.mode,
            dtype=args.dtype,
            seq_len=args.seq_len,
            seed=args.seed,
            gpu_id=args.gpu_id,
            use_mps=use_mps,
            status="runtime_error",
            code="runtime_error",
            message=str(exc),
        )
        code = 1

    ensure_dirs(args.output_dir)
    write_case_json(
        payload,
        output_dir=args.output_dir,
        json_output=args.json_output,
        strict_schema=args.strict_schema,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
