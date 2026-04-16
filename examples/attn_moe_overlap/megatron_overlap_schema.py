"""Schema and bookkeeping helpers for Megatron EP overlap experiments."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

CASE_SCHEMA_VERSION = "megatron_ep_overlap.case.v5"
MATRIX_SCHEMA_VERSION = "megatron_ep_overlap.matrix.v5"
RUNTIME_BACKENDS = ("mps_only", "mps_green_ctx")
TORCH_PROFILER_SELECTIONS = ("representative", "all-successful")
MOE_ROUTING_MODES = ("normal", "equal_tokens")
TORCH_COMPILE_REQUESTED_VALUES = ("on", "off")
TORCH_COMPILE_STATUS_VALUES = ("eager", "compiled", "compile_failed")

REQUIRED_STATUS_KEYS = (
    "ok",
    "oom",
    "timeout",
    "numerical_mismatch",
    "invalid_environment",
    "runtime_error",
    "nsys_capture_failed",
)
RETRYABLE_STATUS_KEYS = {"oom", "timeout"}
PROFILER_STATUS_KEYS = (*REQUIRED_STATUS_KEYS, "torch_profiler_capture_failed")

_DTYPE_ALIASES = {
    "float32": "fp32",
    "fp32": "fp32",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "fp16": "fp16",
}

_DTYPE_TOLERANCES = {
    "fp32": {"max_abs_diff": 1e-5, "max_rel_diff": 1e-4, "eps": 1e-12},
    "bf16": {"max_abs_diff": 5e-3, "max_rel_diff": 5e-2, "eps": 1e-6},
    "fp16": {"max_abs_diff": 5e-3, "max_rel_diff": 5e-2, "eps": 1e-6},
}

_SIGNATURE_NUMERIC_KEYS = ("sum", "mean", "std", "max_abs")


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_dtype_name(dtype_name: str) -> str:
    key = dtype_name.strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return _DTYPE_ALIASES[key]


def torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    normalized = normalize_dtype_name(dtype_name)
    if normalized == "fp32":
        return torch.float32
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def tolerance_for_dtype(dtype_name: str) -> dict[str, float]:
    normalized = normalize_dtype_name(dtype_name)
    return dict(_DTYPE_TOLERANCES[normalized])


def normalize_runtime_backend(runtime_backend: str) -> str:
    normalized = runtime_backend.strip().lower()
    if normalized not in RUNTIME_BACKENDS:
        raise ValueError(f"Unsupported runtime backend: {runtime_backend}")
    return normalized


def normalize_moe_routing_mode(moe_routing_mode: str) -> str:
    normalized = moe_routing_mode.strip().lower()
    if normalized not in MOE_ROUTING_MODES:
        raise ValueError(f"Unsupported MoE routing mode: {moe_routing_mode}")
    return normalized


def normalize_torch_compile_requested(requested: str) -> str:
    normalized = requested.strip().lower()
    if normalized not in TORCH_COMPILE_REQUESTED_VALUES:
        raise ValueError(f"Unsupported torch compile request: {requested}")
    return normalized


def normalize_torch_compile_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized not in TORCH_COMPILE_STATUS_VALUES:
        raise ValueError(f"Unsupported torch compile status: {status}")
    return normalized


def build_torch_compile_metadata(
    *,
    requested: str = "off",
    by_role: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_requested = normalize_torch_compile_requested(requested)
    raw_by_role = by_role or {}
    normalized_by_role: dict[str, dict[str, str]] = {}
    for role in ("attn", "moe"):
        raw_role = raw_by_role.get(role) or {}
        status = raw_role.get("status", "eager")
        normalized_by_role[role] = {"status": normalize_torch_compile_status(str(status))}
    return {
        "requested": normalized_requested,
        "by_role": normalized_by_role,
    }


def parse_int_csv(raw: str, *, field_name: str) -> list[int]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be comma-separated integers: {raw!r}") from exc
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return values


def parse_seq_lens(raw: str) -> list[int]:
    values = parse_int_csv(raw, field_name="seq-lens")
    for value in values:
        if value <= 0:
            raise ValueError(f"seq-lens must be > 0: got {value}")
    return values


def parse_batch_sizes(raw: str) -> list[int]:
    values = parse_int_csv(raw, field_name="batch-sizes")
    for value in values:
        if value <= 0:
            raise ValueError(f"batch-sizes must be > 0: got {value}")
    return values


def parse_runtime_backends(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        normalized = normalize_runtime_backend(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    if not values:
        raise ValueError("runtime-backends must not be empty")
    return values


def parse_gpu_ids(raw: str, *, field_name: str) -> list[int]:
    values = parse_int_csv(raw, field_name=field_name)
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value < 0:
            raise ValueError(f"{field_name} must contain non-negative GPU ids: got {value}")
        if value in seen:
            raise ValueError(f"{field_name} contains duplicate GPU id: {value}")
        seen.add(value)
        deduped.append(value)
    return deduped


def parse_dtypes(raw: str) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        normalized = normalize_dtype_name(text)
        if normalized in seen:
            continue
        seen.add(normalized)
        values.append(normalized)
    if not values:
        raise ValueError("dtypes must not be empty")
    return values


def canonical_nccl_tuple(nccl_tuple: tuple[int, int, int] | None) -> str:
    if nccl_tuple is None:
        return "off"
    return f"{nccl_tuple[0]},{nccl_tuple[1]},{nccl_tuple[2]}"


def parse_nccl_tuples(
    *,
    nccl_tuples: str | None,
    nccl_socket_nthreads: int | None,
    nccl_max_nchannels: int | None,
    nccl_max_ctas: int | None,
) -> list[tuple[int, int, int] | None]:
    has_fallback = any(
        value is not None for value in (nccl_socket_nthreads, nccl_max_nchannels, nccl_max_ctas)
    )
    if nccl_tuples and has_fallback:
        raise ValueError("Do not mix --nccl-tuples with --nccl-socket-nthreads/--nccl-max-nchannels/--nccl-max-ctas")

    parsed: list[tuple[int, int, int] | None] = []
    if nccl_tuples:
        for raw_tuple in nccl_tuples.split(";"):
            token = raw_tuple.strip()
            if not token:
                continue
            if token.lower() in {"off", "none", "unset", "disabled"}:
                parsed.append(None)
                continue
            parts = [part.strip() for part in token.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid NCCL tuple {token!r}: expected socket_nthreads,max_nchannels,max_ctas")
            try:
                triple = tuple(int(part) for part in parts)
            except ValueError as exc:
                raise ValueError(f"Invalid NCCL tuple {token!r}: values must be integers") from exc
            parsed.append(triple)  # type: ignore[arg-type]
    elif has_fallback:
        triple = (
            int(nccl_socket_nthreads or 4),
            int(nccl_max_nchannels or 16),
            int(nccl_max_ctas or 32),
        )
        parsed.append(triple)
    else:
        parsed.append(None)

    if parsed.count(None) and len(parsed) > 1:
        raise ValueError("Do not mix disabled NCCL tuning with explicit NCCL tuples")

    deduped: list[tuple[int, int, int] | None] = []
    seen: set[str] = set()
    for triple in parsed:
        canonical = canonical_nccl_tuple(triple)
        if triple is None:
            if canonical in seen:
                continue
            seen.add(canonical)
            deduped.append(triple)
            continue
        if len(triple) != 3:
            raise ValueError(f"Invalid NCCL tuple length: {triple}")
        if any(value <= 0 for value in triple):
            raise ValueError(f"NCCL tuple values must be positive integers: {triple}")
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(triple)
    return deduped


def _gpu_ids_fragment(gpu_ids: Iterable[int]) -> str:
    values = list(gpu_ids)
    if not values:
        return "none"
    return "_".join(str(value) for value in values)


def build_case_id(
    *,
    mode: str,
    seq_len: int,
    batch_size: int,
    runtime_backend: str,
    green_ctx_attn_sms: int | None,
    green_ctx_moe_sms: int | None,
    dtype: str,
    seed: int,
    attn_dp_size: int,
    moe_ep_size: int,
    attn_gpu_ids: Iterable[int],
    moe_gpu_ids: Iterable[int],
    nccl_tuple: tuple[int, int, int] | None,
    moe_routing_mode: str = "normal",
) -> str:
    nccl_fragment = "off" if nccl_tuple is None else f"{nccl_tuple[0]}_{nccl_tuple[1]}_{nccl_tuple[2]}"
    runtime_backend = normalize_runtime_backend(runtime_backend)
    if runtime_backend == "mps_green_ctx":
        green_ctx_fragment = f"gc-{int(green_ctx_attn_sms or 0)}_{int(green_ctx_moe_sms or 0)}"
    else:
        green_ctx_fragment = "gc-off"
    return "__".join(
        (
            f"mode-{mode}",
            f"routing-{normalize_moe_routing_mode(moe_routing_mode)}",
            f"seq-{seq_len}",
            f"batch-{batch_size}",
            f"backend-{runtime_backend}",
            green_ctx_fragment,
            f"dtype-{normalize_dtype_name(dtype)}",
            f"seed-{seed}",
            f"dp-{attn_dp_size}",
            f"ep-{moe_ep_size}",
            f"attn-{_gpu_ids_fragment(attn_gpu_ids)}",
            f"moe-{_gpu_ids_fragment(moe_gpu_ids)}",
            f"nccl-{nccl_fragment}",
        )
    )


def case_output_path(output_dir: str | Path, case_id: str) -> Path:
    return Path(output_dir) / "cases" / f"{case_id}.json"


def tensor_signature(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return None

    t = tensor.detach().to("cpu")
    dtype_name = str(t.dtype).replace("torch.", "")
    shape = list(t.shape)
    if t.numel() == 0:
        return {
            "shape": shape,
            "dtype": dtype_name,
            "numel": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        }

    if not torch.is_floating_point(t):
        t = t.float()
    else:
        t = t.to(torch.float32)

    abs_tensor = t.abs()
    return {
        "shape": shape,
        "dtype": dtype_name,
        "numel": int(t.numel()),
        "sum": float(t.sum().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "max_abs": float(abs_tensor.max().item()),
    }


def _signature_diff(
    *,
    test_signature: dict[str, Any] | None,
    ref_signature: dict[str, Any] | None,
    dtype: str,
) -> tuple[float | None, float | None, float]:
    eps = tolerance_for_dtype(dtype)["eps"]
    if test_signature is None or ref_signature is None:
        return None, None, eps
    max_abs_diff = 0.0
    max_rel_diff = 0.0
    for key in _SIGNATURE_NUMERIC_KEYS:
        if key not in test_signature or key not in ref_signature:
            return None, None, eps
        test_value = float(test_signature[key])
        ref_value = float(ref_signature[key])
        abs_diff = abs(test_value - ref_value)
        rel_diff = abs_diff / max(abs(ref_value), eps)
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_diff = max(max_rel_diff, rel_diff)
    return max_abs_diff, max_rel_diff, eps


def evaluate_stage_diff(
    *,
    stage_name: str,
    dtype: str,
    test_signature: dict[str, Any] | None,
    ref_signature: dict[str, Any] | None,
) -> dict[str, Any]:
    tol = tolerance_for_dtype(dtype)
    max_abs_diff, max_rel_diff, eps = _signature_diff(
        test_signature=test_signature,
        ref_signature=ref_signature,
        dtype=dtype,
    )
    within_tolerance = (
        max_abs_diff is not None
        and max_rel_diff is not None
        and max_abs_diff <= tol["max_abs_diff"]
        and max_rel_diff <= tol["max_rel_diff"]
    )
    return {
        "stage": stage_name,
        "max_abs_diff": max_abs_diff,
        "max_rel_diff": max_rel_diff,
        "eps": eps,
        "tolerance": {
            "max_abs_diff": tol["max_abs_diff"],
            "max_rel_diff": tol["max_rel_diff"],
        },
        "within_tolerance": within_tolerance,
    }


def compute_speedup(serial_total_ms: float | None, overlap_total_ms: float | None) -> float | None:
    if serial_total_ms is None or overlap_total_ms is None:
        return None
    if overlap_total_ms <= 0:
        return None
    return float(serial_total_ms / overlap_total_ms)


def compute_host_enqueue_overlap_ms(
    attn_windows: list[tuple[float, float]],
    moe_windows: list[tuple[float, float]],
) -> float:
    overlap_values_ms: list[float] = []
    for attn_window, moe_window in zip(attn_windows, moe_windows):
        overlap_s = max(0.0, min(attn_window[1], moe_window[1]) - max(attn_window[0], moe_window[0]))
        overlap_values_ms.append(overlap_s * 1000.0)
    if not overlap_values_ms:
        return 0.0
    return float(sum(overlap_values_ms) / len(overlap_values_ms))


def _normalize_worker_sms(values: list[int | None] | None) -> list[int]:
    normalized: list[int] = []
    for value in values or []:
        if value is None:
            continue
        normalized.append(int(value))
    return normalized


def build_runtime_metadata(
    *,
    runtime_backend: str,
    green_ctx_attn_sms: int | None = None,
    green_ctx_moe_sms: int | None = None,
    granted_sms_by_role: dict[str, list[int | None] | None] | None = None,
    device_total_sms_by_role: dict[str, list[int | None] | None] | None = None,
) -> dict[str, Any]:
    runtime_backend = normalize_runtime_backend(runtime_backend)
    green_ctx_enabled = runtime_backend == "mps_green_ctx"
    return {
        "green_ctx_enabled": green_ctx_enabled,
        "requested_sms_by_role": {
            "attn": None if not green_ctx_enabled or green_ctx_attn_sms is None else int(green_ctx_attn_sms),
            "moe": None if not green_ctx_enabled or green_ctx_moe_sms is None else int(green_ctx_moe_sms),
        },
        "granted_sms_by_role": {
            "attn": _normalize_worker_sms((granted_sms_by_role or {}).get("attn")),
            "moe": _normalize_worker_sms((granted_sms_by_role or {}).get("moe")),
        },
        "device_total_sms_by_role": {
            "attn": _normalize_worker_sms((device_total_sms_by_role or {}).get("attn")),
            "moe": _normalize_worker_sms((device_total_sms_by_role or {}).get("moe")),
        },
    }


def build_config_fingerprint(identity_fields: dict[str, Any]) -> str:
    encoded = json.dumps(identity_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_case_payload(
    *,
    case_id: str,
    status: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    runtime_backend: str,
    dtype: str,
    seed: int,
    topology: dict[str, Any],
    nccl_env: dict[str, Any],
    moe_routing_mode: str = "normal",
    runtime: dict[str, Any] | None = None,
    timing_ms: dict[str, Any] | None = None,
    overlap_ms: float | None = None,
    finite: dict[str, Any] | None = None,
    stage_signatures: dict[str, Any] | None = None,
    baseline_diff: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    profiler: dict[str, Any] | None = None,
    torch_compile: dict[str, Any] | None = None,
    attempt_count: int = 1,
    retry_trigger: str = "none",
    artifact_path: str | None = None,
    tokens_per_expert: list[int] | None = None,
    tokens_per_expert_min: int | None = None,
    tokens_per_expert_max: int | None = None,
    tokens_per_expert_spread: int | None = None,
) -> dict[str, Any]:
    normalized_dtype = normalize_dtype_name(dtype)
    normalized_runtime_backend = normalize_runtime_backend(runtime_backend)
    normalized_moe_routing_mode = normalize_moe_routing_mode(moe_routing_mode)
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": status,
        "mode": mode,
        "moe_routing_mode": normalized_moe_routing_mode,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "runtime_backend": normalized_runtime_backend,
        "dtype": normalized_dtype,
        "seed": int(seed),
        "topology": topology,
        "nccl": nccl_env,
        "runtime": runtime
        if runtime is not None
        else build_runtime_metadata(runtime_backend=normalized_runtime_backend),
        "timing_ms": timing_ms
        if timing_ms is not None
        else {"total": None, "timed_wall": None, "attn": None, "moe": None},
        "overlap": {
            "host_enqueue_overlap_ms": float(overlap_ms) if overlap_ms is not None else 0.0,
            "speedup_vs_serial": None,
            "timed_speedup_vs_serial": None,
        },
        "finite": finite
        if finite is not None
        else {
            "all_finite": False,
            "first_nonfinite": None,
        },
        "stage_signatures": stage_signatures if stage_signatures is not None else {"attn": None, "moe": None},
        "baseline_diff": baseline_diff
        if baseline_diff is not None
        else {
            "baseline_case_id": None,
            "all_within_tolerance": None,
            "stages": {"attn": None, "moe": None},
        },
        "error": error
        if error is not None
        else {
            "code": None,
            "message": None,
            "traceback": None,
        },
        "profiler": profiler
        if profiler is not None
        else {
            "capture_requested": False,
            "selection": None,
            "wait_iters": None,
            "active_iters": None,
        },
        "torch_compile": (
            build_torch_compile_metadata()
            if torch_compile is None
            else build_torch_compile_metadata(
                requested=str(torch_compile.get("requested", "off")),
                by_role=torch_compile.get("by_role"),
            )
        ),
        "attempt_count": int(attempt_count),
        "retry_trigger": retry_trigger,
        "artifact_path": artifact_path,
        "tokens_per_expert": tokens_per_expert,
        "tokens_per_expert_min": tokens_per_expert_min,
        "tokens_per_expert_max": tokens_per_expert_max,
        "tokens_per_expert_spread": tokens_per_expert_spread,
    }


def build_invalid_environment_payload(
    *,
    case_id: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    runtime_backend: str,
    dtype: str,
    seed: int,
    topology: dict[str, Any],
    nccl_env: dict[str, Any],
    runtime: dict[str, Any] | None,
    message: str,
    moe_routing_mode: str = "normal",
    profiler: dict[str, Any] | None = None,
    torch_compile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_case_payload(
        case_id=case_id,
        status="invalid_environment",
        mode=mode,
        seq_len=seq_len,
        batch_size=batch_size,
        runtime_backend=runtime_backend,
        dtype=dtype,
        seed=seed,
        topology=topology,
        nccl_env=nccl_env,
        moe_routing_mode=moe_routing_mode,
        runtime=runtime,
        error={"code": "invalid_environment", "message": message, "traceback": None},
        profiler=profiler,
        torch_compile=torch_compile,
    )


def validate_case_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "case_id",
        "status",
        "mode",
        "moe_routing_mode",
        "seq_len",
        "batch_size",
        "runtime_backend",
        "dtype",
        "seed",
        "topology",
        "nccl",
        "runtime",
        "timing_ms",
        "overlap",
        "finite",
        "stage_signatures",
        "baseline_diff",
        "error",
        "profiler",
        "torch_compile",
        "attempt_count",
        "retry_trigger",
        "artifact_path",
        "tokens_per_expert",
        "tokens_per_expert_min",
        "tokens_per_expert_max",
        "tokens_per_expert_spread",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing key: {key}")

    if payload.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {CASE_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )

    status = payload.get("status")
    if status not in REQUIRED_STATUS_KEYS:
        errors.append(f"status must be one of {REQUIRED_STATUS_KEYS}, got {status!r}")

    moe_routing_mode = payload.get("moe_routing_mode")
    if moe_routing_mode not in MOE_ROUTING_MODES:
        errors.append(f"moe_routing_mode must be one of {MOE_ROUTING_MODES}, got {moe_routing_mode!r}")

    batch_size = payload.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        errors.append(f"batch_size must be a positive integer, got {batch_size!r}")

    runtime_backend = payload.get("runtime_backend")
    if runtime_backend not in RUNTIME_BACKENDS:
        errors.append(f"runtime_backend must be one of {RUNTIME_BACKENDS}, got {runtime_backend!r}")

    runtime = payload.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime must be an object")
    else:
        if "green_ctx_enabled" not in runtime:
            errors.append("runtime.green_ctx_enabled missing")
        requested_sms = runtime.get("requested_sms_by_role")
        granted_sms = runtime.get("granted_sms_by_role")
        device_total_sms = runtime.get("device_total_sms_by_role")
        for key, value in (
            ("requested_sms_by_role", requested_sms),
            ("granted_sms_by_role", granted_sms),
            ("device_total_sms_by_role", device_total_sms),
        ):
            if not isinstance(value, dict):
                errors.append(f"runtime.{key} must be an object")
                continue
            for role in ("attn", "moe"):
                if role not in value:
                    errors.append(f"runtime.{key}.{role} missing")
        for role in ("attn", "moe"):
            requested_value = (requested_sms or {}).get(role)
            if requested_value is not None and (not isinstance(requested_value, int) or requested_value <= 0):
                errors.append(f"runtime.requested_sms_by_role.{role} must be null or a positive integer")
            granted_value = (granted_sms or {}).get(role)
            if granted_value is not None:
                if not isinstance(granted_value, list):
                    errors.append(f"runtime.granted_sms_by_role.{role} must be a list")
                elif any(not isinstance(item, int) or item <= 0 for item in granted_value):
                    errors.append(f"runtime.granted_sms_by_role.{role} must contain positive integers")
            device_total_value = (device_total_sms or {}).get(role)
            if device_total_value is not None:
                if not isinstance(device_total_value, list):
                    errors.append(f"runtime.device_total_sms_by_role.{role} must be a list")
                elif any(not isinstance(item, int) or item <= 0 for item in device_total_value):
                    errors.append(f"runtime.device_total_sms_by_role.{role} must contain positive integers")

    timing_ms = payload.get("timing_ms")
    if not isinstance(timing_ms, dict):
        errors.append("timing_ms must be an object")
    else:
        for key in ("total", "timed_wall", "attn", "moe"):
            if key not in timing_ms:
                errors.append(f"timing_ms.{key} missing")

    overlap = payload.get("overlap")
    if not isinstance(overlap, dict):
        errors.append("overlap must be an object")
    else:
        if "host_enqueue_overlap_ms" not in overlap:
            errors.append("overlap.host_enqueue_overlap_ms missing")
        if "speedup_vs_serial" not in overlap:
            errors.append("overlap.speedup_vs_serial missing")
        if "timed_speedup_vs_serial" not in overlap:
            errors.append("overlap.timed_speedup_vs_serial missing")

    finite = payload.get("finite")
    if not isinstance(finite, dict):
        errors.append("finite must be an object")
    else:
        if "all_finite" not in finite:
            errors.append("finite.all_finite missing")
        if "first_nonfinite" not in finite:
            errors.append("finite.first_nonfinite missing")

    stage_signatures = payload.get("stage_signatures")
    if not isinstance(stage_signatures, dict):
        errors.append("stage_signatures must be an object")
    else:
        for key in ("attn", "moe"):
            if key not in stage_signatures:
                errors.append(f"stage_signatures.{key} missing")

    baseline_diff = payload.get("baseline_diff")
    if not isinstance(baseline_diff, dict):
        errors.append("baseline_diff must be an object")
    else:
        for key in ("baseline_case_id", "all_within_tolerance", "stages"):
            if key not in baseline_diff:
                errors.append(f"baseline_diff.{key} missing")
        stages = baseline_diff.get("stages", {})
        if not isinstance(stages, dict):
            errors.append("baseline_diff.stages must be an object")
        else:
            for key in ("attn", "moe"):
                if key not in stages:
                    errors.append(f"baseline_diff.stages.{key} missing")

    profiler = payload.get("profiler")
    if not isinstance(profiler, dict):
        errors.append("profiler must be an object")
    else:
        capture_requested = profiler.get("capture_requested")
        if not isinstance(capture_requested, bool):
            errors.append("profiler.capture_requested must be a boolean")
        selection = profiler.get("selection")
        if selection is not None and selection not in TORCH_PROFILER_SELECTIONS:
            errors.append(
                f"profiler.selection must be null or one of {TORCH_PROFILER_SELECTIONS}, got {selection!r}"
            )
        for key in ("wait_iters", "active_iters"):
            value = profiler.get(key)
            if value is not None and (not isinstance(value, int) or value <= 0):
                errors.append(f"profiler.{key} must be null or a positive integer")

    torch_compile = payload.get("torch_compile")
    if not isinstance(torch_compile, dict):
        errors.append("torch_compile must be an object")
    else:
        requested = torch_compile.get("requested")
        if requested not in TORCH_COMPILE_REQUESTED_VALUES:
            errors.append(
                "torch_compile.requested must be one of "
                f"{TORCH_COMPILE_REQUESTED_VALUES}, got {requested!r}"
            )
        by_role = torch_compile.get("by_role")
        if not isinstance(by_role, dict):
            errors.append("torch_compile.by_role must be an object")
        else:
            for role in ("attn", "moe"):
                role_payload = by_role.get(role)
                if not isinstance(role_payload, dict):
                    errors.append(f"torch_compile.by_role.{role} must be an object")
                    continue
                role_status = role_payload.get("status")
                if role_status not in TORCH_COMPILE_STATUS_VALUES:
                    errors.append(
                        f"torch_compile.by_role.{role}.status must be one of "
                        f"{TORCH_COMPILE_STATUS_VALUES}, got {role_status!r}"
                    )

    tokens_per_expert = payload.get("tokens_per_expert")
    tokens_per_expert_min = payload.get("tokens_per_expert_min")
    tokens_per_expert_max = payload.get("tokens_per_expert_max")
    tokens_per_expert_spread = payload.get("tokens_per_expert_spread")
    requires_equal_token_metadata = moe_routing_mode == "equal_tokens" and status in {
        "ok",
        "numerical_mismatch",
    }
    if requires_equal_token_metadata:
        if not isinstance(tokens_per_expert, list) or not tokens_per_expert:
            errors.append("tokens_per_expert must be a non-empty list for equal_tokens cases")
        elif any(not isinstance(value, int) or value < 0 for value in tokens_per_expert):
            errors.append("tokens_per_expert must contain non-negative integers")
        if not isinstance(tokens_per_expert_min, int) or tokens_per_expert_min < 0:
            errors.append("tokens_per_expert_min must be a non-negative integer for equal_tokens cases")
        if not isinstance(tokens_per_expert_max, int) or tokens_per_expert_max < 0:
            errors.append("tokens_per_expert_max must be a non-negative integer for equal_tokens cases")
        if not isinstance(tokens_per_expert_spread, int) or tokens_per_expert_spread < 0:
            errors.append("tokens_per_expert_spread must be a non-negative integer for equal_tokens cases")
        if (
            isinstance(tokens_per_expert, list)
            and tokens_per_expert
            and isinstance(tokens_per_expert_min, int)
            and isinstance(tokens_per_expert_max, int)
            and isinstance(tokens_per_expert_spread, int)
        ):
            if tokens_per_expert_min != min(tokens_per_expert):
                errors.append("tokens_per_expert_min must equal min(tokens_per_expert)")
            if tokens_per_expert_max != max(tokens_per_expert):
                errors.append("tokens_per_expert_max must equal max(tokens_per_expert)")
            if tokens_per_expert_spread != (tokens_per_expert_max - tokens_per_expert_min):
                errors.append("tokens_per_expert_spread must equal tokens_per_expert_max - tokens_per_expert_min")
    elif moe_routing_mode == "normal":
        for key, value in (
            ("tokens_per_expert", tokens_per_expert),
            ("tokens_per_expert_min", tokens_per_expert_min),
            ("tokens_per_expert_max", tokens_per_expert_max),
            ("tokens_per_expert_spread", tokens_per_expert_spread),
        ):
            if value is not None:
                errors.append(f"{key} must be null for normal routing cases")

    return errors


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_case_json(
    *,
    output_dir: str | Path,
    payload: dict[str, Any],
    strict_schema: bool = True,
) -> Path:
    if strict_schema:
        errors = validate_case_payload(payload)
        if errors:
            joined = "\n".join(errors)
            raise RuntimeError(f"Case payload failed schema validation:\n{joined}")
    path = case_output_path(output_dir, payload["case_id"])
    payload = dict(payload)
    payload["artifact_path"] = str(path)
    write_json_atomic(path, payload)
    return path


def load_case_payload(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def is_terminal_status(status: str) -> bool:
    return status in REQUIRED_STATUS_KEYS


def should_retry(status: str, attempt_count: int) -> bool:
    return attempt_count == 1 and status in RETRYABLE_STATUS_KEYS


def should_skip_existing(existing_payload: dict[str, Any], rerun_existing: bool) -> bool:
    if rerun_existing:
        return False
    if existing_payload.get("schema_version") != CASE_SCHEMA_VERSION:
        return False
    batch_size = existing_payload.get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        return False
    status = str(existing_payload.get("status"))
    return is_terminal_status(status)


def _compact_case_row(payload: dict[str, Any]) -> dict[str, Any]:
    runtime = payload.get("runtime") or {}
    return {
        "case_id": payload.get("case_id"),
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "moe_routing_mode": payload.get("moe_routing_mode"),
        "seq_len": payload.get("seq_len"),
        "batch_size": payload.get("batch_size"),
        "runtime_backend": payload.get("runtime_backend"),
        "green_ctx_sms": dict(runtime.get("requested_sms_by_role") or {"attn": None, "moe": None}),
        "dtype": payload.get("dtype"),
        "nccl_tuple": payload.get("nccl", {}).get("tuple"),
        "attempt_count": payload.get("attempt_count", 1),
        "tokens_per_expert_spread": payload.get("tokens_per_expert_spread"),
        "artifact_path": payload.get("artifact_path"),
    }


def _comparison_green_ctx_sms(payload: dict[str, Any]) -> tuple[int | None, int | None]:
    runtime = payload.get("runtime") or {}
    requested = runtime.get("requested_sms_by_role") or {}
    return (
        None if requested.get("attn") is None else int(requested["attn"]),
        None if requested.get("moe") is None else int(requested["moe"]),
    )


def _comparison_group_key(
    payload: dict[str, Any],
) -> tuple[int, int, str, str, str, str, int | None, int | None]:
    green_ctx_attn_sms, green_ctx_moe_sms = _comparison_green_ctx_sms(payload)
    return (
        int(payload.get("seq_len") or 0),
        int(payload.get("batch_size") or 0),
        str(payload.get("runtime_backend") or ""),
        str(payload.get("moe_routing_mode") or "normal"),
        str(payload.get("dtype") or ""),
        str(payload.get("nccl", {}).get("tuple") or ""),
        -1 if green_ctx_attn_sms is None else green_ctx_attn_sms,
        -1 if green_ctx_moe_sms is None else green_ctx_moe_sms,
    )


def _comparison_row_template(payload: dict[str, Any]) -> dict[str, Any]:
    green_ctx_attn_sms, green_ctx_moe_sms = _comparison_green_ctx_sms(payload)
    return {
        "seq_len": int(payload.get("seq_len") or 0),
        "batch_size": int(payload.get("batch_size") or 0),
        "runtime_backend": payload.get("runtime_backend"),
        "moe_routing_mode": payload.get("moe_routing_mode"),
        "green_ctx_sms": {
            "attn": green_ctx_attn_sms,
            "moe": green_ctx_moe_sms,
        },
        "dtype": payload.get("dtype"),
        "nccl_tuple": payload.get("nccl", {}).get("tuple"),
        "serial_case_id": None,
        "serial_status": "missing",
        "serial_attn_ms": None,
        "serial_moe_ms": None,
        "serial_total_ms": None,
        "serial_timed_wall_ms": None,
        "overlap_case_id": None,
        "overlap_status": "missing",
        "overlap_attn_ms": None,
        "overlap_moe_ms": None,
        "overlap_total_ms": None,
        "overlap_timed_wall_ms": None,
        "timed_speedup_vs_serial": None,
    }


def _build_comparison_rows(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str, str, str, int | None, int | None], dict[str, Any]] = {}
    for case in cases:
        key = _comparison_group_key(case)
        row = grouped.setdefault(key, _comparison_row_template(case))
        mode = case.get("mode")
        prefix = "serial" if mode == "serial" else "overlap" if mode == "overlap" else None
        if prefix is None:
            continue
        status = case.get("status") or "missing"
        timing_ms = case.get("timing_ms") or {}
        row[f"{prefix}_case_id"] = case.get("case_id")
        row[f"{prefix}_status"] = status
        if status == "ok":
            row[f"{prefix}_attn_ms"] = timing_ms.get("attn")
            row[f"{prefix}_moe_ms"] = timing_ms.get("moe")
            row[f"{prefix}_total_ms"] = timing_ms.get("total")
            row[f"{prefix}_timed_wall_ms"] = timing_ms.get("timed_wall")
        if prefix == "overlap" and status == "ok":
            row["timed_speedup_vs_serial"] = (case.get("overlap") or {}).get("timed_speedup_vs_serial")

    return [grouped[key] for key in sorted(grouped)]


def _backend_pair_common_key(row: dict[str, Any]) -> tuple[int, int, str, str, str]:
    return (
        int(row.get("seq_len") or 0),
        int(row.get("batch_size") or 0),
        str(row.get("moe_routing_mode") or "normal"),
        str(row.get("dtype") or ""),
        str(row.get("nccl_tuple") or ""),
    )


def _build_backend_pair_id(
    *,
    seq_len: int,
    batch_size: int,
    moe_routing_mode: str,
    dtype: str,
    nccl_tuple: str,
    green_ctx_attn_sms: int | None,
    green_ctx_moe_sms: int | None,
) -> str:
    return "__".join(
        (
            f"pair-seq-{seq_len}",
            f"batch-{batch_size}",
            f"routing-{normalize_moe_routing_mode(moe_routing_mode)}",
            f"dtype-{dtype}",
            f"nccl-{nccl_tuple.replace(',', '_')}",
            (
                f"gc-{int(green_ctx_attn_sms or 0)}_{int(green_ctx_moe_sms or 0)}"
                if green_ctx_attn_sms is not None or green_ctx_moe_sms is not None
                else "gc-off"
            ),
        )
    )


def _pair_status(backend_pair_row: dict[str, Any]) -> str:
    mps_only_serial_status = backend_pair_row.get("mps_only_serial_status")
    mps_only_overlap_status = backend_pair_row.get("mps_only_overlap_status")
    mps_green_ctx_serial_status = backend_pair_row.get("mps_green_ctx_serial_status")
    mps_green_ctx_overlap_status = backend_pair_row.get("mps_green_ctx_overlap_status")
    mps_only_missing = mps_only_serial_status is None and mps_only_overlap_status is None
    mps_green_ctx_missing = (
        mps_green_ctx_serial_status is None and mps_green_ctx_overlap_status is None
    )
    if mps_only_missing:
        return "missing_mps_only"
    if mps_green_ctx_missing:
        return "missing_mps_green_ctx"
    mps_only_ok = (mps_only_serial_status, mps_only_overlap_status) == ("ok", "ok")
    mps_green_ctx_ok = (mps_green_ctx_serial_status, mps_green_ctx_overlap_status) == ("ok", "ok")
    if not mps_only_ok and not mps_green_ctx_ok:
        return "both_failed"
    if not mps_only_ok:
        return "mps_only_failed"
    if not mps_green_ctx_ok:
        return "mps_green_ctx_failed"
    return "ok"


def _build_backend_pair_rows(run_config: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runtime_backends = [str(value) for value in run_config.get("runtime_backends") or []]
    if "mps_only" not in runtime_backends or "mps_green_ctx" not in runtime_backends:
        return []

    mps_only_rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    mps_green_ctx_rows: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for row in comparison_rows:
        key = _backend_pair_common_key(row)
        backend = row.get("runtime_backend")
        if backend == "mps_only":
            mps_only_rows[key] = row
        elif backend == "mps_green_ctx":
            mps_green_ctx_rows[key] = row

    requested_green_ctx_sms = dict(run_config.get("green_ctx_sms") or {"attn": None, "moe": None})
    pair_rows: list[dict[str, Any]] = []
    for key in sorted(set(mps_only_rows) | set(mps_green_ctx_rows)):
        seq_len, batch_size, moe_routing_mode, dtype, nccl_tuple = key
        mps_only_row = mps_only_rows.get(key)
        mps_green_ctx_row = mps_green_ctx_rows.get(key)
        row = {
            "pair_id": _build_backend_pair_id(
                seq_len=seq_len,
                batch_size=batch_size,
                moe_routing_mode=moe_routing_mode,
                dtype=dtype,
                nccl_tuple=nccl_tuple,
                green_ctx_attn_sms=requested_green_ctx_sms.get("attn"),
                green_ctx_moe_sms=requested_green_ctx_sms.get("moe"),
            ),
            "seq_len": seq_len,
            "batch_size": batch_size,
            "moe_routing_mode": moe_routing_mode,
            "dtype": dtype,
            "nccl_tuple": nccl_tuple,
            "green_ctx_sms": requested_green_ctx_sms,
            "mps_only_serial_case_id": None if mps_only_row is None else mps_only_row.get("serial_case_id"),
            "mps_only_serial_status": None if mps_only_row is None else mps_only_row.get("serial_status"),
            "mps_only_serial_timed_wall_ms": (
                None if mps_only_row is None else mps_only_row.get("serial_timed_wall_ms")
            ),
            "mps_only_serial_timed_speedup_vs_serial": (
                1.0 if mps_only_row is not None and mps_only_row.get("serial_status") == "ok" else None
            ),
            "mps_only_overlap_case_id": None if mps_only_row is None else mps_only_row.get("overlap_case_id"),
            "mps_only_overlap_status": None if mps_only_row is None else mps_only_row.get("overlap_status"),
            "mps_only_overlap_timed_wall_ms": (
                None if mps_only_row is None else mps_only_row.get("overlap_timed_wall_ms")
            ),
            "mps_only_overlap_timed_speedup_vs_serial": (
                None if mps_only_row is None else mps_only_row.get("timed_speedup_vs_serial")
            ),
            "mps_green_ctx_serial_case_id": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("serial_case_id")
            ),
            "mps_green_ctx_serial_status": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("serial_status")
            ),
            "mps_green_ctx_serial_timed_wall_ms": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("serial_timed_wall_ms")
            ),
            "mps_green_ctx_serial_timed_speedup_vs_serial": (
                1.0
                if mps_green_ctx_row is not None and mps_green_ctx_row.get("serial_status") == "ok"
                else None
            ),
            "mps_green_ctx_overlap_case_id": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("overlap_case_id")
            ),
            "mps_green_ctx_overlap_status": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("overlap_status")
            ),
            "mps_green_ctx_overlap_timed_wall_ms": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("overlap_timed_wall_ms")
            ),
            "mps_green_ctx_overlap_timed_speedup_vs_serial": (
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("timed_speedup_vs_serial")
            ),
            "serial_timed_speedup_mps_green_ctx_vs_mps_only": compute_speedup(
                None if mps_only_row is None else mps_only_row.get("serial_timed_wall_ms"),
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("serial_timed_wall_ms"),
            ),
            "overlap_timed_speedup_mps_green_ctx_vs_mps_only": compute_speedup(
                None if mps_only_row is None else mps_only_row.get("overlap_timed_wall_ms"),
                None if mps_green_ctx_row is None else mps_green_ctx_row.get("overlap_timed_wall_ms"),
            ),
            "delta_overlap_total_ms": (
                None
                if mps_only_row is None
                or mps_green_ctx_row is None
                or mps_only_row.get("overlap_total_ms") is None
                or mps_green_ctx_row.get("overlap_total_ms") is None
                else float(mps_only_row["overlap_total_ms"]) - float(mps_green_ctx_row["overlap_total_ms"])
            ),
            "delta_overlap_timed_wall_ms": (
                None
                if mps_only_row is None
                or mps_green_ctx_row is None
                or mps_only_row.get("overlap_timed_wall_ms") is None
                or mps_green_ctx_row.get("overlap_timed_wall_ms") is None
                else float(mps_only_row["overlap_timed_wall_ms"]) - float(mps_green_ctx_row["overlap_timed_wall_ms"])
            ),
            "delta_timed_speedup_vs_serial": (
                None
                if mps_only_row is None
                or mps_green_ctx_row is None
                or mps_only_row.get("timed_speedup_vs_serial") is None
                or mps_green_ctx_row.get("timed_speedup_vs_serial") is None
                else float(mps_green_ctx_row["timed_speedup_vs_serial"])
                - float(mps_only_row["timed_speedup_vs_serial"])
            ),
        }
        row["pair_status"] = _pair_status(row)
        pair_rows.append(row)
    return pair_rows


def build_matrix_summary(
    *,
    run_config: dict[str, Any],
    cases: list[dict[str, Any]],
    total_points: int,
) -> dict[str, Any]:
    by_status = {status: 0 for status in REQUIRED_STATUS_KEYS}
    attempted_cases = 0
    for case in cases:
        status = str(case.get("status"))
        if status not in by_status:
            by_status[status] = 0
        by_status[status] += 1
        attempted_cases += max(int(case.get("attempt_count", 1)), 1)

    comparison_rows = _build_comparison_rows(cases)
    backend_pair_rows = _build_backend_pair_rows(run_config, comparison_rows)
    summary = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "run_config": run_config,
        "counts": {
            "total_points": int(total_points),
            "attempted_cases": int(attempted_cases),
            "completed_cases": len(cases),
            "comparison_points": len(comparison_rows),
            "backend_pair_points": len(backend_pair_rows),
            "by_status": by_status,
        },
        "cases": [_compact_case_row(case) for case in cases],
        "comparison_rows": comparison_rows,
        "backend_pair_rows": backend_pair_rows,
        "comparisons": {
            "backend_pairs": backend_pair_rows,
        },
        "generated_at": now_utc_iso(),
    }
    return summary


def validate_matrix_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = (
        "schema_version",
        "run_config",
        "counts",
        "cases",
        "comparison_rows",
        "backend_pair_rows",
        "comparisons",
        "generated_at",
    )
    for key in required_top:
        if key not in summary:
            errors.append(f"missing key: {key}")

    if summary.get("schema_version") != MATRIX_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {MATRIX_SCHEMA_VERSION!r}, got {summary.get('schema_version')!r}"
        )

    counts = summary.get("counts")
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
        return errors

    for key in ("total_points", "attempted_cases", "completed_cases", "comparison_points", "backend_pair_points", "by_status"):
        if key not in counts:
            errors.append(f"counts.{key} missing")

    by_status = counts.get("by_status", {})
    if not isinstance(by_status, dict):
        errors.append("counts.by_status must be an object")
        return errors

    for status in REQUIRED_STATUS_KEYS:
        if status not in by_status:
            errors.append(f"counts.by_status.{status} missing")

    completed_cases = int(counts.get("completed_cases", 0))
    status_sum = sum(int(value) for value in by_status.values()) if by_status else 0
    if status_sum != completed_cases:
        errors.append(
            f"sum(counts.by_status.values()) must equal completed_cases: {status_sum} != {completed_cases}"
        )

    run_config = summary.get("run_config")
    if not isinstance(run_config, dict):
        errors.append("run_config must be an object")
    else:
        batch_sizes = run_config.get("batch_sizes")
        if not isinstance(batch_sizes, list) or not batch_sizes:
            errors.append("run_config.batch_sizes missing or empty")
        elif any(not isinstance(value, int) or value <= 0 for value in batch_sizes):
            errors.append("run_config.batch_sizes must contain positive integers")
        if "batch_size" in run_config and batch_sizes:
            if len(batch_sizes) != 1 or int(run_config["batch_size"]) != int(batch_sizes[0]):
                errors.append("run_config.batch_size must mirror the only entry in run_config.batch_sizes")
        runtime_backends = run_config.get("runtime_backends")
        if not isinstance(runtime_backends, list) or not runtime_backends:
            errors.append("run_config.runtime_backends missing or empty")
        elif any(str(value) not in RUNTIME_BACKENDS for value in runtime_backends):
            errors.append(f"run_config.runtime_backends must contain only {RUNTIME_BACKENDS}")
        green_ctx_sms = run_config.get("green_ctx_sms")
        if not isinstance(green_ctx_sms, dict):
            errors.append("run_config.green_ctx_sms missing or invalid")
        else:
            for role in ("attn", "moe"):
                if role not in green_ctx_sms:
                    errors.append(f"run_config.green_ctx_sms.{role} missing")
        device_sm_signature = run_config.get("device_sm_signature")
        if not isinstance(device_sm_signature, dict):
            errors.append("run_config.device_sm_signature missing or invalid")
        elif any(not isinstance(value, int) or value <= 0 for value in device_sm_signature.values()):
            errors.append("run_config.device_sm_signature must map GPU ids to positive integer SM counts")
        config_fingerprint = run_config.get("config_fingerprint")
        if not isinstance(config_fingerprint, str) or not config_fingerprint.strip():
            errors.append("run_config.config_fingerprint missing or invalid")
        moe_routing_mode = run_config.get("moe_routing_mode")
        if moe_routing_mode not in MOE_ROUTING_MODES:
            errors.append(f"run_config.moe_routing_mode must be one of {MOE_ROUTING_MODES}")
        torch_compile = run_config.get("torch_compile")
        if torch_compile not in TORCH_COMPILE_REQUESTED_VALUES:
            errors.append(
                f"run_config.torch_compile must be one of {TORCH_COMPILE_REQUESTED_VALUES}, got {torch_compile!r}"
            )

    cases = summary.get("cases")
    if not isinstance(cases, list):
        errors.append("cases must be an array")
    else:
        for index, row in enumerate(cases):
            if not isinstance(row, dict):
                errors.append(f"cases[{index}] must be an object")
                continue
            if "batch_size" not in row:
                errors.append(f"cases[{index}].batch_size missing")
            if "runtime_backend" not in row:
                errors.append(f"cases[{index}].runtime_backend missing")
            if "moe_routing_mode" not in row:
                errors.append(f"cases[{index}].moe_routing_mode missing")

    comparison_rows = summary.get("comparison_rows")
    if not isinstance(comparison_rows, list):
        errors.append("comparison_rows must be an array")
    else:
        required_comparison_keys = (
            "seq_len",
            "batch_size",
            "runtime_backend",
            "moe_routing_mode",
            "green_ctx_sms",
            "serial_case_id",
            "serial_status",
            "serial_attn_ms",
            "serial_moe_ms",
            "serial_total_ms",
            "serial_timed_wall_ms",
            "overlap_case_id",
            "overlap_status",
            "overlap_attn_ms",
            "overlap_moe_ms",
            "overlap_total_ms",
            "overlap_timed_wall_ms",
            "timed_speedup_vs_serial",
        )
        for index, row in enumerate(comparison_rows):
            if not isinstance(row, dict):
                errors.append(f"comparison_rows[{index}] must be an object")
                continue
            for key in required_comparison_keys:
                if key not in row:
                    errors.append(f"comparison_rows[{index}].{key} missing")
        if int(counts.get("comparison_points", 0)) != len(comparison_rows):
            errors.append(
                "counts.comparison_points must equal len(comparison_rows): "
                f"{counts.get('comparison_points')} != {len(comparison_rows)}"
            )
    backend_pair_rows = summary.get("backend_pair_rows")
    if not isinstance(backend_pair_rows, list):
        errors.append("backend_pair_rows must be an array")
    else:
        required_pair_keys = (
            "seq_len",
            "batch_size",
            "moe_routing_mode",
            "mps_only_serial_case_id",
            "mps_only_serial_status",
            "mps_only_serial_timed_wall_ms",
            "mps_only_serial_timed_speedup_vs_serial",
            "mps_only_overlap_case_id",
            "mps_only_overlap_status",
            "mps_only_overlap_timed_wall_ms",
            "mps_only_overlap_timed_speedup_vs_serial",
            "mps_green_ctx_serial_case_id",
            "mps_green_ctx_serial_status",
            "mps_green_ctx_serial_timed_wall_ms",
            "mps_green_ctx_overlap_case_id",
            "mps_green_ctx_overlap_status",
            "mps_green_ctx_overlap_timed_wall_ms",
            "mps_green_ctx_overlap_timed_speedup_vs_serial",
            "serial_timed_speedup_mps_green_ctx_vs_mps_only",
            "overlap_timed_speedup_mps_green_ctx_vs_mps_only",
        )
        for index, row in enumerate(backend_pair_rows):
            if not isinstance(row, dict):
                errors.append(f"backend_pair_rows[{index}] must be an object")
                continue
            for key in required_pair_keys:
                if key not in row:
                    errors.append(f"backend_pair_rows[{index}].{key} missing")
            for status_key in (
                "mps_only_serial_status",
                "mps_only_overlap_status",
                "mps_green_ctx_serial_status",
                "mps_green_ctx_overlap_status",
            ):
                status = row.get(status_key)
                if status is not None and status not in REQUIRED_STATUS_KEYS:
                    errors.append(
                        f"backend_pair_rows[{index}].{status_key} must be one of {REQUIRED_STATUS_KEYS} or null"
                    )
        if int(counts.get("backend_pair_points", 0)) != len(backend_pair_rows):
            errors.append(
                "counts.backend_pair_points must equal len(backend_pair_rows): "
                f"{counts.get('backend_pair_points')} != {len(backend_pair_rows)}"
            )

    comparisons = summary.get("comparisons")
    if not isinstance(comparisons, dict):
        errors.append("comparisons must be an object")
    else:
        backend_pairs = comparisons.get("backend_pairs")
        if backend_pairs != backend_pair_rows:
            errors.append("comparisons.backend_pairs must mirror backend_pair_rows")
    return errors


def write_matrix_summary(
    *,
    output_dir: str | Path,
    summary: dict[str, Any],
    strict_schema: bool = True,
) -> Path:
    if strict_schema:
        errors = validate_matrix_summary(summary)
        if errors:
            joined = "\n".join(errors)
            raise RuntimeError(f"matrix_summary failed schema validation:\n{joined}")
    path = Path(output_dir) / "matrix_summary.json"
    write_json_atomic(path, summary)
    return path


def render_matrix_summary_markdown(summary: dict[str, Any]) -> str:
    counts = summary.get("counts", {})
    lines: list[str] = []
    lines.append("# Megatron EP Overlap Matrix Summary")
    lines.append("")
    lines.append(f"- generated_at: `{summary.get('generated_at')}`")
    lines.append(f"- total_points: `{counts.get('total_points')}`")
    lines.append(f"- attempted_cases: `{counts.get('attempted_cases')}`")
    lines.append(f"- completed_cases: `{counts.get('completed_cases')}`")
    lines.append(f"- comparison_points: `{counts.get('comparison_points')}`")
    lines.append(f"- backend_pair_points: `{counts.get('backend_pair_points')}`")
    lines.append("")
    lines.append("| case_id | status | mode | seq_len | batch_size | runtime_backend | dtype | nccl_tuple | attempt_count |")
    lines.append("|---|---|---|---:|---:|---|---|---|---:|")
    for row in summary.get("cases", []):
        lines.append(
            "| {case_id} | {status} | {mode}/{moe_routing_mode} | {seq_len} | {batch_size} | {runtime_backend} | {dtype} | {nccl_tuple} | {attempt_count} |".format(
                case_id=row.get("case_id"),
                status=row.get("status"),
                mode=row.get("mode"),
                moe_routing_mode=row.get("moe_routing_mode"),
                seq_len=row.get("seq_len"),
                batch_size=row.get("batch_size"),
                runtime_backend=row.get("runtime_backend"),
                dtype=row.get("dtype"),
                nccl_tuple=row.get("nccl_tuple"),
                attempt_count=row.get("attempt_count"),
            )
        )
    lines.append("")
    lines.append("## Comparison Rows")
    lines.append("")
    lines.append(
        "| seq_len | batch_size | runtime_backend | moe_routing_mode | serial_status | serial_attn_ms | serial_moe_ms | serial_total_ms | overlap_status | overlap_attn_ms | overlap_moe_ms | overlap_total_ms | timed_speedup_vs_serial |"
    )
    lines.append(
        "|---:|---:|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|"
    )
    for row in summary.get("comparison_rows", []):
        lines.append(
            "| {seq_len} | {batch_size} | {runtime_backend} | {moe_routing_mode} | {serial_status} | {serial_attn_ms} | {serial_moe_ms} | {serial_total_ms} | {overlap_status} | {overlap_attn_ms} | {overlap_moe_ms} | {overlap_total_ms} | {timed_speedup_vs_serial} |".format(
                seq_len=row.get("seq_len"),
                batch_size=row.get("batch_size"),
                runtime_backend=row.get("runtime_backend"),
                moe_routing_mode=row.get("moe_routing_mode"),
                serial_status=row.get("serial_status"),
                serial_attn_ms=row.get("serial_attn_ms"),
                serial_moe_ms=row.get("serial_moe_ms"),
                serial_total_ms=row.get("serial_total_ms"),
                overlap_status=row.get("overlap_status"),
                overlap_attn_ms=row.get("overlap_attn_ms"),
                overlap_moe_ms=row.get("overlap_moe_ms"),
                overlap_total_ms=row.get("overlap_total_ms"),
                timed_speedup_vs_serial=row.get("timed_speedup_vs_serial"),
            )
        )
    lines.append("")
    lines.append("## Backend Pair Rows")
    lines.append("")
    lines.append(
        "| pair_id | pair_status | seq_len | batch_size | moe_routing_mode | mps_only_overlap_status | mps_green_ctx_overlap_status | overlap_timed_speedup_mps_green_ctx_vs_mps_only |"
    )
    lines.append("|---|---|---:|---:|---|---|---|---:|")
    for row in summary.get("backend_pair_rows", []):
        lines.append(
            "| {pair_id} | {pair_status} | {seq_len} | {batch_size} | {moe_routing_mode} | {mps_only_overlap_status} | {mps_green_ctx_overlap_status} | {overlap_timed_speedup_mps_green_ctx_vs_mps_only} |".format(
                pair_id=row.get("pair_id"),
                pair_status=row.get("pair_status"),
                seq_len=row.get("seq_len"),
                batch_size=row.get("batch_size"),
                moe_routing_mode=row.get("moe_routing_mode"),
                mps_only_overlap_status=row.get("mps_only_overlap_status"),
                mps_green_ctx_overlap_status=row.get("mps_green_ctx_overlap_status"),
                overlap_timed_speedup_mps_green_ctx_vs_mps_only=row.get(
                    "overlap_timed_speedup_mps_green_ctx_vs_mps_only"
                ),
            )
        )
    lines.append("")
    lines.append("## Status Counts")
    lines.append("")
    for status in REQUIRED_STATUS_KEYS:
        value = summary.get("counts", {}).get("by_status", {}).get(status, 0)
        lines.append(f"- `{status}`: `{value}`")
    return "\n".join(lines) + "\n"


def write_matrix_summary_markdown(output_dir: str | Path, summary: dict[str, Any]) -> Path:
    path = Path(output_dir) / "matrix_summary.md"
    _write_text_atomic(path, render_matrix_summary_markdown(summary))
    return path
