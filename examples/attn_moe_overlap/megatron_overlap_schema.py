"""Schema and bookkeeping helpers for Megatron EP overlap experiments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch

CASE_SCHEMA_VERSION = "megatron_ep_overlap.case.v1"
MATRIX_SCHEMA_VERSION = "megatron_ep_overlap.matrix.v1"

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


def canonical_nccl_tuple(nccl_tuple: tuple[int, int, int]) -> str:
    return f"{nccl_tuple[0]},{nccl_tuple[1]},{nccl_tuple[2]}"


def parse_nccl_tuples(
    *,
    nccl_tuples: str | None,
    nccl_socket_nthreads: int | None,
    nccl_max_nchannels: int | None,
    nccl_max_ctas: int | None,
) -> list[tuple[int, int, int]]:
    has_fallback = any(
        value is not None for value in (nccl_socket_nthreads, nccl_max_nchannels, nccl_max_ctas)
    )
    if nccl_tuples and has_fallback:
        raise ValueError("Do not mix --nccl-tuples with --nccl-socket-nthreads/--nccl-max-nchannels/--nccl-max-ctas")

    parsed: list[tuple[int, int, int]] = []
    if nccl_tuples:
        for raw_tuple in nccl_tuples.split(";"):
            token = raw_tuple.strip()
            if not token:
                continue
            parts = [part.strip() for part in token.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid NCCL tuple {token!r}: expected socket_nthreads,max_nchannels,max_ctas")
            try:
                triple = tuple(int(part) for part in parts)
            except ValueError as exc:
                raise ValueError(f"Invalid NCCL tuple {token!r}: values must be integers") from exc
            parsed.append(triple)  # type: ignore[arg-type]
    else:
        # Keep defaults modest and explicit when tuple list is omitted.
        triple = (
            int(nccl_socket_nthreads or 4),
            int(nccl_max_nchannels or 16),
            int(nccl_max_ctas or 32),
        )
        parsed.append(triple)

    deduped: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for triple in parsed:
        if len(triple) != 3:
            raise ValueError(f"Invalid NCCL tuple length: {triple}")
        if any(value <= 0 for value in triple):
            raise ValueError(f"NCCL tuple values must be positive integers: {triple}")
        if triple in seen:
            continue
        seen.add(triple)
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
    dtype: str,
    seed: int,
    attn_dp_size: int,
    moe_ep_size: int,
    attn_gpu_ids: Iterable[int],
    moe_gpu_ids: Iterable[int],
    nccl_tuple: tuple[int, int, int],
) -> str:
    return "__".join(
        (
            f"mode-{mode}",
            f"seq-{seq_len}",
            f"dtype-{normalize_dtype_name(dtype)}",
            f"seed-{seed}",
            f"dp-{attn_dp_size}",
            f"ep-{moe_ep_size}",
            f"attn-{_gpu_ids_fragment(attn_gpu_ids)}",
            f"moe-{_gpu_ids_fragment(moe_gpu_ids)}",
            f"nccl-{nccl_tuple[0]}_{nccl_tuple[1]}_{nccl_tuple[2]}",
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


def build_case_payload(
    *,
    case_id: str,
    status: str,
    mode: str,
    seq_len: int,
    dtype: str,
    seed: int,
    topology: dict[str, Any],
    nccl_env: dict[str, Any],
    timing_ms: dict[str, Any] | None = None,
    overlap_ms: float | None = None,
    finite: dict[str, Any] | None = None,
    stage_signatures: dict[str, Any] | None = None,
    baseline_diff: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    attempt_count: int = 1,
    retry_trigger: str = "none",
    artifact_path: str | None = None,
) -> dict[str, Any]:
    normalized_dtype = normalize_dtype_name(dtype)
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "status": status,
        "mode": mode,
        "seq_len": int(seq_len),
        "dtype": normalized_dtype,
        "seed": int(seed),
        "topology": topology,
        "nccl": nccl_env,
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
        "attempt_count": int(attempt_count),
        "retry_trigger": retry_trigger,
        "artifact_path": artifact_path,
    }


def build_invalid_environment_payload(
    *,
    case_id: str,
    mode: str,
    seq_len: int,
    dtype: str,
    seed: int,
    topology: dict[str, Any],
    nccl_env: dict[str, Any],
    message: str,
) -> dict[str, Any]:
    return build_case_payload(
        case_id=case_id,
        status="invalid_environment",
        mode=mode,
        seq_len=seq_len,
        dtype=dtype,
        seed=seed,
        topology=topology,
        nccl_env=nccl_env,
        error={"code": "invalid_environment", "message": message, "traceback": None},
    )


def validate_case_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "case_id",
        "status",
        "mode",
        "seq_len",
        "dtype",
        "seed",
        "topology",
        "nccl",
        "timing_ms",
        "overlap",
        "finite",
        "stage_signatures",
        "baseline_diff",
        "error",
        "attempt_count",
        "retry_trigger",
        "artifact_path",
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
    status = str(existing_payload.get("status"))
    return is_terminal_status(status)


def _compact_case_row(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": payload.get("case_id"),
        "status": payload.get("status"),
        "mode": payload.get("mode"),
        "seq_len": payload.get("seq_len"),
        "dtype": payload.get("dtype"),
        "nccl_tuple": payload.get("nccl", {}).get("tuple"),
        "attempt_count": payload.get("attempt_count", 1),
        "artifact_path": payload.get("artifact_path"),
    }


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

    summary = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "run_config": run_config,
        "counts": {
            "total_points": int(total_points),
            "attempted_cases": int(attempted_cases),
            "completed_cases": len(cases),
            "by_status": by_status,
        },
        "cases": [_compact_case_row(case) for case in cases],
        "generated_at": now_utc_iso(),
    }
    return summary


def validate_matrix_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_top = ("schema_version", "run_config", "counts", "cases", "generated_at")
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

    for key in ("total_points", "attempted_cases", "completed_cases", "by_status"):
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
    lines.append("")
    lines.append("| case_id | status | mode | seq_len | dtype | nccl_tuple | attempt_count |")
    lines.append("|---|---|---|---:|---|---|---:|")
    for row in summary.get("cases", []):
        lines.append(
            "| {case_id} | {status} | {mode} | {seq_len} | {dtype} | {nccl_tuple} | {attempt_count} |".format(
                case_id=row.get("case_id"),
                status=row.get("status"),
                mode=row.get("mode"),
                seq_len=row.get("seq_len"),
                dtype=row.get("dtype"),
                nccl_tuple=row.get("nccl_tuple"),
                attempt_count=row.get("attempt_count"),
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
