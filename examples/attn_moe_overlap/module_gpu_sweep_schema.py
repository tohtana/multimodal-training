"""Schema helpers for the Step-8 isolated attention/MoE GPU sweep."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

CASE_SCHEMA_VERSION = "module_gpu_sweep.case.v1"
SUMMARY_SCHEMA_VERSION = "module_gpu_sweep.summary.v1"
VALID_STAGE_ROLES = ("attn", "moe")
VALID_STATUS_KEYS = (
    "ok",
    "oom",
    "timeout",
    "runtime_error",
    "unsupported_dtype",
    "unavailable_hardware",
    "workspace_bootstrap_failed",
)

_DTYPE_ALIASES = {
    "float32": "fp32",
    "fp32": "fp32",
    "bfloat16": "bf16",
    "bf16": "bf16",
    "float16": "fp16",
    "fp16": "fp16",
}


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_stage_role(stage_role: str) -> str:
    normalized = stage_role.strip().lower()
    if normalized not in VALID_STAGE_ROLES:
        raise ValueError(f"Unsupported stage_role: {stage_role}")
    return normalized


def normalize_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.strip().lower()
    if normalized not in _DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return _DTYPE_ALIASES[normalized]


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


def parse_gpu_ids(raw: str) -> list[int]:
    values = parse_int_csv(raw, field_name="gpu-ids")
    seen: set[int] = set()
    deduped: list[int] = []
    for value in values:
        if value < 0:
            raise ValueError(f"gpu-ids must contain non-negative integers: got {value}")
        if value in seen:
            raise ValueError(f"gpu-ids contains duplicate GPU id: {value}")
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


def build_case_id(
    *,
    stage_role: str,
    seq_len: int,
    batch_size: int,
    dtype: str,
    seed: int,
    world_size: int,
) -> str:
    module = normalize_stage_role(stage_role)
    normalized_dtype = normalize_dtype_name(dtype)
    return "__".join(
        (
            f"module-{module}",
            f"seq-{int(seq_len)}",
            f"batch-{int(batch_size)}",
            f"dtype-{normalized_dtype}",
            f"seed-{int(seed)}",
            f"world-{int(world_size)}",
        )
    )


def relative_case_path(case_id: str) -> str:
    return f"cases/{case_id}.json"


def build_config_fingerprint(payload: dict[str, Any]) -> str:
    serializable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serializable.encode("utf-8")).hexdigest()[:16]


def build_case_payload(
    *,
    case_id: str,
    attempt_id: str,
    module: str,
    gpu_type: str,
    seq_len: int,
    batch_size: int,
    dtype: str,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    gpu_ids: list[int],
    world_size: int,
    model_name: str,
    model_type: str,
    status: str,
    status_reason: str | None,
    attention_backend: str,
    moe_grouped_gemm: bool,
    moe_token_dispatcher_type: str,
    moe_routing_mode: str,
    num_experts: int | None,
    forward_pass_ms: float | None = None,
    module_timing_ms: float | None = None,
    tokens_per_iter: int | None = None,
    tokens_per_second: float | None = None,
    timing_ms: dict[str, Any] | None = None,
    runtime: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    superproject_commit: str | None = None,
    multimodal_training_commit: str | None = None,
    artifact_path: str | None = None,
    pricing_status: str | None = None,
    per_gpu_hourly_cost_usd: float | None = None,
    node_hourly_cost_usd: float | None = None,
    cost_per_token_usd: float | None = None,
    runpod_model: str | None = None,
    tokens_per_expert: list[int] | None = None,
) -> dict[str, Any]:
    normalized_module = normalize_stage_role(module)
    normalized_dtype = normalize_dtype_name(dtype)
    if status not in VALID_STATUS_KEYS:
        raise ValueError(f"Unsupported status: {status}")
    return {
        "schema_version": CASE_SCHEMA_VERSION,
        "case_id": case_id,
        "attempt_id": str(attempt_id),
        "module": normalized_module,
        "gpu_type": str(gpu_type),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "dtype": normalized_dtype,
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "seed": int(seed),
        "gpu_ids": [int(value) for value in gpu_ids],
        "world_size": int(world_size),
        "model_name": str(model_name),
        "model_type": str(model_type),
        "status": status,
        "status_reason": status_reason,
        "attention_backend": str(attention_backend),
        "moe_grouped_gemm": bool(moe_grouped_gemm),
        "moe_token_dispatcher_type": str(moe_token_dispatcher_type),
        "moe_routing_mode": str(moe_routing_mode),
        "num_experts": None if num_experts is None else int(num_experts),
        "forward_pass_ms": None if forward_pass_ms is None else float(forward_pass_ms),
        "module_timing_ms": None if module_timing_ms is None else float(module_timing_ms),
        "tokens_per_iter": None if tokens_per_iter is None else int(tokens_per_iter),
        "tokens_per_second": None if tokens_per_second is None else float(tokens_per_second),
        "timing_ms": timing_ms
        if timing_ms is not None
        else {
            "timed_wall_total": None,
            "timed_wall_per_iter": None,
            "module_cuda": None,
            "module_step_total": None,
        },
        "runtime": runtime
        if runtime is not None
        else {
            "detected_gpu_name": None,
            "device_names_by_rank": [],
            "device_capabilities_by_rank": [],
        },
        "error": error
        if error is not None
        else {"code": None, "message": None, "traceback": None},
        "tokens_per_expert": tokens_per_expert,
        "pricing_status": pricing_status,
        "per_gpu_hourly_cost_usd": None if per_gpu_hourly_cost_usd is None else float(per_gpu_hourly_cost_usd),
        "node_hourly_cost_usd": None if node_hourly_cost_usd is None else float(node_hourly_cost_usd),
        "cost_per_token_usd": None if cost_per_token_usd is None else float(cost_per_token_usd),
        "runpod_model": runpod_model,
        "superproject_commit": superproject_commit,
        "multimodal_training_commit": multimodal_training_commit,
        "artifact_path": artifact_path,
        "generated_at": now_utc_iso(),
    }


def validate_case_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "case_id",
        "attempt_id",
        "module",
        "gpu_type",
        "seq_len",
        "batch_size",
        "dtype",
        "warmup_iters",
        "timed_iters",
        "seed",
        "gpu_ids",
        "world_size",
        "model_name",
        "model_type",
        "status",
        "status_reason",
        "forward_pass_ms",
        "module_timing_ms",
        "tokens_per_iter",
        "tokens_per_second",
        "timing_ms",
        "runtime",
        "error",
        "pricing_status",
        "per_gpu_hourly_cost_usd",
        "node_hourly_cost_usd",
        "cost_per_token_usd",
        "runpod_model",
        "superproject_commit",
        "multimodal_training_commit",
        "artifact_path",
    )
    for key in required:
        if key not in payload:
            errors.append(f"missing key: {key}")

    if payload.get("schema_version") != CASE_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {CASE_SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )

    module = payload.get("module")
    if module not in VALID_STAGE_ROLES:
        errors.append(f"module must be one of {VALID_STAGE_ROLES}, got {module!r}")

    status = payload.get("status")
    if status not in VALID_STATUS_KEYS:
        errors.append(f"status must be one of {VALID_STATUS_KEYS}, got {status!r}")

    for key in ("seq_len", "batch_size", "warmup_iters", "timed_iters", "seed", "world_size"):
        value = payload.get(key)
        if not isinstance(value, int):
            errors.append(f"{key} must be an integer, got {value!r}")
    if isinstance(payload.get("seq_len"), int) and payload["seq_len"] <= 0:
        errors.append("seq_len must be > 0")
    if isinstance(payload.get("batch_size"), int) and payload["batch_size"] <= 0:
        errors.append("batch_size must be > 0")
    if isinstance(payload.get("timed_iters"), int) and payload["timed_iters"] <= 0:
        errors.append("timed_iters must be > 0")
    if isinstance(payload.get("world_size"), int) and payload["world_size"] <= 0:
        errors.append("world_size must be > 0")

    try:
        normalize_dtype_name(str(payload.get("dtype")))
    except ValueError as exc:
        errors.append(str(exc))

    gpu_ids = payload.get("gpu_ids")
    if not isinstance(gpu_ids, list) or any(not isinstance(value, int) or value < 0 for value in gpu_ids):
        errors.append("gpu_ids must be a list of non-negative integers")
    elif isinstance(payload.get("world_size"), int) and len(gpu_ids) != payload["world_size"]:
        errors.append("len(gpu_ids) must equal world_size")

    for key in ("forward_pass_ms", "module_timing_ms", "tokens_per_second", "per_gpu_hourly_cost_usd", "node_hourly_cost_usd", "cost_per_token_usd"):
        value = payload.get(key)
        if value is not None and not isinstance(value, (int, float)):
            errors.append(f"{key} must be null or numeric, got {value!r}")

    tokens_per_iter = payload.get("tokens_per_iter")
    if tokens_per_iter is not None and (not isinstance(tokens_per_iter, int) or tokens_per_iter <= 0):
        errors.append("tokens_per_iter must be null or a positive integer")

    if payload.get("status_reason") is not None and not isinstance(payload.get("status_reason"), str):
        errors.append("status_reason must be null or a string")

    if not isinstance(payload.get("timing_ms"), dict):
        errors.append("timing_ms must be an object")
    if not isinstance(payload.get("runtime"), dict):
        errors.append("runtime must be an object")
    if not isinstance(payload.get("error"), dict):
        errors.append("error must be an object")

    artifact_path = payload.get("artifact_path")
    if not isinstance(artifact_path, str):
        errors.append("artifact_path must be a string")
    elif Path(artifact_path).is_absolute() or artifact_path.startswith("../"):
        errors.append("artifact_path must be output-dir-relative")

    return errors


def build_status_counts(cases: list[dict[str, Any]]) -> dict[str, int]:
    counts = {status: 0 for status in VALID_STATUS_KEYS}
    for case in cases:
        status = str(case.get("status"))
        counts.setdefault(status, 0)
        counts[status] += 1
    return counts


def build_module_summary(
    *,
    stage_role: str,
    attempt_id: str,
    command: list[str],
    gpu_type: str,
    detected_gpu_name: str | None,
    superproject_commit: str | None,
    multimodal_training_commit: str | None,
    case_paths: list[str],
    status_counts: dict[str, int],
    model_name: str,
    model_type: str,
    gpu_ids: list[int],
    seq_lens: list[int],
    batch_sizes: list[int],
    dtypes: list[str],
    warmup_iters: int,
    timed_iters: int,
    attention_backend: str,
    moe_grouped_gemm: bool,
    moe_token_dispatcher_type: str,
    moe_routing_mode: str,
    num_experts: int | None,
    config_fingerprint: str | None = None,
) -> dict[str, Any]:
    normalized_stage_role = normalize_stage_role(stage_role)
    normalized_case_paths = sorted(case_paths)
    summary_payload = {
        "stage_role": normalized_stage_role,
        "gpu_type": str(gpu_type),
        "model_name": str(model_name),
        "model_type": str(model_type),
        "gpu_ids": [int(value) for value in gpu_ids],
        "seq_lens": [int(value) for value in seq_lens],
        "batch_sizes": [int(value) for value in batch_sizes],
        "dtypes": [normalize_dtype_name(value) for value in dtypes],
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "attention_backend": str(attention_backend),
        "moe_grouped_gemm": bool(moe_grouped_gemm),
        "moe_token_dispatcher_type": str(moe_token_dispatcher_type),
        "moe_routing_mode": str(moe_routing_mode),
        "num_experts": None if num_experts is None else int(num_experts),
    }
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "attempt_id": str(attempt_id),
        "stage_role": normalized_stage_role,
        "command": [str(part) for part in command],
        "gpu_type": str(gpu_type),
        "detected_gpu_name": detected_gpu_name,
        "superproject_commit": superproject_commit,
        "multimodal_training_commit": multimodal_training_commit,
        "case_paths": normalized_case_paths,
        "status_counts": {
            status: int(status_counts.get(status, 0))
            for status in sorted(set(VALID_STATUS_KEYS) | set(status_counts))
        },
        "model_name": str(model_name),
        "model_type": str(model_type),
        "gpu_ids": [int(value) for value in gpu_ids],
        "seq_lens": [int(value) for value in seq_lens],
        "batch_sizes": [int(value) for value in batch_sizes],
        "dtypes": [normalize_dtype_name(value) for value in dtypes],
        "warmup_iters": int(warmup_iters),
        "timed_iters": int(timed_iters),
        "attention_backend": str(attention_backend),
        "moe_grouped_gemm": bool(moe_grouped_gemm),
        "moe_token_dispatcher_type": str(moe_token_dispatcher_type),
        "moe_routing_mode": str(moe_routing_mode),
        "num_experts": None if num_experts is None else int(num_experts),
        "config_fingerprint": config_fingerprint or build_config_fingerprint(summary_payload),
        "generated_at": now_utc_iso(),
    }


def validate_module_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = (
        "schema_version",
        "attempt_id",
        "stage_role",
        "command",
        "gpu_type",
        "detected_gpu_name",
        "superproject_commit",
        "multimodal_training_commit",
        "case_paths",
        "status_counts",
        "model_name",
        "model_type",
        "gpu_ids",
        "seq_lens",
        "batch_sizes",
        "dtypes",
        "warmup_iters",
        "timed_iters",
        "config_fingerprint",
    )
    for key in required:
        if key not in summary:
            errors.append(f"missing key: {key}")

    if summary.get("schema_version") != SUMMARY_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SUMMARY_SCHEMA_VERSION!r}, got {summary.get('schema_version')!r}"
        )

    stage_role = summary.get("stage_role")
    if stage_role not in VALID_STAGE_ROLES:
        errors.append(f"stage_role must be one of {VALID_STAGE_ROLES}, got {stage_role!r}")

    command = summary.get("command")
    if not isinstance(command, list) or any(not isinstance(part, str) for part in command):
        errors.append("command must be a list of strings")

    case_paths = summary.get("case_paths")
    if not isinstance(case_paths, list) or any(not isinstance(path, str) for path in case_paths):
        errors.append("case_paths must be a list of strings")
    else:
        for path in case_paths:
            if Path(path).is_absolute() or path.startswith("../"):
                errors.append("case_paths must be output-dir-relative")
                break

    status_counts = summary.get("status_counts")
    if not isinstance(status_counts, dict):
        errors.append("status_counts must be an object")
    else:
        for status in VALID_STATUS_KEYS:
            if status not in status_counts:
                errors.append(f"status_counts.{status} missing")
            elif not isinstance(status_counts[status], int) or status_counts[status] < 0:
                errors.append(f"status_counts.{status} must be a non-negative integer")

    for list_key in ("gpu_ids", "seq_lens", "batch_sizes", "dtypes"):
        value = summary.get(list_key)
        if not isinstance(value, list):
            errors.append(f"{list_key} must be a list")

    return errors


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def write_case_json(output_dir: str | Path, payload: dict[str, Any], *, strict_schema: bool = True) -> Path:
    path = Path(output_dir) / relative_case_path(str(payload["case_id"]))
    materialized = dict(payload)
    materialized["artifact_path"] = relative_case_path(str(payload["case_id"]))
    if strict_schema:
        errors = validate_case_payload(materialized)
        if errors:
            raise RuntimeError("Case payload failed schema validation:\n" + "\n".join(errors))
    write_json_atomic(path, materialized)
    return path


def write_module_summary(output_dir: str | Path, summary: dict[str, Any], *, strict_schema: bool = True) -> Path:
    if strict_schema:
        errors = validate_module_summary(summary)
        if errors:
            raise RuntimeError("Module summary failed schema validation:\n" + "\n".join(errors))
    path = Path(output_dir) / "module_summary.json"
    write_json_atomic(path, summary)
    return path
