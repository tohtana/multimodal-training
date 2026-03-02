"""Shared utilities for overlap NaN minimal repro harnesses."""

from __future__ import annotations

import json
import os
import random
import shutil
import traceback
from pathlib import Path
from typing import Any

import torch

SCHEMA_VERSION = "mps_overlap_repro.v1"
BF16_ATOL = 5e-2
BF16_RTOL = 5e-2
FP32_ATOL = 1e-5
FP32_RTOL = 1e-4


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def tolerance_for_dtype(name: str) -> tuple[float, float]:
    if name == "bf16":
        return BF16_ATOL, BF16_RTOL
    if name == "fp32":
        return FP32_ATOL, FP32_RTOL
    raise ValueError(f"Unsupported dtype: {name}")


def _normalize_thread_pct(pct: int | None) -> str:
    return "dflt" if pct is None else str(pct)


def build_case_id(
    *,
    experiment: str,
    mode: str,
    use_mps: bool,
    dtype: str,
    seq_len: int,
    seed: int,
    gpu_id: int,
    attn_thread_pct: int | None = None,
    moe_thread_pct: int | None = None,
) -> str:
    parts = [
        f"exp-{experiment}",
        f"mode-{mode}",
        f"mps-{'on' if use_mps else 'off'}",
        f"dtype-{dtype}",
        f"seq-{seq_len}",
        f"seed-{seed}",
        f"gpu-{gpu_id}",
        f"attn-{_normalize_thread_pct(attn_thread_pct)}",
        f"moe-{_normalize_thread_pct(moe_thread_pct)}",
    ]
    return "__".join(parts)


def case_output_path(output_dir: str | Path, case_id: str) -> Path:
    return Path(output_dir) / "cases" / f"{case_id}.json"


def baseline_output_path(output_dir: str | Path, baseline_case_id: str) -> Path:
    return Path(output_dir) / "baselines" / f"{baseline_case_id}.pt"


def _default_finite() -> dict[str, Any]:
    return {
        "all_finite": True,
        "first_nonfinite": None,
    }


def _default_numeric_diff(atol: float, rtol: float) -> dict[str, Any]:
    return {
        "baseline_case_id": None,
        "baseline_metadata_match": False,
        "max_abs_diff": None,
        "relative_error": None,
        "atol": float(atol),
        "rtol": float(rtol),
    }


def _default_timing() -> dict[str, Any]:
    return {
        "step_total": None,
        "attn_cuda": None,
        "moe_cuda": None,
    }


def _default_overlap() -> dict[str, Any]:
    return {
        "required": False,
        "host_enqueue_overlap_ms": 0.0,
        "min_required_host_overlap_ms": 0.0,
        "valid": True,
    }


def _default_error() -> dict[str, Any]:
    return {
        "code": None,
        "message": None,
        "traceback": None,
    }


def build_case_payload(
    *,
    case_id: str,
    experiment: str,
    mode: str,
    dtype: str,
    seq_len: int,
    seed: int,
    gpu_id: int,
    use_mps: bool,
    status: str,
    finite: dict[str, Any] | None = None,
    numeric_diff: dict[str, Any] | None = None,
    timing_ms: dict[str, Any] | None = None,
    overlap: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    atol, rtol = tolerance_for_dtype(dtype)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "experiment": experiment,
        "mode": mode,
        "dtype": dtype,
        "seq_len": int(seq_len),
        "seed": int(seed),
        "gpu_id": int(gpu_id),
        "use_mps": bool(use_mps),
        "status": status,
        "finite": finite if finite is not None else _default_finite(),
        "numeric_diff": numeric_diff if numeric_diff is not None else _default_numeric_diff(atol, rtol),
        "timing_ms": timing_ms if timing_ms is not None else _default_timing(),
        "overlap": overlap if overlap is not None else _default_overlap(),
        "error": error if error is not None else _default_error(),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


def validate_case_payload(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    required_top = [
        "schema_version",
        "case_id",
        "experiment",
        "mode",
        "dtype",
        "seq_len",
        "seed",
        "gpu_id",
        "use_mps",
        "status",
        "finite",
        "numeric_diff",
        "timing_ms",
        "overlap",
        "error",
    ]
    for key in required_top:
        if key not in payload:
            errors.append(f"missing key: {key}")

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be '{SCHEMA_VERSION}', got '{payload.get('schema_version')}'"
        )

    finite = payload.get("finite", {})
    if not isinstance(finite, dict):
        errors.append("finite must be an object")
    else:
        if "all_finite" not in finite:
            errors.append("finite.all_finite missing")
        if "first_nonfinite" not in finite:
            errors.append("finite.first_nonfinite missing")
        first_nonfinite = finite.get("first_nonfinite")
        if first_nonfinite is not None:
            if not isinstance(first_nonfinite, dict):
                errors.append("finite.first_nonfinite must be object or null")
            else:
                for key in ["module", "phase", "tensor", "iter"]:
                    if key not in first_nonfinite:
                        errors.append(f"finite.first_nonfinite.{key} missing")

    numeric_diff = payload.get("numeric_diff", {})
    if not isinstance(numeric_diff, dict):
        errors.append("numeric_diff must be an object")
    else:
        for key in [
            "baseline_case_id",
            "baseline_metadata_match",
            "max_abs_diff",
            "relative_error",
            "atol",
            "rtol",
        ]:
            if key not in numeric_diff:
                errors.append(f"numeric_diff.{key} missing")

    timing_ms = payload.get("timing_ms", {})
    if not isinstance(timing_ms, dict):
        errors.append("timing_ms must be an object")
    else:
        for key in ["step_total", "attn_cuda", "moe_cuda"]:
            if key not in timing_ms:
                errors.append(f"timing_ms.{key} missing")

    overlap = payload.get("overlap", {})
    if not isinstance(overlap, dict):
        errors.append("overlap must be an object")
    else:
        for key in [
            "required",
            "host_enqueue_overlap_ms",
            "min_required_host_overlap_ms",
            "valid",
        ]:
            if key not in overlap:
                errors.append(f"overlap.{key} missing")

    error = payload.get("error", {})
    if not isinstance(error, dict):
        errors.append("error must be an object")
    else:
        for key in ["code", "message", "traceback"]:
            if key not in error:
                errors.append(f"error.{key} missing")

    return errors


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text)
    tmp_path.replace(path)


def _canonical_json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def write_case_json(
    payload: dict[str, Any],
    *,
    output_dir: str | Path,
    json_output: str | None,
    strict_schema: bool,
) -> Path:
    if strict_schema:
        errors = validate_case_payload(payload)
        if errors:
            joined = "\n".join(errors)
            raise RuntimeError(f"Case payload validation failed:\n{joined}")

    canonical = case_output_path(output_dir, payload["case_id"])
    text = _canonical_json_text(payload)
    _write_text_atomic(canonical, text)

    if json_output is not None:
        mirror = Path(json_output)
        _write_text_atomic(mirror, text)
        if canonical.read_bytes() != mirror.read_bytes():
            raise RuntimeError(
                f"Mirror payload mismatch between canonical '{canonical}' and mirror '{mirror}'"
            )

    return canonical


def save_baseline_artifact(
    *,
    output_dir: str | Path,
    baseline_case_id: str,
    artifact: dict[str, Any],
    baseline_override_path: str | None = None,
) -> Path:
    path = Path(baseline_override_path) if baseline_override_path else baseline_output_path(output_dir, baseline_case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(artifact, tmp_path)
    tmp_path.replace(path)
    return path


def load_baseline_artifact(
    *,
    output_dir: str | Path,
    baseline_case_id: str,
    baseline_override_path: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    path = Path(baseline_override_path) if baseline_override_path else baseline_output_path(output_dir, baseline_case_id)
    if not path.exists():
        raise FileNotFoundError(f"Baseline artifact does not exist: {path}")
    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise RuntimeError(f"Baseline artifact is not a dictionary: {path}")
    return path, artifact


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def first_nonfinite_in_tensors(
    *,
    iter_idx: int,
    module: str,
    phase: str,
    named_tensors: list[tuple[str, torch.Tensor | None]],
) -> dict[str, Any] | None:
    for name, tensor in named_tensors:
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            continue
        if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
            return {
                "module": module,
                "phase": phase,
                "tensor": name,
                "iter": int(iter_idx),
            }
    return None


def tensor_signature(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    if not torch.is_tensor(tensor):
        return None
    t = tensor.detach().to(device="cpu")
    if t.numel() == 0:
        return {
            "numel": 0,
            "sum": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0.0,
        }
    if not torch.is_floating_point(t):
        t = t.to(torch.float32)
    else:
        t = t.float()
    abs_t = t.abs()
    return {
        "numel": int(t.numel()),
        "sum": float(t.sum().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
        "max_abs": float(abs_t.max().item()),
    }


def signature_diff(
    current: dict[str, Any] | None,
    baseline: dict[str, Any] | None,
) -> tuple[float | None, float | None]:
    if current is None or baseline is None:
        return None, None

    keys = ["sum", "mean", "std", "max_abs"]
    max_abs_diff = 0.0
    max_rel_err = 0.0
    for key in keys:
        if key not in current or key not in baseline:
            return None, None
        c_val = float(current[key])
        b_val = float(baseline[key])
        abs_diff = abs(c_val - b_val)
        denom = max(abs(b_val), 1e-12)
        rel_err = abs_diff / denom
        max_abs_diff = max(max_abs_diff, abs_diff)
        max_rel_err = max(max_rel_err, rel_err)
    return max_abs_diff, max_rel_err


def tensor_numeric_diff(
    current_tensors: dict[str, torch.Tensor | None],
    baseline_tensors: dict[str, torch.Tensor | None],
) -> tuple[float | None, float | None]:
    max_abs_diff: float | None = None
    max_rel_err: float | None = None
    for key, cur_tensor in current_tensors.items():
        base_tensor = baseline_tensors.get(key)
        if cur_tensor is None or base_tensor is None:
            continue
        c = cur_tensor.detach().to(device="cpu")
        b = base_tensor.detach().to(device="cpu")
        if c.shape != b.shape:
            return None, None
        if not torch.is_floating_point(c):
            c = c.to(torch.float32)
        else:
            c = c.float()
        if not torch.is_floating_point(b):
            b = b.to(torch.float32)
        else:
            b = b.float()
        abs_diff = (c - b).abs()
        local_abs = float(abs_diff.max().item()) if abs_diff.numel() else 0.0
        base_max = float(b.abs().max().item()) if b.numel() else 0.0
        local_rel = local_abs / max(base_max, 1e-12)
        if max_abs_diff is None:
            max_abs_diff = local_abs
            max_rel_err = local_rel
        else:
            max_abs_diff = max(max_abs_diff, local_abs)
            max_rel_err = max(max_rel_err, local_rel)
    return max_abs_diff, max_rel_err


def grad_checksums(modules: dict[str, torch.nn.Module]) -> dict[str, float]:
    checksums: dict[str, float] = {}
    for module_name, module in modules.items():
        total = 0.0
        for param in module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            total += float(grad.abs().sum().item())
        checksums[module_name] = total
    return checksums


def preflight_checks(
    *,
    experiment: str,
    mode: str,
    gpu_id: int,
    dtype: str,
    seq_len: int,
    seed: int,
    use_mps: bool,
    output_dir: str,
    strict_schema: bool,
    nsys_required: bool = False,
    nsys_bin: str = "nsys",
    attn_thread_pct: int | None = None,
    moe_thread_pct: int | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    case_id = build_case_id(
        experiment=experiment,
        mode=mode,
        use_mps=use_mps,
        dtype=dtype,
        seq_len=seq_len,
        seed=seed,
        gpu_id=gpu_id,
        attn_thread_pct=attn_thread_pct,
        moe_thread_pct=moe_thread_pct,
    )

    errors: list[str] = []
    if not torch.cuda.is_available():
        errors.append("CUDA is not available")
    else:
        device_count = torch.cuda.device_count()
        if device_count <= 0:
            errors.append("No CUDA devices are visible")
        elif gpu_id < 0 or gpu_id >= device_count:
            errors.append(f"gpu-id {gpu_id} is out of range [0, {device_count - 1}]")

    try:
        from examples.attn_moe_overlap.model_utils import (  # noqa: F401
            Qwen3AttentionStage,
            Qwen3MoEStageWithHead,
            create_qwen3_config,
        )
    except Exception as exc:
        errors.append(f"Failed to import Qwen3 stage utilities: {exc}")

    if use_mps:
        if shutil.which("nvidia-cuda-mps-control") is None:
            errors.append("nvidia-cuda-mps-control not found in PATH")
        if torch.cuda.is_available() and 0 <= gpu_id < torch.cuda.device_count():
            cap = torch.cuda.get_device_capability(gpu_id)
            if cap[0] < 7:
                errors.append(
                    f"Selected gpu-id {gpu_id} has compute capability {cap[0]}.{cap[1]}; MPS requires >= 7.0"
                )

    if nsys_required:
        nsys_path = shutil.which(nsys_bin)
        if nsys_path is None and not Path(nsys_bin).exists():
            errors.append(f"Nsight Systems binary not found: {nsys_bin}")

    if errors:
        payload = build_case_payload(
            case_id=case_id,
            experiment=experiment,
            mode=mode,
            dtype=dtype,
            seq_len=seq_len,
            seed=seed,
            gpu_id=gpu_id,
            use_mps=use_mps,
            status="invalid_environment",
            error={
                "code": "invalid_environment",
                "message": "; ".join(errors),
                "traceback": None,
            },
            metadata={
                "strict_schema": strict_schema,
                "output_dir": output_dir,
            },
        )
        return False, payload

    return True, None


def make_error_payload(
    *,
    case_id: str,
    experiment: str,
    mode: str,
    dtype: str,
    seq_len: int,
    seed: int,
    gpu_id: int,
    use_mps: bool,
    status: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return build_case_payload(
        case_id=case_id,
        experiment=experiment,
        mode=mode,
        dtype=dtype,
        seq_len=seq_len,
        seed=seed,
        gpu_id=gpu_id,
        use_mps=use_mps,
        status=status,
        error={
            "code": code,
            "message": message,
            "traceback": traceback.format_exc(),
        },
    )


def ensure_dirs(output_dir: str | Path) -> None:
    base = Path(output_dir)
    (base / "cases").mkdir(parents=True, exist_ok=True)
    (base / "baselines").mkdir(parents=True, exist_ok=True)


def set_mps_thread_pct(thread_pct: int | None) -> None:
    if thread_pct is None:
        os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
    else:
        os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
