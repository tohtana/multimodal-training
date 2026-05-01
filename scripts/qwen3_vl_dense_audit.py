"""Audit dense Qwen3-VL compatibility and layer-only truncation paths."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL_IDS = ["Qwen/Qwen3-VL-8B-Instruct"]
NON_LAYER_FIELD_PATHS = (
    "text_config.hidden_size",
    "text_config.num_attention_heads",
    "text_config.num_key_value_heads",
    "text_config.intermediate_size",
    "text_config.vocab_size",
    "text_config.rope_theta",
    "vision_config.hidden_size",
    "vision_config.out_hidden_size",
    "vision_config.num_heads",
    "vision_config.patch_size",
    "vision_config.temporal_patch_size",
    "vision_config.spatial_merge_size",
)


@dataclass
class DenseAuditReport:
    selected_model_id: str | None
    selected_model_revision: str | None
    layer_truncation: dict[str, Any]
    probe_parameter_path: dict[str, str | None]
    per_id_results: list[dict[str, Any]]
    optimizer_construction_added: bool = False
    dtype_assumptions: dict[str, Any] = field(default_factory=dict)
    generated_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def audit_dense_model_ids(
    model_ids: list[str],
    phase0_audit_path: Path,
    output_path: Path,
    *,
    allow_weight_download: bool = False,
) -> DenseAuditReport:
    phase0 = _read_json(phase0_audit_path)
    per_id_results = []
    selected: dict[str, Any] | None = None

    for model_id in model_ids:
        result = _audit_one_model_id(model_id, phase0, allow_weight_download=allow_weight_download)
        per_id_results.append(result)
        if selected is None and result["status"] == "ok":
            selected = result

    if selected is None:
        report = DenseAuditReport(
            selected_model_id=None,
            selected_model_revision=None,
            layer_truncation=_empty_layer_truncation(),
            probe_parameter_path={"vision": None, "text": None},
            per_id_results=per_id_results,
            dtype_assumptions={"bf16_master_weights_off": False},
        )
    else:
        report = DenseAuditReport(
            selected_model_id=selected["model_id"],
            selected_model_revision=selected.get("revision"),
            layer_truncation=selected["layer_truncation"],
            probe_parameter_path=selected["probe_parameter_path"],
            per_id_results=per_id_results,
            optimizer_construction_added=False,
            dtype_assumptions={"bf16_master_weights_off": False},
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output_path, asdict(report))
    return report


def assert_non_layer_fields_unchanged(original: Any, effective: Any, *, allowed_layer_fields: set[str]) -> None:
    """Raise if a non-layer architecture field differs between two configs."""
    diffs = []
    for path in NON_LAYER_FIELD_PATHS:
        if path in allowed_layer_fields:
            continue
        before = _get_path(original, path, default=None)
        after = _get_path(effective, path, default=None)
        if before != after:
            diffs.append({"path": path, "original": before, "effective": after})
    if diffs:
        raise ValueError(f"Non-layer architecture fields changed: {diffs}")


def _audit_one_model_id(model_id: str, phase0: dict[str, Any], *, allow_weight_download: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model_id": model_id,
        "dense": _is_dense_qwen3_vl_id(model_id),
        "status": "blocked",
        "failures": [],
        "checks": {},
    }
    if not result["dense"]:
        result["failures"].append("model id is not a dense Qwen3-VL candidate")
        return result

    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        result["checks"]["auto_config"] = "ok"
    except Exception as exc:  # pragma: no cover - exercised in integration environments
        result["checks"]["auto_config"] = "failed"
        result["failures"].append(f"AutoConfig failed: {_excerpt(exc)}")
        return result

    try:
        from swift.megatron.model import get_megatron_model_meta
        from swift.megatron.utils import convert_hf_config

        model_meta = get_megatron_model_meta("qwen3_vl")
        result["checks"]["ms_swift_model_meta"] = "ok" if model_meta is not None else "missing"
        if model_meta is None:
            result["failures"].append("swift.megatron model meta for qwen3_vl is missing")
        megatron_kwargs = convert_hf_config(hf_config)
        result["checks"]["hf_to_megatron_conversion"] = "ok"
        result["converted_keys"] = sorted(str(key) for key in megatron_kwargs.keys())
    except Exception as exc:  # pragma: no cover - depends on installed ms-swift
        result["checks"]["ms_swift_model_meta"] = result["checks"].get("ms_swift_model_meta", "failed")
        result["checks"]["hf_to_megatron_conversion"] = "failed"
        result["failures"].append(f"ms-swift conversion failed: {_excerpt(exc)}")

    result["local_weight_cache"] = _local_weight_cache_status(model_id)
    result["checks"]["local_weight_files"] = "ok" if result["local_weight_cache"]["complete"] else "incomplete"
    result["weight_download_allowed"] = allow_weight_download
    result["layer_truncation"] = _select_layer_truncation(hf_config, phase0)
    result["probe_parameter_path"] = _select_probe_paths(phase0)
    result["checks"]["layer_override_roundtrip"] = (
        "ok" if _layer_override_roundtrip_claim_is_supported(result["layer_truncation"]) else "unvalidated"
    )

    if not result["local_weight_cache"]["complete"] and not allow_weight_download:
        result["failures"].append(
            "local dense weight shards are incomplete and weight download is disabled: "
            f"{result['local_weight_cache']['summary']}"
        )
    if result["failures"]:
        return result
    if any(value is None for value in result["probe_parameter_path"].values()):
        result["failures"].append("no validated probe parameter path for at least one stage")
        return result

    result["status"] = "ok"
    return result


def _select_layer_truncation(hf_config: Any, phase0: dict[str, Any]) -> dict[str, Any]:
    layer_paths = phase0.get("layer_count_paths", {})
    vision_source = _first_existing_path(hf_config, ["vision_config.depth", "vision_config.num_hidden_layers"])
    language_source = _first_existing_path(hf_config, ["text_config.num_hidden_layers", "num_hidden_layers"])
    vision_original = _get_path(hf_config, vision_source, default=None) if vision_source else None
    language_original = _get_path(hf_config, language_source, default=None) if language_source else None
    return {
        "vision_source_field": vision_source,
        "language_source_field": language_source,
        "vision_override_path": layer_paths.get("vision", {}).get("override_path"),
        "language_override_path": layer_paths.get("language", {}).get("override_path"),
        "vision_original_layers": vision_original,
        "language_original_layers": language_original,
        "vision_effective_layers": vision_original,
        "language_effective_layers": language_original,
        "requested_ratio": None,
        "rounding_rule": None,
        "override_roundtrip_validated": False,
    }


def _select_probe_paths(phase0: dict[str, Any]) -> dict[str, str | None]:
    candidates = phase0.get("optimizer_probe_candidates", {})
    return {
        "vision": _first(candidates.get("vision", [])),
        "text": _first(candidates.get("text", [])),
    }


def _layer_override_roundtrip_claim_is_supported(layer_truncation: dict[str, Any]) -> bool:
    return bool(layer_truncation.get("override_roundtrip_validated"))


def _local_weight_cache_status(model_id: str) -> dict[str, Any]:
    cache_roots = [
        os.environ.get("HF_HUB_CACHE"),
        (Path(os.environ["HF_HOME"]) / "hub") if os.environ.get("HF_HOME") else None,
        Path(os.path.expanduser("~/.cache/huggingface")) / "hub",
        "/mnt/cluster_storage/hf_cache",
        "/mnt/cluster_storage/hf_cache/hub",
        "/mnt/local_storage/huggingface",
        "/mnt/local_storage/huggingface/hub",
        "/mnt/user_storage/huggingface",
        "/mnt/user_storage/huggingface/hub",
    ]
    rel = "hub/models--" + model_id.replace("/", "--")
    direct_rel = "models--" + model_id.replace("/", "--")
    hf_candidates: list[Path] = []
    for root in cache_roots:
        if not root:
            continue
        root_path = Path(root)
        hf_candidates.append(root_path / direct_rel)
        hf_candidates.append(root_path / rel)

    modelscope_roots = [
        os.environ.get("MODELSCOPE_CACHE"),
        os.path.expanduser("~/.cache/modelscope/hub"),
        "/mnt/local_storage/modelscope_cache",
        "/mnt/user_storage/modelscope_cache",
    ]
    modelscope_rel = Path("models") / model_id
    modelscope_candidates = [Path(root) / modelscope_rel for root in modelscope_roots if root]

    candidate_roots: list[Path] = []
    for root in hf_candidates:
        snapshot_root = root / "snapshots"
        if snapshot_root.exists():
            candidate_roots.extend(path for path in snapshot_root.iterdir() if path.is_dir())
        candidate_roots.append(root)
    candidate_roots.extend(modelscope_candidates)
    candidate_roots = _dedupe_paths(candidate_roots)

    statuses = [_weight_file_status(path) for path in candidate_roots]
    existing = [status for status in statuses if status["exists"]]
    complete = [status for status in statuses if status["complete"]]
    partial = [status for status in statuses if status["index_exists"] and not status["complete"]]
    summary = _weight_cache_summary(existing=existing, complete=complete, partial=partial)
    return {
        "exists": bool(existing),
        "complete": bool(complete),
        "paths_checked": [str(path) for path in _dedupe_paths([*hf_candidates, *modelscope_candidates])],
        "existing_paths": [status["path"] for status in existing],
        "complete_paths": [status["path"] for status in complete],
        "partial_paths": [status["path"] for status in partial],
        "candidate_statuses": statuses,
        "summary": summary,
    }


def _weight_file_status(path: Path) -> dict[str, Any]:
    index_path = path / "model.safetensors.index.json"
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "index_exists": False,
            "complete": False,
            "required_files": [],
            "present_files": [],
            "missing_files": [],
            "required_count": 0,
            "present_count": 0,
            "expected_total_bytes": None,
            "present_bytes": 0,
        }

    try:
        index_payload = _read_json(index_path) if index_path.exists() else {}
    except json.JSONDecodeError:
        index_payload = {}
    required_files = _required_safetensor_files(index_payload)
    expected_total_bytes = index_payload.get("metadata", {}).get("total_size")
    present_files = []
    missing_files = []
    present_bytes = 0
    for filename in required_files:
        shard_path = path / filename
        if shard_path.exists() and shard_path.stat().st_size > 0:
            present_files.append(filename)
            present_bytes += shard_path.stat().st_size
        else:
            missing_files.append(filename)
    return {
        "path": str(path),
        "exists": True,
        "index_exists": index_path.exists(),
        "complete": bool(required_files) and not missing_files,
        "required_files": required_files,
        "present_files": present_files,
        "missing_files": missing_files,
        "required_count": len(required_files),
        "present_count": len(present_files),
        "expected_total_bytes": expected_total_bytes,
        "present_bytes": present_bytes,
    }


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    result = []
    seen = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def _required_safetensor_files(payload: dict[str, Any]) -> list[str]:
    return sorted(set(str(filename) for filename in payload.get("weight_map", {}).values()))


def _weight_cache_summary(
    *,
    existing: list[dict[str, Any]],
    complete: list[dict[str, Any]],
    partial: list[dict[str, Any]],
) -> str:
    if complete:
        return f"complete weights under {complete[0]['path']}"
    if partial:
        first = partial[0]
        return (
            f"incomplete weights under {first['path']} "
            f"({len(first['present_files'])}/{len(first['required_files'])} shards present)"
        )
    if existing:
        return "cache directory exists but no model.safetensors.index.json with complete shards was found"
    return "no local cache directory found"


def _is_dense_qwen3_vl_id(model_id: str) -> bool:
    lowered = model_id.lower()
    return "qwen3-vl" in lowered and "a3b" not in lowered and "moe" not in lowered


def _empty_layer_truncation() -> dict[str, Any]:
    return {
        "vision_source_field": None,
        "language_source_field": None,
        "vision_override_path": None,
        "language_override_path": None,
        "vision_original_layers": None,
        "language_original_layers": None,
        "vision_effective_layers": None,
        "language_effective_layers": None,
        "requested_ratio": None,
        "rounding_rule": None,
        "override_roundtrip_validated": False,
    }


def _first(values: list[str] | tuple[str, ...]) -> str | None:
    return values[0] if values else None


def _first_existing_path(obj: Any, paths: list[str]) -> str | None:
    for path in paths:
        if _get_path(obj, path, default=None) is not None:
            return path
    return None


def _get_path(obj: Any, path: str | None, *, default: Any = None) -> Any:
    if not path:
        return default
    value = obj
    for part in path.split("."):
        if isinstance(value, dict):
            if part not in value:
                return default
            value = value[part]
        else:
            if not hasattr(value, part):
                return default
            value = getattr(value, part)
    return value


def _excerpt(exc: Exception) -> str:
    text = str(exc)
    return text if len(text) <= 4096 else text[:4081] + "... [truncated]"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    default_artifacts = default_root / "todo/docs/qwen3-vl-dense-pipeline-benchmark/artifacts"
    parser.add_argument("--model-ids", default=",".join(DEFAULT_MODEL_IDS))
    parser.add_argument("--phase0-audit", type=Path, default=default_artifacts / "phase0_audit.json")
    parser.add_argument("--output", type=Path, default=default_artifacts / "dense_model_audit.json")
    parser.add_argument("--allow-weight-download", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_ids = [item.strip() for item in args.model_ids.split(",") if item.strip()]
    audit_dense_model_ids(
        model_ids,
        args.phase0_audit,
        args.output,
        allow_weight_download=args.allow_weight_download,
    )


if __name__ == "__main__":
    main()
