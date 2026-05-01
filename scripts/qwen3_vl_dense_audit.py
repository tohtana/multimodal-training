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
    result["weight_download_allowed"] = allow_weight_download
    result["layer_truncation"] = _select_layer_truncation(hf_config, phase0)
    result["probe_parameter_path"] = _select_probe_paths(phase0)

    if result["failures"]:
        return result
    if not _layer_override_roundtrip_claim_is_supported(result["layer_truncation"]):
        result["failures"].append("layer override path was identified but not runtime-roundtrip validated")
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
        os.environ.get("HF_HOME"),
        os.path.expanduser("~/.cache/huggingface"),
        "/mnt/local_storage/huggingface",
        "/mnt/user_storage/huggingface",
    ]
    rel = "hub/models--" + model_id.replace("/", "--")
    candidates = [str(Path(root) / rel) for root in cache_roots if root]
    existing = [path for path in candidates if os.path.exists(path)]
    return {"exists": bool(existing), "paths_checked": candidates, "existing_paths": existing}


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
