"""Emit dense Qwen3-VL flexible-pipeline smoke/benchmark artifact rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.pipeline.run_artifact_schema import (
    BlockerRow,
    EdgeRow,
    LayerTruncationRow,
    MetricsRow,
    RunRow,
    StageRow,
    TrainingRow,
    project_summary,
    SUMMARY_PROJECTION,
)


CELL_CHOICES = ("compat", "collocated", "separated", "asymmetric")
NON_DEFAULT_RAY_PORT = 6379


def main() -> None:
    args, hydra_overrides = _parse_args()
    if args.ray_port == NON_DEFAULT_RAY_PORT:
        raise SystemExit("--ray-port must not use Ray's default port 6379")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "configs").mkdir(exist_ok=True)
    _write_env(output_dir)

    dense_audit = _read_json(args.dense_audit) if args.dense_audit.exists() else None
    cells = _requested_cells(args.cells)
    for cell in cells:
        row = _build_row(args, hydra_overrides, output_dir, cell, dense_audit)
        _append_validated_row(output_dir, row)


def _build_row(
    args: argparse.Namespace,
    hydra_overrides: list[str],
    output_dir: Path,
    cell: str,
    dense_audit: dict[str, Any] | None,
) -> RunRow:
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{cell}-{uuid.uuid4().hex[:8]}"
    started = datetime.now(timezone.utc).isoformat()
    command = [sys.executable, str(Path(__file__).name), *sys.argv[1:]]
    config = _load_config(args.config_path, args.config_name)
    _write_config(output_dir, run_id, config)

    dense_model_id = dense_audit.get("selected_model_id") if dense_audit else None
    target_model_id = _target_dense_model_id(dense_audit)
    model_id = dense_model_id or target_model_id
    framework_only = False
    layer = _layer_from_audit(dense_audit)
    stages = _stage_rows_from_config(config, cell)
    edges = _edge_rows_from_config(config, stages)

    if args.dry_run:
        return _row(
            run_id=run_id,
            status="skipped",
            cell=cell,
            dry_run=True,
            framework_only=False,
            model_id=model_id,
            dense=_is_dense_qwen3_vl_id(model_id),
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=None,
        )

    if dense_model_id is None:
        blocker_category = _dense_blocker_category(dense_audit)
        blocker = BlockerRow(
            category=blocker_category,
            reason=_dense_blocker_reason(blocker_category),
            code_path="multimodal-training/scripts/qwen3_vl_dense_audit.py:120",
            code_path_git_sha=_git_sha(Path(__file__).resolve().parents[1]),
            error_excerpt=_dense_failure_excerpt(dense_audit),
            config_diff=None,
            next_action=_dense_blocker_next_action(blocker_category),
        )
        return _row(
            run_id=run_id,
            status="blocked",
            cell=cell,
            dry_run=False,
            framework_only=framework_only,
            model_id=model_id,
            dense=_is_dense_qwen3_vl_id(model_id),
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=blocker,
        )

    if cell == "asymmetric":
        blocker = BlockerRow(
            category="asymmetric_routing_unsupported",
            reason="Forward routing supports M:N refs, but real-VLM backward gradient aggregation is not implemented.",
            code_path="multimodal-training/python/pipeline/vlm_runner.py:240",
            code_path_git_sha=_git_sha(Path(__file__).resolve().parents[1]),
            error_excerpt=(
                '"Asymmetric real-VLM backward routing is not implemented" from '
                "VLMPipelineRunner._ensure_backward_routing_supported"
            ),
            config_diff="pipeline.resource_sets: vision.num_gpus != text.num_gpus",
            next_action=(
                "Implement vision-gradient aggregation for multiple downstream text ranks in VLMPipelineRunner."
            ),
        )
        return _row(
            run_id=run_id,
            status="blocked",
            cell=cell,
            dry_run=False,
            framework_only=False,
            model_id=dense_model_id,
            dense=True,
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=blocker,
        )

    blocker, runtime_metadata = _run_training_subprocess(args, hydra_overrides, output_dir, run_id, dense_audit)
    if blocker is not None:
        return _row(
            run_id=run_id,
            status=blocker.category if blocker.category == "oom" else "blocked",
            cell=cell,
            dry_run=False,
            framework_only=False,
            model_id=dense_model_id,
            dense=True,
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=blocker if blocker.category != "oom" else None,
            error_excerpt=blocker.error_excerpt,
        )

    if runtime_metadata is None:
        blocker = BlockerRow(
            category="other",
            reason="Dense Qwen3-VL subprocess exited successfully but did not emit runtime evidence.",
            code_path="multimodal-training/python/train_pipeline.py:94",
            code_path_git_sha=_git_sha(Path(__file__).resolve().parents[1]),
            error_excerpt="VLM_PIPELINE_METADATA_PATH was missing or empty after subprocess return code 0.",
            config_diff=None,
            next_action="Keep train_pipeline runtime metadata emission enabled and rerun the smoke cell.",
        )
        return _row(
            run_id=run_id,
            status="blocked",
            cell=cell,
            dry_run=False,
            framework_only=False,
            model_id=dense_model_id,
            dense=True,
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=blocker,
        )

    runtime = runtime_metadata.get("post_training") or runtime_metadata.get("post_init")
    runtime_training = runtime_metadata.get("training") or {}
    return _row(
        run_id=run_id,
        status="ok",
        cell=cell,
        dry_run=False,
        framework_only=False,
        model_id=dense_model_id,
        dense=True,
        command=command,
        started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(),
        layer=_layer_from_runtime(dense_audit, runtime),
        stages=_stage_rows_from_runtime(config, runtime, cell),
        edges=_edge_rows_from_runtime(runtime, _stage_rows_from_runtime(config, runtime, cell)),
        blocker=None,
        training=_training_from_runtime(runtime_training, args.max_iterations),
        metrics=_metrics_from_runtime(runtime_training),
    )


def _run_training_subprocess(
    args: argparse.Namespace,
    hydra_overrides: list[str],
    output_dir: Path,
    run_id: str,
    dense_audit: dict[str, Any] | None,
) -> tuple[BlockerRow | None, dict[str, Any] | None]:
    project_root = Path(__file__).resolve().parents[1]
    log_path = output_dir / "logs" / f"{run_id}.txt"
    metadata_path = output_dir / "logs" / f"{run_id}.metadata.json"
    bridge_load_path = _complete_weight_path_from_audit(dense_audit)
    bridge_overrides = []
    if bridge_load_path:
        bridge_overrides = [
            f"+vision.engine_config.bridge_load_path={bridge_load_path}",
            f"+text.engine_config.bridge_load_path={bridge_load_path}",
        ]
    cmd = [
        sys.executable,
        "-m",
        "python.train_pipeline",
        f"--config-path={args.config_path}",
        f"--config-name={args.config_name}",
        f"training.num_iterations={args.max_iterations}",
        "training.num_epochs=1",
        "training.warmup_steps=0",
        "training.no_checkpoint=true",
        *bridge_overrides,
        *hydra_overrides,
    ]
    env = os.environ.copy()
    pythonpath_parts = [
        str(project_root),
        str(project_root.parent / "ms-swift"),
        str(project_root.parent / "Megatron-LM"),
        str(project_root.parent / "DeepSpeed"),
    ]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_ADDRESS"] = env.get("RAY_ADDRESS", "auto")
    env["RAY_NAMESPACE"] = env.get("RAY_NAMESPACE", f"todo-qwen3-vl-{run_id}")
    if args.ray_port is not None:
        env["VLM_REQUESTED_RAY_PORT"] = str(args.ray_port)
    env["VLM_PIPELINE_METADATA_PATH"] = str(metadata_path)
    env["VLM_PIPELINE_DENSE_AUDIT_PATH"] = str(args.dense_audit)
    if bridge_load_path and str(bridge_load_path).startswith("/mnt/cluster_storage/hf_cache/"):
        env.setdefault("HF_HUB_CACHE", "/mnt/cluster_storage/hf_cache")
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        excerpt = _truncate((exc.stdout or "") + "\n" + (exc.stderr or "") + "\n" + str(exc))
        log_path.write_text(_subprocess_log(cmd, bridge_load_path, excerpt), encoding="utf-8")
        return BlockerRow(
            category="other",
            reason="Dense Qwen3-VL smoke subprocess timed out.",
            code_path="multimodal-training/python/train_pipeline.py:98",
            code_path_git_sha=_git_sha(project_root),
            error_excerpt=excerpt,
            config_diff=None,
            next_action="Reduce the layer count only or inspect the timed-out Ray actor logs.",
        ), _read_json(metadata_path) if metadata_path.exists() else None

    combined = result.stdout + "\n" + result.stderr
    log_path.write_text(_subprocess_log(cmd, bridge_load_path, combined), encoding="utf-8")
    if result.returncode == 0:
        if not metadata_path.exists():
            return None, None
        metadata = _read_json(metadata_path)
        if metadata.get("status") != "ok":
            return BlockerRow(
                category="other",
                reason="Dense Qwen3-VL subprocess returned 0 but runtime metadata is not ok.",
                code_path="multimodal-training/python/train_pipeline.py:96",
                code_path_git_sha=_git_sha(project_root),
                error_excerpt=_truncate(json.dumps(metadata.get("error_excerpt") or metadata.get("status"))),
                config_diff=None,
                next_action="Inspect the metadata JSON and make train_pipeline fail non-zero for this condition.",
            ), metadata
        return None, metadata

    category, reason, next_action = _subprocess_failure_classification(combined)
    return BlockerRow(
        category=category,
        reason=reason,
        code_path="multimodal-training/python/train_pipeline.py:98",
        code_path_git_sha=_git_sha(project_root),
        error_excerpt=_truncate(combined),
        config_diff=None,
        next_action=next_action,
    ), _read_json(metadata_path) if metadata_path.exists() else None


def _row(
    *,
    run_id: str,
    status: str,
    cell: str,
    dry_run: bool,
    framework_only: bool,
    model_id: str | None,
    dense: bool,
    command: list[str],
    started_utc: str,
    finished_utc: str | None,
    layer: LayerTruncationRow,
    stages: list[StageRow],
    edges: list[EdgeRow],
    blocker: BlockerRow | None,
    error_excerpt: str | None = None,
    training_ok: bool = False,
    training: TrainingRow | None = None,
    metrics: MetricsRow | None = None,
) -> RunRow:
    return RunRow(
        run_id=run_id,
        status=status,
        cell=cell,
        dry_run=dry_run,
        framework_only=framework_only,
        model_id=model_id,
        dense=dense,
        git_sha=_git_sha(Path(__file__).resolve().parents[1]),
        submodule_shas=_submodule_shas(Path(__file__).resolve().parents[2]),
        command=command,
        started_utc=started_utc,
        finished_utc=finished_utc,
        error_excerpt=error_excerpt,
        layer_truncation=layer,
        stages=stages,
        edges=edges,
        training=training or TrainingRow(
            iterations=0 if dry_run else 1,
            warmup_iterations=0,
            batch_size=1,
            loss_values=[],
            backward_completed=False,
            selected_parameter_grad_nonzero=False,
            parameter_norm_delta=None,
            optimizer_update_verified=training_ok,
            iteration_step_counter_advanced=training_ok,
            expected_optimizer_step_delta=0 if dry_run else 1,
            actual_optimizer_step_delta=0 if dry_run else (1 if training_ok else None),
        ),
        metrics=metrics or MetricsRow(
            loss_finite=training_ok,
            iter_time_ms_p50=None,
            iter_time_ms_p90=None,
            cuda_max_allocated_bytes=None,
            cuda_max_reserved_bytes=None,
        ),
        blocker=blocker,
    )


def _stage_rows_from_config(config: dict[str, Any], cell: str) -> list[StageRow]:
    pipeline = config.get("pipeline", {})
    resource_sets = {rs["name"]: rs for rs in pipeline.get("resource_sets", [])}
    placements = {p["stage"]: p["resource_set"] for p in pipeline.get("placements", [])}
    rows = []
    for stage in pipeline.get("stages", []):
        name = stage["name"]
        rs_name = placements[name]
        rs = resource_sets[rs_name]
        actor_count = int(rs["num_gpus"])
        rows.append(
            StageRow(
                name=name,
                resource_set=rs_name,
                actor_count=actor_count,
                physical_gpu_ids={i: f"unrun:{rs_name}:{i}" for i in range(actor_count)},
                requested_device_ids=rs.get("device_ids"),
                placement_match=False if rs.get("device_ids") else True,
                process_group_world_size=actor_count,
                megatron={"tp": actor_count, "cp": 1, "pp": 1, "ep": 1, "cell": cell},
            )
        )
        if rows[-1].actor_count != actor_count:
            raise AssertionError("StageRow.actor_count must equal ResourceSet.num_gpus")
    return rows


def _edge_rows_from_config(config: dict[str, Any], stages: list[StageRow]) -> list[EdgeRow]:
    stage_counts = {stage.name: stage.actor_count for stage in stages}
    rows = []
    for edge in config.get("pipeline", {}).get("edges", []):
        num_src = stage_counts[edge["src"]]
        num_dst = stage_counts[edge["dst"]]
        rows.append(
            EdgeRow(
                from_stage=edge["src"],
                to_stage=edge["dst"],
                transfer_tier="unrun",
                routing_map={"num_src": num_src, "num_dst": num_dst, "policy": "broadcast"},
            )
        )
    return rows


def _stage_rows_from_runtime(config: dict[str, Any], runtime: dict[str, Any] | None, cell: str) -> list[StageRow]:
    if runtime is None:
        raise ValueError("runtime metadata is required for ok rows")
    pipeline = config.get("pipeline", {})
    resource_sets = {rs["name"]: rs for rs in pipeline.get("resource_sets", [])}
    placements = {p["stage"]: p["resource_set"] for p in pipeline.get("placements", [])}
    runtime_stages = runtime.get("stages", {})
    rows = []
    for stage in pipeline.get("stages", []):
        name = stage["name"]
        rs_name = placements[name]
        rs = resource_sets[rs_name]
        expected_actor_count = int(rs["num_gpus"])
        stage_runtime = runtime_stages.get(name)
        if stage_runtime is None:
            raise ValueError(f"runtime metadata missing stage {name}")
        actor_runtime = _int_keyed_dict(stage_runtime.get("actor_runtime", {}))
        physical_gpu_ids = _int_keyed_dict(stage_runtime.get("physical_gpu_ids", {}))
        cuda_visible_devices = _int_keyed_dict(stage_runtime.get("cuda_visible_devices", {}))
        if not physical_gpu_ids and actor_runtime:
            physical_gpu_ids = {
                rank: str(metadata.get("physical_gpu_id", ""))
                for rank, metadata in actor_runtime.items()
            }
        if not cuda_visible_devices and actor_runtime:
            cuda_visible_devices = {
                rank: str(metadata.get("cuda_visible_devices", ""))
                for rank, metadata in actor_runtime.items()
            }
        first_runtime = actor_runtime.get(0, {})
        process_group = first_runtime.get("process_group", {})
        actor_count = int(stage_runtime.get("actor_count") or len(actor_runtime))
        if actor_count != expected_actor_count:
            raise AssertionError(
                f"StageRow.actor_count must equal ResourceSet.num_gpus for {name}: "
                f"{actor_count} != {expected_actor_count}"
            )
        rows.append(
            StageRow(
                name=name,
                resource_set=rs_name,
                actor_count=actor_count,
                physical_gpu_ids=physical_gpu_ids,
                cuda_visible_devices=cuda_visible_devices,
                requested_device_ids=rs.get("device_ids"),
                placement_match=bool(stage_runtime.get("placement_match")),
                process_group_world_size=int(process_group.get("world_size", actor_count)),
                megatron={**(first_runtime.get("megatron") or {}), "cell": cell},
                actor_runtime=actor_runtime,
            )
        )
    return rows


def _edge_rows_from_runtime(runtime: dict[str, Any] | None, stages: list[StageRow]) -> list[EdgeRow]:
    if runtime is None:
        raise ValueError("runtime metadata is required for ok rows")
    stage_counts = {stage.name: stage.actor_count for stage in stages}
    rows = []
    for edge in runtime.get("edges", []):
        routing_map = dict(edge.get("routing_map") or {})
        routing_map.setdefault("num_src", stage_counts[edge["from_stage"]])
        routing_map.setdefault("num_dst", stage_counts[edge["to_stage"]])
        rows.append(
            EdgeRow(
                from_stage=edge["from_stage"],
                to_stage=edge["to_stage"],
                transfer_tier=edge.get("transfer_tier") or "unknown",
                routing_map=routing_map,
            )
        )
    return rows


def _layer_from_runtime(dense_audit: dict[str, Any] | None, runtime: dict[str, Any] | None) -> LayerTruncationRow:
    layer = (dense_audit or {}).get("layer_truncation", {})
    vision_effective = layer.get("vision_effective_layers")
    language_effective = layer.get("language_effective_layers")
    if runtime is not None:
        vision_counts = _first_actor_layer_counts(runtime, "vision")
        text_counts = _first_actor_layer_counts(runtime, "text")
        vision_effective = vision_counts.get("vision_effective_layers") or vision_effective
        language_effective = text_counts.get("language_effective_layers") or language_effective
    return LayerTruncationRow(
        vision_source_field=layer.get("vision_source_field"),
        language_source_field=layer.get("language_source_field"),
        vision_override_path=layer.get("vision_override_path"),
        language_override_path=layer.get("language_override_path"),
        vision_effective_layers=vision_effective,
        language_effective_layers=language_effective,
        vision_original_layers=layer.get("vision_original_layers"),
        language_original_layers=layer.get("language_original_layers"),
    )


def _training_from_runtime(raw: dict[str, Any], max_iterations: int) -> TrainingRow:
    return TrainingRow(
        iterations=int(raw.get("iterations", max_iterations)),
        warmup_iterations=0,
        batch_size=1,
        loss_values=[float(value) for value in raw.get("loss_values", [])],
        backward_completed=bool(raw.get("backward_completed")),
        selected_parameter_grad_nonzero=bool(raw.get("selected_parameter_grad_nonzero")),
        parameter_norm_delta=raw.get("parameter_norm_delta"),
        optimizer_update_verified=bool(raw.get("optimizer_update_verified")),
        iteration_step_counter_advanced=bool(raw.get("iteration_step_counter_advanced")),
        expected_optimizer_step_delta=int(raw.get("expected_optimizer_step_delta", max_iterations)),
        actual_optimizer_step_delta=raw.get("actual_optimizer_step_delta"),
        optimizer_probe=dict(raw.get("optimizer_probe") or {}),
    )


def _metrics_from_runtime(raw: dict[str, Any]) -> MetricsRow:
    return MetricsRow(
        loss_finite=bool(raw.get("loss_finite")),
        iter_time_ms_p50=raw.get("iter_time_ms_p50"),
        iter_time_ms_p90=raw.get("iter_time_ms_p90"),
        cuda_max_allocated_bytes=raw.get("cuda_max_allocated_bytes"),
        cuda_max_reserved_bytes=raw.get("cuda_max_reserved_bytes"),
    )


def _first_actor_layer_counts(runtime: dict[str, Any], stage_name: str) -> dict[str, Any]:
    layer_counts = ((runtime.get("stages", {}).get(stage_name) or {}).get("layer_counts") or {})
    if "0" in layer_counts:
        return layer_counts["0"]
    if 0 in layer_counts:
        return layer_counts[0]
    return {}


def _int_keyed_dict(raw: dict[Any, Any]) -> dict[int, Any]:
    return {int(key): value for key, value in raw.items()}


def _layer_from_audit(dense_audit: dict[str, Any] | None) -> LayerTruncationRow:
    layer = (dense_audit or {}).get("layer_truncation", {})
    return LayerTruncationRow(
        vision_source_field=layer.get("vision_source_field"),
        language_source_field=layer.get("language_source_field"),
        vision_override_path=layer.get("vision_override_path"),
        language_override_path=layer.get("language_override_path"),
        vision_effective_layers=layer.get("vision_effective_layers"),
        language_effective_layers=layer.get("language_effective_layers"),
        vision_original_layers=layer.get("vision_original_layers"),
        language_original_layers=layer.get("language_original_layers"),
    )


def _append_validated_row(output_dir: Path, row: RunRow) -> None:
    runs_path = output_dir / "runs.jsonl"
    summary_path = output_dir / "summary.csv"
    try:
        payload = row.model_dump_json()
        RunRow.model_validate_json(payload)
    except Exception as exc:
        failure_path = output_dir / "logs" / f"schema_validation_failure_{_now_slug()}.json"
        failure_path.write_text(
            json.dumps({"row": row.model_dump(mode="json"), "error": str(exc)}, indent=2),
            encoding="utf-8",
        )
        raise

    with runs_path.open("a", encoding="utf-8") as f:
        f.write(payload + "\n")

    summary = project_summary(row)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[name for name, _ in SUMMARY_PROJECTION])
        if write_header:
            writer.writeheader()
        writer.writerow(summary)


def _load_config(config_path: Path, config_name: str) -> dict[str, Any]:
    path = config_path / f"{config_name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_config(output_dir: Path, run_id: str, config: dict[str, Any]) -> None:
    with (output_dir / "configs" / f"{run_id}.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=True)


def _write_env(output_dir: Path) -> None:
    env_path = output_dir / "env.json"
    if env_path.exists():
        return
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "git_sha": _git_sha(Path(__file__).resolve().parents[1]),
        "submodule_shas": _submodule_shas(Path(__file__).resolve().parents[2]),
    }
    env_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _requested_cells(raw: str) -> list[str]:
    cells = [cell.strip() for cell in raw.split(",") if cell.strip()]
    unknown = sorted(set(cells) - set(CELL_CHOICES))
    if unknown:
        raise SystemExit(f"unknown cells: {', '.join(unknown)}")
    return cells


def _dense_failure_excerpt(dense_audit: dict[str, Any] | None) -> str:
    if dense_audit is None:
        return "dense_model_audit.json is missing"
    failures = []
    for result in dense_audit.get("per_id_results", []):
        failures.extend(result.get("failures", []))
    text = "; ".join(failures) or "No dense model selected"
    return _truncate(text)


def _target_dense_model_id(dense_audit: dict[str, Any] | None) -> str:
    if dense_audit is not None:
        for result in dense_audit.get("per_id_results", []):
            model_id = result.get("model_id")
            if isinstance(model_id, str) and _is_dense_qwen3_vl_id(model_id):
                return model_id
    return "Qwen/Qwen3-VL-8B-Instruct"


def _is_dense_qwen3_vl_id(model_id: str | None) -> bool:
    if model_id is None:
        return False
    lowered = model_id.lower()
    return "qwen3-vl" in lowered and "a3b" not in lowered and "moe" not in lowered


def _complete_weight_path_from_audit(dense_audit: dict[str, Any] | None) -> str | None:
    if dense_audit is None:
        return None
    complete_paths = []
    for result in dense_audit.get("per_id_results", []):
        if result.get("model_id") == dense_audit.get("selected_model_id"):
            complete_paths = result.get("local_weight_cache", {}).get("complete_paths", [])
            break
    if not complete_paths:
        return None
    return str(complete_paths[0])


def _dense_blocker_category(dense_audit: dict[str, Any] | None) -> str:
    excerpt = _dense_failure_excerpt(dense_audit).lower()
    if "weight shards" in excerpt or "local weights" in excerpt:
        return "dense_weights_unavailable"
    return "dense_model_unavailable"


def _dense_blocker_reason(category: str) -> str:
    if category == "dense_weights_unavailable":
        return "Dense Qwen3-VL compatibility passed, but complete local safetensor shards are unavailable."
    return "No dense Qwen3-VL candidate passed the local compatibility and layer-override audit."


def _dense_blocker_next_action(category: str) -> str:
    if category == "dense_weights_unavailable":
        return "Pre-populate complete Qwen3-VL-8B weight shards under /mnt/local_storage or a shared HF cache."
    return "Validate a dense Qwen3-VL override path that round-trips through the Megatron bridge."


def _subprocess_failure_classification(combined: str) -> tuple[str, str, str]:
    lowered = combined.lower()
    if "out of memory" in lowered:
        return (
            "oom",
            "Dense Qwen3-VL smoke subprocess ran out of GPU memory.",
            "Apply layer-only truncation and rerun the same cell with runtime layer-count validation.",
        )
    if "downloading [model-" in lowered and "safetensors" in lowered:
        return (
            "dense_weights_unavailable",
            "Dense Qwen3-VL smoke reached real Ray actor construction but blocked while downloading missing "
            "weight shards.",
            "Pre-populate complete Qwen3-VL-8B safetensor shards under /mnt/local_storage or a shared HF "
            "cache, then rerun.",
        )
    return (
        "other",
        "Dense Qwen3-VL smoke subprocess failed.",
        "Inspect the per-run log and either fix the code path or record the resource blocker.",
    )


def _subprocess_log(cmd: list[str], bridge_load_path: str | None, body: str) -> str:
    header = [
        "internal_train_pipeline_command:",
        " ".join(cmd),
        f"bridge_load_path={bridge_load_path or '<none>'}",
        "",
    ]
    return "\n".join(header) + body


def _git_sha(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _submodule_shas(path: Path) -> dict[str, str]:
    try:
        output = subprocess.check_output(["git", "-C", str(path), "submodule", "status"], text=True)
    except subprocess.CalledProcessError:
        return {}
    result = {}
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            result[parts[1]] = parts[0].lstrip("+-")
    return result


def _truncate(text: str, limit: int = 4096) -> str:
    return text if len(text) <= limit else text[: limit - 15] + "... [truncated]"


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    project_root = Path(__file__).resolve().parents[1]
    default_artifacts = project_root.parent / "todo/docs/qwen3-vl-dense-pipeline-benchmark/artifacts"
    parser.add_argument("--config-path", type=Path, default=project_root / "configs")
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output-dir", type=Path, default=default_artifacts)
    parser.add_argument("--cells", default="compat,collocated,separated,asymmetric")
    parser.add_argument("--dense-audit", type=Path, default=default_artifacts / "dense_model_audit.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--ray-port", type=int, default=None)
    args, rest = parser.parse_known_args()
    if rest and rest[0] == "--":
        rest = rest[1:]
    return args, rest


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
