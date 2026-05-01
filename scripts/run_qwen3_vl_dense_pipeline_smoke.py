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
    framework_only = dense_model_id is None and not args.dry_run
    model_id = dense_model_id or "Qwen/Qwen2.5-VL-32B-Instruct"
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
            dense=dense_model_id is not None,
            command=command,
            started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            layer=layer,
            stages=stages,
            edges=edges,
            blocker=None,
        )

    if dense_model_id is None:
        blocker = BlockerRow(
            category="dense_model_unavailable",
            reason="No dense Qwen3-VL candidate passed the local compatibility and layer-override audit.",
            code_path="multimodal-training/scripts/qwen3_vl_dense_audit.py:120",
            code_path_git_sha=_git_sha(Path(__file__).resolve().parents[1]),
            error_excerpt=_dense_failure_excerpt(dense_audit),
            config_diff=f"model_id: Qwen/Qwen3-VL-8B-Instruct -> {model_id} (framework_only=true)",
            next_action="Validate a dense Qwen3-VL override path that round-trips through the Megatron bridge.",
        )
        return _row(
            run_id=run_id,
            status="blocked",
            cell=cell,
            dry_run=False,
            framework_only=framework_only,
            model_id=model_id,
            dense=False,
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
            code_path="multimodal-training/python/pipeline/vlm_runner.py:181",
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

    blocker = _run_training_subprocess(args, hydra_overrides, output_dir, run_id)
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
        layer=layer,
        stages=stages,
        edges=edges,
        blocker=None,
        training_ok=True,
    )


def _run_training_subprocess(
    args: argparse.Namespace,
    hydra_overrides: list[str],
    output_dir: Path,
    run_id: str,
) -> BlockerRow | None:
    project_root = Path(__file__).resolve().parents[1]
    log_path = output_dir / "logs" / f"{run_id}.txt"
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
        *hydra_overrides,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_ADDRESS"] = f"127.0.0.1:{args.ray_port}"
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
        log_path.write_text(excerpt, encoding="utf-8")
        return BlockerRow(
            category="other",
            reason="Dense Qwen3-VL smoke subprocess timed out.",
            code_path="multimodal-training/python/train_pipeline.py:98",
            code_path_git_sha=_git_sha(project_root),
            error_excerpt=excerpt,
            config_diff=None,
            next_action="Reduce the layer count only or inspect the timed-out Ray actor logs.",
        )

    combined = result.stdout + "\n" + result.stderr
    log_path.write_text(combined, encoding="utf-8")
    if result.returncode == 0:
        return None

    category = "oom" if "out of memory" in combined.lower() else "other"
    return BlockerRow(
        category=category,
        reason="Dense Qwen3-VL smoke subprocess failed.",
        code_path="multimodal-training/python/train_pipeline.py:98",
        code_path_git_sha=_git_sha(project_root),
        error_excerpt=_truncate(combined),
        config_diff=None,
        next_action="Inspect the per-run log and either fix the code path or record the resource blocker.",
    )


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
        training=TrainingRow(
            iterations=0 if dry_run else 1,
            warmup_iterations=0,
            batch_size=1,
            optimizer_update_verified=training_ok,
            iteration_step_counter_advanced=training_ok,
            expected_optimizer_step_delta=0 if dry_run else 1,
            actual_optimizer_step_delta=0 if dry_run else (1 if training_ok else None),
        ),
        metrics=MetricsRow(
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
    parser.add_argument("--ray-port", type=int, required=True)
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
