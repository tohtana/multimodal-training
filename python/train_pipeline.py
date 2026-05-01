"""
Pipeline-based training entry point (M6: VLM Migration).

Uses the pipeline framework for DAG-based orchestration of VLM training,
replacing the hardcoded two-stage logic in train_ray.py.

The pipeline structure is defined in the config YAML under the `pipeline` key;
model, training, data, and DeepSpeed sections remain the same as legacy format.
"""

import json
import logging
import multiprocessing as mp
import os
import statistics
import time
import traceback
from pathlib import Path

import hydra
import ray
import torch
from omegaconf import DictConfig

from .pipeline.config_loader import parse_pipeline_dict
from .pipeline.stage import Pipeline
from .pipeline.vlm_placement import build_vlm_stage_groups, configure_vlm_cross_stage_communication
from .pipeline.vlm_runner import VLMPipelineRunner
from .ray.utils import ensure_run_log_file

# Set log file path BEFORE importing logger module
if "RAY_TRAIN_LOG_FILE" not in os.environ:
    ensure_run_log_file()

from .checkpoint import find_latest_checkpoint
from .ray.logger import setup_logging
from .ray.utils import initialize_ray
from .train_ray import normalize_component_config
from .trainer_registry import resolve_trainer

# See: https://docs.ray.io/en/latest/ray-core/patterns/fork-new-processes.html
mp.set_start_method("spawn", force=True)

logger = logging.getLogger(__name__)

config_dir = str(Path(__file__).parent.parent.parent / "configs")


def _build_pipeline_from_config(cfg: DictConfig) -> Pipeline:
    """Parse the pipeline section of the config into a Pipeline object."""
    pipeline_raw = dict(cfg.pipeline)

    # Convert OmegaConf lists to plain Python lists
    stages = [dict(s) for s in pipeline_raw.get("stages", [])]
    edges = [dict(e) for e in pipeline_raw.get("edges", [])]
    resource_sets = [dict(r) for r in pipeline_raw.get("resource_sets", [])]
    placements = [dict(p) for p in pipeline_raw.get("placements", [])]

    raw = {
        "stages": stages,
        "edges": edges,
        "resource_sets": resource_sets,
        "placements": placements,
        "dp_size": pipeline_raw.get("dp_size", cfg.training.get("dp_size", 1)),
        "num_microbatches": pipeline_raw.get("num_microbatches", 1),
    }

    return parse_pipeline_dict(raw)


def _build_stage_config(cfg: DictConfig, component_type: str) -> dict:
    """Build a merged config dict for a stage (same merging logic as train_ray.py)."""
    component_config = dict(cfg[component_type]) if component_type in cfg else {}

    if "training" in cfg:
        component_config.update(dict(cfg.training))
    if "data" in cfg:
        component_config.update(dict(cfg.data))
    if "deepspeed" in cfg:
        component_config.update(dict(cfg.deepspeed))

    component_config = normalize_component_config(component_config, component_type)

    # Set DP/TP configuration
    dp_size = cfg.training.dp_size
    parallel_size = cfg.training.parallel_size

    if component_type == "vision":
        component_config["sequence_parallel_size"] = parallel_size
    elif component_type == "text":
        if component_config.get("parallelism") == "autotp" and component_config.get("autotp_size") is None:
            component_config["autotp_size"] = parallel_size
    elif component_type == "bridge":
        pass  # Bridge needs no SP/TP params

    return component_config


def _load_optimizer_probe_paths() -> dict[str, str | None]:
    audit_path = os.environ.get("VLM_PIPELINE_DENSE_AUDIT_PATH")
    if not audit_path:
        return {}
    try:
        with open(audit_path, "r", encoding="utf-8") as f:
            audit = json.load(f)
    except FileNotFoundError:
        return {}
    return dict(audit.get("probe_parameter_path") or {})


def _write_runtime_metadata(payload: dict) -> None:
    metadata_path = os.environ.get("VLM_PIPELINE_METADATA_PATH")
    if not metadata_path:
        return
    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(percentile) - 1]


def _summarize_training(iteration_records: list[dict], iteration_times: list[float], runtime: dict | None) -> dict:
    losses = [float(record["loss"]) for record in iteration_records]
    probe_results = {
        stage: result
        for record in iteration_records
        for stage, result in (record.get("optimizer_probe") or {}).items()
    }
    parameter_deltas = [
        float(result.get("param_norm_delta", 0.0))
        for result in probe_results.values()
        if result.get("param_norm_delta") is not None
    ]
    max_allocated = 0
    max_reserved = 0
    if runtime is not None:
        for stage in runtime.get("stages", {}).values():
            for memory in stage.get("memory", {}).values():
                max_allocated = max(max_allocated, int(memory.get("max_allocated_bytes", 0)))
                max_reserved = max(max_reserved, int(memory.get("max_reserved_bytes", 0)))

    return {
        "iterations": len(iteration_records),
        "loss_values": losses,
        "loss_finite": all(torch.isfinite(torch.tensor(loss)).item() for loss in losses),
        "backward_completed": bool(iteration_records)
        and all(record.get("backward_completed") for record in iteration_records),
        "optimizer_probe": probe_results,
        "optimizer_update_verified": bool(probe_results)
        and all(result.get("optimizer_update_verified") for result in probe_results.values()),
        "iteration_step_counter_advanced": bool(probe_results)
        and all(result.get("iteration_step_counter_advanced") for result in probe_results.values()),
        "selected_parameter_grad_nonzero": bool(probe_results)
        and all((result.get("before") or {}).get("has_nonzero_grad") for result in probe_results.values()),
        "parameter_norm_delta": min(parameter_deltas) if parameter_deltas else None,
        "expected_optimizer_step_delta": len(iteration_records),
        "actual_optimizer_step_delta": len(iteration_records)
        if probe_results and all(result.get("iteration_step_counter_advanced") for result in probe_results.values())
        else None,
        "iter_time_ms_p50": _percentile([value * 1000.0 for value in iteration_times], 50),
        "iter_time_ms_p90": _percentile([value * 1000.0 for value in iteration_times], 90),
        "cuda_max_allocated_bytes": max_allocated,
        "cuda_max_reserved_bytes": max_reserved,
    }


@hydra.main(config_path=config_dir, config_name="pipeline_sample", version_base=None)
def main(cfg: DictConfig):
    setup_logging(force=True)
    logger.info(f"Pipeline training configuration: {cfg}")

    runner = None
    stage_groups = {}
    run_metadata: dict = {
        "status": "starting",
        "training": {},
        "post_init": None,
        "post_training": None,
        "error_excerpt": None,
    }
    try:
        initialize_ray()

        # Parse pipeline structure
        pipeline = _build_pipeline_from_config(cfg)
        logger.info(f"Pipeline: {[s.name for s in pipeline.stages]} stages, {len(pipeline.edges)} edges")

        # Build per-stage configs using the same merging logic as train_ray.py
        stage_configs: dict[str, dict] = {}
        for stage in pipeline.stages:
            # Get component_type from stage config or infer from name
            component_type = stage.config.get("component_type")
            if component_type is None:
                # Try to get from the pipeline YAML stage definition
                for s in cfg.pipeline.stages:
                    if s.name == stage.name:
                        component_type = s.get("component_type", stage.name)
                        break
                if component_type is None:
                    component_type = stage.name
            stage_configs[stage.name] = _build_stage_config(cfg, component_type)

        parallel_size = cfg.training.parallel_size
        dp_size = cfg.training.dp_size
        logger.info("DP size: %s, global parallel_size: %s", dp_size, parallel_size)

        def trainer_resolver(stage_name: str, stage_cfg: dict):
            component_type = stage_cfg.get("component_type")
            if component_type is None:
                for raw_stage in cfg.pipeline.stages:
                    if raw_stage.name == stage_name:
                        component_type = raw_stage.get("component_type", stage_name)
                        break
            if component_type is None:
                component_type = stage_name

            return resolve_trainer(
                component_type=component_type,
                engine=stage_cfg.get("engine"),
                model_type=stage_cfg["model_type"],
                config=stage_cfg,
            )

        placement_plan, vlm_stage_groups = build_vlm_stage_groups(
            pipeline=pipeline,
            stage_configs=stage_configs,
            trainer_resolver=trainer_resolver,
        )
        stage_groups = {name: group for name, group in vlm_stage_groups.items()}

        for name in stage_groups:
            stage_groups[name].execute_all("build_model")
            logger.info("Stage '%s' model built", name)

        for name in stage_groups:
            stage_groups[name].execute_all("initialize_trainer")
        logger.info("All trainers initialized")

        router = configure_vlm_cross_stage_communication(
            pipeline=pipeline,
            placement_plan=placement_plan,
            stage_configs=stage_configs,
        )

        runner = VLMPipelineRunner(pipeline, stage_groups, placement_plan=placement_plan, router=router)
        run_metadata["post_init"] = runner.collect_runtime_metadata()
        runner.reset_cuda_memory_stats()

        # Training configuration
        num_epochs = cfg.training.num_epochs
        num_iterations = cfg.training.num_iterations
        warmup_steps = cfg.training.warmup_steps
        no_checkpoint = cfg.training.no_checkpoint
        log_interval = cfg.training.log_interval
        clip_grad_norm = cfg.training.get("clip_grad_norm", False)
        max_grad_norm = cfg.training.get("max_grad_norm", 1.0)
        checkpoint_dir = None
        if not no_checkpoint:
            checkpoint_dir = os.path.abspath(cfg.training.checkpoint_dir)
            logger.info(f"Checkpointing enabled. Directory: {checkpoint_dir}")

        # Check for existing checkpoint and auto-load
        start_epoch = 0
        if checkpoint_dir:
            latest_epoch = find_latest_checkpoint(checkpoint_dir)
            if latest_epoch is not None:
                logger.info(f"Found existing checkpoint at epoch {latest_epoch}, loading...")
                success = runner.load_checkpoint(checkpoint_dir, latest_epoch)
                if success:
                    start_epoch = latest_epoch + 1
                    logger.info(f"Resuming training from epoch {start_epoch}")
                else:
                    logger.warning("Failed to load checkpoint, starting from scratch")

        # Training loop
        logger.info(
            f"Starting pipeline training: {num_epochs} epochs, {num_iterations} iterations/epoch, "
            f"warmup={warmup_steps} steps"
        )

        iteration_times = []
        iteration_records = []
        optimizer_probe_paths = _load_optimizer_probe_paths()
        global_step = 0

        for epoch in range(start_epoch, num_epochs):
            logger.info("=" * 60)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            logger.info("=" * 60)

            epoch_start = time.perf_counter()
            epoch_loss = 0.0

            for iteration in range(num_iterations):
                measure_metrics = global_step >= warmup_steps
                iteration_start = time.perf_counter() if measure_metrics else None

                # Run one pipeline iteration
                result = runner.run_iteration(
                    iteration=global_step,
                    dp_size=dp_size,
                    parallel_size=parallel_size,
                    clip_grad_norm=clip_grad_norm,
                    max_grad_norm=max_grad_norm,
                    optimizer_probe_paths=optimizer_probe_paths,
                )

                avg_loss = result["loss"]
                epoch_loss += avg_loss
                iteration_records.append(
                    {
                        "iteration": global_step,
                        "loss": avg_loss,
                        "backward_completed": result.get("backward_completed", False),
                        "optimizer_probe": result.get("optimizer_probe", {}),
                    }
                )

                if measure_metrics and iteration_start is not None:
                    iteration_elapsed = time.perf_counter() - iteration_start
                    iteration_times.append(iteration_elapsed)

                # Log at specified interval
                if (iteration + 1) % log_interval == 0 or iteration == 0:
                    status = "warmup" if global_step < warmup_steps else "training"
                    if measure_metrics and iteration_start is not None:
                        iter_time = time.perf_counter() - iteration_start
                        logger.info(
                            f"Epoch {epoch + 1}/{num_epochs}, Iter {iteration + 1}/{num_iterations} "
                            f"({status}) - Loss: {avg_loss:.4f}, Iteration time: {iter_time:.3f}s"
                        )
                    else:
                        logger.info(
                            f"Epoch {epoch + 1}/{num_epochs}, Iter {iteration + 1}/{num_iterations} "
                            f"({status}) - Loss: {avg_loss:.4f}"
                        )

                if iteration > 100:
                    break

                global_step += 1

            # End of epoch
            epoch_elapsed = time.perf_counter() - epoch_start
            avg_epoch_loss = epoch_loss / max(1, num_iterations)
            logger.info("=" * 60)
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_elapsed:.2f}s - Avg Loss: {avg_epoch_loss:.4f}"
            )
            logger.info("=" * 60)

            # Save checkpoint
            if checkpoint_dir:
                runner.save_checkpoint(checkpoint_dir, epoch, config=cfg)

        run_metadata["post_training"] = runner.collect_runtime_metadata()
        run_metadata["training"] = _summarize_training(
            iteration_records,
            iteration_times,
            run_metadata["post_training"],
        )
        run_metadata["status"] = "ok"

        # Final summary
        if iteration_times:
            total_time = sum(iteration_times)
            avg_time = total_time / len(iteration_times)
            logger.info("=" * 80)
            logger.info(
                f"Pipeline training completed! {num_epochs} epochs, "
                f"{len(iteration_times)} measured steps (avg {avg_time:.3f}s/step)"
            )
            logger.info("=" * 80)
        else:
            logger.info("Pipeline training completed!")
    except BaseException as exc:
        run_metadata["status"] = "failed"
        run_metadata["error_excerpt"] = "".join(traceback.format_exception_only(type(exc), exc))[:4096]
        if runner is not None and stage_groups and run_metadata.get("post_training") is None:
            try:
                run_metadata["post_failure"] = runner.collect_runtime_metadata()
            except Exception as metadata_exc:
                run_metadata["post_failure_error"] = str(metadata_exc)[:4096]
        raise
    finally:
        _write_runtime_metadata(run_metadata)
        for group in stage_groups.values():
            try:
                group.shutdown()
            except Exception:
                logger.exception("Failed to shut down VLM stage group")
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
