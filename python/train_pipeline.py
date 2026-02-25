"""
Pipeline-based training entry point (M6: VLM Migration).

Uses the pipeline framework for DAG-based orchestration of VLM training,
replacing the hardcoded two-stage logic in train_ray.py.

The pipeline structure is defined in the config YAML under the `pipeline` key;
model, training, data, and DeepSpeed sections remain the same as legacy format.
"""

import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

import hydra
import ray
import torch
from omegaconf import DictConfig
from ray.experimental.collective import create_collective_group

from .pipeline.config_loader import parse_pipeline_dict
from .pipeline.stage import Pipeline
from .pipeline.vlm_runner import VLMPipelineRunner
from .ray.utils import ensure_run_log_file

# Set log file path BEFORE importing logger module
if "RAY_TRAIN_LOG_FILE" not in os.environ:
    ensure_run_log_file()

from .checkpoint import find_latest_checkpoint
from .ray.actor_group import ActorGroup
from .ray.logger import setup_logging
from .ray.tensor_transfer import gather_gpu_ids
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


@hydra.main(config_path=config_dir, config_name="pipeline_sample", version_base=None)
def main(cfg: DictConfig):
    setup_logging(force=True)
    logger.info(f"Pipeline training configuration: {cfg}")

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

    # Training parameters
    parallel_size = cfg.training.parallel_size
    dp_size = cfg.training.dp_size
    collocate = cfg.training.collocate
    total_actors = dp_size * parallel_size

    # Collocation factor = number of collocated stage groups sharing the same GPUs
    num_stages = len(pipeline.stages)
    collocation_factor = num_stages if collocate else 1

    logger.info(f"DP size: {dp_size}, Parallel size: {parallel_size}, Total actors: {total_actors}")
    logger.info(f"Collocation: {collocate}, collocation_factor: {collocation_factor}")

    # Create actor groups for each stage
    stage_groups: dict[str, ActorGroup] = {}
    prev_pg = None

    for stage in pipeline.stages:
        stage_cfg = stage_configs[stage.name]
        component_type = stage_cfg.get("component_type")
        if component_type is None:
            for s in cfg.pipeline.stages:
                if s.name == stage.name:
                    component_type = s.get("component_type", stage.name)
                    break

        model_type = stage_cfg["model_type"]
        engine = stage_cfg.get("engine")

        TrainerClass, init_kwargs = resolve_trainer(
            component_type=component_type,
            engine=engine,
            model_type=model_type,
            config=stage_cfg,
        )

        group = ActorGroup(
            stage_cfg,
            TrainerClass,
            num_actors=total_actors,
            collocate=collocate,
            placement_group_handle=prev_pg if collocate else None,
            actor_init_kwargs=init_kwargs,
            collocation_factor=collocation_factor,
        )

        stage_groups[stage.name] = group
        if collocate and prev_pg is None:
            prev_pg = group.placement_group

        logger.info(f"Created ActorGroup for stage '{stage.name}': {total_actors} actors, engine={engine}")

    # Build models and initialize trainers
    for name in stage_groups:
        stage_groups[name].execute_all("build_model")
        logger.info(f"Stage '{name}' model built")

    for name in stage_groups:
        stage_groups[name].execute_all("initialize_trainer")
    logger.info("All trainers initialized")

    # Set up cross-stage communication
    if collocate:
        logger.info("Setting up CUDA IPC for collocated actors...")
        # For collocated stages, set up CUDA IPC between adjacent stages in DAG order.
        # Skip edges involving bridge stages — bridge doesn't use IPC (Ray ObjectRef
        # passing between same-GPU actors is efficient enough for v1).
        ordered_names = [s.name for s in pipeline.stages]
        for i in range(len(ordered_names) - 1):
            src_name = ordered_names[i]
            dst_name = ordered_names[i + 1]

            src_component = stage_configs[src_name].get("component_type", src_name)
            dst_component = stage_configs[dst_name].get("component_type", dst_name)
            if src_component == "bridge" or dst_component == "bridge":
                logger.info(f"Skipping IPC setup for edge {src_name}->{dst_name} (bridge stage)")
                continue

            src_group = stage_groups[src_name]
            dst_group = stage_groups[dst_name]

            src_gpu_ids = gather_gpu_ids(src_group._actors)
            dst_gpu_ids = gather_gpu_ids(dst_group._actors)
            logger.info(f"{src_name} GPU IDs: {src_gpu_ids}, {dst_name} GPU IDs: {dst_gpu_ids}")

            for j in range(total_actors):
                src_group._actors[j].set_receiver_info.remote([dst_gpu_ids[j]], use_ipc=True)
                dst_group._actors[j].set_receiver_info.remote([src_gpu_ids[j]], use_ipc=True)

            ray.get([a.get_rank.remote() for a in src_group._actors])
        logger.info("CUDA IPC setup complete")
    else:
        logger.info("Creating NCCL collective group for cross-GPU communication...")
        all_actors = []
        for name in stage_groups:
            all_actors.extend(stage_groups[name]._actors)
        create_collective_group(all_actors, backend="nccl")

    # Create pipeline runner
    runner = VLMPipelineRunner(pipeline, stage_groups)

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
            )

            avg_loss = result["loss"]
            epoch_loss += avg_loss

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
        avg_epoch_loss = epoch_loss / num_iterations
        logger.info("=" * 60)
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} completed in {epoch_elapsed:.2f}s - Avg Loss: {avg_epoch_loss:.4f}"
        )
        logger.info("=" * 60)

        # Save checkpoint
        if checkpoint_dir:
            runner.save_checkpoint(checkpoint_dir, epoch, config=cfg)

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

    ray.shutdown()


if __name__ == "__main__":
    main()
