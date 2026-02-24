"""VLM Pipeline Runner: orchestrates VLM training using the pipeline DAG with existing trainers.

Uses PipelineDAG for stage ordering and scheduling, dispatching to existing
VisionTrainer/TextTrainer actors via their native interface. This validates the
pipeline framework (DAG, scheduling, config) while maintaining exact training
parity with the legacy train_ray.py path.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

import ray
import torch

from .dag import PipelineDAG
from .scheduler import OpType, PipelineScheduler, SequentialScheduler
from .stage import Pipeline

logger = logging.getLogger(__name__)


class VLMPipelineRunner:
    """Pipeline runner for VLM training using existing VisionTrainer/TextTrainer actors.

    The runner uses the pipeline's DAG scheduling to determine stage ordering,
    then dispatches to existing trainer ActorGroups via their native interface
    (forward_step, backward_step, optimizer_step, etc.).

    This provides pipeline framework validation while reusing the existing
    trainer model building, data loading, and training logic for exact parity.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        stage_groups: dict[str, Any],
        scheduler: PipelineScheduler | None = None,
    ):
        """
        Args:
            pipeline: Pipeline configuration object (stages, edges, etc.).
            stage_groups: Dict mapping stage name to ActorGroup (from ray.actor_group).
            scheduler: Pipeline scheduler (defaults to SequentialScheduler).
        """
        self.pipeline = pipeline
        self.stage_groups = stage_groups
        self.dag = PipelineDAG(pipeline)
        self.topo_order = self.dag.topological_sort()
        self.scheduler = scheduler or SequentialScheduler()

        # Identify source and terminal stages from the pipeline config
        self._source_stages = [s.name for s in pipeline.stages if s.is_source]
        self._terminal_stages = [s.name for s in pipeline.stages if s.is_terminal]

        if not self._source_stages:
            raise ValueError("Pipeline must have at least one source stage")
        if not self._terminal_stages:
            raise ValueError("Pipeline must have at least one terminal stage")

        logger.info(
            f"VLMPipelineRunner initialized: stages={self.topo_order}, "
            f"source={self._source_stages}, terminal={self._terminal_stages}"
        )

    def run_iteration(
        self,
        iteration: int,
        dp_size: int = 1,
        parallel_size: int = 1,
        clip_grad_norm: bool = False,
        max_grad_norm: float = 1.0,
    ) -> dict[str, Any]:
        """Run one pipeline iteration following the DAG schedule.

        Args:
            iteration: Current training iteration (global step).
            dp_size: Data parallel size (for grad norm aggregation).
            parallel_size: TP/SP parallel size (for grad norm aggregation).
            clip_grad_norm: Whether to clip gradients.
            max_grad_norm: Maximum gradient norm for clipping.

        Returns:
            Dict with "loss" and optionally "global_grad_norm".
        """
        schedule = self.scheduler.generate_schedule(self.topo_order, num_microbatches=1)

        # Zero gradients for all stages
        for name in self.topo_order:
            self.stage_groups[name].execute_all("zero_grad")

        # Track output refs per stage (for passing between stages)
        stage_output_refs: dict[str, list] = {}
        stage_grad_refs: dict[str, list] = {}

        for step in schedule:
            name = step.stage_name
            group = self.stage_groups[name]

            if step.op == OpType.FORWARD:
                preds = self.dag.predecessors(name)

                if not preds:
                    # Source stage (e.g., vision): forward_step(iteration)
                    refs = group.execute_all_async("forward_step", iteration)
                else:
                    # Non-source (e.g., text): forward_step(upstream_refs, iteration_list)
                    pred_name = preds[0]
                    pred_refs = stage_output_refs[pred_name]
                    iteration_list = [iteration] * len(group._actors)
                    refs = group.execute_all_async("forward_step", pred_refs, iteration_list)

                stage_output_refs[name] = refs

            elif step.op == OpType.BACKWARD:
                succs = self.dag.successors(name)

                if not succs:
                    # Terminal stage (e.g., text): backward_step() — no args
                    refs = group.execute_all_async("backward_step")
                else:
                    # Non-terminal (e.g., vision): backward_step(downstream_grad_refs)
                    succ_name = succs[0]
                    grad_refs = stage_grad_refs[succ_name]
                    refs = group.execute_all_async("backward_step", grad_refs)

                stage_grad_refs[name] = refs

        # Extract loss from terminal stage (wait for forward results)
        avg_loss = self._extract_loss(stage_output_refs)

        # Wait for all backward passes to complete
        for name in self.topo_order:
            if name in stage_grad_refs:
                ray.get(stage_grad_refs[name])

        # Gradient clipping
        global_grad_norm = None
        if clip_grad_norm:
            global_grad_norm = self._compute_global_grad_norm(dp_size, parallel_size)

        # Optimizer steps (all stages)
        for name in self.topo_order:
            self.stage_groups[name].execute_all("optimizer_step", global_grad_norm)

        return {"loss": avg_loss, "global_grad_norm": global_grad_norm}

    def _extract_loss(self, stage_output_refs: dict[str, list]) -> float:
        """Extract average loss from terminal stage forward results."""
        terminal_name = self._terminal_stages[0]
        terminal_refs = stage_output_refs.get(terminal_name, [])
        if not terminal_refs:
            return 0.0

        terminal_results = ray.get(terminal_refs)
        loss_values = []
        for result in terminal_results:
            if isinstance(result, dict):
                loss = result.get("loss")
                loss_values.append(loss.item() if torch.is_tensor(loss) else loss)
            else:
                loss_values.append(result.item() if torch.is_tensor(result) else result)

        return sum(loss_values) / len(loss_values) if loss_values else 0.0

    def _compute_global_grad_norm(self, dp_size: int, parallel_size: int) -> float:
        """Compute global gradient norm across all stages.

        Uses existing compute_grad_norm_contribution() on each actor and
        aggregates using the legacy aggregate_grad_norms() logic.
        """
        from ..train_ray import aggregate_grad_norms

        # Collect per-stage norms
        all_norms: dict[str, list[dict]] = {}
        for name in self.topo_order:
            group = self.stage_groups[name]
            norm_refs = group.execute_all_async("compute_grad_norm_contribution")
            all_norms[name] = ray.get(norm_refs)

        # For two-stage VLM, delegate to legacy aggregation
        # The legacy function handles different parallelism types (SP, TP, AutoTP, etc.)
        vision_norms = []
        text_norms = []
        for name in self.topo_order:
            stage = self.pipeline.get_stage(name)
            is_source = stage.is_source
            if is_source:
                vision_norms.extend(all_norms[name])
            else:
                text_norms.extend(all_norms[name])

        return aggregate_grad_norms(vision_norms, text_norms, dp_size, parallel_size)

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        epoch: int,
        config=None,
    ) -> None:
        """Save checkpoint for all pipeline stages.

        Args:
            checkpoint_dir: Base directory for checkpoints.
            epoch: Current epoch number.
            config: Optional config for metadata.
        """
        import json
        import os
        from datetime import datetime

        all_paths = {}
        for name in self.topo_order:
            group = self.stage_groups[name]
            refs = group.execute_all_async("save_checkpoint", checkpoint_dir, epoch)
            paths = ray.get(refs)
            all_paths[name] = paths

        # Write metadata
        metadata = {
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
            "format_version": 1,
            "stage_names": list(self.topo_order),
        }
        for name, paths in all_paths.items():
            metadata[f"{name}_checkpoints"] = paths

        epoch_dir = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        os.makedirs(epoch_dir, exist_ok=True)
        metadata_path = os.path.join(epoch_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved pipeline checkpoint for epoch {epoch} ({len(self.topo_order)} stages)")

    def load_checkpoint(self, checkpoint_dir: str, epoch: int) -> bool:
        """Load checkpoint for all pipeline stages.

        Args:
            checkpoint_dir: Base directory for checkpoints.
            epoch: Epoch number to load.

        Returns:
            True if all stages loaded successfully.
        """
        all_success = True
        for name in self.topo_order:
            group = self.stage_groups[name]
            refs = group.execute_all_async("load_checkpoint", checkpoint_dir, epoch)
            results = ray.get(refs)
            if not all(results):
                logger.error(f"Failed to load checkpoint for stage '{name}' at epoch {epoch}")
                all_success = False

        return all_success

    def shutdown(self) -> None:
        """Shut down the runner. Idempotent."""
        pass
