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
from .placement import PlacementPlan
from .router import CrossStageRouter
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
        placement_plan: PlacementPlan | None = None,
        router: CrossStageRouter | None = None,
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
        self.placement_plan = placement_plan
        self.router = router

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
        optimizer_probe_paths: dict[str, str | None] | None = None,
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
                    pred_refs = self._route_forward_refs(pred_name, name, stage_output_refs[pred_name])
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
                    self._ensure_backward_routing_supported(name, succ_name)
                    grad_refs = stage_grad_refs[succ_name]
                    refs = group.execute_all_async("backward_step", grad_refs)

                stage_grad_refs[name] = refs

        # Extract loss from terminal stage (wait for forward results)
        avg_loss = self._extract_loss(stage_output_refs)

        # Wait for all backward passes to complete
        for name in self.topo_order:
            if name in stage_grad_refs:
                ray.get(stage_grad_refs[name])

        optimizer_probe_before: dict[str, Any] = {}
        if optimizer_probe_paths:
            for name, parameter_path in optimizer_probe_paths.items():
                if name not in self.stage_groups or not parameter_path:
                    continue
                actor = self.stage_groups[name]._actors[0]
                optimizer_probe_before[name] = ray.get(
                    actor.get_optimizer_probe_snapshot.remote(parameter_path, True)
                )

        # Gradient clipping
        global_grad_norm = None
        if clip_grad_norm:
            global_grad_norm = self._compute_global_grad_norm(dp_size, parallel_size)

        # Optimizer steps (all stages)
        for name in self.topo_order:
            self.stage_groups[name].execute_all("optimizer_step", global_grad_norm)

        optimizer_probe_after: dict[str, Any] = {}
        for name, before in optimizer_probe_before.items():
            actor = self.stage_groups[name]._actors[0]
            optimizer_probe_after[name] = ray.get(actor.verify_optimizer_update.remote(before))

        return {
            "loss": avg_loss,
            "global_grad_norm": global_grad_norm,
            "backward_completed": True,
            "optimizer_probe": optimizer_probe_after,
        }

    def reset_cuda_memory_stats(self) -> None:
        for name in self.topo_order:
            self.stage_groups[name].execute_all("reset_cuda_memory_stats")

    def collect_runtime_metadata(self) -> dict[str, Any]:
        """Collect actor placement, process-group, layer, memory, and edge metadata."""
        stages: dict[str, Any] = {}
        for name in self.topo_order:
            group = self.stage_groups[name]
            actor_runtime = ray.get(group.execute_all_async("get_runtime_metadata"))
            layer_counts = ray.get(group.execute_all_async("get_effective_layer_counts"))
            memory = ray.get(group.execute_all_async("get_cuda_memory_stats"))
            stages[name] = {
                "resource_set": getattr(group, "resource_set", None),
                "actor_count": len(group._actors),
                "physical_gpu_ids": getattr(group, "physical_gpu_ids", {}),
                "cuda_visible_devices": getattr(group, "cuda_visible_devices", {}),
                "requested_device_ids": getattr(group, "requested_device_ids", None),
                "placement_match": getattr(group, "placement_match", None),
                "actor_runtime": {idx: value for idx, value in enumerate(actor_runtime)},
                "layer_counts": {idx: value for idx, value in enumerate(layer_counts)},
                "memory": {idx: value for idx, value in enumerate(memory)},
            }

        edges: list[dict[str, Any]] = []
        for edge in self.pipeline.edges:
            routing = self.router.get_routing(edge.src, edge.dst) if self.router is not None else None
            edges.append(
                {
                    "from_stage": edge.src,
                    "to_stage": edge.dst,
                    "transfer_tier": self.router.get_transport(edge.src, edge.dst) if self.router is not None else "unknown",
                    "routing_map": {
                        "num_src": routing.num_src if routing is not None else len(self.stage_groups[edge.src]._actors),
                        "num_dst": routing.num_dst if routing is not None else len(self.stage_groups[edge.dst]._actors),
                        "src_to_dst": routing.src_to_dst if routing is not None else {},
                        "dst_to_src": routing.dst_to_src if routing is not None else {},
                    },
                }
            )
        return {"stages": stages, "edges": edges}

    def _route_forward_refs(self, src_stage: str, dst_stage: str, pred_refs: list) -> list:
        if self.router is None:
            return pred_refs

        routing = self.router.get_routing(src_stage, dst_stage)
        if routing.num_src != len(pred_refs):
            raise RuntimeError(
                f"VLM routing source count mismatch for {src_stage}->{dst_stage}: "
                f"routing.num_src={routing.num_src}, refs={len(pred_refs)}"
            )

        dst_group = self.stage_groups[dst_stage]
        routed_refs = []
        for dst_rank in range(len(dst_group._actors)):
            if dst_rank not in routing.dst_to_src:
                raise RuntimeError(f"No source rank routed to {dst_stage}[{dst_rank}]")
            routed_refs.append(pred_refs[routing.dst_to_src[dst_rank]])
        return routed_refs

    def _ensure_backward_routing_supported(self, src_stage: str, dst_stage: str) -> None:
        if self.router is None or self.router.is_symmetric(src_stage, dst_stage):
            return
        routing = self.router.get_routing(src_stage, dst_stage)
        raise NotImplementedError(
            "Asymmetric real-VLM backward routing is not implemented: "
            f"{src_stage}->{dst_stage} has {routing.num_src} source actors and {routing.num_dst} "
            "destination actors. Vision-gradient aggregation across multiple downstream text ranks "
            "must be implemented before this row can be marked supported."
        )

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

        Iterates per-stage, dispatching aggregation by parallelism type reported
        in each actor's norm contribution dict.  Supports arbitrary N-stage
        pipelines (not limited to vision + text).
        """
        total_norm_sq = 0.0

        for name in self.topo_order:
            group = self.stage_groups[name]
            norm_refs = group.execute_all_async("compute_grad_norm_contribution")
            norms = ray.get(norm_refs)

            if not norms:
                continue

            ptype = norms[0]["type"]

            if ptype == "sequence":
                # SP: all ranks in a DP replica share identical grads after sync.
                # Sample one actor per DP replica: indices 0, parallel_size, 2*parallel_size, ...
                for dp_rank in range(dp_size):
                    idx = dp_rank * parallel_size
                    if idx < len(norms):
                        total_norm_sq += norms[idx]["norm_sq"]

            elif ptype == "tensor":
                # TP: first actor per DP replica provides replicated + sharded norms.
                for dp_rank in range(dp_size):
                    idx = dp_rank * parallel_size
                    if idx < len(norms):
                        total_norm_sq += norms[idx]["replicated_norm_sq"] + norms[idx]["sharded_norm_sq"]

            else:
                # "deepspeed", "none", or unknown: sum all contributions.
                total_norm_sq += sum(n["norm_sq"] for n in norms)

        return math.sqrt(total_norm_sq)

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
