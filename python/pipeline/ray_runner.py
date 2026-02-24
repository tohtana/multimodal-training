"""Ray-based pipeline runner: dispatches forward/backward/optimizer to ActorGroups."""

from __future__ import annotations

import logging
import math
from typing import Any

import ray

import torch

from ..ray.payloads import StageGradients, StageOutputs
from .dag import PipelineDAG
from .placement import PipelineActorGroup, PlacementPlan
from .router import CrossStageRouter
from .scheduler import OpType, PipelineScheduler, SequentialScheduler
from .stage import Pipeline

logger = logging.getLogger(__name__)


class PipelineIterationError(Exception):
    """Raised when a pipeline iteration fails."""

    def __init__(self, stage_name: str, op: str, microbatch_id: int = 0, cause: Exception | None = None):
        self.stage_name = stage_name
        self.op = op
        self.microbatch_id = microbatch_id
        self.cause = cause
        msg = f"Pipeline iteration failed at stage '{stage_name}', op={op}, microbatch={microbatch_id}"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


class RayPipelineRunner:
    """Runs a pipeline using Ray ActorGroups.

    For T0 (all stages on same ActorGroup): forward and backward are dispatched
    as individual per-stage calls to each actor, with ObjectRefs passed between
    calls to avoid CUDA tensor serialization to the driver.

    For T2 (stages on different ActorGroups): ObjectRefs are passed directly
    between actors; RDT/NCCL transports CUDA tensors GPU-to-GPU without
    hitting the driver.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        placement_plan: PlacementPlan,
        failure_injection: dict | None = None,
        scheduler: PipelineScheduler | None = None,
    ):
        self.pipeline = pipeline
        self.plan = placement_plan
        self.dag = PipelineDAG(pipeline)
        self.topo_order = self.dag.topological_sort()
        self.scheduler = scheduler or SequentialScheduler()
        self.failure_injection = failure_injection
        self.router = CrossStageRouter(self.dag, self.plan)

        # Pre-compute: check if all stages share a single ActorGroup (T0)
        group_ids = {id(self.plan.stage_to_actor_group[name]) for name in self.topo_order}
        self._all_same_group = len(group_ids) == 1

        # Pre-compute successors map for run_full_backward
        self._stage_successors = {name: self.dag.successors(name) for name in self.topo_order}

        # Set up NCCL collective groups for cross-GPU edges (T2)
        if self.router.has_cross_group_edges():
            self.router.setup_collective_groups()

    def run_iteration(
        self,
        data: Any = None,
        labels: Any = None,
        max_norm: float | None = None,
        iteration: int = 0,
        num_microbatches: int = 1,
    ) -> dict[str, Any]:
        """Run one iteration of the pipeline.

        Args:
            num_microbatches: Number of microbatches to split the batch into.
                When > 1, gradients are accumulated across microbatches before
                the optimizer step.
        """

        # Check failure injection for backward
        for name in self.topo_order:
            if self._should_inject_failure(name, "backward", iteration):
                raise PipelineIterationError(name, "backward", 0)
            if self._should_inject_failure(name, "forward", iteration):
                raise PipelineIterationError(name, "forward", 0)

        # T0 optimized path only for single microbatch + all same group
        if self._all_same_group and num_microbatches == 1:
            return self._run_iteration_t0(data, labels, max_norm, iteration)
        else:
            return self._run_iteration_mixed(data, labels, max_norm, iteration, num_microbatches)

    def _run_iteration_t0(
        self,
        data: Any,
        labels: Any,
        max_norm: float | None,
        iteration: int,
    ) -> dict[str, Any]:
        """Optimized path: all stages on same ActorGroup (T0 zero-copy).

        Forward passes ObjectRefs between stages. Backward runs as a batched
        call on each actor to avoid CUDA tensors hitting the driver.
        """
        # Forward: dispatch per stage, passing ObjectRefs between same-actor calls
        group = self.plan.stage_to_actor_group[self.topo_order[0]]
        stage_output_refs: dict[str, list] = {}

        for stage_name in self.topo_order:
            preds = self.dag.predecessors(stage_name)
            if not preds:
                # Source stage
                refs = []
                for actor in group.actors:
                    ref = actor.forward_step.remote(stage_name, StageOutputs(activations=data), labels)
                    refs.append(ref)
            else:
                pred_refs = stage_output_refs[preds[0]]
                refs = []
                for actor, pred_ref in zip(group.actors, pred_refs):
                    ref = actor.forward_step.remote(stage_name, pred_ref, labels)
                    refs.append(ref)
            stage_output_refs[stage_name] = refs

        # Backward: run all stages in reverse order on each actor (in-process, no serialization)
        reversed_stages = list(reversed(self.topo_order))
        backward_refs = []
        for actor in group.actors:
            ref = actor.run_full_backward.remote(reversed_stages, self._stage_successors)
            backward_refs.append(ref)
        ray.get(backward_refs)  # Returns bool True (scalar, safe to deserialize)

        # Extract loss (scalar only)
        loss_value = None
        for stage_cfg in self.pipeline.stages:
            if stage_cfg.is_terminal:
                loss_value = ray.get(group.actors[0].get_last_loss.remote(stage_cfg.name))

        # Compute global gradient norm (scalar only)
        total_norm_sq = 0.0
        for name in self.topo_order:
            norm_sq = ray.get(group.actors[0].compute_grad_norm_sq.remote(name))
            total_norm_sq += norm_sq
        global_grad_norm = math.sqrt(total_norm_sq)

        # Optimizer step
        opt_refs = []
        for name in self.topo_order:
            for actor in group.actors:
                ref = actor.optimizer_step.remote(
                    name,
                    global_grad_norm=global_grad_norm if max_norm is not None else None,
                    max_norm=max_norm or 1.0,
                )
                opt_refs.append(ref)
        ray.get(opt_refs)

        return {"loss": loss_value, "global_grad_norm": global_grad_norm}

    def _run_iteration_mixed(
        self,
        data: Any,
        labels: Any,
        max_norm: float | None,
        iteration: int,
        num_microbatches: int = 1,
    ) -> dict[str, Any]:
        """General path: stages on different ActorGroups (mixed T0/T2).

        Supports multiple microbatches with gradient accumulation.
        ObjectRefs are passed directly between actors — the driver never
        calls ray.get() on tensor-containing refs.
        """
        schedule = self.scheduler.generate_schedule(self.topo_order, num_microbatches)

        # Split batch into microbatches
        mb_data, mb_labels = self._split_batch(data, labels, num_microbatches)

        # Per-stage, per-microbatch ref tracking
        # stage_output_refs[stage_name][microbatch_id] = list of refs (one per actor)
        stage_output_refs: dict[str, dict[int, list]] = {}
        stage_grad_refs: dict[str, dict[int, list]] = {}

        for step in schedule:
            group = self.plan.stage_to_actor_group[step.stage_name]
            mb_id = step.microbatch_id

            if step.op == OpType.FORWARD:
                preds = self.dag.predecessors(step.stage_name)
                succs = self.dag.successors(step.stage_name)
                use_rdt = any(not self.router.is_same_group(step.stage_name, s) for s in succs)

                if not preds:
                    refs = [
                        self._forward_remote(
                            actor, step.stage_name,
                            StageOutputs(activations=mb_data[mb_id]),
                            mb_labels[mb_id], use_rdt,
                        )
                        for actor in group.actors
                    ]
                else:
                    pred_name = preds[0]
                    pred_refs = stage_output_refs[pred_name][mb_id]
                    routing = self.router.get_routing(pred_name, step.stage_name)

                    refs = []
                    for dst_rank, actor in enumerate(group.actors):
                        src_rank = routing.dst_to_src[dst_rank]
                        pred_ref = pred_refs[src_rank]
                        ref = self._forward_remote(actor, step.stage_name, pred_ref, mb_labels[mb_id], use_rdt)
                        refs.append(ref)

                stage_output_refs.setdefault(step.stage_name, {})[mb_id] = refs

            elif step.op == OpType.BACKWARD:
                succs_of = self.dag.successors(step.stage_name)
                preds_of = self.dag.predecessors(step.stage_name)
                use_rdt_upstream = any(not self.router.is_same_group(p, step.stage_name) for p in preds_of)

                if not succs_of:
                    refs = [
                        self._backward_remote(actor, step.stage_name, None, use_rdt_upstream)
                        for actor in group.actors
                    ]
                else:
                    succ_name = succs_of[0]
                    succ_grad_refs = stage_grad_refs.get(succ_name, {}).get(mb_id, [])
                    routing = self.router.get_routing(step.stage_name, succ_name)
                    use_rdt_downstream = not self.router.is_same_group(step.stage_name, succ_name)

                    refs = []
                    for src_rank, actor in enumerate(group.actors):
                        dst_ranks = routing.src_to_dst[src_rank]
                        grad_ref = self._gather_grad_refs(
                            actor, succ_grad_refs, dst_ranks, use_rdt_downstream
                        )
                        ref = self._backward_remote(actor, step.stage_name, grad_ref, use_rdt_upstream)
                        refs.append(ref)

                stage_grad_refs.setdefault(step.stage_name, {})[mb_id] = refs

        # Loss (scalar only — safe for ray.get; actor task ordering ensures backward is done)
        loss_value = None
        for stage_cfg in self.pipeline.stages:
            if stage_cfg.is_terminal:
                group = self.plan.stage_to_actor_group[stage_cfg.name]
                loss_value = ray.get(group.actors[0].get_last_loss.remote(stage_cfg.name))

        # Grad norm (scalar only)
        total_norm_sq = 0.0
        for name in self.topo_order:
            group = self.plan.stage_to_actor_group[name]
            norm_sq = ray.get(group.actors[0].compute_grad_norm_sq.remote(name))
            total_norm_sq += norm_sq
        global_grad_norm = math.sqrt(total_norm_sq)

        # Optimizer step
        opt_refs = []
        for name in self.topo_order:
            group = self.plan.stage_to_actor_group[name]
            for actor in group.actors:
                ref = actor.optimizer_step.remote(
                    name,
                    global_grad_norm=global_grad_norm if max_norm else None,
                    max_norm=max_norm or 1.0,
                )
                opt_refs.append(ref)
        ray.get(opt_refs)

        return {"loss": loss_value, "global_grad_norm": global_grad_norm}

    @staticmethod
    def _split_batch(
        data: Any, labels: Any, num_microbatches: int
    ) -> tuple[list[Any], list[Any]]:
        """Split data and labels into microbatches along dim 0."""
        if num_microbatches == 1:
            return [data], [labels]
        if isinstance(data, torch.Tensor):
            mb_data = list(data.chunk(num_microbatches, dim=0))
        else:
            mb_data = [data] * num_microbatches
        if isinstance(labels, torch.Tensor):
            mb_labels = list(labels.chunk(num_microbatches, dim=0))
        else:
            mb_labels = [labels] * num_microbatches
        return mb_data, mb_labels

    @staticmethod
    def _forward_remote(actor, stage_name, inputs, labels, use_rdt: bool):
        """Dispatch forward_step, optionally using RDT/NCCL for the return value."""
        if use_rdt:
            return actor.forward_step.options(tensor_transport="nccl").remote(stage_name, inputs, labels)
        return actor.forward_step.remote(stage_name, inputs, labels)

    @staticmethod
    def _backward_remote(actor, stage_name, downstream_grad, use_rdt: bool):
        """Dispatch backward_step, optionally using RDT/NCCL for the return value."""
        if use_rdt:
            return actor.backward_step.options(tensor_transport="nccl").remote(stage_name, downstream_grad)
        return actor.backward_step.remote(stage_name, downstream_grad)

    @staticmethod
    def _gather_grad_refs(actor, succ_grad_refs, dst_ranks, use_rdt: bool):
        """Gather and aggregate gradient refs for one src actor.

        For 1:1 mapping, returns the single grad ref directly.
        For 1:N expansion, aggregates N grad refs via sum_gradients on the actor.
        """
        if not succ_grad_refs or not dst_ranks:
            return None
        if len(dst_ranks) == 1:
            return succ_grad_refs[dst_ranks[0]]
        # Multiple successor actors → aggregate gradients on this actor
        grad_refs = [succ_grad_refs[d] for d in dst_ranks]
        if use_rdt:
            return actor.sum_gradients.options(tensor_transport="nccl").remote(grad_refs)
        return actor.sum_gradients.remote(grad_refs)

    def _should_inject_failure(self, stage_name: str, op: str, iteration: int) -> bool:
        if self.failure_injection is None:
            return False
        return (
            self.failure_injection.get("stage_name") == stage_name
            and self.failure_injection.get("op") == op
            and self.failure_injection.get("iteration") == iteration
        )

    def shutdown(self) -> None:
        """Shut down the runner. Idempotent."""
        pass
