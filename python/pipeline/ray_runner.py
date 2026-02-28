"""Ray-based pipeline runner: dispatches forward/backward/optimizer to ActorGroups."""

from __future__ import annotations

import logging
import math
import os
import re
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
        self._verify_t1_cleanup = os.environ.get("MM_VERIFY_T1_CACHE_CLEANUP", "0") == "1"
        self._last_t1_cache_cleanup_stats: list[dict[str, Any]] = []

        # Pre-compute: check if all stages share a single ActorGroup (T0)
        group_ids = {id(self.plan.stage_to_actor_group[name]) for name in self.topo_order}
        self._all_same_group = len(group_ids) == 1
        self._has_t1 = self.router.has_t1_edges()

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
            if self._should_inject_failure(stage_name, "forward", iteration, microbatch_id=0):
                raise PipelineIterationError(stage_name, "forward", 0)
            preds = self.dag.predecessors(stage_name)
            if not preds:
                # Source stage
                refs = []
                for actor in group.actors:
                    ref = actor.forward_step.remote(stage_name, StageOutputs(activations=data), labels, 0)
                    refs.append(ref)
            else:
                pred_refs = stage_output_refs[preds[0]]
                refs = []
                for actor, pred_ref in zip(group.actors, pred_refs):
                    ref = actor.forward_step.remote(stage_name, pred_ref, labels, 0)
                    refs.append(ref)
            stage_output_refs[stage_name] = refs

        # Backward: run all stages in reverse order on each actor (in-process, no serialization)
        reversed_stages = list(reversed(self.topo_order))
        for stage_name in reversed_stages:
            if self._should_inject_failure(stage_name, "backward", iteration, microbatch_id=0):
                raise PipelineIterationError(stage_name, "backward", 0)
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
        run_error: Exception | None = None
        result: dict[str, Any] | None = None

        try:
            for step in schedule:
                group = self.plan.stage_to_actor_group[step.stage_name]
                mb_id = step.microbatch_id
                step_op = step.op.value
                if self._should_inject_failure(step.stage_name, step_op, iteration, microbatch_id=mb_id):
                    raise PipelineIterationError(step.stage_name, step_op, mb_id)

                if step.op == OpType.FORWARD:
                    preds = self.dag.predecessors(step.stage_name)
                    succs = self.dag.successors(step.stage_name)
                    # Only use RDT/NCCL if any successor edge requires T2 transport
                    use_rdt = any(
                        self.router.get_transport(step.stage_name, s) in ("t2", "mixed")
                        for s in succs
                    )

                    if not preds:
                        refs = [
                            self._forward_remote(
                                actor,
                                step.stage_name,
                                StageOutputs(activations=mb_data[mb_id]),
                                mb_labels[mb_id],
                                mb_id,
                                use_rdt,
                            )
                            for actor in group.actors
                        ]
                    else:
                        pred_name = preds[0]
                        pred_refs = stage_output_refs[pred_name][mb_id]
                        pred_group = self.plan.stage_to_actor_group[pred_name]
                        routing = self.router.get_routing(pred_name, step.stage_name)

                        refs = []
                        for dst_rank, actor in enumerate(group.actors):
                            src_rank = routing.dst_to_src[dst_rank]
                            transport = self.router.get_actor_pair_transport(
                                pred_name, step.stage_name, src_rank, dst_rank
                            )

                            if transport == "t1":
                                # T1: CUDA IPC — get IPC handle for (stage, microbatch), reconstruct on receiver
                                ipc_ref = pred_group.actors[src_rank].create_ipc_for_output.remote(pred_name, mb_id)
                                ref = actor.forward_from_ipc.remote(
                                    step.stage_name, ipc_ref, mb_labels[mb_id], mb_id
                                )
                            else:
                                pred_ref = pred_refs[src_rank]
                                ref = self._forward_remote(
                                    actor,
                                    step.stage_name,
                                    pred_ref,
                                    mb_labels[mb_id],
                                    mb_id,
                                    use_rdt=(transport == "t2"),
                                )
                            refs.append(ref)

                    stage_output_refs.setdefault(step.stage_name, {})[mb_id] = refs

                elif step.op == OpType.BACKWARD:
                    succs_of = self.dag.successors(step.stage_name)
                    preds_of = self.dag.predecessors(step.stage_name)
                    use_rdt_upstream = any(
                        self.router.get_transport(p, step.stage_name) in ("t2", "mixed")
                        for p in preds_of
                    )

                    if not succs_of:
                        refs = [
                            self._backward_remote(actor, step.stage_name, None, mb_id, use_rdt_upstream)
                            for actor in group.actors
                        ]
                    else:
                        succ_name = succs_of[0]
                        succ_grad_refs = stage_grad_refs.get(succ_name, {}).get(mb_id, [])
                        succ_group = self.plan.stage_to_actor_group[succ_name]
                        routing = self.router.get_routing(step.stage_name, succ_name)

                        refs = []
                        for src_rank, actor in enumerate(group.actors):
                            dst_ranks = routing.src_to_dst[src_rank]
                            grad_ref = self._gather_grad_refs_with_transport(
                                actor,
                                step.stage_name,
                                succ_name,
                                succ_group,
                                succ_grad_refs,
                                dst_ranks,
                                src_rank,
                                mb_id,
                            )
                            ref = self._backward_remote(actor, step.stage_name, grad_ref, mb_id, use_rdt_upstream)
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

            result = {"loss": loss_value, "global_grad_norm": global_grad_norm}
        except Exception as exc:  # noqa: PERF203
            run_error = exc
        finally:
            cleanup_error = None
            if self._has_t1:
                try:
                    self._clear_t1_ipc_cache(iteration)
                except Exception as exc:  # noqa: PERF203
                    cleanup_error = exc

            if cleanup_error is not None:
                if run_error is None:
                    raise self._wrap_iteration_exception(cleanup_error) from cleanup_error
                logger.error(f"Failed to clear T1 IPC cache after iteration error: {cleanup_error}")

        if run_error is not None:
            if isinstance(run_error, PipelineIterationError):
                raise run_error
            raise self._wrap_iteration_exception(run_error) from run_error

        assert result is not None
        return result

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
    def _forward_remote(actor, stage_name, inputs, labels, microbatch_id: int, use_rdt: bool):
        """Dispatch forward_step, optionally using RDT/NCCL for the return value."""
        if use_rdt:
            return actor.forward_step.options(tensor_transport="nccl").remote(stage_name, inputs, labels, microbatch_id)
        return actor.forward_step.remote(stage_name, inputs, labels, microbatch_id)

    @staticmethod
    def _backward_remote(actor, stage_name, downstream_grad, microbatch_id: int, use_rdt: bool):
        """Dispatch backward_step, optionally using RDT/NCCL for the return value."""
        if use_rdt:
            return actor.backward_step.options(tensor_transport="nccl").remote(stage_name, downstream_grad, microbatch_id)
        return actor.backward_step.remote(stage_name, downstream_grad, microbatch_id)

    @staticmethod
    def _gather_grad_refs(actor, succ_grad_refs, dst_ranks, use_rdt: bool):
        """Gather and aggregate gradient refs for one src actor.

        For 1:1 mapping, returns the single grad ref directly.
        For 1:N expansion, aggregates N grad refs via sequential accumulation.
        """
        if not succ_grad_refs or not dst_ranks:
            return None
        if len(dst_ranks) == 1:
            return succ_grad_refs[dst_ranks[0]]
        # Multiple successor actors → accumulate sequentially
        result_ref = succ_grad_refs[dst_ranks[0]]
        for d in dst_ranks[1:]:
            result_ref = actor.accumulate_gradient.remote(result_ref, succ_grad_refs[d])
        return result_ref

    def _gather_grad_refs_with_transport(
        self,
        actor,
        src_stage: str,
        succ_stage: str,
        succ_group,
        succ_grad_refs,
        dst_ranks,
        src_rank: int,
        mb_id: int,
    ):
        """Gather gradient refs using per-actor-pair transport (T0/T1/T2).

        For T1 pairs, fetches IPC handles from successor actors.
        For T2 pairs, uses RDT.
        For T0 pairs, passes ObjectRefs directly.
        """
        if not succ_grad_refs or not dst_ranks:
            return None

        if len(dst_ranks) == 1:
            dst_rank = dst_ranks[0]
            transport = self.router.get_actor_pair_transport(src_stage, succ_stage, src_rank, dst_rank)
            if transport == "t1":
                return succ_group.actors[dst_rank].create_ipc_for_grad.remote(succ_stage, mb_id)
            return succ_grad_refs[dst_rank]

        # Multiple successor actors → gather and aggregate sequentially.
        # We use sequential accumulate_gradient (two top-level args) instead of
        # sum_gradients (list arg) because Ray does not resolve ObjectRefs inside
        # lists on tensor-transport-enabled actors.
        gathered_refs = []
        for dst_rank in dst_ranks:
            transport = self.router.get_actor_pair_transport(src_stage, succ_stage, src_rank, dst_rank)
            if transport == "t1":
                gathered_refs.append(succ_group.actors[dst_rank].create_ipc_for_grad.remote(succ_stage, mb_id))
            else:
                gathered_refs.append(succ_grad_refs[dst_rank])

        # Chain accumulate_gradient calls: each takes two resolved ObjectRefs
        result_ref = gathered_refs[0]
        for ref in gathered_refs[1:]:
            result_ref = actor.accumulate_gradient.remote(result_ref, ref)
        return result_ref

    def _iter_unique_actor_groups(self):
        seen: set[int] = set()
        for stage_name in self.topo_order:
            group = self.plan.stage_to_actor_group[stage_name]
            group_id = id(group)
            if group_id in seen:
                continue
            seen.add(group_id)
            yield group

    def _clear_t1_ipc_cache(self, iteration: int) -> None:
        cleanup_refs = []
        actor_contexts: list[tuple[str, int, Any]] = []
        for group in self._iter_unique_actor_groups():
            for actor_rank, actor in enumerate(group.actors):
                cleanup_refs.append(actor.clear_t1_ipc_cache.remote())
                actor_contexts.append((group.resource_set_name, actor_rank, actor))

        if not cleanup_refs:
            self._last_t1_cache_cleanup_stats = []
            return

        cleanup_stats = ray.get(cleanup_refs)
        self._last_t1_cache_cleanup_stats = []
        for idx, stats in enumerate(cleanup_stats):
            resource_set_name, actor_rank, _ = actor_contexts[idx]
            entry = {"resource_set": resource_set_name, "actor_rank": actor_rank, "iteration": iteration}
            entry.update(stats or {})
            self._last_t1_cache_cleanup_stats.append(entry)

        if not self._verify_t1_cleanup:
            return

        verify_refs = [actor.get_t1_ipc_cache_stats.remote() for _, _, actor in actor_contexts]
        verify_stats = ray.get(verify_refs)
        for idx, stats in enumerate(verify_stats):
            if (stats or {}).get("forward_current_entries", 0) != 0 or (stats or {}).get("backward_current_entries", 0) != 0:
                resource_set_name, actor_rank, _ = actor_contexts[idx]
                raise RuntimeError(
                    f"T1 cache cleanup verification failed at {resource_set_name}[{actor_rank}] "
                    f"(iteration={iteration}): {stats}"
                )

    def _wrap_iteration_exception(self, exc: Exception) -> PipelineIterationError:
        if isinstance(exc, PipelineIterationError):
            return exc

        text = str(exc)
        stage_name = "unknown"
        op = "iteration"
        microbatch_id = -1

        full_match = re.search(r"stage='([^']+)', microbatch=(\d+)", text)
        if full_match:
            stage_name = full_match.group(1)
            microbatch_id = int(full_match.group(2))
        else:
            stage_match = re.search(r"stage='([^']+)'", text)
            if stage_match:
                stage_name = stage_match.group(1)

        if "forward output" in text or "create_ipc_for_output" in text or "forward_from_ipc" in text:
            op = "forward"
        elif "backward grad" in text or "create_ipc_for_grad" in text or "backward_from_ipc" in text:
            op = "backward"
        elif "cleanup" in text:
            op = "cleanup"

        return PipelineIterationError(stage_name, op, microbatch_id, cause=exc)

    def get_last_t1_cache_cleanup_stats(self) -> list[dict[str, Any]]:
        """Return per-actor T1 cache stats captured during the last cleanup."""
        return [dict(entry) for entry in self._last_t1_cache_cleanup_stats]

    def _should_inject_failure(
        self,
        stage_name: str,
        op: str,
        iteration: int,
        microbatch_id: int | None = None,
    ) -> bool:
        if self.failure_injection is None:
            return False
        if (
            self.failure_injection.get("stage_name") != stage_name
            or self.failure_injection.get("op") != op
            or self.failure_injection.get("iteration") != iteration
        ):
            return False

        injected_mb = self.failure_injection.get("microbatch_id")
        if injected_mb is not None and microbatch_id is not None and injected_mb != microbatch_id:
            return False
        return True

    def shutdown(self) -> None:
        """Shut down the runner. Idempotent."""
        pass
