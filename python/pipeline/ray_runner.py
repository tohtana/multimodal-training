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
        self._has_t1 = self.router.has_t1_edges()

        # Shared buffer state (T19)
        self._shared_buffer_requested = False
        self._shared_buffers_ready = False
        self._shared_buffer_error: str | None = None
        self._shared_num_microbatches: int = 0
        self._overlap_pairs: dict[tuple, dict] = {}

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

        # Shared buffer overlap path (T19)
        if self._shared_buffer_requested:
            if self._shared_buffers_ready:
                return self._run_iteration_overlap(data, labels, max_norm, iteration, num_microbatches)
            logger.warning(f"shared_buffer disabled; fallback to mixed path: {self._shared_buffer_error}")

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
                # Only use RDT/NCCL if any successor edge requires T2 transport
                use_rdt = any(self.router.get_transport(step.stage_name, s) in ("t2", "mixed") for s in succs)

                if not preds:
                    refs = [
                        self._forward_remote(
                            actor,
                            step.stage_name,
                            StageOutputs(activations=mb_data[mb_id]),
                            mb_labels[mb_id],
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
                        transport = self.router.get_actor_pair_transport(pred_name, step.stage_name, src_rank, dst_rank)

                        if transport == "t1":
                            # T1: CUDA IPC — get IPC handle from sender, reconstruct on receiver
                            ipc_ref = pred_group.actors[src_rank].create_ipc_for_output.remote(pred_name)
                            ref = actor.forward_from_ipc.remote(step.stage_name, ipc_ref, mb_labels[mb_id])
                        else:
                            pred_ref = pred_refs[src_rank]
                            ref = self._forward_remote(
                                actor,
                                step.stage_name,
                                pred_ref,
                                mb_labels[mb_id],
                                use_rdt=(transport == "t2"),
                            )
                        refs.append(ref)

                stage_output_refs.setdefault(step.stage_name, {})[mb_id] = refs

            elif step.op == OpType.BACKWARD:
                succs_of = self.dag.successors(step.stage_name)
                preds_of = self.dag.predecessors(step.stage_name)
                use_rdt_upstream = any(
                    self.router.get_transport(p, step.stage_name) in ("t2", "mixed") for p in preds_of
                )

                if not succs_of:
                    refs = [
                        self._backward_remote(actor, step.stage_name, None, use_rdt_upstream) for actor in group.actors
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

    # ── Shared buffer overlap (T19) ──

    def enable_shared_buffer_overlap(
        self,
        *,
        num_microbatches: int,
        activation_specs: dict[tuple[str, str], dict],
        scheduler_name: str,
    ) -> tuple[bool, str | None]:
        """Preflight + setup for shared-buffer overlap. Returns (ok, reason)."""
        self._shared_buffer_requested = True
        self._shared_num_microbatches = num_microbatches
        self._shared_buffers_ready = False
        self._shared_buffer_error = None

        # v1 guardrails
        if scheduler_name != "gpipe":
            self._shared_buffer_error = "shared_buffer v1 requires --scheduler gpipe"
            return False, self._shared_buffer_error
        if len(self.topo_order) != 2:
            self._shared_buffer_error = "shared_buffer v1 requires exactly 2 stages"
            return False, self._shared_buffer_error

        # Every overlap edge must be pure T1 and bijective 1:1
        for (src, dst), tier in self.router.edge_transports.items():
            if tier != "t1":
                self._shared_buffer_error = f"edge {src}->{dst} transport={tier}; requires pure t1"
                return False, self._shared_buffer_error
            routing = self.router.get_routing(src, dst)
            if routing.num_src != routing.num_dst or any(len(v) != 1 for v in routing.src_to_dst.values()):
                self._shared_buffer_error = f"edge {src}->{dst} is not 1:1"
                return False, self._shared_buffer_error
            if (src, dst) not in activation_specs:
                self._shared_buffer_error = f"missing activation spec for edge {src}->{dst}"
                return False, self._shared_buffer_error
            spec = activation_specs[(src, dst)]
            if spec.get("payload_mode") != "activations_only":
                self._shared_buffer_error = (
                    f"edge {src}->{dst} payload_mode={spec.get('payload_mode')}; "
                    "shared_buffer v1 requires activations_only"
                )
                return False, self._shared_buffer_error
            if spec.get("required_meta_keys"):
                self._shared_buffer_error = (
                    f"edge {src}->{dst} requires meta keys {spec['required_meta_keys']}; "
                    "shared_buffer v1 requires empty meta"
                )
                return False, self._shared_buffer_error

        try:
            self._setup_shared_buffers(num_microbatches, activation_specs)
            self._shared_buffers_ready = True
            return True, None
        except Exception as e:
            cleanup_err = self._cleanup_shared_buffers()
            self._shared_buffer_error = f"setup failed: {e}"
            if cleanup_err:
                self._shared_buffer_error += f"; cleanup warning: {cleanup_err}"
            return False, self._shared_buffer_error

    def _setup_shared_buffers(self, num_microbatches: int, activation_specs: dict[tuple[str, str], dict]) -> None:
        """Create/open forward+backward shared buffers for every eligible actor pair."""
        self._overlap_pairs = {}

        for (src, dst), _tier in self.router.edge_transports.items():
            spec = activation_specs[(src, dst)]
            shape, dtype = spec["shape"], spec["dtype"]
            src_group = self.plan.stage_to_actor_group[src]
            dst_group = self.plan.stage_to_actor_group[dst]
            routing = self.router.get_routing(src, dst)

            for src_rank, dst_ranks in routing.src_to_dst.items():
                dst_rank = dst_ranks[0]
                fwd_buffer_id = f"{src}->{dst}:fwd:r{src_rank}->r{dst_rank}"
                bwd_buffer_id = f"{src}->{dst}:bwd:r{dst_rank}->r{src_rank}"

                # Forward buffer: producer=src, consumer=dst
                ipc_data = ray.get(
                    src_group.actors[src_rank].setup_shared_buffers.remote(
                        fwd_buffer_id, num_microbatches, shape, dtype
                    )
                )
                ray.get(dst_group.actors[dst_rank].open_shared_buffers.remote(ipc_data))

                # Backward buffer: producer=dst, consumer=src
                ipc_data = ray.get(
                    dst_group.actors[dst_rank].setup_shared_buffers.remote(
                        bwd_buffer_id, num_microbatches, shape, dtype
                    )
                )
                ray.get(src_group.actors[src_rank].open_shared_buffers.remote(ipc_data))

                self._overlap_pairs[(src, dst, src_rank, dst_rank)] = {
                    "fwd_buffer_id": fwd_buffer_id,
                    "bwd_buffer_id": bwd_buffer_id,
                }

    def _cleanup_shared_buffers(self) -> str | None:
        """Best-effort rollback used after setup failure. Never raises."""
        cleanup_refs = []
        for group in self.plan.resource_set_to_actor_group.values():
            for actor in group.actors:
                cleanup_refs.append(actor.clear_shared_buffers.remote())
        if not cleanup_refs:
            return None
        try:
            ray.get(cleanup_refs, timeout=15)
            return None
        except Exception as cleanup_e:
            logger.warning(f"shared_buffer cleanup had errors: {cleanup_e}")
            return str(cleanup_e)

    def _run_iteration_overlap(
        self,
        data: Any,
        labels: Any,
        max_norm: float | None,
        iteration: int,
        num_microbatches: int,
    ) -> dict[str, Any]:
        """Pipeline iteration with concurrent dispatch for true stage overlap.

        For a 2-stage pipeline with M microbatches:

        Forward phase:
          All forward ops dispatched at once. Per-actor FIFO ordering ensures
          mb0 runs before mb1 on each actor. CUDA events enforce cross-actor
          data dependencies. MPS enables GPU kernel overlap.

        Backward phase (symmetric):
          All backward ops dispatched at once. FIFO ordering preserved.

        Dependencies enforced by CUDA events (GPU-side), not by Ray task ordering.
        ray.get at phase boundaries prevents cross-iteration slot/event reuse races.
        """
        if num_microbatches != self._shared_num_microbatches:
            raise ValueError(
                f"shared_buffer configured for {self._shared_num_microbatches} microbatches, got {num_microbatches}"
            )

        mb_data, mb_labels = self._split_batch(data, labels, num_microbatches)

        # Forward phase: enqueue producer+consumer per microbatch for each actor pair.
        pending_refs = []
        for mb in range(num_microbatches):
            for (src, dst, src_rank, dst_rank), ids in self._overlap_pairs.items():
                src_actor = self.plan.stage_to_actor_group[src].actors[src_rank]
                dst_actor = self.plan.stage_to_actor_group[dst].actors[dst_rank]
                pending_refs.append(
                    src_actor.forward_to_buffer.remote(
                        src, ids["fwd_buffer_id"], mb, StageOutputs(activations=mb_data[mb]), mb_labels[mb]
                    )
                )
                pending_refs.append(dst_actor.forward_from_buffer.remote(dst, ids["fwd_buffer_id"], mb, mb_labels[mb]))
        ray.get(pending_refs)  # iteration fence for forward slot/event reuse

        # Backward phase: preserve FIFO microbatch order per actor.
        pending_refs = []
        for mb in range(num_microbatches):
            for (src, dst, src_rank, dst_rank), ids in self._overlap_pairs.items():
                src_actor = self.plan.stage_to_actor_group[src].actors[src_rank]
                dst_actor = self.plan.stage_to_actor_group[dst].actors[dst_rank]
                pending_refs.append(dst_actor.backward_to_buffer.remote(dst, ids["bwd_buffer_id"], mb, None))
                pending_refs.append(src_actor.backward_from_buffer.remote(src, ids["bwd_buffer_id"], mb))
        ray.get(pending_refs)  # iteration fence for backward slot/event reuse

        # Loss + grad norm + optimizer (same as _run_iteration_mixed)
        loss_value = None
        for stage_cfg in self.pipeline.stages:
            if stage_cfg.is_terminal:
                group = self.plan.stage_to_actor_group[stage_cfg.name]
                loss_value = ray.get(group.actors[0].get_last_loss.remote(stage_cfg.name))

        total_norm_sq = 0.0
        for name in self.topo_order:
            group = self.plan.stage_to_actor_group[name]
            norm_sq = ray.get(group.actors[0].compute_grad_norm_sq.remote(name))
            total_norm_sq += norm_sq
        global_grad_norm = math.sqrt(total_norm_sq)

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
    def _split_batch(data: Any, labels: Any, num_microbatches: int) -> tuple[list[Any], list[Any]]:
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
                return succ_group.actors[dst_rank].create_ipc_for_grad.remote(succ_stage)
            return succ_grad_refs[dst_rank]

        # Multiple successor actors → gather and aggregate sequentially.
        # We use sequential accumulate_gradient (two top-level args) instead of
        # sum_gradients (list arg) because Ray does not resolve ObjectRefs inside
        # lists on tensor-transport-enabled actors.
        gathered_refs = []
        for dst_rank in dst_ranks:
            transport = self.router.get_actor_pair_transport(src_stage, succ_stage, src_rank, dst_rank)
            if transport == "t1":
                gathered_refs.append(succ_group.actors[dst_rank].create_ipc_for_grad.remote(succ_stage))
            else:
                gathered_refs.append(succ_grad_refs[dst_rank])

        # Chain accumulate_gradient calls: each takes two resolved ObjectRefs
        result_ref = gathered_refs[0]
        for ref in gathered_refs[1:]:
            result_ref = actor.accumulate_gradient.remote(result_ref, ref)
        return result_ref

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
