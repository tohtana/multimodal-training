"""Pipeline schedulers: determine the execution order of forward/backward across stages and microbatches."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum


class OpType(str, Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass(frozen=True)
class ScheduleStep:
    """A single step in a pipeline schedule."""

    op: OpType
    stage_name: str
    microbatch_id: int


class PipelineScheduler(ABC):
    """Base class for pipeline schedulers."""

    @abstractmethod
    def generate_schedule(self, stage_order: list[str], num_microbatches: int) -> list[ScheduleStep]:
        """Generate a list of schedule steps for one iteration.

        Args:
            stage_order: Topological order of stages.
            num_microbatches: Number of microbatches to schedule.

        Returns:
            List of ScheduleStep in execution order.
        """
        ...


class OneFOneBScheduler(PipelineScheduler):
    """1F1B (one-forward-one-backward) pipeline schedule.

    Reduces pipeline bubble time by interleaving forward and backward passes
    across microbatches:
      - Warmup: fill pipeline with forward passes (deeper stages start later)
      - Steady state: alternate backward/forward at each stage
      - Cooldown: drain remaining backward passes

    Produces identical gradients to SequentialScheduler (same total gradient
    accumulation, just different operation ordering).
    """

    def generate_schedule(self, stage_order: list[str], num_microbatches: int) -> list[ScheduleStep]:
        N = len(stage_order)
        M = num_microbatches
        if M < 1:
            return []

        # Build per-stage operation queues.
        # Stage s has warmup W = min(N-1-s, M) forward passes before its first backward.
        queues: list[list[tuple[str, int]]] = []
        for s in range(N):
            W = min(N - 1 - s, M)
            ops: list[tuple[str, int]] = []
            fwd_mb, bwd_mb = 0, 0

            if W == 0:
                # Last stage(s): alternate F,B immediately
                while fwd_mb < M:
                    ops.append(("F", fwd_mb))
                    fwd_mb += 1
                    if bwd_mb < M:
                        ops.append(("B", bwd_mb))
                        bwd_mb += 1
            else:
                # Warmup forwards
                for _ in range(W):
                    ops.append(("F", fwd_mb))
                    fwd_mb += 1
                # Steady state: B,F pairs
                while fwd_mb < M:
                    ops.append(("B", bwd_mb))
                    bwd_mb += 1
                    ops.append(("F", fwd_mb))
                    fwd_mb += 1
                # Cooldown: remaining backwards
                while bwd_mb < M:
                    ops.append(("B", bwd_mb))
                    bwd_mb += 1

            queues.append(ops)

        # Interleave per-stage queues respecting data dependencies:
        #   F(s, mb) requires F(s-1, mb)
        #   B(s, mb) requires F(s, mb) and B(s+1, mb) [or s is last stage]
        pointers = [0] * N
        fwd_done: set[tuple[int, int]] = set()
        bwd_done: set[tuple[int, int]] = set()
        steps: list[ScheduleStep] = []
        total_ops = sum(len(q) for q in queues)

        while len(steps) < total_ops:
            emitted = False

            # Pass 1: backward operations (last stage first)
            for s in range(N - 1, -1, -1):
                if pointers[s] >= len(queues[s]):
                    continue
                op, mb = queues[s][pointers[s]]
                if op != "B":
                    continue
                can_run = (s, mb) in fwd_done
                if s < N - 1:
                    can_run = can_run and ((s + 1, mb) in bwd_done)
                if can_run:
                    steps.append(ScheduleStep(OpType.BACKWARD, stage_order[s], mb))
                    bwd_done.add((s, mb))
                    pointers[s] += 1
                    emitted = True

            # Pass 2: forward operations (first stage first)
            for s in range(N):
                if pointers[s] >= len(queues[s]):
                    continue
                op, mb = queues[s][pointers[s]]
                if op != "F":
                    continue
                can_run = (s == 0) or ((s - 1, mb) in fwd_done)
                if can_run:
                    steps.append(ScheduleStep(OpType.FORWARD, stage_order[s], mb))
                    fwd_done.add((s, mb))
                    pointers[s] += 1
                    emitted = True

            if not emitted:
                raise RuntimeError(
                    f"1F1B schedule deadlock at step {len(steps)}/{total_ops}. "
                    f"Pointers: {pointers}, Queue lengths: {[len(q) for q in queues]}"
                )

        return steps


class SequentialScheduler(PipelineScheduler):
    """Sequential (fill-drain) schedule: all forwards then all backwards for each microbatch.

    For num_microbatches=1 this is the simplest pipeline schedule:
      forward(stage_0), forward(stage_1), ..., forward(stage_N),
      backward(stage_N), backward(stage_N-1), ..., backward(stage_0)
    """

    def generate_schedule(self, stage_order: list[str], num_microbatches: int) -> list[ScheduleStep]:
        steps: list[ScheduleStep] = []
        for mb in range(num_microbatches):
            # Forward in topological order
            for stage in stage_order:
                steps.append(ScheduleStep(op=OpType.FORWARD, stage_name=stage, microbatch_id=mb))
            # Backward in reverse topological order
            for stage in reversed(stage_order):
                steps.append(ScheduleStep(op=OpType.BACKWARD, stage_name=stage, microbatch_id=mb))
        return steps
