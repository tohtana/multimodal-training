"""CrossStageRouter: determines transport tier and routes tensors between pipeline stages (design §6.4)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .dag import PipelineDAG
from .placement import PlacementPlan

logger = logging.getLogger(__name__)


@dataclass
class RoutingPlan:
    """Mapping from src actors to dst actors for an edge.

    Supports M:N routing where M src actors send to N dst actors.
    """

    src_to_dst: dict[int, list[int]]  # src_rank → [dst_ranks]
    dst_to_src: dict[int, int]  # dst_rank → src_rank
    policy: str  # "broadcast" or "scatter"
    num_src: int = 0
    num_dst: int = 0

    @staticmethod
    def build(num_src: int, num_dst: int, policy: str = "broadcast") -> RoutingPlan:
        """Build a routing plan for M:N actor mapping.

        Assigns dst actors evenly across src actors in ascending rank order.
        For non-divisible cases, lower-rank src actors get one extra dst.

        Args:
            num_src: Number of source actors (M).
            num_dst: Number of destination actors (N).
            policy: "broadcast" (full copy) or "scatter" (batch split).
        """
        src_to_dst: dict[int, list[int]] = {}
        dst_to_src: dict[int, int] = {}

        if num_src == num_dst:
            # 1:1 mapping
            for i in range(num_src):
                src_to_dst[i] = [i]
                dst_to_src[i] = i
        elif num_dst > num_src:
            # M → N expansion (M < N)
            base = num_dst // num_src
            remainder = num_dst % num_src
            dst_idx = 0
            for src_idx in range(num_src):
                count = base + (1 if src_idx < remainder else 0)
                dst_list = list(range(dst_idx, dst_idx + count))
                src_to_dst[src_idx] = dst_list
                for d in dst_list:
                    dst_to_src[d] = src_idx
                dst_idx += count
        else:
            # N → M contraction (N < M) — each src sends to its mapped dst
            base = num_src // num_dst
            remainder = num_src % num_dst
            src_idx = 0
            for dst_idx in range(num_dst):
                count = base + (1 if dst_idx < remainder else 0)
                for s in range(src_idx, src_idx + count):
                    src_to_dst[s] = [dst_idx]
                    dst_to_src[dst_idx] = s  # Last src in the group
                src_idx += count

        return RoutingPlan(
            src_to_dst=src_to_dst,
            dst_to_src=dst_to_src,
            policy=policy,
            num_src=num_src,
            num_dst=num_dst,
        )


class CrossStageRouter:
    """Routes tensors between pipeline stages using the appropriate transport.

    Transport tiers:
    - T0: Same ActorGroup (same process) — pass ObjectRef directly (zero-copy)
    - T2: Different ActorGroups, different GPUs — use RDT/NCCL
    - T1: Different ActorGroups, same GPU — CUDA IPC (deferred to Milestone 11)

    Supports asymmetric M:N actor counts with broadcast or scatter routing.
    """

    def __init__(self, dag: PipelineDAG, plan: PlacementPlan):
        self.dag = dag
        self.plan = plan
        self.edge_transports: dict[tuple[str, str], str] = {}
        self.edge_routing: dict[tuple[str, str], RoutingPlan] = {}

        for edge in dag.pipeline.edges:
            src_group = plan.stage_to_actor_group[edge.src]
            dst_group = plan.stage_to_actor_group[edge.dst]
            if id(src_group) == id(dst_group):
                self.edge_transports[(edge.src, edge.dst)] = "t0"
            else:
                self.edge_transports[(edge.src, edge.dst)] = "t2"

            # Build routing plan for this edge
            routing = RoutingPlan.build(
                num_src=src_group.num_actors,
                num_dst=dst_group.num_actors,
                policy="broadcast",
            )
            self.edge_routing[(edge.src, edge.dst)] = routing

        logger.info(f"CrossStageRouter transport tiers: {self.edge_transports}")

    def get_transport(self, src: str, dst: str) -> str:
        return self.edge_transports.get((src, dst), "t2")

    def get_routing(self, src: str, dst: str) -> RoutingPlan:
        return self.edge_routing[(src, dst)]

    def is_same_group(self, src: str, dst: str) -> bool:
        return self.get_transport(src, dst) == "t0"

    def is_symmetric(self, src: str, dst: str) -> bool:
        """Check if src and dst have the same number of actors (1:1 mapping)."""
        routing = self.edge_routing.get((src, dst))
        return routing is not None and routing.num_src == routing.num_dst

    def has_cross_group_edges(self) -> bool:
        return any(tier == "t2" for tier in self.edge_transports.values())

    def setup_collective_groups(self) -> None:
        """Set up NCCL collective groups for T2 edges."""
        if not self.has_cross_group_edges():
            return

        from ray.experimental.collective import create_collective_group

        done_pairs: set[tuple[int, int]] = set()
        for (src, dst), tier in self.edge_transports.items():
            if tier != "t2":
                continue
            src_group = self.plan.stage_to_actor_group[src]
            dst_group = self.plan.stage_to_actor_group[dst]
            pair_key = tuple(sorted([id(src_group), id(dst_group)]))
            if pair_key in done_pairs:
                continue
            done_pairs.add(pair_key)

            all_actors = src_group.actors + dst_group.actors
            create_collective_group(all_actors, backend="nccl")
            logger.info(
                f"Created NCCL collective group for "
                f"{src_group.resource_set_name} <-> {dst_group.resource_set_name}"
            )
