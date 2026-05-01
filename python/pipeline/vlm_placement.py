"""Placement-aware legacy VLM actor construction.

This module adapts the flexible pipeline placement model to the existing
Vision/Text Ray trainers. It intentionally reuses ``PlacementPlan`` so routing
and metadata consumers see the same contract as generic pipeline execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import ray

from ..ray.actor_group import ActorGroup
from ..ray.tensor_transfer import gather_gpu_ids
from .dag import PipelineDAG
from .placement import PlacementPlan
from .router import CrossStageRouter
from .stage import Pipeline, ResourceSet, Stage

logger = logging.getLogger(__name__)


TrainerResolver = Callable[[str, dict], tuple[type, dict]]
ActorGroupFactory = Callable[..., Any]
GpuIdCollector = Callable[[list], list[str]]


@dataclass
class VLMStageGroup:
    stage_name: str
    resource_set: str
    actor_group: Any
    actor_count: int
    physical_gpu_ids: dict[int, str]
    requested_device_ids: list[int] | None
    placement_match: bool

    @property
    def actors(self) -> list:
        return self.actor_group._actors

    @property
    def _actors(self) -> list:
        return self.actor_group._actors

    @property
    def num_actors(self) -> int:
        return self.actor_count

    @property
    def resource_set_name(self) -> str:
        return self.resource_set

    @property
    def placement_group_handle(self):
        return getattr(self.actor_group, "placement_group", None)

    def execute_all_async(self, method_name: str, *args, **kwargs):
        return self.actor_group.execute_all_async(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.actor_group.execute_all(method_name, *args, **kwargs)

    def shutdown(self) -> None:
        self.actor_group.shutdown()


def build_vlm_stage_groups(
    pipeline: Pipeline,
    stage_configs: dict[str, dict],
    trainer_resolver: TrainerResolver,
    *,
    actor_group_factory: ActorGroupFactory = ActorGroup,
    gpu_id_collector: GpuIdCollector = gather_gpu_ids,
) -> tuple[PlacementPlan, dict[str, VLMStageGroup]]:
    """Build real VLM actor groups from pipeline resource sets.

    Actor counts come from the stage's ``ResourceSet.num_gpus``. The returned
    ``PlacementPlan`` stores the exact ``VLMStageGroup`` objects returned to the
    caller, preserving one source of placement truth for routing and metadata.
    """
    _validate_stage_configs(pipeline, stage_configs)

    stage_by_rs = _stage_names_by_resource_set(pipeline)
    sharing_count = _sharing_count_by_resource_set(pipeline, stage_by_rs)
    built_resource_sets: set[str] = set()
    stage_groups: dict[str, VLMStageGroup] = {}
    placement_plan = PlacementPlan()

    def build_resource_set(rs_name: str) -> None:
        if rs_name in built_resource_sets:
            return

        rs = pipeline.get_resource_set(rs_name)
        if rs.subset_of is not None:
            build_resource_set(rs.subset_of)

        parent_pg = None
        if rs.subset_of is not None:
            parent_stage_name = next(iter(stage_by_rs.get(rs.subset_of, [])), None)
            if parent_stage_name is not None:
                parent_pg = stage_groups[parent_stage_name].placement_group_handle

        for stage in [pipeline.get_stage(name) for name in stage_by_rs.get(rs_name, [])]:
            group = _create_single_stage_group(
                pipeline=pipeline,
                stage=stage,
                resource_set=rs,
                stage_config=stage_configs[stage.name],
                trainer_resolver=trainer_resolver,
                actor_group_factory=actor_group_factory,
                gpu_id_collector=gpu_id_collector,
                sharing_count=sharing_count[rs_name],
                placement_group_handle=parent_pg,
            )
            stage_groups[stage.name] = group
            placement_plan.stage_to_actor_group[stage.name] = group
            placement_plan.resource_set_to_actor_group[rs.name] = group
            placement_plan.actor_gpu_ids[rs.name] = group.physical_gpu_ids

            if parent_pg is None and group.placement_group_handle is not None:
                parent_pg = group.placement_group_handle

        built_resource_sets.add(rs_name)

    try:
        for resource_set in pipeline.resource_sets:
            build_resource_set(resource_set.name)
    except Exception:
        for group in stage_groups.values():
            group.shutdown()
        raise

    return placement_plan, stage_groups


def configure_vlm_cross_stage_communication(
    pipeline: Pipeline,
    placement_plan: PlacementPlan,
    stage_configs: dict[str, dict],
) -> CrossStageRouter:
    """Create routing metadata and initialize cross-stage transport hooks."""
    router = CrossStageRouter(PipelineDAG(pipeline), placement_plan)

    for edge in pipeline.edges:
        src_component = stage_configs[edge.src].get("component_type", edge.src)
        dst_component = stage_configs[edge.dst].get("component_type", edge.dst)
        if src_component == "bridge" or dst_component == "bridge":
            continue

        src_group = placement_plan.stage_to_actor_group[edge.src]
        dst_group = placement_plan.stage_to_actor_group[edge.dst]
        routing = router.get_routing(edge.src, edge.dst)

        if router.get_transport(edge.src, edge.dst) in {"t1", "mixed"}:
            dst_rs = pipeline.get_placement(edge.dst).resource_set
            src_rs = pipeline.get_placement(edge.src).resource_set
            dst_gpus = placement_plan.actor_gpu_ids.get(dst_rs, {})
            src_gpus = placement_plan.actor_gpu_ids.get(src_rs, {})
            for src_rank, dst_ranks in routing.src_to_dst.items():
                receiver_ids = [dst_gpus[dst_rank] for dst_rank in dst_ranks if dst_rank in dst_gpus]
                src_group._actors[src_rank].set_receiver_info.remote(receiver_ids, use_ipc=True)
            for dst_rank, src_rank in routing.dst_to_src.items():
                receiver_ids = [src_gpus[src_rank]] if src_rank in src_gpus else []
                dst_group._actors[dst_rank].set_receiver_info.remote(receiver_ids, use_ipc=True)
            ray.get([actor.get_rank.remote() for actor in src_group._actors])

    router.setup_collective_groups()
    return router


def _validate_stage_configs(pipeline: Pipeline, stage_configs: dict[str, dict]) -> None:
    missing = [stage.name for stage in pipeline.stages if stage.name not in stage_configs]
    if missing:
        raise ValueError(f"Missing stage config for: {', '.join(missing)}")


def _stage_names_by_resource_set(pipeline: Pipeline) -> dict[str, list[str]]:
    by_rs: dict[str, list[str]] = {rs.name: [] for rs in pipeline.resource_sets}
    for stage in pipeline.stages:
        by_rs[pipeline.get_placement(stage.name).resource_set].append(stage.name)
    return by_rs


def _sharing_count_by_resource_set(pipeline: Pipeline, stage_by_rs: dict[str, list[str]]) -> dict[str, int]:
    child_count: dict[str, int] = {rs.name: 0 for rs in pipeline.resource_sets}
    for rs in pipeline.resource_sets:
        if rs.subset_of is not None:
            child_count[rs.subset_of] += max(1, len(stage_by_rs.get(rs.name, [])))

    sharing: dict[str, int] = {}
    for rs in pipeline.resource_sets:
        local_stages = max(1, len(stage_by_rs.get(rs.name, [])))
        if rs.subset_of is None:
            sharing[rs.name] = max(local_stages + child_count[rs.name], local_stages)
        else:
            parent = pipeline.get_resource_set(rs.subset_of)
            sharing[rs.name] = max(1, len(stage_by_rs.get(parent.name, [])) + child_count[parent.name])
    return sharing


def _create_single_stage_group(
    *,
    pipeline: Pipeline,
    stage: Stage,
    resource_set: ResourceSet,
    stage_config: dict,
    trainer_resolver: TrainerResolver,
    actor_group_factory: ActorGroupFactory,
    gpu_id_collector: GpuIdCollector,
    sharing_count: int,
    placement_group_handle,
) -> VLMStageGroup:
    trainer_cls, init_kwargs = trainer_resolver(stage.name, stage_config)
    collocate = sharing_count > 1
    actor_group = actor_group_factory(
        stage_config,
        trainer_cls,
        num_actors=resource_set.num_gpus,
        collocate=collocate,
        placement_group_handle=placement_group_handle if collocate else None,
        actor_init_kwargs=init_kwargs,
        collocation_factor=sharing_count,
    )
    physical_gpu_ids = {rank: gpu_id for rank, gpu_id in enumerate(gpu_id_collector(actor_group._actors))}
    requested = list(resource_set.device_ids) if resource_set.device_ids is not None else None
    placement_match = _placement_matches(requested, physical_gpu_ids)

    logger.info(
        "Created VLM stage group '%s': resource_set=%s actors=%s requested_device_ids=%s "
        "physical_gpu_ids=%s placement_match=%s",
        stage.name,
        resource_set.name,
        resource_set.num_gpus,
        requested,
        physical_gpu_ids,
        placement_match,
    )
    if resource_set.num_gpus != len(actor_group._actors):
        raise RuntimeError(
            f"Stage '{stage.name}' actor count mismatch: ResourceSet.num_gpus={resource_set.num_gpus}, "
            f"actors={len(actor_group._actors)}"
        )
    if pipeline.get_placement(stage.name).resource_set != resource_set.name:
        raise RuntimeError(f"Stage '{stage.name}' placement changed during VLM group construction")

    return VLMStageGroup(
        stage_name=stage.name,
        resource_set=resource_set.name,
        actor_group=actor_group,
        actor_count=resource_set.num_gpus,
        physical_gpu_ids=physical_gpu_ids,
        requested_device_ids=requested,
        placement_match=placement_match,
    )


def _placement_matches(requested_device_ids: list[int] | None, physical_gpu_ids: dict[int, str]) -> bool:
    if requested_device_ids is None:
        return True
    return {str(device_id) for device_id in requested_device_ids} == set(physical_gpu_ids.values())
