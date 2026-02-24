"""PlacementManager: creates Ray placement groups and ActorGroups for pipeline stages (design §6.3)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import ray
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from ..ray.multi_stage_actor import MultiStageActor
from .stage import Pipeline, ResourceSet, Stage

logger = logging.getLogger(__name__)


@dataclass
class PlacementPlan:
    """Result of placement planning: maps stages to ActorGroups."""

    stage_to_actor_group: dict[str, "PipelineActorGroup"] = field(default_factory=dict)
    resource_set_to_actor_group: dict[str, "PipelineActorGroup"] = field(default_factory=dict)
    # Physical GPU ID per actor: actor_gpu_ids[resource_set_name][actor_rank] = gpu_uuid
    actor_gpu_ids: dict[str, dict[int, str]] = field(default_factory=dict)


@dataclass
class PipelineActorGroup:
    """An ActorGroup of MultiStageActors for a resource set."""

    resource_set_name: str
    actors: list  # list of Ray actor handles
    num_actors: int
    placement_group_handle: Any = None
    stage_names: list[str] = field(default_factory=list)

    def execute_all_async(self, method_name: str, *args, **kwargs) -> list:
        """Execute method on all actors asynchronously. Returns list of ObjectRefs."""
        results = []
        for actor in self.actors:
            remote_call = getattr(actor, method_name)
            results.append(remote_call.remote(*args, **kwargs))
        return results

    def execute_all(self, method_name: str, *args, **kwargs) -> list:
        """Execute method on all actors synchronously."""
        refs = self.execute_all_async(method_name, *args, **kwargs)
        return ray.get(refs)

    def shutdown(self) -> None:
        """Shut down actors and placement group. Idempotent."""
        for actor in self.actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.actors = []

        if self.placement_group_handle is not None:
            try:
                ray.util.remove_placement_group(self.placement_group_handle)
            except Exception:
                pass
            self.placement_group_handle = None


@dataclass
class StageModelSpec:
    """Serializable model specification for constructing stages on Ray actors."""

    stage_name: str
    model_cls: type
    model_kwargs: dict
    state_dict: dict | None = None
    is_terminal: bool = False
    optimizer_cls: type | None = None
    optimizer_kwargs: dict | None = None
    loss_cls: type | None = None
    loss_kwargs: dict | None = None
    engine: str = "native"  # "native" or "deepspeed"
    ds_config: dict | None = None  # DeepSpeed config dict (required when engine="deepspeed")


class PlacementManager:
    """Creates placement groups and ActorGroups for pipeline resource sets."""

    def __init__(
        self,
        pipeline: Pipeline,
        model_specs: list[StageModelSpec] | None = None,
        actor_cls: type = MultiStageActor,
        num_cpus_per_actor: int = 1,
    ):
        self.pipeline = pipeline
        self.model_specs = {spec.stage_name: spec for spec in (model_specs or [])}
        self.actor_cls = actor_cls
        self.num_cpus_per_actor = num_cpus_per_actor
        self._groups: list[PipelineActorGroup] = []

    def plan(self) -> PlacementPlan:
        """Create placement groups and ActorGroups for all resource sets.

        Handles ``subset_of`` overlapping resource sets by placing parent and
        child actors on a shared placement group with fractional GPUs so that
        actors on the same physical GPU share memory (enabling T1/CUDA IPC).

        Returns:
            PlacementPlan mapping stages to their ActorGroups.
        """
        result = PlacementPlan()

        # Group stages by resource set
        rs_to_stages: dict[str, list[Stage]] = {}
        for stage in self.pipeline.stages:
            placement = self.pipeline.get_placement(stage.name)
            rs_name = placement.resource_set
            rs_to_stages.setdefault(rs_name, []).append(stage)

        # Identify overlap groups: parent → [children]
        rs_by_name = {rs.name: rs for rs in self.pipeline.resource_sets}
        children_of: dict[str, list[ResourceSet]] = {}  # parent_name → [child_rs]
        standalone: list[str] = []
        child_names: set[str] = set()

        for rs in self.pipeline.resource_sets:
            if rs.subset_of is not None:
                children_of.setdefault(rs.subset_of, []).append(rs)
                child_names.add(rs.name)

        for rs in self.pipeline.resource_sets:
            if rs.name not in child_names:
                if rs.name in children_of:
                    pass  # Will be handled as overlap parent
                else:
                    standalone.append(rs.name)

        # Create standalone ActorGroups (no overlap)
        for rs_name in standalone:
            rs = rs_by_name[rs_name]
            stages = rs_to_stages.get(rs_name, [])
            group = self._create_actor_group(rs, stages)
            self._groups.append(group)
            result.resource_set_to_actor_group[rs_name] = group
            for stage in stages:
                result.stage_to_actor_group[stage.name] = group

        # Create overlapping ActorGroups (parent + children share PG)
        for parent_name, child_rss in children_of.items():
            parent_rs = rs_by_name[parent_name]
            parent_stages = rs_to_stages.get(parent_name, [])
            child_stages_map = {c.name: rs_to_stages.get(c.name, []) for c in child_rss}

            parent_group, child_groups = self._create_overlap_groups(
                parent_rs, child_rss, parent_stages, child_stages_map
            )
            self._groups.append(parent_group)
            result.resource_set_to_actor_group[parent_name] = parent_group
            for stage in parent_stages:
                result.stage_to_actor_group[stage.name] = parent_group

            for c_name, c_group in child_groups.items():
                self._groups.append(c_group)
                result.resource_set_to_actor_group[c_name] = c_group
                for stage in rs_to_stages.get(c_name, []):
                    result.stage_to_actor_group[stage.name] = c_group

        # Query physical GPU IDs from all actors
        self._query_gpu_ids(result)

        return result

    def build_models(self, plan: PlacementPlan) -> None:
        """Build models on all actors using the model specs.

        This sends model_cls, model_kwargs, and state_dicts to actors —
        all serializable (no lambdas or closures).
        """
        for stage_name, spec in self.model_specs.items():
            group = plan.stage_to_actor_group[stage_name]
            refs = []
            for actor in group.actors:
                ref = actor.build_model_from_state_dict.remote(
                    stage_name=spec.stage_name,
                    model_cls=spec.model_cls,
                    model_kwargs=spec.model_kwargs,
                    state_dict=spec.state_dict,
                    is_terminal=spec.is_terminal,
                    optimizer_cls=spec.optimizer_cls,
                    optimizer_kwargs=spec.optimizer_kwargs,
                    loss_cls=spec.loss_cls,
                    loss_kwargs=spec.loss_kwargs,
                    engine=spec.engine,
                    ds_config=spec.ds_config,
                )
                refs.append(ref)
            ray.get(refs)

    def _create_overlap_groups(
        self,
        parent_rs: ResourceSet,
        child_rss: list[ResourceSet],
        parent_stages: list[Stage],
        child_stages_map: dict[str, list[Stage]],
    ) -> tuple[PipelineActorGroup, dict[str, PipelineActorGroup]]:
        """Create ActorGroups for a parent and its child resource sets (subset_of).

        Creates a single placement group with 1 GPU per bundle (parent GPU count).
        Parent actors get 0.5 GPU each. Child actors also get 0.5 GPU each and are
        placed on the same bundle as their corresponding parent actor, so they
        share the same physical GPU (enabling T1 / CUDA IPC transport).
        """
        num_parent_gpus = parent_rs.num_gpus
        # Count max actors per GPU (for CPU allocation in shared bundles)
        child_device_sets = [set(c.device_ids) for c in child_rss if c.device_ids]
        all_child_devices = set().union(*child_device_sets) if child_device_sets else set()
        # Each bundle needs enough CPUs for parent actor + any collocated child actors
        bundles = []
        for i, dev_id in enumerate(parent_rs.device_ids):
            num_actors_on_gpu = 1 + sum(1 for cs in child_device_sets if dev_id in cs)
            bundles.append({
                "GPU": 1,
                "CPU": self.num_cpus_per_actor * num_actors_on_gpu,
            })
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

        # Create parent actors (0.5 GPU each)
        parent_remote_cls = ray.remote(
            num_cpus=self.num_cpus_per_actor, num_gpus=0.5, enable_tensor_transport=True
        )(self.actor_cls)

        parent_actors = []
        for i in range(num_parent_gpus):
            actor = parent_remote_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote({}, i)
            parent_actors.append(actor)

        parent_group = PipelineActorGroup(
            resource_set_name=parent_rs.name,
            actors=parent_actors,
            num_actors=num_parent_gpus,
            placement_group_handle=pg,
            stage_names=[s.name for s in parent_stages],
        )

        # Map parent device_ids to bundle indices
        parent_device_to_bundle = {d: idx for idx, d in enumerate(parent_rs.device_ids)}

        # Create child actor groups
        child_groups: dict[str, PipelineActorGroup] = {}
        for child_rs in child_rss:
            child_remote_cls = ray.remote(
                num_cpus=self.num_cpus_per_actor, num_gpus=0.5, enable_tensor_transport=True
            )(self.actor_cls)

            child_actors = []
            for j, dev_id in enumerate(child_rs.device_ids):
                bundle_idx = parent_device_to_bundle[dev_id]
                actor = child_remote_cls.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=bundle_idx,
                    ),
                ).remote({}, j)
                child_actors.append(actor)

            child_stages = child_stages_map.get(child_rs.name, [])
            child_group = PipelineActorGroup(
                resource_set_name=child_rs.name,
                actors=child_actors,
                num_actors=child_rs.num_gpus,
                placement_group_handle=None,  # Shared PG managed by parent
                stage_names=[s.name for s in child_stages],
            )
            child_groups[child_rs.name] = child_group

            logger.info(
                f"Created overlapping ActorGroup for child '{child_rs.name}' "
                f"({child_rs.num_gpus} actors, subset_of '{parent_rs.name}')"
            )

        logger.info(
            f"Created overlap group: parent='{parent_rs.name}' ({num_parent_gpus} GPUs), "
            f"children={[c.name for c in child_rss]}"
        )

        return parent_group, child_groups

    def _query_gpu_ids(self, plan: PlacementPlan) -> None:
        """Query physical GPU IDs from all actors and store in PlacementPlan."""
        for rs_name, group in plan.resource_set_to_actor_group.items():
            refs = [actor.get_physical_gpu_id.remote() for actor in group.actors]
            gpu_ids = ray.get(refs)
            plan.actor_gpu_ids[rs_name] = {i: gpu_id for i, gpu_id in enumerate(gpu_ids)}
            logger.info(f"GPU IDs for resource set '{rs_name}': {plan.actor_gpu_ids[rs_name]}")

    def _create_actor_group(self, rs: ResourceSet, stages: list[Stage]) -> PipelineActorGroup:
        """Create a PipelineActorGroup for a resource set."""
        num_actors = rs.num_gpus
        stage_names = [s.name for s in stages]

        # Create placement group
        bundles = [{"GPU": 1, "CPU": self.num_cpus_per_actor} for _ in range(num_actors)]
        pg = placement_group(bundles, strategy="PACK")
        ray.get(pg.ready())

        # Create remote actor class (enable_tensor_transport for RDT/NCCL cross-GPU transfers)
        remote_cls = ray.remote(
            num_cpus=self.num_cpus_per_actor, num_gpus=1, enable_tensor_transport=True
        )(self.actor_cls)

        # Create actors
        actors = []
        for i in range(num_actors):
            actor = remote_cls.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote({}, i)
            actors.append(actor)

        group = PipelineActorGroup(
            resource_set_name=rs.name,
            actors=actors,
            num_actors=num_actors,
            placement_group_handle=pg,
            stage_names=stage_names,
        )

        logger.info(f"Created ActorGroup for resource set '{rs.name}': {num_actors} actors, stages={stage_names}")
        return group

    def shutdown(self) -> None:
        """Shut down all managed ActorGroups. Idempotent."""
        for group in self._groups:
            group.shutdown()
        self._groups = []
