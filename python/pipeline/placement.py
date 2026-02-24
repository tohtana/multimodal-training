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

        # Create one ActorGroup per resource set
        for rs_name, stages in rs_to_stages.items():
            rs = self.pipeline.get_resource_set(rs_name)
            group = self._create_actor_group(rs, stages)
            self._groups.append(group)
            result.resource_set_to_actor_group[rs_name] = group
            for stage in stages:
                result.stage_to_actor_group[stage.name] = group

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
