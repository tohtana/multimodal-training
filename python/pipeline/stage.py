"""Core dataclasses for flexible pipeline parallelism (design §5.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class ParallelismType(str, Enum):
    NONE = "none"
    SEQUENCE = "sequence"
    TENSOR = "tensor"
    AUTOTP = "autotp"
    EXPERT = "expert"


class EngineType(str, Enum):
    NATIVE = "native"
    DEEPSPEED = "deepspeed"
    MEGATRON = "megatron"


class MergePolicy(str, Enum):
    """Merge policy for fan-in stages that receive inputs from multiple upstream stages."""

    CONCAT = "concat"
    SUM = "sum"
    DICT = "dict"


@dataclass(frozen=True)
class ResourceSet:
    """A named set of GPU devices for stage placement."""

    name: str
    num_gpus: int
    device_ids: tuple[int, ...] | None = None
    subset_of: str | None = None

    def __post_init__(self):
        if self.num_gpus <= 0:
            raise ValueError(f"ResourceSet '{self.name}': num_gpus must be > 0, got {self.num_gpus}")
        if self.device_ids is not None and len(self.device_ids) != self.num_gpus:
            raise ValueError(
                f"ResourceSet '{self.name}': device_ids length ({len(self.device_ids)}) "
                f"must match num_gpus ({self.num_gpus})"
            )


@dataclass(frozen=True)
class Placement:
    """Maps a stage to a resource set."""

    stage_name: str
    resource_set: str


@dataclass(frozen=True)
class EdgeConfig:
    """Configuration for a directed edge (data flow) between two stages."""

    src: str
    dst: str
    transfer_fn: Callable | None = None
    layout_fn: Callable | None = None
    merge_policy: MergePolicy | None = None


@dataclass
class Stage:
    """A single pipeline stage: model + parallelism + engine."""

    name: str
    model_fn: Callable | None = None
    trainer_cls: type | None = None
    trainer_resolver_key: str | None = None
    parallelism: ParallelismType = ParallelismType.NONE
    engine: EngineType = EngineType.NATIVE
    is_source: bool = False
    is_terminal: bool = False
    dataloader_fn: Callable | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.is_source and self.dataloader_fn is None and self.model_fn is None:
            pass  # source stages may have dataloader provided later via config
        if not self.is_source and self.dataloader_fn is not None:
            raise ValueError(f"Stage '{self.name}': non-source stages must not have a dataloader_fn")


@dataclass
class Pipeline:
    """Complete pipeline configuration: stages, edges, resource sets, placements."""

    stages: list[Stage]
    edges: list[EdgeConfig]
    resource_sets: list[ResourceSet]
    placements: list[Placement]
    dp_size: int = 1
    num_microbatches: int = 1
    gradient_accumulation_steps: int = 1

    def validate(self) -> list[str]:
        """Validate the pipeline configuration. Returns list of error messages (empty = valid)."""
        errors = []

        stage_names = {s.name for s in self.stages}

        # Check for duplicate stage names
        if len(stage_names) != len(self.stages):
            seen = set()
            for s in self.stages:
                if s.name in seen:
                    errors.append(f"Duplicate stage name: '{s.name}'")
                seen.add(s.name)

        # Check resource set names
        rs_names = {rs.name for rs in self.resource_sets}
        if len(rs_names) != len(self.resource_sets):
            errors.append("Duplicate resource set names")

        # Check placements reference valid stages and resource sets
        placed_stages = set()
        for p in self.placements:
            if p.stage_name not in stage_names:
                errors.append(f"Placement references unknown stage: '{p.stage_name}'")
            if p.resource_set not in rs_names:
                errors.append(f"Placement references unknown resource set: '{p.resource_set}'")
            if p.stage_name in placed_stages:
                errors.append(f"Stage '{p.stage_name}' has multiple placements")
            placed_stages.add(p.stage_name)

        # Every stage must have exactly one placement
        for s in self.stages:
            if s.name not in placed_stages:
                errors.append(f"Stage '{s.name}' has no placement")

        # Check edges reference valid stages
        for e in self.edges:
            if e.src not in stage_names:
                errors.append(f"Edge references unknown source stage: '{e.src}'")
            if e.dst not in stage_names:
                errors.append(f"Edge references unknown destination stage: '{e.dst}'")

        # Check fan-in: stages with multiple inputs must have merge policy
        dst_counts: dict[str, int] = {}
        edge_merge_policies: dict[str, MergePolicy | None] = {}
        for e in self.edges:
            dst_counts[e.dst] = dst_counts.get(e.dst, 0) + 1
            if e.merge_policy is not None:
                edge_merge_policies[e.dst] = e.merge_policy

        for dst, count in dst_counts.items():
            if count > 1 and dst not in edge_merge_policies:
                errors.append(f"Fan-in stage '{dst}' has {count} inputs but no merge policy specified")

        # dp_size == 1 enforced (phase 1 constraint)
        if self.dp_size != 1:
            errors.append(f"dp_size must be 1 (phase 1 constraint), got {self.dp_size}")

        # gradient_accumulation_steps constraint with microbatches
        if self.num_microbatches > 1 and self.gradient_accumulation_steps > 1:
            errors.append("gradient_accumulation_steps must be <= 1 when num_microbatches > 1")

        # Source stage data ownership: only source stages may have dataloaders
        for s in self.stages:
            if not s.is_source and s.dataloader_fn is not None:
                errors.append(f"Non-source stage '{s.name}' must not have a dataloader_fn")

        # Check subset_of references and device_ids constraints
        rs_by_name = {rs.name: rs for rs in self.resource_sets}
        for rs in self.resource_sets:
            if rs.subset_of is not None:
                if rs.subset_of not in rs_names:
                    errors.append(f"ResourceSet '{rs.name}' references unknown parent: '{rs.subset_of}'")
                else:
                    parent = rs_by_name[rs.subset_of]
                    # Both must have device_ids
                    if rs.device_ids is None:
                        errors.append(f"ResourceSet '{rs.name}' uses subset_of but has no device_ids")
                    if parent.device_ids is None:
                        errors.append(f"ResourceSet '{rs.name}' parent '{rs.subset_of}' has no device_ids")
                    # Child device_ids must be a subset of parent device_ids
                    if rs.device_ids is not None and parent.device_ids is not None:
                        if not set(rs.device_ids).issubset(set(parent.device_ids)):
                            errors.append(
                                f"ResourceSet '{rs.name}' device_ids {rs.device_ids} "
                                f"not a subset of parent '{rs.subset_of}' device_ids {parent.device_ids}"
                            )

        return errors

    def get_stage(self, name: str) -> Stage:
        """Get a stage by name."""
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(f"No stage named '{name}'")

    def get_placement(self, stage_name: str) -> Placement:
        """Get the placement for a stage."""
        for p in self.placements:
            if p.stage_name == stage_name:
                return p
        raise KeyError(f"No placement for stage '{stage_name}'")

    def get_resource_set(self, name: str) -> ResourceSet:
        """Get a resource set by name."""
        for rs in self.resource_sets:
            if rs.name == name:
                return rs
        raise KeyError(f"No resource set named '{name}'")

    def stage_resource_set(self, stage_name: str) -> ResourceSet:
        """Get the resource set for a stage."""
        placement = self.get_placement(stage_name)
        return self.get_resource_set(placement.resource_set)
