"""Canonical YAML → Pipeline parser for pipeline configs (design §8.1, §8.2).

This is the only YAML parsing module for pipeline configurations. It converts
pipeline-format YAML into typed Pipeline/Stage/EdgeConfig objects.
"""

from __future__ import annotations

from typing import Any

import yaml

from .stage import (
    EdgeConfig,
    EngineType,
    MergePolicy,
    ParallelismType,
    Pipeline,
    Placement,
    ResourceSet,
    Stage,
)

# Registry for trainer resolver keys → (trainer_cls, model_fn) tuples
# Populated by trainer_registry or by user code before loading configs.
_TRAINER_REGISTRY: dict[str, dict[str, Any]] = {}


def register_trainer(key: str, trainer_cls: type | None = None, model_fn: Any = None) -> None:
    """Register a trainer resolver key for YAML config resolution."""
    _TRAINER_REGISTRY[key] = {"trainer_cls": trainer_cls, "model_fn": model_fn}


def get_registered_trainer(key: str) -> dict[str, Any]:
    """Look up a registered trainer by resolver key."""
    if key not in _TRAINER_REGISTRY:
        raise ConfigLoadError(f"Unknown trainer resolver key: '{key}'. Registered keys: {list(_TRAINER_REGISTRY)}")
    return _TRAINER_REGISTRY[key]


class ConfigLoadError(Exception):
    """Raised when a pipeline YAML config is invalid or cannot be parsed."""


def load_pipeline_config(yaml_path: str) -> Pipeline:
    """Load a pipeline YAML config and return a validated Pipeline object.

    Args:
        yaml_path: Path to the YAML config file.

    Returns:
        A Pipeline object with all stages, edges, resource sets, and placements.

    Raises:
        ConfigLoadError: If the YAML is invalid or cannot be parsed.
    """
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigLoadError(f"Expected top-level dict in {yaml_path}, got {type(raw).__name__}")

    return parse_pipeline_dict(raw)


def parse_pipeline_dict(raw: dict[str, Any]) -> Pipeline:
    """Parse a raw dict (from YAML or Python) into a Pipeline object.

    Args:
        raw: Dict with keys: stages, edges, resource_sets, placements, and optional pipeline-level settings.

    Returns:
        A validated Pipeline object.

    Raises:
        ConfigLoadError: If required fields are missing or values are invalid.
    """
    try:
        stages = _parse_stages(raw.get("stages", []))
        edges = _parse_edges(raw.get("edges", []))
        resource_sets = _parse_resource_sets(raw.get("resource_sets", []))
        placements = _parse_placements(raw.get("placements", []))

        pipeline = Pipeline(
            stages=stages,
            edges=edges,
            resource_sets=resource_sets,
            placements=placements,
            dp_size=raw.get("dp_size", 1),
            num_microbatches=raw.get("num_microbatches", 1),
            gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 1),
        )

        errors = pipeline.validate()
        if errors:
            raise ConfigLoadError("Pipeline validation failed:\n  " + "\n  ".join(errors))

        return pipeline

    except ConfigLoadError:
        raise
    except Exception as e:
        raise ConfigLoadError(f"Failed to parse pipeline config: {e}") from e


def _parse_stages(raw_stages: list[dict]) -> list[Stage]:
    """Parse stage definitions from YAML."""
    if not raw_stages:
        raise ConfigLoadError("Pipeline must have at least one stage")

    stages = []
    for i, s in enumerate(raw_stages):
        if not isinstance(s, dict):
            raise ConfigLoadError(f"Stage {i}: expected dict, got {type(s).__name__}")
        name = s.get("name")
        if not name:
            raise ConfigLoadError(f"Stage {i}: 'name' is required")

        # Resolve parallelism
        parallelism_str = s.get("parallelism", "none")
        try:
            parallelism = ParallelismType(parallelism_str)
        except ValueError:
            raise ConfigLoadError(
                f"Stage '{name}': invalid parallelism '{parallelism_str}'. "
                f"Valid: {[p.value for p in ParallelismType]}"
            )

        # Resolve engine
        engine_str = s.get("engine", "native")
        try:
            engine = EngineType(engine_str)
        except ValueError:
            raise ConfigLoadError(
                f"Stage '{name}': invalid engine '{engine_str}'. " f"Valid: {[e.value for e in EngineType]}"
            )

        # Resolve trainer from registry if resolver key is provided
        trainer_cls = None
        model_fn = None
        resolver_key = s.get("trainer_resolver_key")
        if resolver_key:
            try:
                reg = get_registered_trainer(resolver_key)
                trainer_cls = reg.get("trainer_cls")
                model_fn = reg.get("model_fn")
            except ConfigLoadError:
                raise ConfigLoadError(
                    f"Stage '{name}': unknown trainer_resolver_key '{resolver_key}'. "
                    f"Registered keys: {list(_TRAINER_REGISTRY)}"
                )

        stage = Stage(
            name=name,
            model_fn=model_fn,
            trainer_cls=trainer_cls,
            trainer_resolver_key=resolver_key,
            parallelism=parallelism,
            engine=engine,
            is_source=s.get("is_source", False),
            is_terminal=s.get("is_terminal", False),
            dataloader_fn=None,  # YAML stages don't carry callables; set via code
            config=s.get("config", {}),
        )
        stages.append(stage)

    return stages


def _parse_edges(raw_edges: list[dict]) -> list[EdgeConfig]:
    """Parse edge definitions from YAML."""
    edges = []
    for i, e in enumerate(raw_edges):
        if not isinstance(e, dict):
            raise ConfigLoadError(f"Edge {i}: expected dict, got {type(e).__name__}")

        src = e.get("src")
        dst = e.get("dst")
        if not src or not dst:
            raise ConfigLoadError(f"Edge {i}: 'src' and 'dst' are required")

        merge_policy = None
        if "merge_policy" in e:
            try:
                merge_policy = MergePolicy(e["merge_policy"])
            except ValueError:
                raise ConfigLoadError(
                    f"Edge {i} ({src}->{dst}): invalid merge_policy '{e['merge_policy']}'. "
                    f"Valid: {[m.value for m in MergePolicy]}"
                )

        edges.append(
            EdgeConfig(
                src=src,
                dst=dst,
                merge_policy=merge_policy,
            )
        )

    return edges


def _parse_resource_sets(raw_rs: list[dict]) -> list[ResourceSet]:
    """Parse resource set definitions from YAML."""
    if not raw_rs:
        raise ConfigLoadError("Pipeline must have at least one resource set")

    resource_sets = []
    for i, rs in enumerate(raw_rs):
        if not isinstance(rs, dict):
            raise ConfigLoadError(f"ResourceSet {i}: expected dict, got {type(rs).__name__}")

        name = rs.get("name")
        if not name:
            raise ConfigLoadError(f"ResourceSet {i}: 'name' is required")

        num_gpus = rs.get("num_gpus")
        if num_gpus is None:
            raise ConfigLoadError(f"ResourceSet '{name}': 'num_gpus' is required")

        device_ids = rs.get("device_ids")
        if device_ids is not None:
            device_ids = tuple(device_ids)

        resource_sets.append(
            ResourceSet(
                name=name,
                num_gpus=num_gpus,
                device_ids=device_ids,
                subset_of=rs.get("subset_of"),
            )
        )

    return resource_sets


def _parse_placements(raw_placements: list[dict]) -> list[Placement]:
    """Parse placement definitions from YAML."""
    if not raw_placements:
        raise ConfigLoadError("Pipeline must have at least one placement")

    placements = []
    for i, p in enumerate(raw_placements):
        if not isinstance(p, dict):
            raise ConfigLoadError(f"Placement {i}: expected dict, got {type(p).__name__}")

        stage_name = p.get("stage")
        resource_set = p.get("resource_set")
        if not stage_name or not resource_set:
            raise ConfigLoadError(f"Placement {i}: 'stage' and 'resource_set' are required")

        placements.append(Placement(stage_name=stage_name, resource_set=resource_set))

    return placements
