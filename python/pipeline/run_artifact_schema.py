"""Structured run-artifact schema for dense Qwen3-VL pipeline smoke rows."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


Status = Literal["ok", "blocked", "failed", "oom", "skipped"]
Cell = Literal["compat", "collocated", "separated", "asymmetric"]
BlockerCategory = Literal[
    "dense_model_unavailable",
    "dense_weights_unavailable",
    "optimizer_construction_missing",
    "placement_pinning_unsupported",
    "asymmetric_routing_unsupported",
    "tensor_shape_mismatch",
    "oom",
    "other",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BlockerRow(StrictModel):
    category: BlockerCategory
    reason: str
    code_path: str | list[str]
    code_path_git_sha: str
    error_excerpt: str = Field(max_length=4096)
    config_diff: str | None
    next_action: str

    @model_validator(mode="after")
    def _require_non_empty_fields(self):
        required = {
            "reason": self.reason,
            "code_path_git_sha": self.code_path_git_sha,
            "error_excerpt": self.error_excerpt,
            "next_action": self.next_action,
        }
        for name, value in required.items():
            if not str(value).strip():
                raise ValueError(f"BlockerRow.{name} must be non-empty")
        paths = self.code_path if isinstance(self.code_path, list) else [self.code_path]
        if not paths or any(not path.strip() for path in paths):
            raise ValueError("BlockerRow.code_path must be non-empty")
        return self


class StageRow(StrictModel):
    name: str
    resource_set: str
    actor_count: int
    physical_gpu_ids: dict[int, str]
    cuda_visible_devices: dict[int, str] = Field(default_factory=dict)
    requested_device_ids: list[int] | None
    placement_match: bool
    process_group_world_size: int
    megatron: dict[str, Any]
    actor_runtime: dict[int, dict[str, Any]] = Field(default_factory=dict)


class EdgeRow(StrictModel):
    from_stage: str
    to_stage: str
    transfer_tier: str
    routing_map: dict[str, Any]


class TrainingRow(StrictModel):
    iterations: int
    warmup_iterations: int
    batch_size: int
    loss_values: list[float] = Field(default_factory=list)
    backward_completed: bool = False
    selected_parameter_grad_nonzero: bool = False
    parameter_norm_delta: float | None = None
    optimizer_update_verified: bool
    iteration_step_counter_advanced: bool
    expected_optimizer_step_delta: int
    actual_optimizer_step_delta: int | None
    optimizer_probe: dict[str, Any] = Field(default_factory=dict)


class MetricsRow(StrictModel):
    loss_finite: bool
    iter_time_ms_p50: float | None
    iter_time_ms_p90: float | None
    cuda_max_allocated_bytes: int | None
    cuda_max_reserved_bytes: int | None


class LayerTruncationRow(StrictModel):
    vision_source_field: str | None
    language_source_field: str | None
    vision_override_path: str | None
    language_override_path: str | None
    vision_effective_layers: int | None
    language_effective_layers: int | None
    vision_original_layers: int | None
    language_original_layers: int | None


class RunRow(StrictModel):
    run_id: str
    status: Status
    cell: Cell
    dry_run: bool
    framework_only: bool
    model_id: str | None
    dense: bool
    git_sha: str
    submodule_shas: dict[str, str]
    command: list[str]
    started_utc: str
    finished_utc: str | None
    error_excerpt: str | None = Field(default=None, max_length=4096)
    layer_truncation: LayerTruncationRow
    stages: list[StageRow]
    edges: list[EdgeRow]
    training: TrainingRow
    metrics: MetricsRow
    blocker: BlockerRow | None

    @model_validator(mode="after")
    def _status_consistency(self):
        if self.status == "blocked" and self.blocker is None:
            raise ValueError("blocked RunRow requires blocker")
        if self.status != "blocked" and self.blocker is not None:
            raise ValueError("Only blocked RunRow may carry blocker")
        if self.status == "ok":
            self._validate_ok_runtime_evidence()
        return self

    def _validate_ok_runtime_evidence(self) -> None:
        if not self.training.loss_values:
            raise ValueError("ok RunRow requires runtime loss_values")
        if not self.training.backward_completed:
            raise ValueError("ok RunRow requires backward_completed=true")
        if not self.training.selected_parameter_grad_nonzero:
            raise ValueError("ok RunRow requires selected_parameter_grad_nonzero=true")
        if self.training.parameter_norm_delta is None or self.training.parameter_norm_delta <= 0:
            raise ValueError("ok RunRow requires positive parameter_norm_delta")
        if not self.training.optimizer_update_verified or not self.training.iteration_step_counter_advanced:
            raise ValueError("ok RunRow requires optimizer update and step-counter proof")
        if self.training.actual_optimizer_step_delta != self.training.expected_optimizer_step_delta:
            raise ValueError("ok RunRow optimizer step delta mismatch")
        if not self.metrics.loss_finite:
            raise ValueError("ok RunRow requires finite loss")
        for field_name in ("iter_time_ms_p50", "iter_time_ms_p90", "cuda_max_allocated_bytes", "cuda_max_reserved_bytes"):
            if getattr(self.metrics, field_name) is None:
                raise ValueError(f"ok RunRow requires metrics.{field_name}")
        for stage in self.stages:
            if len(stage.physical_gpu_ids) != stage.actor_count:
                raise ValueError(f"ok RunRow stage {stage.name} missing physical GPU IDs")
            if len(stage.cuda_visible_devices) != stage.actor_count:
                raise ValueError(f"ok RunRow stage {stage.name} missing CUDA_VISIBLE_DEVICES")
            if len(stage.actor_runtime) != stage.actor_count:
                raise ValueError(f"ok RunRow stage {stage.name} missing actor runtime metadata")
            if any(str(value).startswith("unrun:") for value in stage.physical_gpu_ids.values()):
                raise ValueError(f"ok RunRow stage {stage.name} has unrun physical GPU placeholders")
            for rank in range(stage.actor_count):
                runtime = stage.actor_runtime.get(rank)
                if runtime is None:
                    raise ValueError(f"ok RunRow stage {stage.name} missing runtime rank {rank}")
                if not str(runtime.get("cuda_visible_devices", "")).strip():
                    raise ValueError(f"ok RunRow stage {stage.name} rank {rank} missing CUDA_VISIBLE_DEVICES")
                if not str(runtime.get("physical_gpu_id", "")).strip():
                    raise ValueError(f"ok RunRow stage {stage.name} rank {rank} missing physical GPU ID")
                process_group = runtime.get("process_group") or {}
                if not process_group.get("initialized"):
                    raise ValueError(f"ok RunRow stage {stage.name} rank {rank} process group not initialized")
                if int(process_group.get("world_size", 0)) != stage.process_group_world_size:
                    raise ValueError(f"ok RunRow stage {stage.name} rank {rank} world size mismatch")
        for edge in self.edges:
            if edge.transfer_tier == "unrun":
                raise ValueError("ok RunRow edge transfer_tier cannot be unrun")

    @field_validator("run_id", "git_sha", "started_utc")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value


SUMMARY_PROJECTION: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("run_id", ("run_id",)),
    ("status", ("status",)),
    ("cell", ("cell",)),
    ("dry_run", ("dry_run",)),
    ("framework_only", ("framework_only",)),
    ("model_id", ("model_id",)),
    ("dense", ("dense",)),
    ("git_sha", ("git_sha",)),
    ("optimizer_update_verified", ("training", "optimizer_update_verified")),
    ("loss_finite", ("metrics", "loss_finite")),
    ("iter_time_ms_p50", ("metrics", "iter_time_ms_p50")),
    ("cuda_max_allocated_bytes", ("metrics", "cuda_max_allocated_bytes")),
    ("blocker_category", ("blocker", "category")),
    ("blocker_reason", ("blocker", "reason")),
)


def project_summary(row: RunRow) -> dict[str, Any]:
    projected: dict[str, Any] = {}
    for column, path in SUMMARY_PROJECTION:
        value: Any = row
        for part in path:
            if value is None:
                break
            value = getattr(value, part)
        projected[column] = value
    return projected
