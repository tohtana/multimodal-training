"""Prototype same-process scheduler for coarse Attention/MoE overlap."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

SCHEMA_VERSION = "v1"
SUPPORTED_SCHEDULES = ("serial", "stream")
SUPPORTED_DEVICES = ("auto", "cpu", "cuda")
SUPPORTED_GROUPS = ("attention", "moe")
SUPPORTED_BLOCK_KINDS = {
    "attention_scores": "attention",
    "attention_probs": "attention",
    "attention_output": "attention",
    "router_probs": "moe",
    "topk_dispatch": "moe",
    "expert_hidden": "moe",
    "expert_output": "moe",
    "moe_output": "moe",
}
INITIAL_VALUE_SOURCES = {
    "attention_input": "input",
    "moe_tokens": "input",
    "router": "weight",
    "w1": "weight",
    "w2": "weight",
}
FINAL_OUTPUTS = ("attention_output", "moe_output")
PROFILE_LABEL_PREFIX = "attn_moe_block::"


@dataclass(frozen=True)
class ShapeConfig:
    batch: int = 1
    seq_len: int = 32
    hidden: int = 64
    num_experts: int = 4
    top_k: int = 2

    def validate(self) -> None:
        values = {
            "batch": self.batch,
            "seq_len": self.seq_len,
            "hidden": self.hidden,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
        }
        for name, value in values.items():
            if int(value) <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.top_k > self.num_experts:
            raise ValueError(f"top_k ({self.top_k}) must be <= num_experts ({self.num_experts})")

    def to_json(self) -> dict[str, int]:
        return {
            "batch": int(self.batch),
            "seq_len": int(self.seq_len),
            "hidden": int(self.hidden),
            "num_experts": int(self.num_experts),
            "top_k": int(self.top_k),
        }


@dataclass(frozen=True)
class BlockSpec:
    name: str
    kind: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    group: str | None = None
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def resolved_group(self) -> str | None:
        return self.group if self.group is not None else SUPPORTED_BLOCK_KINDS.get(self.kind)

    def to_json(self) -> dict[str, Any]:
        group = self.resolved_group
        return {
            "name": self.name,
            "group": group,
            "module": group,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True)
class ValidationError:
    code: str
    message: str
    block: str | None = None
    dependency: str | None = None

    def to_json(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "block": self.block,
            "dependency": self.dependency,
        }


@dataclass
class _WorkloadState:
    values: dict[str, torch.Tensor]
    value_sources: dict[str, str]


def default_block_specs() -> list[BlockSpec]:
    return [
        BlockSpec(
            name="A0_attention_scores",
            group="attention",
            kind="attention_scores",
            inputs=("attention_input",),
            outputs=("attention_scores",),
        ),
        BlockSpec(
            name="A1_attention_probs",
            group="attention",
            kind="attention_probs",
            depends_on=("A0_attention_scores",),
            inputs=("attention_scores",),
            outputs=("attention_probs",),
        ),
        BlockSpec(
            name="A2_attention_output",
            group="attention",
            kind="attention_output",
            depends_on=("A1_attention_probs",),
            inputs=("attention_probs", "attention_input"),
            outputs=("attention_output",),
        ),
        BlockSpec(
            name="M0_router_probs",
            group="moe",
            kind="router_probs",
            inputs=("moe_tokens", "router"),
            outputs=("router_probs",),
        ),
        BlockSpec(
            name="M1_topk_dispatch",
            group="moe",
            kind="topk_dispatch",
            depends_on=("M0_router_probs",),
            inputs=("router_probs",),
            outputs=("dispatch",),
        ),
        BlockSpec(
            name="M2_expert_hidden",
            group="moe",
            kind="expert_hidden",
            inputs=("moe_tokens", "w1"),
            outputs=("expert_hidden",),
        ),
        BlockSpec(
            name="M3_expert_output",
            group="moe",
            kind="expert_output",
            depends_on=("M2_expert_hidden",),
            inputs=("expert_hidden", "w2"),
            outputs=("expert_output",),
        ),
        BlockSpec(
            name="M4_moe_output",
            group="moe",
            kind="moe_output",
            depends_on=("M1_topk_dispatch", "M3_expert_output"),
            inputs=("dispatch", "expert_output"),
            outputs=("moe_output",),
        ),
    ]


def validate_schedule(blocks: Sequence[BlockSpec]) -> list[ValidationError]:
    errors: list[ValidationError] = []
    counts: dict[str, int] = {}
    for block in blocks:
        counts[block.name] = counts.get(block.name, 0) + 1

    for name, count in counts.items():
        if count > 1:
            errors.append(
                ValidationError(
                    code="duplicate_block",
                    message=f"Block name {name!r} appears {count} times",
                    block=name,
                )
            )

    names = set(counts)
    for block in blocks:
        expected_group = SUPPORTED_BLOCK_KINDS.get(block.kind)
        if expected_group is None:
            errors.append(
                ValidationError(
                    code="unsupported_block_kind",
                    message=f"Block {block.name!r} has unsupported kind {block.kind!r}",
                    block=block.name,
                )
            )
        group = block.resolved_group
        if group not in SUPPORTED_GROUPS:
            errors.append(
                ValidationError(
                    code="unsupported_block_group",
                    message=f"Block {block.name!r} has unsupported group {group!r}",
                    block=block.name,
                )
            )
        elif expected_group is not None and group != expected_group:
            errors.append(
                ValidationError(
                    code="unsupported_block_group",
                    message=(
                        f"Block {block.name!r} kind {block.kind!r} belongs to group "
                        f"{expected_group!r}, not {group!r}"
                    ),
                    block=block.name,
                )
            )
        for dep in block.depends_on:
            if dep == block.name:
                errors.append(
                    ValidationError(
                        code="self_dependency",
                        message=f"Block {block.name!r} depends on itself",
                        block=block.name,
                        dependency=dep,
                    )
                )
            elif dep not in names:
                errors.append(
                    ValidationError(
                        code="unknown_dependency",
                        message=f"Block {block.name!r} depends on unknown block {dep!r}",
                        block=block.name,
                        dependency=dep,
                    )
                )

    if errors:
        return errors

    waves, cycle_blocks = _topological_waves(blocks)
    if not waves and cycle_blocks:
        return [
            ValidationError(
                code="cycle",
                message=f"Dependency cycle includes block {cycle_blocks[0]!r}",
                block=cycle_blocks[0],
            )
        ]
    if cycle_blocks:
        errors.append(
            ValidationError(
                code="cycle",
                message=f"Dependency cycle includes block {cycle_blocks[0]!r}",
                block=cycle_blocks[0],
            )
        )
    return errors


def _topological_waves(blocks: Sequence[BlockSpec]) -> tuple[list[list[BlockSpec]], list[str]]:
    by_name = {block.name: block for block in blocks}
    remaining = set(by_name)
    satisfied: set[str] = set()
    waves: list[list[BlockSpec]] = []

    while remaining:
        ready_names = [
            block.name
            for block in blocks
            if block.name in remaining and all(dep in satisfied for dep in by_name[block.name].depends_on)
        ]
        if not ready_names:
            return waves, sorted(remaining)
        waves.append([by_name[name] for name in ready_names])
        satisfied.update(ready_names)
        remaining.difference_update(ready_names)
    return waves, []


def _resolve_device(requested: str) -> torch.device:
    if requested not in SUPPORTED_DEVICES:
        raise ValueError(f"Unsupported device {requested!r}; expected one of {SUPPORTED_DEVICES}")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _probe_green_context(cuda_available: bool) -> dict[str, Any]:
    try:
        from examples.attn_moe_overlap import green_context_utils
    except Exception as exc:  # pragma: no cover - import failure depends on environment
        return {
            "visible": False,
            "usable": False,
            "reason": f"import_failed: {exc}",
        }

    visible = all(
        hasattr(green_context_utils, name) for name in ("green_context_supported", "create_green_context_stream")
    )
    if not visible:
        return {
            "visible": False,
            "usable": False,
            "reason": "green_context_helpers_missing",
        }
    if not cuda_available:
        return {
            "visible": True,
            "usable": False,
            "reason": "cuda_unavailable",
        }

    try:
        usable, reason = green_context_utils.green_context_supported()
    except Exception as exc:  # pragma: no cover - driver behavior is host-specific
        return {
            "visible": True,
            "usable": False,
            "reason": f"probe_failed: {exc}",
        }
    return {
        "visible": True,
        "usable": bool(usable),
        "reason": reason,
    }


def probe_runtime_capabilities(device: str = "auto") -> dict[str, Any]:
    resolved = _resolve_device(device)
    cuda_available = bool(torch.cuda.is_available())
    cuda_used = resolved.type == "cuda"
    cuda_info: dict[str, Any] = {
        "available": cuda_available,
        "requested": device,
        "used": cuda_used,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "device_index": None,
        "device_name": None,
        "device_capability": None,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }
    if cuda_used:
        index = int(resolved.index or 0)
        cuda_info.update(
            {
                "device_index": index,
                "device_name": torch.cuda.get_device_name(index),
                "device_capability": list(torch.cuda.get_device_capability(index)),
            }
        )

    return {
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "cuda": cuda_info,
        "green_context": _probe_green_context(cuda_available),
    }


def _make_tensor(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    seed: int,
    scale: float = 1.0,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    tensor = torch.randn(shape, generator=generator, dtype=torch.float32) * float(scale)
    return tensor.to(device=device)


def _make_linspace_tensor(shape: tuple[int, ...], *, device: torch.device, scale: float) -> torch.Tensor:
    numel = math.prod(shape)
    tensor = torch.linspace(-1.0, 1.0, steps=numel, dtype=torch.float32).reshape(shape)
    return (tensor * float(scale)).to(device=device)


def _build_workload(shape: ShapeConfig, device: torch.device, seed: int) -> _WorkloadState:
    shape.validate()
    batch, seq_len, hidden = shape.batch, shape.seq_len, shape.hidden
    return _WorkloadState(
        values={
            "attention_input": _make_tensor((batch, seq_len, hidden), device=device, seed=seed + 11, scale=0.2),
            "moe_tokens": _make_tensor((batch * seq_len, hidden), device=device, seed=seed + 23, scale=0.2),
            "router": _make_linspace_tensor((hidden, shape.num_experts), device=device, scale=0.05),
            "w1": _make_linspace_tensor((shape.num_experts, hidden, hidden), device=device, scale=0.04),
            "w2": _make_linspace_tensor((shape.num_experts, hidden, hidden), device=device, scale=0.04),
        },
        value_sources=dict(INITIAL_VALUE_SOURCES),
    )


def _run_block(block: BlockSpec, state: _WorkloadState, shape: ShapeConfig) -> torch.Tensor:
    values = state.values
    if block.kind == "attention_scores":
        x = values["attention_input"]
        output = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(float(x.shape[-1]))
    elif block.kind == "attention_probs":
        output = torch.softmax(values["attention_scores"], dim=-1)
    elif block.kind == "attention_output":
        output = torch.matmul(values["attention_probs"], values["attention_input"])
    elif block.kind == "router_probs":
        output = torch.softmax(values["moe_tokens"] @ values["router"], dim=-1)
    elif block.kind == "topk_dispatch":
        router_probs = values["router_probs"]
        top_values, top_indices = torch.topk(router_probs, k=shape.top_k, dim=-1)
        dispatch = torch.zeros_like(router_probs)
        dispatch.scatter_(dim=-1, index=top_indices, src=top_values)
        output = dispatch / dispatch.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    elif block.kind == "expert_hidden":
        output = torch.tanh(torch.einsum("nh,ehm->nem", values["moe_tokens"], values["w1"]))
    elif block.kind == "expert_output":
        output = torch.einsum("nem,emh->neh", values["expert_hidden"], values["w2"])
    elif block.kind == "moe_output":
        combined = torch.sum(values["expert_output"] * values["dispatch"].unsqueeze(-1), dim=1)
        output = combined.reshape(shape.batch, shape.seq_len, shape.hidden)
    else:
        raise ValueError(f"Unsupported block kind: {block.kind}")

    if len(block.outputs) != 1:
        raise ValueError(f"Block {block.name!r} must produce exactly one output value")
    values[block.outputs[0]] = output
    state.value_sources[block.outputs[0]] = block.name
    return output


def _tensor_checksum(tensor: torch.Tensor) -> float:
    value = tensor.detach().float().mean()
    return float(value.item())


def _block_record(
    block: BlockSpec,
    *,
    stream_role: str,
    timed_ms: float,
    output: torch.Tensor,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finite = bool(torch.isfinite(output).all().item())
    checksum = _tensor_checksum(output)
    group = block.resolved_group
    return {
        "name": block.name,
        "group": group,
        "module": group,
        "kind": block.kind,
        "depends_on": list(block.depends_on),
        "inputs": list(block.inputs),
        "outputs": list(block.outputs),
        "stream_role": stream_role,
        "timed_ms": float(timed_ms),
        "finite": finite,
        "checksum": checksum,
        "profile": profile,
    }


def _tensor_metadata(tensor: torch.Tensor) -> dict[str, Any]:
    numel = int(tensor.numel())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": numel,
        "nbytes": int(numel * tensor.element_size()),
    }


def _value_lifetimes(
    *,
    blocks: Sequence[BlockSpec],
    state: _WorkloadState,
) -> list[dict[str, Any]]:
    block_index = {block.name: index for index, block in enumerate(blocks)}
    consumed_by: dict[str, list[str]] = {}
    produced_by: dict[str, str] = {}
    for block in blocks:
        for value_name in block.inputs:
            consumed_by.setdefault(value_name, []).append(block.name)
        for value_name in block.outputs:
            produced_by[value_name] = block.name

    value_names = sorted(set(state.values) | set(consumed_by) | set(produced_by))
    lifetimes: list[dict[str, Any]] = []
    for value_name in value_names:
        tensor = state.values.get(value_name)
        producer = produced_by.get(value_name)
        consumers = consumed_by.get(value_name, [])
        consumer_indexes = [block_index[name] for name in consumers if name in block_index]
        producer_index = block_index[producer] if producer is not None and producer in block_index else None
        first_candidates = consumer_indexes + ([] if producer_index is None else [producer_index])
        last_candidates = consumer_indexes + ([] if producer_index is None else [producer_index])
        source = state.value_sources.get(value_name, "intermediate")
        record: dict[str, Any] = {
            "name": value_name,
            "source": source,
            "persistent": source in {"input", "weight"},
            "input": source in {"input", "weight"},
            "produced_by": producer,
            "consumed_by": consumers,
            "first_block_index": min(first_candidates) if first_candidates else None,
            "last_block_index": max(last_candidates) if last_candidates else None,
        }
        if tensor is None:
            record.update(
                {
                    "shape": None,
                    "dtype": None,
                    "device": None,
                    "numel": None,
                    "nbytes": None,
                }
            )
        else:
            record.update(_tensor_metadata(tensor))
        lifetimes.append(record)
    return lifetimes


def _final_checksums(state: _WorkloadState) -> dict[str, float]:
    return {name: _tensor_checksum(state.values[name]) for name in FINAL_OUTPUTS if name in state.values}


def _timed_cpu_block(block: BlockSpec, state: _WorkloadState, shape: ShapeConfig) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    output = _run_block(block, state, shape)
    return output, (time.perf_counter() - start) * 1000.0


def _run_serial(
    *,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
    stream_role: str,
) -> tuple[list[dict[str, Any]], float]:
    block_ms = {block.name: 0.0 for block in blocks}
    outputs: dict[str, torch.Tensor] = {}
    timed_wall_start: float | None = None

    for iter_idx in range(warmup_iters + timed_iters):
        timed = iter_idx >= warmup_iters
        if timed and timed_wall_start is None:
            timed_wall_start = time.perf_counter()
        for wave in waves:
            for block in wave:
                if device.type == "cuda":
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    start_event.record()
                    output = _run_block(block, state, shape)
                    end_event.record()
                    end_event.synchronize()
                    elapsed_ms = float(start_event.elapsed_time(end_event))
                else:
                    output, elapsed_ms = _timed_cpu_block(block, state, shape)
                outputs[block.name] = output
                if timed:
                    block_ms[block.name] += elapsed_ms
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_clock_ms = 0.0 if timed_wall_start is None else (time.perf_counter() - timed_wall_start) * 1000.0
    return (
        [
            _block_record(
                block,
                stream_role=stream_role,
                timed_ms=block_ms[block.name],
                output=outputs[block.name],
            )
            for block in blocks
        ],
        wall_clock_ms,
    )


def _run_stream_cuda(
    *,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
) -> tuple[list[dict[str, Any]], float]:
    streams = {block.name: torch.cuda.Stream(device=device) for block in blocks}
    block_ms = {block.name: 0.0 for block in blocks}
    outputs: dict[str, torch.Tensor] = {}
    timed_wall_start: float | None = None
    ready_event = torch.cuda.Event(enable_timing=False)
    ready_event.record(torch.cuda.current_stream(device))

    for iter_idx in range(warmup_iters + timed_iters):
        timed = iter_idx >= warmup_iters
        if timed and timed_wall_start is None:
            timed_wall_start = time.perf_counter()
        completion_events: dict[str, torch.cuda.Event] = {}
        timing_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}

        for wave in waves:
            for block in wave:
                stream = streams[block.name]
                completion = torch.cuda.Event(enable_timing=False)
                with torch.cuda.stream(stream):
                    stream.wait_event(ready_event)
                    for dependency in block.depends_on:
                        stream.wait_event(completion_events[dependency])
                    if timed:
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record(stream)
                        outputs[block.name] = _run_block(block, state, shape)
                        end_event.record(stream)
                        timing_events[block.name] = (start_event, end_event)
                    else:
                        outputs[block.name] = _run_block(block, state, shape)
                    completion.record(stream)
                completion_events[block.name] = completion

        for event in completion_events.values():
            event.synchronize()
        if timed:
            for block_name, (start_event, end_event) in timing_events.items():
                block_ms[block_name] += float(start_event.elapsed_time(end_event))

    torch.cuda.current_stream(device).wait_stream(streams[blocks[-1].name])
    torch.cuda.synchronize(device)
    wall_clock_ms = 0.0 if timed_wall_start is None else (time.perf_counter() - timed_wall_start) * 1000.0
    return (
        [
            _block_record(
                block,
                stream_role="per_block",
                timed_ms=block_ms[block.name],
                output=outputs[block.name],
            )
            for block in blocks
        ],
        wall_clock_ms,
    )


def _memory_peaks(device: torch.device) -> tuple[int | None, int | None]:
    if device.type != "cuda":
        return None, None
    return (
        int(torch.cuda.max_memory_allocated(device)),
        int(torch.cuda.max_memory_reserved(device)),
    )


def _profile_label(block: BlockSpec) -> str:
    return f"{PROFILE_LABEL_PREFIX}{block.name}"


def _profiler_summary_disabled() -> dict[str, Any]:
    return {
        "enabled": False,
        "activities": [],
        "profile_memory": False,
        "trace_path": None,
        "trace_exported": False,
        "synchronizes_blocks_for_memory_peaks": False,
        "utilization_proxy_note": (
            "PyTorch profiler does not report raw SM utilization here; CUDA/device time divided by "
            "host wall time is only a utilization-like proxy."
        ),
    }


def _profiler_summary_skipped(*, reason: str, trace_path: str | None = None) -> dict[str, Any]:
    summary = _profiler_summary_disabled()
    summary.update(
        {
            "requested": True,
            "skip_reason": reason,
            "trace_path": trace_path,
        }
    )
    return summary


def _event_metric_ms(event: Any, names: Sequence[str]) -> float | None:
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return float(value) / 1000.0
    return None


def _event_metric_int(event: Any, names: Sequence[str]) -> int | None:
    for name in names:
        if hasattr(event, name):
            value = getattr(event, name)
            if value is not None:
                return int(value)
    return None


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _extract_profiler_metrics(profiler: Any, *, cuda_profiled: bool) -> dict[str, dict[str, Any]]:
    metrics: dict[str, dict[str, Any]] = {}
    for event in profiler.key_averages():
        key = getattr(event, "key", None)
        if not isinstance(key, str) or not key.startswith(PROFILE_LABEL_PREFIX):
            continue
        device_time_total_ms = _event_metric_ms(event, ("device_time_total", "cuda_time_total"))
        self_device_time_total_ms = _event_metric_ms(event, ("self_device_time_total", "self_cuda_time_total"))
        metrics[key] = {
            "profiler_event_count": int(getattr(event, "count", 0)),
            "profiler_cpu_time_total_ms": _event_metric_ms(event, ("cpu_time_total",)),
            "profiler_self_cpu_time_total_ms": _event_metric_ms(event, ("self_cpu_time_total",)),
            "profiler_device_time_total_ms": device_time_total_ms,
            "profiler_self_device_time_total_ms": self_device_time_total_ms,
            "profiler_cuda_time_total_ms": device_time_total_ms if cuda_profiled else None,
            "profiler_self_cuda_time_total_ms": self_device_time_total_ms if cuda_profiled else None,
            "profiler_cpu_memory_usage_bytes": _event_metric_int(event, ("cpu_memory_usage",)),
            "profiler_self_cpu_memory_usage_bytes": _event_metric_int(event, ("self_cpu_memory_usage",)),
            "profiler_device_memory_usage_bytes": (
                _event_metric_int(event, ("device_memory_usage",)) if cuda_profiled else None
            ),
            "profiler_self_device_memory_usage_bytes": (
                _event_metric_int(event, ("self_device_memory_usage",)) if cuda_profiled else None
            ),
        }
    return metrics


def _new_block_profile(block: BlockSpec) -> dict[str, Any]:
    return {
        "record_function": _profile_label(block),
        "wall_ms": 0.0,
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
    }


def _accumulate_profile_wall(profile: dict[str, Any], wall_ms: float) -> None:
    profile["wall_ms"] = float(profile["wall_ms"]) + float(wall_ms)


def _accumulate_profile_peaks(
    profile: dict[str, Any],
    *,
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
) -> None:
    if peak_allocated_bytes is not None:
        current = profile["peak_allocated_bytes"]
        profile["peak_allocated_bytes"] = (
            peak_allocated_bytes if current is None else max(current, peak_allocated_bytes)
        )
    if peak_reserved_bytes is not None:
        current = profile["peak_reserved_bytes"]
        profile["peak_reserved_bytes"] = peak_reserved_bytes if current is None else max(current, peak_reserved_bytes)


def _finalize_block_profiles(
    *,
    block_records: list[dict[str, Any]],
    profiler_metrics: dict[str, dict[str, Any]],
    cuda_profiled: bool,
) -> None:
    for record in block_records:
        profile = record["profile"] or {}
        metrics = profiler_metrics.get(str(profile.get("record_function")), {})
        profile.update(metrics)
        cuda_time_total_ms = profile.get("profiler_cuda_time_total_ms") if cuda_profiled else None
        profile["cuda_time_total_ms_per_wall_ms_proxy"] = _safe_ratio(cuda_time_total_ms, profile.get("wall_ms"))
        busy_fraction = profile["cuda_time_total_ms_per_wall_ms_proxy"]
        profile["device_busy_fraction_proxy"] = None if busy_fraction is None else min(1.0, float(busy_fraction))
        profile["utilization_proxy_note"] = (
            "CUDA/device time over host wall time from PyTorch profiler; this is not raw SM utilization."
        )
        record["profile"] = profile


def _run_warmup_iterations(
    *,
    schedule: str,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    warmup_iters: int,
) -> None:
    if warmup_iters <= 0:
        return
    if schedule == "stream" and device.type == "cuda":
        _run_stream_cuda(
            blocks=blocks,
            waves=waves,
            state=state,
            shape=shape,
            device=device,
            warmup_iters=0,
            timed_iters=warmup_iters,
        )
    else:
        _run_serial(
            blocks=blocks,
            waves=waves,
            state=state,
            shape=shape,
            device=device,
            warmup_iters=0,
            timed_iters=warmup_iters,
            stream_role="cpu" if schedule == "stream" and device.type == "cpu" else "default",
        )


def _run_profiled_serial(
    *,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    timed_iters: int,
    stream_role: str,
) -> tuple[list[dict[str, Any]], float]:
    block_ms = {block.name: 0.0 for block in blocks}
    outputs: dict[str, torch.Tensor] = {}
    profiles = {block.name: _new_block_profile(block) for block in blocks}

    timed_wall_start = time.perf_counter()
    for _ in range(timed_iters):
        for wave in waves:
            for block in wave:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    wall_start = time.perf_counter()
                    with torch.profiler.record_function(_profile_label(block)):
                        start_event.record()
                        output = _run_block(block, state, shape)
                        end_event.record()
                    end_event.synchronize()
                    elapsed_ms = float(start_event.elapsed_time(end_event))
                    peak_allocated, peak_reserved = _memory_peaks(device)
                    wall_ms = (time.perf_counter() - wall_start) * 1000.0
                else:
                    wall_start = time.perf_counter()
                    with torch.profiler.record_function(_profile_label(block)):
                        output = _run_block(block, state, shape)
                    wall_ms = (time.perf_counter() - wall_start) * 1000.0
                    elapsed_ms = wall_ms
                    peak_allocated, peak_reserved = None, None
                outputs[block.name] = output
                block_ms[block.name] += elapsed_ms
                _accumulate_profile_wall(profiles[block.name], wall_ms)
                _accumulate_profile_peaks(
                    profiles[block.name],
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_clock_ms = (time.perf_counter() - timed_wall_start) * 1000.0
    return (
        [
            _block_record(
                block,
                stream_role=stream_role,
                timed_ms=block_ms[block.name],
                output=outputs[block.name],
                profile=profiles[block.name],
            )
            for block in blocks
        ],
        wall_clock_ms,
    )


def _run_profiled_stream_cuda(
    *,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    timed_iters: int,
) -> tuple[list[dict[str, Any]], float]:
    streams = {block.name: torch.cuda.Stream(device=device) for block in blocks}
    block_ms = {block.name: 0.0 for block in blocks}
    outputs: dict[str, torch.Tensor] = {}
    profiles = {block.name: _new_block_profile(block) for block in blocks}
    ready_event = torch.cuda.Event(enable_timing=False)
    ready_event.record(torch.cuda.current_stream(device))

    timed_wall_start = time.perf_counter()
    for _ in range(timed_iters):
        completion_events: dict[str, torch.cuda.Event] = {}
        for wave in waves:
            for block in wave:
                stream = streams[block.name]
                completion = torch.cuda.Event(enable_timing=False)
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
                wall_start = time.perf_counter()
                with torch.cuda.stream(stream):
                    stream.wait_event(ready_event)
                    for dependency in block.depends_on:
                        stream.wait_event(completion_events[dependency])
                    with torch.profiler.record_function(_profile_label(block)):
                        start_event.record(stream)
                        outputs[block.name] = _run_block(block, state, shape)
                        end_event.record(stream)
                    completion.record(stream)
                completion.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
                peak_allocated, peak_reserved = _memory_peaks(device)
                wall_ms = (time.perf_counter() - wall_start) * 1000.0
                completion_events[block.name] = completion
                block_ms[block.name] += elapsed_ms
                _accumulate_profile_wall(profiles[block.name], wall_ms)
                _accumulate_profile_peaks(
                    profiles[block.name],
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                )

    torch.cuda.synchronize(device)
    wall_clock_ms = (time.perf_counter() - timed_wall_start) * 1000.0
    return (
        [
            _block_record(
                block,
                stream_role="per_block",
                timed_ms=block_ms[block.name],
                output=outputs[block.name],
                profile=profiles[block.name],
            )
            for block in blocks
        ],
        wall_clock_ms,
    )


def _run_profiled_schedule(
    *,
    schedule: str,
    blocks: Sequence[BlockSpec],
    waves: Sequence[Sequence[BlockSpec]],
    state: _WorkloadState,
    shape: ShapeConfig,
    device: torch.device,
    warmup_iters: int,
    timed_iters: int,
    trace_output: str | Path | None,
) -> tuple[list[dict[str, Any]], float, dict[str, Any]]:
    _run_warmup_iterations(
        schedule=schedule,
        blocks=blocks,
        waves=waves,
        state=state,
        shape=shape,
        device=device,
        warmup_iters=warmup_iters,
    )

    activities = [torch.profiler.ProfilerActivity.CPU]
    activity_names = ["CPU"]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
        activity_names.append("CUDA")

    with torch.profiler.profile(activities=activities, profile_memory=True, acc_events=True) as profiler:
        if schedule == "stream" and device.type == "cuda":
            block_records, wall_clock_ms = _run_profiled_stream_cuda(
                blocks=blocks,
                waves=waves,
                state=state,
                shape=shape,
                device=device,
                timed_iters=timed_iters,
            )
        else:
            block_records, wall_clock_ms = _run_profiled_serial(
                blocks=blocks,
                waves=waves,
                state=state,
                shape=shape,
                device=device,
                timed_iters=timed_iters,
                stream_role="cpu" if schedule == "stream" and device.type == "cpu" else "default",
            )

    trace_path = str(trace_output) if trace_output is not None else None
    trace_exported = False
    if trace_output is not None:
        path = Path(trace_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(path))
        trace_exported = True

    _finalize_block_profiles(
        block_records=block_records,
        profiler_metrics=_extract_profiler_metrics(profiler, cuda_profiled=device.type == "cuda"),
        cuda_profiled=device.type == "cuda",
    )
    return (
        block_records,
        wall_clock_ms,
        {
            "enabled": True,
            "activities": activity_names,
            "profile_memory": True,
            "trace_path": trace_path,
            "trace_exported": trace_exported,
            "synchronizes_blocks_for_memory_peaks": device.type == "cuda",
            "record_function_prefix": PROFILE_LABEL_PREFIX,
            "utilization_proxy_note": (
                "PyTorch profiler does not report raw SM utilization here; per-block "
                "device_busy_fraction_proxy is min(1, profiler CUDA/device time / block wall time)."
            ),
        },
    )


def _invalid_payload(
    *,
    schedule: str,
    requested_device: str,
    resolved_device: torch.device,
    shape: ShapeConfig,
    warmup_iters: int,
    timed_iters: int,
    capabilities: dict[str, Any],
    errors: Sequence[ValidationError],
    profiler: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule": schedule,
        "device": {
            "requested": requested_device,
            "resolved": resolved_device.type,
            "index": resolved_device.index,
        },
        "shape": shape.to_json(),
        "iterations": {
            "warmup": int(warmup_iters),
            "timed": int(timed_iters),
        },
        "capabilities": capabilities,
        "blocks": [],
        "summary": {
            "wall_clock_ms": None,
            "all_finite": None,
            "schedule_valid": False,
            "combined_checksum": None,
            "final_checksums": None,
            "value_lifetimes": [],
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
            "profiler": profiler if profiler is not None else _profiler_summary_disabled(),
            "validation_errors": [error.to_json() for error in errors],
        },
    }


def run_composite_schedule(
    schedule: str = "serial",
    device: str = "auto",
    shape: ShapeConfig | None = None,
    warmup_iters: int = 1,
    timed_iters: int = 1,
    seed: int = 0,
    blocks: Sequence[BlockSpec] | None = None,
    profile: bool = False,
    profile_trace_output: str | Path | None = None,
) -> dict[str, Any]:
    if schedule not in SUPPORTED_SCHEDULES:
        raise ValueError(f"Unsupported schedule {schedule!r}; expected one of {SUPPORTED_SCHEDULES}")
    if warmup_iters < 0 or timed_iters < 0:
        raise ValueError("warmup_iters and timed_iters must be non-negative")

    shape = shape or ShapeConfig()
    shape.validate()
    resolved_device = _resolve_device(device)
    capabilities = probe_runtime_capabilities(device)
    specs = list(blocks) if blocks is not None else default_block_specs()
    validation_errors = validate_schedule(specs)
    if validation_errors:
        return _invalid_payload(
            schedule=schedule,
            requested_device=device,
            resolved_device=resolved_device,
            shape=shape,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            capabilities=capabilities,
            errors=validation_errors,
            profiler=(
                _profiler_summary_skipped(reason="schedule_invalid", trace_path=str(profile_trace_output))
                if profile
                else None
            ),
        )

    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    torch.manual_seed(int(seed))
    state = _build_workload(shape, resolved_device, seed)
    waves, cycle_blocks = _topological_waves(specs)
    if cycle_blocks:
        raise RuntimeError(f"Unexpected cycle after validation: {cycle_blocks}")

    profiler_summary = _profiler_summary_disabled()
    if profile:
        block_records, wall_clock_ms, profiler_summary = _run_profiled_schedule(
            schedule=schedule,
            blocks=specs,
            waves=waves,
            state=state,
            shape=shape,
            device=resolved_device,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            trace_output=profile_trace_output,
        )
    elif schedule == "stream" and resolved_device.type == "cuda":
        block_records, wall_clock_ms = _run_stream_cuda(
            blocks=specs,
            waves=waves,
            state=state,
            shape=shape,
            device=resolved_device,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
        )
    else:
        block_records, wall_clock_ms = _run_serial(
            blocks=specs,
            waves=waves,
            state=state,
            shape=shape,
            device=resolved_device,
            warmup_iters=warmup_iters,
            timed_iters=timed_iters,
            stream_role="cpu" if schedule == "stream" and resolved_device.type == "cpu" else "default",
        )

    peak_allocated, peak_reserved = _memory_peaks(resolved_device)
    capabilities["cuda"]["peak_allocated_bytes"] = peak_allocated
    capabilities["cuda"]["peak_reserved_bytes"] = peak_reserved
    all_finite = all(bool(record["finite"]) for record in block_records)
    combined_checksum = float(sum(float(record["checksum"]) for record in block_records))
    final_checksums = _final_checksums(state)
    value_lifetimes = _value_lifetimes(blocks=specs, state=state)
    return {
        "schema_version": SCHEMA_VERSION,
        "schedule": schedule,
        "device": {
            "requested": device,
            "resolved": resolved_device.type,
            "index": resolved_device.index,
        },
        "shape": shape.to_json(),
        "iterations": {
            "warmup": int(warmup_iters),
            "timed": int(timed_iters),
        },
        "capabilities": capabilities,
        "blocks": block_records,
        "summary": {
            "wall_clock_ms": float(wall_clock_ms),
            "all_finite": all_finite,
            "schedule_valid": True,
            "combined_checksum": combined_checksum,
            "final_checksums": final_checksums,
            "value_lifetimes": value_lifetimes,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "profiler": profiler_summary,
            "validation_errors": [],
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", choices=SUPPORTED_SCHEDULES, default="serial")
    parser.add_argument("--device", choices=SUPPORTED_DEVICES, default="auto")
    parser.add_argument("--batch", type=int, default=ShapeConfig.batch)
    parser.add_argument("--seq-len", type=int, default=ShapeConfig.seq_len)
    parser.add_argument("--hidden", type=int, default=ShapeConfig.hidden)
    parser.add_argument("--num-experts", type=int, default=ShapeConfig.num_experts)
    parser.add_argument("--top-k", type=int, default=ShapeConfig.top_k)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--timed-iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-trace-output", type=str, default=None)
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    profile_trace_output = args.profile_trace_output
    if args.profile and profile_trace_output is None:
        profile_trace_output = "block_profile_trace.json"
    payload = run_composite_schedule(
        schedule=args.schedule,
        device=args.device,
        shape=ShapeConfig(
            batch=args.batch,
            seq_len=args.seq_len,
            hidden=args.hidden,
            num_experts=args.num_experts,
            top_k=args.top_k,
        ),
        warmup_iters=args.warmup_iters,
        timed_iters=args.timed_iters,
        seed=args.seed,
        profile=args.profile,
        profile_trace_output=profile_trace_output,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
