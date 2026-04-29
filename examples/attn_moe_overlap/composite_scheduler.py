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

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
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
    attention_input: torch.Tensor
    moe_input: torch.Tensor
    moe_router: torch.Tensor
    moe_w1: torch.Tensor
    moe_w2: torch.Tensor


def default_block_specs() -> list[BlockSpec]:
    return [
        BlockSpec(name="attention", kind="attention"),
        BlockSpec(name="moe", kind="moe"),
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
        if block.kind not in {"attention", "moe"}:
            errors.append(
                ValidationError(
                    code="unknown_kind",
                    message=f"Block {block.name!r} has unsupported kind {block.kind!r}",
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
        ready_names = sorted(name for name in remaining if all(dep in satisfied for dep in by_name[name].depends_on))
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
        attention_input=_make_tensor((batch, seq_len, hidden), device=device, seed=seed + 11, scale=0.2),
        moe_input=_make_tensor((batch, seq_len, hidden), device=device, seed=seed + 23, scale=0.2),
        moe_router=_make_linspace_tensor((hidden, shape.num_experts), device=device, scale=0.05),
        moe_w1=_make_linspace_tensor((shape.num_experts, hidden, hidden), device=device, scale=0.04),
        moe_w2=_make_linspace_tensor((shape.num_experts, hidden, hidden), device=device, scale=0.04),
    )


def _run_attention(state: _WorkloadState) -> torch.Tensor:
    x = state.attention_input
    scores = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(float(x.shape[-1]))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, x)


def _run_moe(state: _WorkloadState, shape: ShapeConfig) -> torch.Tensor:
    x = state.moe_input
    tokens = x.reshape(-1, shape.hidden)
    router_probs = torch.softmax(tokens @ state.moe_router, dim=-1)
    top_values, top_indices = torch.topk(router_probs, k=shape.top_k, dim=-1)
    dispatch = torch.zeros_like(router_probs)
    dispatch.scatter_(dim=-1, index=top_indices, src=top_values)
    dispatch = dispatch / dispatch.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    hidden = torch.tanh(torch.einsum("nh,ehm->nem", tokens, state.moe_w1))
    expert_out = torch.einsum("nem,emh->neh", hidden, state.moe_w2)
    combined = torch.sum(expert_out * dispatch.unsqueeze(-1), dim=1)
    return combined.reshape(shape.batch, shape.seq_len, shape.hidden)


def _run_block(block: BlockSpec, state: _WorkloadState, shape: ShapeConfig) -> torch.Tensor:
    if block.kind == "attention":
        return _run_attention(state)
    if block.kind == "moe":
        return _run_moe(state, shape)
    raise ValueError(f"Unsupported block kind: {block.kind}")


def _tensor_checksum(tensor: torch.Tensor) -> float:
    value = tensor.detach().float().mean()
    return float(value.item())


def _block_record(
    block: BlockSpec,
    *,
    stream_role: str,
    timed_ms: float,
    output: torch.Tensor,
) -> dict[str, Any]:
    finite = bool(torch.isfinite(output).all().item())
    checksum = _tensor_checksum(output)
    return {
        "name": block.name,
        "kind": block.kind,
        "depends_on": list(block.depends_on),
        "stream_role": stream_role,
        "timed_ms": float(timed_ms),
        "finite": finite,
        "checksum": checksum,
    }


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
            "peak_allocated_bytes": None,
            "peak_reserved_bytes": None,
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
        )

    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)

    torch.manual_seed(int(seed))
    state = _build_workload(shape, resolved_device, seed)
    waves, cycle_blocks = _topological_waves(specs)
    if cycle_blocks:
        raise RuntimeError(f"Unexpected cycle after validation: {cycle_blocks}")

    if schedule == "stream" and resolved_device.type == "cuda":
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
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
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
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
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
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.json_output:
        Path(args.json_output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
