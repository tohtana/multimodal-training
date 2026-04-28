"""Step 8: isolated attention/MoE CUDA memory sweep."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

DEFAULT_SEQ_LENS = (1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_BATCH_SIZES = (1, 2)
DEFAULT_MODEL_NAME = "Qwen/Qwen3-30B-A3B"
VALID_MODULES = ("attention", "moe")
VALID_MODES = ("forward", "forward_backward")
DTYPE_ALIASES = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp16": "fp16",
    "float16": "fp16",
    "fp32": "fp32",
    "float32": "fp32",
}
TORCH_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}
MEMORY_PREFIXES = (
    "max_memory_allocated",
    "max_memory_reserved",
    "memory_allocated_before",
    "memory_reserved_before",
    "memory_allocated_after",
    "memory_reserved_after",
)
ROW_FIELDNAMES = (
    "module",
    "mode",
    "seq_len",
    "batch_size",
    "status",
    "status_reason",
    "warmup_iters",
    "timed_iters",
    "elapsed_ms_total",
    "elapsed_ms_mean",
    "tokens_per_iter",
    "tokens_per_second",
    "max_memory_allocated_bytes",
    "max_memory_allocated_mib",
    "max_memory_allocated_gib",
    "max_memory_reserved_bytes",
    "max_memory_reserved_mib",
    "max_memory_reserved_gib",
    "memory_allocated_before_bytes",
    "memory_allocated_before_mib",
    "memory_allocated_before_gib",
    "memory_reserved_before_bytes",
    "memory_reserved_before_mib",
    "memory_reserved_before_gib",
    "memory_allocated_after_bytes",
    "memory_allocated_after_mib",
    "memory_allocated_after_gib",
    "memory_reserved_after_bytes",
    "memory_reserved_after_mib",
    "memory_reserved_after_gib",
    "gpu_type",
    "gpu_name",
    "device",
    "dtype",
    "world_size",
    "num_gpus_used",
    "expert_parallel_size",
    "tensor_parallel_size",
    "model_name",
    "model_revision",
    "model_commit_hash",
    "model_type",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "attention_backend",
    "config_error",
    "cuda_version",
    "cuda_driver_version",
    "pytorch_version",
    "python_version",
    "commit_sha",
    "multimodal_training_commit_sha",
    "branch",
    "command",
    "started_at_utc",
    "completed_at_utc",
)


@dataclass(frozen=True)
class ResolvedModelConfig:
    model_name: str
    model_revision: str | None
    model_commit_hash: str | None
    model_type: str | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    num_experts: int | None
    num_experts_per_tok: int | None
    moe_intermediate_size: int | None
    attention_backend: str | None
    config_available: bool
    config_error: str | None = None


@dataclass(frozen=True)
class BenchmarkConfig:
    modules: tuple[str, ...]
    modes: tuple[str, ...]
    seq_lens: tuple[int, ...]
    batch_sizes: tuple[int, ...]
    dtype_name: str
    device: str
    warmup_iters: int
    timed_iters: int
    seed: int
    model: ResolvedModelConfig


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_int_csv(raw: str, *, field_name: str) -> tuple[int, ...]:
    values: list[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be comma-separated integers: {raw!r}") from exc
        if value <= 0:
            raise ValueError(f"{field_name} values must be positive: got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(values)


def parse_modules(raw: str) -> tuple[str, ...]:
    return _parse_choices(raw, valid=VALID_MODULES, aliases={"attn": "attention"}, field_name="modules")


def parse_modes(raw: str) -> tuple[str, ...]:
    aliases = {
        "fwd": "forward",
        "forward-only": "forward",
        "forward_only": "forward",
        "fwd_bwd": "forward_backward",
        "forward+backward": "forward_backward",
        "forward-backward": "forward_backward",
    }
    return _parse_choices(raw, valid=VALID_MODES, aliases=aliases, field_name="modes")


def normalize_dtype_name(raw: str) -> str:
    normalized = raw.strip().lower()
    if normalized not in DTYPE_ALIASES:
        raise ValueError(f"dtype must be one of {sorted(DTYPE_ALIASES)}, got {raw!r}")
    return DTYPE_ALIASES[normalized]


def _parse_choices(
    raw: str,
    *,
    valid: tuple[str, ...],
    aliases: dict[str, str],
    field_name: str,
) -> tuple[str, ...]:
    if raw.strip().lower() == "all":
        return valid
    values: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        text = chunk.strip().lower()
        if not text:
            continue
        normalized = aliases.get(text, text)
        if normalized not in valid:
            raise ValueError(f"{field_name} must contain values from {valid} or 'all', got {text!r}")
        if normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    return tuple(values)


def bytes_to_units(value: int | None) -> dict[str, int | float | None]:
    if value is None:
        return {"bytes": None, "mib": None, "gib": None}
    return {
        "bytes": int(value),
        "mib": float(value) / (1024.0**2),
        "gib": float(value) / (1024.0**3),
    }


def add_memory_units(row: dict[str, Any], prefix: str, value: int | None) -> None:
    units = bytes_to_units(value)
    row[f"{prefix}_bytes"] = units["bytes"]
    row[f"{prefix}_mib"] = units["mib"]
    row[f"{prefix}_gib"] = units["gib"]


def canonical_matrix(seq_lens: tuple[int, ...], batch_sizes: tuple[int, ...]) -> list[tuple[int, int]]:
    return [(seq_len, batch_size) for seq_len in seq_lens for batch_size in batch_sizes]


def expected_row_keys(config: BenchmarkConfig) -> list[tuple[str, str, int, int]]:
    return [
        (module, mode, seq_len, batch_size)
        for module in config.modules
        for mode in config.modes
        for seq_len, batch_size in canonical_matrix(config.seq_lens, config.batch_sizes)
    ]


def _get_int_attr(config_obj: Any, attr_name: str) -> int | None:
    value = getattr(config_obj, attr_name, None)
    if value is None:
        return None
    return int(value)


def _config_commit_hash(config_obj: Any) -> str | None:
    commit_hash = getattr(config_obj, "_commit_hash", None)
    if commit_hash:
        return str(commit_hash)
    try:
        raw_dict = config_obj.to_dict()
    except Exception:
        return None
    commit_hash = raw_dict.get("_commit_hash")
    return str(commit_hash) if commit_hash else None


def _attention_backend(config_obj: Any) -> str:
    for attr_name in ("attention_backend", "_attn_implementation", "attn_implementation"):
        value = getattr(config_obj, attr_name, None)
        if value:
            return str(value)
    return "sdpa"


def resolve_model_config(
    *,
    model_name: str,
    model_revision: str | None,
    auto_config_cls: Any | None = None,
) -> ResolvedModelConfig:
    """Resolve Qwen module dimensions from transformers config without model weights."""
    try:
        if auto_config_cls is None:
            from transformers import AutoConfig

            auto_config_cls = AutoConfig
        kwargs: dict[str, Any] = {}
        if model_revision:
            kwargs["revision"] = model_revision
        config_obj = auto_config_cls.from_pretrained(model_name, **kwargs)
    except Exception as exc:  # noqa: BLE001 - unavailable config becomes skipped rows.
        return ResolvedModelConfig(
            model_name=model_name,
            model_revision=model_revision,
            model_commit_hash=None,
            model_type=None,
            hidden_size=None,
            num_attention_heads=None,
            num_key_value_heads=None,
            head_dim=None,
            num_experts=None,
            num_experts_per_tok=None,
            moe_intermediate_size=None,
            attention_backend=None,
            config_available=False,
            config_error=f"{exc.__class__.__name__}: {exc}",
        )

    hidden_size = _get_int_attr(config_obj, "hidden_size")
    num_attention_heads = _get_int_attr(config_obj, "num_attention_heads")
    num_key_value_heads = _get_int_attr(config_obj, "num_key_value_heads")
    head_dim = _get_int_attr(config_obj, "head_dim")
    num_experts = _get_int_attr(config_obj, "num_experts")
    num_experts_per_tok = _get_int_attr(config_obj, "num_experts_per_tok")
    moe_intermediate_size = _get_int_attr(config_obj, "moe_intermediate_size")
    if head_dim is None and hidden_size is not None and num_attention_heads is not None:
        if hidden_size % num_attention_heads != 0:
            return ResolvedModelConfig(
                model_name=model_name,
                model_revision=model_revision,
                model_commit_hash=_config_commit_hash(config_obj),
                model_type=getattr(config_obj, "model_type", None),
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=None,
                num_experts=num_experts,
                num_experts_per_tok=num_experts_per_tok,
                moe_intermediate_size=moe_intermediate_size,
                attention_backend=_attention_backend(config_obj),
                config_available=False,
                config_error="hidden_size is not divisible by num_attention_heads",
            )
        head_dim = hidden_size // num_attention_heads

    missing = [
        field_name
        for field_name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", num_attention_heads),
            ("num_key_value_heads", num_key_value_heads),
            ("head_dim", head_dim),
            ("num_experts", num_experts),
            ("num_experts_per_tok", num_experts_per_tok),
            ("moe_intermediate_size", moe_intermediate_size),
        )
        if value is None
    ]
    config_available = not missing
    config_error = None if config_available else f"missing config fields: {', '.join(missing)}"
    commit_hash = _config_commit_hash(config_obj)
    return ResolvedModelConfig(
        model_name=model_name,
        model_revision=commit_hash or model_revision,
        model_commit_hash=commit_hash,
        model_type=getattr(config_obj, "model_type", None),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        moe_intermediate_size=moe_intermediate_size,
        attention_backend=_attention_backend(config_obj),
        config_available=config_available,
        config_error=config_error,
    )


def require_model_int(model: ResolvedModelConfig, field_name: str) -> int:
    value = getattr(model, field_name)
    if value is None:
        raise RuntimeError(f"model config field unavailable: {field_name}")
    return int(value)


class IsolatedAttention(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
    ) -> None:
        super().__init__()
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.hidden_size = int(hidden_size)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.q_width = self.num_attention_heads * self.head_dim
        self.kv_width = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(self.hidden_size, self.q_width, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_width, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_width, bias=False)
        self.out = nn.Linear(self.q_width, self.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        y = self._scaled_dot_product_attention(q, k, v)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_width)
        return self.out(y)

    def _scaled_dot_product_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self.num_attention_heads == self.num_key_value_heads:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        try:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, enable_gqa=True)
        except TypeError:
            repeat_factor = self.num_attention_heads // self.num_key_value_heads
            return F.scaled_dot_product_attention(
                q,
                k.repeat_interleave(repeat_factor, dim=1),
                v.repeat_interleave(repeat_factor, dim=1),
                dropout_p=0.0,
                is_causal=False,
            )


class ExpertMLP(nn.Module):
    def __init__(self, *, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class BalancedTopKMoE(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in [1, num_experts]")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.experts = nn.ModuleList(
            ExpertMLP(hidden_size=hidden_size, intermediate_size=intermediate_size)
            for _ in range(self.num_experts)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        positions = torch.arange(flat.shape[0], device=flat.device)
        route_offsets = torch.arange(self.top_k, device=flat.device)
        routed_experts = ((positions[:, None] + route_offsets[None, :]) % self.num_experts).reshape(-1)
        routed_tokens = positions.repeat_interleave(self.top_k)
        output = torch.zeros_like(flat)
        for expert_idx, expert in enumerate(self.experts):
            route_idx = torch.nonzero(routed_experts == expert_idx, as_tuple=False).flatten()
            if route_idx.numel() == 0:
                continue
            token_idx = routed_tokens.index_select(0, route_idx)
            expert_input = flat.index_select(0, token_idx)
            expert_output = expert(expert_input)
            output.index_add_(0, token_idx, expert_output)
        return (output / float(self.top_k)).view_as(x)


def build_module(config: BenchmarkConfig, module_name: str, dtype: torch.dtype, device: torch.device) -> nn.Module:
    model = config.model
    if module_name == "attention":
        module = IsolatedAttention(
            hidden_size=require_model_int(model, "hidden_size"),
            num_attention_heads=require_model_int(model, "num_attention_heads"),
            num_key_value_heads=require_model_int(model, "num_key_value_heads"),
            head_dim=require_model_int(model, "head_dim"),
        )
    elif module_name == "moe":
        module = BalancedTopKMoE(
            hidden_size=require_model_int(model, "hidden_size"),
            intermediate_size=require_model_int(model, "moe_intermediate_size"),
            num_experts=require_model_int(model, "num_experts"),
            top_k=require_model_int(model, "num_experts_per_tok"),
        )
    else:
        raise ValueError(f"Unsupported module: {module_name}")
    return module.to(device=device, dtype=dtype).train()


def resolve_device(device_name: str) -> torch.device:
    device = torch.device(device_name)
    if device.type == "cuda" and device.index is None:
        return torch.device("cuda:0")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_grads(module: nn.Module, x: torch.Tensor, *, set_to_none: bool) -> None:
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        if set_to_none:
            parameter.grad = None
        else:
            parameter.grad.zero_()
    if x.grad is not None:
        if set_to_none:
            x.grad = None
        else:
            x.grad.zero_()


def run_step(module: nn.Module, x: torch.Tensor, mode: str) -> None:
    if mode == "forward":
        with torch.no_grad():
            module(x)
        return

    clear_grads(module, x, set_to_none=False)
    y = module(x)
    loss = y.float().square().mean()
    loss.backward()
    clear_grads(module, x, set_to_none=False)


def cuda_memory_snapshot(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "allocated": None,
            "reserved": None,
            "max_allocated": None,
            "max_reserved": None,
        }
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "max_allocated": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cleanup_cuda(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


def classify_exception(exc: BaseException) -> tuple[str, str]:
    message = str(exc).splitlines()[0] if str(exc) else exc.__class__.__name__
    if isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in message.lower():
        return "oom", message
    return "runtime_error", message


def measure_case(
    *,
    config: BenchmarkConfig,
    module_name: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    started_at = now_utc_iso()
    device = resolve_device(config.device)
    dtype = TORCH_DTYPES[config.dtype_name]
    row = base_row(
        config=config,
        module_name=module_name,
        mode=mode,
        seq_len=seq_len,
        batch_size=batch_size,
        metadata=metadata,
        started_at=started_at,
    )

    module: nn.Module | None = None
    x: torch.Tensor | None = None
    try:
        hidden_size = require_model_int(config.model, "hidden_size")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
        if device.type == "cuda":
            torch.cuda.set_device(device)

        torch.manual_seed(config.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        cleanup_cuda(device)

        module = build_module(config, module_name, dtype, device)
        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        if mode == "forward_backward":
            x.requires_grad_(True)

        synchronize(device)
        for _ in range(config.warmup_iters):
            run_step(module, x, mode)
        synchronize(device)

        before = cuda_memory_snapshot(device)
        reset_peak_memory(device)
        synchronize(device)

        start_s = time.perf_counter()
        for _ in range(config.timed_iters):
            run_step(module, x, mode)
        synchronize(device)
        elapsed_ms = (time.perf_counter() - start_s) * 1000.0
        after = cuda_memory_snapshot(device)

        tokens_per_iter = int(seq_len * batch_size)
        elapsed_mean = elapsed_ms / float(config.timed_iters)
        row.update(
            {
                "status": "ok",
                "status_reason": "",
                "elapsed_ms_total": elapsed_ms,
                "elapsed_ms_mean": elapsed_mean,
                "tokens_per_iter": tokens_per_iter,
                "tokens_per_second": tokens_per_iter / (elapsed_mean / 1000.0),
            }
        )
        add_memory_units(row, "memory_allocated_before", before["allocated"])
        add_memory_units(row, "memory_reserved_before", before["reserved"])
        add_memory_units(row, "memory_allocated_after", after["allocated"])
        add_memory_units(row, "memory_reserved_after", after["reserved"])
        add_memory_units(row, "max_memory_allocated", after["max_allocated"])
        add_memory_units(row, "max_memory_reserved", after["max_reserved"])
    except BaseException as exc:  # noqa: BLE001 - every matrix cell must become a status row.
        status, reason = classify_exception(exc)
        snapshot = cuda_memory_snapshot(device) if device.type == "cuda" and torch.cuda.is_available() else {}
        row.update(
            {
                "status": status,
                "status_reason": reason,
                "traceback": traceback.format_exc(),
            }
        )
        add_memory_units(row, "memory_allocated_after", snapshot.get("allocated"))
        add_memory_units(row, "memory_reserved_after", snapshot.get("reserved"))
        add_memory_units(row, "max_memory_allocated", snapshot.get("max_allocated"))
        add_memory_units(row, "max_memory_reserved", snapshot.get("max_reserved"))
    finally:
        if module is not None and x is not None:
            clear_grads(module, x, set_to_none=True)
        del module
        del x
        cleanup_cuda(device)

    row["completed_at_utc"] = now_utc_iso()
    return row


def skipped_case(
    *,
    config: BenchmarkConfig,
    module_name: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    row = base_row(
        config=config,
        module_name=module_name,
        mode=mode,
        seq_len=seq_len,
        batch_size=batch_size,
        metadata=metadata,
        started_at=now_utc_iso(),
    )
    row.update(
        {
            "status": "skipped",
            "status_reason": "config_unavailable",
            "config_error": config.model.config_error,
            "completed_at_utc": now_utc_iso(),
        }
    )
    return row


def base_row(
    *,
    config: BenchmarkConfig,
    module_name: str,
    mode: str,
    seq_len: int,
    batch_size: int,
    metadata: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    model = config.model
    row: dict[str, Any] = {
        "module": module_name,
        "mode": mode,
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "status": "not_run",
        "status_reason": "",
        "warmup_iters": int(config.warmup_iters),
        "timed_iters": int(config.timed_iters),
        "elapsed_ms_total": None,
        "elapsed_ms_mean": None,
        "tokens_per_iter": int(seq_len * batch_size),
        "tokens_per_second": None,
        "gpu_type": metadata.get("gpu_type"),
        "gpu_name": metadata.get("gpu_name"),
        "device": config.device,
        "dtype": config.dtype_name,
        "world_size": 1,
        "num_gpus_used": metadata.get("num_gpus_used"),
        "expert_parallel_size": 1,
        "tensor_parallel_size": 1,
        "model_name": model.model_name,
        "model_revision": model.model_revision,
        "model_commit_hash": model.model_commit_hash,
        "model_type": model.model_type,
        "hidden_size": model.hidden_size,
        "num_attention_heads": model.num_attention_heads,
        "num_key_value_heads": model.num_key_value_heads,
        "head_dim": model.head_dim,
        "num_experts": model.num_experts,
        "num_experts_per_tok": model.num_experts_per_tok,
        "moe_intermediate_size": model.moe_intermediate_size,
        "attention_backend": model.attention_backend,
        "config_error": model.config_error,
        "cuda_version": metadata.get("cuda_version"),
        "cuda_driver_version": metadata.get("cuda_driver_version"),
        "pytorch_version": metadata.get("pytorch_version"),
        "python_version": metadata.get("python_version"),
        "commit_sha": metadata.get("commit_sha"),
        "multimodal_training_commit_sha": metadata.get("multimodal_training_commit_sha"),
        "branch": metadata.get("branch"),
        "command": metadata.get("command"),
        "started_at_utc": started_at,
        "completed_at_utc": None,
    }
    for prefix in MEMORY_PREFIXES:
        add_memory_units(row, prefix, None)
    return row


def detect_gpu_type(name: str | None) -> str:
    if not name:
        return "unknown"
    for token in ("H100", "A100", "L40S", "L40", "T4"):
        if token in name.upper():
            return token
    return name


def run_text(cmd: list[str], *, cwd: Path) -> str | None:
    try:
        result = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
    except Exception:
        return None
    return result.stdout.strip() or None


def query_nvidia_driver() -> str | None:
    result = run_text(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        cwd=Path.cwd(),
    )
    if not result:
        return None
    return result.splitlines()[0].strip()


def build_metadata(config: BenchmarkConfig, argv: list[str]) -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    multimodal_root = script_path.parents[2]
    repo_root = script_path.parents[3]
    device = resolve_device(config.device)
    gpu_name = None
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(device)
        gpu_name = torch.cuda.get_device_name(device)

    return {
        "schema_version": "step8_attn_moe_memory_sweep.v1",
        "command": shlex.join([sys.executable, str(script_path), *argv]),
        "cwd": str(Path.cwd()),
        "commit_sha": run_text(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "multimodal_training_commit_sha": run_text(["git", "rev-parse", "HEAD"], cwd=multimodal_root),
        "branch": run_text(["git", "branch", "--show-current"], cwd=repo_root) or "detached",
        "gpu_name": gpu_name,
        "gpu_type": detect_gpu_type(gpu_name),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "num_gpus_used": 1 if device.type == "cuda" and torch.cuda.is_available() else 0,
        "cuda_version": torch.version.cuda,
        "cuda_driver_version": query_nvidia_driver(),
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at_utc": now_utc_iso(),
        "matrix_cell_count": len(canonical_matrix(config.seq_lens, config.batch_sizes)),
        "expected_row_count": len(expected_row_keys(config)),
        "model": {
            "model_name": config.model.model_name,
            "model_revision": config.model.model_revision,
            "model_commit_hash": config.model.model_commit_hash,
            "model_type": config.model.model_type,
            "hidden_size": config.model.hidden_size,
            "num_attention_heads": config.model.num_attention_heads,
            "num_key_value_heads": config.model.num_key_value_heads,
            "head_dim": config.model.head_dim,
            "num_experts": config.model.num_experts,
            "num_experts_per_tok": config.model.num_experts_per_tok,
            "moe_intermediate_size": config.model.moe_intermediate_size,
            "attention_backend": config.model.attention_backend,
            "config_available": config.model.config_available,
            "config_error": config.model.config_error,
        },
        "config": {
            "modules": list(config.modules),
            "modes": list(config.modes),
            "seq_lens": list(config.seq_lens),
            "batch_sizes": list(config.batch_sizes),
            "dtype": config.dtype_name,
            "device": config.device,
            "warmup_iters": config.warmup_iters,
            "timed_iters": config.timed_iters,
            "seed": config.seed,
        },
    }


def run_sweep(config: BenchmarkConfig, metadata: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for module_name, mode, seq_len, batch_size in expected_row_keys(config):
        print(f"[case] module={module_name} mode={mode} seq_len={seq_len} batch={batch_size}", flush=True)
        if not config.model.config_available:
            rows.append(
                skipped_case(
                    config=config,
                    module_name=module_name,
                    mode=mode,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    metadata=metadata,
                )
            )
        else:
            rows.append(
                measure_case(
                    config=config,
                    module_name=module_name,
                    mode=mode,
                    seq_len=seq_len,
                    batch_size=batch_size,
                    metadata=metadata,
                )
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted({key for row in rows for key in row} - set(ROW_FIELDNAMES))
    fieldnames = [*ROW_FIELDNAMES, *extra_fields]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fieldnames})


def write_json(path: Path, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "step8_attn_moe_memory_sweep.v1",
        "metadata": metadata,
        "rows": rows,
        "status_counts": build_status_counts(rows),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def format_float(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return ""
    return f"{float(value):.{digits}f}"


def format_memory(row: dict[str, Any], prefix: str) -> str:
    raw = row.get(f"{prefix}_bytes")
    if raw in (None, ""):
        return ""
    return f"{float(row[f'{prefix}_gib']):.3f} GiB ({float(row[f'{prefix}_mib']):.1f} MiB; {int(raw)} B)"


def markdown_cell(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def render_results_markdown(
    *,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> str:
    status_counts = build_status_counts(rows)
    model = metadata["model"]
    command = str(metadata.get("command"))
    if metadata.get("cuda_visible_devices"):
        command = f"CUDA_VISIBLE_DEVICES={metadata['cuda_visible_devices']} {command}"
    lines = [
        "# Step-8 Attention/MoE Memory Sweep Results",
        "",
        "## Summary",
        "",
        f"- Command: `{command}`",
        f"- Commit SHA: `{metadata.get('commit_sha')}`",
        f"- multimodal-training commit SHA: `{metadata.get('multimodal_training_commit_sha')}`",
        f"- Branch: `{metadata.get('branch')}`",
        f"- GPU type/name: `{metadata.get('gpu_type')}` / `{metadata.get('gpu_name')}`",
        f"- GPU count visible: `{metadata.get('gpu_count')}`",
        f"- CUDA_VISIBLE_DEVICES: `{metadata.get('cuda_visible_devices')}`",
        f"- CUDA runtime / driver: `{metadata.get('cuda_version')}` / `{metadata.get('cuda_driver_version')}`",
        f"- PyTorch / Python: `{metadata.get('pytorch_version')}` / `{metadata.get('python_version')}`",
        f"- Warmup/timed iterations: `{metadata['config']['warmup_iters']}` / `{metadata['config']['timed_iters']}`",
        f"- Dtype: `{metadata['config']['dtype']}`",
        f"- Model: `{model['model_name']}`",
        f"- Model revision / commit hash: `{model['model_revision']}` / `{model['model_commit_hash']}`",
        f"- Model type: `{model['model_type']}`",
        (
            f"- Qwen attention shape: hidden_size=`{model['hidden_size']}`, "
            f"num_attention_heads=`{model['num_attention_heads']}`, "
            f"num_key_value_heads=`{model['num_key_value_heads']}`, head_dim=`{model['head_dim']}`, "
            f"attention_backend=`{model['attention_backend']}`"
        ),
        (
            f"- Qwen MoE shape: num_experts=`{model['num_experts']}`, "
            f"num_experts_per_tok=`{model['num_experts_per_tok']}`, "
            f"moe_intermediate_size=`{model['moe_intermediate_size']}`"
        ),
        f"- CSV artifact: `{csv_path.as_posix()}`",
        f"- JSON artifact: `{json_path.as_posix()}`",
        f"- Status counts: `{status_counts}`",
        "",
        "The issue text mentioned 10 combinations, but the enumerated grid is 6 sequence lengths x 2 batch "
        "sizes = 12 cells. This report accounts for all 12 cells for each module and execution mode.",
        "",
        "This corrected run supersedes the prior canonical artifact set that used placeholder synthetic "
        "dimensions. The module sizes and artifact metadata now come from "
        f"`transformers.AutoConfig.from_pretrained(\"{model['model_name']}\")`; only random module weights and "
        "synthetic inputs are used, so checkpoint shards are not downloaded.",
        "",
        "The runs are isolated attention-only or isolated MoE-only synthetic module runs. No overlapped "
        "Attention+MoE measurement is included.",
        "",
    ]

    for module_name in VALID_MODULES:
        for mode in VALID_MODES:
            section_rows = [
                row for row in rows if row["module"] == module_name and row["mode"] == mode
            ]
            lines.extend(
                [
                    f"## {module_name.title()} {mode.replace('_', '+')}",
                    "",
                    "| seq_len | batch | status | mean ms | tokens/s | max allocated | max reserved | notes |",
                    "| ---: | ---: | --- | ---: | ---: | --- | --- | --- |",
                ]
            )
            for row in sorted(section_rows, key=lambda item: (int(item["seq_len"]), int(item["batch_size"]))):
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            str(row["seq_len"]),
                            str(row["batch_size"]),
                            markdown_cell(row["status"]),
                            format_float(row.get("elapsed_ms_mean")),
                            format_float(row.get("tokens_per_second"), digits=1),
                            markdown_cell(format_memory(row, "max_memory_allocated")),
                            markdown_cell(format_memory(row, "max_memory_reserved")),
                            markdown_cell(row.get("status_reason")),
                        ]
                    )
                    + " |"
                )
            lines.append("")

    skipped_or_failed = [row for row in rows if row.get("status") != "ok"]
    lines.extend(["## OOM And Skipped Cells", ""])
    if not skipped_or_failed:
        lines.append("All cells completed with `status=ok`.")
    else:
        lines.extend(
            [
                "| module | mode | seq_len | batch | status | reason |",
                "| --- | --- | ---: | ---: | --- | --- |",
            ]
        )
        for row in skipped_or_failed:
            lines.append(
                "| "
                + " | ".join(
                    [
                        markdown_cell(row["module"]),
                        markdown_cell(row["mode"]),
                        str(row["seq_len"]),
                        str(row["batch_size"]),
                        markdown_cell(row["status"]),
                        markdown_cell(row.get("status_reason")),
                    ]
                )
                + " |"
            )
    lines.append("")
    return "\n".join(lines)


def write_results_markdown(
    path: Path,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    csv_path: Path,
    json_path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_results_markdown(metadata=metadata, rows=rows, csv_path=csv_path, json_path=json_path),
        encoding="utf-8",
    )


def default_results_path(output_dir: Path) -> Path:
    if output_dir.name == "verification":
        return output_dir.parent / "results.md"
    return output_dir / "results.md"


def build_config(args: argparse.Namespace) -> BenchmarkConfig:
    dtype_name = normalize_dtype_name(args.dtype)
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")
    if args.timed_iters <= 0:
        raise ValueError("--timed-iters must be > 0")
    model = resolve_model_config(model_name=args.model_name, model_revision=args.model_revision)
    return BenchmarkConfig(
        modules=parse_modules(args.modules),
        modes=parse_modes(args.modes),
        seq_lens=parse_int_csv(args.seq_lens, field_name="seq-lens"),
        batch_sizes=parse_int_csv(args.batch_sizes, field_name="batch-sizes"),
        dtype_name=dtype_name,
        device=args.device,
        warmup_iters=int(args.warmup_iters),
        timed_iters=int(args.timed_iters),
        seed=int(args.seed),
        model=model,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure isolated attention/MoE CUDA timing and memory.")
    parser.add_argument("--modules", default="all", help="attention,moe or all")
    parser.add_argument("--modes", default="all", help="forward,forward_backward or all")
    parser.add_argument("--seq-lens", default=",".join(str(value) for value in DEFAULT_SEQ_LENS))
    parser.add_argument("--batch-sizes", default=",".join(str(value) for value in DEFAULT_BATCH_SIZES))
    parser.add_argument("--dtype", default="bf16", choices=sorted(DTYPE_ALIASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--timed-iters", type=int, default=5)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--results-md", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.csv_path) if args.csv_path else output_dir / "step8_attn_moe_memory_sweep.csv"
    json_path = Path(args.json_path) if args.json_path else output_dir / "step8_attn_moe_memory_sweep.json"
    results_path = Path(args.results_md) if args.results_md else default_results_path(output_dir)

    metadata = build_metadata(config, sys.argv[1:] if argv is None else argv)
    rows = run_sweep(config, metadata)
    metadata["completed_at_utc"] = now_utc_iso()
    write_csv(csv_path, rows)
    write_json(json_path, metadata, rows)
    write_results_markdown(results_path, metadata, rows, csv_path, json_path)

    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote JSON: {json_path}")
    print(f"Wrote results: {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
