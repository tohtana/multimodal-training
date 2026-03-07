"""Megatron single-layer runtime helpers for attention/MoE overlap experiments."""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Any
from threading import BrokenBarrierError

import torch

try:
    from examples.attn_moe_overlap.megatron_overlap_schema import (
        normalize_dtype_name,
        tensor_signature,
    )
except ModuleNotFoundError:
    from megatron_overlap_schema import (  # type: ignore[no-redef]
        normalize_dtype_name,
        tensor_signature,
    )


def _trainer_dtype_name(dtype: str) -> str:
    normalized = normalize_dtype_name(dtype)
    if normalized == "fp32":
        return "float32"
    if normalized == "bf16":
        return "bfloat16"
    if normalized == "fp16":
        return "float16"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _torch_dtype(dtype: str) -> torch.dtype:
    normalized = normalize_dtype_name(dtype)
    if normalized == "fp32":
        return torch.float32
    if normalized == "bf16":
        return torch.bfloat16
    if normalized == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {dtype}")


@dataclass
class RuntimeConfig:
    model_name: str
    model_type: str
    stage_role: str
    dtype: str
    seq_len: int
    batch_size: int
    seed: int
    expert_model_parallel_size: int
    num_experts: int | None = None


class _SingleLayerMegatronTrainer:
    """Thin wrapper over MegatronBaseTrainer for non-Ray local process usage."""

    def __init__(self, config: dict[str, Any], rank: int):
        from python.ray.megatron_trainer import MegatronBaseTrainer

        class _Impl(MegatronBaseTrainer):
            def build_model(self):
                self._build_megatron_model()
                self.megatron_model.eval()

        self._impl = _Impl(config, rank)

    @property
    def megatron_model(self):
        return self._impl.megatron_model

    def build_model(self) -> None:
        self._impl.build_model()


class MegatronSingleLayerRuntime:
    """Runs one Megatron decoder layer split into attention and MoE-stage paths."""

    def __init__(self, config: RuntimeConfig):
        if config.stage_role not in {"attn", "moe"}:
            raise ValueError(f"Unsupported stage_role: {config.stage_role}")
        self.config = config
        self.device = torch.device("cuda:0")
        self.trainer: _SingleLayerMegatronTrainer | None = None
        self.layer = None
        self.hidden_states: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None

    def initialize(self) -> None:
        torch.cuda.set_device(self.device)
        rank = int(os.environ.get("RANK", "0"))
        trainer_config = self._build_trainer_config()
        self.trainer = _SingleLayerMegatronTrainer(trainer_config, rank)
        self.trainer.build_model()
        self.layer = self._resolve_decoder_layer(self.trainer.megatron_model)
        self.layer.eval()
        hidden_size = int(self.layer.config.hidden_size)
        self.hidden_states = self._build_hidden_states(hidden_size)
        self.attention_mask = self._build_attention_mask()

    def _build_trainer_config(self) -> dict[str, Any]:
        engine_config: dict[str, Any] = {
            "tensor_parallel_size": 1,
            "sequence_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": int(self.config.expert_model_parallel_size),
            "attention_backend": "unfused",
            "megatron_num_layers": 1,
            "load_weights": False,
            "use_cpu_initialization": True,
        }
        if self.config.num_experts is not None:
            engine_config["num_experts"] = int(self.config.num_experts)

        return {
            "model_name": self.config.model_name,
            "model_type": self.config.model_type,
            "engine": "megatron",
            "engine_config": engine_config,
            "parallelism": "tensor",
            "dtype": _trainer_dtype_name(self.config.dtype),
            "attention_backend": "sdpa",
            "activation_checkpointing": False,
            "autocast": False,
            "seed": int(self.config.seed),
            "dp_size": 1,
            "parallel_size": 1,
            "text_seq_len": int(self.config.seq_len),
        }

    def _resolve_decoder_layer(self, megatron_model):
        model = megatron_model
        if hasattr(model, "language_model"):
            model = model.language_model
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            raise RuntimeError("Megatron model does not expose decoder")
        layers = getattr(decoder, "layers", None)
        if layers is None or len(layers) == 0:
            raise RuntimeError("Megatron decoder has no layers")
        return layers[0]

    def _build_hidden_states(self, hidden_size: int) -> torch.Tensor:
        generator = torch.Generator(device=self.device)
        stage_seed_offset = 101 if self.config.stage_role == "attn" else 202
        generator.manual_seed(int(self.config.seed) + stage_seed_offset)
        base = torch.randn(
            int(self.config.seq_len),
            int(self.config.batch_size),
            hidden_size,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        return base.to(_torch_dtype(self.config.dtype))

    def _build_attention_mask(self) -> torch.Tensor:
        seq_len = int(self.config.seq_len)
        batch = int(self.config.batch_size)
        mask = ~torch.tril(torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool))
        return mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)

    def run_stage(
        self,
        *,
        warmup_iters: int,
        timed_iters: int,
        timed_start_barrier: Any | None = None,
    ) -> dict[str, Any]:
        if self.layer is None or self.hidden_states is None or self.attention_mask is None:
            self.initialize()

        assert self.layer is not None
        assert self.hidden_states is not None
        assert self.attention_mask is not None

        total_iters = int(warmup_iters) + int(timed_iters)
        if total_iters <= 0:
            raise ValueError("warmup_iters + timed_iters must be > 0")

        cuda_ms: list[float] = []
        step_ms: list[float] = []
        enqueue_windows: list[tuple[float, float]] = []
        timed_start_s: float | None = None
        timed_end_s: float | None = None
        first_nonfinite: dict[str, Any] | None = None
        output_tensor: torch.Tensor | None = None

        import torch.distributed as dist

        for iter_idx in range(total_iters):
            if iter_idx == warmup_iters and timed_start_barrier is not None:
                try:
                    timed_start_barrier.wait()
                except BrokenBarrierError as exc:
                    raise RuntimeError("Timed-start barrier broke before timed iterations") from exc

            if dist.is_available() and dist.is_initialized():
                dist.barrier()

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            step_start = time.perf_counter()
            start_event.record()
            enqueue_start = time.perf_counter()
            with torch.no_grad():
                if self.config.stage_role == "attn":
                    output_tensor, _ = self.layer._forward_attention(
                        self.hidden_states,
                        attention_mask=self.attention_mask,
                    )
                else:
                    output_tensor = self.layer._forward_mlp(self.hidden_states, inference_context=None)
            enqueue_end = time.perf_counter()
            end_event.record()
            torch.cuda.synchronize(self.device)
            step_end = time.perf_counter()

            if iter_idx >= warmup_iters:
                if timed_start_s is None:
                    timed_start_s = step_start
                cuda_ms.append(float(start_event.elapsed_time(end_event)))
                step_ms.append((step_end - step_start) * 1000.0)
                enqueue_windows.append((enqueue_start, enqueue_end))
                timed_end_s = step_end

            if first_nonfinite is None and output_tensor is not None and torch.is_floating_point(output_tensor):
                if not torch.isfinite(output_tensor).all():
                    first_nonfinite = {
                        "module": f"{self.config.stage_role}_layer",
                        "phase": "forward",
                        "tensor": "output",
                        "iter": int(iter_idx),
                    }

        mean_cuda_ms = float(sum(cuda_ms) / len(cuda_ms)) if cuda_ms else None
        mean_step_ms = float(sum(step_ms) / len(step_ms)) if step_ms else None
        timed_wall_ms = (
            float((timed_end_s - timed_start_s) * 1000.0)
            if timed_start_s is not None and timed_end_s is not None
            else None
        )
        return {
            "status": "ok",
            "stage_role": self.config.stage_role,
            "timing_ms": {
                "cuda": mean_cuda_ms,
                "step_total": mean_step_ms,
                "timed_wall": timed_wall_ms,
            },
            "timed_window_s": {
                "start_s": timed_start_s,
                "end_s": timed_end_s,
                "duration_ms": timed_wall_ms,
            },
            "enqueue_windows": enqueue_windows,
            "finite": {
                "all_finite": first_nonfinite is None,
                "first_nonfinite": first_nonfinite,
            },
            "output_signature": tensor_signature(output_tensor),
        }


def cleanup_distributed_state() -> None:
    """Best-effort cleanup for model-parallel + process-group state."""
    try:
        from megatron.core import parallel_state

        parallel_state.destroy_model_parallel()
    except Exception:
        pass
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass


def classify_exception(exc: BaseException) -> tuple[str, dict[str, Any]]:
    message = str(exc)
    lower = message.lower()
    if "out of memory" in lower:
        status = "oom"
    else:
        status = "runtime_error"
    return (
        status,
        {
            "code": status,
            "message": message,
            "traceback": traceback.format_exc(),
        },
    )
