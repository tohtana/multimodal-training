"""Megatron single-layer runtime helpers for attention/MoE overlap experiments."""

from __future__ import annotations

import os
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from types import MethodType
from typing import Any, Callable
from threading import BrokenBarrierError

import torch

try:
    from examples.attn_moe_overlap.green_context_utils import (
        GreenContextError,
        GreenContextStreamOwner,
        create_green_context_stream,
    )
    from examples.attn_moe_overlap.megatron_overlap_schema import (
        normalize_dtype_name,
        tensor_signature,
    )
except ModuleNotFoundError:
    from green_context_utils import (  # type: ignore[no-redef]
        GreenContextError,
        GreenContextStreamOwner,
        create_green_context_stream,
    )
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


def _resolve_profiler_schedule(
    *,
    warmup_iters: int,
    timed_iters: int,
    profiler_wait_iters: int | None,
    profiler_active_timed_iters: int | None,
) -> tuple[int, int]:
    wait_iters = int(
        warmup_iters if profiler_wait_iters is None else profiler_wait_iters
    )
    if wait_iters < 0:
        raise ValueError("profiler_wait_iters must be >= 0")

    active_timed_iters = int(profiler_active_timed_iters or timed_iters)
    active_timed_iters = max(1, min(int(timed_iters), active_timed_iters))
    return wait_iters, active_timed_iters


def _timed_window_from_bounds(
    start_s: float | None, end_s: float | None
) -> dict[str, float | None]:
    if start_s is None or end_s is None:
        return {"start_s": None, "end_s": None, "duration_ms": None}
    return {
        "start_s": start_s,
        "end_s": end_s,
        "duration_ms": max(0.0, (end_s - start_s) * 1000.0),
    }


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name.lower()
    return str(value)


MEGATRON_BLOCKS_BY_ROLE = {
    "attn": ("A0_qkv", "A1_rotary", "A2_core_attention", "A3_output_projection"),
    "moe": (
        "M0_router",
        "M1_dispatch_preprocess",
        "M2_token_dispatch",
        "M3_dispatch_postprocess",
        "M4_experts",
        "M5_combine_preprocess",
        "M6_token_combine",
    ),
}


def describe_attention_runtime(layer: Any, requested_backend: str) -> dict[str, Any]:
    config = getattr(layer, "config", None)
    self_attention = getattr(layer, "self_attention", None)
    core_attention = (
        getattr(self_attention, "core_attention", None)
        if self_attention is not None
        else None
    )
    return {
        "requested_backend": str(requested_backend),
        "config_attention_backend": _enum_name(
            getattr(config, "attention_backend", None)
        ),
        "transformer_impl": getattr(config, "transformer_impl", None),
        "layer_class": type(layer).__name__,
        "self_attention_class": type(self_attention).__name__
        if self_attention is not None
        else None,
        "core_attention_class": type(core_attention).__name__
        if core_attention is not None
        else None,
        "nvte_backend_flags": {
            "flash": os.getenv("NVTE_FLASH_ATTN"),
            "fused": os.getenv("NVTE_FUSED_ATTN"),
            "unfused": os.getenv("NVTE_UNFUSED_ATTN"),
        },
    }


class ScheduleAborted(RuntimeError):
    """Raised when a peer failure aborts the shared launch schedule."""


def _run_iteration_schedule(
    *,
    execution_schedule: str,
    stage_role: str,
    stage_label: str | None = None,
    serial_phase_order: tuple[str, ...] | None = None,
    total_iters: int,
    warmup_iters: int,
    barrier_wait: Callable[[str, int], None],
    run_forward: Callable[[str, int, bool], None],
    profiler_step: Callable[[int], None] | None = None,
    now: Callable[[], float] = time.perf_counter,
) -> dict[str, dict[str, float | None]]:
    if execution_schedule not in {"overlap", "serial_lockstep"}:
        raise ValueError(f"Unsupported execution_schedule: {execution_schedule}")
    if stage_role not in {"attn", "moe"}:
        raise ValueError(f"Unsupported stage_role: {stage_role}")
    active_stage_label = str(stage_label or stage_role)
    default_phase_order = (
        ("attn", "moe")
        if execution_schedule == "serial_lockstep"
        else (active_stage_label,)
    )
    phase_order = tuple(serial_phase_order or default_phase_order)
    if execution_schedule == "serial_lockstep":
        if not phase_order:
            raise ValueError("serial_phase_order must not be empty for serial_lockstep")
        if active_stage_label not in phase_order:
            raise ValueError(
                f"serial_phase_order must contain the active stage label {active_stage_label!r}"
            )

    stage_timed_start_s: float | None = None
    stage_timed_end_s: float | None = None
    schedule_timed_start_s: float | None = None
    schedule_timed_end_s: float | None = None

    def _serial_barrier_name(phase_index: int, phase_count: int) -> str:
        if phase_index >= phase_count - 1:
            return "serial_end"
        if phase_count == 2 and phase_index == 0:
            return "serial_between"
        return f"serial_between_{phase_index}"

    def _run_active_phase(
        phase_label: str, phase_name: str, iter_idx: int, is_timed: bool
    ) -> None:
        nonlocal stage_timed_start_s, stage_timed_end_s
        if active_stage_label != phase_label:
            return
        phase_start_s = now()
        run_forward(phase_name, iter_idx, is_timed)
        phase_end_s = now()
        if is_timed:
            if stage_timed_start_s is None:
                stage_timed_start_s = phase_start_s
            stage_timed_end_s = phase_end_s

    for iter_idx in range(int(total_iters)):
        is_timed = iter_idx >= int(warmup_iters)
        if is_timed and schedule_timed_start_s is None:
            schedule_timed_start_s = now()

        if execution_schedule == "overlap":
            barrier_wait("overlap_start", iter_idx)
            _run_active_phase(
                active_stage_label, active_stage_label, iter_idx, is_timed
            )
            barrier_wait("overlap_end", iter_idx)
        else:
            barrier_wait("serial_start", iter_idx)
            for phase_index, phase_label in enumerate(phase_order):
                _run_active_phase(phase_label, phase_label, iter_idx, is_timed)
                barrier_wait(
                    _serial_barrier_name(phase_index, len(phase_order)), iter_idx
                )

        if is_timed:
            schedule_timed_end_s = now()
        if profiler_step is not None:
            profiler_step(iter_idx)

    return {
        "stage_timed_window_s": _timed_window_from_bounds(
            stage_timed_start_s, stage_timed_end_s
        ),
        "schedule_timed_window_s": _timed_window_from_bounds(
            schedule_timed_start_s, schedule_timed_end_s
        ),
    }


@dataclass
class RuntimeConfig:
    model_name: str
    model_type: str
    stage_role: str
    runtime_backend: str
    attention_backend: str
    moe_grouped_gemm: bool
    moe_token_dispatcher_type: str
    overlap_moe_expert_parallel_comm: bool
    dtype: str
    seq_len: int
    batch_size: int
    seed: int
    expert_model_parallel_size: int
    stage_label: str | None = None
    pair_index: int = 0
    green_ctx_attn_sms: int | None = None
    green_ctx_moe_sms: int | None = None
    num_experts: int | None = None
    moe_routing_mode: str = "normal"
    torch_compile_enabled: bool = False


@dataclass
class EqualTokenRoutingState:
    assigned_expert_ids: torch.Tensor
    probs: torch.Tensor
    routing_map: torch.Tensor
    tokens_per_expert: list[int]
    local_tokens_per_expert: list[int]


def _build_equal_token_routing_state(
    *,
    hidden_states: torch.Tensor,
    num_experts: int,
    top_k: int,
    local_expert_indices: list[int],
) -> EqualTokenRoutingState:
    if hidden_states.ndim < 2:
        raise RuntimeError(
            "equal_tokens requires hidden_states with an explicit hidden dimension"
        )
    if int(num_experts) <= 0:
        raise RuntimeError("equal_tokens requires num_experts > 0")
    if int(top_k) <= 0:
        raise RuntimeError("equal_tokens requires top_k > 0")
    if int(top_k) > int(num_experts):
        raise RuntimeError("equal_tokens requires top_k <= num_experts")
    num_tokens = int(hidden_states.numel() // hidden_states.shape[-1])
    device = hidden_states.device
    token_indices = torch.arange(num_tokens, device=device, dtype=torch.long).unsqueeze(
        1
    )
    expert_offsets = torch.arange(
        int(top_k), device=device, dtype=torch.long
    ).unsqueeze(0)
    assigned_expert_ids = (token_indices + expert_offsets) % int(num_experts)
    routing_map = torch.zeros(
        (num_tokens, int(num_experts)), device=device, dtype=torch.bool
    )
    routing_map.scatter_(1, assigned_expert_ids, True)
    probs = torch.zeros(
        (num_tokens, int(num_experts)), device=device, dtype=hidden_states.dtype
    )
    probs.scatter_(1, assigned_expert_ids, 1.0 / float(top_k))
    global_counts = torch.bincount(
        assigned_expert_ids.reshape(-1), minlength=int(num_experts)
    ).to(dtype=torch.int64)
    local_counts = torch.zeros_like(global_counts)
    if local_expert_indices:
        local_idx = torch.tensor(local_expert_indices, device=device, dtype=torch.long)
        local_counts[local_idx] = global_counts[local_idx]
    return EqualTokenRoutingState(
        assigned_expert_ids=assigned_expert_ids,
        probs=probs,
        routing_map=routing_map,
        tokens_per_expert=[int(value) for value in global_counts.cpu().tolist()],
        local_tokens_per_expert=[int(value) for value in local_counts.cpu().tolist()],
    )


class TorchCompileFailure(RuntimeError):
    """Raised when torch.compile preparation or preflight fails."""


@dataclass
class PreparedStageCallable:
    run: Callable[[], torch.Tensor]
    compile_payload: dict[str, str]
    consume_equal_token_routing_state: Callable[[], EqualTokenRoutingState | None]
    cleanup: Callable[[], None]


def _noop_cleanup() -> None:
    return None


def _chain_cleanups(*cleanups: Callable[[], None]) -> Callable[[], None]:
    active_cleanups = [cleanup for cleanup in cleanups if cleanup is not _noop_cleanup]
    if not active_cleanups:
        return _noop_cleanup

    def _restore() -> None:
        for cleanup in reversed(active_cleanups):
            cleanup()

    return _restore


def _install_torch_compile_safe_global(
    method: Any,
    global_name: str,
    marker_name: str,
) -> Callable[[], None]:
    method_globals = getattr(getattr(method, "__func__", None), "__globals__", None)
    disable_driver = getattr(getattr(torch, "_dynamo", None), "disable", None)
    if not isinstance(method_globals, dict) or disable_driver is None:
        return _noop_cleanup

    original = method_globals.get(global_name)
    if not callable(original) or getattr(original, marker_name, False):
        return _noop_cleanup

    compile_safe_wrapper = disable_driver(original)
    setattr(compile_safe_wrapper, marker_name, True)
    method_globals[global_name] = compile_safe_wrapper

    def _restore() -> None:
        if method_globals.get(global_name) is compile_safe_wrapper:
            method_globals[global_name] = original

    return _restore


def _install_torch_compile_safe_method(
    target: Any,
    method_name: str,
    marker_name: str,
) -> Callable[[], None]:
    disable_driver = getattr(getattr(torch, "_dynamo", None), "disable", None)
    bound_method = getattr(target, method_name, None)
    method = getattr(bound_method, "__func__", None)
    if not callable(method) or disable_driver is None or getattr(method, marker_name, False):
        return _noop_cleanup

    target_dict = getattr(target, "__dict__", None)
    had_instance_attr = isinstance(target_dict, dict) and method_name in target_dict
    original_instance_attr = target_dict.get(method_name) if had_instance_attr else None

    compile_safe_method = disable_driver(method)
    setattr(compile_safe_method, marker_name, True)
    setattr(target, method_name, MethodType(compile_safe_method, target))

    def _restore() -> None:
        current = getattr(target, method_name, None)
        if getattr(current, "__func__", None) is not compile_safe_method:
            return
        if had_instance_attr and isinstance(target_dict, dict):
            target_dict[method_name] = original_instance_attr
            return
        try:
            delattr(target, method_name)
        except AttributeError:
            pass

    return _restore


def _install_torch_compile_safe_moe_cpu_handoff(token_dispatcher: Any) -> Callable[[], None]:
    return _chain_cleanups(
        _install_torch_compile_safe_global(
            getattr(token_dispatcher, "_maybe_dtoh_and_synchronize", None),
            "maybe_move_tensor_to_cpu",
            "__codex_compile_safe_moe_cpu_handoff__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "dispatch_preprocess",
            "__codex_compile_safe_dispatch_preprocess__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "token_dispatch",
            "__codex_compile_safe_token_dispatch__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "dispatch_postprocess",
            "__codex_compile_safe_dispatch_postprocess__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "combine_preprocess",
            "__codex_compile_safe_combine_preprocess__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "token_combine",
            "__codex_compile_safe_token_combine__",
        ),
        _install_torch_compile_safe_method(
            token_dispatcher,
            "combine_postprocess",
            "__codex_compile_safe_combine_postprocess__",
        ),
    )


def _torch_compile_requested(torch_compile_enabled: bool) -> str:
    return "on" if torch_compile_enabled else "off"


def _prepare_stage_callable(
    *,
    config: RuntimeConfig,
    layer: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    compile_fn: Callable[[Callable[..., torch.Tensor]], Callable[..., torch.Tensor]] | None = None,
) -> PreparedStageCallable:
    stage_state: dict[str, EqualTokenRoutingState | None] = {"equal_token_routing_state": None}
    cleanup = _noop_cleanup

    def _consume_equal_token_routing_state() -> EqualTokenRoutingState | None:
        state = stage_state["equal_token_routing_state"]
        stage_state["equal_token_routing_state"] = None
        return state

    try:
        if config.stage_role == "attn":
            def stage_impl(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                stage_state["equal_token_routing_state"] = None
                output_tensor, _ = layer._forward_attention(
                    hidden_states, attention_mask=attention_mask
                )
                return output_tensor

            eager_runner = lambda: stage_impl(hidden_states, attention_mask)
        elif config.stage_role == "moe":
            mlp = layer.mlp
            router = mlp.router
            token_dispatcher = mlp.token_dispatcher
            if config.torch_compile_enabled:
                cleanup = _install_torch_compile_safe_moe_cpu_handoff(token_dispatcher)

            def stage_impl(hidden_states: torch.Tensor) -> torch.Tensor:
                return layer._forward_mlp(hidden_states, inference_context=None)

            equal_token_state: EqualTokenRoutingState | None = None
            if config.moe_routing_mode == "equal_tokens":
                local_expert_indices = list(getattr(token_dispatcher, "local_expert_indices"))
                equal_token_state = _build_equal_token_routing_state(
                    hidden_states=hidden_states,
                    num_experts=int(config.num_experts or 0),
                    top_k=int(getattr(layer.config, "moe_router_topk", 0) or 0),
                    local_expert_indices=local_expert_indices,
                )

            def _run_moe_impl(stage_callable: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
                stage_state["equal_token_routing_state"] = equal_token_state
                if equal_token_state is None:
                    return stage_callable(hidden_states)

                original_forward = router.forward

                def _equal_tokens_forward(_router_self: Any, input_tensor: torch.Tensor):
                    del input_tensor
                    return equal_token_state.probs, equal_token_state.routing_map

                router.forward = MethodType(_equal_tokens_forward, router)
                try:
                    return stage_callable(hidden_states)
                finally:
                    router.forward = original_forward

            eager_runner = lambda: _run_moe_impl(stage_impl)
        else:
            raise ValueError(f"Unsupported stage_role: {config.stage_role}")
    except Exception:
        cleanup()
        raise

    if not config.torch_compile_enabled:
        return PreparedStageCallable(
            run=eager_runner,
            compile_payload={"requested": _torch_compile_requested(False), "status": "eager"},
            consume_equal_token_routing_state=_consume_equal_token_routing_state,
            cleanup=cleanup,
        )

    compile_driver = compile_fn if compile_fn is not None else getattr(torch, "compile", None)
    if compile_driver is None:
        cleanup()
        raise TorchCompileFailure("torch.compile is unavailable in this environment")

    try:
        compiled_impl = compile_driver(stage_impl)
    except Exception as exc:
        cleanup()
        raise TorchCompileFailure(f"torch.compile failed during stage preparation: {exc}") from exc

    if config.stage_role == "attn":
        runner = lambda: compiled_impl(hidden_states, attention_mask)
    else:
        runner = lambda: _run_moe_impl(compiled_impl)

    return PreparedStageCallable(
        run=runner,
        compile_payload={"requested": _torch_compile_requested(True), "status": "compiled"},
        consume_equal_token_routing_state=_consume_equal_token_routing_state,
        cleanup=cleanup,
    )


class _SingleLayerMegatronTrainer:
    """Thin wrapper over MegatronBaseTrainer for non-Ray local process usage."""

    def __init__(self, config: dict[str, Any], rank: int):
        from python.ray.megatron_trainer import MegatronBaseTrainer

        class _Impl(MegatronBaseTrainer):
            def _get_device(self):
                local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                device = torch.device(f"cuda:{local_rank}")
                torch.cuda.set_device(device)
                return device

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
        self.device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
        self.trainer: _SingleLayerMegatronTrainer | None = None
        self.layer = None
        self.hidden_states: torch.Tensor | None = None
        self.attention_mask: torch.Tensor | None = None
        self.attention_runtime: dict[str, Any] | None = None
        self.execution_stream: torch.cuda.Stream | None = None
        self.green_ctx_stream_owner: GreenContextStreamOwner | None = None
        self.runtime_metadata = {
            "requested_sms": None,
            "granted_sms": None,
            "device_total_sms": None,
        }
        self._stage_runner: Callable[[], torch.Tensor] | None = None
        self._consume_equal_token_routing_state: Callable[[], EqualTokenRoutingState | None] = lambda: None
        self._stage_cleanup: Callable[[], None] = _noop_cleanup
        self._torch_compile_status = "eager"
        self._torch_compile_preflight_done = False

    def initialize(self) -> None:
        torch.cuda.set_device(self.device)
        rank = int(os.environ.get("RANK", "0"))
        trainer_config = self._build_trainer_config()
        self.trainer = _SingleLayerMegatronTrainer(trainer_config, rank)
        self.trainer.build_model()
        self.layer = self._resolve_decoder_layer(self.trainer.megatron_model)
        self.layer.eval()
        self._validate_moe_routing_mode_support()
        self.attention_runtime = describe_attention_runtime(
            self.layer, self.config.attention_backend
        )
        hidden_size = int(self.layer.config.hidden_size)
        self.hidden_states = self._build_hidden_states(hidden_size)
        self.attention_mask = self._build_attention_mask()
        self._initialize_execution_stream()
        self._prepare_stage_runner(requires_grad=False)

    def _prepare_stage_runner(self, *, requires_grad: bool) -> None:
        if self.layer is None or self.hidden_states is None or self.attention_mask is None:
            raise RuntimeError("Cannot prepare stage runner before layer/input initialization")

        self._stage_cleanup()
        self._stage_cleanup = _noop_cleanup
        self.hidden_states = self.hidden_states.detach()
        if requires_grad:
            self.hidden_states.requires_grad_(True)
        try:
            prepared_stage = _prepare_stage_callable(
                config=self.config,
                layer=self.layer,
                hidden_states=self.hidden_states,
                attention_mask=self.attention_mask,
            )
        except TorchCompileFailure:
            self._torch_compile_status = "compile_failed"
            raise
        self._stage_runner = prepared_stage.run
        self._consume_equal_token_routing_state = prepared_stage.consume_equal_token_routing_state
        self._stage_cleanup = prepared_stage.cleanup
        self._torch_compile_status = str(prepared_stage.compile_payload["status"])

    def _clear_gradients(self) -> None:
        if self.layer is None:
            return
        for parameter in self.layer.parameters():
            parameter.grad = None
        if self.hidden_states is not None:
            self.hidden_states.grad = None

    def describe_layer_mapping(self) -> dict[str, Any]:
        if self.layer is None:
            self.initialize()
        assert self.layer is not None

        def _children(module: Any) -> list[dict[str, str]]:
            if module is None or not hasattr(module, "named_children"):
                return []
            return [
                {"name": name, "class": type(child).__name__}
                for name, child in module.named_children()
            ]

        layer = self.layer
        self_attention = getattr(layer, "self_attention", None)
        mlp = getattr(layer, "mlp", None)
        return {
            "stage_role": self.config.stage_role,
            "measured_blocks": list(MEGATRON_BLOCKS_BY_ROLE[self.config.stage_role]),
            "layer_class": type(layer).__name__,
            "stage_method": (
                "layer._forward_attention"
                if self.config.stage_role == "attn"
                else "layer._forward_mlp"
            ),
            "self_attention": {
                "class": type(self_attention).__name__ if self_attention is not None else None,
                "module": type(self_attention).__module__ if self_attention is not None else None,
                "children": _children(self_attention),
                "core_attention_class": (
                    type(getattr(self_attention, "core_attention", None)).__name__
                    if getattr(self_attention, "core_attention", None) is not None
                    else None
                ),
                "core_attention_module": (
                    type(getattr(self_attention, "core_attention", None)).__module__
                    if getattr(self_attention, "core_attention", None) is not None
                    else None
                ),
            },
            "mlp": {
                "class": type(mlp).__name__ if mlp is not None else None,
                "module": type(mlp).__module__ if mlp is not None else None,
                "children": _children(mlp),
                "public_methods": [
                    name
                    for name in (
                        "router_and_preprocess",
                        "dispatch",
                        "routed_experts_compute",
                        "combine",
                        "shared_experts_compute",
                    )
                    if callable(getattr(mlp, name, None))
                ],
            },
            "block_mapping": {
                "A0_qkv": "Attention.forward get_query_key_value_tensors(...) including projection/split setup",
                "A1_rotary": "Attention.forward rotary embedding application block",
                "A2_core_attention": "Attention.forward self.core_attention(...) or checkpointed core attention",
                "A3_output_projection": "Attention.forward self.linear_proj(core_attn_out)",
                "M0_router": "MoELayer.router_and_preprocess self.router(hidden_states)",
                "M1_dispatch_preprocess": "MoELayer.router_and_preprocess token_dispatcher.dispatch_preprocess(...)",
                "M2_token_dispatch": "MoELayer.dispatch token_dispatcher.token_dispatch(...)",
                "M3_dispatch_postprocess": "MoELayer.routed_experts_compute token_dispatcher.dispatch_postprocess(...)",
                "M4_experts": "MoELayer.routed_experts_compute self.experts(...)",
                "M5_combine_preprocess": "MoELayer.routed_experts_compute token_dispatcher.combine_preprocess(...)",
                "M6_token_combine": "MoELayer.combine token_combine(...) plus combine_postprocess(...)",
            },
            "block_mapping_note": (
                "Fine blocks are instrumented inside Megatron-LM internals with "
                "megatron.core.transformer.profiling.profile_block. The bridge benchmark "
                "runs only MegatronSingleLayerRuntime stage paths and reads the helper stats."
            ),
        }

    def run_stage_measurement(
        self,
        *,
        warmup_iters: int,
        timed_iters: int,
        include_backward: bool,
        enable_cuda_profiler: bool = False,
    ) -> dict[str, Any]:
        if self.layer is None or self.hidden_states is None or self.attention_mask is None:
            self.initialize()

        assert self.layer is not None
        assert self.hidden_states is not None
        assert self.execution_stream is not None
        assert self._stage_runner is not None

        total_iters = int(warmup_iters) + int(timed_iters)
        if total_iters <= 0:
            raise ValueError("warmup_iters + timed_iters must be > 0")
        if int(timed_iters) <= 0:
            raise ValueError("timed_iters must be > 0")

        self._prepare_stage_runner(requires_grad=include_backward)
        assert self._stage_runner is not None
        self._run_torch_compile_preflight()

        import torch.distributed as dist
        from megatron.core.transformer.profiling import (
            get_block_stats,
            reset_block_stats,
            set_block_profiling,
        )

        forward_cuda_ms: list[float] = []
        total_cuda_ms: list[float] = []
        forward_wall_ms: list[float] = []
        total_wall_ms: list[float] = []
        first_nonfinite: dict[str, Any] | None = None
        output_tensor: torch.Tensor | None = None
        profiler_started = False
        memory_before: dict[str, int] | None = None
        memory_after: dict[str, int] | None = None

        block_stats: dict[str, Any] = {}

        try:
            set_block_profiling(
                True,
                collect_stats=False,
                collect_cuda_timing=True,
                collect_memory=True,
            )
            reset_block_stats()
            for iter_idx in range(total_iters):
                is_timed = iter_idx >= int(warmup_iters)
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()
                if is_timed and not forward_cuda_ms:
                    self.execution_stream.synchronize()
                    reset_block_stats()
                    torch.cuda.reset_peak_memory_stats(self.device)
                    memory_before = {
                        "allocated": int(torch.cuda.memory_allocated(self.device)),
                        "reserved": int(torch.cuda.memory_reserved(self.device)),
                    }
                    if enable_cuda_profiler:
                        torch.cuda.cudart().cudaProfilerStart()
                        profiler_started = True

                self._clear_gradients()
                forward_start_event = torch.cuda.Event(enable_timing=True)
                forward_end_event = torch.cuda.Event(enable_timing=True)
                total_start_event = torch.cuda.Event(enable_timing=True)
                total_end_event = torch.cuda.Event(enable_timing=True)

                active_stream = self.execution_stream
                total_start_s = time.perf_counter()
                total_start_event.record(active_stream)
                forward_start_event.record(active_stream)
                forward_start_s = time.perf_counter()
                set_block_profiling(
                    True,
                    collect_stats=is_timed and not enable_cuda_profiler,
                    collect_cuda_timing=True,
                    collect_memory=True,
                )
                try:
                    with torch.cuda.stream(active_stream):
                        if include_backward:
                            output_tensor = self._stage_runner()
                        else:
                            with torch.no_grad():
                                output_tensor = self._stage_runner()
                finally:
                    set_block_profiling(False, collect_stats=False)
                forward_end_event.record(active_stream)
                active_stream.synchronize()
                forward_end_s = time.perf_counter()

                if self.config.stage_role == "moe" and self.config.moe_routing_mode == "equal_tokens":
                    state = self._consume_equal_token_routing_state()
                    if state is None:
                        raise RuntimeError("equal_tokens stage runner did not produce routing state")
                else:
                    self._consume_equal_token_routing_state()

                if include_backward:
                    if output_tensor is None:
                        raise RuntimeError("stage runner produced no output tensor")
                    loss = output_tensor.float().square().mean()
                    loss.backward()

                total_end_event.record(active_stream)
                active_stream.synchronize()
                total_end_s = time.perf_counter()

                if is_timed:
                    forward_cuda_ms.append(float(forward_start_event.elapsed_time(forward_end_event)))
                    total_cuda_ms.append(float(total_start_event.elapsed_time(total_end_event)))
                    forward_wall_ms.append((forward_end_s - forward_start_s) * 1000.0)
                    total_wall_ms.append((total_end_s - total_start_s) * 1000.0)

                if (
                    first_nonfinite is None
                    and output_tensor is not None
                    and torch.is_floating_point(output_tensor)
                ):
                    if not torch.isfinite(output_tensor).all():
                        first_nonfinite = {
                            "module": f"{self.config.stage_role}_layer",
                            "phase": "forward",
                            "tensor": "output",
                            "iter": int(iter_idx),
                        }

            if profiler_started:
                torch.cuda.cudart().cudaProfilerStop()
                profiler_started = False
            self.execution_stream.synchronize()
            block_stats = get_block_stats(reset=True)
            memory_after = {
                "allocated": int(torch.cuda.memory_allocated(self.device)),
                "reserved": int(torch.cuda.memory_reserved(self.device)),
                "max_allocated": int(torch.cuda.max_memory_allocated(self.device)),
                "max_reserved": int(torch.cuda.max_memory_reserved(self.device)),
            }
        finally:
            set_block_profiling(False, collect_stats=False)
            if profiler_started:
                torch.cuda.cudart().cudaProfilerStop()
            self._clear_gradients()
            self.cleanup()

        def _mean(values: list[float]) -> float | None:
            return float(sum(values) / len(values)) if values else None

        return {
            "status": "ok",
            "stage_role": self.config.stage_role,
            "blocks": list(MEGATRON_BLOCKS_BY_ROLE[self.config.stage_role]),
            "mode": "forward_backward" if include_backward else "forward",
            "attention_backend": self.config.attention_backend,
            "attention_impl": self.attention_runtime,
            "moe_grouped_gemm": self.config.moe_grouped_gemm,
            "moe_token_dispatcher_type": self.config.moe_token_dispatcher_type,
            "moe_routing_mode": self.config.moe_routing_mode,
            "timing_ms": {
                "forward_cuda_mean": _mean(forward_cuda_ms),
                "forward_wall_mean": _mean(forward_wall_ms),
                "total_cuda_mean": _mean(total_cuda_ms),
                "total_wall_mean": _mean(total_wall_ms),
            },
            "memory": {
                "before": memory_before,
                "after": memory_after,
            },
            "finite": {
                "all_finite": first_nonfinite is None,
                "first_nonfinite": first_nonfinite,
            },
            "output_signature": tensor_signature(output_tensor),
            "block_stats": block_stats,
            "torch_compile": self.torch_compile_payload(),
        }

    def cleanup(self) -> None:
        try:
            self._stage_cleanup()
        finally:
            self._stage_cleanup = _noop_cleanup
            if self.green_ctx_stream_owner is not None:
                self.green_ctx_stream_owner.cleanup()
                self.green_ctx_stream_owner = None

    def torch_compile_payload(self) -> dict[str, str]:
        return {
            "requested": _torch_compile_requested(self.config.torch_compile_enabled),
            "status": self._torch_compile_status,
        }

    def _run_torch_compile_preflight(self) -> None:
        if not self.config.torch_compile_enabled or self._torch_compile_preflight_done:
            return
        if self._stage_runner is None or self.execution_stream is None:
            raise RuntimeError("torch.compile preflight requires initialized stage runner and execution stream")

        import torch.distributed as dist

        try:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            with torch.cuda.stream(self.execution_stream), torch.no_grad():
                _ = self._stage_runner()
            self.execution_stream.synchronize()
            self._consume_equal_token_routing_state()
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            self._torch_compile_preflight_done = True
        except Exception as exc:
            self._torch_compile_status = "compile_failed"
            raise TorchCompileFailure(f"torch.compile preflight failed: {exc}") from exc

    def _requested_green_ctx_sms(self) -> int | None:
        if self.config.stage_role == "attn":
            return self.config.green_ctx_attn_sms
        return self.config.green_ctx_moe_sms

    def _initialize_execution_stream(self) -> None:
        if self.config.runtime_backend == "mps_only":
            self.execution_stream = torch.cuda.current_stream(device=self.device)
            return
        if self.config.runtime_backend != "mps_green_ctx":
            raise ValueError(
                f"Unsupported runtime_backend: {self.config.runtime_backend}"
            )

        requested_sms = self._requested_green_ctx_sms()
        if requested_sms is None:
            raise ValueError(
                f"Missing Green Context SM budget for role={self.config.stage_role} under {self.config.runtime_backend}"
            )

        try:
            stream_owner = create_green_context_stream(
                device_id=int(self.device.index or 0), requested_sms=requested_sms
            )
        except GreenContextError as exc:
            raise RuntimeError(f"{exc.code}: {exc.message}") from exc
        stream_owner.wait_for_current_stream()
        self.green_ctx_stream_owner = stream_owner
        self.execution_stream = stream_owner.external_stream
        self.runtime_metadata = {
            "requested_sms": int(stream_owner.requested_sms),
            "granted_sms": int(stream_owner.granted_sms),
            "device_total_sms": int(stream_owner.total_sms),
        }

    def _build_trainer_config(self) -> dict[str, Any]:
        engine_config: dict[str, Any] = {
            "tensor_parallel_size": 1,
            "sequence_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "expert_model_parallel_size": int(self.config.expert_model_parallel_size),
            "attention_backend": str(self.config.attention_backend),
            "megatron_moe_grouped_gemm": True if self.config.moe_grouped_gemm else None,
            "megatron_moe_token_dispatcher_type": str(
                self.config.moe_token_dispatcher_type
            ),
            "megatron_overlap_moe_expert_parallel_comm": (
                True if self.config.overlap_moe_expert_parallel_comm else None
            ),
            "megatron_seed": int(self.config.seed),
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

    def _validate_moe_routing_mode_support(self) -> None:
        if self.config.moe_routing_mode == "normal" or self.config.stage_role != "moe":
            return
        if self.config.moe_routing_mode != "equal_tokens":
            raise RuntimeError(
                f"Unsupported moe_routing_mode={self.config.moe_routing_mode}"
            )
        if self.config.num_experts is None or int(self.config.num_experts) <= 0:
            raise RuntimeError("equal_tokens requires --num-experts > 0")
        if (
            int(self.config.num_experts) % int(self.config.expert_model_parallel_size)
            != 0
        ):
            raise RuntimeError(
                "equal_tokens requires num_experts to be divisible by expert_model_parallel_size"
            )
        mlp = getattr(self.layer, "mlp", None)
        router = getattr(mlp, "router", None) if mlp is not None else None
        token_dispatcher = (
            getattr(mlp, "token_dispatcher", None) if mlp is not None else None
        )
        if mlp is None or router is None or token_dispatcher is None:
            raise RuntimeError(
                "equal_tokens requires layer.mlp.router and layer.mlp.token_dispatcher"
            )
        router_topk = getattr(
            getattr(self.layer, "config", None), "moe_router_topk", None
        )
        if int(router_topk or 0) <= 0:
            raise RuntimeError("equal_tokens requires an effective moe_router_topk > 0")
        if int(router_topk or 0) > int(self.config.num_experts):
            raise RuntimeError("equal_tokens requires moe_router_topk <= num_experts")
        local_expert_indices = getattr(token_dispatcher, "local_expert_indices", None)
        if not isinstance(local_expert_indices, list) or not local_expert_indices:
            raise RuntimeError(
                "equal_tokens requires token_dispatcher.local_expert_indices"
            )

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
        mask = ~torch.tril(
            torch.ones((seq_len, seq_len), device=self.device, dtype=torch.bool)
        )
        return mask.view(1, 1, seq_len, seq_len).expand(batch, 1, seq_len, seq_len)

    def run_stage(
        self,
        *,
        warmup_iters: int,
        timed_iters: int,
        execution_schedule: str = "overlap",
        serial_phase_order: tuple[str, ...] | None = None,
        iteration_barrier: Any | None = None,
        abort_event: Any | None = None,
        barrier_timeout_s: float | None = None,
        profiler_trace_dir: str | None = None,
        profiler_worker_name: str | None = None,
        profiler_wait_iters: int | None = None,
        profiler_active_timed_iters: int | None = None,
    ) -> dict[str, Any]:
        if (
            self.layer is None
            or self.hidden_states is None
            or self.attention_mask is None
        ):
            self.initialize()

        assert self.layer is not None
        assert self.hidden_states is not None
        assert self.attention_mask is not None
        assert self.execution_stream is not None
        assert self._stage_runner is not None

        total_iters = int(warmup_iters) + int(timed_iters)
        if total_iters <= 0:
            raise ValueError("warmup_iters + timed_iters must be > 0")

        self._run_torch_compile_preflight()

        cuda_ms: list[float] = []
        step_ms: list[float] = []
        enqueue_windows: list[tuple[float, float]] = []
        timed_start_s: float | None = None
        timed_end_s: float | None = None
        first_nonfinite: dict[str, Any] | None = None
        output_tensor: torch.Tensor | None = None
        profiler: Any | None = None
        stable_tokens_per_expert: list[int] | None = None
        local_tokens_per_expert: list[int] | None = None

        import torch.distributed as dist

        try:
            if profiler_trace_dir is not None:
                os.makedirs(profiler_trace_dir, exist_ok=True)
                wait_iters, active_timed_iters = _resolve_profiler_schedule(
                    warmup_iters=int(warmup_iters),
                    timed_iters=int(timed_iters),
                    profiler_wait_iters=profiler_wait_iters,
                    profiler_active_timed_iters=profiler_active_timed_iters,
                )
                profiler = torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=wait_iters,
                        warmup=0,
                        active=active_timed_iters,
                        repeat=1,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        profiler_trace_dir,
                        worker_name=profiler_worker_name,
                    ),
                    record_shapes=True,
                    with_stack=False,
                )
                profiler.__enter__()

            def _wait_for_schedule_phase(phase_name: str, iter_idx: int) -> None:
                if abort_event is not None and abort_event.is_set():
                    raise ScheduleAborted(
                        f"Schedule aborted before {phase_name} at iter {iter_idx}"
                    )
                if iteration_barrier is None:
                    return
                try:
                    iteration_barrier.wait(timeout=barrier_timeout_s)
                except BrokenBarrierError as exc:
                    if abort_event is not None and abort_event.is_set():
                        raise ScheduleAborted(
                            f"Schedule aborted during {phase_name} at iter {iter_idx}"
                        ) from exc
                    raise RuntimeError(
                        f"Iteration barrier broke during {phase_name} at iter {iter_idx}"
                    ) from exc
                if abort_event is not None and abort_event.is_set():
                    raise ScheduleAborted(
                        f"Schedule aborted after {phase_name} at iter {iter_idx}"
                    )

            def _run_forward_phase(
                _phase_name: str, iter_idx: int, is_timed: bool
            ) -> None:
                nonlocal first_nonfinite, output_tensor, timed_start_s, timed_end_s
                nonlocal stable_tokens_per_expert, local_tokens_per_expert
                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                active_stream = self.execution_stream

                step_start = time.perf_counter()
                start_event.record(active_stream)
                enqueue_start = time.perf_counter()
                record_ctx = (
                    torch.profiler.record_function(
                        f"{self.config.stage_role}.iter_{iter_idx:04d}"
                    )
                    if profiler is not None
                    else nullcontext()
                )
                with record_ctx:
                    with torch.cuda.stream(active_stream), torch.no_grad():
                        output_tensor = self._stage_runner()

                if self.config.stage_role == "moe" and self.config.moe_routing_mode == "equal_tokens":
                    state = self._consume_equal_token_routing_state()
                    if state is None:
                        raise RuntimeError("equal_tokens stage runner did not produce routing state")
                    if is_timed:
                        if stable_tokens_per_expert is None:
                            stable_tokens_per_expert = list(state.tokens_per_expert)
                        elif stable_tokens_per_expert != state.tokens_per_expert:
                            raise RuntimeError(
                                "equal_tokens produced inconsistent tokens_per_expert across timed iterations"
                            )
                        local_tokens_per_expert = list(state.local_tokens_per_expert)
                else:
                    self._consume_equal_token_routing_state()
                enqueue_end = time.perf_counter()
                end_event.record(active_stream)
                active_stream.synchronize()
                step_end = time.perf_counter()

                if is_timed:
                    if timed_start_s is None:
                        timed_start_s = step_start
                    cuda_ms.append(float(start_event.elapsed_time(end_event)))
                    step_ms.append((step_end - step_start) * 1000.0)
                    enqueue_windows.append((enqueue_start, enqueue_end))
                    timed_end_s = step_end

                if (
                    first_nonfinite is None
                    and output_tensor is not None
                    and torch.is_floating_point(output_tensor)
                ):
                    if not torch.isfinite(output_tensor).all():
                        first_nonfinite = {
                            "module": f"{self.config.stage_role}_layer",
                            "phase": "forward",
                            "tensor": "output",
                            "iter": int(iter_idx),
                        }

            schedule_windows = _run_iteration_schedule(
                execution_schedule=execution_schedule,
                stage_role=self.config.stage_role,
                stage_label=self.config.stage_label,
                serial_phase_order=serial_phase_order,
                total_iters=total_iters,
                warmup_iters=int(warmup_iters),
                barrier_wait=_wait_for_schedule_phase,
                run_forward=_run_forward_phase,
                profiler_step=(lambda _iter_idx: profiler.step())
                if profiler is not None
                else None,
            )
        finally:
            if profiler is not None:
                profiler.__exit__(None, None, None)
            self.cleanup()

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
            "stage_label": str(self.config.stage_label or self.config.stage_role),
            "pair_index": int(self.config.pair_index),
            "runtime_backend": self.config.runtime_backend,
            "attention_backend": self.config.attention_backend,
            "attention_impl": self.attention_runtime,
            "moe_grouped_gemm": self.config.moe_grouped_gemm,
            "moe_token_dispatcher_type": self.config.moe_token_dispatcher_type,
            "overlap_moe_expert_parallel_comm": self.config.overlap_moe_expert_parallel_comm,
            "runtime": dict(self.runtime_metadata),
            "timing_ms": {
                "cuda": mean_cuda_ms,
                "step_total": mean_step_ms,
                "timed_wall": timed_wall_ms,
            },
            "timed_window_s": {
                **schedule_windows["stage_timed_window_s"],
            },
            "schedule_timed_window_s": schedule_windows["schedule_timed_window_s"],
            "enqueue_windows": enqueue_windows,
            "finite": {
                "all_finite": first_nonfinite is None,
                "first_nonfinite": first_nonfinite,
            },
            "output_signature": tensor_signature(output_tensor),
            "torch_compile": self.torch_compile_payload(),
            "moe_routing_mode": self.config.moe_routing_mode,
            "tokens_per_expert": stable_tokens_per_expert,
            "local_tokens_per_expert": local_tokens_per_expert,
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
    if isinstance(exc, TorchCompileFailure):
        status = "oom" if "out of memory" in lower else "runtime_error"
        code = "torch_compile_failed"
    elif "out of memory" in lower:
        status = "oom"
        code = status
    else:
        status = "runtime_error"
        code = status
    if isinstance(exc, GreenContextError):
        code = exc.code
    elif status == "runtime_error" and ":" in message and not isinstance(exc, TorchCompileFailure):
        possible_code, _, remainder = message.partition(":")
        if possible_code.startswith("green_context_") and remainder.strip():
            code = possible_code
            message = remainder.strip()
    return (
        status,
        {
            "code": code,
            "message": message,
            "traceback": traceback.format_exc(),
        },
    )
