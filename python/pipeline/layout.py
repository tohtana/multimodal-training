"""Layout adapters for tensor redistribution between parallelism strategies (design §6.5).

When an edge crosses parallelism boundaries (e.g., SP → TP), the tensor layout
must be transformed. Adapters are pure functions: (tensor, rank, world_size) → tensor.

Forward adapters transform activations; backward adapters are the exact inverse
(applied to gradients flowing in the opposite direction).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch


class LayoutContractError(Exception):
    """Raised when a layout adapter encounters an invalid tensor shape or metadata."""


class LayoutAdapterFn(Protocol):
    """Protocol for layout adapter functions."""

    def __call__(self, tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor: ...


@dataclass(frozen=True)
class LayoutAdapter:
    """A pair of forward/backward layout adapters for an edge.

    forward_fn transforms activations: src layout → dst layout.
    backward_fn transforms gradients: dst layout → src layout (exact inverse).
    """

    forward_fn: LayoutAdapterFn
    backward_fn: LayoutAdapterFn
    name: str = "custom"


def identity_adapter() -> LayoutAdapter:
    """No-op adapter for same parallelism / same layout."""

    def _identity(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        return tensor

    return LayoutAdapter(forward_fn=_identity, backward_fn=_identity, name="identity")


def shard_dim_adapter(dim: int) -> LayoutAdapter:
    """Shard a tensor along `dim` in forward; gather along `dim` in backward.

    Forward: full tensor → shard for this rank (e.g., none → TP on hidden dim).
    Backward: gradient shard → full gradient (gather from all ranks).
    """

    def _shard(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        if tensor.size(dim) % world_size != 0:
            raise LayoutContractError(
                f"Cannot evenly shard dim {dim} (size {tensor.size(dim)}) across {world_size} ranks"
            )
        chunks = tensor.chunk(world_size, dim=dim)
        return chunks[rank].contiguous()

    def _gather(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        # In a distributed setting, this would be an all-gather.
        # For local/single-process use, we pad to the expected full size.
        # The actual distributed all-gather is handled by the transport layer.
        return tensor

    return LayoutAdapter(forward_fn=_shard, backward_fn=_gather, name=f"shard_dim{dim}")


def gather_dim_adapter(dim: int) -> LayoutAdapter:
    """Gather shards along `dim` in forward; shard along `dim` in backward.

    Forward: sequence shard → full tensor (e.g., SP → none via all-gather on seq dim).
    Backward: full gradient → gradient shard for this rank.
    """

    def _gather(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        # In a distributed setting, this would be an all-gather.
        # For local/single-process, pass through (the transport layer handles gathering).
        return tensor

    def _shard(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        if tensor.size(dim) % world_size != 0:
            raise LayoutContractError(
                f"Cannot evenly shard dim {dim} (size {tensor.size(dim)}) across {world_size} ranks"
            )
        chunks = tensor.chunk(world_size, dim=dim)
        return chunks[rank].contiguous()

    return LayoutAdapter(forward_fn=_gather, backward_fn=_shard, name=f"gather_dim{dim}")


def sp_to_tp_adapter(seq_dim: int = 0, hidden_dim: int = -1) -> LayoutAdapter:
    """Compose: gather sequence dim (SP exit) then shard hidden dim (TP entry).

    Forward: SP shard → all-gather seq → shard hidden → TP shard.
    Backward: TP grad shard → all-gather hidden → shard seq → SP grad shard.
    """
    gather_seq = gather_dim_adapter(seq_dim)
    shard_hidden = shard_dim_adapter(hidden_dim)

    def _forward(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        t = gather_seq.forward_fn(tensor, rank, world_size)
        return shard_hidden.forward_fn(t, rank, world_size)

    def _backward(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        t = shard_hidden.backward_fn(tensor, rank, world_size)
        return gather_seq.backward_fn(t, rank, world_size)

    return LayoutAdapter(forward_fn=_forward, backward_fn=_backward, name="sp_to_tp")


def tp_to_ep_adapter() -> LayoutAdapter:
    """TP → EP layout adapter: identity per-actor.

    In a full distributed implementation, this would perform an all-to-all
    redistribution (gather hidden-dim shards, scatter by expert routing).
    In the pipeline framework, cross-actor redistribution is handled by
    M:N routing; the per-actor adapter is identity.
    """

    def _identity(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        return tensor

    return LayoutAdapter(forward_fn=_identity, backward_fn=_identity, name="tp_to_ep")


def ep_to_tp_adapter() -> LayoutAdapter:
    """EP → TP layout adapter: identity per-actor.

    In a full distributed implementation, this would perform an all-to-all
    redistribution (gather expert outputs, shard hidden dim).
    In the pipeline framework, cross-actor redistribution is handled by
    M:N routing; the per-actor adapter is identity.
    """

    def _identity(tensor: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
        return tensor

    return LayoutAdapter(forward_fn=_identity, backward_fn=_identity, name="ep_to_tp")


def resolve_layout_adapter(
    src_parallelism: str,
    dst_parallelism: str,
    custom_fn: Callable | None = None,
) -> LayoutAdapter:
    """Resolve the layout adapter for an edge based on parallelism types.

    Args:
        src_parallelism: Source stage parallelism ("none", "sequence", "tensor", etc.).
        dst_parallelism: Destination stage parallelism.
        custom_fn: Optional custom EdgeConfig.layout_fn override. If provided and it
                   returns a LayoutAdapter, use it; otherwise fall back to auto-resolution.
    """
    if custom_fn is not None:
        result = custom_fn(src_parallelism, dst_parallelism)
        if isinstance(result, LayoutAdapter):
            return result

    if src_parallelism == dst_parallelism:
        return identity_adapter()

    key = (src_parallelism, dst_parallelism)

    if key == ("sequence", "none"):
        return gather_dim_adapter(dim=0)
    if key == ("none", "tensor"):
        return shard_dim_adapter(dim=-1)
    if key == ("sequence", "tensor"):
        return sp_to_tp_adapter(seq_dim=0, hidden_dim=-1)
    if key == ("tensor", "none"):
        return gather_dim_adapter(dim=-1)
    if key == ("none", "sequence"):
        return shard_dim_adapter(dim=0)
    if key == ("tensor", "sequence"):
        # Reverse of sp_to_tp: gather hidden, shard seq
        gather_hidden = gather_dim_adapter(dim=-1)
        shard_seq = shard_dim_adapter(dim=0)

        def _forward(tensor, rank, world_size):
            t = gather_hidden.forward_fn(tensor, rank, world_size)
            return shard_seq.forward_fn(t, rank, world_size)

        def _backward(tensor, rank, world_size):
            t = shard_seq.backward_fn(tensor, rank, world_size)
            return gather_hidden.backward_fn(t, rank, world_size)

        return LayoutAdapter(forward_fn=_forward, backward_fn=_backward, name="tp_to_sp")

    if key == ("tensor", "expert"):
        return tp_to_ep_adapter()
    if key == ("expert", "tensor"):
        return ep_to_tp_adapter()

    # Default: identity with a warning
    return identity_adapter()
