"""M8: Layout redistribution tests — tensor reshaping at parallelism boundaries.

CPU-only tests verify:
  1. identity_adapter is a no-op.
  2. shard_dim_adapter correctly shards and round-trips.
  3. gather_dim_adapter correctly gathers and round-trips.
  4. sp_to_tp_adapter composes gather-seq + shard-hidden.
  5. resolve_layout_adapter selects correct adapter for parallelism pairs.
  6. LayoutContractError on non-divisible dimensions.
  7. Custom EdgeConfig.layout_fn overrides auto-resolution.
"""

import pytest
import torch

from python.pipeline.layout import (
    LayoutAdapter,
    LayoutContractError,
    gather_dim_adapter,
    identity_adapter,
    resolve_layout_adapter,
    shard_dim_adapter,
    sp_to_tp_adapter,
)

pytestmark = [pytest.mark.cpu_only]

WORLD_SIZE = 4


class TestIdentityAdapter:
    def test_forward_noop(self):
        adapter = identity_adapter()
        t = torch.randn(8, 16)
        result = adapter.forward_fn(t, rank=0, world_size=WORLD_SIZE)
        assert result is t

    def test_backward_noop(self):
        adapter = identity_adapter()
        t = torch.randn(8, 16)
        result = adapter.backward_fn(t, rank=0, world_size=WORLD_SIZE)
        assert result is t


class TestShardDimAdapter:
    def test_shard_dim0(self):
        """Shard along dim 0 (batch/sequence)."""
        adapter = shard_dim_adapter(dim=0)
        t = torch.arange(16).reshape(4, 4)  # [4, 4]
        for rank in range(WORLD_SIZE):
            shard = adapter.forward_fn(t, rank=rank, world_size=WORLD_SIZE)
            assert shard.shape == (1, 4)
            assert torch.equal(shard, t[rank : rank + 1])

    def test_shard_dim_last(self):
        """Shard along last dim (hidden)."""
        adapter = shard_dim_adapter(dim=-1)
        t = torch.arange(32).reshape(2, 16)  # [2, 16]
        for rank in range(WORLD_SIZE):
            shard = adapter.forward_fn(t, rank=rank, world_size=WORLD_SIZE)
            assert shard.shape == (2, 4)
            assert torch.equal(shard, t[:, rank * 4 : (rank + 1) * 4])

    def test_non_divisible_raises(self):
        adapter = shard_dim_adapter(dim=0)
        t = torch.randn(5, 4)  # 5 not divisible by 4
        with pytest.raises(LayoutContractError, match="Cannot evenly shard"):
            adapter.forward_fn(t, rank=0, world_size=WORLD_SIZE)

    def test_shard_all_ranks_cover_full_tensor(self):
        """Concatenating all shards reconstructs the original tensor."""
        adapter = shard_dim_adapter(dim=0)
        t = torch.randn(8, 16)
        shards = [adapter.forward_fn(t, rank=r, world_size=WORLD_SIZE) for r in range(WORLD_SIZE)]
        reconstructed = torch.cat(shards, dim=0)
        assert torch.equal(reconstructed, t)


class TestGatherDimAdapter:
    def test_backward_shards_correctly(self):
        """Backward (shard) of gather_dim is inverse: full tensor → rank shard."""
        adapter = gather_dim_adapter(dim=0)
        t = torch.arange(16).reshape(4, 4)
        for rank in range(WORLD_SIZE):
            shard = adapter.backward_fn(t, rank=rank, world_size=WORLD_SIZE)
            assert shard.shape == (1, 4)
            assert torch.equal(shard, t[rank : rank + 1])

    def test_backward_non_divisible_raises(self):
        adapter = gather_dim_adapter(dim=0)
        t = torch.randn(5, 4)
        with pytest.raises(LayoutContractError, match="Cannot evenly shard"):
            adapter.backward_fn(t, rank=0, world_size=WORLD_SIZE)


class TestSpToTpAdapter:
    def test_forward_shard_hidden(self):
        """SP→TP forward: (local) shard hidden dim."""
        adapter = sp_to_tp_adapter(seq_dim=0, hidden_dim=-1)
        # Simulate: each SP rank has a sequence shard [2, 16]
        # After gather seq (no-op locally), shard hidden: [2, 4]
        t = torch.randn(2, 16)
        result = adapter.forward_fn(t, rank=1, world_size=WORLD_SIZE)
        assert result.shape == (2, 4)
        assert torch.equal(result, t[:, 4:8])

    def test_all_ranks_cover_hidden_dim(self):
        adapter = sp_to_tp_adapter(seq_dim=0, hidden_dim=-1)
        t = torch.randn(8, 32)
        shards = [adapter.forward_fn(t, rank=r, world_size=WORLD_SIZE) for r in range(WORLD_SIZE)]
        reconstructed = torch.cat(shards, dim=-1)
        assert torch.equal(reconstructed, t)


class TestResolveLayoutAdapter:
    def test_same_parallelism_identity(self):
        adapter = resolve_layout_adapter("sequence", "sequence")
        assert adapter.name == "identity"

    def test_sp_to_none(self):
        adapter = resolve_layout_adapter("sequence", "none")
        assert adapter.name == "gather_dim0"

    def test_none_to_tp(self):
        adapter = resolve_layout_adapter("none", "tensor")
        assert adapter.name == "shard_dim-1"

    def test_sp_to_tp(self):
        adapter = resolve_layout_adapter("sequence", "tensor")
        assert adapter.name == "sp_to_tp"

    def test_tp_to_none(self):
        adapter = resolve_layout_adapter("tensor", "none")
        assert adapter.name == "gather_dim-1"

    def test_unknown_pair_returns_identity(self):
        adapter = resolve_layout_adapter("expert", "unknown")
        assert adapter.name == "identity"

    def test_custom_fn_override(self):
        custom = LayoutAdapter(
            forward_fn=lambda t, r, w: t,
            backward_fn=lambda t, r, w: t,
            name="my_custom",
        )

        def my_layout_fn(src, dst):
            return custom

        adapter = resolve_layout_adapter("sequence", "tensor", custom_fn=my_layout_fn)
        assert adapter.name == "my_custom"

    def test_custom_fn_returns_none_falls_through(self):
        """If custom_fn returns non-LayoutAdapter, fall through to auto-resolution."""

        def my_layout_fn(src, dst):
            return None

        adapter = resolve_layout_adapter("sequence", "tensor", custom_fn=my_layout_fn)
        assert adapter.name == "sp_to_tp"
