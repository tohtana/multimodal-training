"""Qwen3-VL vision sequence-parallel helpers.

The DeepSpeed HF Ulysses wrapper is text-oriented: it requires position_ids and
assumes causal/packed text metadata. Qwen3-VL vision applies RoPE before the
attention interface and uses global cu_seqlens for non-causal varlen attention.
This module keeps those Qwen3 vision contracts while reusing DeepSpeed's
Ulysses all-to-all primitive.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _qwen3_vl_modeling():
    return importlib.import_module("transformers.models.qwen3_vl.modeling_qwen3_vl")


def _ds_dist():
    import deepspeed.comm as dist

    return dist


def _dim_zero_all_to_all():
    from deepspeed.sequence.layer import _DimZeroAllToAll

    return _DimZeroAllToAll


def _sp_rank_world(process_group) -> tuple[int, int]:
    if process_group is None:
        return 0, 1
    dist = _ds_dist()
    return dist.get_rank(process_group), dist.get_world_size(process_group)


def _all_to_all_dim0(tensor: torch.Tensor, process_group) -> torch.Tensor:
    if process_group is None:
        return tensor
    return _dim_zero_all_to_all().apply(process_group, tensor)


def _combine_local_sequences(
    tensor: torch.Tensor,
    *,
    process_group,
    local_seq_length: int,
    batch_size: int,
    head_count: int,
    head_dim: int,
) -> torch.Tensor:
    """All-to-all from local sequence/all heads to global sequence/local heads."""

    _, world_size = _sp_rank_world(process_group)
    if head_count % world_size != 0:
        raise ValueError(f"Qwen3 vision num_heads={head_count} must be divisible by SP size {world_size}")

    local_head_count = head_count // world_size
    tensor = tensor.reshape(local_seq_length, batch_size, world_size, local_head_count, head_dim)
    tensor = tensor.permute(2, 0, 1, 3, 4).contiguous()
    tensor = _all_to_all_dim0(tensor, process_group)
    return tensor.reshape(local_seq_length * world_size, batch_size, local_head_count, head_dim).contiguous()


def _partition_global_sequence(
    tensor: torch.Tensor,
    *,
    process_group,
    local_seq_length: int,
    batch_size: int,
    hidden_size: int,
) -> torch.Tensor:
    """All-to-all from global sequence/local heads back to local sequence/all heads."""

    _, world_size = _sp_rank_world(process_group)
    local_hidden = hidden_size // world_size
    tensor = tensor.reshape(world_size, local_seq_length, batch_size, local_hidden).contiguous()
    tensor = _all_to_all_dim0(tensor, process_group)
    tensor = tensor.permute(1, 2, 0, 3).contiguous()
    return tensor.reshape(local_seq_length, batch_size, hidden_size).contiguous()


def _all_gather_dim0(tensor: torch.Tensor, process_group) -> torch.Tensor:
    if process_group is None:
        return tensor
    import torch.distributed.nn.functional as dist_nn

    pieces = dist_nn.all_gather(tensor, group=process_group)
    return torch.cat(tuple(pieces), dim=0)


class Qwen3VLUlyssesVisionAttention(nn.Module):
    """Qwen3-VL vision attention with DeepSpeed Ulysses all-to-all.

    Input and output stay Qwen3-compatible: hidden states are local sequence
    shards shaped [seq_local, hidden], cu_seqlens remain global, attention is
    non-causal, and position_embeddings are the local slice for the same shard.
    """

    def __init__(self, attention: nn.Module, process_group) -> None:
        super().__init__()
        self.dim = attention.dim
        self.num_heads = attention.num_heads
        self.head_dim = attention.head_dim
        self.num_key_value_groups = attention.num_key_value_groups
        self.qkv = attention.qkv
        self.proj = attention.proj
        self.scaling = attention.scaling
        self.config = attention.config
        self.attention_dropout = attention.attention_dropout
        self.is_causal = False
        self.process_group = process_group

    def _attention_interface(self):
        modeling = _qwen3_vl_modeling()
        return modeling.ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            modeling.eager_attention_forward,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if position_embeddings is None:
            raise ValueError("Qwen3 vision Ulysses attention requires position_embeddings=(cos, sin)")
        if hidden_states.dim() != 2:
            raise ValueError(f"Expected hidden_states [seq_local, hidden], got {tuple(hidden_states.shape)}")

        _, world_size = _sp_rank_world(self.process_group)
        local_seq_length = hidden_states.shape[0]
        global_seq_length = int(cu_seqlens[-1].item())
        if local_seq_length * world_size != global_seq_length:
            raise ValueError(
                "Qwen3 vision Ulysses requires equal contiguous sequence shards: "
                f"local_seq={local_seq_length}, world={world_size}, cu_seqlens[-1]={global_seq_length}"
            )

        cos, sin = position_embeddings
        if cos.shape[0] != local_seq_length or sin.shape[0] != local_seq_length:
            raise ValueError(
                "position_embeddings must be sliced to the local sequence shard: "
                f"cos={tuple(cos.shape)}, sin={tuple(sin.shape)}, local_seq={local_seq_length}"
            )

        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(local_seq_length, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )

        modeling = _qwen3_vl_modeling()
        query_states, key_states = modeling.apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        batch_size = 1
        query_states = query_states.unsqueeze(1)
        key_states = key_states.unsqueeze(1)
        value_states = value_states.unsqueeze(1)

        query_states = _combine_local_sequences(
            query_states,
            process_group=self.process_group,
            local_seq_length=local_seq_length,
            batch_size=batch_size,
            head_count=self.num_heads,
            head_dim=self.head_dim,
        )
        key_states = _combine_local_sequences(
            key_states,
            process_group=self.process_group,
            local_seq_length=local_seq_length,
            batch_size=batch_size,
            head_count=self.num_heads,
            head_dim=self.head_dim,
        )
        value_states = _combine_local_sequences(
            value_states,
            process_group=self.process_group,
            local_seq_length=local_seq_length,
            batch_size=batch_size,
            head_count=self.num_heads,
            head_dim=self.head_dim,
        )

        query_states = query_states.permute(1, 2, 0, 3).contiguous()
        key_states = key_states.permute(1, 2, 0, 3).contiguous()
        value_states = value_states.permute(1, 2, 0, 3).contiguous()

        attention_interface = self._attention_interface()
        if modeling.is_flash_attention_requested(self.config):
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
            attn_output, _ = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                cu_seq_lens_q=cu_seqlens,
                cu_seq_lens_k=cu_seqlens,
                max_length_q=max_seqlen,
                max_length_k=max_seqlen,
                is_causal=False,
                **kwargs,
            )
        else:
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            splits = [
                torch.split(tensor, lengths.tolist(), dim=2)
                for tensor in (query_states, key_states, value_states)
            ]
            attn_output = torch.cat(
                [
                    attention_interface(
                        self,
                        q,
                        k,
                        v,
                        attention_mask=None,
                        scaling=self.scaling,
                        dropout=0.0 if not self.training else self.attention_dropout,
                        is_causal=False,
                        **kwargs,
                    )[0]
                    for q, k, v in zip(*splits)
                ],
                dim=1,
            )

        attn_output = attn_output.reshape(batch_size, global_seq_length, -1).permute(1, 0, 2).contiguous()
        attn_output = _partition_global_sequence(
            attn_output,
            process_group=self.process_group,
            local_seq_length=local_seq_length,
            batch_size=batch_size,
            hidden_size=self.dim,
        )
        attn_output = attn_output.squeeze(1)
        return self.proj(attn_output)


def apply_qwen3_vision_ulysses(model: nn.Module, process_group) -> int:
    """Replace Qwen3 dense vision attention modules with the Ulysses adapter."""

    replaced = 0
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is None:
            continue
        if attn.__class__.__name__ == "Qwen3VLVisionAttention":
            block.attn = Qwen3VLUlyssesVisionAttention(attn, process_group)
            replaced += 1
    if replaced == 0:
        raise ValueError("No Qwen3VLVisionAttention modules were found to wrap for Ulysses")
    return replaced


def _qwen3_cu_seqlens(grid_thw: torch.Tensor) -> torch.Tensor:
    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    return F.pad(cu_seqlens, (1, 0), value=0)


def _slice_local_sequence(tensor: torch.Tensor, process_group) -> torch.Tensor:
    rank, world_size = _sp_rank_world(process_group)
    if tensor.shape[0] % world_size != 0:
        raise ValueError(f"Sequence length {tensor.shape[0]} must be divisible by SP size {world_size}")
    local_seq_length = tensor.shape[0] // world_size
    start = rank * local_seq_length
    end = start + local_seq_length
    return tensor[start:end].contiguous()


def qwen3_vision_sequence_parallel_forward(
    model: nn.Module,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    process_group,
    **kwargs: Any,
):
    """Run Qwen3 dense vision with sequence-sharded blocks and gathered outputs."""

    modeling = _qwen3_vl_modeling()

    hidden_states = model.patch_embed(hidden_states)
    hidden_states = hidden_states + model.fast_pos_embed_interpolate(grid_thw)

    rotary_pos_emb = model.rot_pos_emb(grid_thw)
    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = _qwen3_cu_seqlens(grid_thw).to(device=hidden_states.device)

    local_hidden_states = _slice_local_sequence(hidden_states, process_group)
    local_position_embeddings = tuple(_slice_local_sequence(tensor, process_group) for tensor in position_embeddings)
    local_seq_length = local_hidden_states.shape[0]
    if local_seq_length % model.spatial_merge_unit != 0:
        raise ValueError(
            "Qwen3 vision SP shards must preserve merger groups: "
            f"local_seq={local_seq_length}, spatial_merge_unit={model.spatial_merge_unit}"
        )

    deepstack_features = []
    for layer_num, block in enumerate(model.blocks):
        local_hidden_states = block(
            local_hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=local_position_embeddings,
            **kwargs,
        )
        if layer_num in model.deepstack_visual_indexes:
            local_deepstack_feature = model.deepstack_merger_list[model.deepstack_visual_indexes.index(layer_num)](
                local_hidden_states
            )
            deepstack_features.append(_all_gather_dim0(local_deepstack_feature, process_group))

    local_pooler_output = model.merger(local_hidden_states)
    last_hidden_state = _all_gather_dim0(local_hidden_states, process_group)
    pooler_output = _all_gather_dim0(local_pooler_output, process_group)

    return modeling.BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        deepstack_features=deepstack_features,
    )
