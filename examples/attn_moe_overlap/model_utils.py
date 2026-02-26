"""Shared wrapper modules, config factory, and batch generation for Qwen3 MoE pipeline example."""

import torch
import torch.nn as nn
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeAttention,
    Qwen3MoeConfig,
    Qwen3MoeRMSNorm,
    Qwen3MoeRotaryEmbedding,
    Qwen3MoeSparseMoeBlock,
    create_causal_mask,
)


class Qwen3AttentionStage(nn.Module):
    """Wraps Qwen3MoeAttention + input_layernorm + RoPE + residual for pipeline use."""

    def __init__(self, config: Qwen3MoeConfig, dtype: torch.dtype = None):
        super().__init__()
        # Required when instantiating Qwen3 sublayers directly (without Qwen3MoeModel).
        if getattr(config, "_attn_implementation", None) is None:
            config._attn_implementation = "sdpa"
        self.config = config
        self.layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = Qwen3MoeAttention(config, layer_idx=0)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        residual = x
        x = self.layernorm(x)
        position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        cos, sin = self.rotary_emb(x, position_ids)
        # For SDPA/flash attention, pass attention_mask=None to use the efficient
        # is_causal=True path instead of materializing the full [seq, seq] mask.
        # For eager attention, we need the explicit causal mask.
        if self.config._attn_implementation == "eager":
            cache_position = position_ids[0]
            causal_mask = create_causal_mask(
                config=self.config,
                input_embeds=x,
                attention_mask=None,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=position_ids,
            )
        else:
            causal_mask = None
        attn_output, _ = self.attn(
            x,
            position_embeddings=(cos, sin),
            attention_mask=causal_mask,
        )
        return residual + attn_output


class Qwen3MoEStage(nn.Module):
    """Wraps Qwen3MoeSparseMoeBlock + post_attention_layernorm + residual for pipeline use."""

    def __init__(self, config: Qwen3MoeConfig, dtype: torch.dtype = None):
        super().__init__()
        self._dtype = dtype
        self.layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = Qwen3MoeSparseMoeBlock(config)
        # Fused experts (transformers >= 5.x) use torch.empty() without init.
        # When used standalone (not via Qwen3MoeForCausalLM), _init_weights is
        # never called, leaving expert weights at zero. Initialize them here.
        self._init_expert_weights(config)
        if dtype is not None:
            self.to(dtype=dtype)

    @torch.no_grad()
    def _init_expert_weights(self, config):
        """Initialize fused expert parameters that torch.empty() left uninitialized."""
        std = config.initializer_range
        experts = self.moe.experts
        if hasattr(experts, "gate_up_proj"):
            nn.init.normal_(experts.gate_up_proj, mean=0.0, std=std)
        if hasattr(experts, "down_proj"):
            nn.init.normal_(experts.down_proj, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        residual = x
        x = self.layernorm(x)
        moe_out = self.moe(x)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        return residual + moe_out


class Qwen3MoEStageWithHead(nn.Module):
    """Terminal stage: MoE block + mean pooling + classification head."""

    def __init__(self, config: Qwen3MoeConfig, num_classes: int, dtype: torch.dtype = None):
        super().__init__()
        self.moe_stage = Qwen3MoEStage(config, dtype=dtype)
        self.head = nn.Linear(config.hidden_size, num_classes)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, hidden_size]
        x = self.moe_stage(x)
        x = x.mean(dim=1)  # pool over sequence dim -> [batch, hidden]
        return self.head(x)  # [batch, num_classes]


def create_qwen3_config(attn_implementation="sdpa", **overrides) -> Qwen3MoeConfig:
    """Create reduced Qwen3MoeConfig suitable for single-GPU training."""
    defaults = dict(
        num_experts=8,
        num_experts_per_tok=2,
        hidden_size=2048,
        num_attention_heads=32,
        num_key_value_heads=4,
        moe_intermediate_size=768,
    )
    defaults.update(overrides)
    config = Qwen3MoeConfig(**defaults)
    config._attn_implementation = attn_implementation
    return config


def generate_dummy_batch(
    config: Qwen3MoeConfig,
    batch_size: int = 4,
    seq_len: int = 128,
    num_classes: int = 10,
    device: str = "cpu",
    dtype: torch.dtype = None,
) -> tuple:
    """Generate (hidden_states, labels) for training.

    hidden_states: [batch, seq_len, hidden_size] random normal
    labels: [batch] random integers in [0, num_classes)
    """
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device, dtype=dtype)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return hidden_states, labels
