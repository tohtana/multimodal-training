"""MoE sublayer model factories for interleaved pipeline execution (design §4.5).

Provides simplified attention and MoE FFN blocks as separate nn.Module
implementations. Each block includes a residual connection internally
(design §11.2: output = sublayer(x) + x).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SimpleAttentionBlock(nn.Module):
    """Simplified attention sublayer: linear projection + residual.

    In a real transformer this would be multi-head self-attention.
    For pipeline testing, a linear projection exercises the same
    forward/backward/optimizer mechanics.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(x)


class SimpleMoEBlock(nn.Module):
    """Simplified MoE FFN sublayer: top-k gated experts + residual.

    Each expert is a single linear layer. The gating network routes each
    token to ``top_k`` experts via softmax-weighted combination.
    """

    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_logits = self.gate(x)  # [batch, num_experts]
        topk_weights, topk_indices = gate_logits.topk(self.top_k, dim=-1)
        topk_weights = torch.softmax(topk_weights, dim=-1)  # [batch, top_k]

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # [batch, num_experts, hidden]
        topk_outputs = expert_outputs.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        output = (topk_weights.unsqueeze(-1) * topk_outputs).sum(dim=1)  # [batch, hidden]

        return x + output


class SimpleMoEBlockWithHead(nn.Module):
    """Terminal MoE block with a classification head for loss computation."""

    def __init__(self, hidden_dim: int, num_classes: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.moe = SimpleMoEBlock(hidden_dim, num_experts, top_k)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.moe(x))
