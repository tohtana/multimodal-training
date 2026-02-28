"""Deterministic regression tests for FLOPs computation formulas."""

import pytest

from examples.attn_moe_overlap.flops_utils import compute_attn_forward_flops, compute_moe_forward_flops

# Default config: batch=8, seq=8192, hidden=2048, heads=32, kv_heads=4, head_dim=64,
# experts=8, k=2, moe_intermediate=768, classes=10
BATCH = 8
SEQ = 8192
HIDDEN = 2048
HEADS = 32
KV_HEADS = 4
HEAD_DIM = 64
EXPERTS = 8
K = 2
MOE_INTERMEDIATE = 768
NUM_CLASSES = 10


@pytest.mark.cpu_only
def test_attn_forward_flops():
    result = compute_attn_forward_flops(
        batch_size=BATCH,
        seq_len=SEQ,
        hidden_size=HIDDEN,
        num_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
    )
    assert result == 5634997092352, f"Expected 5634997092352, got {result}"


@pytest.mark.cpu_only
def test_moe_forward_flops():
    result = compute_moe_forward_flops(
        batch_size=BATCH,
        seq_len=SEQ,
        hidden_size=HIDDEN,
        num_experts=EXPERTS,
        num_experts_per_tok=K,
        moe_intermediate_size=MOE_INTERMEDIATE,
        num_classes=NUM_CLASSES,
    )
    assert result == 1239098392576, f"Expected 1239098392576, got {result}"
