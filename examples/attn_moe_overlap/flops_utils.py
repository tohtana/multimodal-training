"""FLOPs computation and MFU utilities for MPS overlap benchmarking."""

from __future__ import annotations

import torch

# NVIDIA datasheet theoretical peaks for bf16 (TFLOP/s).
# Sources: NVIDIA product specification pages.
GPU_PEAK_TFLOPS_BF16 = {
    "H100": 1979.0,  # H100 SXM5, bf16 tensor core
    "H200": 1979.0,  # Same compute die as H100
    "A100": 312.0,  # A100 80GB SXM, bf16 tensor core
    "A10G": 70.0,  # A10G, bf16 tensor core
    "L40S": 362.0,  # L40S, bf16 tensor core
    "L4": 121.0,  # L4, bf16 tensor core
}


def compute_attn_forward_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> int:
    """Theoretical forward FLOPs for Qwen3MoeAttention stage (single layer).

    Components:
    - QKV projections: 2 * T * H * (Nh + 2*Nkv) * D
    - Attention matmuls (QK^T + AV): 4 * B * Nh * S^2 * D
    - Output projection: 2 * T * Nh * D * H
    """
    T = batch_size * seq_len
    qkv = 2 * T * hidden_size * (num_heads + 2 * num_kv_heads) * head_dim
    attn = 4 * batch_size * num_heads * seq_len * seq_len * head_dim
    proj = 2 * T * num_heads * head_dim * hidden_size
    return qkv + attn + proj


def compute_moe_forward_flops(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    num_experts: int,
    num_experts_per_tok: int,
    moe_intermediate_size: int,
    num_classes: int,
) -> int:
    """Theoretical forward FLOPs for Qwen3MoEStageWithHead (terminal stage).

    Components:
    - Router gate: 2 * T * H * E
    - Active experts FFN (gate_proj + up_proj + down_proj): 2 * T * K * H * I * 3
    - Classifier head: 2 * B * H * C
    """
    T = batch_size * seq_len
    gate = 2 * T * hidden_size * num_experts
    ffn = 2 * T * num_experts_per_tok * hidden_size * moe_intermediate_size * 3
    head = 2 * batch_size * hidden_size * num_classes
    return gate + ffn + head


def get_gpu_peak_tflops(dtype: torch.dtype = torch.bfloat16) -> float | None:
    """Look up GPU peak TFLOP/s from the constant table.

    Returns None if GPU model is not in the table or dtype is not bf16.
    """
    if dtype != torch.bfloat16:
        return None
    if not torch.cuda.is_available():
        return None
    name = torch.cuda.get_device_name(0)
    for key, tflops in GPU_PEAK_TFLOPS_BF16.items():
        if key in name:
            return tflops
    return None


def compute_mfu(achieved_flops_per_sec: float, peak_tflops: float | None) -> float | None:
    """Compute Model FLOPs Utilization = achieved / peak.

    Returns None if peak is not available.
    """
    if peak_tflops is None or peak_tflops <= 0:
        return None
    peak_flops_per_sec = peak_tflops * 1e12
    return achieved_flops_per_sec / peak_flops_per_sec
