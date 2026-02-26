"""Step 3: Two-stage baseline (single GPU, no pipeline framework).

Verifies attention + MoE compose correctly, providing a single-GPU reference
for pipeline correctness comparison in Steps 4 and 5.
"""

import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
    generate_dummy_batch,
)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    config = create_qwen3_config()
    num_classes = 10

    attn_stage = Qwen3AttentionStage(config).to(device=device, dtype=dtype)
    moe_stage = Qwen3MoEStageWithHead(config, num_classes).to(device=device, dtype=dtype)

    total_params = sum(p.numel() for p in attn_stage.parameters()) + sum(p.numel() for p in moe_stage.parameters())
    print(f"Total parameters: {total_params:,}")

    # Single optimizer over both stages
    all_params = list(attn_stage.parameters()) + list(moe_stage.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Fixed batch for stable convergence
    torch.manual_seed(42)
    x, labels = generate_dummy_batch(config, batch_size=8, seq_len=8192, num_classes=num_classes, device=device, dtype=dtype)

    losses = []
    for step in range(20):
        h = attn_stage(x)
        logits = moe_stage(h)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Step {step:3d} | Loss: {loss.item():.6f}")

    # Convergence check: mean of last 5 < mean of first 5
    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    assert last_5 < first_5, f"Not converging: first_5={first_5:.6f}, last_5={last_5:.6f}"
    print("PASSED: Loss converged")


if __name__ == "__main__":
    main()
