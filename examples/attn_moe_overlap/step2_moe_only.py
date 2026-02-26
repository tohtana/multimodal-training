"""Step 2: MoE-only training loop (single GPU, native PyTorch).

Verifies Qwen3MoEStage trains correctly in isolation using MSE loss
against a fixed target tensor.
"""

import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import Qwen3MoEStage, create_qwen3_config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    config = create_qwen3_config()
    model = Qwen3MoEStage(config).to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print(f"MoE stage parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Fixed input and target for stable convergence
    torch.manual_seed(42)
    x = torch.randn(8, 8192, config.hidden_size, device=device, dtype=dtype)
    target = torch.randn_like(x)

    losses = []
    for step in range(20):
        output = model(x)
        loss = nn.functional.mse_loss(output, target)

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
