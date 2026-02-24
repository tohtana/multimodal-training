"""Minimal StageTrainer for native PyTorch pipeline execution (no Ray, no DeepSpeed).

Each StageTrainer wraps an nn.Module and provides forward_step / backward_step /
optimizer_step for use by the native_runner.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Callable

import torch
import torch.nn as nn

from ..ray.payloads import StageGradients, StageOutputs

logger = logging.getLogger(__name__)


class StageTrainer:
    """Wraps an nn.Module for pipeline stage execution.

    Lifecycle:
      1. forward_step(inputs) → StageOutputs  (retains output for backward)
      2. backward_step(grad)  → StageGradients or None
      3. optimizer_step(global_grad_norm)

    Supports multiple in-flight microbatches: forward pushes activations onto
    a FIFO queue, backward pops them in the same order. Gradients accumulate
    naturally across microbatches (no zero_grad between them).
    """

    def __init__(
        self,
        stage_name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        loss_fn: Callable | None = None,
        is_terminal: bool = False,
        device: torch.device | None = None,
    ):
        self.stage_name = stage_name
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.is_terminal = is_terminal
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # FIFO queue of (saved_tensor, saved_input) for microbatch support.
        # saved_tensor is either loss (terminal) or output (non-terminal).
        self._activation_queue: deque[tuple[torch.Tensor, torch.Tensor | None]] = deque()
        self._last_loss_value: float | None = None
        self._last_forward_meta: dict = {}

    def forward_step(self, inputs: StageOutputs | None = None, labels: Any = None) -> StageOutputs:
        """Run the model forward.

        For non-source stages: inputs.activations is the input tensor.
        For source stages: inputs may be None (dataloader-provided data handled externally).

        The output tensor is retained (with its autograd graph) for backward.
        Terminal stages compute and store loss instead.
        """
        if inputs is not None:
            x = inputs.activations
        else:
            raise ValueError(f"Stage '{self.stage_name}': forward_step requires inputs")

        # At stage boundaries, detach the input tensor and create a new leaf tensor
        # with requires_grad=True. This ensures that backward() will populate .grad
        # on this tensor, which we return as the upstream gradient.
        if isinstance(x, torch.Tensor):
            # Move to model device if needed (e.g., CPU→CUDA when receiving from Ray driver)
            x = x.to(self.device).detach().requires_grad_(True)
            self._saved_input = x
        else:
            self._saved_input = None

        output = self.model(x)

        if self.is_terminal:
            if self.loss_fn is None:
                raise ValueError(f"Terminal stage '{self.stage_name}' requires a loss_fn")
            if labels is None:
                raise ValueError(f"Terminal stage '{self.stage_name}' requires labels")
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
            loss = self.loss_fn(output, labels)
            self._activation_queue.append((loss, x))
            self._last_loss_value = loss.item()
            self._last_forward_meta = {"loss": self._last_loss_value}
            return StageOutputs(activations=output, meta={"loss": self._last_loss_value})
        else:
            self._activation_queue.append((output, x))
            return StageOutputs(activations=output)

    def backward_step(self, downstream_grad: StageGradients | None = None) -> StageGradients | None:
        """Run backward for the oldest queued microbatch (FIFO).

        Terminal stages: loss.backward().
        Non-terminal stages: output.backward(grad) where grad comes from downstream.

        Returns StageGradients with grad w.r.t. this stage's input (for upstream),
        or None if this is a source stage.
        """
        assert self._activation_queue, f"Stage '{self.stage_name}': backward called before forward"
        saved_tensor, saved_input = self._activation_queue.popleft()

        if self.is_terminal:
            saved_tensor.backward()
        else:
            assert downstream_grad is not None, f"Non-terminal stage '{self.stage_name}' requires downstream_grad"
            grad = downstream_grad.grad
            if isinstance(grad, torch.Tensor):
                grad = grad.to(self.device)
            saved_tensor.backward(grad)

        # Return gradient w.r.t. this stage's input for upstream stages
        if saved_input is not None and saved_input.grad is not None:
            upstream_grad = StageGradients(grad=saved_input.grad.clone())
            return upstream_grad

        return None

    def compute_grad_norm_sq(self) -> float:
        """Compute sum of squared gradient norms for all parameters."""
        norm_sq = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                norm_sq += param.grad.data.float().norm(2).item() ** 2
        return norm_sq

    def optimizer_step(self, global_grad_norm: float | None = None, max_norm: float = 1.0) -> None:
        """Step the optimizer with optional gradient clipping, then zero grad."""
        if self.optimizer is None:
            return

        if global_grad_norm is not None and global_grad_norm > 0:
            clip_coeff = min(1.0, max_norm / (global_grad_norm + 1e-6))
            if clip_coeff < 1.0:
                for param in self.model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coeff)

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def zero_grad(self) -> None:
        """Zero out gradients."""
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        else:
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad = None
