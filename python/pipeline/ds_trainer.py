"""DeepSpeed-aware StageTrainer for pipeline execution (design §6.2, Milestone 4).

Wraps an nn.Module with deepspeed.initialize() and provides the same
forward_step / backward_step / optimizer_step interface as StageTrainer.
"""

from __future__ import annotations

import logging
import os
import socket
from typing import Any, Callable

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from ..ray.payloads import StageGradients, StageOutputs

logger = logging.getLogger(__name__)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class DeepSpeedStageTrainer:
    """Wraps an nn.Module with DeepSpeed engine for pipeline stage execution.

    Same interface as StageTrainer:
      1. forward_step(inputs) -> StageOutputs
      2. backward_step(grad)  -> StageGradients or None
      3. optimizer_step(global_grad_norm)
    """

    def __init__(
        self,
        stage_name: str,
        model: nn.Module,
        ds_config: dict,
        loss_fn: Callable | None = None,
        is_terminal: bool = False,
        device: torch.device | None = None,
    ):
        self.stage_name = stage_name
        self.is_terminal = is_terminal
        self.loss_fn = loss_fn
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize torch.distributed for DeepSpeed (single-rank)
        self._ensure_distributed()

        # Initialize DeepSpeed engine
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=ds_config,
            dist_init_required=False,
        )
        self.engine = model_engine
        self.optimizer = optimizer

        # Saved between forward and backward
        self._saved_output: torch.Tensor | None = None
        self._saved_input: torch.Tensor | None = None
        self._loss: torch.Tensor | None = None
        self._last_loss_value: float | None = None
        self._last_forward_meta: dict = {}

    @staticmethod
    def _ensure_distributed():
        """Initialize a single-rank process group if not already initialized."""
        if dist.is_initialized():
            return
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)

    def forward_step(self, inputs: StageOutputs | None = None, labels: Any = None) -> StageOutputs:
        """Run forward through the DeepSpeed engine."""
        if inputs is not None:
            x = inputs.activations
        else:
            raise ValueError(f"Stage '{self.stage_name}': forward_step requires inputs")

        if isinstance(x, torch.Tensor):
            x = x.to(self.device).detach().requires_grad_(True)
            self._saved_input = x
        else:
            self._saved_input = None

        output = self.engine(x)

        if self.is_terminal:
            if self.loss_fn is None:
                raise ValueError(f"Terminal stage '{self.stage_name}' requires a loss_fn")
            if labels is None:
                raise ValueError(f"Terminal stage '{self.stage_name}' requires labels")
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device)
            loss = self.loss_fn(output, labels)
            self._loss = loss
            self._last_loss_value = loss.item()
            self._last_forward_meta = {"loss": self._last_loss_value}
            self._saved_output = None
            return StageOutputs(activations=output, meta={"loss": self._last_loss_value})
        else:
            self._saved_output = output
            self._loss = None
            return StageOutputs(activations=output)

    def backward_step(self, downstream_grad: StageGradients | None = None) -> StageGradients | None:
        """Run backward through the DeepSpeed engine."""
        if self.is_terminal:
            assert self._loss is not None, f"Stage '{self.stage_name}': backward called before forward"
            self.engine.backward(self._loss)
            self._loss = None
        else:
            assert self._saved_output is not None, f"Stage '{self.stage_name}': backward called before forward"
            assert downstream_grad is not None, f"Non-terminal stage '{self.stage_name}' requires downstream_grad"
            grad = downstream_grad.grad
            if isinstance(grad, torch.Tensor):
                grad = grad.to(self.device)
            self._saved_output.backward(grad)
            self._saved_output = None

        # Return gradient w.r.t. this stage's input for upstream stages
        if self._saved_input is not None and self._saved_input.grad is not None:
            upstream_grad = StageGradients(grad=self._saved_input.grad.clone())
            self._saved_input = None
            return upstream_grad

        self._saved_input = None
        return None

    def compute_grad_norm_sq(self) -> float:
        """Compute sum of squared gradient norms for all parameters."""
        norm_sq = 0.0
        for param in self.engine.parameters():
            if param.grad is not None:
                norm_sq += param.grad.data.float().norm(2).item() ** 2
        return norm_sq

    def optimizer_step(self, global_grad_norm: float | None = None, max_norm: float = 1.0) -> None:
        """Step the DeepSpeed engine with optional gradient clipping, then zero grad."""
        if global_grad_norm is not None and global_grad_norm > 0:
            clip_coeff = min(1.0, max_norm / (global_grad_norm + 1e-6))
            if clip_coeff < 1.0:
                for param in self.engine.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(clip_coeff)

        self.engine.step()

    def zero_grad(self) -> None:
        """Zero out gradients."""
        self.engine.zero_grad()
