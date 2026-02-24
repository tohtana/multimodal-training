"""Local pipeline runner: executes N-stage pipeline in a single process (no Ray).

Used for correctness testing and as the reference implementation for gradient matching.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable

import torch

from ..ray.payloads import StageGradients, StageOutputs
from .dag import PipelineDAG
from .native_trainer import StageTrainer
from .scheduler import SequentialScheduler
from .stage import Pipeline

logger = logging.getLogger(__name__)


class NativeRunner:
    """Runs a pipeline locally (no Ray) with native PyTorch.

    Each stage's model is instantiated in the current process.
    Forward/backward follow topological order via SequentialScheduler.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        trainers: dict[str, StageTrainer],
        dataloader_fn: Callable | None = None,
        labels_fn: Callable | None = None,
    ):
        """
        Args:
            pipeline: The Pipeline config.
            trainers: Map from stage name → StageTrainer.
            dataloader_fn: Callable that returns (input_tensor, labels) for source stages.
            labels_fn: Callable that returns labels for terminal stages (if not provided via dataloader).
        """
        self.pipeline = pipeline
        self.trainers = trainers
        self.dag = PipelineDAG(pipeline)
        self.topo_order = self.dag.topological_sort()
        self.scheduler = SequentialScheduler()
        self.dataloader_fn = dataloader_fn
        self.labels_fn = labels_fn

    def run_iteration(
        self,
        data: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        max_norm: float | None = None,
    ) -> dict[str, Any]:
        """Run one full iteration: forward all stages, backward all stages, optimizer step.

        Args:
            data: Input tensor for source stages.
            labels: Labels for terminal stages.
            max_norm: Max gradient norm for clipping (None = no clipping).

        Returns:
            Dict with 'loss', 'global_grad_norm', per-stage info.
        """
        schedule = self.scheduler.generate_schedule(self.topo_order, num_microbatches=1)

        # Storage for inter-stage activations and gradients
        stage_outputs: dict[str, StageOutputs] = {}
        stage_grads: dict[str, StageGradients | None] = {}
        loss_value = None

        for step in schedule:
            trainer = self.trainers[step.stage_name]

            if step.op.value == "forward":
                # Determine inputs
                preds = self.dag.predecessors(step.stage_name)
                if not preds:
                    # Source stage: use provided data
                    assert data is not None, "Source stage requires data"
                    inputs = StageOutputs(activations=data)
                else:
                    # Non-source stage: get outputs from predecessor(s)
                    # For single predecessor (linear pipeline)
                    assert len(preds) == 1, "Fan-in not yet implemented in native_runner"
                    pred_output = stage_outputs[preds[0]]
                    inputs = StageOutputs(activations=pred_output.activations)

                # Forward
                stage_labels = labels if trainer.is_terminal else None
                output = trainer.forward_step(inputs, labels=stage_labels)
                stage_outputs[step.stage_name] = output

                if trainer.is_terminal and "loss" in output.meta:
                    loss_value = output.meta["loss"]

            elif step.op.value == "backward":
                # Determine downstream gradient
                succs = self.dag.successors(step.stage_name)
                if not succs:
                    # Terminal stage: backward from loss
                    upstream_grad = trainer.backward_step(downstream_grad=None)
                else:
                    # Non-terminal stage: get gradient from successor
                    assert len(succs) == 1, "Fan-out not yet implemented in native_runner"
                    downstream_grad = stage_grads.get(succs[0])
                    upstream_grad = trainer.backward_step(downstream_grad=downstream_grad)

                stage_grads[step.stage_name] = upstream_grad

        # Compute global gradient norm
        total_norm_sq = sum(t.compute_grad_norm_sq() for t in self.trainers.values())
        global_grad_norm = math.sqrt(total_norm_sq)

        # Optimizer step for all stages
        for name in self.topo_order:
            self.trainers[name].optimizer_step(
                global_grad_norm=global_grad_norm if max_norm is not None else None,
                max_norm=max_norm or 1.0,
            )

        return {
            "loss": loss_value,
            "global_grad_norm": global_grad_norm,
        }
