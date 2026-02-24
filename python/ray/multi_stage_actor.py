"""MultiStageActor: a Ray actor hosting multiple pipeline stage trainers (design §6.2).

Each actor in an ActorGroup holds one or more StageTrainer instances (one per
stage placed on this resource set). The actor dispatches forward/backward/optimizer
calls to the named stage trainer.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import ray
import torch
import torch.nn as nn

from ..ray.payloads import StageGradients, StageOutputs

logger = logging.getLogger(__name__)


class MultiStageActor:
    """A Ray actor that hosts one or more pipeline stage trainers.

    Unlike RayActor, this does not set up torch.distributed — the pipeline
    framework handles placement and transport separately.

    Model construction happens via build_model_from_state_dict(), which receives:
    - A model_cls and constructor kwargs (must be serializable)
    - An initial state_dict (transferred via Ray ObjectRef)
    - Optimizer and loss configuration
    """

    def __init__(self, config: dict, rank: int):
        self.rank = rank
        self.config = config
        self._trainers: dict[str, StageTrainer] = {}

    def build_model_from_state_dict(
        self,
        stage_name: str,
        model_cls: type,
        model_kwargs: dict,
        state_dict: dict | None = None,
        is_terminal: bool = False,
        optimizer_cls: type | None = None,
        optimizer_kwargs: dict | None = None,
        loss_cls: type | None = None,
        loss_kwargs: dict | None = None,
        engine: str = "native",
        ds_config: dict | None = None,
    ) -> bool:
        """Build a stage trainer from a model class + state dict.

        All arguments must be serializable (no lambdas or closures).
        The model is constructed as model_cls(**model_kwargs), then state_dict is loaded.

        Args:
            stage_name: Name of the stage.
            model_cls: The nn.Module class (e.g., nn.Linear).
            model_kwargs: Constructor kwargs (e.g., {"in_features": 32, "out_features": 64}).
            state_dict: Optional initial state dict.
            is_terminal: Whether this is a terminal (loss-computing) stage.
            optimizer_cls: Optimizer class (e.g., torch.optim.Adam).
            optimizer_kwargs: Optimizer kwargs excluding params (e.g., {"lr": 0.01}).
            loss_cls: Loss function class for terminal stages (e.g., nn.CrossEntropyLoss).
            loss_kwargs: Loss function kwargs.
            engine: "native" or "deepspeed".
            ds_config: DeepSpeed config dict (required when engine="deepspeed").

        Returns:
            True on success.
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Construct model
        model = model_cls(**model_kwargs).to(device)
        if state_dict is not None:
            model.load_state_dict(state_dict)

        # Construct loss
        loss_fn = None
        if loss_cls is not None and is_terminal:
            loss_fn = loss_cls(**(loss_kwargs or {}))

        if engine == "deepspeed":
            from ..pipeline.ds_trainer import DeepSpeedStageTrainer

            if ds_config is None:
                raise ValueError(f"Stage '{stage_name}': engine='deepspeed' requires ds_config")
            trainer = DeepSpeedStageTrainer(
                stage_name=stage_name,
                model=model,
                ds_config=ds_config,
                loss_fn=loss_fn,
                is_terminal=is_terminal,
                device=device,
            )
        else:
            from ..pipeline.native_trainer import StageTrainer

            # Construct optimizer (native only — DeepSpeed manages its own)
            optimizer = None
            if optimizer_cls is not None:
                opt_kwargs = optimizer_kwargs or {}
                optimizer = optimizer_cls(model.parameters(), **opt_kwargs)

            trainer = StageTrainer(
                stage_name=stage_name,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                is_terminal=is_terminal,
                device=device,
            )

        self._trainers[stage_name] = trainer
        logger.info(f"[r{self.rank}] Built {engine} model for stage '{stage_name}' on {device}")
        return True

    def forward_step(
        self,
        stage_name: str,
        inputs: StageOutputs | None = None,
        labels: Any = None,
    ) -> StageOutputs:
        """Run forward for a named stage.

        For cross-GPU transport, call with .options(tensor_transport="nccl").remote()
        so the returned StageOutputs tensor is delivered via RDT/NCCL.
        """
        trainer = self._get_trainer(stage_name)
        return trainer.forward_step(inputs, labels=labels)

    def backward_step(
        self,
        stage_name: str,
        downstream_grad: StageGradients | None = None,
    ) -> StageGradients | None:
        """Run backward for a named stage.

        For cross-GPU transport, call with .options(tensor_transport="nccl").remote()
        so the returned StageGradients tensor is delivered via RDT/NCCL.
        """
        trainer = self._get_trainer(stage_name)
        return trainer.backward_step(downstream_grad)

    def optimizer_step(
        self,
        stage_name: str,
        global_grad_norm: float | None = None,
        max_norm: float = 1.0,
    ) -> None:
        """Step optimizer for a named stage."""
        trainer = self._get_trainer(stage_name)
        trainer.optimizer_step(global_grad_norm=global_grad_norm, max_norm=max_norm)

    def compute_grad_norm_sq(self, stage_name: str) -> float:
        """Compute sum-of-squares gradient norm for a named stage."""
        trainer = self._get_trainer(stage_name)
        return trainer.compute_grad_norm_sq()

    def zero_grad(self, stage_name: str) -> None:
        """Zero gradients for a named stage."""
        trainer = self._get_trainer(stage_name)
        trainer.zero_grad()

    def get_last_loss(self, stage_name: str) -> float | None:
        """Get the loss from the last forward of a terminal stage.

        Returns the scalar loss value (safe to ray.get without CUDA deserialization).
        """
        trainer = self._get_trainer(stage_name)
        return getattr(trainer, "_last_loss_value", None)

    def get_last_forward_meta(self, stage_name: str) -> dict:
        """Get metadata from the last forward (scalars only, no CUDA tensors)."""
        trainer = self._get_trainer(stage_name)
        return getattr(trainer, "_last_forward_meta", {})

    def run_full_backward(self, stage_names_reversed: list[str], stage_successors: dict[str, list[str]]) -> bool:
        """Run backward for all stages within this actor, passing gradients locally.

        Args:
            stage_names_reversed: Stage names in reverse topological order.
            stage_successors: Map from stage name to its successor stage names.

        Returns:
            True when all backward ops complete (safe scalar for ray.get).
        """
        grad_map: dict[str, "StageGradients | None"] = {}

        for name in stage_names_reversed:
            trainer = self._trainers[name]
            succs = stage_successors.get(name, [])

            if not succs:
                # Terminal stage
                upstream_grad = trainer.backward_step(downstream_grad=None)
            else:
                # Get gradient from successor
                succ_name = succs[0]
                downstream_grad = grad_map.get(succ_name)
                upstream_grad = trainer.backward_step(downstream_grad=downstream_grad)

            grad_map[name] = upstream_grad

        return True

    def sum_gradients(self, grads: list[StageGradients]) -> StageGradients:
        """Sum multiple gradient payloads into one (used for M:N backward aggregation).

        When a source stage maps to multiple destination actors, the backward pass
        produces one gradient per destination actor. This method aggregates them
        by summing on-device before passing to the source stage's backward_step.
        """
        if len(grads) == 1:
            return grads[0]
        summed = grads[0].grad.clone()
        for g in grads[1:]:
            summed = summed + g.grad
        return StageGradients(grad=summed, meta=grads[0].meta)

    def _get_trainer(self, stage_name: str) -> StageTrainer:
        if stage_name not in self._trainers:
            raise ValueError(f"[r{self.rank}] Stage '{stage_name}' not built. Call build_model first.")
        return self._trainers[stage_name]
