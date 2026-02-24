"""Canonical trainer-construction path for pipeline stages."""

from __future__ import annotations

import logging
from typing import Any

from .native_trainer import StageTrainer
from .stage import Stage

logger = logging.getLogger(__name__)


class StageFactoryError(Exception):
    """Raised when a stage trainer cannot be constructed."""

    def __init__(self, stage_name: str, resolver_key: str | None, cause: Exception | None = None):
        self.stage_name = stage_name
        self.resolver_key = resolver_key
        self.cause = cause
        msg = f"Failed to create trainer for stage '{stage_name}'"
        if resolver_key:
            msg += f" (resolver_key='{resolver_key}')"
        if cause:
            msg += f": {cause}"
        super().__init__(msg)


def create_stage_trainer(stage_cfg: Stage, rank: int, **kwargs) -> StageTrainer:
    """Create a StageTrainer from a Stage config.

    For Python API stages (model_fn provided directly):
      - Calls model_fn() to get the nn.Module
      - Wraps in StageTrainer

    For YAML stages (trainer_resolver_key):
      - Looks up the trainer class/model_fn from the registry
      - Constructs the trainer

    Args:
        stage_cfg: Stage configuration
        rank: Actor rank
        **kwargs: Additional keyword arguments (device, optimizer_fn, loss_fn, etc.)

    Returns:
        A configured StageTrainer instance.

    Raises:
        StageFactoryError: If the trainer cannot be created.
    """
    try:
        model_fn = stage_cfg.model_fn
        if model_fn is None and stage_cfg.trainer_resolver_key:
            from .config_loader import get_registered_trainer

            reg = get_registered_trainer(stage_cfg.trainer_resolver_key)
            model_fn = reg.get("model_fn")

        if model_fn is None:
            raise StageFactoryError(stage_cfg.name, stage_cfg.trainer_resolver_key, ValueError("No model_fn provided"))

        model = model_fn()
        device = kwargs.get("device")
        if device is not None:
            model = model.to(device)

        optimizer_fn = kwargs.get("optimizer_fn")
        optimizer = optimizer_fn(model.parameters()) if optimizer_fn else None

        loss_fn = kwargs.get("loss_fn") if stage_cfg.is_terminal else None

        return StageTrainer(
            stage_name=stage_cfg.name,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            is_terminal=stage_cfg.is_terminal,
            device=device,
        )

    except StageFactoryError:
        raise
    except Exception as e:
        raise StageFactoryError(stage_cfg.name, stage_cfg.trainer_resolver_key, e) from e
