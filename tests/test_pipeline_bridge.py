"""UC2: Three-stage pipeline with BridgeTrainer tests.

CPU tests:
  - BridgeProjection forward/backward shape correctness
  - trainer_registry accepts "bridge" component
  - resolve_trainer("bridge", "native", "generic") returns BridgeTrainer

GPU integration test:
  - Run train_pipeline.py with pipeline_bridge_sample config as subprocess
  - Verify losses are finite, non-zero, and change across iterations
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from python.ray.bridge import BridgeProjection
from python.trainer_registry import _normalize_component, resolve_trainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
BRIDGE_CONFIG = "pipeline_bridge_sample"
NUM_ITERS = 3
NUM_GPUS_REQUIRED = 4


# ── CPU unit tests ──


@pytest.mark.cpu_only
class TestBridgeProjectionCPU:
    """Unit tests for BridgeProjection MLP (no GPU required)."""

    def test_forward_shape(self):
        """Forward output has correct shape."""
        model = BridgeProjection(input_dim=128, output_dim=64, hidden_dim=96)
        x = torch.randn(2, 10, 128)  # [batch, seq, input_dim]
        y = model(x)
        assert y.shape == (2, 10, 64), f"Expected (2, 10, 64), got {y.shape}"

    def test_forward_shape_default_hidden(self):
        """Default hidden_dim = (input + output) // 2."""
        model = BridgeProjection(input_dim=100, output_dim=200)
        assert model.fc1.in_features == 100
        assert model.fc1.out_features == 150  # (100 + 200) // 2
        assert model.fc2.in_features == 150
        assert model.fc2.out_features == 200

    def test_backward_produces_grads(self):
        """Backward pass produces gradients on all parameters."""
        model = BridgeProjection(input_dim=32, output_dim=16, hidden_dim=24)
        x = torch.randn(1, 5, 32, requires_grad=True)
        y = model(x)
        loss = y.sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.shape == param.shape

        assert x.grad is not None, "No gradient for input"
        assert x.grad.shape == x.shape

    def test_backward_shape_matches_input(self):
        """Gradient w.r.t. input has same shape as input."""
        model = BridgeProjection(input_dim=64, output_dim=64)
        x = torch.randn(4, 20, 64, requires_grad=True)
        y = model(x)
        grad_output = torch.randn_like(y)
        y.backward(gradient=grad_output)
        assert x.grad.shape == (4, 20, 64)


@pytest.mark.cpu_only
class TestBridgeRegistryCPU:
    """Unit tests for trainer_registry bridge support."""

    def test_normalize_component_bridge(self):
        """_normalize_component accepts 'bridge'."""
        assert _normalize_component("bridge") == "bridge"
        assert _normalize_component("Bridge") == "bridge"
        assert _normalize_component("  BRIDGE  ") == "bridge"

    def test_resolve_trainer_bridge_native(self):
        """resolve_trainer returns BridgeTrainer for bridge/native/generic."""
        from python.ray.bridge import BridgeTrainer

        trainer_cls, init_kwargs = resolve_trainer("bridge", "native", "generic")
        assert trainer_cls is BridgeTrainer
        assert init_kwargs == {}

    def test_resolve_trainer_bridge_any_model_type(self):
        """Bridge with native engine works regardless of model_type."""
        from python.ray.bridge import BridgeTrainer

        trainer_cls, _ = resolve_trainer("bridge", "native", "some_custom_model")
        assert trainer_cls is BridgeTrainer

    def test_resolve_trainer_bridge_unsupported_engine(self):
        """Bridge with non-native engine raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported trainer combination"):
            resolve_trainer("bridge", "deepspeed", "generic")


# ── GPU integration test ──


def _check_preconditions():
    """Check that GPUs, model weights, and data are available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.device_count() < NUM_GPUS_REQUIRED:
        pytest.skip(f"At least {NUM_GPUS_REQUIRED} GPUs required, found {torch.cuda.device_count()}")
    required_paths = [
        "/mnt/local_storage/laion/laion_pop_train.jsonl",
        "/mnt/local_storage/laion/images",
    ]
    for path in required_paths:
        if not os.path.exists(path):
            pytest.skip(f"Required path not found: {path}")
    hf_cache = "/mnt/local_storage/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct"
    if not os.path.exists(hf_cache):
        pytest.skip(f"Model weights not found at {hf_cache}")


def _parse_losses(output: str) -> list[float]:
    """Extract per-iteration loss values from training log output."""
    pattern = r"Iter\s+\d+/\d+\s+\([^)]+\)\s+-\s+Loss:\s+([\d.]+)"
    return [float(m) for m in re.findall(pattern, output)]


@pytest.mark.gpu
@pytest.mark.integration
class TestPipelineBridgeGPU:
    """Integration test: three-stage pipeline with real Qwen2.5-VL model."""

    def test_bridge_pipeline_smoke(self):
        """Three-stage pipeline (vision→bridge→text) produces finite, changing loss."""
        _check_preconditions()

        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        env["HF_HOME"] = "/mnt/local_storage/huggingface"
        env.setdefault("HYDRA_FULL_ERROR", "1")

        cmd = [
            sys.executable,
            "-m",
            "python.train_pipeline",
            f"--config-path={CONFIGS_DIR}",
            f"--config-name={BRIDGE_CONFIG}",
            f"training.num_iterations={NUM_ITERS}",
            "training.num_epochs=1",
            "training.warmup_steps=0",
            "training.no_checkpoint=true",
            "training.log_interval=1",
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=1200,
        )

        combined = result.stdout + "\n" + result.stderr
        if result.returncode != 0:
            print("=== STDOUT ===")
            print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
            print("=== STDERR ===")
            print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
            pytest.fail(f"train_pipeline exited with code {result.returncode}")

        losses = _parse_losses(combined)

        assert len(losses) >= NUM_ITERS, (
            f"Expected at least {NUM_ITERS} loss values, got {len(losses)}. "
            f"Check training output for errors."
        )

        for i, loss in enumerate(losses):
            assert loss == loss, f"Loss is NaN at iteration {i}"
            assert loss < float("inf"), f"Loss is inf at iteration {i}"
            assert loss > 0, f"Loss is non-positive at iteration {i}: {loss}"

        unique_losses = set(f"{l:.4f}" for l in losses)
        assert len(unique_losses) > 1, f"Loss is stuck at {losses[0]:.4f} across all iterations"

        print(f"Bridge pipeline losses: {[f'{l:.4f}' for l in losses]}")
