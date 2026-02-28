"""M6: VLM Migration correctness gate test.

Validates that the pipeline framework (VLMPipelineRunner + train_pipeline.py)
produces correct training behavior with real Qwen2.5-VL models, and that
loss values match the legacy train_ray.py path within tolerance.

Tests:
  1. Pipeline path runs correctly with real Qwen2.5-VL model (loss is finite).
  2. Parity: pipeline and legacy paths produce matching loss within 1e-2.

Requires:
  - 4+ GPUs (H100 80GB recommended)
  - Qwen2.5-VL-32B-Instruct model weights
  - LAION dataset at /mnt/local_storage/laion/
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
PIPELINE_CONFIG = "pipeline_sample"
NUM_ITERS = 5
NUM_GPUS_REQUIRED = 4

# Paths that must exist for the test to run
REQUIRED_PATHS = [
    "/mnt/local_storage/laion/laion_pop_train.jsonl",
    "/mnt/local_storage/laion/images",
]


def _check_preconditions():
    """Check that GPUs, model weights, and data are available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.device_count() < NUM_GPUS_REQUIRED:
        pytest.skip(f"At least {NUM_GPUS_REQUIRED} GPUs required, found {torch.cuda.device_count()}")
    for path in REQUIRED_PATHS:
        if not os.path.exists(path):
            pytest.skip(f"Required path not found: {path}")
    # Check model weights via HF cache
    hf_cache = "/mnt/local_storage/huggingface/hub/models--Qwen--Qwen2.5-VL-32B-Instruct"
    if not os.path.exists(hf_cache):
        pytest.skip(f"Model weights not found at {hf_cache}")


def _parse_losses(output: str) -> list[float]:
    """Extract per-iteration loss values from training log output."""
    # Matches: "Iter N/M (warmup|training) - Loss: X.XXXX"
    pattern = r"Iter\s+\d+/\d+\s+\([^)]+\)\s+-\s+Loss:\s+([\d.]+)"
    matches = re.findall(pattern, output)
    return [float(m) for m in matches]


def _run_training(module: str, config_name: str, extra_overrides: list[str] | None = None, timeout: int = 1200) -> str:
    """Run a training module as a subprocess and return combined stdout+stderr.

    Args:
        module: Python module to run (e.g. "python.train_pipeline").
        config_name: Hydra config name.
        extra_overrides: Additional Hydra CLI overrides.
        timeout: Subprocess timeout in seconds (default 20 min).

    Returns:
        Combined stdout and stderr as a string.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    # Use local HF cache
    env["HF_HOME"] = "/mnt/local_storage/huggingface"
    # Avoid Hydra multirun directory collisions
    env.setdefault("HYDRA_FULL_ERROR", "1")

    cmd = [
        sys.executable,
        "-m",
        module,
        f"--config-path={CONFIGS_DIR}",
        f"--config-name={config_name}",
        f"training.num_iterations={NUM_ITERS}",
        "training.num_epochs=1",
        "training.warmup_steps=0",
        "training.no_checkpoint=true",
        "training.log_interval=1",
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    combined = result.stdout + "\n" + result.stderr
    if result.returncode != 0:
        # Print full output for debugging
        print(f"=== {module} STDOUT ===")
        print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
        print(f"=== {module} STDERR ===")
        print(result.stderr[-5000:] if len(result.stderr) > 5000 else result.stderr)
        pytest.fail(f"{module} exited with code {result.returncode}")

    return combined


class TestPipelineVLM:
    """M6 VLM Migration tests using real Qwen2.5-VL model."""

    def test_pipeline_smoke(self):
        """Pipeline path runs correctly with real model: loss is finite and non-zero."""
        _check_preconditions()

        output = _run_training("python.train_pipeline", PIPELINE_CONFIG)
        losses = _parse_losses(output)

        assert len(losses) >= NUM_ITERS, (
            f"Expected at least {NUM_ITERS} loss values, got {len(losses)}. " f"Check training output for errors."
        )

        for i, loss in enumerate(losses):
            assert loss == loss, f"Loss is NaN at iteration {i}"  # NaN check
            assert loss < float("inf"), f"Loss is inf at iteration {i}"
            assert loss > 0, f"Loss is non-positive at iteration {i}: {loss}"

        # Loss should not be stuck at the same value
        unique_losses = set(f"{l:.4f}" for l in losses)
        assert len(unique_losses) > 1, f"Loss is stuck at {losses[0]:.4f} across all iterations"

        print(f"Pipeline losses: {[f'{l:.4f}' for l in losses]}")

    def test_parity_with_legacy(self):
        """Pipeline and legacy paths produce matching loss within tolerance.

        Runs both paths sequentially with the same config (seed, data, model),
        then compares per-iteration losses. The pipeline path uses VLMPipelineRunner
        (DAG-based scheduling) while the legacy path uses train_ray.py (hardcoded
        two-stage loop). Both use identical ActorGroup/Trainer/DeepSpeed machinery.
        """
        _check_preconditions()

        # --- Run legacy path first ---
        # train_ray.py can use the pipeline config — it reads vision/text/training/data/deepspeed
        # and ignores the pipeline section
        print("Running legacy path (train_ray.py)...")
        legacy_output = _run_training(
            "python.train_ray",
            PIPELINE_CONFIG,
            extra_overrides=["training.seed=42"],
        )
        legacy_losses = _parse_losses(legacy_output)

        assert len(legacy_losses) >= NUM_ITERS, f"Legacy path: expected {NUM_ITERS} losses, got {len(legacy_losses)}"

        # --- Run pipeline path ---
        print("Running pipeline path (train_pipeline.py)...")
        pipeline_output = _run_training(
            "python.train_pipeline",
            PIPELINE_CONFIG,
            extra_overrides=["training.seed=42"],
        )
        pipeline_losses = _parse_losses(pipeline_output)

        assert (
            len(pipeline_losses) >= NUM_ITERS
        ), f"Pipeline path: expected {NUM_ITERS} losses, got {len(pipeline_losses)}"

        # --- Compare losses ---
        print(f"Legacy   losses: {[f'{l:.6f}' for l in legacy_losses[:NUM_ITERS]]}")
        print(f"Pipeline losses: {[f'{l:.6f}' for l in pipeline_losses[:NUM_ITERS]]}")

        for i in range(min(NUM_ITERS, len(legacy_losses), len(pipeline_losses))):
            leg = legacy_losses[i]
            pip = pipeline_losses[i]
            # Relative difference
            if leg != 0:
                rel_diff = abs(pip - leg) / abs(leg)
            else:
                rel_diff = abs(pip - leg)

            # Tolerance: 1e-2 relative — accounts for Ray scheduling nondeterminism
            # and floating-point ordering differences
            assert rel_diff < 1e-2, (
                f"Iteration {i}: loss mismatch — legacy={leg:.6f}, pipeline={pip:.6f}, "
                f"rel_diff={rel_diff:.6f} (threshold=1e-2)"
            )

        print("Parity check PASSED: all losses match within 1e-2 relative tolerance")
