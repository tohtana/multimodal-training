"""MPS (Multi-Process Service) lifecycle utilities for single-GPU compute overlap."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time

import torch

logger = logging.getLogger(__name__)


def check_mps_support() -> bool:
    """Check if NVIDIA MPS is available (binary exists and GPU compute >= 7.0)."""
    if shutil.which("nvidia-cuda-mps-control") is None:
        return False
    if not torch.cuda.is_available():
        return False
    cap = torch.cuda.get_device_capability(0)
    return cap[0] >= 7


def get_gpu_arch() -> str:
    """Return GPU architecture name based on compute capability."""
    if not torch.cuda.is_available():
        return "unknown"
    cap = torch.cuda.get_device_capability(0)
    major = cap[0]
    arch_map = {
        7: "Volta/Turing",
        8: "Ampere",
        9: "Hopper",
        10: "Blackwell",
    }
    return arch_map.get(major, f"compute_{major}.{cap[1]}")


class MPSContext:
    """Context manager for MPS daemon lifecycle.

    Uses PID-scoped pipe/log directories so multiple users can run concurrently
    on the same host without sharing MPS state.
    """

    def __init__(
        self,
        gpu_id: int = 0,
        pipe_dir: str | None = None,
        log_dir: str | None = None,
        active_thread_pct: int | None = None,
    ):
        self.gpu_id = gpu_id
        self.pipe_dir = pipe_dir or f"/tmp/mm-mps-{os.getuid()}-{os.getpid()}"
        self.log_dir = log_dir or f"/tmp/mm-mps-log-{os.getuid()}-{os.getpid()}"
        self.active_thread_pct = active_thread_pct
        self.started_daemon = False
        self._saved_env: dict[str, str | None] = {}

    def __enter__(self) -> MPSContext:
        if shutil.which("nvidia-cuda-mps-control") is None:
            raise RuntimeError("nvidia-cuda-mps-control not found in PATH")

        # Fail fast if pipe_dir already has a live daemon
        control_socket = os.path.join(self.pipe_dir, "control")
        if os.path.exists(control_socket):
            raise RuntimeError(
                f"MPS pipe directory '{self.pipe_dir}' already has an active daemon (control socket exists). "
                "Use a different pipe_dir or stop the existing daemon first."
            )

        # Create directories
        os.makedirs(self.pipe_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save and set env vars
        env_vars = {
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
        }
        if self.active_thread_pct is not None:
            env_vars["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.active_thread_pct)

        for key, value in env_vars.items():
            self._saved_env[key] = os.environ.get(key)
            os.environ[key] = value

        # Start daemon
        result = subprocess.run(
            ["nvidia-cuda-mps-control", "-d"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            self._restore_env()
            raise RuntimeError(f"Failed to start MPS daemon: {result.stderr.strip()}")

        # Verify daemon started (poll for control socket)
        for _ in range(20):
            if os.path.exists(control_socket):
                break
            time.sleep(0.1)
        else:
            self._restore_env()
            raise RuntimeError("MPS daemon started but control socket did not appear within 2s")

        self.started_daemon = True
        logger.info(f"MPS daemon started (pipe={self.pipe_dir}, gpu={self.gpu_id})")
        return self

    def __exit__(self, *exc) -> None:
        if self.started_daemon:
            try:
                env = os.environ.copy()
                env["CUDA_MPS_PIPE_DIRECTORY"] = self.pipe_dir
                result = subprocess.run(
                    ["nvidia-cuda-mps-control"],
                    input="quit\n",
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=10,
                )
                if result.returncode != 0:
                    logger.warning(f"MPS quit returned code {result.returncode}: {result.stderr.strip()}")
            except subprocess.TimeoutExpired:
                logger.warning("MPS daemon quit timed out")
            except Exception as e:
                logger.warning(f"Error stopping MPS daemon: {e}")

            # Wait for daemon to exit
            time.sleep(0.5)

            # Clean up directories
            for d in (self.pipe_dir, self.log_dir):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass

            self.started_daemon = False
            logger.info("MPS daemon stopped and cleaned up")

        self._restore_env()

    def get_env_vars(self) -> dict[str, str]:
        """Return env vars for Ray actor runtime_env."""
        env = {
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
        }
        if self.active_thread_pct is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(self.active_thread_pct)
        return env

    def get_env_vars_for_stage(self, thread_pct: int | None = None) -> dict[str, str]:
        """Return env vars for an MPS client with a specific thread percentage.

        Args:
            thread_pct: CUDA_MPS_ACTIVE_THREAD_PERCENTAGE value (1-100),
                or None to leave the variable unset (unconstrained).
        """
        env = {
            "CUDA_MPS_PIPE_DIRECTORY": self.pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": self.log_dir,
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
        }
        if thread_pct is not None:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(thread_pct)
        return env

    def _restore_env(self) -> None:
        """Restore environment variables to their pre-context values."""
        for key, old_value in self._saved_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
        self._saved_env = {}
