from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _ensure_local_deepspeed() -> None:
    """Prefer the repo DeepSpeed checkout and ensure HAS_TRITON exists."""
    deepspeed_root = REPO_ROOT / "DeepSpeed"
    if deepspeed_root.exists():
        sys.path.insert(0, str(deepspeed_root))

    try:
        import deepspeed  # noqa: WPS433
    except Exception:
        return

    if not hasattr(deepspeed, "HAS_TRITON"):
        deepspeed.HAS_TRITON = False


_ensure_local_deepspeed()


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "cpu_only: test does not require GPUs or distributed backends")
    config.addinivalue_line("markers", "gpu: test expects CUDA/NCCL and multi-GPU setup")
    config.addinivalue_line("markers", "integration: end-to-end or long-running scenario")
