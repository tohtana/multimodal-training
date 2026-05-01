import os
import site
import sys

import pytest

from python.ray.utils import prepare_runtime_environment


@pytest.mark.cpu_only
def test_prepare_runtime_environment_includes_active_venv_site_packages(monkeypatch, tmp_path):
    venv_site = "/mnt/user_storage/venvs/task/lib/python3.12/site-packages"
    conda_site = "/home/ray/anaconda3/lib/python3.12/site-packages"
    unrelated_site = "/tmp/other/lib/python3.12/site-packages"

    monkeypatch.setattr(sys, "prefix", "/mnt/user_storage/venvs/task")
    monkeypatch.setattr(sys, "exec_prefix", "/mnt/user_storage/venvs/task")
    monkeypatch.setattr(site, "getsitepackages", lambda: [venv_site, conda_site, unrelated_site])
    monkeypatch.setenv("CONDA_PREFIX", "/home/ray/anaconda3")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(["relative-lib", "/extra"]))
    monkeypatch.setenv("RAY_TRAIN_LOG_FILE", "/tmp/train.log")

    env_vars = prepare_runtime_environment()
    pythonpath = env_vars["PYTHONPATH"].split(os.pathsep)

    assert pythonpath[:2] == [venv_site, conda_site]
    assert str(tmp_path / "relative-lib") in pythonpath
    assert "/extra" in pythonpath
    assert unrelated_site not in pythonpath
