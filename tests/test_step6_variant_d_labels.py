"""CPU-only checks for variant D split label/path normalization."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.attn_moe_overlap.step6_mps_overlap as step6


@pytest.mark.cpu_only
def test_variant_d_behavioral_check_uses_canonical_dflt_labels(monkeypatch, tmp_path):
    results = {
        "100:100": {"status": "ok"},
        "90:dflt": {"status": "ok"},
    }
    splits = [(90, None), (100, 100)]
    seen_paths: list[str] = []

    def _fake_load(path: str):
        seen_paths.append(path)
        if path.endswith("variant_d/attn100_moe100/mps_overlap_metrics.json"):
            return {"overlap_pct": 10.0, "attn_full_sm_pct": 10.0, "moe_full_sm_pct": 10.0}
        if path.endswith("variant_d/attn90_moedflt/mps_overlap_metrics.json"):
            return {"overlap_pct": 20.5, "attn_full_sm_pct": 10.0, "moe_full_sm_pct": 10.0}
        return None

    monkeypatch.setattr(step6, "load_overlap_metrics", _fake_load)

    status = step6.run_variant_d_behavioral_check(results, str(tmp_path), splits)
    assert status == "effective"
    assert any("attn90_moedflt" in p for p in seen_paths)


@pytest.mark.cpu_only
def test_variant_d_behavioral_check_falls_back_to_legacy_none_trace(monkeypatch, tmp_path):
    results = {
        "100:100": {"status": "ok"},
        "dflt:dflt": {"status": "ok"},
    }
    splits = [(None, None), (100, 100)]
    seen_paths: list[str] = []

    def _fake_load(path: str):
        seen_paths.append(path)
        if path.endswith("variant_d/attn100_moe100/mps_overlap_metrics.json"):
            return {"overlap_pct": 60.0, "attn_full_sm_pct": 10.0, "moe_full_sm_pct": 10.0}
        if path.endswith("variant_d/attnNone_moeNone/mps_overlap_metrics.json"):
            return {"overlap_pct": 50.0, "attn_full_sm_pct": 16.0, "moe_full_sm_pct": 10.0}
        return None

    monkeypatch.setattr(step6, "load_overlap_metrics", _fake_load)

    status = step6.run_variant_d_behavioral_check(results, str(tmp_path), splits)
    assert status == "effective"
    assert any("attnNone_moeNone" in p for p in seen_paths)
