import json

import pytest

from scripts.qwen3_vl_dense_audit import assert_non_layer_fields_unchanged, _local_weight_cache_status


class Obj:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _config(num_layers=4, hidden_size=16, patch_size=14):
    return Obj(
        text_config=Obj(
            num_hidden_layers=num_layers,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_key_value_heads=4,
            intermediate_size=64,
            vocab_size=32000,
            rope_theta=1000000,
        ),
        vision_config=Obj(
            depth=num_layers,
            hidden_size=hidden_size,
            out_hidden_size=hidden_size,
            num_heads=4,
            patch_size=patch_size,
            temporal_patch_size=2,
            spatial_merge_size=2,
        ),
    )


@pytest.mark.cpu_only
def test_non_layer_guard_allows_layer_only_changes():
    original = _config(num_layers=32)
    effective = _config(num_layers=8)

    assert_non_layer_fields_unchanged(
        original,
        effective,
        allowed_layer_fields={"text_config.num_hidden_layers", "vision_config.depth"},
    )


@pytest.mark.cpu_only
def test_non_layer_guard_rejects_hidden_size_change():
    original = _config(hidden_size=16)
    effective = _config(hidden_size=32)

    with pytest.raises(ValueError, match="hidden_size"):
        assert_non_layer_fields_unchanged(
            original,
            effective,
            allowed_layer_fields={"text_config.num_hidden_layers", "vision_config.depth"},
        )


@pytest.mark.cpu_only
def test_local_weight_cache_requires_all_indexed_safetensor_shards(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf"
    snapshot = cache_root / "hub/models--Unit--Test-Model/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 12},
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"partial")

    monkeypatch.setenv("HF_HOME", str(cache_root))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope"))

    status = _local_weight_cache_status("Unit/Test-Model")

    assert status["exists"] is True
    assert status["complete"] is False
    assert status["partial_paths"] == [str(snapshot)]
    partial_status = next(item for item in status["candidate_statuses"] if item["path"] == str(snapshot))
    assert partial_status["required_count"] == 2
    assert partial_status["present_count"] == 1
    assert partial_status["expected_total_bytes"] == 12
    assert "1/2 shards present" in status["summary"]

    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"complete")
    status = _local_weight_cache_status("Unit/Test-Model")

    assert status["complete"] is True
    assert status["complete_paths"] == [str(snapshot)]
    complete_status = next(item for item in status["candidate_statuses"] if item["path"] == str(snapshot))
    assert complete_status["present_count"] == 2
    assert complete_status["present_bytes"] == len(b"partial") + len(b"complete")


@pytest.mark.cpu_only
def test_local_weight_cache_checks_direct_hf_hub_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "hf-hub-cache"
    snapshot = cache_root / "models--Unit--Direct-Model/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 7},
                "weight_map": {
                    "a": "model-00001-of-00001.safetensors",
                },
            }
        ),
        encoding="utf-8",
    )
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    monkeypatch.setenv("MODELSCOPE_CACHE", str(tmp_path / "modelscope"))

    status = _local_weight_cache_status("Unit/Direct-Model")

    assert status["complete"] is True
    assert status["complete_paths"] == [str(snapshot)]
    complete_status = next(item for item in status["candidate_statuses"] if item["path"] == str(snapshot))
    assert complete_status["required_count"] == 1
    assert complete_status["present_count"] == 1
    assert complete_status["expected_total_bytes"] == 7
