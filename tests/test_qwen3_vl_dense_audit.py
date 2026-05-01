import pytest

from scripts.qwen3_vl_dense_audit import assert_non_layer_fields_unchanged


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
