import pytest
import torch

from python.trainer_registry import resolve_trainer


pytestmark = [pytest.mark.cpu_only]


def _qwen3_trainer_class():
    from python.ray.vision import Qwen3VLVisionTrainer

    return Qwen3VLVisionTrainer.__ray_metadata__.modified_class


def _tiny_trainer_config():
    return {
        "seed": 1234,
        "model_name": "__tiny_qwen3_vl__",
        "model_type": "qwen3_vl",
        "engine": "deepspeed",
        "parallelism": "sequence",
        "dtype": "float32",
        "attention_backend": "sdpa",
        "activation_checkpointing": False,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "zero_stage": 0,
        "reduce_bucket_size": 1000000,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "warmup_ratio": 0.0,
        "warmup_steps": 0,
        "lr_scheduler_type": "constant",
        "num_iterations": 1,
        "vision_config_overrides": {
            "depth": 1,
            "hidden_size": 16,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "spatial_merge_size": 2,
            "temporal_patch_size": 1,
            "out_hidden_size": 24,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
    }


def test_registry_resolves_qwen3_dense_vision_deepspeed():
    trainer_cls, init_kwargs = resolve_trainer("vision", "deepspeed", "qwen3_vl", config={})

    from python.ray.vision import Qwen3VLVisionTrainer

    assert trainer_cls is Qwen3VLVisionTrainer
    assert init_kwargs == {}


def test_registry_does_not_claim_qwen3_moe_coverage():
    with pytest.raises(ValueError, match="Unsupported trainer combination"):
        resolve_trainer("vision", "deepspeed", "qwen3_moe_vl", config={})


def test_qwen3_dense_vision_trainer_tiny_construction_cpu():
    trainer = _qwen3_trainer_class()(_tiny_trainer_config(), rank=0)
    model_config = trainer._load_model_config("__tiny_qwen3_vl__")

    assert model_config.vision_config.__class__.__name__ == "Qwen3VLVisionConfig"
    assert model_config.vision_config.deepstack_visual_indexes == [0]

    model, projector = trainer._create_model_instance(model_config)
    assert projector is None
    assert model.__class__.__name__ == "Qwen3VLVisionModel"
    assert model.blocks[0].attn.qkv.__class__.__name__ == "Linear"
    assert model.blocks[0].attn.proj.__class__.__name__ == "Linear"
    assert model.merger.linear_fc1.__class__.__name__ == "Linear"

    pixel_values = torch.randn(16, 3 * 1 * 2 * 2)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    outputs = model(hidden_states=pixel_values, grid_thw=grid_thw)

    assert outputs.last_hidden_state.shape == (16, 16)
    assert outputs.pooler_output.shape == (4, 24)
    assert len(outputs.deepstack_features) == 1
    assert outputs.deepstack_features[0].shape == (4, 24)


def test_qwen3_dense_vision_trainer_model_forward_keeps_deepstack_cpu():
    config = _tiny_trainer_config()
    trainer = _qwen3_trainer_class()(config, rank=0)

    model_config = trainer._load_model_config("__tiny_qwen3_vl__")
    model, projector = trainer._create_model_instance(model_config)
    trainer.model = model
    trainer.projector = projector
    trainer.model.train()

    pixel_values = torch.randn(16, 3 * 1 * 2 * 2, requires_grad=True)
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long)
    outputs = trainer._model_forward({"pixel_values": pixel_values, "image_grid_thw": grid_thw})

    assert outputs.pooler_output.shape == (1, 4, 24)
    assert len(outputs.deepstack_features) == 1
    assert outputs.deepstack_features[0].shape == (1, 4, 24)

    grad = {
        "pooler_output": torch.randn_like(outputs.pooler_output),
        "deepstack_features": [torch.randn_like(outputs.deepstack_features[0])],
    }
    trainer._pending_outputs.append(outputs)
    trainer._apply_vision_backward(grad)

    assert pixel_values.grad is not None
    assert any(param.grad is not None for param in model.parameters())


def test_qwen3_sequence_setup_wraps_attention_without_renaming_qkv():
    from python.ray.qwen3_vision_ulysses import Qwen3VLUlyssesVisionAttention

    trainer = _qwen3_trainer_class()(_tiny_trainer_config(), rank=0)
    model_config = trainer._load_model_config("__tiny_qwen3_vl__")
    model, _ = trainer._create_model_instance(model_config)
    original_qkv = model.blocks[0].attn.qkv

    trainer._setup_sequence_parallel(model, sp_group=object())

    assert isinstance(model.blocks[0].attn, Qwen3VLUlyssesVisionAttention)
    assert model.blocks[0].attn.qkv is original_qkv
    assert model.blocks[0].attn.is_causal is False
    assert "q_proj" not in dict(model.named_modules())


def test_qwen3_ulysses_attention_requires_position_embeddings_cpu():
    from python.ray.qwen3_vision_ulysses import Qwen3VLUlyssesVisionAttention

    trainer = _qwen3_trainer_class()(_tiny_trainer_config(), rank=0)
    model_config = trainer._load_model_config("__tiny_qwen3_vl__")
    model, _ = trainer._create_model_instance(model_config)
    attn = Qwen3VLUlyssesVisionAttention(model.blocks[0].attn, process_group=None)

    hidden_states = torch.randn(4, model_config.vision_config.hidden_size)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(ValueError, match="position_embeddings"):
        attn(hidden_states, cu_seqlens=cu_seqlens)
