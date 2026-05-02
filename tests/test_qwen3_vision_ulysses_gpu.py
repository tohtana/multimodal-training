import copy
import os

import pytest
import torch


pytestmark = [pytest.mark.gpu]


def _require_torchrun():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        pytest.skip("Run with torchrun so DeepSpeed sequence-parallel groups can be initialized")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")


def _init_sp_group():
    _require_torchrun()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    import deepspeed
    import deepspeed.runtime.sequence_parallel.parallel_state_sp as mpu
    import torch.distributed as dist

    if not dist.is_initialized():
        deepspeed.init_distributed()

    world_size = dist.get_world_size()
    if world_size not in {2, 4}:
        pytest.skip(f"Qwen3 vision Ulysses smoke expects SP=2 or SP=4, got WORLD_SIZE={world_size}")

    if getattr(mpu, "_SEQUENCE_PARALLEL_GROUP", None) is None:
        mpu.initialize_sequence_parallel(sequence_parallel_size=world_size)
    return mpu.get_sequence_parallel_group(), local_rank, dist.get_rank(), world_size


def _tiny_vision_config():
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

    config = Qwen3VLVisionConfig(
        depth=1,
        hidden_size=16,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=1,
        out_hidden_size=24,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    config._attn_implementation = "sdpa"
    config.attn_implementation = "sdpa"
    return config


def _assert_close(name, actual, expected, atol=2e-5, rtol=2e-4):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}")


def test_qwen3_vision_ulysses_block_matches_unsharded_forward_backward():
    sp_group, local_rank, rank, world_size = _init_sp_group()
    device = torch.device(f"cuda:{local_rank}")

    from python.ray.qwen3_vision_ulysses import apply_qwen3_vision_ulysses
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionBlock

    torch.manual_seed(9001)
    config = _tiny_vision_config()
    baseline = Qwen3VLVisionBlock(config).to(device=device, dtype=torch.float32)
    sharded = copy.deepcopy(baseline).to(device=device, dtype=torch.float32)
    container = torch.nn.Module()
    container.blocks = torch.nn.ModuleList([sharded])
    apply_qwen3_vision_ulysses(container, sp_group)

    global_seq = 16
    local_seq = global_seq // world_size
    start = rank * local_seq
    end = start + local_seq

    hidden = torch.randn(global_seq, config.hidden_size, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(global_seq, config.hidden_size // config.num_heads, device=device, dtype=torch.float32)
    position_embeddings = (angles.cos(), angles.sin())
    cu_seqlens = torch.tensor([0, 8, 16], device=device, dtype=torch.int32)
    grad_out = torch.randn(global_seq, config.hidden_size, device=device, dtype=torch.float32)

    baseline_out = baseline(hidden, cu_seqlens=cu_seqlens, position_embeddings=position_embeddings)
    baseline_out.backward(grad_out)

    local_hidden = hidden.detach()[start:end].clone().requires_grad_(True)
    local_position_embeddings = tuple(tensor[start:end].contiguous() for tensor in position_embeddings)
    sharded_out = sharded(local_hidden, cu_seqlens=cu_seqlens, position_embeddings=local_position_embeddings)
    sharded_out.backward(grad_out[start:end])

    _assert_close("local block output", sharded_out, baseline_out.detach()[start:end])
    _assert_close("local input grad", local_hidden.grad, hidden.grad.detach()[start:end])

    import torch.distributed as dist

    for (name, param), (_, expected) in zip(sharded.named_parameters(), baseline.named_parameters()):
        if param.grad is None:
            assert expected.grad is None
            continue
        dist.all_reduce(param.grad, group=sp_group)
        _assert_close(f"parameter grad {name}", param.grad, expected.grad)


def test_qwen3_vision_sequence_parallel_forward_matches_unsharded_model():
    sp_group, local_rank, rank, world_size = _init_sp_group()
    device = torch.device(f"cuda:{local_rank}")

    from python.ray.qwen3_vision_ulysses import apply_qwen3_vision_ulysses, qwen3_vision_sequence_parallel_forward
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(9002)
    config = _tiny_vision_config()
    baseline = Qwen3VLVisionModel(config).to(device=device, dtype=torch.float32)
    sharded = copy.deepcopy(baseline).to(device=device, dtype=torch.float32)
    apply_qwen3_vision_ulysses(sharded, sp_group)

    pixel_values = torch.randn(16, 3 * 1 * 2 * 2, device=device, dtype=torch.float32, requires_grad=True)
    sharded_pixel_values = pixel_values.detach().clone().requires_grad_(True)
    grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    grad_pooler = torch.randn(4, config.out_hidden_size, device=device, dtype=torch.float32)
    grad_deepstack = torch.randn(4, config.out_hidden_size, device=device, dtype=torch.float32)

    baseline_out = baseline(hidden_states=pixel_values, grid_thw=grid_thw)
    torch.autograd.backward(
        [baseline_out.pooler_output, baseline_out.deepstack_features[0]],
        [grad_pooler, grad_deepstack],
    )

    sharded_out = qwen3_vision_sequence_parallel_forward(
        sharded,
        hidden_states=sharded_pixel_values,
        grid_thw=grid_thw,
        process_group=sp_group,
    )
    torch.autograd.backward(
        [sharded_out.pooler_output, sharded_out.deepstack_features[0]],
        [grad_pooler / world_size, grad_deepstack / world_size],
    )

    _assert_close("last_hidden_state", sharded_out.last_hidden_state, baseline_out.last_hidden_state.detach())
    _assert_close("pooler_output", sharded_out.pooler_output, baseline_out.pooler_output.detach())
    assert len(sharded_out.deepstack_features) == 1
    _assert_close(
        "deepstack_features[0]",
        sharded_out.deepstack_features[0],
        baseline_out.deepstack_features[0].detach(),
    )

    import torch.distributed as dist

    dist.all_reduce(sharded_pixel_values.grad, group=sp_group)
    _assert_close("pixel grad", sharded_pixel_values.grad, pixel_values.grad)

    for (name, param), (_, expected) in zip(sharded.named_parameters(), baseline.named_parameters()):
        if param.grad is None:
            assert expected.grad is None
            continue
        dist.all_reduce(param.grad, group=sp_group)
        _assert_close(f"model parameter grad {name}", param.grad, expected.grad)


def test_qwen3_vision_trainer_standalone_sequence_forward_backward():
    sp_group, local_rank, rank, world_size = _init_sp_group()
    device = torch.device(f"cuda:{local_rank}")

    from python.ray.vision import Qwen3VLVisionTrainer

    trainer_cls = Qwen3VLVisionTrainer.__ray_metadata__.modified_class
    config = {
        "seed": 9003,
        "model_name": "__tiny_qwen3_vl__",
        "model_type": "qwen3_vl",
        "engine": "deepspeed",
        "parallelism": "sequence",
        "sequence_parallel_size": world_size,
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
    trainer = trainer_cls(config, rank=rank)
    trainer.build_model()

    pixel_values = torch.randn(16, 3 * 1 * 2 * 2, device=device, dtype=torch.float32)
    grid_thw = torch.tensor([[1, 4, 4]], device=device, dtype=torch.long)
    outputs = trainer._model_forward({"pixel_values": pixel_values, "image_grid_thw": grid_thw})
    assert outputs.pooler_output.shape == (1, 4, 24)
    assert len(outputs.deepstack_features) == 1
    assert outputs.deepstack_features[0].shape == (1, 4, 24)
    assert torch.isfinite(outputs.pooler_output).all()
    assert torch.isfinite(outputs.deepstack_features[0]).all()

    grad = {
        "pooler_output": torch.randn_like(outputs.pooler_output) / world_size,
        "deepstack_features": [torch.randn_like(outputs.deepstack_features[0]) / world_size],
    }
    trainer._pending_outputs.append(outputs)
    trainer._apply_vision_backward(grad)

    module = trainer.deepspeed_engine.module
    local_grad_count = sum(1 for param in module.parameters() if param.grad is not None)
    assert local_grad_count > 0
    assert trainer.sp_group is sp_group
