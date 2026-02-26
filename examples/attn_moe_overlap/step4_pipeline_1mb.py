"""Step 4: Pipeline parallel for 2 stages (8 microbatches, microbatch_size=1).

Runs the same attention + MoE stages via the pipeline framework with Ray,
verifying framework integration with real HF layers.

Global batch_size=8, seq_len=8192, split into 8 microbatches of size 1.

Requires at least 1 CUDA GPU. Uses 2-GPU placement by default;
falls back to 1-GPU overlap (subset_of) placement when only 1 GPU is available.
"""

import ray
import torch
import torch.nn as nn

from examples.attn_moe_overlap.model_utils import (
    Qwen3AttentionStage,
    Qwen3MoEStageWithHead,
    create_qwen3_config,
    generate_dummy_batch,
)
from python.pipeline.placement import PlacementManager, StageModelSpec
from python.pipeline.ray_runner import PipelineIterationError, RayPipelineRunner
from python.pipeline.stage import EdgeConfig, ParallelismType, Pipeline, Placement, ResourceSet, Stage


def main():
    config = create_qwen3_config()
    num_classes = 10
    runner = None
    manager = None
    ray_started_here = False

    num_gpus = torch.cuda.device_count()
    if num_gpus >= 2:
        resource_sets = [
            ResourceSet(name="rs_attn", num_gpus=1),
            ResourceSet(name="rs_moe", num_gpus=1),
        ]
        placements = [
            Placement(stage_name="attn", resource_set="rs_attn"),
            Placement(stage_name="moe", resource_set="rs_moe"),
        ]
    elif num_gpus == 1:
        resource_sets = [
            ResourceSet(name="rs_parent", num_gpus=1, device_ids=(0,)),
            ResourceSet(name="rs_child", num_gpus=1, device_ids=(0,), subset_of="rs_parent"),
        ]
        placements = [
            Placement(stage_name="attn", resource_set="rs_parent"),
            Placement(stage_name="moe", resource_set="rs_child"),
        ]
    else:
        raise RuntimeError("step4_pipeline_1mb requires at least one CUDA GPU")

    pipeline = Pipeline(
        stages=[
            Stage(name="attn", is_source=True, parallelism=ParallelismType.NONE),
            Stage(name="moe", is_terminal=True, parallelism=ParallelismType.NONE),
        ],
        edges=[EdgeConfig(src="attn", dst="moe")],
        resource_sets=resource_sets,
        placements=placements,
        num_microbatches=8,
    )

    errors = pipeline.validate()
    assert not errors, f"Pipeline validation errors: {errors}"

    # Build model specs with deterministic init (bf16 for memory-efficient attention)
    dtype = torch.bfloat16
    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=dtype)
    moe_model = Qwen3MoEStageWithHead(config, num_classes, dtype=dtype)

    specs = [
        StageModelSpec(
            stage_name="attn",
            model_cls=Qwen3AttentionStage,
            model_kwargs={"config": config, "dtype": dtype},
            state_dict=attn_model.state_dict(),
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3, "foreach": False},
        ),
        StageModelSpec(
            stage_name="moe",
            model_cls=Qwen3MoEStageWithHead,
            model_kwargs={"config": config, "num_classes": num_classes, "dtype": dtype},
            state_dict=moe_model.state_dict(),
            is_terminal=True,
            loss_cls=nn.CrossEntropyLoss,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": 1e-3, "foreach": False},
        ),
    ]

    # Generate fixed batch (on CPU, bf16 — will be sent to actors by the runner)
    torch.manual_seed(42)
    data, labels = generate_dummy_batch(
        config, batch_size=8, seq_len=8192, num_classes=num_classes, device="cpu", dtype=dtype
    )

    if not ray.is_initialized():
        ray.init()
        ray_started_here = True

    try:
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        manager.build_models(plan)
        runner = RayPipelineRunner(pipeline, plan)

        losses = []
        for step in range(20):
            result = runner.run_iteration(data=data, labels=labels, iteration=step)
            assert result["loss"] is not None, f"Loss is None at step {step}"
            losses.append(result["loss"])
            print(f"Step {step:3d} | Loss: {result['loss']:.6f}")

        first_5 = sum(losses[:5]) / 5
        last_5 = sum(losses[-5:]) / 5
        assert last_5 < first_5, f"Not converging: first_5={first_5:.6f}, last_5={last_5:.6f}"
        print("PASSED: Loss decreased")
    except PipelineIterationError as e:
        print(f"PipelineIterationError: stage={e.stage_name}, op={e.op}, microbatch={e.microbatch_id}")
        raise
    except torch.cuda.OutOfMemoryError:
        print("CUDA OOM in step4_pipeline_1mb; retry with batch_size=2, seq_len=64, or hidden_size=1024")
        raise
    finally:
        if runner is not None:
            runner.shutdown()
        if manager is not None:
            manager.shutdown()
        if ray_started_here and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
