"""Step 5: Pipeline parallel for 2 stages (1F1B schedule comparison).

Global batch_size=8, seq_len=8192, microbatch_size=1.

Runs two variants from identical random init:
  - baseline: SequentialScheduler, 1 microbatch (full batch), lr=1e-4
  - target:   OneFOneBScheduler,   8 microbatches of size 1, lr=1e-4/8

Both must converge, and final losses must be within a bounded gap.

Requires at least 1 CUDA GPU.
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
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import OneFOneBScheduler, SequentialScheduler
from python.pipeline.stage import EdgeConfig, ParallelismType, Pipeline, Placement, ResourceSet, Stage


def _make_resource_sets_and_placements():
    """Return resource_sets and placements based on available GPUs."""
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
        raise RuntimeError("step5_pipeline_2mb requires at least one CUDA GPU")
    return resource_sets, placements


def run_variant(
    name,
    config,
    num_classes,
    attn_state_dict,
    moe_state_dict,
    data,
    labels,
    scheduler,
    num_microbatches,
    lr,
    dtype=None,
    num_steps=20,
):
    """Run one training variant from given initial state_dicts. Returns list of per-step losses.

    Creates fresh specs/manager/runner each time and always cleans up in finally.
    """
    resource_sets, placements = _make_resource_sets_and_placements()

    pipeline = Pipeline(
        stages=[
            Stage(name="attn", is_source=True, parallelism=ParallelismType.NONE),
            Stage(name="moe", is_terminal=True, parallelism=ParallelismType.NONE),
        ],
        edges=[EdgeConfig(src="attn", dst="moe")],
        resource_sets=resource_sets,
        placements=placements,
        num_microbatches=num_microbatches,
    )

    errors = pipeline.validate()
    assert not errors, f"Pipeline validation errors: {errors}"

    specs = [
        StageModelSpec(
            stage_name="attn",
            model_cls=Qwen3AttentionStage,
            model_kwargs={"config": config, "dtype": dtype},
            state_dict=attn_state_dict,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": lr, "foreach": False},
        ),
        StageModelSpec(
            stage_name="moe",
            model_cls=Qwen3MoEStageWithHead,
            model_kwargs={"config": config, "num_classes": num_classes, "dtype": dtype},
            state_dict=moe_state_dict,
            is_terminal=True,
            loss_cls=nn.CrossEntropyLoss,
            optimizer_cls=torch.optim.AdamW,
            optimizer_kwargs={"lr": lr, "foreach": False},
        ),
    ]

    runner = None
    manager = None
    try:
        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        manager.build_models(plan)
        runner = RayPipelineRunner(pipeline, plan, scheduler=scheduler)

        losses = []
        for step in range(num_steps):
            result = runner.run_iteration(data=data, labels=labels, iteration=step, num_microbatches=num_microbatches)
            assert result["loss"] is not None, f"Loss is None at step {step}"
            losses.append(result["loss"])
            print(f"  [{name}] Step {step:3d} | Loss: {result['loss']:.6f}")

        return losses
    finally:
        if runner is not None:
            runner.shutdown()
        if manager is not None:
            manager.shutdown()


def main():
    config = create_qwen3_config()
    num_classes = 10
    batch_size = 8  # global batch; split into microbatches of size 1
    dtype = torch.bfloat16

    # Create initial models with deterministic seed and save state_dicts
    # Both variants will start from identical weights
    torch.manual_seed(42)
    attn_model = Qwen3AttentionStage(config, dtype=dtype)
    moe_model = Qwen3MoEStageWithHead(config, num_classes, dtype=dtype)
    attn_state_dict = attn_model.state_dict()
    moe_state_dict = moe_model.state_dict()

    # Fixed batch (on CPU, bf16)
    torch.manual_seed(42)
    data, labels = generate_dummy_batch(
        config, batch_size=batch_size, seq_len=8192, num_classes=num_classes, device="cpu", dtype=dtype
    )

    ray_started_here = False
    if not ray.is_initialized():
        ray.init()
        ray_started_here = True

    try:
        # Variant 1: baseline (1 microbatch, sequential scheduler)
        base_lr = 1e-3
        num_steps_1mb = 20
        print(f"=== Baseline: 1 microbatch, SequentialScheduler, lr={base_lr}, {num_steps_1mb} steps ===")
        losses_1mb = run_variant(
            name="1mb",
            config=config,
            num_classes=num_classes,
            attn_state_dict=attn_state_dict,
            moe_state_dict=moe_state_dict,
            data=data,
            labels=labels,
            scheduler=SequentialScheduler(),
            num_microbatches=1,
            lr=base_lr,
            dtype=dtype,
            num_steps=num_steps_1mb,
        )

        # Variant 2: target (8 microbatches of size 1, 1F1B)
        # LR scaled down by 1/num_microbatches since StageTrainer sums gradients without normalization.
        # More steps needed to compensate for the lower effective LR.
        num_mb = batch_size
        scaled_lr = base_lr / num_mb
        num_steps_nmb = num_steps_1mb * num_mb
        print(f"\n=== Target: {num_mb} microbatches, OneFOneBScheduler, lr={scaled_lr}, {num_steps_nmb} steps ===")
        losses_2mb = run_variant(
            name=f"{num_mb}mb",
            config=config,
            num_classes=num_classes,
            attn_state_dict=attn_state_dict,
            moe_state_dict=moe_state_dict,
            data=data,
            labels=labels,
            scheduler=OneFOneBScheduler(),
            num_microbatches=num_mb,
            lr=scaled_lr,
            dtype=dtype,
            num_steps=num_steps_nmb,
        )

        # Check convergence for both
        for label, losses in [("1mb", losses_1mb), (f"{num_mb}mb", losses_2mb)]:
            first_5 = sum(losses[:5]) / 5
            last_5 = sum(losses[-5:]) / 5
            assert last_5 < first_5, f"{label} not converging: first_5={first_5:.6f}, last_5={last_5:.6f}"
            print(f"\n{label}: first_5={first_5:.6f}, last_5={last_5:.6f} — converged")

        # Check final-loss gap (both should reach near-zero)
        final_1mb = losses_1mb[-1]
        final_2mb = losses_2mb[-1]
        gap = abs(final_2mb - final_1mb)
        tolerance = 0.10
        print(f"\nFinal loss gap: |{final_2mb:.6f} - {final_1mb:.6f}| = {gap:.6f} (tolerance: {tolerance:.6f})")
        assert gap <= tolerance, f"Final loss gap {gap:.6f} exceeds tolerance {tolerance:.6f}"
        print("PASSED: Both variants converged with bounded final-loss gap")
    finally:
        if ray_started_here and ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()
