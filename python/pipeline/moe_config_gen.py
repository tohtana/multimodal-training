"""Programmatic config generator for N-layer interleaved MoE pipelines (design §5.2).

Generates Pipeline and StageModelSpec objects for models where attention sublayers
run on one GPU set (TP) and MoE FFN sublayers run on another GPU set (EP),
alternating every layer. The attention GPU set can be ``subset_of`` the MoE set
to enable overlapping execution on shared GPUs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .moe_models import SimpleAttentionBlock, SimpleMoEBlock, SimpleMoEBlockWithHead
from .placement import StageModelSpec
from .stage import EdgeConfig, ParallelismType, Pipeline, Placement, ResourceSet, Stage


def generate_moe_pipeline(
    num_layers: int,
    hidden_dim: int,
    num_attn_gpus: int,
    num_moe_gpus: int,
    attn_device_ids: tuple[int, ...] | None = None,
    moe_device_ids: tuple[int, ...] | None = None,
    num_microbatches: int = 1,
    use_subset: bool = True,
) -> Pipeline:
    """Generate a Pipeline for an N-layer interleaved MoE model.

    Creates stages: attn_0 → moe_0 → attn_1 → moe_1 → ... → attn_{N-1} → moe_{N-1}
    Attention stages use ``ag_attn`` (TP), MoE stages use ``ag_moe`` (EP).

    Args:
        num_layers: Number of transformer layers (each = 1 attn + 1 MoE stage).
        hidden_dim: Hidden dimension for all stages.
        num_attn_gpus: Number of GPUs for attention (TP) stages.
        num_moe_gpus: Number of GPUs for MoE (EP) stages.
        attn_device_ids: Physical device IDs for attention GPUs.
        moe_device_ids: Physical device IDs for MoE GPUs.
        num_microbatches: Number of microbatches for pipeline scheduling.
        use_subset: If True and device_ids are provided, make ag_attn a subset_of ag_moe.
    """
    stages = []
    edges = []
    placements = []

    for i in range(num_layers):
        attn_name = f"attn_{i}"
        moe_name = f"moe_{i}"

        stages.append(
            Stage(
                name=attn_name,
                is_source=(i == 0),
                parallelism=ParallelismType.TENSOR,
            )
        )
        stages.append(
            Stage(
                name=moe_name,
                is_terminal=(i == num_layers - 1),
                parallelism=ParallelismType.EXPERT,
            )
        )

        # Intra-layer edge: attn → moe
        edges.append(EdgeConfig(src=attn_name, dst=moe_name))

        # Inter-layer edge: moe → next attn
        if i < num_layers - 1:
            edges.append(EdgeConfig(src=moe_name, dst=f"attn_{i + 1}"))

        placements.append(Placement(stage_name=attn_name, resource_set="ag_attn"))
        placements.append(Placement(stage_name=moe_name, resource_set="ag_moe"))

    # Resource sets
    resource_sets = [
        ResourceSet(name="ag_moe", num_gpus=num_moe_gpus, device_ids=moe_device_ids),
    ]

    can_subset = use_subset and attn_device_ids is not None and moe_device_ids is not None
    resource_sets.append(
        ResourceSet(
            name="ag_attn",
            num_gpus=num_attn_gpus,
            device_ids=attn_device_ids,
            subset_of="ag_moe" if can_subset else None,
        )
    )

    return Pipeline(
        stages=stages,
        edges=edges,
        resource_sets=resource_sets,
        placements=placements,
        num_microbatches=num_microbatches,
    )


def generate_moe_model_specs(
    num_layers: int,
    hidden_dim: int,
    output_dim: int,
    num_experts: int = 4,
    top_k: int = 2,
    lr: float = 0.01,
    seed: int = 42,
) -> list[StageModelSpec]:
    """Generate StageModelSpec list for an interleaved MoE pipeline.

    Each attention stage gets a ``SimpleAttentionBlock``, each non-terminal MoE
    stage gets a ``SimpleMoEBlock``, and the terminal MoE stage gets a
    ``SimpleMoEBlockWithHead`` with a classification head.

    Returns deterministically-initialized model specs (one per stage).
    """
    specs = []

    for i in range(num_layers):
        attn_name = f"attn_{i}"
        moe_name = f"moe_{i}"
        is_terminal = i == num_layers - 1

        # Attention stage
        torch.manual_seed(seed + i * 100)
        attn_sd = SimpleAttentionBlock(hidden_dim).state_dict()
        specs.append(
            StageModelSpec(
                stage_name=attn_name,
                model_cls=SimpleAttentionBlock,
                model_kwargs={"hidden_dim": hidden_dim},
                state_dict=attn_sd,
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": lr, "foreach": False},
            )
        )

        # MoE stage
        torch.manual_seed(seed + i * 100 + 50)
        if is_terminal:
            moe_model = SimpleMoEBlockWithHead(hidden_dim, output_dim, num_experts, top_k)
            specs.append(
                StageModelSpec(
                    stage_name=moe_name,
                    model_cls=SimpleMoEBlockWithHead,
                    model_kwargs={
                        "hidden_dim": hidden_dim,
                        "num_classes": output_dim,
                        "num_experts": num_experts,
                        "top_k": top_k,
                    },
                    state_dict=moe_model.state_dict(),
                    is_terminal=True,
                    optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={"lr": lr, "foreach": False},
                    loss_cls=nn.CrossEntropyLoss,
                )
            )
        else:
            moe_model = SimpleMoEBlock(hidden_dim, num_experts, top_k)
            specs.append(
                StageModelSpec(
                    stage_name=moe_name,
                    model_cls=SimpleMoEBlock,
                    model_kwargs={
                        "hidden_dim": hidden_dim,
                        "num_experts": num_experts,
                        "top_k": top_k,
                    },
                    state_dict=moe_model.state_dict(),
                    optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={"lr": lr, "foreach": False},
                )
            )

    return specs
