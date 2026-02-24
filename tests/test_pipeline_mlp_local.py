"""M1: 3-stage MLP pipeline on single GPU with gradient matching against monolithic model.

Tests:
  1. Loss decreases over 100 iterations.
  2. Gradient flow matches monolithic nn.Sequential(A, B, C) within tolerance.
  3. Global grad norm matches monolithic model's grad norm.
"""

import copy
import math

import pytest
import torch
import torch.nn as nn

from python.pipeline.dag import PipelineDAG
from python.pipeline.native_runner import NativeRunner
from python.pipeline.native_trainer import StageTrainer
from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage
from python.ray.payloads import StageOutputs

# Dimensions
INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 10
BATCH_SIZE = 16
SEED = 42
NUM_TRAIN_ITERS = 100
LR = 0.01


def _make_mlp(in_dim: int, out_dim: int) -> nn.Linear:
    """Simple linear layer (MLP without activation for exact gradient matching)."""
    return nn.Linear(in_dim, out_dim, bias=True)


def _make_pipeline_and_trainers(device: torch.device, seed: int = SEED):
    """Build a 3-stage linear MLP pipeline with trainers."""
    torch.manual_seed(seed)

    model_a = _make_mlp(INPUT_DIM, HIDDEN_DIM).to(device)
    model_b = _make_mlp(HIDDEN_DIM, HIDDEN_DIM).to(device)
    model_c = _make_mlp(HIDDEN_DIM, OUTPUT_DIM).to(device)

    pipeline = Pipeline(
        stages=[
            Stage(name="a", is_source=True),
            Stage(name="b"),
            Stage(name="c", is_terminal=True),
        ],
        edges=[
            EdgeConfig(src="a", dst="b"),
            EdgeConfig(src="b", dst="c"),
        ],
        resource_sets=[ResourceSet(name="gpu", num_gpus=1)],
        placements=[
            Placement(stage_name="a", resource_set="gpu"),
            Placement(stage_name="b", resource_set="gpu"),
            Placement(stage_name="c", resource_set="gpu"),
        ],
    )

    loss_fn = nn.CrossEntropyLoss()

    trainers = {
        "a": StageTrainer(
            "a",
            model_a,
            optimizer=torch.optim.Adam(model_a.parameters(), lr=LR, foreach=False),
            device=device,
        ),
        "b": StageTrainer(
            "b",
            model_b,
            optimizer=torch.optim.Adam(model_b.parameters(), lr=LR, foreach=False),
            device=device,
        ),
        "c": StageTrainer(
            "c",
            model_c,
            optimizer=torch.optim.Adam(model_c.parameters(), lr=LR, foreach=False),
            loss_fn=loss_fn,
            is_terminal=True,
            device=device,
        ),
    }

    return pipeline, trainers, (model_a, model_b, model_c)


def _make_monolithic_model(device: torch.device, seed: int = SEED):
    """Build a monolithic nn.Sequential matching the 3-stage MLP pipeline."""
    torch.manual_seed(seed)

    model_a = _make_mlp(INPUT_DIM, HIDDEN_DIM).to(device)
    model_b = _make_mlp(HIDDEN_DIM, HIDDEN_DIM).to(device)
    model_c = _make_mlp(HIDDEN_DIM, OUTPUT_DIM).to(device)

    model = nn.Sequential(model_a, model_b, model_c).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=False)
    loss_fn = nn.CrossEntropyLoss()

    return model, optimizer, loss_fn


def _generate_batch(device: torch.device, seed_offset: int = 0):
    """Generate a reproducible batch of synthetic data."""
    gen = torch.Generator(device="cpu").manual_seed(SEED + seed_offset)
    data = torch.randn(BATCH_SIZE, INPUT_DIM, generator=gen, device="cpu").to(device)
    labels = torch.randint(0, OUTPUT_DIM, (BATCH_SIZE,), generator=gen, device="cpu").to(device)
    return data, labels


@pytest.mark.gpu
class TestPipelineMLPLocal:
    """GPU tests for M1: single-GPU MLP pipeline."""

    def test_loss_decreases(self):
        """3-stage MLP pipeline trains for 100 iterations on fixed data; final loss < initial loss."""
        device = torch.device("cuda:0")
        pipeline, trainers, _ = _make_pipeline_and_trainers(device)
        runner = NativeRunner(pipeline, trainers)

        # Use a fixed dataset so the model can memorize and loss decreases
        data, labels = _generate_batch(device, seed_offset=0)

        losses = []
        for i in range(NUM_TRAIN_ITERS):
            result = runner.run_iteration(data=data, labels=labels)
            losses.append(result["loss"])

        assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        # Also check that loss decreased significantly (not just noise)
        assert losses[-1] < losses[0] * 0.5, f"Loss decrease too small: {losses[0]:.6f} → {losses[-1]:.6f}"

    def test_gradient_matching(self):
        """Pipeline gradients match monolithic nn.Sequential gradients exactly (fp32)."""
        device = torch.device("cuda:0")

        # Build pipeline
        pipeline, trainers, (pa, pb, pc) = _make_pipeline_and_trainers(device)
        runner = NativeRunner(pipeline, trainers)

        # Build monolithic model with SAME weights
        mono, mono_opt, mono_loss_fn = _make_monolithic_model(device)

        # Verify initial weights match
        for (pp, mp) in zip([pa, pb, pc], mono.children()):
            for (p1, p2) in zip(pp.parameters(), mp.parameters()):
                assert torch.equal(p1.data, p2.data), "Initial weights don't match"

        # Run 5 iterations comparing gradients at each step
        for i in range(5):
            data, labels = _generate_batch(device, seed_offset=i)

            # Pipeline forward+backward (no optimizer step yet — we need to compare grads first)
            from python.pipeline.scheduler import SequentialScheduler
            from python.pipeline.dag import PipelineDAG

            dag = PipelineDAG(pipeline)
            topo_order = dag.topological_sort()
            schedule = SequentialScheduler().generate_schedule(topo_order, 1)

            stage_outputs = {}
            stage_grads_map = {}

            for step in schedule:
                trainer = trainers[step.stage_name]
                if step.op.value == "forward":
                    preds = dag.predecessors(step.stage_name)
                    if not preds:
                        inputs = StageOutputs(activations=data)
                    else:
                        inputs = StageOutputs(activations=stage_outputs[preds[0]].activations)
                    stage_labels = labels if trainer.is_terminal else None
                    output = trainer.forward_step(inputs, labels=stage_labels)
                    stage_outputs[step.stage_name] = output
                elif step.op.value == "backward":
                    succs = dag.successors(step.stage_name)
                    if not succs:
                        upstream_grad = trainer.backward_step(downstream_grad=None)
                    else:
                        downstream_grad = stage_grads_map.get(succs[0])
                        upstream_grad = trainer.backward_step(downstream_grad=downstream_grad)
                    stage_grads_map[step.stage_name] = upstream_grad

            # Monolithic forward+backward
            mono_opt.zero_grad()
            mono_out = mono(data)
            mono_loss = mono_loss_fn(mono_out, labels)
            mono_loss.backward()

            # Compare gradients
            for (stage_model, mono_module) in zip([pa, pb, pc], mono.children()):
                for (p_pipe, p_mono) in zip(stage_model.parameters(), mono_module.parameters()):
                    assert p_pipe.grad is not None, f"Pipeline grad is None for stage"
                    assert p_mono.grad is not None, f"Monolithic grad is None"
                    max_abs_diff = (p_pipe.grad - p_mono.grad).abs().max().item()
                    assert max_abs_diff <= 1e-5, f"Gradient abs diff {max_abs_diff} > 1e-5 at iter {i}"

                    # Relative diff
                    denom = p_mono.grad.abs().clamp(min=1e-8)
                    max_rel_diff = ((p_pipe.grad - p_mono.grad).abs() / denom).max().item()
                    assert max_rel_diff <= 1e-3, f"Gradient rel diff {max_rel_diff} > 1e-3 at iter {i}"

            # Now step both optimizers
            pipeline_norm_sq = sum(t.compute_grad_norm_sq() for t in trainers.values())
            pipeline_grad_norm = math.sqrt(pipeline_norm_sq)

            mono_norm_sq = sum(p.grad.data.float().norm(2).item() ** 2 for p in mono.parameters() if p.grad is not None)
            mono_grad_norm = math.sqrt(mono_norm_sq)

            for name in topo_order:
                trainers[name].optimizer_step()

            mono_opt.step()

    def test_global_grad_norm_matches(self):
        """Global grad norm from pipeline matches monolithic model within tolerance."""
        device = torch.device("cuda:0")

        pipeline, trainers, (pa, pb, pc) = _make_pipeline_and_trainers(device)
        mono, mono_opt, mono_loss_fn = _make_monolithic_model(device)

        data, labels = _generate_batch(device, seed_offset=0)

        # Pipeline forward+backward
        runner = NativeRunner(pipeline, trainers)
        # Do forward+backward manually to get grad norm before optimizer step
        from python.pipeline.scheduler import SequentialScheduler
        from python.pipeline.dag import PipelineDAG

        dag = PipelineDAG(pipeline)
        topo_order = dag.topological_sort()
        schedule = SequentialScheduler().generate_schedule(topo_order, 1)

        stage_outputs = {}
        stage_grads_map = {}

        for step in schedule:
            trainer = trainers[step.stage_name]
            if step.op.value == "forward":
                preds = dag.predecessors(step.stage_name)
                if not preds:
                    inputs = StageOutputs(activations=data)
                else:
                    inputs = StageOutputs(activations=stage_outputs[preds[0]].activations)
                stage_labels = labels if trainer.is_terminal else None
                output = trainer.forward_step(inputs, labels=stage_labels)
                stage_outputs[step.stage_name] = output
            elif step.op.value == "backward":
                succs = dag.successors(step.stage_name)
                if not succs:
                    upstream_grad = trainer.backward_step(downstream_grad=None)
                else:
                    downstream_grad = stage_grads_map.get(succs[0])
                    upstream_grad = trainer.backward_step(downstream_grad=downstream_grad)
                stage_grads_map[step.stage_name] = upstream_grad

        # Pipeline grad norm
        pipeline_norm_sq = sum(t.compute_grad_norm_sq() for t in trainers.values())
        pipeline_grad_norm = math.sqrt(pipeline_norm_sq)

        # Monolithic forward+backward
        mono_opt.zero_grad()
        mono_out = mono(data)
        mono_loss = mono_loss_fn(mono_out, labels)
        mono_loss.backward()

        # Monolithic grad norm
        mono_norm_sq = sum(p.grad.data.float().norm(2).item() ** 2 for p in mono.parameters() if p.grad is not None)
        mono_grad_norm = math.sqrt(mono_norm_sq)

        abs_diff = abs(pipeline_grad_norm - mono_grad_norm)
        assert abs_diff <= 1e-5, (
            f"Grad norm mismatch: pipeline={pipeline_grad_norm:.8f}, mono={mono_grad_norm:.8f}, diff={abs_diff:.8f}"
        )
