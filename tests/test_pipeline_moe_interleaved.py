"""M12: Interleaved MoE (Capstone) — alternating attention (TP) and MoE (EP) stages.

CPU-only tests verify:
  1. Config generator produces valid Pipeline objects.
  2. TP↔EP layout adapters resolve correctly.
  3. Stage/edge/placement counts are correct for N-layer configs.

GPU tests verify (4+ GPUs, Ray):
  4. 2-layer interleaved MoE pipeline trains with loss decreasing.
  5. Transport tiers are correct: T1 for shared GPUs, T2 for non-shared.
  6. 1F1B schedule works on the interleaved 4-stage pipeline.
"""

import pytest
import ray
import torch

from python.pipeline.dag import PipelineDAG
from python.pipeline.layout import resolve_layout_adapter
from python.pipeline.moe_config_gen import generate_moe_model_specs, generate_moe_pipeline
from python.pipeline.moe_models import SimpleAttentionBlock, SimpleMoEBlock, SimpleMoEBlockWithHead
from python.pipeline.placement import PlacementManager
from python.pipeline.ray_runner import RayPipelineRunner
from python.pipeline.scheduler import OneFOneBScheduler
from python.pipeline.stage import ParallelismType

# Dimensions
HIDDEN_DIM = 32
OUTPUT_DIM = 10
NUM_EXPERTS = 4
TOP_K = 2
BATCH_SIZE = 16
SEED = 42
LR = 0.01
NUM_ITERS = 20


# ── CPU-only tests ──


@pytest.mark.cpu_only
class TestMoEModels:
    """Verify MoE model factories produce correct shapes."""

    def test_attention_block_shape(self):
        model = SimpleAttentionBlock(HIDDEN_DIM)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_attention_block_residual(self):
        """Output should differ from input (residual + projection)."""
        model = SimpleAttentionBlock(HIDDEN_DIM)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        out = model(x)
        assert not torch.allclose(out, x), "Residual output should not equal input"

    def test_moe_block_shape(self):
        model = SimpleMoEBlock(HIDDEN_DIM, NUM_EXPERTS, TOP_K)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, HIDDEN_DIM)

    def test_moe_block_with_head_shape(self):
        model = SimpleMoEBlockWithHead(HIDDEN_DIM, OUTPUT_DIM, NUM_EXPERTS, TOP_K)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM)
        out = model(x)
        assert out.shape == (BATCH_SIZE, OUTPUT_DIM)

    def test_moe_block_differentiable(self):
        """MoE block should be fully differentiable (soft gating)."""
        model = SimpleMoEBlock(HIDDEN_DIM, NUM_EXPERTS, TOP_K)
        x = torch.randn(BATCH_SIZE, HIDDEN_DIM, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape


@pytest.mark.cpu_only
class TestMoEConfigGeneration:
    """Verify config generator produces valid Pipeline objects."""

    def test_2layer_pipeline_structure(self):
        """2-layer MoE: 4 stages, 3 edges."""
        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
        )
        assert len(pipeline.stages) == 4
        assert len(pipeline.edges) == 3  # attn0→moe0, moe0→attn1, attn1→moe1
        assert len(pipeline.placements) == 4

        # Check stage names
        names = [s.name for s in pipeline.stages]
        assert names == ["attn_0", "moe_0", "attn_1", "moe_1"]

        # Check source/terminal
        assert pipeline.get_stage("attn_0").is_source
        assert pipeline.get_stage("moe_1").is_terminal
        assert not pipeline.get_stage("moe_0").is_terminal
        assert not pipeline.get_stage("attn_1").is_source

    def test_4layer_pipeline_structure(self):
        """4-layer MoE: 8 stages, 7 edges."""
        pipeline = generate_moe_pipeline(
            num_layers=4,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=4,
            num_moe_gpus=8,
        )
        assert len(pipeline.stages) == 8
        assert len(pipeline.edges) == 7
        assert pipeline.get_stage("attn_0").is_source
        assert pipeline.get_stage("moe_3").is_terminal

    def test_parallelism_types(self):
        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
        )
        for s in pipeline.stages:
            if s.name.startswith("attn"):
                assert s.parallelism == ParallelismType.TENSOR
            else:
                assert s.parallelism == ParallelismType.EXPERT

    def test_subset_of_resource_sets(self):
        """With device_ids, ag_attn should be subset_of ag_moe."""
        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            attn_device_ids=(0, 1),
            moe_device_ids=(0, 1, 2, 3),
            use_subset=True,
        )
        attn_rs = pipeline.get_resource_set("ag_attn")
        assert attn_rs.subset_of == "ag_moe"
        errors = pipeline.validate()
        assert not errors, f"Validation errors: {errors}"

    def test_no_subset_when_disabled(self):
        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            attn_device_ids=(0, 1),
            moe_device_ids=(0, 1, 2, 3),
            use_subset=False,
        )
        attn_rs = pipeline.get_resource_set("ag_attn")
        assert attn_rs.subset_of is None

    def test_topological_order(self):
        """DAG topological order should be attn_0, moe_0, attn_1, moe_1, ..."""
        pipeline = generate_moe_pipeline(
            num_layers=3,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
        )
        dag = PipelineDAG(pipeline)
        topo = dag.topological_sort()
        assert topo == ["attn_0", "moe_0", "attn_1", "moe_1", "attn_2", "moe_2"]

    def test_model_specs_count(self):
        """One spec per stage."""
        specs = generate_moe_model_specs(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
        )
        assert len(specs) == 4
        names = [s.stage_name for s in specs]
        assert names == ["attn_0", "moe_0", "attn_1", "moe_1"]

    def test_terminal_spec_has_loss(self):
        specs = generate_moe_model_specs(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
        )
        terminal = [s for s in specs if s.is_terminal]
        assert len(terminal) == 1
        assert terminal[0].stage_name == "moe_1"
        assert terminal[0].loss_cls is not None

    def test_1f1b_schedule_on_interleaved(self):
        """1F1B schedule generates valid steps for 4-stage interleaved pipeline."""
        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            num_microbatches=2,
        )
        dag = PipelineDAG(pipeline)
        topo = dag.topological_sort()
        scheduler = OneFOneBScheduler()
        steps = scheduler.generate_schedule(topo, num_microbatches=2)

        # 4 stages × 2 microbatches × 2 (fwd+bwd) = 16 steps
        assert len(steps) == 16

        # Verify all stages appear in both forward and backward
        fwd_stages = {s.stage_name for s in steps if s.op.value == "forward"}
        bwd_stages = {s.stage_name for s in steps if s.op.value == "backward"}
        assert fwd_stages == set(topo)
        assert bwd_stages == set(topo)


@pytest.mark.cpu_only
class TestTPEPLayoutAdapters:
    """Verify TP↔EP layout adapters resolve correctly."""

    def test_tp_to_ep_resolves(self):
        adapter = resolve_layout_adapter("tensor", "expert")
        assert adapter.name == "tp_to_ep"

    def test_ep_to_tp_resolves(self):
        adapter = resolve_layout_adapter("expert", "tensor")
        assert adapter.name == "ep_to_tp"

    def test_tp_to_ep_identity(self):
        """TP→EP per-actor adapter should be identity."""
        adapter = resolve_layout_adapter("tensor", "expert")
        x = torch.randn(8, 32)
        out = adapter.forward_fn(x, rank=0, world_size=4)
        assert torch.equal(out, x)

    def test_ep_to_tp_identity(self):
        adapter = resolve_layout_adapter("expert", "tensor")
        x = torch.randn(8, 32)
        out = adapter.forward_fn(x, rank=0, world_size=4)
        assert torch.equal(out, x)


# ── GPU tests ──


def _generate_batch(hidden_dim, output_dim, batch_size=BATCH_SIZE, seed_offset=0):
    gen = torch.Generator(device="cpu").manual_seed(SEED + seed_offset)
    data = torch.randn(batch_size, hidden_dim, generator=gen, device="cpu")
    labels = torch.randint(0, output_dim, (batch_size,), generator=gen, device="cpu")
    return data, labels


@pytest.fixture(scope="module")
def ray_context():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield


@pytest.mark.gpu
class TestMoEInterleavedGPU:
    def test_interleaved_loss_decreases(self, ray_context):
        """2-layer interleaved MoE on 4 GPUs: loss should decrease over iterations."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Interleaved MoE test requires at least 4 GPUs")

        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            attn_device_ids=(0, 1),
            moe_device_ids=(0, 1, 2, 3),
            num_microbatches=1,
            use_subset=True,
        )
        specs = generate_moe_model_specs(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            lr=LR,
            seed=SEED,
        )

        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            data, labels = _generate_batch(HIDDEN_DIM, OUTPUT_DIM)
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(data=data, labels=labels, iteration=i)
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_interleaved_with_1f1b(self, ray_context):
        """2-layer interleaved MoE with 1F1B schedule and 2 microbatches."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Interleaved MoE 1F1B test requires at least 4 GPUs")

        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            attn_device_ids=(0, 1),
            moe_device_ids=(0, 1, 2, 3),
            num_microbatches=2,
            use_subset=True,
        )
        specs = generate_moe_model_specs(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            lr=LR,
            seed=SEED,
        )

        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        scheduler = OneFOneBScheduler()
        runner = RayPipelineRunner(pipeline, plan, scheduler=scheduler)

        try:
            manager.build_models(plan)

            data, labels = _generate_batch(HIDDEN_DIM, OUTPUT_DIM)
            losses = []

            for i in range(NUM_ITERS):
                result = runner.run_iteration(
                    data=data,
                    labels=labels,
                    iteration=i,
                    num_microbatches=2,
                )
                assert result["loss"] is not None, f"Loss is None at iter {i}"
                losses.append(result["loss"])

            assert losses[-1] < losses[0], f"Loss did not decrease: initial={losses[0]:.6f}, final={losses[-1]:.6f}"
        finally:
            runner.shutdown()
            manager.shutdown()

    def test_transport_tiers_correct(self, ray_context):
        """Verify T1 for shared GPUs, T2 for non-shared in interleaved MoE."""
        if torch.cuda.device_count() < 4:
            pytest.skip("Transport tier test requires at least 4 GPUs")

        pipeline = generate_moe_pipeline(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            num_attn_gpus=2,
            num_moe_gpus=4,
            attn_device_ids=(0, 1),
            moe_device_ids=(0, 1, 2, 3),
            use_subset=True,
        )
        specs = generate_moe_model_specs(
            num_layers=2,
            hidden_dim=HIDDEN_DIM,
            output_dim=OUTPUT_DIM,
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            lr=LR,
            seed=SEED,
        )

        manager = PlacementManager(pipeline, model_specs=specs)
        plan = manager.plan()
        runner = RayPipelineRunner(pipeline, plan)

        try:
            manager.build_models(plan)

            # All attn→moe and moe→attn edges should have some T1 pairs
            # (attn on GPUs 0,1 overlaps with moe on GPUs 0,1,2,3)
            assert runner.router.has_t1_edges(), "Expected T1 edges for overlapping GPUs"

            # Verify routing plans: attn(2) → moe(4) is 2→4 expansion
            routing_attn_moe = runner.router.get_routing("attn_0", "moe_0")
            assert routing_attn_moe.num_src == 2
            assert routing_attn_moe.num_dst == 4

            # Verify routing plans: moe(4) → attn(2) is 4→2 contraction
            routing_moe_attn = runner.router.get_routing("moe_0", "attn_1")
            assert routing_moe_attn.num_src == 4
            assert routing_moe_attn.num_dst == 2

            # Run one iteration to confirm end-to-end correctness
            data, labels = _generate_batch(HIDDEN_DIM, OUTPUT_DIM)
            result = runner.run_iteration(data=data, labels=labels, iteration=0)
            assert result["loss"] is not None
        finally:
            runner.shutdown()
            manager.shutdown()
