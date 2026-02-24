"""Tests for pipeline DAG, dataclasses, config loader, and validation (M0)."""

import os
import textwrap
import tempfile

import pytest

from python.pipeline.stage import (
    EdgeConfig,
    EngineType,
    MergePolicy,
    ParallelismType,
    Pipeline,
    Placement,
    ResourceSet,
    Stage,
)
from python.pipeline.dag import PipelineDAG, PipelineDAGError
from python.pipeline.scheduler import OpType, ScheduleStep, SequentialScheduler
from python.pipeline.config_loader import (
    ConfigLoadError,
    load_pipeline_config,
    parse_pipeline_dict,
    register_trainer,
)
from python.ray.payloads import StageOutputs, StageGradients


# ── Helpers ──


def _linear_pipeline(names: list[str], num_gpus: int = 4) -> Pipeline:
    """Build a simple linear pipeline (A → B → C → ...)."""
    stages = []
    for i, name in enumerate(names):
        stages.append(
            Stage(
                name=name,
                is_source=(i == 0),
                is_terminal=(i == len(names) - 1),
            )
        )
    edges = [EdgeConfig(src=names[i], dst=names[i + 1]) for i in range(len(names) - 1)]
    rs = ResourceSet(name="gpus", num_gpus=num_gpus)
    placements = [Placement(stage_name=n, resource_set="gpus") for n in names]
    return Pipeline(stages=stages, edges=edges, resource_sets=[rs], placements=placements)


def _write_yaml(content: str) -> str:
    """Write YAML content to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    os.write(fd, content.encode())
    os.close(fd)
    return path


# ── Dataclass tests ──


@pytest.mark.cpu_only
class TestDataclasses:
    def test_stage_defaults(self):
        s = Stage(name="test")
        assert s.parallelism == ParallelismType.NONE
        assert s.engine == EngineType.NATIVE
        assert not s.is_source
        assert not s.is_terminal

    def test_stage_non_source_with_dataloader_raises(self):
        with pytest.raises(ValueError, match="non-source"):
            Stage(name="bad", is_source=False, dataloader_fn=lambda: None)

    def test_resource_set_validation(self):
        with pytest.raises(ValueError, match="num_gpus must be > 0"):
            ResourceSet(name="bad", num_gpus=0)

    def test_resource_set_device_ids_mismatch(self):
        with pytest.raises(ValueError, match="device_ids length"):
            ResourceSet(name="bad", num_gpus=2, device_ids=(0,))

    def test_resource_set_valid(self):
        rs = ResourceSet(name="ok", num_gpus=2, device_ids=(0, 1))
        assert rs.num_gpus == 2
        assert rs.device_ids == (0, 1)

    def test_edge_config(self):
        e = EdgeConfig(src="a", dst="b")
        assert e.src == "a"
        assert e.dst == "b"
        assert e.merge_policy is None

    def test_pipeline_get_stage(self):
        p = _linear_pipeline(["a", "b"])
        assert p.get_stage("a").name == "a"
        with pytest.raises(KeyError):
            p.get_stage("nonexistent")

    def test_pipeline_stage_resource_set(self):
        p = _linear_pipeline(["a", "b"])
        rs = p.stage_resource_set("a")
        assert rs.name == "gpus"


# ── DAG topology tests ──


@pytest.mark.cpu_only
class TestDAGTopology:
    def test_two_stage_topo_order(self):
        p = _linear_pipeline(["vision", "text"])
        dag = PipelineDAG(p)
        order = dag.topological_sort()
        assert order == ["text", "vision"] or order == ["vision", "text"]
        # vision must come before text
        assert order.index("vision") < order.index("text")

    def test_three_stage_topo_order(self):
        p = _linear_pipeline(["a", "b", "c"])
        dag = PipelineDAG(p)
        order = dag.topological_sort()
        assert order.index("a") < order.index("b") < order.index("c")

    def test_five_stage_linear(self):
        names = ["s0", "s1", "s2", "s3", "s4"]
        p = _linear_pipeline(names)
        dag = PipelineDAG(p)
        order = dag.topological_sort()
        for i in range(len(names) - 1):
            assert order.index(names[i]) < order.index(names[i + 1])

    def test_fan_in_dag(self):
        """Two sources → one sink with merge policy."""
        stages = [
            Stage(name="src1", is_source=True),
            Stage(name="src2", is_source=True),
            Stage(name="sink", is_terminal=True),
        ]
        edges = [
            EdgeConfig(src="src1", dst="sink", merge_policy=MergePolicy.CONCAT),
            EdgeConfig(src="src2", dst="sink", merge_policy=MergePolicy.CONCAT),
        ]
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name=s.name, resource_set="gpus") for s in stages]
        p = Pipeline(stages=stages, edges=edges, resource_sets=[rs], placements=placements)

        dag = PipelineDAG(p)
        order = dag.topological_sort()
        # Both sources must come before sink
        assert order.index("src1") < order.index("sink")
        assert order.index("src2") < order.index("sink")

        # Validation should pass
        errors = dag.validate()
        assert not errors, f"Unexpected errors: {errors}"

    def test_fan_in_without_merge_policy_fails(self):
        """Two sources → one sink WITHOUT merge policy should fail validation."""
        stages = [
            Stage(name="src1", is_source=True),
            Stage(name="src2", is_source=True),
            Stage(name="sink", is_terminal=True),
        ]
        edges = [
            EdgeConfig(src="src1", dst="sink"),
            EdgeConfig(src="src2", dst="sink"),
        ]
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name=s.name, resource_set="gpus") for s in stages]
        p = Pipeline(stages=stages, edges=edges, resource_sets=[rs], placements=placements)

        dag = PipelineDAG(p)
        errors = dag.validate()
        assert any("merge policy" in e.lower() or "Fan-in" in e for e in errors)

    def test_cycle_detection(self):
        stages = [
            Stage(name="a", is_source=True),
            Stage(name="b"),
            Stage(name="c", is_terminal=True),
        ]
        # a → b → c → a (cycle)
        edges = [
            EdgeConfig(src="a", dst="b"),
            EdgeConfig(src="b", dst="c"),
            EdgeConfig(src="c", dst="a"),
        ]
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name=s.name, resource_set="gpus") for s in stages]
        p = Pipeline(stages=stages, edges=edges, resource_sets=[rs], placements=placements)
        dag = PipelineDAG(p)

        with pytest.raises(PipelineDAGError, match="Cycle"):
            dag.topological_sort()

    def test_disconnected_graph(self):
        stages = [
            Stage(name="a", is_source=True, is_terminal=True),
            Stage(name="b", is_source=True, is_terminal=True),
        ]
        # No edges between a and b
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name=s.name, resource_set="gpus") for s in stages]
        p = Pipeline(stages=stages, edges=[], resource_sets=[rs], placements=placements)
        dag = PipelineDAG(p)
        assert not dag.is_connected()
        errors = dag.validate()
        assert any("not connected" in e for e in errors)

    def test_missing_placement(self):
        stages = [Stage(name="a", is_source=True, is_terminal=True)]
        rs = ResourceSet(name="gpus", num_gpus=4)
        # No placement for "a"
        p = Pipeline(stages=stages, edges=[], resource_sets=[rs], placements=[])
        errors = p.validate()
        assert any("no placement" in e.lower() for e in errors)

    def test_sources_and_sinks(self):
        p = _linear_pipeline(["a", "b", "c"])
        dag = PipelineDAG(p)
        assert dag.sources() == ["a"]
        assert dag.sinks() == ["c"]

    def test_predecessors_and_successors(self):
        p = _linear_pipeline(["a", "b", "c"])
        dag = PipelineDAG(p)
        assert dag.predecessors("b") == ["a"]
        assert dag.successors("b") == ["c"]
        assert dag.predecessors("a") == []
        assert dag.successors("c") == []


# ── Pipeline validation tests ──


@pytest.mark.cpu_only
class TestPipelineValidation:
    def test_duplicate_stage_names(self):
        stages = [Stage(name="a", is_source=True), Stage(name="a", is_terminal=True)]
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name="a", resource_set="gpus")]
        p = Pipeline(stages=stages, edges=[], resource_sets=[rs], placements=placements)
        errors = p.validate()
        assert any("Duplicate stage" in e for e in errors)

    def test_dp_size_constraint(self):
        p = _linear_pipeline(["a", "b"])
        p.dp_size = 2
        errors = p.validate()
        assert any("dp_size must be 1" in e for e in errors)

    def test_microbatch_gradient_accumulation_constraint(self):
        p = _linear_pipeline(["a", "b"])
        p.num_microbatches = 4
        p.gradient_accumulation_steps = 2
        errors = p.validate()
        assert any("gradient_accumulation_steps" in e for e in errors)

    def test_valid_two_stage(self):
        p = _linear_pipeline(["vision", "text"])
        errors = p.validate()
        assert not errors, f"Unexpected errors: {errors}"

    def test_unknown_resource_set_in_placement(self):
        stages = [Stage(name="a", is_source=True, is_terminal=True)]
        rs = ResourceSet(name="gpus", num_gpus=4)
        placements = [Placement(stage_name="a", resource_set="nonexistent")]
        p = Pipeline(stages=stages, edges=[], resource_sets=[rs], placements=placements)
        errors = p.validate()
        assert any("unknown resource set" in e.lower() for e in errors)

    def test_subset_of_unknown_parent(self):
        rs1 = ResourceSet(name="all", num_gpus=8)
        rs2 = ResourceSet(name="sub", num_gpus=2, subset_of="nonexistent")
        stages = [Stage(name="a", is_source=True, is_terminal=True)]
        placements = [Placement(stage_name="a", resource_set="all")]
        p = Pipeline(stages=stages, edges=[], resource_sets=[rs1, rs2], placements=placements)
        errors = p.validate()
        assert any("unknown parent" in e.lower() for e in errors)


# ── Scheduler tests ──


@pytest.mark.cpu_only
class TestSequentialScheduler:
    def test_single_microbatch(self):
        scheduler = SequentialScheduler()
        steps = scheduler.generate_schedule(["a", "b", "c"], num_microbatches=1)
        assert len(steps) == 6  # 3 forward + 3 backward
        assert steps[0] == ScheduleStep(OpType.FORWARD, "a", 0)
        assert steps[1] == ScheduleStep(OpType.FORWARD, "b", 0)
        assert steps[2] == ScheduleStep(OpType.FORWARD, "c", 0)
        assert steps[3] == ScheduleStep(OpType.BACKWARD, "c", 0)
        assert steps[4] == ScheduleStep(OpType.BACKWARD, "b", 0)
        assert steps[5] == ScheduleStep(OpType.BACKWARD, "a", 0)

    def test_multiple_microbatches(self):
        scheduler = SequentialScheduler()
        steps = scheduler.generate_schedule(["a", "b"], num_microbatches=3)
        assert len(steps) == 12  # 3 * (2 forward + 2 backward)
        # First microbatch
        assert steps[0].microbatch_id == 0
        assert steps[3].microbatch_id == 0
        # Second microbatch
        assert steps[4].microbatch_id == 1
        # Third microbatch
        assert steps[8].microbatch_id == 2


# ── Config loader tests ──


@pytest.mark.cpu_only
class TestConfigLoader:
    def test_load_two_stage_yaml(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "pipeline_two_stage.yaml")
        pipeline = load_pipeline_config(config_path)
        assert len(pipeline.stages) == 2
        assert pipeline.stages[0].name == "vision"
        assert pipeline.stages[1].name == "text"
        assert pipeline.stages[0].is_source is True
        assert pipeline.stages[1].is_terminal is True

    def test_load_three_stage_yaml(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "pipeline_three_stage_example.yaml")
        pipeline = load_pipeline_config(config_path)
        assert len(pipeline.stages) == 3
        dag = PipelineDAG(pipeline)
        order = dag.topological_sort()
        assert order.index("vision") < order.index("bridge") < order.index("text")

    def test_parse_dict_directly(self):
        raw = {
            "stages": [
                {"name": "a", "is_source": True, "parallelism": "none", "engine": "native"},
                {"name": "b", "is_terminal": True, "parallelism": "none", "engine": "native"},
            ],
            "edges": [{"src": "a", "dst": "b"}],
            "resource_sets": [{"name": "gpus", "num_gpus": 2}],
            "placements": [
                {"stage": "a", "resource_set": "gpus"},
                {"stage": "b", "resource_set": "gpus"},
            ],
        }
        pipeline = parse_pipeline_dict(raw)
        assert len(pipeline.stages) == 2
        assert pipeline.dp_size == 1

    def test_reject_invalid_parallelism(self):
        raw = {
            "stages": [{"name": "a", "is_source": True, "is_terminal": True, "parallelism": "quantum"}],
            "edges": [],
            "resource_sets": [{"name": "gpus", "num_gpus": 1}],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        with pytest.raises(ConfigLoadError, match="invalid parallelism"):
            parse_pipeline_dict(raw)

    def test_reject_invalid_engine(self):
        raw = {
            "stages": [{"name": "a", "is_source": True, "is_terminal": True, "engine": "turbo"}],
            "edges": [],
            "resource_sets": [{"name": "gpus", "num_gpus": 1}],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        with pytest.raises(ConfigLoadError, match="invalid engine"):
            parse_pipeline_dict(raw)

    def test_reject_unknown_trainer_resolver_key(self):
        raw = {
            "stages": [
                {
                    "name": "a",
                    "is_source": True,
                    "is_terminal": True,
                    "trainer_resolver_key": "nonexistent_trainer",
                }
            ],
            "edges": [],
            "resource_sets": [{"name": "gpus", "num_gpus": 1}],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        with pytest.raises(ConfigLoadError, match="unknown trainer_resolver_key"):
            parse_pipeline_dict(raw)

    def test_reject_missing_stages(self):
        raw = {
            "stages": [],
            "edges": [],
            "resource_sets": [{"name": "gpus", "num_gpus": 1}],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        with pytest.raises(ConfigLoadError, match="at least one stage"):
            parse_pipeline_dict(raw)

    def test_reject_missing_resource_sets(self):
        raw = {
            "stages": [{"name": "a", "is_source": True, "is_terminal": True}],
            "edges": [],
            "resource_sets": [],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        with pytest.raises(ConfigLoadError, match="at least one resource set"):
            parse_pipeline_dict(raw)

    def test_reject_cycle_in_yaml(self):
        content = textwrap.dedent("""\
            stages:
              - name: a
                is_source: true
              - name: b
              - name: c
                is_terminal: true
            edges:
              - src: a
                dst: b
              - src: b
                dst: c
              - src: c
                dst: a
            resource_sets:
              - name: gpus
                num_gpus: 4
            placements:
              - stage: a
                resource_set: gpus
              - stage: b
                resource_set: gpus
              - stage: c
                resource_set: gpus
        """)
        path = _write_yaml(content)
        try:
            pipeline = load_pipeline_config(path)
            dag = PipelineDAG(pipeline)
            errors = dag.validate()
            assert any("ycle" in e for e in errors)
        finally:
            os.unlink(path)

    def test_trainer_registry_resolution(self):
        """Register a trainer and verify YAML config resolves it."""
        register_trainer("test_mlp", trainer_cls=object, model_fn=lambda: None)
        raw = {
            "stages": [
                {
                    "name": "a",
                    "is_source": True,
                    "is_terminal": True,
                    "trainer_resolver_key": "test_mlp",
                }
            ],
            "edges": [],
            "resource_sets": [{"name": "gpus", "num_gpus": 1}],
            "placements": [{"stage": "a", "resource_set": "gpus"}],
        }
        pipeline = parse_pipeline_dict(raw)
        assert pipeline.stages[0].trainer_cls is object


# ── Payload tests ──


@pytest.mark.cpu_only
class TestPayloads:
    def test_stage_outputs(self):
        out = StageOutputs(activations="tensor_data", attention_mask="mask", meta={"key": "val"})
        assert out.activations == "tensor_data"
        d = out.to_dict()
        assert d["activations"] == "tensor_data"
        assert d["attention_mask"] == "mask"
        assert d["meta"]["key"] == "val"

    def test_stage_gradients(self):
        grad = StageGradients(grad="grad_tensor", meta={"step": 1})
        assert grad.grad == "grad_tensor"
        d = grad.to_dict()
        assert d["grad"] == "grad_tensor"
        assert d["meta"]["step"] == 1

    def test_stage_outputs_defaults(self):
        out = StageOutputs(activations="data")
        assert out.attention_mask is None
        assert out.meta == {}

    def test_stage_gradients_defaults(self):
        grad = StageGradients(grad="g")
        assert grad.meta == {}
