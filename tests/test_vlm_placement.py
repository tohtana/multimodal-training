import pytest

from python.pipeline.stage import EdgeConfig, Pipeline, Placement, ResourceSet, Stage
from python.pipeline.vlm_placement import build_vlm_stage_groups


class FakeActorGroup:
    created = []
    fail_on_call = None

    def __init__(
        self,
        config,
        actor_cls,
        num_actors,
        num_cpus=6,
        num_gpus=1,
        collocate=False,
        placement_group_handle=None,
        actor_init_kwargs=None,
        collocation_factor=2,
    ):
        call_index = len(FakeActorGroup.created)
        if FakeActorGroup.fail_on_call == call_index:
            raise RuntimeError("injected actor group failure")
        self.config = config
        self.actor_cls = actor_cls
        self.num_actors = num_actors
        self.collocate = collocate
        self.placement_group = placement_group_handle or f"pg-{call_index}"
        self._actors = [f"actor-{call_index}-{rank}" for rank in range(num_actors)]
        self.shutdown_called = False
        FakeActorGroup.created.append(self)

    def execute_all_async(self, method_name, *args, **kwargs):
        return []

    def execute_all(self, method_name, *args, **kwargs):
        return []

    def shutdown(self):
        self.shutdown_called = True


def _resolver(stage_name, stage_config):
    return object, {"stage_name": stage_name}


def _gpu_ids(actors):
    return [f"gpu-{actor.split('-')[-1]}" for actor in actors]


def _pipeline(vision_gpus=2, text_gpus=4):
    return Pipeline(
        stages=[Stage(name="vision", is_source=True), Stage(name="text", is_terminal=True)],
        edges=[EdgeConfig(src="vision", dst="text")],
        resource_sets=[
            ResourceSet(name="vision_rs", num_gpus=vision_gpus),
            ResourceSet(name="text_rs", num_gpus=text_gpus),
        ],
        placements=[
            Placement(stage_name="vision", resource_set="vision_rs"),
            Placement(stage_name="text", resource_set="text_rs"),
        ],
    )


@pytest.fixture(autouse=True)
def reset_fake_actor_group():
    FakeActorGroup.created = []
    FakeActorGroup.fail_on_call = None
    yield
    FakeActorGroup.created = []
    FakeActorGroup.fail_on_call = None


@pytest.mark.cpu_only
def test_vlm_stage_groups_use_asymmetric_resource_set_actor_counts():
    pipeline = _pipeline(vision_gpus=2, text_gpus=4)
    plan, groups = build_vlm_stage_groups(
        pipeline,
        {"vision": {"seed": 1}, "text": {"seed": 1}},
        _resolver,
        actor_group_factory=FakeActorGroup,
        gpu_id_collector=_gpu_ids,
    )

    assert groups["vision"].actor_count == 2
    assert groups["text"].actor_count == 4
    assert plan.stage_to_actor_group["vision"] is groups["vision"]
    assert plan.stage_to_actor_group["text"] is groups["text"]


@pytest.mark.cpu_only
def test_vlm_stage_groups_support_symmetric_resource_sets():
    pipeline = _pipeline(vision_gpus=4, text_gpus=4)
    plan, groups = build_vlm_stage_groups(
        pipeline,
        {"vision": {"seed": 1}, "text": {"seed": 1}},
        _resolver,
        actor_group_factory=FakeActorGroup,
        gpu_id_collector=_gpu_ids,
    )

    assert groups["vision"].actor_count == 4
    assert groups["text"].actor_count == 4
    assert plan.actor_gpu_ids["vision_rs"] == {0: "gpu-0", 1: "gpu-1", 2: "gpu-2", 3: "gpu-3"}


@pytest.mark.cpu_only
def test_vlm_stage_groups_preserve_plan_object_identity():
    pipeline = _pipeline()
    plan, groups = build_vlm_stage_groups(
        pipeline,
        {"vision": {"seed": 1}, "text": {"seed": 1}},
        _resolver,
        actor_group_factory=FakeActorGroup,
        gpu_id_collector=_gpu_ids,
    )

    assert plan.resource_set_to_actor_group["vision_rs"] is groups["vision"]
    assert plan.resource_set_to_actor_group["text_rs"] is groups["text"]
    assert plan.stage_to_actor_group["vision"] is plan.resource_set_to_actor_group["vision_rs"]


@pytest.mark.cpu_only
def test_vlm_stage_group_tears_down_created_groups_on_failure():
    FakeActorGroup.fail_on_call = 1
    pipeline = _pipeline()

    with pytest.raises(RuntimeError, match="injected actor group failure"):
        build_vlm_stage_groups(
            pipeline,
            {"vision": {"seed": 1}, "text": {"seed": 1}},
            _resolver,
            actor_group_factory=FakeActorGroup,
            gpu_id_collector=_gpu_ids,
        )

    assert len(FakeActorGroup.created) == 1
    assert FakeActorGroup.created[0].shutdown_called is True
