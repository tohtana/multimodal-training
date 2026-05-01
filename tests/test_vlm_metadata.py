import pytest

from python.ray.megatron_trainer import MegatronBaseTrainer


class DummyParam:
    grad = None

    def __init__(self, value):
        self.value = float(value)

    def detach(self):
        return self

    def float(self):
        return self

    def norm(self):
        return self

    def item(self):
        return self.value


class DummyModel:
    def __init__(self):
        self.param = DummyParam(1.0)

    def named_parameters(self):
        return [("layers.0.weight", self.param)]


@pytest.mark.cpu_only
def test_optimizer_probe_requires_explicit_parameter_path():
    trainer = MegatronBaseTrainer({"seed": 1}, rank=0)
    trainer.megatron_model = DummyModel()

    with pytest.raises(ValueError, match="parameter_path is required"):
        trainer.get_optimizer_probe_snapshot()


@pytest.mark.cpu_only
def test_optimizer_probe_snapshot_and_verify_update():
    trainer = MegatronBaseTrainer({"seed": 1}, rank=0)
    trainer.megatron_model = DummyModel()
    before = trainer.get_optimizer_probe_snapshot("megatron_model.layers.0.weight")

    trainer.megatron_model.param.value = 1.25
    trainer._optimizer_step_count = 1
    result = trainer.verify_optimizer_update(before)

    assert result["param_norm_changed"] is True
    assert result["iteration_step_counter_advanced"] is True
    assert result["optimizer_update_verified"] is True
