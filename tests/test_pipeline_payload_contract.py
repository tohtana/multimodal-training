"""M5: Payload contract tests — generic StageOutputs/StageGradients are interchangeable
with legacy VisionOutputs/TextBackwardOutputs.

CPU-only tests verifying:
  1. StageOutputs.embeddings property returns activations (backward compat)
  2. VisionOutputs.activations property returns embeddings (forward compat)
  3. normalize_vision_outputs accepts StageOutputs and returns VisionOutputs
  4. normalize_text_backward_outputs accepts StageGradients and returns TextBackwardOutputs
  5. normalize_stage_outputs accepts VisionOutputs and returns StageOutputs
  6. normalize_stage_gradients accepts TextBackwardOutputs and returns StageGradients
  7. Dict normalization works for both legacy and generic key formats
"""

import pytest
import torch

from python.ray.payloads import (
    StageGradients,
    StageOutputs,
    TextBackwardOutputs,
    VisionOutputs,
    normalize_stage_gradients,
    normalize_stage_outputs,
    normalize_text_backward_outputs,
    normalize_vision_outputs,
)

pytestmark = [pytest.mark.cpu_only]


class TestStageOutputsCompat:
    def test_stage_outputs_embeddings_property(self):
        """StageOutputs.embeddings returns activations for backward compat."""
        t = torch.randn(2, 3)
        so = StageOutputs(activations=t, meta={"key": 1})
        assert so.embeddings is t
        assert so.activations is t

    def test_vision_outputs_activations_property(self):
        """VisionOutputs.activations returns embeddings for forward compat."""
        t = torch.randn(2, 3)
        vo = VisionOutputs(embeddings=t, meta={"key": 1})
        assert vo.activations is t
        assert vo.embeddings is t

    def test_stage_gradients_matches_text_backward(self):
        """StageGradients and TextBackwardOutputs have same fields."""
        t = torch.randn(2, 3)
        sg = StageGradients(grad=t, meta={"key": 1})
        tbo = TextBackwardOutputs(grad=t, meta={"key": 1})
        assert sg.grad is t
        assert tbo.grad is t


class TestNormalizeVisionOutputs:
    def test_from_vision_outputs(self):
        vo = VisionOutputs(embeddings="emb", attention_mask="mask", meta={"a": 1})
        result = normalize_vision_outputs(vo)
        assert isinstance(result, VisionOutputs)
        assert result.embeddings == "emb"
        assert result.attention_mask == "mask"

    def test_from_stage_outputs(self):
        so = StageOutputs(activations="emb", attention_mask="mask", meta={"a": 1})
        result = normalize_vision_outputs(so)
        assert isinstance(result, VisionOutputs)
        assert result.embeddings == "emb"
        assert result.attention_mask == "mask"

    def test_from_legacy_dict(self):
        d = {"vision_embeddings": "emb", "vision_attention_mask": "mask", "meta": {"a": 1}}
        result = normalize_vision_outputs(d)
        assert isinstance(result, VisionOutputs)
        assert result.embeddings == "emb"

    def test_from_generic_dict(self):
        d = {"activations": "emb", "attention_mask": "mask", "meta": {"a": 1}}
        result = normalize_vision_outputs(d)
        assert isinstance(result, VisionOutputs)
        assert result.embeddings == "emb"

    def test_dict_missing_key_raises(self):
        with pytest.raises(RuntimeError, match="missing"):
            normalize_vision_outputs({"unrelated": 1})


class TestNormalizeTextBackward:
    def test_from_text_backward(self):
        tbo = TextBackwardOutputs(grad="g", meta={"b": 2})
        result = normalize_text_backward_outputs(tbo)
        assert isinstance(result, TextBackwardOutputs)
        assert result.grad == "g"

    def test_from_stage_gradients(self):
        sg = StageGradients(grad="g", meta={"b": 2})
        result = normalize_text_backward_outputs(sg)
        assert isinstance(result, TextBackwardOutputs)
        assert result.grad == "g"

    def test_from_dict(self):
        d = {"grad": "g", "meta": {"b": 2}}
        result = normalize_text_backward_outputs(d)
        assert isinstance(result, TextBackwardOutputs)
        assert result.grad == "g"


class TestNormalizeStageOutputs:
    def test_from_stage_outputs(self):
        so = StageOutputs(activations="act", meta={"c": 3})
        result = normalize_stage_outputs(so)
        assert isinstance(result, StageOutputs)
        assert result.activations == "act"

    def test_from_vision_outputs(self):
        vo = VisionOutputs(embeddings="emb", attention_mask="mask", meta={"c": 3})
        result = normalize_stage_outputs(vo)
        assert isinstance(result, StageOutputs)
        assert result.activations == "emb"
        assert result.attention_mask == "mask"

    def test_from_generic_dict(self):
        d = {"activations": "act", "attention_mask": "mask"}
        result = normalize_stage_outputs(d)
        assert isinstance(result, StageOutputs)
        assert result.activations == "act"

    def test_from_legacy_dict(self):
        d = {"vision_embeddings": "emb", "vision_attention_mask": "mask"}
        result = normalize_stage_outputs(d)
        assert isinstance(result, StageOutputs)
        assert result.activations == "emb"


class TestNormalizeStageGradients:
    def test_from_stage_gradients(self):
        sg = StageGradients(grad="g", meta={"d": 4})
        result = normalize_stage_gradients(sg)
        assert isinstance(result, StageGradients)
        assert result.grad == "g"

    def test_from_text_backward(self):
        tbo = TextBackwardOutputs(grad="g", meta={"d": 4})
        result = normalize_stage_gradients(tbo)
        assert isinstance(result, StageGradients)
        assert result.grad == "g"

    def test_from_dict(self):
        d = {"grad": "g", "meta": {"d": 4}}
        result = normalize_stage_gradients(d)
        assert isinstance(result, StageGradients)
        assert result.grad == "g"


class TestToDict:
    def test_stage_outputs_to_dict(self):
        so = StageOutputs(activations="act", attention_mask="mask", meta={"k": 1})
        d = so.to_dict()
        assert d["activations"] == "act"
        assert d["attention_mask"] == "mask"

    def test_vision_outputs_to_dict_legacy_keys(self):
        vo = VisionOutputs(embeddings="emb", meta={"k": 1})
        d = vo.to_dict()
        assert d["vision_embeddings"] == "emb"

    def test_stage_gradients_to_dict(self):
        sg = StageGradients(grad="g", meta={"k": 1})
        d = sg.to_dict()
        assert d["grad"] == "g"
