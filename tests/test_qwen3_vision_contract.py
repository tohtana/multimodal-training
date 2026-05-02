from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from python.ray.qwen3_vision_contract import (
    apply_qwen3_vision_backward,
    build_qwen3_vision_payload,
    collect_qwen3_vision_gradients,
    coerce_qwen3_vision_gradients,
    extract_qwen3_vision_embeddings,
    is_qwen3_vl_model_type,
    make_qwen3_payload_leaf,
    prepare_qwen3_text_inputs,
    QWEN3_VISUAL_PAYLOAD_META_KEY,
    Qwen3VisionPayload,
)
from python.ray.bridge import BridgeProjection
from python.ray.megatron_trainer import _PrecomputedQwen3VisualAdapter
from python.ray.payloads import TextBackwardOutputs, normalize_text_backward_outputs
from python.ray.text import BaseTextTrainer

pytestmark = [pytest.mark.cpu_only]


def _generic_extractor_pick(outputs):
    for attr in ("last_hidden_state", "vision_embeddings", "embeddings"):
        value = getattr(outputs, attr, None)
        if isinstance(value, torch.Tensor):
            return attr, value
    return None, None


def test_qwen3_model_type_predicate():
    assert is_qwen3_vl_model_type("qwen3_vl")
    assert is_qwen3_vl_model_type(" QWEN3_MOE_VL ")
    assert not is_qwen3_vl_model_type("qwen2_5_vl")
    assert not is_qwen3_vl_model_type(None)


def test_qwen3_extractor_selects_pooler_output_and_preserves_deepstack():
    last_hidden_state = torch.randn(4, 8)
    pooler_output = torch.randn(1, 16, requires_grad=True)
    deepstack = [torch.randn(1, 16, requires_grad=True), torch.randn(1, 16, requires_grad=True)]
    outputs = SimpleNamespace(
        last_hidden_state=last_hidden_state,
        pooler_output=pooler_output,
        deepstack_features=deepstack,
    )

    generic_name, generic_tensor = _generic_extractor_pick(outputs)
    assert generic_name == "last_hidden_state"
    assert generic_tensor is last_hidden_state

    selected, meta = extract_qwen3_vision_embeddings(
        outputs,
        image_grid_thw=torch.tensor([[1, 2, 2]], dtype=torch.long),
        spatial_merge_size=2,
        spatial_merge_size_source="test",
    )

    assert selected is pooler_output
    assert meta["selected_tensor"] == "pooler_output"
    assert meta["grid_token_count"] == 1
    assert meta["deepstack_present"] is True
    payload = meta[QWEN3_VISUAL_PAYLOAD_META_KEY]
    assert isinstance(payload, Qwen3VisionPayload)
    assert payload.primary_embeddings is pooler_output
    assert payload.deepstack_visual_embeds == tuple(deepstack)


def test_qwen3_extractor_concatenates_split_pooler_output():
    split_pooler = (torch.randn(1, 16), torch.randn(2, 16))
    deepstack = [torch.randn(3, 16)]
    outputs = {"pooler_output": split_pooler, "deepstack_features": deepstack}

    selected, meta = extract_qwen3_vision_embeddings(
        outputs,
        image_grid_thw=torch.tensor([[1, 2, 2], [1, 2, 4]], dtype=torch.long),
        spatial_merge_size=2,
        spatial_merge_size_source="test",
    )

    assert selected.shape == (3, 16)
    torch.testing.assert_close(selected, torch.cat(split_pooler, dim=0))
    assert meta["image_token_counts"] == (1, 2)


def test_qwen3_token_count_and_hidden_size_validation():
    payload = Qwen3VisionPayload(
        primary_embeddings=torch.randn(2, 16),
        image_token_counts=(2,),
        spatial_merge_size=2,
    )
    input_ids = torch.tensor([[1, 7, 2]], dtype=torch.long)
    inputs_embeds = torch.randn(1, 3, 16)

    with pytest.raises(RuntimeError, match="token mismatch"):
        prepare_qwen3_text_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            payload=payload,
            image_token_id=7,
            video_token_id=8,
        )

    hidden_mismatch = Qwen3VisionPayload(
        primary_embeddings=torch.randn(1, 15),
        image_token_counts=(1,),
        spatial_merge_size=2,
    )
    with pytest.raises(RuntimeError, match="hidden size"):
        prepare_qwen3_text_inputs(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            payload=hidden_mismatch,
            image_token_id=7,
            video_token_id=8,
        )


def test_qwen3_megatron_adapter_preserves_token_order_deepstack_kwargs_and_gradients():
    image_token_id = 7
    video_token_id = 8
    primary = torch.tensor(
        [
            [10.0, 11.0, 12.0, 13.0],
            [20.0, 21.0, 22.0, 23.0],
            [30.0, 31.0, 32.0, 33.0],
        ]
    )
    deepstack0 = primary + 100.0
    deepstack1 = primary + 200.0
    payload = Qwen3VisionPayload(
        primary_embeddings=primary,
        deepstack_visual_embeds=(deepstack0, deepstack1),
        image_token_counts=(2,),
        video_token_counts=(1,),
    )
    adapter = _PrecomputedQwen3VisualAdapter(
        payload,
        primary_embeddings=primary,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
    )
    input_ids = torch.tensor([[image_token_id, video_token_id, image_token_id, 3]], dtype=torch.long)
    inputs_embeds = torch.zeros(1, 4, 4)

    result = adapter.get_inputs_embeds(inputs_embeds, input_ids=input_ids)

    expected_inputs = inputs_embeds.clone()
    expected_inputs[0, 0] = primary[0]
    expected_inputs[0, 1] = primary[2]
    expected_inputs[0, 2] = primary[1]
    torch.testing.assert_close(result["inputs_embeds"], expected_inputs)

    expected_visual_mask = (input_ids == image_token_id) | (input_ids == video_token_id)
    torch.testing.assert_close(result["visual_pos_masks"], expected_visual_mask.transpose(0, 1))
    assert result["deepstack_visual_embeds"].shape == (2, 3, 4)
    torch.testing.assert_close(
        result["deepstack_visual_embeds"][0],
        torch.stack((deepstack0[0], deepstack0[2], deepstack0[1]), dim=0),
    )
    torch.testing.assert_close(
        result["deepstack_visual_embeds"][1],
        torch.stack((deepstack1[0], deepstack1[2], deepstack1[1]), dim=0),
    )

    hidden_states = result["inputs_embeds"].transpose(0, 1).contiguous()
    visual_hidden_states = hidden_states[result["visual_pos_masks"], :]
    loss = visual_hidden_states.square().sum() + result["deepstack_visual_embeds"].square().sum()
    loss.backward()

    gradients = collect_qwen3_vision_gradients(adapter.payload)
    assert gradients.primary_grad.shape == primary.shape
    assert len(gradients.deepstack_grads) == 2
    assert gradients.primary_grad.abs().sum().item() > 0
    assert gradients.deepstack_grads[0].abs().sum().item() > 0
    assert gradients.deepstack_grads[1].abs().sum().item() > 0


def test_qwen3_structured_gradients_survive_legacy_backward_payload_normalization():
    leaf_payload = make_qwen3_payload_leaf(
        Qwen3VisionPayload(
            primary_embeddings=torch.randn(1, 4),
            deepstack_visual_embeds=(torch.randn(1, 4),),
            image_token_counts=(1,),
        )
    )
    loss = leaf_payload.primary_embeddings.sum() + sum(tensor.sum() for tensor in leaf_payload.deepstack_visual_embeds)
    loss.backward()
    gradients = collect_qwen3_vision_gradients(leaf_payload)
    payload = TextBackwardOutputs(grad=gradients, meta={"qwen3_structured_grad_transport": "ray_object_fallback"})
    normalized = normalize_text_backward_outputs(payload.to_dict())

    assert normalized.grad is gradients
    assert normalized.meta["qwen3_structured_grad_transport"] == "ray_object_fallback"
    assert coerce_qwen3_vision_gradients(normalized.grad).primary_grad.shape == (1, 4)


class _TextBackwardFallbackTrainer(BaseTextTrainer):
    def _create_model_and_lm_head(self, model_config):
        return None, None

    def _get_embedding_module(self, model):
        return None


def test_qwen3_structured_backward_uses_ray_fallback_when_ipc_is_enabled():
    trainer = _TextBackwardFallbackTrainer({"seed": 1234}, rank=0)
    leaf_payload = make_qwen3_payload_leaf(
        Qwen3VisionPayload(
            primary_embeddings=torch.randn(1, 4),
            deepstack_visual_embeds=(torch.randn(1, 4),),
            image_token_counts=(1,),
        )
    )
    trainer.loss = leaf_payload.primary_embeddings.sum() + sum(
        tensor.sum() for tensor in leaf_payload.deepstack_visual_embeds
    )
    trainer._qwen3_visual_payload = leaf_payload
    trainer.vision_embeddings = leaf_payload.primary_embeddings
    trainer.use_ipc = True
    trainer.receiver_gpu_ids = ["same-gpu"]

    result = trainer.backward_step()

    assert isinstance(result.grad, type(collect_qwen3_vision_gradients(leaf_payload)))
    assert result.meta["qwen3_structured_grad_transport"] == "ray_object_fallback"
    assert result.grad.primary_grad.shape == leaf_payload.primary_embeddings.shape
    assert len(result.grad.deepstack_grads) == len(leaf_payload.deepstack_visual_embeds)


def test_qwen3_structured_bridge_projection_returns_primary_and_deepstack_gradients():
    torch.manual_seed(7)
    original_payload = Qwen3VisionPayload(
        primary_embeddings=torch.randn(2, 4),
        deepstack_visual_embeds=(torch.randn(2, 4), torch.randn(2, 4)),
        image_token_counts=(2,),
    )
    bridge = BridgeProjection(input_dim=4, output_dim=6, hidden_dim=5)
    bridge_input = make_qwen3_payload_leaf(original_payload)

    projected_payload = Qwen3VisionPayload(
        primary_embeddings=bridge(bridge_input.primary_embeddings),
        deepstack_visual_embeds=tuple(bridge(tensor) for tensor in bridge_input.deepstack_visual_embeds),
        image_token_counts=bridge_input.image_token_counts,
    )
    downstream_leaf = make_qwen3_payload_leaf(projected_payload)
    downstream_loss = downstream_leaf.primary_embeddings.square().sum()
    downstream_loss = downstream_loss + sum(tensor.square().sum() for tensor in downstream_leaf.deepstack_visual_embeds)
    downstream_loss.backward()

    downstream_grads = collect_qwen3_vision_gradients(downstream_leaf)
    apply_qwen3_vision_backward(projected_payload, downstream_grads)
    upstream_grads = collect_qwen3_vision_gradients(bridge_input)

    assert upstream_grads.primary_grad.shape == original_payload.primary_embeddings.shape
    assert len(upstream_grads.deepstack_grads) == len(original_payload.deepstack_visual_embeds)
    for grad, tensor in zip(upstream_grads.deepstack_grads, original_payload.deepstack_visual_embeds):
        assert grad.shape == tensor.shape
        assert grad.abs().sum().item() > 0


def _tiny_qwen3_model():
    pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import (
        Qwen3VLConfig,
        Qwen3VLTextConfig,
        Qwen3VLVisionConfig,
    )
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    text_config = Qwen3VLTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        use_cache=False,
        attention_dropout=0.0,
    )
    vision_config = Qwen3VLVisionConfig(
        hidden_size=8,
        out_hidden_size=16,
        depth=2,
        num_heads=2,
        intermediate_size=16,
        spatial_merge_size=2,
        patch_size=2,
        temporal_patch_size=1,
        in_channels=3,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1],
    )
    config = Qwen3VLConfig(
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=7,
        video_token_id=8,
        tie_word_embeddings=False,
    )
    return Qwen3VLForConditionalGeneration(config)


def _split_qwen3_forward(model, input_ids, labels, pixel_values, image_grid_thw):
    vision_output = model.model.visual(pixel_values, grid_thw=image_grid_thw, return_dict=True)
    original_payload = build_qwen3_vision_payload(
        vision_output,
        image_grid_thw=image_grid_thw,
        spatial_merge_size=model.model.visual.spatial_merge_size,
        spatial_merge_size_source="model.model.visual.spatial_merge_size",
    )
    text_payload = make_qwen3_payload_leaf(original_payload)

    inputs_embeds = model.get_input_embeddings()(input_ids)
    prepared = prepare_qwen3_text_inputs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        payload=text_payload,
        image_token_id=model.config.image_token_id,
        video_token_id=model.config.video_token_id,
    )
    position_ids, _ = model.model.get_rope_index(input_ids, image_grid_thw, None, attention_mask=None)
    text_outputs = model.model.language_model(
        input_ids=None,
        inputs_embeds=prepared.inputs_embeds,
        position_ids=position_ids,
        visual_pos_masks=prepared.visual_pos_masks,
        deepstack_visual_embeds=list(prepared.deepstack_visual_embeds),
        use_cache=False,
    )
    logits = model.lm_head(text_outputs.last_hidden_state)
    loss = model.loss_function(logits=logits, labels=labels, vocab_size=model.config.text_config.vocab_size)
    return logits, loss, original_payload, text_payload


def test_tiny_qwen3_split_matches_monolithic_with_deepstack_and_gradients():
    torch.manual_seed(20260502)
    model = _tiny_qwen3_model()
    model.eval()

    input_ids = torch.tensor([[1, 7, 2, 3, 4]], dtype=torch.long)
    labels = input_ids.clone()
    labels[input_ids == model.config.image_token_id] = -100
    image_grid_thw = torch.tensor([[1, 2, 2]], dtype=torch.long)
    pixel_values = torch.randn(1, 3, 1, 4, 4)

    with torch.no_grad():
        monolithic = model(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values.clone(),
            image_grid_thw=image_grid_thw,
            use_cache=False,
        )

    pixel_values_split = pixel_values.clone().requires_grad_(True)
    split_logits, split_loss, original_payload, text_payload = _split_qwen3_forward(
        model,
        input_ids,
        labels,
        pixel_values_split,
        image_grid_thw,
    )

    torch.testing.assert_close(split_logits, monolithic.logits, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(split_loss, monolithic.loss, rtol=1e-5, atol=1e-6)
    assert len(text_payload.deepstack_visual_embeds) == 2

    split_loss.backward()
    gradients = collect_qwen3_vision_gradients(text_payload)
    assert gradients.primary_grad.shape == original_payload.primary_embeddings.shape
    assert len(gradients.deepstack_grads) == len(original_payload.deepstack_visual_embeds)
    for grad, tensor in zip(gradients.deepstack_grads, original_payload.deepstack_visual_embeds):
        assert grad.shape == tensor.shape

    assert pixel_values_split.grad is None
    apply_qwen3_vision_backward(original_payload, gradients)
    assert pixel_values_split.grad is not None
    assert pixel_values_split.grad.abs().sum().item() > 0
