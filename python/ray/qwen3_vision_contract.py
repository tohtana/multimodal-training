from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


QWEN3_VISUAL_PAYLOAD_META_KEY = "qwen3_visual_payload"


@dataclass(frozen=True)
class Qwen3VisionPayload:
    """Structured Qwen3-VL visual payload consumed by the text stage.

    `primary_embeddings` is the post-merger `pooler_output` tensor. Deepstack
    tensors are post-merger tensors in the same flat media-token order.
    """

    primary_embeddings: torch.Tensor
    deepstack_visual_embeds: tuple[torch.Tensor, ...] = ()
    image_grid_thw: torch.Tensor | None = None
    video_grid_thw: torch.Tensor | None = None
    image_token_counts: tuple[int, ...] = ()
    video_token_counts: tuple[int, ...] = ()
    spatial_merge_size: int = 1
    spatial_merge_size_source: str = "unknown"
    selected_tensor: str = "pooler_output"
    last_hidden_state_shape: tuple[int, ...] | None = None

    @property
    def image_token_count(self) -> int:
        return sum(self.image_token_counts)

    @property
    def video_token_count(self) -> int:
        return sum(self.video_token_counts)

    @property
    def total_token_count(self) -> int:
        return self.image_token_count + self.video_token_count


@dataclass(frozen=True)
class Qwen3PreparedTextInputs:
    inputs_embeds: torch.Tensor
    visual_pos_masks: torch.Tensor | None
    deepstack_visual_embeds: tuple[torch.Tensor, ...]
    payload: Qwen3VisionPayload


@dataclass(frozen=True)
class Qwen3VisionGradients:
    primary_grad: torch.Tensor
    deepstack_grads: tuple[torch.Tensor, ...] = ()


def is_qwen3_vl_model_type(model_type: str | None) -> bool:
    return (model_type or "").strip().lower() in {"qwen3_vl", "qwen3_moe_vl"}


def get_qwen3_visual_payload(meta: dict[str, Any] | None) -> Qwen3VisionPayload | None:
    if not meta:
        return None
    payload = meta.get(QWEN3_VISUAL_PAYLOAD_META_KEY)
    if payload is None:
        return None
    if not isinstance(payload, Qwen3VisionPayload):
        raise RuntimeError(f"Invalid Qwen3 visual payload type: {type(payload)}")
    return payload


def _get_output_value(output: Any, key: str) -> Any:
    if isinstance(output, dict):
        return output.get(key)
    value = getattr(output, key, None)
    if value is not None:
        return value
    if isinstance(output, (list, tuple)):
        # ms-swift's Qwen3 visual wrapper returns (pooler_output, deepstack_features).
        if key == "pooler_output" and output and isinstance(output[0], torch.Tensor):
            return output[0]
        if key in {"deepstack_features", "deepstack_visual_embeds"} and len(output) > 1:
            return output[1]
    return None


def _flatten_pooler_output(pooler_output: Any) -> torch.Tensor:
    if isinstance(pooler_output, torch.Tensor):
        return pooler_output
    if isinstance(pooler_output, (list, tuple)):
        tensors = [item for item in pooler_output if isinstance(item, torch.Tensor)]
        if len(tensors) != len(pooler_output):
            raise RuntimeError("Qwen3 pooler_output list/tuple contains a non-tensor item.")
        if not tensors:
            raise RuntimeError("Qwen3 pooler_output list/tuple is empty.")
        return torch.cat(tensors, dim=0)
    raise RuntimeError(f"Qwen3 pooler_output must be a tensor or tensor sequence, got {type(pooler_output)}.")


def _as_tensor_tuple(value: Any, *, name: str) -> tuple[torch.Tensor, ...]:
    if value is None:
        return ()
    if isinstance(value, torch.Tensor):
        if value.dim() == 3:
            return tuple(value.unbind(0))
        return (value,)
    if isinstance(value, (list, tuple)):
        tensors: list[torch.Tensor] = []
        for item in value:
            if not isinstance(item, torch.Tensor):
                raise RuntimeError(f"{name} contains a non-tensor item: {type(item)}")
            tensors.append(item)
        return tuple(tensors)
    raise RuntimeError(f"{name} must be a tensor or tensor sequence, got {type(value)}.")


def _clone_grid(grid: torch.Tensor | None) -> torch.Tensor | None:
    if grid is None:
        return None
    if not isinstance(grid, torch.Tensor):
        grid = torch.as_tensor(grid, dtype=torch.long)
    return grid.detach().clone()


def qwen3_grid_token_counts(grid_thw: torch.Tensor | None, spatial_merge_size: int) -> tuple[int, ...]:
    if grid_thw is None:
        return ()
    if spatial_merge_size <= 0:
        raise RuntimeError(f"spatial_merge_size must be positive, got {spatial_merge_size}.")
    if grid_thw.dim() != 2 or grid_thw.shape[-1] != 3:
        raise RuntimeError(f"grid_thw must have shape [num_media, 3], got {tuple(grid_thw.shape)}.")
    merge_area = int(spatial_merge_size) ** 2
    counts = grid_thw.detach().cpu().prod(dim=-1) // merge_area
    return tuple(int(count.item()) for count in counts)


def build_qwen3_vision_payload(
    output: Any,
    *,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    spatial_merge_size_source: str = "unknown",
    video_grid_thw: torch.Tensor | None = None,
) -> Qwen3VisionPayload:
    pooler_output = _get_output_value(output, "pooler_output")
    if pooler_output is None:
        raise RuntimeError("Qwen3 vision output missing required pooler_output.")
    primary_embeddings = _flatten_pooler_output(pooler_output)

    deepstack_value = _get_output_value(output, "deepstack_features")
    if deepstack_value is None:
        deepstack_value = _get_output_value(output, "deepstack_visual_embeds")
    deepstack_visual_embeds = _as_tensor_tuple(deepstack_value, name="deepstack_features")

    image_counts = qwen3_grid_token_counts(image_grid_thw, spatial_merge_size)
    video_counts = qwen3_grid_token_counts(video_grid_thw, spatial_merge_size)
    expected_tokens = sum(image_counts) + sum(video_counts)
    if expected_tokens and primary_embeddings.shape[0] != expected_tokens:
        raise RuntimeError(
            "Qwen3 pooler_output token count mismatch: "
            f"selected={primary_embeddings.shape[0]}, grid_expected={expected_tokens}, "
            f"image_counts={image_counts}, video_counts={video_counts}, "
            f"spatial_merge_size={spatial_merge_size}."
        )

    for index, tensor in enumerate(deepstack_visual_embeds):
        if tensor.shape[0] != primary_embeddings.shape[0]:
            raise RuntimeError(
                f"Qwen3 deepstack feature {index} token count {tensor.shape[0]} "
                f"does not match primary token count {primary_embeddings.shape[0]}."
            )
        if tensor.shape[-1] != primary_embeddings.shape[-1]:
            raise RuntimeError(
                f"Qwen3 deepstack feature {index} hidden size {tensor.shape[-1]} "
                f"does not match primary hidden size {primary_embeddings.shape[-1]}."
            )

    last_hidden_state = _get_output_value(output, "last_hidden_state")
    last_hidden_state_shape = tuple(last_hidden_state.shape) if isinstance(last_hidden_state, torch.Tensor) else None

    return Qwen3VisionPayload(
        primary_embeddings=primary_embeddings,
        deepstack_visual_embeds=deepstack_visual_embeds,
        image_grid_thw=_clone_grid(image_grid_thw),
        video_grid_thw=_clone_grid(video_grid_thw),
        image_token_counts=image_counts,
        video_token_counts=video_counts,
        spatial_merge_size=int(spatial_merge_size),
        spatial_merge_size_source=spatial_merge_size_source,
        last_hidden_state_shape=last_hidden_state_shape,
    )


def qwen3_payload_metadata(payload: Qwen3VisionPayload) -> dict[str, Any]:
    return {
        QWEN3_VISUAL_PAYLOAD_META_KEY: payload,
        "selected_tensor": payload.selected_tensor,
        "selected_shape": tuple(payload.primary_embeddings.shape),
        "selected_hidden_size": int(payload.primary_embeddings.shape[-1]),
        "image_grid_thw_shape": tuple(payload.image_grid_thw.shape) if payload.image_grid_thw is not None else None,
        "video_grid_thw_shape": tuple(payload.video_grid_thw.shape) if payload.video_grid_thw is not None else None,
        "grid_token_count": payload.total_token_count,
        "image_token_counts": payload.image_token_counts,
        "video_token_counts": payload.video_token_counts,
        "spatial_merge_size": payload.spatial_merge_size,
        "spatial_merge_size_source": payload.spatial_merge_size_source,
        "deepstack_present": bool(payload.deepstack_visual_embeds),
        "deepstack_count": len(payload.deepstack_visual_embeds),
        "qwen3_primary_embedding_only": not bool(payload.deepstack_visual_embeds),
        "last_hidden_state_shape": payload.last_hidden_state_shape,
    }


def extract_qwen3_vision_embeddings(
    output: Any,
    *,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    spatial_merge_size_source: str = "unknown",
    video_grid_thw: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    payload = build_qwen3_vision_payload(
        output,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        spatial_merge_size=spatial_merge_size,
        spatial_merge_size_source=spatial_merge_size_source,
    )
    return payload.primary_embeddings, qwen3_payload_metadata(payload)


def make_qwen3_payload_leaf(
    payload: Qwen3VisionPayload,
    *,
    primary_embeddings: torch.Tensor | None = None,
    clone: bool = True,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> Qwen3VisionPayload:
    def make_leaf(tensor: torch.Tensor) -> torch.Tensor:
        to_kwargs: dict[str, Any] = {}
        if device is not None:
            to_kwargs["device"] = device
        if dtype is not None and tensor.is_floating_point():
            to_kwargs["dtype"] = dtype
        if to_kwargs:
            tensor = tensor.to(**to_kwargs)
        tensor = tensor.detach()
        if clone:
            tensor = tensor.clone()
        tensor = tensor.requires_grad_(True)
        tensor.retain_grad()
        return tensor

    primary_source = payload.primary_embeddings if primary_embeddings is None else primary_embeddings
    primary = make_leaf(primary_source)

    deepstack = tuple(make_leaf(tensor) for tensor in payload.deepstack_visual_embeds)
    return replace(payload, primary_embeddings=primary, deepstack_visual_embeds=deepstack)


def _split_media_embeddings(payload: Qwen3VisionPayload) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    image_tokens = payload.image_token_count
    video_tokens = payload.video_token_count
    primary = payload.primary_embeddings

    if image_tokens and video_tokens:
        return primary[:image_tokens], primary[image_tokens : image_tokens + video_tokens]
    if image_tokens:
        return primary[:image_tokens], None
    if video_tokens:
        return None, primary[:video_tokens]
    return primary, None


def _validate_hidden_size(features: torch.Tensor, inputs_embeds: torch.Tensor, label: str) -> None:
    if features.shape[-1] != inputs_embeds.shape[-1]:
        raise RuntimeError(
            f"Qwen3 {label} hidden size {features.shape[-1]} does not match text hidden size "
            f"{inputs_embeds.shape[-1]}; configure a structured bridge/projection first."
        )


def _scatter_features(
    inputs_embeds: torch.Tensor,
    mask: torch.Tensor,
    features: torch.Tensor,
    *,
    label: str,
) -> torch.Tensor:
    token_count = int(mask.sum().item())
    if token_count != features.shape[0]:
        raise RuntimeError(
            f"Qwen3 {label} token mismatch: {token_count} placeholders but {features.shape[0]} features."
        )
    _validate_hidden_size(features, inputs_embeds, label)
    expanded_mask = mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
    features = features.to(inputs_embeds.device, inputs_embeds.dtype)
    return inputs_embeds.masked_scatter(expanded_mask, features)


def prepare_qwen3_text_inputs(
    *,
    input_ids: torch.Tensor,
    inputs_embeds: torch.Tensor,
    payload: Qwen3VisionPayload,
    image_token_id: int,
    video_token_id: int | None = None,
) -> Qwen3PreparedTextInputs:
    image_features, video_features = _split_media_embeddings(payload)
    image_mask = input_ids == image_token_id
    if image_features is not None:
        inputs_embeds = _scatter_features(inputs_embeds, image_mask, image_features, label="image")
    elif image_mask.any():
        raise RuntimeError("Qwen3 input_ids contain image placeholders but the visual payload has no image features.")

    video_mask = torch.zeros_like(image_mask, dtype=torch.bool)
    if video_token_id is not None:
        video_mask = input_ids == video_token_id
    if video_features is not None:
        inputs_embeds = _scatter_features(inputs_embeds, video_mask, video_features, label="video")
    elif video_mask.any():
        raise RuntimeError("Qwen3 input_ids contain video placeholders but the visual payload has no video features.")

    visual_pos_masks = None
    deepstack_visual_embeds: tuple[torch.Tensor, ...] = ()
    if image_features is not None or video_features is not None:
        visual_pos_masks = image_mask | video_mask

    if payload.deepstack_visual_embeds:
        prepared_deepstack = []
        for index, tensor in enumerate(payload.deepstack_visual_embeds):
            _validate_hidden_size(tensor, inputs_embeds, f"deepstack[{index}]")
            prepared_deepstack.append(tensor.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype))

        if image_features is not None and video_features is not None:
            image_tokens = payload.image_token_count
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            joined: list[torch.Tensor] = []
            for tensor in prepared_deepstack:
                image_tensor = tensor[:image_tokens]
                video_tensor = tensor[image_tokens:]
                joint = tensor.new_zeros((int(visual_pos_masks.sum().item()), tensor.shape[-1]))
                joint[image_mask_joint.to(joint.device), :] = image_tensor
                joint[video_mask_joint.to(joint.device), :] = video_tensor
                joined.append(joint)
            deepstack_visual_embeds = tuple(joined)
        elif image_features is not None:
            deepstack_visual_embeds = tuple(tensor[: payload.image_token_count] for tensor in prepared_deepstack)
        elif video_features is not None:
            deepstack_visual_embeds = tuple(tensor[: payload.video_token_count] for tensor in prepared_deepstack)

    return Qwen3PreparedTextInputs(
        inputs_embeds=inputs_embeds,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        payload=payload,
    )


def collect_qwen3_vision_gradients(payload: Qwen3VisionPayload) -> Qwen3VisionGradients:
    if payload.primary_embeddings.grad is None:
        raise RuntimeError("Qwen3 primary vision embedding gradient is None after text backward.")
    primary_grad = payload.primary_embeddings.grad.detach().clone()

    deepstack_grads: list[torch.Tensor] = []
    for index, tensor in enumerate(payload.deepstack_visual_embeds):
        if tensor.grad is None:
            raise RuntimeError(f"Qwen3 deepstack vision embedding gradient {index} is None after text backward.")
        deepstack_grads.append(tensor.grad.detach().clone())

    return Qwen3VisionGradients(primary_grad=primary_grad, deepstack_grads=tuple(deepstack_grads))


def coerce_qwen3_vision_gradients(gradients: Qwen3VisionGradients | dict[str, Any]) -> Qwen3VisionGradients:
    if isinstance(gradients, Qwen3VisionGradients):
        return gradients
    if isinstance(gradients, dict):
        primary = gradients.get("primary_grad")
        deepstack = gradients.get("deepstack_grads", ())
        if primary is None:
            raise RuntimeError("Qwen3 gradient dict missing primary_grad.")
        return Qwen3VisionGradients(primary_grad=primary, deepstack_grads=tuple(deepstack))
    raise RuntimeError(f"Unsupported Qwen3 gradient payload type: {type(gradients)}")


def apply_qwen3_vision_backward(
    payload: Qwen3VisionPayload,
    gradients: Qwen3VisionGradients | dict[str, Any],
) -> None:
    grad_payload = coerce_qwen3_vision_gradients(gradients)
    if len(grad_payload.deepstack_grads) != len(payload.deepstack_visual_embeds):
        raise RuntimeError(
            "Qwen3 deepstack gradient count mismatch: "
            f"{len(grad_payload.deepstack_grads)} gradients for {len(payload.deepstack_visual_embeds)} tensors."
        )

    tensors = [payload.primary_embeddings, *payload.deepstack_visual_embeds]
    grads = [grad_payload.primary_grad, *grad_payload.deepstack_grads]
    for tensor, grad in zip(tensors, grads):
        if tuple(tensor.shape) != tuple(grad.shape):
            raise RuntimeError(f"Qwen3 gradient shape {tuple(grad.shape)} does not match tensor {tuple(tensor.shape)}.")
    torch.autograd.backward(tensors, grad_tensors=grads, retain_graph=False)
