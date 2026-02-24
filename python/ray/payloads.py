from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Generic pipeline payloads (design §4.3) ──


@dataclass(frozen=True)
class StageOutputs:
    """Generic forward payload between pipeline stages."""

    activations: Any
    attention_mask: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def embeddings(self) -> Any:
        """Backward-compatible alias for activations (used by legacy VisionOutputs callers)."""
        return self.activations

    def to_dict(self) -> dict[str, Any]:
        return {
            "activations": self.activations,
            "attention_mask": self.attention_mask,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class StageGradients:
    """Generic backward payload between pipeline stages."""

    grad: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"grad": self.grad, "meta": dict(self.meta)}


# ── Legacy payloads (deprecated — use StageOutputs / StageGradients) ──


@dataclass(frozen=True)
class VisionOutputs:
    """Deprecated: use StageOutputs instead. Kept for backward compatibility."""

    embeddings: Any
    attention_mask: Any | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def activations(self) -> Any:
        """Forward-compatible alias mapping to StageOutputs.activations."""
        return self.embeddings

    def to_dict(self) -> dict[str, Any]:
        return {
            "vision_embeddings": self.embeddings,
            "vision_attention_mask": self.attention_mask,
            "meta": dict(self.meta),
        }


# TextBackwardOutputs is structurally identical to StageGradients
@dataclass(frozen=True)
class TextBackwardOutputs:
    """Deprecated: use StageGradients instead. Kept for backward compatibility."""

    grad: Any
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"grad": self.grad, "meta": dict(self.meta)}


def normalize_vision_outputs(payload: Any) -> VisionOutputs:
    """Normalize legacy dicts, StageOutputs, or VisionOutputs into VisionOutputs."""
    if isinstance(payload, VisionOutputs):
        return payload

    if isinstance(payload, StageOutputs):
        return VisionOutputs(
            embeddings=payload.activations,
            attention_mask=payload.attention_mask,
            meta=dict(payload.meta),
        )

    if isinstance(payload, dict):
        # Support both legacy ("vision_embeddings") and generic ("activations") dict keys
        if "vision_embeddings" in payload:
            embeddings = payload["vision_embeddings"]
        elif "activations" in payload:
            embeddings = payload["activations"]
        else:
            raise RuntimeError("Vision payload dict missing 'vision_embeddings' or 'activations'.")
        meta = dict(payload.get("meta") or {})
        for key in ("sample_index", "iteration", "forward_time_ms"):
            if key in payload and key not in meta:
                meta[key] = payload[key]
        return VisionOutputs(
            embeddings=embeddings,
            attention_mask=payload.get("vision_attention_mask") or payload.get("attention_mask"),
            meta=meta,
        )

    raise RuntimeError(f"Unsupported vision payload type: {type(payload)}")


def normalize_text_backward_outputs(payload: Any) -> TextBackwardOutputs:
    """Normalize legacy dicts, StageGradients, or TextBackwardOutputs into TextBackwardOutputs."""
    if isinstance(payload, TextBackwardOutputs):
        return payload

    if isinstance(payload, StageGradients):
        return TextBackwardOutputs(grad=payload.grad, meta=dict(payload.meta))

    if isinstance(payload, dict):
        if "grad" not in payload:
            raise RuntimeError("Text backward payload dict missing 'grad'.")
        meta = dict(payload.get("meta") or {})
        if "backward_time_ms" in payload and "backward_time_ms" not in meta:
            meta["backward_time_ms"] = payload["backward_time_ms"]
        return TextBackwardOutputs(grad=payload.get("grad"), meta=meta)

    raise RuntimeError(f"Unsupported text backward payload type: {type(payload)}")


def normalize_stage_outputs(payload: Any) -> StageOutputs:
    """Normalize any forward payload (VisionOutputs, dict, StageOutputs) into StageOutputs."""
    if isinstance(payload, StageOutputs):
        return payload

    if isinstance(payload, VisionOutputs):
        return StageOutputs(
            activations=payload.embeddings,
            attention_mask=payload.attention_mask,
            meta=dict(payload.meta),
        )

    if isinstance(payload, dict):
        if "activations" in payload:
            activations = payload["activations"]
        elif "vision_embeddings" in payload:
            activations = payload["vision_embeddings"]
        else:
            raise RuntimeError("Payload dict missing 'activations' or 'vision_embeddings'.")
        return StageOutputs(
            activations=activations,
            attention_mask=payload.get("attention_mask") or payload.get("vision_attention_mask"),
            meta=dict(payload.get("meta") or {}),
        )

    raise RuntimeError(f"Unsupported forward payload type: {type(payload)}")


def normalize_stage_gradients(payload: Any) -> StageGradients:
    """Normalize any backward payload (TextBackwardOutputs, dict, StageGradients) into StageGradients."""
    if isinstance(payload, StageGradients):
        return payload

    if isinstance(payload, TextBackwardOutputs):
        return StageGradients(grad=payload.grad, meta=dict(payload.meta))

    if isinstance(payload, dict):
        if "grad" not in payload:
            raise RuntimeError("Payload dict missing 'grad'.")
        return StageGradients(grad=payload.get("grad"), meta=dict(payload.get("meta") or {}))

    raise RuntimeError(f"Unsupported backward payload type: {type(payload)}")
