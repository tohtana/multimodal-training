"""Pre-shared GPU ring buffer + CUDA event pool for zero-overhead inter-process tensor transfer.

Replaces per-microbatch IPC handle creation with setup-time-only IPC sharing:
- SharedActivationBuffer: K pre-allocated GPU slots, IPC handles exported once
- SharedEventPool: K interprocess CUDA events, IPC handles exported once

Usage:
  Producer: buffer.write(slot, tensor); events.record(slot)
  Consumer: events.wait(slot); tensor = buffer.read(slot)
"""

from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)


class SharedActivationBuffer:
    """Pre-allocated GPU ring buffer for zero-overhead inter-process tensor sharing.

    Created by the producer actor, IPC handles exported once at setup time.
    Consumer actor opens handles once -> persistent GPU memory pointers.

    Memory layout: K contiguous slots of (shape, dtype) on a single GPU.
    Slot assignment (v1): slot == microbatch_id, with K == num_microbatches.
    """

    def __init__(self, num_slots: int, shape: tuple, dtype: torch.dtype, device: torch.device):
        self.num_slots = num_slots
        self.shape = shape
        self.dtype = dtype
        self.device = device
        # Allocate a single contiguous tensor and view as K slots
        slot_size = math.prod(shape)
        total_elements = num_slots * slot_size
        self._storage = torch.empty(total_elements, dtype=dtype, device=device)
        self._slots = [self._storage[i * slot_size : (i + 1) * slot_size].view(shape) for i in range(num_slots)]

    def write(self, slot: int, tensor: torch.Tensor) -> None:
        """Copy tensor into buffer[slot]. Must be followed by event.record()."""
        self._slots[slot].copy_(tensor)

    def read(self, slot: int) -> torch.Tensor:
        """Return a view of buffer[slot]. Caller must ensure event sync first."""
        return self._slots[slot]

    def export_ipc_handles(self) -> dict:
        """Export versioned IPC metadata (contiguous primary + per-slot fallback).

        Returns a dict with schema_version, layout, and IPC handles.
        CPU tensors return a cpu_schema_only layout (for testing).
        """
        if not self._storage.is_cuda:
            return {
                "schema_version": 1,
                "layout": "cpu_schema_only",
                "gpu_id": None,
                "num_slots": self.num_slots,
                "shape": self.shape,
                "dtype": str(self.dtype),
                "contiguous_ipc": None,
                "slot_ipc": [],
            }

        from .tensor_transfer import create_ipc_handle

        contiguous_ipc, gpu_id, _ = create_ipc_handle(self._storage)
        slot_ipc = [create_ipc_handle(slot)[0] for slot in self._slots]
        return {
            "schema_version": 1,
            "layout": "contiguous_with_slot_fallback",
            "gpu_id": gpu_id,
            "num_slots": self.num_slots,
            "shape": self.shape,
            "dtype": str(self.dtype),
            "contiguous_ipc": contiguous_ipc,
            "slot_ipc": slot_ipc,
        }

    @classmethod
    def open_from_ipc(cls, ipc_data: dict, device: torch.device) -> SharedActivationBuffer:
        """Reconstruct buffer from versioned IPC schema on consumer side."""
        from .tensor_transfer import reconstruct_tensor_from_ipc
        from .utils import get_physical_gpu_id

        required = {"schema_version", "gpu_id", "num_slots", "shape", "dtype", "contiguous_ipc"}
        missing = required - set(ipc_data.keys())
        if missing:
            raise ValueError(f"shared-buffer IPC payload missing keys: {sorted(missing)}")
        if ipc_data["schema_version"] != 1:
            raise ValueError(f"unsupported shared-buffer schema_version={ipc_data['schema_version']}")
        if ipc_data.get("layout") == "cpu_schema_only":
            raise ValueError("CPU schema export cannot be opened via CUDA IPC")

        num_slots = ipc_data["num_slots"]
        shape = tuple(ipc_data["shape"])
        slot_size = math.prod(shape)
        my_gpu = get_physical_gpu_id()

        instance = cls.__new__(cls)
        instance.num_slots = num_slots
        instance.shape = shape
        instance.dtype = getattr(torch, ipc_data["dtype"].split(".")[-1])
        instance.device = device

        # Primary path: single contiguous IPC handle.
        try:
            storage = reconstruct_tensor_from_ipc(
                ipc_data["contiguous_ipc"],
                my_gpu,
                ipc_data["gpu_id"],
            )
            instance._storage = storage
            instance._slots = [storage[i * slot_size : (i + 1) * slot_size].view(shape) for i in range(num_slots)]
            instance._ipc_layout = "contiguous"
            return instance
        except Exception:
            # Setup-time fallback: reconstruct each slot handle independently.
            slot_handles = ipc_data.get("slot_ipc") or []
            if len(slot_handles) != num_slots:
                raise
            instance._storage = None
            instance._slots = [
                reconstruct_tensor_from_ipc(h, my_gpu, ipc_data["gpu_id"]).view(shape) for h in slot_handles
            ]
            instance._ipc_layout = "slot_fallback"
            return instance


class SharedEventPool:
    """Pool of CUDA interprocess events for ring buffer synchronization.

    Created by the producer, IPC handles exported once.
    Consumer reconstructs events from handles and calls stream.wait_event().
    """

    def __init__(self, num_events: int, device: torch.device):
        self.num_events = num_events
        self.device = device
        self._events = [torch.cuda.Event(interprocess=True) for _ in range(num_events)]

    def record(self, slot: int) -> None:
        """Record event[slot] on current stream (producer side)."""
        torch.cuda.current_stream().record_event(self._events[slot])

    def wait(self, slot: int) -> None:
        """Wait on event[slot] on current stream (consumer side)."""
        torch.cuda.current_stream().wait_event(self._events[slot])

    def export_ipc_handles(self) -> list:
        """Export IPC handles for all events."""
        return [e.ipc_handle() for e in self._events]

    @classmethod
    def open_from_ipc(cls, ipc_handles: list, device: torch.device) -> SharedEventPool:
        """Reconstruct events from IPC handles (consumer side)."""
        instance = cls.__new__(cls)
        instance.num_events = len(ipc_handles)
        instance.device = device
        device_id = torch.cuda.current_device()
        instance._events = [torch.cuda.Event.from_ipc_handle(device=device_id, handle=h) for h in ipc_handles]
        return instance
