"""Tests for SharedActivationBuffer and SharedEventPool (T19).

CPU-only tests: allocation, write/read, slot isolation, schema validation.
GPU tests: IPC roundtrip, event pool IPC, slot/event reuse gate.
"""

import math

import pytest
import torch

from python.ray.shared_buffer import SharedActivationBuffer, SharedEventPool


# ── CPU-only tests ──


@pytest.mark.cpu_only
class TestSharedActivationBufferCPU:
    def test_init_slot_count_cpu(self):
        """Verify num_slots views are created on CPU device."""
        buf = SharedActivationBuffer(4, (2, 3), torch.float32, torch.device("cpu"))
        assert buf.num_slots == 4
        assert len(buf._slots) == 4
        for slot in buf._slots:
            assert slot.shape == (2, 3)

    def test_write_read_identity_cpu(self):
        """Write/read roundtrip preserves data."""
        buf = SharedActivationBuffer(2, (4, 8), torch.float32, torch.device("cpu"))
        tensor = torch.randn(4, 8)
        buf.write(0, tensor)
        result = buf.read(0)
        assert torch.equal(result, tensor)

    def test_slot_isolation_cpu(self):
        """Writing to slot 0 doesn't affect slot 1."""
        buf = SharedActivationBuffer(2, (4,), torch.float32, torch.device("cpu"))
        t0 = torch.ones(4)
        t1 = torch.zeros(4)
        buf.write(0, t0)
        buf.write(1, t1)
        assert torch.equal(buf.read(0), t0)
        assert torch.equal(buf.read(1), t1)

    def test_export_schema_keys(self):
        """Export payload contains required schema fields."""
        buf = SharedActivationBuffer(2, (4,), torch.float32, torch.device("cpu"))
        ipc_data = buf.export_ipc_handles()
        assert ipc_data["schema_version"] == 1
        assert ipc_data["layout"] == "cpu_schema_only"
        assert ipc_data["gpu_id"] is None
        assert ipc_data["num_slots"] == 2
        assert ipc_data["shape"] == (4,)
        assert "float32" in ipc_data["dtype"]
        assert ipc_data["contiguous_ipc"] is None
        assert ipc_data["slot_ipc"] == []

    def test_open_from_ipc_rejects_bad_schema(self):
        """Invalid schema_version raises clear ValueError."""
        bad_data = {
            "schema_version": 99,
            "gpu_id": "fake",
            "num_slots": 2,
            "shape": (4,),
            "dtype": "torch.float32",
            "contiguous_ipc": None,
        }
        with pytest.raises(ValueError, match="unsupported shared-buffer schema_version"):
            SharedActivationBuffer.open_from_ipc(bad_data, torch.device("cpu"))

    def test_open_from_ipc_rejects_missing_keys(self):
        """Missing required keys raises ValueError."""
        bad_data = {"schema_version": 1}
        with pytest.raises(ValueError, match="missing keys"):
            SharedActivationBuffer.open_from_ipc(bad_data, torch.device("cpu"))

    def test_open_from_ipc_rejects_cpu_schema(self):
        """CPU schema export cannot be opened via IPC."""
        buf = SharedActivationBuffer(2, (4,), torch.float32, torch.device("cpu"))
        ipc_data = buf.export_ipc_handles()
        with pytest.raises(ValueError, match="CPU schema export cannot be opened"):
            SharedActivationBuffer.open_from_ipc(ipc_data, torch.device("cpu"))

    def test_contiguous_storage(self):
        """All slots share the same underlying storage."""
        buf = SharedActivationBuffer(3, (4, 8), torch.float32, torch.device("cpu"))
        # Each slot should be a view of the contiguous storage
        assert buf._storage.numel() == 3 * 4 * 8
        for slot in buf._slots:
            assert slot.data_ptr() >= buf._storage.data_ptr()
            assert slot.data_ptr() < buf._storage.data_ptr() + buf._storage.nelement() * buf._storage.element_size()


# ── GPU tests ──


@pytest.mark.gpu
class TestSharedEventPoolGPU:
    def test_create_events_gpu(self):
        """Verify num_events interprocess events created."""
        pool = SharedEventPool(4, torch.device("cuda:0"))
        assert pool.num_events == 4
        assert len(pool._events) == 4

    def test_export_ipc_handles_length(self):
        """Verify handle list length matches num_events."""
        pool = SharedEventPool(3, torch.device("cuda:0"))
        handles = pool.export_ipc_handles()
        assert len(handles) == 3

    def test_record_wait_cycle(self):
        """Record and wait on event completes without error."""
        pool = SharedEventPool(2, torch.device("cuda:0"))
        pool.record(0)
        pool.wait(0)
        torch.cuda.synchronize()


@pytest.mark.gpu
class TestSharedBufferGPU:
    def test_write_read_gpu(self):
        """Write/read roundtrip on GPU preserves data."""
        buf = SharedActivationBuffer(2, (4, 8), torch.bfloat16, torch.device("cuda:0"))
        tensor = torch.randn(4, 8, dtype=torch.bfloat16, device="cuda:0")
        buf.write(0, tensor)
        torch.cuda.synchronize()
        result = buf.read(0)
        assert torch.equal(result, tensor)

    def test_export_ipc_handles_gpu(self):
        """GPU buffer exports contiguous_with_slot_fallback layout."""
        buf = SharedActivationBuffer(2, (4, 8), torch.bfloat16, torch.device("cuda:0"))
        ipc_data = buf.export_ipc_handles()
        assert ipc_data["schema_version"] == 1
        assert ipc_data["layout"] == "contiguous_with_slot_fallback"
        assert ipc_data["gpu_id"] is not None
        assert ipc_data["num_slots"] == 2
        assert ipc_data["contiguous_ipc"] is not None
        assert len(ipc_data["slot_ipc"]) == 2


@pytest.mark.gpu
class TestSharedBufferIPCSchema:
    """GPU IPC schema tests (export validation, not cross-process open).

    CUDA IPC handles cannot be opened in the same process. Cross-process IPC
    is validated by the pipeline integration tests (test_pipeline_shared_buffer.py)
    which use Ray actors running in separate processes.
    """

    def test_buffer_export_schema_gpu(self):
        """GPU buffer exports correct versioned IPC schema."""
        buf = SharedActivationBuffer(4, (2, 16), torch.bfloat16, torch.device("cuda:0"))
        ipc_data = buf.export_ipc_handles()
        assert ipc_data["schema_version"] == 1
        assert ipc_data["layout"] == "contiguous_with_slot_fallback"
        assert ipc_data["gpu_id"] is not None
        assert ipc_data["num_slots"] == 4
        assert ipc_data["shape"] == (2, 16)
        assert "bfloat16" in ipc_data["dtype"]
        assert ipc_data["contiguous_ipc"] is not None
        assert len(ipc_data["slot_ipc"]) == 4

    def test_event_export_handles_gpu(self):
        """Event pool exports correct number of IPC handles."""
        pool = SharedEventPool(4, torch.device("cuda:0"))
        handles = pool.export_ipc_handles()
        assert len(handles) == 4

    def test_buffer_write_read_multiple_slots_gpu(self):
        """Direct write/read (no IPC) works correctly on GPU."""
        num_slots = 4
        shape = (2, 16)
        buf = SharedActivationBuffer(num_slots, shape, torch.float32, torch.device("cuda:0"))
        tensors = [torch.randn(*shape, device="cuda:0") for _ in range(num_slots)]
        for i, t in enumerate(tensors):
            buf.write(i, t)
        torch.cuda.synchronize()
        for i, expected in enumerate(tensors):
            result = buf.read(i)
            assert torch.equal(result, expected), f"Slot {i} mismatch"

    def test_buffer_event_producer_flow_gpu(self):
        """Full producer flow: write+record, then direct read (same-process)."""
        shape = (2, 8)
        buf = SharedActivationBuffer(2, shape, torch.bfloat16, torch.device("cuda:0"))
        events = SharedEventPool(2, torch.device("cuda:0"))

        tensor = torch.randn(*shape, dtype=torch.bfloat16, device="cuda:0")
        buf.write(0, tensor)
        events.record(0)
        events.wait(0)
        result = buf.read(0)
        torch.cuda.synchronize()
        assert torch.equal(result, tensor)

    def test_slot_reuse_over_iterations_gpu(self):
        """Slots can be reused across 100+ iterations with correct data (same-process)."""
        num_slots = 2
        shape = (4, 8)
        buf = SharedActivationBuffer(num_slots, shape, torch.float32, torch.device("cuda:0"))
        events = SharedEventPool(num_slots, torch.device("cuda:0"))

        num_iters = 100
        for iteration in range(num_iters):
            for slot in range(num_slots):
                value = float(iteration * 1000 + slot)
                tensor = torch.full(shape, value, device="cuda:0")
                buf.write(slot, tensor)
                events.record(slot)
                events.wait(slot)
                result = buf.read(slot)
                torch.cuda.synchronize()

                expected = torch.full(shape, value, device="cuda:0")
                assert torch.equal(result, expected), (
                    f"Mismatch at iter={iteration}, slot={slot}: "
                    f"expected {value}, got {result.flatten()[0].item()}"
                )
