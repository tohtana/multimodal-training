"""ctypes helpers for CUDA Green Context-backed streams."""

from __future__ import annotations

import ctypes
import ctypes.util
import threading
from dataclasses import dataclass
from typing import Any

import torch

CU_STREAM_NON_BLOCKING = 0x1
CU_GREEN_CTX_DEFAULT_STREAM = 0x1
CU_DEV_RESOURCE_TYPE_SM = 0x1

_CUDA_SUCCESS = 0
_RESOURCE_PADDING_BYTES = 92
_RESOURCE_EXTERNAL_BYTES = 48


class GreenContextError(RuntimeError):
    """Raised when CUDA Green Context setup or teardown fails."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class GreenContextUnavailableError(GreenContextError):
    """Raised when the current host cannot use CUDA Green Context APIs."""


class GreenContextConfigError(GreenContextError):
    """Raised when the requested Green Context configuration is invalid."""


class _CUdevSmResource(ctypes.Structure):
    _fields_ = [("smCount", ctypes.c_uint)]


class _CUdevResourcePayload(ctypes.Union):
    _fields_ = [
        ("sm", _CUdevSmResource),
        ("_oversize", ctypes.c_ubyte * _RESOURCE_EXTERNAL_BYTES),
    ]


class _CUdevResource(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("_internal_padding", ctypes.c_ubyte * _RESOURCE_PADDING_BYTES),
        ("payload", _CUdevResourcePayload),
    ]


def _resource_sm_count(resource: _CUdevResource) -> int:
    return int(resource.payload.sm.smCount)


class _CudaDriver:
    _instance: "_CudaDriver | None" = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        library_path = ctypes.util.find_library("cuda") or "libcuda.so.1"
        try:
            self.lib = ctypes.CDLL(library_path)
        except OSError as exc:
            raise GreenContextUnavailableError("green_context_unavailable", f"Failed to load {library_path}: {exc}") from exc

        self._bind("cuGetErrorName", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int)
        self._bind("cuGetErrorString", [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)], ctypes.c_int)
        self._bind("cuInit", [ctypes.c_uint], ctypes.c_int)
        self._bind("cuDeviceGet", [ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int)
        self._bind(
            "cuDeviceGetDevResource",
            [ctypes.c_int, ctypes.POINTER(_CUdevResource), ctypes.c_int],
            ctypes.c_int,
        )
        self._bind(
            "cuDevSmResourceSplitByCount",
            [
                ctypes.POINTER(_CUdevResource),
                ctypes.POINTER(ctypes.c_uint),
                ctypes.POINTER(_CUdevResource),
                ctypes.POINTER(_CUdevResource),
                ctypes.c_uint,
                ctypes.c_uint,
            ],
            ctypes.c_int,
        )
        self._bind(
            "cuDevResourceGenerateDesc",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(_CUdevResource), ctypes.c_uint],
            ctypes.c_int,
        )
        self._bind(
            "cuGreenCtxCreate",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_int, ctypes.c_uint],
            ctypes.c_int,
        )
        self._bind("cuGreenCtxDestroy", [ctypes.c_void_p], ctypes.c_int)
        self._bind(
            "cuGreenCtxGetDevResource",
            [ctypes.c_void_p, ctypes.POINTER(_CUdevResource), ctypes.c_int],
            ctypes.c_int,
        )
        self._bind(
            "cuGreenCtxStreamCreate",
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint, ctypes.c_int],
            ctypes.c_int,
        )
        self._bind("cuStreamDestroy", [ctypes.c_void_p], ctypes.c_int)

        self._check(self.lib.cuInit(0), "cuInit", "green_context_unavailable")

    @classmethod
    def instance(cls) -> "_CudaDriver":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _bind(self, name: str, argtypes: list[Any], restype: Any) -> None:
        fn = getattr(self.lib, name, None)
        if fn is None:
            raise GreenContextUnavailableError("green_context_unavailable", f"CUDA driver symbol {name} is unavailable")
        fn.argtypes = argtypes
        fn.restype = restype

    def _format_error(self, result: int) -> str:
        name_ptr = ctypes.c_char_p()
        text_ptr = ctypes.c_char_p()
        name = f"CUDA error {result}"
        text = "unknown error"
        if self.lib.cuGetErrorName(result, ctypes.byref(name_ptr)) == _CUDA_SUCCESS and name_ptr.value is not None:
            name = name_ptr.value.decode("utf-8", errors="replace")
        if self.lib.cuGetErrorString(result, ctypes.byref(text_ptr)) == _CUDA_SUCCESS and text_ptr.value is not None:
            text = text_ptr.value.decode("utf-8", errors="replace")
        return f"{name}: {text}"

    def _check(self, result: int, api_name: str, code: str = "green_context_runtime_error") -> None:
        if result == _CUDA_SUCCESS:
            return
        raise GreenContextError(code, f"{api_name} failed with {self._format_error(result)}")

    def _device_handle(self, device_id: int) -> int:
        handle = ctypes.c_int()
        self._check(
            self.lib.cuDeviceGet(ctypes.byref(handle), int(device_id)),
            "cuDeviceGet",
            "green_context_unavailable",
        )
        return int(handle.value)

    def total_sms_for_device(self, device_id: int) -> int:
        resource = _CUdevResource()
        self._check(
            self.lib.cuDeviceGetDevResource(
                self._device_handle(device_id),
                ctypes.byref(resource),
                CU_DEV_RESOURCE_TYPE_SM,
            ),
            "cuDeviceGetDevResource",
            "green_context_unavailable",
        )
        return _resource_sm_count(resource)

    def create_stream_owner(self, *, device_id: int, requested_sms: int) -> "GreenContextStreamOwner":
        total_resource = _CUdevResource()
        device = self._device_handle(device_id)
        self._check(
            self.lib.cuDeviceGetDevResource(
                device,
                ctypes.byref(total_resource),
                CU_DEV_RESOURCE_TYPE_SM,
            ),
            "cuDeviceGetDevResource",
            "green_context_unavailable",
        )
        total_sms = _resource_sm_count(total_resource)
        if requested_sms <= 0:
            raise GreenContextConfigError("green_context_invalid_config", f"requested_sms must be > 0, got {requested_sms}")
        if requested_sms > total_sms:
            raise GreenContextConfigError(
                "green_context_invalid_config",
                f"requested_sms ({requested_sms}) exceeds device total ({total_sms})",
            )

        split_groups = (_CUdevResource * 1)()
        split_count = ctypes.c_uint(1)
        remaining = _CUdevResource()
        self._check(
            self.lib.cuDevSmResourceSplitByCount(
                split_groups,
                ctypes.byref(split_count),
                ctypes.byref(total_resource),
                ctypes.byref(remaining),
                0,
                int(requested_sms),
            ),
            "cuDevSmResourceSplitByCount",
            "green_context_invalid_config",
        )
        if int(split_count.value) < 1:
            raise GreenContextConfigError(
                "green_context_invalid_config",
                f"Unable to create a Green Context SM partition for requested_sms={requested_sms}",
            )

        desc = ctypes.c_void_p()
        self._check(
            self.lib.cuDevResourceGenerateDesc(
                ctypes.byref(desc),
                split_groups,
                1,
            ),
            "cuDevResourceGenerateDesc",
            "green_context_runtime_error",
        )

        green_ctx = ctypes.c_void_p()
        stream = ctypes.c_void_p()
        try:
            self._check(
                self.lib.cuGreenCtxCreate(
                    ctypes.byref(green_ctx),
                    desc,
                    device,
                    CU_GREEN_CTX_DEFAULT_STREAM,
                ),
                "cuGreenCtxCreate",
                "green_context_runtime_error",
            )
            self._check(
                self.lib.cuGreenCtxStreamCreate(
                    ctypes.byref(stream),
                    green_ctx,
                    CU_STREAM_NON_BLOCKING,
                    0,
                ),
                "cuGreenCtxStreamCreate",
                "green_context_runtime_error",
            )
            granted_resource = _CUdevResource()
            self._check(
                self.lib.cuGreenCtxGetDevResource(
                    green_ctx,
                    ctypes.byref(granted_resource),
                    CU_DEV_RESOURCE_TYPE_SM,
                ),
                "cuGreenCtxGetDevResource",
                "green_context_runtime_error",
            )
        except Exception:
            if stream.value:
                self.destroy_stream(stream)
            if green_ctx.value:
                self.destroy_green_ctx(green_ctx)
            raise

        return GreenContextStreamOwner(
            device_id=int(device_id),
            requested_sms=int(requested_sms),
            granted_sms=_resource_sm_count(granted_resource),
            total_sms=int(total_sms),
            _driver=self,
            _green_ctx=green_ctx,
            _stream=stream,
            external_stream=torch.cuda.ExternalStream(int(stream.value), device=int(device_id)),
        )

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream.value is None:
            return
        self._check(
            self.lib.cuStreamDestroy(stream),
            "cuStreamDestroy",
            "green_context_cleanup_failed",
        )

    def destroy_green_ctx(self, green_ctx: ctypes.c_void_p) -> None:
        if green_ctx.value is None:
            return
        self._check(
            self.lib.cuGreenCtxDestroy(green_ctx),
            "cuGreenCtxDestroy",
            "green_context_cleanup_failed",
        )


@dataclass
class GreenContextStreamOwner:
    device_id: int
    requested_sms: int
    granted_sms: int
    total_sms: int
    _driver: _CudaDriver
    _green_ctx: ctypes.c_void_p
    _stream: ctypes.c_void_p
    external_stream: torch.cuda.ExternalStream
    _closed: bool = False

    def wait_for_current_stream(self) -> None:
        self.external_stream.wait_stream(torch.cuda.current_stream(device=self.device_id))

    def synchronize(self) -> None:
        self.external_stream.synchronize()

    def cleanup(self) -> None:
        if self._closed:
            return
        try:
            self.synchronize()
        except Exception:
            pass
        try:
            self._driver.destroy_stream(self._stream)
        finally:
            self._stream = ctypes.c_void_p()
            try:
                self._driver.destroy_green_ctx(self._green_ctx)
            finally:
                self._green_ctx = ctypes.c_void_p()
                self._closed = True


def green_context_supported() -> tuple[bool, str | None]:
    try:
        _CudaDriver.instance()
    except GreenContextError as exc:
        return False, exc.message
    return True, None


def get_device_total_sms(device_id: int) -> int:
    return _CudaDriver.instance().total_sms_for_device(int(device_id))


def create_green_context_stream(*, device_id: int, requested_sms: int) -> GreenContextStreamOwner:
    return _CudaDriver.instance().create_stream_owner(device_id=int(device_id), requested_sms=int(requested_sms))
