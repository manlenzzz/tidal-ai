from __future__ import annotations

import torch

__all__ = ["resolve_device", "resolve_dtype"]

_DTYPE_BY_NAME = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _npu_available() -> bool:
    """Return True only if torch_npu is importable and reports an available NPU."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    npu = getattr(torch, "npu", None)
    try:
        return bool(npu is not None and npu.is_available())
    except Exception:
        return False


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve a torch device, supporting NVIDIA CUDA and Ascend NPU.

    Explicit selections are honored. Selecting an ``npu`` device imports
    ``torch_npu`` first and raises a clear error if it is unavailable. When
    ``device`` is ``None`` the accelerator is auto-detected in the order
    CUDA -> NPU -> CPU; the CUDA/CPU paths never import ``torch_npu``.
    """
    if isinstance(device, torch.device):
        device = str(device)

    if device is not None:
        normalized = str(device).strip().lower()
        if normalized.startswith("npu"):
            try:
                import torch_npu  # noqa: F401
            except Exception as exc:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "device 'npu' was requested but torch_npu is not importable; "
                    "install the Ascend torch_npu extension matching your CANN version"
                ) from exc
        return torch.device(normalized)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _npu_available():
        return torch.device("npu")
    return torch.device("cpu")


def resolve_dtype(
    dtype: str | torch.dtype | None,
    device: torch.device | object,
) -> torch.dtype | str:
    """Resolve a torch dtype, defaulting to a device-appropriate choice.

    Explicit dtypes (``float32``/``float16``/``bfloat16`` or a ``torch.dtype``)
    are honored. ``"auto"`` is passed through. When ``dtype`` is ``None`` the
    default is ``bfloat16`` on CUDA/NPU accelerators and ``"auto"`` on CPU,
    preserving the prior Hugging Face ``torch_dtype="auto"`` CPU behavior.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is not None:
        name = str(dtype).strip().lower()
        if name == "auto":
            return "auto"
        if name in _DTYPE_BY_NAME:
            return _DTYPE_BY_NAME[name]
        raise ValueError(f"unsupported dtype: {dtype!r}")

    device_type = getattr(device, "type", str(device))
    if device_type in ("cuda", "npu"):
        return torch.bfloat16
    return "auto"
