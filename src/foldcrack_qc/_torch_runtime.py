"""Shared, dependency-light PyTorch runtime primitives.

This private module owns optional PyTorch import, device selection, and backend
synchronization.  Higher-level modules keep their existing public interfaces
and policy-specific validation while delegating the duplicated mechanics here.
"""

from __future__ import annotations

from typing import Any


def import_torch() -> Any:
    """Import PyTorch lazily with the repository's actionable error message."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "Foundation feature extraction requires PyTorch. Install the "
            "project's 'foundation' extra or inject an approved torch runtime."
        ) from error
    return torch


def select_torch_device(
    requested: str = "auto",
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve the current ``auto|mps|cpu`` policy without silent fallback."""

    torch = import_torch() if torch_module is None else torch_module
    normalized = str(requested).lower()
    if normalized not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of 'auto', 'mps', or 'cpu'")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and mps_backend.is_available()
    )
    if normalized == "auto":
        return "mps" if mps_available else "cpu"
    if normalized == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested but is unavailable in this PyTorch/macOS runtime"
        )
    return normalized


def synchronize_torch_device(
    torch_module: Any,
    device: str,
    *,
    require_available: bool = False,
) -> None:
    """Synchronize the selected backend under the current CPU/MPS contract.

    ``require_available`` preserves the two existing smoke-runner policies:
    the general foundation smoke tolerates runtimes without an MPS synchronize
    hook, while the locked SigLIP2 smoke fails closed.
    """

    if device != "mps":
        return
    synchronize = getattr(getattr(torch_module, "mps", None), "synchronize", None)
    if callable(synchronize):
        synchronize()
        return
    if require_available:
        raise TypeError("MPS synchronization is unavailable")


__all__ = [
    "import_torch",
    "select_torch_device",
    "synchronize_torch_device",
]
