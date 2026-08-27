"""Paths to immutable runtime resources shipped with the Python package."""

from __future__ import annotations

from pathlib import Path


def resource_path(*parts: str) -> Path:
    """Return a path below the installed package's resource directory."""

    root = Path(__file__).resolve().parent
    candidate = root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:  # pragma: no cover - defensive programming
        raise ValueError("Resource paths must remain inside the package") from error
    return candidate


__all__ = ["resource_path"]
