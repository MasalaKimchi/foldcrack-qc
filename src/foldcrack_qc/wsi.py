"""Physical-scale tiled inference primitives for WSI/pyramid integration.

This module does not pretend that NumPy is a production slide reader.  It
defines and tests the coordinate, halo, and stitching contract that an
OpenSlide, cuCIM, OME-Zarr, or vendor adapter must implement.  The included
``ArrayPyramidSource`` is a deterministic test double and ROI prototype.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


def _pair(
    value: float | Sequence[float], name: str, *, allow_zero: bool = False
) -> tuple[float, float]:
    if np.isscalar(value):
        pair = (float(value), float(value))
    else:
        pair = tuple(float(item) for item in value)
        if len(pair) != 2:
            raise ValueError(f"{name} must be a scalar or (y, x) pair")
    invalid_minimum = min(pair) < 0.0 if allow_zero else min(pair) <= 0.0
    if not np.all(np.isfinite(pair)) or invalid_minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must contain {qualifier} finite values")
    return pair


@dataclass(frozen=True)
class TileWindow:
    """One core tile plus a clipped read halo, in one pyramid level."""

    core: tuple[int, int, int, int]
    read: tuple[int, int, int, int]

    @property
    def core_in_read(self) -> tuple[slice, slice]:
        y0, x0, y1, x1 = self.core
        read_y0, read_x0, _, _ = self.read
        return (
            slice(y0 - read_y0, y1 - read_y0),
            slice(x0 - read_x0, x1 - read_x0),
        )

    @property
    def read_shape(self) -> tuple[int, int]:
        y0, x0, y1, x1 = self.read
        return y1 - y0, x1 - x0


@runtime_checkable
class PyramidSource(Protocol):
    @property
    def level_count(self) -> int: ...

    def level_shape(self, level: int) -> tuple[int, int]: ...

    def level_pixel_size_um(self, level: int) -> tuple[float, float]: ...

    def read_region(
        self, level: int, window: tuple[int, int, int, int]
    ) -> np.ndarray: ...


class ArrayPyramidSource:
    """In-memory pyramid used for deterministic integration tests and ROIs."""

    def __init__(
        self,
        levels: Sequence[np.ndarray],
        pixel_sizes_um: Sequence[float | Sequence[float]],
    ) -> None:
        if not levels or len(levels) != len(pixel_sizes_um):
            raise ValueError("levels and pixel_sizes_um must be non-empty and aligned")
        arrays: list[np.ndarray] = []
        spacings: list[tuple[float, float]] = []
        for index, (level, spacing) in enumerate(
            zip(levels, pixel_sizes_um, strict=True)
        ):
            array = np.asarray(level)
            if array.ndim not in (2, 3) or min(array.shape[:2]) <= 0:
                raise ValueError(f"Pyramid level {index} must be HxW[xC]")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Pyramid level {index} contains non-finite values")
            arrays.append(array)
            spacings.append(_pair(spacing, f"pixel_sizes_um[{index}]"))
        self._levels = tuple(arrays)
        self._spacings = tuple(spacings)

    @property
    def level_count(self) -> int:
        return len(self._levels)

    def level_shape(self, level: int) -> tuple[int, int]:
        array = self._levels[level]
        return int(array.shape[0]), int(array.shape[1])

    def level_pixel_size_um(self, level: int) -> tuple[float, float]:
        return self._spacings[level]

    def read_region(self, level: int, window: tuple[int, int, int, int]) -> np.ndarray:
        y0, x0, y1, x1 = (int(item) for item in window)
        height, width = self.level_shape(level)
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"Read window is outside pyramid level {level}")
        return np.ascontiguousarray(self._levels[level][y0:y1, x0:x1])


def choose_level(source: PyramidSource, target_pixel_size_um: float) -> int:
    """Choose the level closest to a target physical sampling on a log scale."""

    target = float(target_pixel_size_um)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("target_pixel_size_um must be finite and positive")
    effective = [
        math_sqrt_product(source.level_pixel_size_um(i))
        for i in range(source.level_count)
    ]
    distances = [abs(np.log(value / target)) for value in effective]
    return int(np.argmin(distances))


def math_sqrt_product(pair: Sequence[float]) -> float:
    return float(np.sqrt(float(pair[0]) * float(pair[1])))


def iter_tile_windows(
    shape: Sequence[int],
    *,
    pixel_size_um: float | Sequence[float],
    tile_size_um: float | Sequence[float],
    halo_um: float | Sequence[float],
) -> tuple[TileWindow, ...]:
    """Create gap-free non-overlapping cores with overlapping read halos."""

    if len(shape) != 2 or min(int(item) for item in shape) <= 0:
        raise ValueError("shape must be a positive (height, width) pair")
    height, width = (int(item) for item in shape)
    spacing = _pair(pixel_size_um, "pixel_size_um")
    tile_um = _pair(tile_size_um, "tile_size_um")
    halo_pair = _pair(halo_um, "halo_um", allow_zero=True)
    tile_px = tuple(
        max(1, round(value / spacing[index])) for index, value in enumerate(tile_um)
    )
    halo_px = tuple(
        max(0, round(value / spacing[index])) for index, value in enumerate(halo_pair)
    )

    windows: list[TileWindow] = []
    for y0 in range(0, height, tile_px[0]):
        y1 = min(height, y0 + tile_px[0])
        for x0 in range(0, width, tile_px[1]):
            x1 = min(width, x0 + tile_px[1])
            read = (
                max(0, y0 - halo_px[0]),
                max(0, x0 - halo_px[1]),
                min(height, y1 + halo_px[0]),
                min(width, x1 + halo_px[1]),
            )
            windows.append(TileWindow((y0, x0, y1, x1), read))
    return tuple(windows)


def stitch_core_predictions(
    windows: Sequence[TileWindow],
    predictions: Sequence[np.ndarray],
    shape: Sequence[int],
) -> np.ndarray:
    """Discard halo predictions and stitch each core exactly once."""

    if len(windows) != len(predictions):
        raise ValueError("windows and predictions must have the same length")
    if len(shape) != 2:
        raise ValueError("shape must be (height, width)")
    output = np.zeros((int(shape[0]), int(shape[1])), dtype=np.float64)
    coverage = np.zeros_like(output, dtype=np.uint8)
    for window, raw_prediction in zip(windows, predictions, strict=True):
        prediction = np.asarray(raw_prediction, dtype=float)
        if prediction.shape != window.read_shape:
            raise ValueError(
                f"Prediction shape {prediction.shape} does not match read halo {window.read_shape}"
            )
        y0, x0, y1, x1 = window.core
        core = prediction[window.core_in_read]
        output[y0:y1, x0:x1] = core
        coverage[y0:y1, x0:x1] += 1
    if not np.all(coverage == 1):
        raise ValueError("Tile cores must cover every output pixel exactly once")
    return output


def level_box_to_level0(
    box: Sequence[int],
    *,
    level_pixel_size_um: Sequence[float],
    level0_pixel_size_um: Sequence[float],
) -> tuple[int, int, int, int]:
    """Map an axis-aligned level box into level-0 pixel coordinates."""

    if len(box) != 4:
        raise ValueError("box must be (y0, x0, y1, x1)")
    level_spacing = _pair(level_pixel_size_um, "level_pixel_size_um")
    level0_spacing = _pair(level0_pixel_size_um, "level0_pixel_size_um")
    scale_y = level_spacing[0] / level0_spacing[0]
    scale_x = level_spacing[1] / level0_spacing[1]
    y0, x0, y1, x1 = (int(item) for item in box)
    return (
        round(y0 * scale_y),
        round(x0 * scale_x),
        round(y1 * scale_y),
        round(x1 * scale_x),
    )


__all__ = [
    "ArrayPyramidSource",
    "PyramidSource",
    "TileWindow",
    "choose_level",
    "iter_tile_windows",
    "level_box_to_level0",
    "stitch_core_predictions",
]
