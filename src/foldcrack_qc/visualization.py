"""Small OpenCV-only visualization helpers for QC error review."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def robust_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an arbitrary HxW[xC] image to displayable BGR uint8."""

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {array.shape}")

    channels: list[np.ndarray] = []
    for index in range(min(array.shape[-1], 3)):
        channel = array[..., index].astype(np.float32)
        finite = np.isfinite(channel)
        if not finite.any():
            channels.append(np.zeros(channel.shape, dtype=np.uint8))
            continue
        low, high = np.percentile(channel[finite], (1.0, 99.0))
        if high <= low:
            scaled = np.zeros_like(channel)
        else:
            scaled = np.clip((channel - low) / (high - low), 0.0, 1.0)
        channels.append(np.rint(255.0 * scaled).astype(np.uint8))

    if len(channels) == 1:
        rgb = np.repeat(channels[0][..., None], 3, axis=-1)
    else:
        while len(channels) < 3:
            channels.append(channels[-1])
        rgb = np.stack(channels[:3], axis=-1)
    # OpenCV writes BGR. H&E data arrive as RGB; fluorescence is a synthetic
    # structural composite, for which this conversion is also deterministic.
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _contours(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def qc_overlay(
    image: np.ndarray,
    *,
    target_fold: np.ndarray | None = None,
    target_crack: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    alpha: float = 0.34,
) -> np.ndarray:
    """Render target regions and prediction contours without hiding errors.

    Target fold is green, target crack is cyan, and prediction boundaries are
    red.  The deliberately distinct encodings make false positives and misses
    visible instead of blending two masks into a single success color.
    """

    canvas = robust_uint8(image)
    fill = canvas.copy()
    if target_fold is not None:
        fill[np.asarray(target_fold, dtype=bool)] = (40, 190, 40)
    if target_crack is not None:
        fill[np.asarray(target_crack, dtype=bool)] = (220, 190, 20)
    canvas = cv2.addWeighted(fill, alpha, canvas, 1.0 - alpha, 0.0)
    if prediction is not None:
        cv2.drawContours(canvas, _contours(prediction), -1, (20, 20, 240), 2)
    return canvas


def write_overlay(path: str | Path, image: np.ndarray, **kwargs: object) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = qc_overlay(image, **kwargs)
    if not cv2.imwrite(str(destination), rendered):
        raise OSError(f"OpenCV could not write overlay to {destination}")
    return destination


__all__ = ["qc_overlay", "robust_uint8", "write_overlay"]
