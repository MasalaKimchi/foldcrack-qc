"""Deterministic synthetic multimodal fixtures for smoke tests and stress tests.

These images are intentionally *not* a substitute for expert-annotated clinical
validation.  They provide known geometry for engineering tests, threshold
sanity checks, and modality/channel ablations before private data is available.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

from .adapters import adapt_image
from .schema import Modality, QCSample


@dataclass(frozen=True)
class SyntheticConfig:
    """Configuration for one reproducible synthetic QC field."""

    size: tuple[int, int] = (256, 256)
    seed: int = 0
    pixel_size_um: float | tuple[float, float] = 0.5
    include_fold: bool = True
    include_crack: bool = True
    include_hard_negatives: bool = True

    def __post_init__(self) -> None:
        if len(self.size) != 2 or min(self.size) < 64:
            raise ValueError(
                "Synthetic image size must be (height, width), each at least 64"
            )
        if int(self.seed) < 0:
            raise ValueError("Synthetic seed must be non-negative")


def _polyline_mask(
    shape: tuple[int, int], points: Sequence[tuple[int, int]], width: int
) -> np.ndarray:
    canvas = np.zeros(shape, dtype=np.uint8)
    cv2.polylines(
        canvas,
        [np.asarray(points, dtype=np.int32)],
        isClosed=False,
        color=1,
        thickness=max(1, int(width)),
        lineType=cv2.LINE_AA,
    )
    return canvas.astype(bool)


def _geometry(config: SyntheticConfig) -> dict[str, np.ndarray]:
    height, width = config.size
    short = min(height, width)
    geometry_rng = np.random.default_rng(config.seed + 91_173)

    def jitter(scale: float) -> float:
        return float(geometry_rng.uniform(-scale, scale))

    tissue_u8 = np.zeros((height, width), dtype=np.uint8)
    center = (int(width * (0.50 + jitter(0.012))), int(height * (0.50 + jitter(0.012))))
    axes = (max(20, int(width * 0.43)), max(20, int(height * 0.39)))
    cv2.ellipse(
        tissue_u8,
        center,
        axes,
        -4 + jitter(3.0),
        0,
        360,
        1,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    # A small protrusion makes the tissue boundary less like a perfect phantom.
    protrusion = np.asarray(
        [
            (int(width * 0.16), int(height * 0.48)),
            (int(width * 0.06), int(height * 0.56)),
            (int(width * 0.18), int(height * 0.64)),
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(tissue_u8, [protrusion], 1, lineType=cv2.LINE_AA)
    tissue = tissue_u8.astype(bool)

    # A broad, gently curved band models the doubled/thickened morphology of a
    # section fold.  Keep it above center so it does not overlap the tear.
    fold_x = np.linspace(0.23 * width, 0.78 * width, 8)
    fold_y = (0.37 + jitter(0.018)) * height + 0.045 * height * np.sin(
        np.linspace(-0.6 + jitter(0.2), 2.4 + jitter(0.2), 8)
    )
    fold_points = [(int(x), int(y)) for x, y in zip(fold_x, fold_y)]
    fold = _polyline_mask(config.size, fold_points, max(7, short // 17)) & tissue

    # The crack/tear is a narrow irregular discontinuity.  Its slight branches
    # exercise topology-aware metrics without making the mask unrealistically
    # wide for pixel-overlap methods.
    crack_points = [
        (int(width * (0.59 + jitter(0.012))), int(height * 0.54)),
        (int(width * (0.61 + jitter(0.012))), int(height * 0.60)),
        (int(width * (0.59 + jitter(0.012))), int(height * 0.66)),
        (int(width * (0.63 + jitter(0.012))), int(height * 0.72)),
        (int(width * (0.61 + jitter(0.012))), int(height * 0.80)),
    ]
    crack = _polyline_mask(config.size, crack_points, max(2, short // 85))
    branch = _polyline_mask(
        config.size,
        [crack_points[2], (int(width * 0.69), int(height * 0.69))],
        max(1, short // 110),
    )
    crack = (crack | branch) & tissue & ~fold

    # Lumen/vessel-like empty regions and a thin natural cleft are deliberately
    # rendered to resemble cracks but remain negative labels.
    hard_u8 = np.zeros(config.size, dtype=np.uint8)
    cv2.ellipse(
        hard_u8,
        (int(width * 0.35), int(height * 0.65)),
        (max(4, int(width * 0.055)), max(3, int(height * 0.035))),
        18,
        0,
        360,
        1,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )
    cleft = _polyline_mask(
        config.size,
        [
            (int(width * 0.72), int(height * 0.53)),
            (int(width * 0.76), int(height * 0.57)),
            (int(width * 0.79), int(height * 0.62)),
        ],
        max(2, short // 100),
    )
    hard_negative = (hard_u8.astype(bool) | cleft) & tissue & ~fold & ~crack

    if not config.include_fold:
        fold.fill(False)
    if not config.include_crack:
        crack.fill(False)
    if not config.include_hard_negatives:
        hard_negative.fill(False)
    return {
        "tissue": tissue,
        "fold": fold,
        "crack": crack,
        "hard_negative": hard_negative,
    }


def _smooth_noise(
    rng: np.random.Generator, shape: tuple[int, int], sigma: float
) -> np.ndarray:
    noise = rng.normal(0.0, 1.0, size=shape).astype(np.float32)
    smooth = cv2.GaussianBlur(noise, (0, 0), sigmaX=sigma, sigmaY=sigma)
    maximum = float(np.max(np.abs(smooth)))
    return smooth / maximum if maximum > 0 else smooth


def _nuclear_texture(
    rng: np.random.Generator, tissue: np.ndarray, density: float = 0.018
) -> np.ndarray:
    seeds = ((rng.random(tissue.shape) < density) & tissue).astype(np.float32)
    nuclei = cv2.GaussianBlur(seeds, (0, 0), sigmaX=1.15, sigmaY=1.15)
    maximum = float(nuclei.max())
    return nuclei / maximum if maximum > 0 else nuclei


def _render_he(
    rng: np.random.Generator, masks: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    tissue = masks["tissue"]
    height, width = tissue.shape
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = np.asarray([247.0, 246.0, 241.0], dtype=np.float32)

    low_frequency = _smooth_noise(
        rng, tissue.shape, sigma=max(3.0, min(tissue.shape) / 35)
    )
    base = np.stack(
        [
            218.0 + 12.0 * low_frequency,
            164.0 + 15.0 * low_frequency,
            193.0 + 11.0 * low_frequency,
        ],
        axis=-1,
    )
    image[tissue] = base[tissue]

    nuclei = _nuclear_texture(rng, tissue)
    nuclear_strength = np.clip(nuclei * 1.35, 0.0, 1.0)[..., np.newaxis]
    nuclear_color = np.asarray([77.0, 50.0, 117.0], dtype=np.float32)
    image = image * (1.0 - 0.78 * nuclear_strength) + nuclear_color * (
        0.78 * nuclear_strength
    )

    hard = masks["hard_negative"]
    hard_edge = (
        cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        & ~hard
    )
    image[hard] = np.asarray([242.0, 232.0, 235.0], dtype=np.float32)
    image[hard_edge & tissue] *= np.asarray([0.82, 0.72, 0.79], dtype=np.float32)

    fold = masks["fold"]
    if np.any(fold):
        fold_texture = (0.84 + 0.08 * low_frequency)[..., np.newaxis]
        image[fold] *= fold_texture[fold] * np.asarray(
            [0.69, 0.52, 0.68], dtype=np.float32
        )
        fold_edge = cv2.morphologyEx(
            fold.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)
        ).astype(bool)
        image[fold_edge & tissue] *= np.asarray([0.78, 0.63, 0.76], dtype=np.float32)

    crack = masks["crack"]
    if np.any(crack):
        crack_edge = (
            cv2.dilate(crack.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & ~crack
        )
        image[crack_edge & tissue] = np.asarray([112.0, 71.0, 119.0], dtype=np.float32)
        image[crack] = np.asarray([250.0, 248.0, 243.0], dtype=np.float32)

    sensor_noise = rng.normal(0.0, 1.8, size=image.shape).astype(np.float32)
    image += sensor_noise
    return np.clip(np.rint(image), 0, 255).astype(np.uint8), ["red", "green", "blue"]


def _render_fluorescence(
    modality: Modality, rng: np.random.Generator, masks: dict[str, np.ndarray]
) -> tuple[np.ndarray, list[str]]:
    tissue = masks["tissue"]
    smooth = _smooth_noise(rng, tissue.shape, sigma=max(2.0, min(tissue.shape) / 42))
    nuclei = _nuclear_texture(rng, tissue, density=0.022)

    if modality is Modality.COMET:
        names = [
            "DAPI",
            "FITC",
            "autofluorescence_TRITC",
            "autofluorescence_Cy5",
            "Cy7",
        ]
        coefficients = np.asarray([0.88, 0.24, 0.30, 0.25, 0.18], dtype=np.float32)
        roles_texture = np.stack(
            [
                0.92 * nuclei,
                0.18 + 0.20 * smooth + 0.22 * nuclei,
                0.22 + 0.15 * smooth + 0.08 * nuclei,
                0.20 - 0.12 * smooth + 0.10 * nuclei,
                0.16 + 0.16 * smooth + 0.15 * nuclei,
            ],
            axis=-1,
        )
    else:
        names = ["DAPI", "PanCK", "CD45", "CD298_B2M"]
        coefficients = np.asarray([0.90, 0.52, 0.27, 0.46], dtype=np.float32)
        # Broad morphology patterns are correlated but not identical.
        roles_texture = np.stack(
            [
                0.95 * nuclei,
                0.34 + 0.26 * smooth + 0.12 * nuclei,
                0.16 - 0.10 * smooth + 0.28 * nuclei,
                0.30 + 0.12 * smooth + 0.20 * nuclei,
            ],
            axis=-1,
        )

    image = np.zeros((*tissue.shape, len(names)), dtype=np.float32)
    tissue_signal = np.clip(roles_texture, 0.0, 1.0) * coefficients
    image[tissue] = tissue_signal[tissue]
    image += rng.normal(0.008, 0.006, size=image.shape).astype(np.float32)

    hard = masks["hard_negative"]
    hard_edge = (
        cv2.dilate(hard.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        & ~hard
    )
    image[hard] *= 0.08
    image[hard_edge & tissue] *= 1.25

    fold = masks["fold"]
    if np.any(fold):
        # Fold thickness raises fluorescence and autofluorescence across several
        # channels rather than depending on one biological marker.
        fold_gain = np.linspace(1.65, 2.25, image.shape[-1], dtype=np.float32)
        image[fold] = image[fold] * fold_gain + 0.18
        fold_blur = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2, sigmaY=1.2)
        image[fold] = 0.6 * image[fold] + 0.4 * fold_blur[fold]

    crack = masks["crack"]
    if np.any(crack):
        crack_edge = (
            cv2.dilate(crack.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & ~crack
        )
        image[crack] *= 0.015
        image[crack_edge & tissue] = np.clip(
            image[crack_edge & tissue] * 1.45 + 0.025, 0, 1
        )

    image[~tissue] *= 0.25
    return np.clip(image, 0.0, 1.0).astype(np.float32), names


def generate_synthetic_sample(
    modality: Modality | str,
    *,
    seed: int = 0,
    size: tuple[int, int] = (256, 256),
    pixel_size_um: float | tuple[float, float] = 0.5,
    include_fold: bool = True,
    include_crack: bool = True,
    include_hard_negatives: bool = True,
) -> QCSample:
    """Generate one deterministic, fully labeled synthetic image.

    Identical ``seed`` and ``size`` values use the same artifact geometry across
    all modalities, enabling paired detector/channel experiments.
    """

    resolved_modality = Modality.coerce(modality)
    config = SyntheticConfig(
        size=size,
        seed=int(seed),
        pixel_size_um=pixel_size_um,
        include_fold=include_fold,
        include_crack=include_crack,
        include_hard_negatives=include_hard_negatives,
    )
    rng = np.random.default_rng(config.seed)
    masks = _geometry(config)
    if resolved_modality is Modality.HE:
        array, channel_names = _render_he(rng, masks)
    else:
        array, channel_names = _render_fluorescence(resolved_modality, rng, masks)

    metadata = {
        "synthetic": True,
        "seed": config.seed,
        "intended_use": "engineering_smoke_test_not_clinical_validation",
        "artifact_burden_fraction": {
            key: float(value.sum() / max(1, masks["tissue"].sum()))
            for key, value in masks.items()
            if key != "tissue"
        },
    }
    image = adapt_image(
        array,
        resolved_modality,
        channel_names=channel_names,
        pixel_size_um=config.pixel_size_um,
        metadata=metadata,
        channel_axis=-1,
    )
    return QCSample(
        sample_id=f"synthetic-{resolved_modality.value}-{config.seed:06d}",
        image=image,
        masks=masks,
        metadata=metadata,
    )


def generate_synthetic_dataset(
    *,
    n_per_modality: int = 3,
    modalities: Iterable[Modality | str] = (
        Modality.HE,
        Modality.COMET,
        Modality.COSMX,
    ),
    seed: int = 0,
    size: tuple[int, int] = (256, 256),
    pixel_size_um: float | tuple[float, float] = 0.5,
) -> list[QCSample]:
    """Generate a reproducible list with unique samples for each modality."""

    if int(n_per_modality) <= 0:
        raise ValueError("n_per_modality must be positive")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    resolved = tuple(Modality.coerce(modality) for modality in modalities)
    if not resolved:
        raise ValueError("At least one modality is required")

    samples: list[QCSample] = []
    for modality_index, modality in enumerate(resolved):
        for sample_index in range(int(n_per_modality)):
            sample_seed = int(seed) + modality_index * 10_007 + sample_index
            samples.append(
                generate_synthetic_sample(
                    modality,
                    seed=sample_seed,
                    size=size,
                    pixel_size_um=pixel_size_um,
                )
            )
    return samples


__all__ = [
    "SyntheticConfig",
    "generate_synthetic_dataset",
    "generate_synthetic_sample",
]
