"""Classical and clean-reference detectors for fold/crack QC.

The classical functions provide inspectable candidate masks, while
``CleanReferenceAnomalyDetector`` learns a compact robust distribution from
artifact-free patch features.  ``HybridQCDetector`` combines both signals
without requiring a GPU or a supervised training set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .features import (
    FeatureTable,
    _channels_last,
    _disk,
    _gradient_magnitude,
    _local_standard_deviation,
    _robust_unit_scale,
    extract_patch_feature_table,
    rgb_to_optical_density,
    tissue_mask,
)


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class CandidateMasks:
    """Classical outputs and their explicit native-resolution review support.

    ``tissue`` is the conservative support inferred from image signal.  A tear
    can legitimately be absent from that mask, so ``review_support`` is kept as
    a separate, auditable concept and always includes proposed fold/crack
    pixels.  Downstream fusion must use the latter when deciding whether an
    artifact is reviewable.
    """

    tissue: BoolArray
    fold: BoolArray
    crack: BoolArray
    fold_score: FloatArray
    crack_score: FloatArray
    review_support: BoolArray | None = None

    def __post_init__(self) -> None:
        tissue = np.asarray(self.tissue, dtype=bool)
        if tissue.ndim != 2:
            raise ValueError("Candidate masks must be two-dimensional")
        arrays = {
            "fold": np.asarray(self.fold, dtype=bool),
            "crack": np.asarray(self.crack, dtype=bool),
            "fold_score": np.asarray(self.fold_score, dtype=np.float64),
            "crack_score": np.asarray(self.crack_score, dtype=np.float64),
        }
        if any(array.shape != tissue.shape for array in arrays.values()):
            raise ValueError("All CandidateMasks arrays must have the same shape")
        review_support = (
            tissue.copy()
            if self.review_support is None
            else np.asarray(self.review_support, dtype=bool)
        )
        if review_support.shape != tissue.shape:
            raise ValueError("review_support must have the same shape as tissue")
        review_support = review_support | tissue | arrays["fold"] | arrays["crack"]
        object.__setattr__(self, "tissue", tissue)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "review_support", review_support)


@dataclass(frozen=True)
class HybridResult:
    """Hybrid detector outputs, including evidence needed for review/debugging."""

    candidates: CandidateMasks
    anomaly_score: FloatArray
    fused_score: FloatArray
    predicted_mask: BoolArray
    feature_table: FeatureTable

    def __post_init__(self) -> None:
        shape = self.candidates.tissue.shape
        anomaly = np.asarray(self.anomaly_score, dtype=np.float64)
        fused = np.asarray(self.fused_score, dtype=np.float64)
        predicted = np.asarray(self.predicted_mask, dtype=bool)
        if anomaly.shape != shape or fused.shape != shape or predicted.shape != shape:
            raise ValueError("Hybrid result maps must match the candidate-mask shape")
        object.__setattr__(self, "anomaly_score", anomaly)
        object.__setattr__(self, "fused_score", fused)
        object.__setattr__(self, "predicted_mask", predicted)


def _mode(modality: str) -> tuple[str, bool]:
    normalized = modality.lower().replace("-", "").replace("_", "")
    if normalized in {"he", "h&e", "brightfield", "brightfieldhe"}:
        return "he", True
    if normalized in {"comet", "cosmx", "fluorescence", "if", "multiplexif"}:
        return "fluorescence", False
    raise ValueError(f"Unsupported modality: {modality!r}")


def _validate_support(
    tissue: ArrayLike | None, shape: tuple[int, int]
) -> BoolArray | None:
    if tissue is None:
        return None
    support = np.asarray(tissue, dtype=bool)
    if support.shape != shape:
        raise ValueError(
            f"tissue shape {support.shape} does not match image shape {shape}"
        )
    return support


def _normalize_pixel_size_um(
    pixel_size_um: float | Sequence[float] | None,
) -> tuple[float, float] | None:
    """Normalize optional ``(y, x)`` pixel spacing without imposing a default."""

    if pixel_size_um is None:
        return None
    values = np.asarray(pixel_size_um, dtype=np.float64)
    if values.ndim == 0:
        spacing = (float(values), float(values))
    elif values.shape == (2,):
        spacing = (float(values[0]), float(values[1]))
    else:
        raise ValueError("pixel_size_um must be a scalar or (y_um, x_um)")
    if not np.isfinite(spacing).all() or min(spacing) <= 0:
        raise ValueError("pixel_size_um values must be finite and strictly positive")
    return spacing


def _physical_disk(radius_um: float, spacing: tuple[float, float]) -> BoolArray:
    """Return an anisotropic footprint representing a disk in physical space."""

    radius = float(radius_um)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("physical radii must be finite and non-negative")
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    radius_y = max(1, int(np.ceil(radius / spacing[0])))
    radius_x = max(1, int(np.ceil(radius / spacing[1])))
    yy, xx = np.ogrid[-radius_y : radius_y + 1, -radius_x : radius_x + 1]
    footprint = ((yy * spacing[0]) / radius) ** 2 + (
        (xx * spacing[1]) / radius
    ) ** 2 <= 1.0
    footprint[radius_y, radius_x] = True
    return footprint


def _physical_size_to_pixels(
    size_um: float | Sequence[float],
    spacing: tuple[float, float] | None,
    *,
    name: str,
) -> tuple[int, int]:
    """Convert a scalar or ``(y, x)`` physical context to nearest pixels."""

    if spacing is None:
        raise ValueError(f"pixel_size_um is required when {name} is supplied")
    values = np.asarray(size_um, dtype=np.float64)
    if values.ndim == 0:
        physical = (float(values), float(values))
    elif values.shape == (2,):
        physical = (float(values[0]), float(values[1]))
    else:
        raise ValueError(f"{name} must be a scalar or (y_um, x_um)")
    if not np.isfinite(physical).all() or min(physical) <= 0:
        raise ValueError(f"{name} values must be finite and strictly positive")
    return tuple(
        max(1, int(np.rint(length / pixel_spacing)))
        for length, pixel_spacing in zip(physical, spacing, strict=True)
    )  # type: ignore[return-value]


def _resolve_footprint(
    radius_pixels: int,
    *,
    radius_um: float | None,
    spacing: tuple[float, float] | None,
) -> BoolArray:
    if radius_um is None:
        return _disk(radius_pixels)
    if spacing is None:
        raise ValueError("pixel_size_um is required when a physical radius is supplied")
    return _physical_disk(radius_um, spacing)


def _resolve_component_areas(
    min_area: int,
    max_area: int | None,
    *,
    min_area_um2: float | None,
    max_area_um2: float | None,
    spacing: tuple[float, float] | None,
) -> tuple[int, int | None]:
    if min_area_um2 is not None or max_area_um2 is not None:
        if spacing is None:
            raise ValueError(
                "pixel_size_um is required when a physical component area is supplied"
            )
    pixel_area = None if spacing is None else spacing[0] * spacing[1]
    if min_area_um2 is not None:
        value = float(min_area_um2)
        if not np.isfinite(value) or value < 0:
            raise ValueError("min_area_um2 must be finite and non-negative")
        assert pixel_area is not None
        min_area = int(np.ceil(value / pixel_area))
    if max_area_um2 is not None:
        value = float(max_area_um2)
        if not np.isfinite(value) or value < 0:
            raise ValueError("max_area_um2 must be finite and non-negative")
        assert pixel_area is not None
        max_area = int(np.floor(value / pixel_area))
    return int(min_area), None if max_area is None else int(max_area)


def _upper_tail_score(
    array: FloatArray, support: BoolArray, low_q: float, high_q: float
) -> FloatArray:
    values = array[support & np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float64)
    low, high = np.percentile(values, (low_q, high_q))
    spread = float(high - low)
    if spread <= 1e-10:
        return np.zeros_like(array, dtype=np.float64)
    return np.clip((array - low) / spread, 0.0, 1.0)


def _lower_signal_score(array: FloatArray, support: BoolArray) -> FloatArray:
    values = array[support & np.isfinite(array)]
    if values.size == 0:
        return np.zeros_like(array, dtype=np.float64)
    low = float(np.percentile(values, 25.0))
    high = float(np.percentile(values, 75.0))
    denominator = max(low * 0.75, high - low, 0.025)
    return np.clip((low - array) / denominator, 0.0, 1.0)


def _refine_he_support(
    channels: FloatArray,
    support: BoolArray,
    *,
    min_component_size: int,
) -> BoolArray:
    """Reject weak noisy background while leaving low-OD internal gaps open."""

    rgb = channels[..., :3]
    od_total = np.sum(rgb_to_optical_density(rgb), axis=-1)
    maximum = np.max(rgb, axis=-1)
    saturation = (maximum - np.min(rgb, axis=-1)) / np.maximum(maximum, 1e-6)
    structural_signal = (od_total > 0.18) | (saturation > 0.055)
    refined = support & structural_signal
    return connected_component_cleanup(
        refined,
        min_area=max(4, min_component_size),
        closing_radius=1,
    )


def connected_component_cleanup(
    mask: ArrayLike,
    *,
    min_area: int = 16,
    max_area: int | None = None,
    fill_holes: bool = False,
    closing_radius: int = 0,
    opening_radius: int = 0,
    pixel_size_um: float | Sequence[float] | None = None,
    min_area_um2: float | None = None,
    max_area_um2: float | None = None,
    closing_radius_um: float | None = None,
    opening_radius_um: float | None = None,
) -> BoolArray:
    """Apply morphology and deterministic area filtering to a binary mask.

    Pixel-unit arguments preserve the original API.  When physical geometry is
    supplied, ``pixel_size_um`` converts areas and creates anisotropic
    structuring elements so thresholds represent the same specimen geometry at
    different scan resolutions.
    """

    result = np.asarray(mask, dtype=bool)
    if result.ndim != 2:
        raise ValueError("mask must be two-dimensional")
    spacing = _normalize_pixel_size_um(pixel_size_um)
    min_area, max_area = _resolve_component_areas(
        min_area,
        max_area,
        min_area_um2=min_area_um2,
        max_area_um2=max_area_um2,
        spacing=spacing,
    )
    if min_area < 0:
        raise ValueError("min_area cannot be negative")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area must be at least min_area")
    opening_footprint = _resolve_footprint(
        opening_radius,
        radius_um=opening_radius_um,
        spacing=spacing,
    )
    closing_footprint = _resolve_footprint(
        closing_radius,
        radius_um=closing_radius_um,
        spacing=spacing,
    )
    if opening_radius > 0 or (opening_radius_um is not None and opening_radius_um > 0):
        result = ndimage.binary_opening(result, structure=opening_footprint)
    if closing_radius > 0 or (closing_radius_um is not None and closing_radius_um > 0):
        result = ndimage.binary_closing(result, structure=closing_footprint)
    if fill_holes:
        result = ndimage.binary_fill_holes(result)
    labels, count = ndimage.label(result)
    if count == 0:
        return np.zeros_like(result, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_area)
    if max_area is not None:
        keep &= areas <= int(max_area)
    keep[0] = False
    return keep[labels]


def _elongated_component_filter(
    mask: BoolArray,
    min_area: int,
    *,
    pixel_size_um: tuple[float, float] | None = None,
) -> BoolArray:
    labels, count = ndimage.label(mask)
    output = np.zeros_like(mask, dtype=bool)
    for label_id in range(1, count + 1):
        yy, xx = np.nonzero(labels == label_id)
        area = yy.size
        if area < min_area:
            continue
        height = int(yy.max() - yy.min() + 1)
        width = int(xx.max() - xx.min() + 1)
        if max(height, width) < 5:
            continue
        coordinates = np.column_stack((yy, xx)).astype(np.float64)
        if pixel_size_um is not None:
            coordinates *= np.asarray(pixel_size_um, dtype=np.float64)
        if area >= 3:
            covariance = np.cov(coordinates, rowvar=False)
            eigenvalues = np.linalg.eigvalsh(np.atleast_2d(covariance))
            elongation = float(
                np.sqrt((eigenvalues[-1] + 1e-6) / (max(eigenvalues[0], 0.0) + 1e-6))
            )
        else:
            elongation = float(max(height, width) / max(min(height, width), 1))
        occupancy = area / float(height * width)
        # Thin or branching discontinuities pass; compact gland/lumen-like blobs
        # are suppressed.  This remains a proposal rule, not a semantic claim.
        if elongation >= 1.7 or occupancy <= 0.38:
            output[labels == label_id] = True
    return output


def classical_fold_candidates(
    image: ArrayLike,
    *,
    modality: str = "he",
    tissue: ArrayLike | None = None,
    channel_axis: int = -1,
    min_component_size: int = 16,
    pixel_size_um: float | Sequence[float] | None = None,
    min_component_area_um2: float | None = None,
    morphology_radius_um: float | None = None,
) -> tuple[BoolArray, FloatArray]:
    """Return a conservative fold candidate mask and continuous score map.

    The H&E score combines high optical density, saturation, and local texture.
    The fluorescence score combines unusually high structural signal, local
    contrast, and edges.  Scores are cohort-free within-image tail scores and
    should be calibrated on the intended scanner/panel before deployment.
    """

    mode, is_he = _mode(modality)
    spacing = _normalize_pixel_size_um(pixel_size_um)
    channels = _channels_last(image, channel_axis)
    support_was_supplied = tissue is not None
    support = _validate_support(tissue, channels.shape[:2])
    if support is None:
        support = tissue_mask(
            channels,
            modality=mode,
            channel_axis=-1,
            min_component_size=max(4, min_component_size),
            closing_radius=1,
        )
    if is_he and not support_was_supplied:
        support = _refine_he_support(
            channels,
            support,
            min_component_size=min_component_size,
        )
    if not np.any(support):
        empty_score = np.zeros(channels.shape[:2], dtype=np.float64)
        return np.zeros_like(empty_score, dtype=bool), empty_score

    if is_he:
        if channels.shape[-1] < 3:
            raise ValueError("H&E fold detection requires three RGB channels")
        rgb = channels[..., :3]
        od_total = np.sum(rgb_to_optical_density(rgb), axis=-1)
        maximum = np.max(rgb, axis=-1)
        saturation = (maximum - np.min(rgb, axis=-1)) / np.maximum(maximum, 1e-6)
        texture = _local_standard_deviation(od_total, size=5)
        score = (
            0.57 * _upper_tail_score(od_total, support, 65.0, 98.5)
            + 0.28 * _upper_tail_score(saturation, support, 65.0, 98.5)
            + 0.15 * _upper_tail_score(texture, support, 70.0, 99.0)
        )
    else:
        scaled = np.stack(
            [
                _robust_unit_scale(channels[..., index], support)
                for index in range(channels.shape[-1])
            ],
            axis=-1,
        )
        aggregate = np.max(scaled, axis=-1)
        texture = _local_standard_deviation(aggregate, size=5)
        gradient = _gradient_magnitude(aggregate)
        score = (
            0.60 * _upper_tail_score(aggregate, support, 70.0, 99.0)
            + 0.22 * _upper_tail_score(texture, support, 70.0, 99.0)
            + 0.18 * _upper_tail_score(gradient, support, 70.0, 99.0)
        )

    score = np.clip(score, 0.0, 1.0) * support
    candidate = connected_component_cleanup(
        score >= 0.53,
        min_area=min_component_size,
        closing_radius=1,
        pixel_size_um=spacing,
        min_area_um2=min_component_area_um2,
        closing_radius_um=morphology_radius_um,
    )
    return candidate & support, score


def classical_crack_candidates(
    image: ArrayLike,
    *,
    modality: str = "he",
    tissue: ArrayLike | None = None,
    channel_axis: int = -1,
    min_component_size: int = 8,
    pixel_size_um: float | Sequence[float] | None = None,
    min_component_area_um2: float | None = None,
    neighborhood_radius_um: float | None = None,
) -> tuple[BoolArray, FloatArray]:
    """Return tear/crack-like low-signal discontinuities and a score map.

    Candidate pixels must have tissue on multiple sides, low structural signal,
    and nearby edge evidence.  An elongation filter suppresses many compact
    lumens, but anatomy remains an expected hard negative for expert review.
    """

    mode, is_he = _mode(modality)
    spacing = _normalize_pixel_size_um(pixel_size_um)
    channels = _channels_last(image, channel_axis)
    support_was_supplied = tissue is not None
    support = _validate_support(tissue, channels.shape[:2])
    if support is None:
        support = tissue_mask(
            channels,
            modality=mode,
            channel_axis=-1,
            min_component_size=max(4, min_component_size),
            closing_radius=1,
        )
    if is_he and not support_was_supplied:
        support = _refine_he_support(
            channels,
            support,
            min_component_size=min_component_size,
        )
    if not np.any(support):
        empty_score = np.zeros(channels.shape[:2], dtype=np.float64)
        return np.zeros_like(empty_score, dtype=bool), empty_score

    if is_he:
        if channels.shape[-1] < 3:
            raise ValueError("H&E crack detection requires three RGB channels")
        signal = np.sum(rgb_to_optical_density(channels[..., :3]), axis=-1)
    else:
        scaled = np.stack(
            [
                _robust_unit_scale(channels[..., index], support)
                for index in range(channels.shape[-1])
            ],
            axis=-1,
        )
        signal = np.max(scaled, axis=-1)

    low_signal = _lower_signal_score(signal, support)
    neighborhood_radius = max(2, min(5, min(signal.shape) // 64 + 2))
    footprint = _resolve_footprint(
        neighborhood_radius,
        radius_um=neighborhood_radius_um,
        spacing=spacing,
    )
    kernel = footprint.astype(np.float64)
    kernel /= float(kernel.sum())
    tissue_density = ndimage.convolve(
        support.astype(np.float64), kernel, mode="constant", cval=0.0
    )
    # A meaningful fraction of a symmetric neighborhood must be tissue.  An
    # internal thin gap satisfies this; the explicit envelope and distance
    # checks below guard against tracing the exterior specimen perimeter.
    surrounding_tissue = np.clip((tissue_density - 0.35) / 0.55, 0.0, 1.0)
    gradient = _gradient_magnitude(signal)
    nearby_gradient = ndimage.maximum_filter(
        gradient, footprint=footprint, mode="reflect"
    )
    edge_evidence = _upper_tail_score(nearby_gradient, support, 60.0, 98.0)
    # Fill only holes enclosed by the detected specimen.  This recovers the
    # canonical low-signal appearance of an internal tear without extending
    # the review region across the external specimen boundary.
    closed_support = ndimage.binary_closing(support, structure=footprint)
    tissue_envelope = ndimage.binary_fill_holes(closed_support)
    gap_evidence = ((~support) & tissue_envelope).astype(np.float64)

    score = (
        0.47 * low_signal
        + 0.28 * surrounding_tissue
        + 0.15 * edge_evidence
        + 0.10 * gap_evidence
    )
    if spacing is None:
        interior_distance = ndimage.distance_transform_edt(tissue_envelope)
        edge_guard = interior_distance > max(1.0, neighborhood_radius * 0.35)
    else:
        interior_distance = ndimage.distance_transform_edt(
            tissue_envelope, sampling=spacing
        )
        physical_radius = (
            float(neighborhood_radius_um)
            if neighborhood_radius_um is not None
            else neighborhood_radius * min(spacing)
        )
        edge_guard = interior_distance > max(min(spacing), physical_radius * 0.35)
    search_region = tissue_envelope & (tissue_density > 0.32) & edge_guard
    score = np.clip(score, 0.0, 1.0) * search_region
    raw_candidate = (score >= 0.54) & search_region
    resolved_min_area, _ = _resolve_component_areas(
        min_component_size,
        None,
        min_area_um2=min_component_area_um2,
        max_area_um2=None,
        spacing=spacing,
    )
    candidate = _elongated_component_filter(
        raw_candidate,
        max(2, resolved_min_area),
        pixel_size_um=spacing,
    )
    candidate = connected_component_cleanup(
        candidate,
        min_area=min_component_size,
        pixel_size_um=spacing,
        min_area_um2=min_component_area_um2,
    )
    return candidate, score


def classical_candidate_masks(
    image: ArrayLike,
    *,
    modality: str = "he",
    tissue: ArrayLike | None = None,
    channel_axis: int = -1,
    min_component_size: int = 16,
    pixel_size_um: float | Sequence[float] | None = None,
    min_component_area_um2: float | None = None,
    crack_neighborhood_radius_um: float | None = None,
    fold_morphology_radius_um: float | None = None,
) -> CandidateMasks:
    """Run both classical proposal branches with a shared tissue mask."""

    mode, _ = _mode(modality)
    channels = _channels_last(image, channel_axis)
    support_was_supplied = tissue is not None
    support = _validate_support(tissue, channels.shape[:2])
    if support is None:
        support = tissue_mask(
            channels,
            modality=mode,
            channel_axis=-1,
            min_component_size=max(4, min_component_size),
            closing_radius=1,
        )
    if mode == "he" and not support_was_supplied:
        support = _refine_he_support(
            channels,
            support,
            min_component_size=min_component_size,
        )
    fold, fold_score = classical_fold_candidates(
        channels,
        modality=mode,
        tissue=support,
        channel_axis=-1,
        min_component_size=min_component_size,
        pixel_size_um=pixel_size_um,
        min_component_area_um2=min_component_area_um2,
        morphology_radius_um=fold_morphology_radius_um,
    )
    crack, crack_score = classical_crack_candidates(
        channels,
        modality=mode,
        tissue=support,
        channel_axis=-1,
        min_component_size=max(4, min_component_size // 2),
        pixel_size_um=pixel_size_um,
        min_component_area_um2=min_component_area_um2,
        neighborhood_radius_um=crack_neighborhood_radius_um,
    )
    return CandidateMasks(
        support,
        fold,
        crack,
        fold_score,
        crack_score,
        review_support=support | fold | crack,
    )


class CleanReferenceAnomalyDetector:
    """Robust PCA/Mahalanobis detector fitted only on reviewed-clean features.

    Median/MAD scaling limits the influence of accidental contamination in the
    clean bank.  PCA removes singular/noisy directions; shrinkage stabilizes the
    covariance when clean examples are limited.  This is an anomaly ranker, not
    proof of artifact identity.
    """

    def __init__(
        self,
        *,
        variance_retained: float = 0.98,
        shrinkage: float = 0.15,
        threshold_quantile: float = 0.995,
        max_components: int | None = None,
        regularization: float = 1e-6,
    ) -> None:
        if not 0.0 < variance_retained <= 1.0:
            raise ValueError("variance_retained must lie in (0, 1]")
        if not 0.0 <= shrinkage <= 1.0:
            raise ValueError("shrinkage must lie in [0, 1]")
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie in (0.5, 1)")
        if max_components is not None and max_components <= 0:
            raise ValueError("max_components must be positive")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        self.variance_retained = float(variance_retained)
        self.shrinkage = float(shrinkage)
        self.threshold_quantile = float(threshold_quantile)
        self.max_components = max_components
        self.regularization = float(regularization)
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _as_matrix(self, features: FeatureTable | ArrayLike) -> FloatArray:
        matrix = (
            features.values
            if isinstance(features, FeatureTable)
            else np.asarray(features)
        )
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2:
            raise ValueError("features must have shape (n_samples, n_features)")
        if matrix.shape[1] == 0:
            raise ValueError("features must contain at least one column")
        return matrix

    def fit(
        self, features: FeatureTable | ArrayLike
    ) -> "CleanReferenceAnomalyDetector":
        matrix = self._as_matrix(features)
        if matrix.shape[0] < 3:
            raise ValueError("At least three clean-reference samples are required")
        finite = np.where(np.isfinite(matrix), matrix, np.nan)
        medians = np.nanmedian(finite, axis=0)
        if np.any(~np.isfinite(medians)):
            raise ValueError(
                "Every feature column needs at least one finite clean value"
            )
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        mad = np.median(np.abs(matrix - medians), axis=0)
        standard_deviation = np.std(matrix, axis=0)
        scale = 1.4826 * mad
        scale = np.where(scale > 1e-10, scale, standard_deviation)
        scale = np.where(scale > 1e-10, scale, 1.0)
        standardized = np.clip((matrix - medians) / scale, -20.0, 20.0)

        _, singular_values, right_vectors = np.linalg.svd(
            standardized, full_matrices=False
        )
        variance = singular_values * singular_values
        if float(np.sum(variance)) <= 1e-12:
            component_count = 1
            components = np.zeros((1, matrix.shape[1]), dtype=np.float64)
            components[0, 0] = 1.0
        else:
            cumulative = np.cumsum(variance) / np.sum(variance)
            component_count = int(
                np.searchsorted(cumulative, self.variance_retained) + 1
            )
            component_count = min(component_count, matrix.shape[0] - 1, matrix.shape[1])
            if self.max_components is not None:
                component_count = min(component_count, self.max_components)
            component_count = max(component_count, 1)
            components = right_vectors[:component_count]

        projected = standardized @ components.T
        projected_center = np.median(projected, axis=0)
        centered = projected - projected_center
        covariance = np.atleast_2d(np.cov(centered, rowvar=False, ddof=1))
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - self.shrinkage) * covariance + self.shrinkage * diagonal
        trace_scale = float(np.trace(covariance) / covariance.shape[0])
        covariance += (
            np.eye(covariance.shape[0]) * self.regularization * max(trace_scale, 1.0)
        )

        self.feature_medians_ = medians
        self.feature_scales_ = scale
        self.components_ = components
        self.projected_center_ = projected_center
        self.precision_ = np.linalg.pinv(covariance, hermitian=True)
        self.n_features_in_ = int(matrix.shape[1])
        self._fitted = True
        training_scores = self.score_samples(matrix)
        threshold = float(np.quantile(training_scores, self.threshold_quantile))
        self.threshold_ = max(threshold, np.finfo(np.float64).eps)
        self.training_scores_ = training_scores
        return self

    def _prepare(self, features: FeatureTable | ArrayLike) -> FloatArray:
        if not self._fitted:
            raise RuntimeError(
                "CleanReferenceAnomalyDetector must be fitted before scoring"
            )
        matrix = self._as_matrix(features)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, received {matrix.shape[1]}"
            )
        matrix = np.where(np.isfinite(matrix), matrix, self.feature_medians_)
        standardized = np.clip(
            (matrix - self.feature_medians_) / self.feature_scales_, -50.0, 50.0
        )
        return standardized @ self.components_.T - self.projected_center_

    def score_samples(self, features: FeatureTable | ArrayLike) -> FloatArray:
        """Return non-negative robust Mahalanobis distances."""

        centered = self._prepare(features)
        squared = np.einsum("ni,ij,nj->n", centered, self.precision_, centered)
        return np.sqrt(np.maximum(squared, 0.0))

    def calibrated_scores(self, features: FeatureTable | ArrayLike) -> FloatArray:
        """Map scores to ``[0, 1]`` with the clean threshold represented by 0.5."""

        scores = self.score_samples(features)
        return np.clip(0.5 * scores / self.threshold_, 0.0, 1.0)

    def predict(self, features: FeatureTable | ArrayLike) -> BoolArray:
        """Flag samples beyond the fitted upper clean-reference quantile."""

        return self.score_samples(features) > self.threshold_


def tile_scores_to_map(
    scores: ArrayLike,
    coordinates: ArrayLike,
    image_shape: Sequence[int],
    *,
    reduction: str = "mean",
) -> FloatArray:
    """Project patch scores into an image map using mean or max overlap fusion."""

    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    boxes = np.asarray(coordinates, dtype=np.int64)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or boxes.shape[0] != values.size:
        raise ValueError("coordinates must have shape (len(scores), 4)")
    shape = tuple(int(item) for item in image_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("image_shape must be a pair of positive integers")
    if reduction not in {"mean", "max"}:
        raise ValueError("reduction must be 'mean' or 'max'")
    output = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64) if reduction == "mean" else None
    for value, (y0, x0, y1, x1) in zip(values, boxes, strict=True):
        if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
            raise ValueError(f"Invalid patch coordinate {(y0, x0, y1, x1)}")
        if reduction == "mean":
            output[y0:y1, x0:x1] += value
            assert count is not None
            count[y0:y1, x0:x1] += 1.0
        else:
            output[y0:y1, x0:x1] = np.maximum(output[y0:y1, x0:x1], value)
    if count is not None:
        output = np.divide(output, count, out=np.zeros_like(output), where=count > 0)
    return output


def fuse_score_maps(
    score_maps: Sequence[ArrayLike],
    *,
    weights: Sequence[float] | None = None,
    tissue: ArrayLike | None = None,
) -> FloatArray:
    """Fuse same-resolution evidence maps with a convex weighted mean."""

    maps = [np.asarray(score, dtype=np.float64) for score in score_maps]
    if not maps:
        raise ValueError("At least one score map is required")
    if any(score.shape != maps[0].shape for score in maps):
        raise ValueError("All score maps must have the same shape")
    if weights is None:
        weight_array = np.ones(len(maps), dtype=np.float64)
    else:
        weight_array = np.asarray(tuple(weights), dtype=np.float64)
        if weight_array.shape != (len(maps),):
            raise ValueError("weights must have one value per score map")
    if (
        np.any(~np.isfinite(weight_array))
        or np.any(weight_array < 0)
        or weight_array.sum() <= 0
    ):
        raise ValueError("weights must be finite, non-negative, and have positive sum")
    weight_array /= weight_array.sum()
    fused = np.zeros_like(maps[0], dtype=np.float64)
    for weight, score in zip(weight_array, maps, strict=True):
        fused += weight * np.clip(np.nan_to_num(score, nan=0.0), 0.0, 1.0)
    if tissue is not None:
        support = np.asarray(tissue, dtype=bool)
        if support.shape != fused.shape:
            raise ValueError("tissue must match score-map shapes")
        fused *= support
    return np.clip(fused, 0.0, 1.0)


class HybridQCDetector:
    """Fuse classical proposals with patch anomaly scores from clean references.

    The final decision is an explicit OR of three reviewable branches: fused
    evidence, classical proposals, and sufficiently strong anomaly-only
    evidence.  This prevents convex averaging from silencing a confident
    anomaly signal and preserves crack candidates outside conservative tissue.
    """

    def __init__(
        self,
        anomaly_detector: CleanReferenceAnomalyDetector | None = None,
        *,
        classical_weight: float = 0.55,
        anomaly_weight: float = 0.45,
        decision_threshold: float = 0.5,
        anomaly_decision_threshold: float = 0.75,
    ) -> None:
        if (
            classical_weight < 0
            or anomaly_weight < 0
            or classical_weight + anomaly_weight <= 0
        ):
            raise ValueError("Fusion weights must be non-negative with a positive sum")
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError("decision_threshold must lie in [0, 1]")
        if not 0.0 <= anomaly_decision_threshold <= 1.0:
            raise ValueError("anomaly_decision_threshold must lie in [0, 1]")
        self.anomaly_detector = anomaly_detector or CleanReferenceAnomalyDetector()
        self.classical_weight = float(classical_weight)
        self.anomaly_weight = float(anomaly_weight)
        self.decision_threshold = float(decision_threshold)
        self.anomaly_decision_threshold = float(anomaly_decision_threshold)

    def fit(self, reference: FeatureTable | ArrayLike) -> "HybridQCDetector":
        self.anomaly_detector.fit(reference)
        return self

    def score(
        self,
        image: ArrayLike,
        *,
        modality: str = "he",
        patch_size: int | Sequence[int] = 64,
        stride: int | Sequence[int] | None = None,
        patch_size_um: float | Sequence[float] | None = None,
        stride_um: float | Sequence[float] | None = None,
        tissue: ArrayLike | None = None,
        channel_axis: int = -1,
        structural_channels: Sequence[int] | None = None,
        pixel_size_um: float | Sequence[float] | None = None,
        min_component_area_um2: float | None = None,
        crack_neighborhood_radius_um: float | None = None,
        fold_morphology_radius_um: float | None = None,
    ) -> HybridResult:
        if not self.anomaly_detector.is_fitted:
            raise RuntimeError(
                "HybridQCDetector.fit must be called with clean-reference features"
            )
        spacing = _normalize_pixel_size_um(pixel_size_um)
        resolved_patch_size = (
            patch_size
            if patch_size_um is None
            else _physical_size_to_pixels(
                patch_size_um,
                spacing,
                name="patch_size_um",
            )
        )
        resolved_stride = (
            stride
            if stride_um is None
            else _physical_size_to_pixels(
                stride_um,
                spacing,
                name="stride_um",
            )
        )
        candidates = classical_candidate_masks(
            image,
            modality=modality,
            tissue=tissue,
            channel_axis=channel_axis,
            pixel_size_um=pixel_size_um,
            min_component_area_um2=min_component_area_um2,
            crack_neighborhood_radius_um=crack_neighborhood_radius_um,
            fold_morphology_radius_um=fold_morphology_radius_um,
        )
        table = extract_patch_feature_table(
            image,
            modality=modality,
            patch_size=resolved_patch_size,
            stride=resolved_stride,
            tissue=candidates.tissue,
            channel_axis=channel_axis,
            structural_channels=structural_channels,
        )
        if len(table):
            patch_anomaly = self.anomaly_detector.calibrated_scores(table)
            anomaly_map = tile_scores_to_map(
                patch_anomaly,
                table.coordinates,
                table.image_shape,
                reduction="mean",
            )
        else:
            anomaly_map = np.zeros(candidates.tissue.shape, dtype=np.float64)
        classical_map = np.maximum(candidates.fold_score, candidates.crack_score)
        fused = fuse_score_maps(
            (classical_map, anomaly_map),
            weights=(self.classical_weight, self.anomaly_weight),
            tissue=candidates.review_support,
        )
        fusion_alert = fused >= self.decision_threshold
        classical_alert = candidates.fold | candidates.crack
        anomaly_alert = (
            anomaly_map >= self.anomaly_decision_threshold
        ) & candidates.review_support
        predicted = connected_component_cleanup(
            fusion_alert | classical_alert | anomaly_alert,
            min_area=max(4, int(np.prod(table.patch_size) // 128)),
            closing_radius=1,
            pixel_size_um=pixel_size_um,
            min_area_um2=min_component_area_um2,
        )
        return HybridResult(candidates, anomaly_map, fused, predicted, table)

    def detect(self, *args: Any, **kwargs: Any) -> HybridResult:
        """Alias for :meth:`score` for detector-oriented call sites."""

        return self.score(*args, **kwargs)


@runtime_checkable
class PatchEncoder(Protocol):
    """Minimal interface for frozen image encoders used in optional experiments."""

    def encode(self, patches: ArrayLike) -> FloatArray:
        """Return a two-dimensional ``(n_patches, embedding_dim)`` array."""


class FrozenDINOv2Encoder:
    """Optional frozen DINOv2 patch encoder with explicit dependency/network use.

    A caller can inject an already-loaded torch model, which is the recommended
    offline/corporate path.  Setting ``allow_download=True`` loads the requested
    official DINOv2 model through torch hub.  The default never accesses the
    network and fails with an actionable message if no model was supplied.
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        *,
        device: str = "auto",
        model: Any | None = None,
        allow_download: bool = False,
        image_size: int = 224,
    ) -> None:
        try:
            import torch
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "FrozenDINOv2Encoder requires PyTorch. Install an approved torch build "
                "or use the default handcrafted features."
            ) from error
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self._torch = torch
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = str(device)
        self.image_size = int(image_size)
        if model is None:
            if not allow_download:
                raise RuntimeError(
                    "No DINOv2 model was supplied. Inject a locally approved frozen torch "
                    "model, or explicitly set allow_download=True to use torch.hub."
                )
            try:  # pragma: no cover - requires network/cache
                model = torch.hub.load(
                    "facebookresearch/dinov2", model_name, pretrained=True
                )
            except Exception as error:
                raise RuntimeError(
                    f"Unable to load {model_name!r} from the DINOv2 torch hub"
                ) from error
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode(self, patches: ArrayLike) -> FloatArray:
        array = np.asarray(patches)
        if array.ndim == 3:
            array = array[None, ...]
        if array.ndim != 4:
            raise ValueError("patches must have shape (N,H,W,C) or (H,W,C)")
        # Accept channels-first batches only when the second dimension is an
        # unmistakable channel count.
        if array.shape[-1] not in (1, 3, 4) and array.shape[1] in (1, 3, 4):
            array = np.moveaxis(array, 1, -1)
        if array.shape[-1] not in (1, 3, 4):
            raise ValueError(
                "FrozenDINOv2Encoder accepts only 1, 3, or 4 image channels; "
                "multiplex inputs require an explicit semantic RGB projection "
                "before encoding"
            )
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        normalized = np.stack(
            [_channels_last(patch, -1)[..., :3] for patch in array], axis=0
        )
        # Keep input validation independent of the optional torch dependency so
        # unsafe multiplex truncation fails before any model/device work.
        torch = self._torch
        tensor = torch.as_tensor(
            normalized,
            dtype=torch.float32,
            device=self.device,
        ).permute(0, 3, 1, 2)
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor((0.485, 0.456, 0.406), device=self.device).view(1, 3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=self.device).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        with torch.inference_mode():
            output = self.model(tensor)
        if isinstance(output, dict):
            if "x_norm_clstoken" in output:
                output = output["x_norm_clstoken"]
            else:
                tensor_values = [
                    value for value in output.values() if torch.is_tensor(value)
                ]
                if not tensor_values:
                    raise TypeError(
                        "DINOv2 model returned a dictionary without tensor outputs"
                    )
                output = tensor_values[0]
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not torch.is_tensor(output):
            raise TypeError("DINOv2 model must return a tensor, tuple, or dictionary")
        output = output.reshape(output.shape[0], -1)
        return output.detach().cpu().numpy().astype(np.float64, copy=False)


__all__ = [
    "CandidateMasks",
    "CleanReferenceAnomalyDetector",
    "FrozenDINOv2Encoder",
    "HybridQCDetector",
    "HybridResult",
    "PatchEncoder",
    "classical_candidate_masks",
    "classical_crack_candidates",
    "classical_fold_candidates",
    "connected_component_cleanup",
    "fuse_score_maps",
    "tile_scores_to_map",
]
