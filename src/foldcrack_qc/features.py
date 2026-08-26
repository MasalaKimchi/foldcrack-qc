"""Dependency-light image features for fold/crack quality control.

The functions in this module intentionally operate on NumPy arrays and return
plain dictionaries/arrays.  They are suitable as transparent baselines and as
inputs to the clean-reference detector in :mod:`foldcrack_qc.detectors`.

Images are expected as ``H x W`` or three-dimensional arrays.  Three-
dimensional arrays default to channels-last; pass ``channel_axis=0`` for
``C x H x W`` microscopy data.  Bright-field values may be uint8, uint16, or
floating point.  Fluorescence intensities are normalized robustly per channel
for feature computation, while the source data are never modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


HE_FEATURE_NAMES = (
    "tissue_fraction",
    "od_mean",
    "od_std",
    "od_p90",
    "hematoxylin_mean",
    "hematoxylin_p90",
    "eosin_mean",
    "eosin_p90",
    "saturation_mean",
    "saturation_p90",
    "gray_entropy",
    "local_contrast_mean",
    "local_contrast_p90",
    "gradient_mean",
    "gradient_p90",
    "edge_fraction",
    "dark_fraction",
)


FLUORESCENCE_FEATURE_NAMES = (
    "tissue_fraction",
    "aggregate_mean",
    "aggregate_std",
    "aggregate_p90",
    "aggregate_p99",
    "foreground_fraction",
    "local_contrast_mean",
    "local_contrast_p90",
    "gradient_mean",
    "gradient_p90",
    "edge_fraction",
    "channel_mean_mean",
    "channel_mean_std",
    "channel_dynamic_range_mean",
    "channel_saturation_mean",
    "cross_channel_correlation",
)


@dataclass(frozen=True)
class FeatureTable:
    """Patch-level features and their spatial provenance.

    ``coordinates`` contains half-open ``(y0, x0, y1, x1)`` image coordinates.
    This lets an anomaly score for each row be projected back to the native
    image without depending on a WSI library.
    """

    values: FloatArray
    coordinates: NDArray[np.int64]
    names: tuple[str, ...]
    patch_size: tuple[int, int]
    image_shape: tuple[int, int]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        coordinates = np.asarray(self.coordinates, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError(
                "FeatureTable.values must have shape (n_patches, n_features)"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 4:
            raise ValueError("FeatureTable.coordinates must have shape (n_patches, 4)")
        if values.shape[0] != coordinates.shape[0]:
            raise ValueError("Feature rows and coordinates must have the same length")
        if values.shape[1] != len(self.names):
            raise ValueError("Feature column count must match names")
        if len(self.patch_size) != 2 or min(self.patch_size) <= 0:
            raise ValueError("patch_size must contain two positive integers")
        if len(self.image_shape) != 2 or min(self.image_shape) <= 0:
            raise ValueError("image_shape must contain two positive integers")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "coordinates", coordinates)

    def __len__(self) -> int:
        return int(self.values.shape[0])


def normalize_image(
    image: ArrayLike, *, white_level: float | None = None
) -> FloatArray:
    """Convert an image to finite float values in ``[0, 1]``.

    Integer arrays use the dtype maximum as their physical white/saturation
    level.  Floating point arrays in ``[0, 255]`` or ``[0, 65535]`` are scaled
    accordingly.  Set ``white_level`` explicitly for unusual scanner exports.
    """

    source = np.asarray(image)
    if source.ndim not in (2, 3):
        raise ValueError(f"Expected a 2-D or 3-D image, got shape {source.shape}")
    if source.size == 0:
        raise ValueError("Image cannot be empty")

    array = source.astype(np.float64, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(
            "Image contains NaN or infinity; abstain or encode acquisition "
            "dropout explicitly instead of converting corruption into signal"
        )
    if white_level is not None:
        if not np.isfinite(white_level) or white_level <= 0:
            raise ValueError("white_level must be a positive finite value")
        scale = float(white_level)
    elif np.issubdtype(source.dtype, np.bool_):
        scale = 1.0
    elif np.issubdtype(source.dtype, np.integer):
        info = np.iinfo(source.dtype)
        scale = float(info.max)
        if info.min < 0:
            array = np.maximum(array, 0.0)
    else:
        maximum = float(np.max(array))
        if maximum <= 1.0:
            scale = 1.0
        elif maximum <= 255.0:
            scale = 255.0
        elif maximum <= 65535.0:
            scale = 65535.0
        else:
            scale = maximum if maximum > 0 else 1.0
    return np.clip(array / max(scale, np.finfo(np.float64).eps), 0.0, 1.0)


def _channels_last(image: ArrayLike, channel_axis: int = -1) -> FloatArray:
    array = normalize_image(image)
    if array.ndim == 2:
        return array[..., None]
    axis = int(channel_axis)
    if axis < 0:
        axis += array.ndim
    if not 0 <= axis < array.ndim:
        raise ValueError(
            f"channel_axis {channel_axis} is out of bounds for an array with {array.ndim} dimensions"
        )
    return np.moveaxis(array, axis, -1)


def rgb_to_optical_density(
    rgb: ArrayLike, *, white_level: float | None = None
) -> FloatArray:
    """Convert a channels-last RGB bright-field image to optical density."""

    image = normalize_image(rgb, white_level=white_level)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError("rgb_to_optical_density expects an H x W x 3(+) image")
    # The half-count offset avoids infinite OD for digitized zero-valued pixels.
    return -np.log(np.clip(image[..., :3], 0.5 / 65535.0, 1.0))


def he_stain_concentrations(rgb: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Estimate H and E concentrations with a fixed Ruifrok-style basis.

    This is deliberately a deterministic baseline rather than a slide-level
    stain-normalization method.  It avoids fitting stain vectors to artifact-
    contaminated tissue.
    """

    od = rgb_to_optical_density(rgb)
    stain_basis = np.asarray(
        [
            [0.650, 0.704, 0.286],  # hematoxylin
            [0.072, 0.990, 0.105],  # eosin
        ],
        dtype=np.float64,
    )
    concentrations = od.reshape(-1, 3) @ np.linalg.pinv(stain_basis)
    concentrations = np.maximum(concentrations, 0.0).reshape(*od.shape[:2], 2)
    return concentrations[..., 0], concentrations[..., 1]


def _disk(radius: int) -> BoolArray:
    radius = int(radius)
    if radius <= 0:
        return np.ones((1, 1), dtype=bool)
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _remove_small_components(mask: BoolArray, min_size: int) -> BoolArray:
    if min_size <= 1 or not np.any(mask):
        return mask.astype(bool, copy=True)
    labels, count = ndimage.label(mask)
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_size)
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: BoolArray, max_size: int) -> BoolArray:
    if max_size <= 0:
        return mask.astype(bool, copy=True)
    background = ~mask
    labels, count = ndimage.label(background)
    if count == 0:
        return mask.astype(bool, copy=True)
    areas = np.bincount(labels.ravel())
    border_labels = np.unique(
        np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
    )
    fill = areas <= int(max_size)
    fill[border_labels] = False
    return mask | fill[labels]


def _robust_unit_scale(
    channel: FloatArray, mask: BoolArray | None = None
) -> FloatArray:
    valid = np.isfinite(channel)
    if mask is not None:
        valid &= mask
    values = channel[valid]
    positive = values[values > 0]
    if positive.size >= 16:
        low, high = np.percentile(positive, (1.0, 99.5))
    elif values.size:
        low, high = float(np.min(values)), float(np.max(values))
    else:
        return np.zeros_like(channel, dtype=np.float64)
    if high <= low + np.finfo(np.float64).eps:
        return np.zeros_like(channel, dtype=np.float64)
    return np.clip((channel - low) / (high - low), 0.0, 1.0)


def _otsu_threshold(values: FloatArray, bins: int = 128) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    minimum, maximum = float(np.min(values)), float(np.max(values))
    if maximum <= minimum + np.finfo(np.float64).eps:
        return maximum
    hist, edges = np.histogram(values, bins=bins, range=(minimum, maximum))
    hist = hist.astype(np.float64)
    centers = (edges[:-1] + edges[1:]) * 0.5
    probability = hist / max(hist.sum(), 1.0)
    omega = np.cumsum(probability)
    mean = np.cumsum(probability * centers)
    total_mean = mean[-1]
    between = (total_mean * omega - mean) ** 2 / np.maximum(
        omega * (1.0 - omega), np.finfo(np.float64).eps
    )
    # The final bin cannot form two classes.
    return float(centers[int(np.argmax(between[:-1]))])


def tissue_mask(
    image: ArrayLike,
    *,
    modality: str = "he",
    channel_axis: int = -1,
    min_component_size: int = 64,
    closing_radius: int = 2,
) -> BoolArray:
    """Create a conservative tissue/support mask for bright-field or fluorescence.

    Supported modality aliases are ``he``, ``h&e``, ``brightfield``, ``comet``,
    ``cosmx``, and ``fluorescence``.  The fluorescence branch intentionally
    merges structural channels; channel selection should happen before calling
    this function if a panel contains non-structural channels.
    """

    mode = modality.lower().replace("-", "").replace("_", "")
    channels = _channels_last(image, channel_axis)
    if mode in {"he", "h&e", "brightfield", "brightfieldhe"}:
        if channels.shape[-1] < 3:
            raise ValueError("H&E tissue masking requires at least three RGB channels")
        rgb = channels[..., :3]
        od_total = np.sum(rgb_to_optical_density(rgb), axis=-1)
        maximum = np.max(rgb, axis=-1)
        saturation = (maximum - np.min(rgb, axis=-1)) / np.maximum(maximum, 1e-6)
        # Either chroma or absorption can establish tissue.  The intensity guard
        # suppresses small numerical departures from a white scanner background.
        mask = ((od_total > 0.12) | (saturation > 0.045)) & (
            np.mean(rgb, axis=-1) < 0.985
        )
    elif mode in {"comet", "cosmx", "fluorescence", "if", "multiplexif"}:
        # A log transform prevents a small saturated/folded region from setting
        # the entire robust range and hiding normally exposed tissue.  Otsu is
        # applied to all pixels so true zero/low background remains a class.
        aggregate = np.max(channels, axis=-1)
        log_aggregate = np.log1p(100.0 * aggregate) / np.log1p(100.0)
        threshold = _otsu_threshold(log_aggregate)
        threshold = float(np.clip(threshold, 1e-4, 0.60))
        mask = log_aggregate > threshold
        # DAPI and membrane markers can be sparse within valid tissue.  A small
        # expansion connects nuclei into a structural support estimate.
        mask = ndimage.binary_dilation(mask, structure=_disk(1), iterations=1)
    else:
        raise ValueError(f"Unsupported modality: {modality!r}")

    if closing_radius > 0:
        mask = ndimage.binary_closing(mask, structure=_disk(closing_radius))
    mask = _remove_small_components(mask.astype(bool), int(min_component_size))
    hole_limit = max(int(min_component_size) * 4, 64)
    return _fill_small_holes(mask, hole_limit)


def _masked_values(array: FloatArray, mask: BoolArray | None) -> FloatArray:
    values = np.asarray(array, dtype=np.float64)
    if mask is not None:
        if mask.shape != values.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {values.shape}"
            )
        values = values[mask]
    else:
        values = values.ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(1, dtype=np.float64)
    return values


def _summary(
    array: FloatArray, mask: BoolArray | None
) -> tuple[float, float, float, float]:
    values = _masked_values(array, mask)
    return (
        float(np.mean(values)),
        float(np.std(values)),
        float(np.percentile(values, 90.0)),
        float(np.percentile(values, 99.0)),
    )


def _entropy(array: FloatArray, mask: BoolArray | None, bins: int = 32) -> float:
    values = _masked_values(np.clip(array, 0.0, 1.0), mask)
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    probability = hist.astype(np.float64)
    probability /= max(float(probability.sum()), 1.0)
    probability = probability[probability > 0]
    return float(-np.sum(probability * np.log2(probability)))


def _gradient_magnitude(array: FloatArray) -> FloatArray:
    gy = ndimage.sobel(array, axis=0, mode="reflect") / 8.0
    gx = ndimage.sobel(array, axis=1, mode="reflect") / 8.0
    return np.hypot(gx, gy)


def _local_standard_deviation(array: FloatArray, size: int = 5) -> FloatArray:
    mean = ndimage.uniform_filter(array, size=size, mode="reflect")
    mean_square = ndimage.uniform_filter(array * array, size=size, mode="reflect")
    return np.sqrt(np.maximum(mean_square - mean * mean, 0.0))


def he_patch_features(
    image: ArrayLike, mask: ArrayLike | None = None
) -> dict[str, float]:
    """Compute interpretable H&E optical-density, texture, and edge features."""

    rgb = _channels_last(image, -1)
    if rgb.shape[-1] < 3:
        raise ValueError("H&E features require an H x W x 3(+) RGB image")
    rgb = rgb[..., :3]
    if mask is None:
        tissue = tissue_mask(rgb, modality="he", min_component_size=4, closing_radius=1)
    else:
        tissue = np.asarray(mask, dtype=bool)
        if tissue.shape != rgb.shape[:2]:
            raise ValueError("mask must match the image height and width")

    od = rgb_to_optical_density(rgb)
    od_total = np.sum(od, axis=-1)
    h_concentration, e_concentration = he_stain_concentrations(rgb)
    maximum = np.max(rgb, axis=-1)
    saturation = (maximum - np.min(rgb, axis=-1)) / np.maximum(maximum, 1e-6)
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    local_contrast = _local_standard_deviation(gray)
    gradient = _gradient_magnitude(gray)

    od_mean, od_std, od_p90, _ = _summary(od_total, tissue)
    h_mean, _, h_p90, _ = _summary(h_concentration, tissue)
    e_mean, _, e_p90, _ = _summary(e_concentration, tissue)
    saturation_mean, _, saturation_p90, _ = _summary(saturation, tissue)
    contrast_mean, _, contrast_p90, _ = _summary(local_contrast, tissue)
    gradient_mean, _, gradient_p90, _ = _summary(gradient, tissue)
    gradient_values = _masked_values(gradient, tissue)
    edge_threshold = max(float(np.percentile(gradient_values, 75.0)), 0.025)
    edge_fraction = float(np.mean(gradient_values > edge_threshold))
    od_values = _masked_values(od_total, tissue)
    dark_threshold = max(
        float(
            np.median(od_values)
            + 2.5 * np.median(np.abs(od_values - np.median(od_values)))
        ),
        1.25,
    )
    dark_fraction = float(np.mean(od_values > dark_threshold))

    values = (
        float(np.mean(tissue)),
        od_mean,
        od_std,
        od_p90,
        h_mean,
        h_p90,
        e_mean,
        e_p90,
        saturation_mean,
        saturation_p90,
        _entropy(gray, tissue),
        contrast_mean,
        contrast_p90,
        gradient_mean,
        gradient_p90,
        edge_fraction,
        dark_fraction,
    )
    return dict(zip(HE_FEATURE_NAMES, values, strict=True))


def fluorescence_patch_features(
    image: ArrayLike,
    mask: ArrayLike | None = None,
    *,
    channel_axis: int = -1,
    structural_channels: Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute panel-agnostic structural features for COMET/CosMx imagery.

    ``structural_channels`` should identify DAPI and broad morphology channels
    from experiment metadata.  Features summarize channels rather than encoding
    marker identities, keeping the vector length stable across panels.
    """

    channels = _channels_last(image, channel_axis)
    if structural_channels is not None:
        indices = np.asarray(tuple(structural_channels), dtype=np.int64)
        if indices.size == 0:
            raise ValueError("structural_channels cannot be empty")
        if np.any(indices < 0) or np.any(indices >= channels.shape[-1]):
            raise IndexError("A structural channel index is out of range")
        channels = channels[..., indices]
    if mask is None:
        tissue = tissue_mask(
            channels,
            modality="fluorescence",
            channel_axis=-1,
            min_component_size=4,
            closing_radius=1,
        )
    else:
        tissue = np.asarray(mask, dtype=bool)
        if tissue.shape != channels.shape[:2]:
            raise ValueError("mask must match the image height and width")

    scaled = np.stack(
        [
            _robust_unit_scale(channels[..., index], tissue)
            for index in range(channels.shape[-1])
        ],
        axis=-1,
    )
    aggregate = np.max(scaled, axis=-1)
    local_contrast = _local_standard_deviation(aggregate)
    gradient = _gradient_magnitude(aggregate)
    aggregate_mean, aggregate_std, aggregate_p90, aggregate_p99 = _summary(
        aggregate, tissue
    )
    contrast_mean, _, contrast_p90, _ = _summary(local_contrast, tissue)
    gradient_mean, _, gradient_p90, _ = _summary(gradient, tissue)
    gradient_values = _masked_values(gradient, tissue)
    edge_threshold = max(float(np.percentile(gradient_values, 75.0)), 0.03)

    channel_means: list[float] = []
    channel_ranges: list[float] = []
    channel_saturation: list[float] = []
    for index in range(channels.shape[-1]):
        values = _masked_values(scaled[..., index], tissue)
        channel_means.append(float(np.mean(values)))
        channel_ranges.append(
            float(np.percentile(values, 99.0) - np.percentile(values, 1.0))
        )
        raw_values = _masked_values(channels[..., index], tissue)
        channel_saturation.append(float(np.mean(raw_values >= 0.995)))

    correlations: list[float] = []
    if channels.shape[-1] > 1 and np.count_nonzero(tissue) >= 4:
        flattened = scaled[tissue]
        for first in range(flattened.shape[1]):
            for second in range(first + 1, flattened.shape[1]):
                if (
                    np.std(flattened[:, first]) > 1e-8
                    and np.std(flattened[:, second]) > 1e-8
                ):
                    correlations.append(
                        float(
                            np.corrcoef(flattened[:, first], flattened[:, second])[0, 1]
                        )
                    )

    aggregate_values = _masked_values(aggregate, tissue)
    values = (
        float(np.mean(tissue)),
        aggregate_mean,
        aggregate_std,
        aggregate_p90,
        aggregate_p99,
        float(np.mean(aggregate_values > 0.1)),
        contrast_mean,
        contrast_p90,
        gradient_mean,
        gradient_p90,
        float(np.mean(gradient_values > edge_threshold)),
        float(np.mean(channel_means)),
        float(np.std(channel_means)),
        float(np.mean(channel_ranges)),
        float(np.mean(channel_saturation)),
        float(np.median(correlations)) if correlations else 0.0,
    )
    return dict(zip(FLUORESCENCE_FEATURE_NAMES, values, strict=True))


def _pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        pair = (int(value), int(value))
    else:
        pair = tuple(int(item) for item in value)
        if len(pair) != 2:
            raise ValueError(f"{name} must be an int or a pair")
    if min(pair) <= 0:
        raise ValueError(f"{name} values must be positive")
    return pair


def _window_starts(length: int, window: int, stride: int) -> list[int]:
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    final = length - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def extract_patch_feature_table(
    image: ArrayLike,
    *,
    modality: str = "he",
    patch_size: int | Sequence[int] = 64,
    stride: int | Sequence[int] | None = None,
    tissue: ArrayLike | None = None,
    min_tissue_fraction: float = 0.05,
    channel_axis: int = -1,
    structural_channels: Sequence[int] | None = None,
) -> FeatureTable:
    """Extract deterministic overlapping patch features and native coordinates."""

    if not 0.0 <= min_tissue_fraction <= 1.0:
        raise ValueError("min_tissue_fraction must lie in [0, 1]")
    patch = _pair(patch_size, "patch_size")
    step = patch if stride is None else _pair(stride, "stride")
    channels = _channels_last(image, channel_axis)
    height, width = channels.shape[:2]
    mode = modality.lower().replace("-", "").replace("_", "")
    is_he = mode in {"he", "h&e", "brightfield", "brightfieldhe"}

    if tissue is None:
        support = tissue_mask(
            channels,
            modality="he" if is_he else "fluorescence",
            channel_axis=-1,
            min_component_size=max(4, min(64, height * width // 1000)),
            closing_radius=1,
        )
    else:
        support = np.asarray(tissue, dtype=bool)
        if support.shape != (height, width):
            raise ValueError("tissue must match the image height and width")

    rows: list[list[float]] = []
    coordinates: list[tuple[int, int, int, int]] = []
    names = HE_FEATURE_NAMES if is_he else FLUORESCENCE_FEATURE_NAMES
    for y0 in _window_starts(height, patch[0], step[0]):
        y1 = min(y0 + patch[0], height)
        for x0 in _window_starts(width, patch[1], step[1]):
            x1 = min(x0 + patch[1], width)
            patch_mask = support[y0:y1, x0:x1]
            if float(np.mean(patch_mask)) < min_tissue_fraction:
                continue
            patch_image = channels[y0:y1, x0:x1]
            if is_he:
                feature_map: Mapping[str, float] = he_patch_features(
                    patch_image, patch_mask
                )
            else:
                feature_map = fluorescence_patch_features(
                    patch_image,
                    patch_mask,
                    channel_axis=-1,
                    structural_channels=structural_channels,
                )
            rows.append([float(feature_map[name]) for name in names])
            coordinates.append((y0, x0, y1, x1))

    values = np.asarray(rows, dtype=np.float64)
    if not rows:
        values = np.empty((0, len(names)), dtype=np.float64)
    coordinate_array = np.asarray(coordinates, dtype=np.int64)
    if not coordinates:
        coordinate_array = np.empty((0, 4), dtype=np.int64)
    return FeatureTable(values, coordinate_array, names, patch, (height, width))


__all__ = [
    "FeatureTable",
    "FLUORESCENCE_FEATURE_NAMES",
    "HE_FEATURE_NAMES",
    "extract_patch_feature_table",
    "fluorescence_patch_features",
    "he_patch_features",
    "he_stain_concentrations",
    "normalize_image",
    "rgb_to_optical_density",
    "tissue_mask",
]
