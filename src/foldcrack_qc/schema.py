"""Canonical, dependency-light data contracts for fold/crack QC.

The project deliberately keeps these objects small and NumPy based.  They are
used at the boundary between modality-specific image loading, detectors, and
evaluation so that none of those components needs to know an instrument's
native array layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


class Modality(str, Enum):
    """Supported acquisition families."""

    HE = "he"
    COMET = "comet"
    COSMX = "cosmx"

    @classmethod
    def coerce(cls, value: "Modality | str") -> "Modality":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace(" ", "")
        aliases = {
            "h&e": cls.HE,
            "h_e": cls.HE,
            "h-e": cls.HE,
            "he": cls.HE,
            "brightfield": cls.HE,
            "comet": cls.COMET,
            "lunaphorecomet": cls.COMET,
            "cosmx": cls.COSMX,
            "cosmxsmi": cls.COSMX,
            "nanostringcosmx": cls.COSMX,
            "brukercosmx": cls.COSMX,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unsupported modality {value!r}; expected one of {choices}"
            ) from exc


class ChannelRole(str, Enum):
    """Semantic channel roles, independent of channel position or marker name."""

    BRIGHTFIELD_RED = "brightfield_red"
    BRIGHTFIELD_GREEN = "brightfield_green"
    BRIGHTFIELD_BLUE = "brightfield_blue"
    NUCLEAR = "nuclear"
    AUTOFLUORESCENCE = "autofluorescence"
    MEMBRANE = "membrane"
    CYTOPLASM = "cytoplasm"
    IMMUNE = "immune"
    MARKER = "marker"
    MORPHOLOGY = "morphology"
    UNKNOWN = "unknown"

    @classmethod
    def coerce(cls, value: "ChannelRole | str") -> "ChannelRole":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace(" ", "_")
        aliases = {
            "red": cls.BRIGHTFIELD_RED,
            "green": cls.BRIGHTFIELD_GREEN,
            "blue": cls.BRIGHTFIELD_BLUE,
            "dapi": cls.NUCLEAR,
            "hoechst": cls.NUCLEAR,
            "nucleus": cls.NUCLEAR,
            "af": cls.AUTOFLUORESCENCE,
            "autofluorescence_tritc": cls.AUTOFLUORESCENCE,
            "autofluorescence_cy5": cls.AUTOFLUORESCENCE,
            "broad_membrane_or_tissue": cls.MEMBRANE,
        }
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError:
            return cls.UNKNOWN


class ArtifactKind(str, Enum):
    """Reference and prediction labels used by the initial benchmark."""

    FOLD = "fold"
    CRACK = "crack"
    HARD_NEGATIVE = "hard_negative"

    @classmethod
    def coerce(cls, value: "ArtifactKind | str") -> "ArtifactKind":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "tear": cls.CRACK,
            "crack_tear": cls.CRACK,
            "crack/tear": cls.CRACK,
            "folding": cls.FOLD,
            "hardnegative": cls.HARD_NEGATIVE,
        }
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


ARTIFACT_KEYS: tuple[str, ...] = (
    ArtifactKind.FOLD.value,
    ArtifactKind.CRACK.value,
    ArtifactKind.HARD_NEGATIVE.value,
)


def _normalize_spacing(value: float | Sequence[float]) -> tuple[float, float]:
    if np.isscalar(value):
        spacing = (float(value), float(value))
    else:
        items = tuple(float(item) for item in value)
        if len(items) != 2:
            raise ValueError("pixel_size_um must be a scalar or (y_um, x_um)")
        spacing = items
    if not np.all(np.isfinite(spacing)) or min(spacing) <= 0:
        raise ValueError("pixel_size_um values must be finite and strictly positive")
    return spacing


def _as_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2 or tuple(array.shape) != shape:
        raise ValueError(f"Mask {name!r} has shape {array.shape}; expected {shape}")
    if not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.number)):
        raise TypeError(f"Mask {name!r} must be numeric or boolean")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"Mask {name!r} cannot be complex-valued")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"Mask {name!r} contains non-finite values")
    if array.dtype != np.bool_:
        allowed = {0.0, 1.0}
        if np.issubdtype(array.dtype, np.integer):
            allowed.add(255.0)
        unique = np.unique(array)
        if any(float(item) not in allowed for item in unique):
            raise ValueError(f"Mask {name!r} is not binary encoded")
    # Make a contiguous boolean copy.  Ground-truth and predicted binary masks
    # should not silently change if a caller later mutates its source array.
    return np.ascontiguousarray(array.astype(bool, copy=False)).copy()


@dataclass
class CanonicalImage:
    """A channels-last image with semantic channel metadata.

    Arrays are not intensity-normalized here.  Preserving native dynamic range
    is important for QC features such as saturation and optical density.
    """

    data: np.ndarray
    modality: Modality | str
    channel_names: Sequence[str] = field(default_factory=tuple)
    channel_roles: Sequence[ChannelRole | str] = field(default_factory=tuple)
    pixel_size_um: float | Sequence[float] = (1.0, 1.0)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_path: str | Path | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.data)
        if array.ndim == 2:
            array = array[..., np.newaxis]
        if array.ndim != 3:
            raise ValueError(
                f"CanonicalImage.data must be HxWxC; got shape {array.shape}"
            )
        if 0 in array.shape:
            raise ValueError("CanonicalImage.data cannot have an empty dimension")
        if not (np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_):
            raise TypeError(f"CanonicalImage.data must be numeric; got {array.dtype}")
        if np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError("Complex-valued images are not supported")
        if not np.all(np.isfinite(array)):
            raise ValueError(
                "CanonicalImage.data contains NaN or infinity; ingestion must "
                "abstain or explicitly encode acquisition dropout"
            )

        self.data = np.ascontiguousarray(array)
        self.modality = Modality.coerce(self.modality)
        self.pixel_size_um = _normalize_spacing(self.pixel_size_um)
        self.metadata = dict(self.metadata)
        self.source_path = None if self.source_path is None else str(self.source_path)

        n_channels = int(array.shape[-1])
        if self.channel_names:
            names = tuple(str(name).strip() for name in self.channel_names)
            if len(names) != n_channels:
                raise ValueError(
                    f"Received {len(names)} channel names for an image with {n_channels} channels"
                )
            if any(not name for name in names):
                raise ValueError("channel_names cannot contain empty strings")
        else:
            names = tuple(f"channel_{index}" for index in range(n_channels))

        if self.channel_roles:
            roles = tuple(ChannelRole.coerce(role) for role in self.channel_roles)
            if len(roles) != n_channels:
                raise ValueError(
                    f"Received {len(roles)} channel roles for an image with {n_channels} channels"
                )
        else:
            roles = (ChannelRole.UNKNOWN,) * n_channels
        self.channel_names = names
        self.channel_roles = roles

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[2])

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def indices_for_role(self, role: ChannelRole | str) -> tuple[int, ...]:
        target = ChannelRole.coerce(role)
        explicit_unknown = role is ChannelRole.UNKNOWN or (
            isinstance(role, str) and role.strip().lower() == ChannelRole.UNKNOWN.value
        )
        if target is ChannelRole.UNKNOWN and not explicit_unknown:
            return ()
        return tuple(
            index
            for index, candidate in enumerate(self.channel_roles)
            if candidate == target
        )

    def channel(
        self, name_or_role: str | ChannelRole, occurrence: int = 0
    ) -> np.ndarray:
        """Return one 2-D channel by exact name or semantic role."""

        if isinstance(name_or_role, str) and name_or_role in self.channel_names:
            indices = (self.channel_names.index(name_or_role),)
        else:
            indices = self.indices_for_role(name_or_role)
        try:
            return self.data[..., indices[occurrence]]
        except IndexError as exc:
            raise KeyError(
                f"No occurrence {occurrence} for channel or role {name_or_role!r}; "
                f"names={self.channel_names}, roles={[role.value for role in self.channel_roles]}"
            ) from exc

    def as_float32(self, scale_integer: bool = False) -> np.ndarray:
        """Return a float view/copy, optionally scaling integer types to [0, 1]."""

        array = self.data.astype(np.float32, copy=False)
        if scale_integer and np.issubdtype(self.data.dtype, np.integer):
            maximum = float(np.iinfo(self.data.dtype).max)
            if maximum > 0:
                array = array / maximum
        return array


@dataclass
class QCSample:
    """One image and optional reference masks used by a detector/evaluator."""

    sample_id: str
    image: CanonicalImage
    masks: Mapping[str | ArtifactKind, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id).strip()
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        if not isinstance(self.image, CanonicalImage):
            raise TypeError("image must be a CanonicalImage")

        normalized: dict[str, np.ndarray] = {}
        for raw_name, value in self.masks.items():
            if isinstance(raw_name, ArtifactKind):
                name = raw_name.value
            else:
                name = str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
                if name in {"tear", "crack_tear", "crack/tear"}:
                    name = ArtifactKind.CRACK.value
            if not name:
                raise ValueError("Mask names cannot be empty")
            normalized[name] = _as_mask(value, self.image.spatial_shape, name)
        self.masks = normalized
        self.metadata = dict(self.metadata)

    @property
    def modality(self) -> Modality:
        return self.image.modality

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.image.spatial_shape

    def mask(
        self, name: str | ArtifactKind, *, required: bool = False
    ) -> np.ndarray | None:
        key = (
            name.value
            if isinstance(name, ArtifactKind)
            else str(name).strip().lower().replace("-", "_").replace(" ", "_")
        )
        if key in {"tear", "crack_tear", "crack/tear"}:
            key = ArtifactKind.CRACK.value
        mask = self.masks.get(key)
        if mask is None and required:
            raise KeyError(f"Sample {self.sample_id!r} has no mask {key!r}")
        return mask

    @property
    def tissue_mask(self) -> np.ndarray:
        tissue = self.masks.get("tissue")
        if tissue is None:
            return np.ones(self.spatial_shape, dtype=bool)
        return tissue

    @property
    def reference_artifact_mask(self) -> np.ndarray:
        combined = np.zeros(self.spatial_shape, dtype=bool)
        for key in (ArtifactKind.FOLD.value, ArtifactKind.CRACK.value):
            if key in self.masks:
                combined |= self.masks[key]
        return combined


@dataclass
class QCResult:
    """Canonical detector output at image resolution.

    ``score_maps`` contain continuous evidence and ``masks`` contain thresholded
    predictions.  Both dictionaries may be empty, which is useful for a failed
    or abstained detector whose status is recorded in ``metadata``.
    """

    sample_id: str
    modality: Modality | str
    score_maps: Mapping[str | ArtifactKind, np.ndarray] = field(default_factory=dict)
    masks: Mapping[str | ArtifactKind, np.ndarray] = field(default_factory=dict)
    summary_scores: Mapping[str, float] = field(default_factory=dict)
    runtime_seconds: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = str(self.sample_id).strip()
        if not self.sample_id:
            raise ValueError("sample_id cannot be empty")
        self.modality = Modality.coerce(self.modality)

        score_maps: dict[str, np.ndarray] = {}
        inferred_shape: tuple[int, int] | None = None
        for raw_name, value in self.score_maps.items():
            name = (
                raw_name.value
                if isinstance(raw_name, ArtifactKind)
                else str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
            )
            array = np.asarray(value, dtype=np.float32)
            if array.ndim == 3 and array.shape[-1] == 1:
                array = array[..., 0]
            if array.ndim != 2:
                raise ValueError(f"Score map {name!r} must be 2-D; got {array.shape}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Score map {name!r} contains non-finite values")
            if inferred_shape is None:
                inferred_shape = tuple(array.shape)
            elif tuple(array.shape) != inferred_shape:
                raise ValueError("All QCResult maps must have the same spatial shape")
            score_maps[name] = np.ascontiguousarray(array)

        masks: dict[str, np.ndarray] = {}
        for raw_name, value in self.masks.items():
            name = (
                raw_name.value
                if isinstance(raw_name, ArtifactKind)
                else str(raw_name).strip().lower().replace("-", "_").replace(" ", "_")
            )
            if inferred_shape is None:
                candidate = np.asarray(value)
                if candidate.ndim == 3 and candidate.shape[-1] == 1:
                    candidate = candidate[..., 0]
                if candidate.ndim != 2:
                    raise ValueError(
                        f"Prediction mask {name!r} must be 2-D; got {candidate.shape}"
                    )
                inferred_shape = tuple(candidate.shape)
            masks[name] = _as_mask(value, inferred_shape, name)

        self.score_maps = score_maps
        self.masks = masks
        self.summary_scores = {
            str(key): float(value) for key, value in self.summary_scores.items()
        }
        if not all(np.isfinite(value) for value in self.summary_scores.values()):
            raise ValueError("summary_scores must contain only finite values")
        if self.runtime_seconds is not None:
            self.runtime_seconds = float(self.runtime_seconds)
            if not np.isfinite(self.runtime_seconds) or self.runtime_seconds < 0:
                raise ValueError("runtime_seconds must be finite and non-negative")
        self.metadata = dict(self.metadata)

    @property
    def spatial_shape(self) -> tuple[int, int] | None:
        values = tuple(self.score_maps.values()) + tuple(self.masks.values())
        return None if not values else tuple(values[0].shape)

    @property
    def artifact_mask(self) -> np.ndarray | None:
        shape = self.spatial_shape
        if shape is None:
            return None
        combined = np.zeros(shape, dtype=bool)
        found = False
        for key in (ArtifactKind.FOLD.value, ArtifactKind.CRACK.value):
            if key in self.masks:
                combined |= self.masks[key]
                found = True
        if not found and "artifact" in self.masks:
            return self.masks["artifact"].copy()
        return combined if found else None


# The longer alias reads naturally in specifications while QCSample remains
# concise in detector code.
CanonicalSample = QCSample


__all__ = [
    "ARTIFACT_KEYS",
    "ArtifactKind",
    "CanonicalImage",
    "CanonicalSample",
    "ChannelRole",
    "Modality",
    "QCSample",
    "QCResult",
]
