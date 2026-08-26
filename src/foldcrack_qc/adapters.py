"""Image readers and modality-aware channel adapters.

The adapters resolve *semantic* roles from names instead of relying on fixed
channel indices.  That matters for COMET and CosMx experiments, whose panels
and exported channel order can change between runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from .schema import CanonicalImage, ChannelRole, Modality, QCSample


_TIFF_SUFFIXES = {".tif", ".tiff"}


def _read_cv2(path: Path) -> np.ndarray:
    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise ValueError(f"OpenCV could not decode image {path}")
    # OpenCV returns color images in BGR(A), whereas our canonical convention
    # (and tifffile/Pillow conventions) is RGB(A).
    if array.ndim == 3 and array.shape[-1] == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    elif array.ndim == 3 and array.shape[-1] == 4:
        array = cv2.cvtColor(array, cv2.COLOR_BGRA2RGBA)
    return array


def read_image(path: str | Path, *, key: str | None = None) -> np.ndarray:
    """Read a NumPy or common raster image without intensity conversion.

    ``.npz`` archives must contain one array, an ``image``/``data``/``arr_0``
    entry, or an explicitly requested ``key``.  TIFF uses ``tifffile`` when it
    is installed and falls back to OpenCV for simple TIFFs.
    """

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    suffix = image_path.suffix.lower()

    if suffix == ".npy":
        if key is not None:
            raise ValueError("key is only valid for .npz input")
        array = np.load(image_path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(image_path, allow_pickle=False) as archive:
            available = tuple(archive.files)
            selected = key
            if selected is None:
                for preferred in ("image", "data", "arr_0"):
                    if preferred in archive.files:
                        selected = preferred
                        break
            if selected is None and len(available) == 1:
                selected = available[0]
            if selected is None:
                raise ValueError(
                    f"NPZ archive {image_path} contains multiple arrays {available}; provide key="
                )
            if selected not in archive.files:
                raise KeyError(
                    f"NPZ archive has no key {selected!r}; available keys are {available}"
                )
            array = np.asarray(archive[selected])
    elif suffix in _TIFF_SUFFIXES:
        try:
            import tifffile  # type: ignore[import-not-found]
        except ImportError:
            array = _read_cv2(image_path)
        else:
            array = np.asarray(tifffile.imread(image_path))
    else:
        array = _read_cv2(image_path)

    if not isinstance(array, np.ndarray) or array.ndim < 2:
        raise ValueError(
            f"Decoded input must be an image-like array; got shape {getattr(array, 'shape', None)}"
        )
    return array


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _to_hwc(
    data: np.ndarray,
    *,
    channel_axis: int | None,
    channel_names: Sequence[str] | None,
) -> tuple[np.ndarray, int | None]:
    array = np.asarray(data)
    if array.ndim == 2:
        return np.ascontiguousarray(array[..., np.newaxis]), None
    if array.ndim != 3:
        raise ValueError(
            f"Expected a 2-D or 3-D image, got shape {array.shape}. "
            "Select an OME series/z-projection before adapting higher-dimensional data."
        )

    resolved_axis = channel_axis
    if resolved_axis is not None:
        resolved_axis = int(resolved_axis) % 3
    elif channel_names:
        matching = [
            axis
            for axis, length in enumerate(array.shape)
            if length == len(channel_names)
        ]
        if len(matching) == 1:
            resolved_axis = matching[0]
        elif 2 in matching:
            resolved_axis = 2
        elif matching:
            resolved_axis = matching[0]
        else:
            raise ValueError(
                f"No axis of shape {array.shape} matches {len(channel_names)} supplied channel names"
            )
    else:
        # Fluorescence stacks are commonly CxHxW; ordinary images are HxWxC.
        # The 64-channel bound covers typical multiplex panels while avoiding
        # treating a small spatial dimension as channels in most crops.
        first_looks_like_channels = array.shape[0] <= 64 and array.shape[0] < min(
            array.shape[1:]
        )
        last_looks_like_channels = array.shape[2] <= 64 and array.shape[2] < min(
            array.shape[:2]
        )
        resolved_axis = (
            0 if first_looks_like_channels and not last_looks_like_channels else 2
        )

    if resolved_axis != 2:
        array = np.moveaxis(array, resolved_axis, 2)
    return np.ascontiguousarray(array), resolved_axis


def _metadata_with_roles(
    metadata: Mapping[str, Any] | None,
    roles: Sequence[ChannelRole],
    original_shape: Sequence[int],
    original_channel_axis: int | None,
) -> dict[str, Any]:
    result = dict(metadata or {})
    role_indices: dict[str, list[int]] = {}
    for index, role in enumerate(roles):
        role_indices.setdefault(role.value, []).append(index)
    structural_roles = {
        ChannelRole.BRIGHTFIELD_RED,
        ChannelRole.BRIGHTFIELD_GREEN,
        ChannelRole.BRIGHTFIELD_BLUE,
        ChannelRole.NUCLEAR,
        ChannelRole.AUTOFLUORESCENCE,
        ChannelRole.MEMBRANE,
        ChannelRole.CYTOPLASM,
        ChannelRole.IMMUNE,
        ChannelRole.MORPHOLOGY,
    }
    structural = [index for index, role in enumerate(roles) if role in structural_roles]
    # A marker channel still contains useful morphology when no explicitly
    # structural channel was identifiable.
    if not structural:
        structural = list(range(len(roles)))
    result.update(
        {
            "input_shape": tuple(int(item) for item in original_shape),
            "input_channel_axis": original_channel_axis,
            "semantic_role_indices": role_indices,
            "structural_channel_indices": structural,
        }
    )
    return result


class ModalityAdapter(ABC):
    """Base interface for converting native arrays to :class:`CanonicalImage`."""

    modality: Modality

    @abstractmethod
    def adapt(
        self,
        data: np.ndarray,
        *,
        channel_names: Sequence[str] | None = None,
        pixel_size_um: float | Sequence[float] = (1.0, 1.0),
        metadata: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
        channel_axis: int | None = None,
        color_order: str = "rgb",
    ) -> CanonicalImage:
        raise NotImplementedError


class HEAdapter(ModalityAdapter):
    """Adapter for raw H&E RGB without stain-normalizing away QC evidence."""

    modality = Modality.HE

    def adapt(
        self,
        data: np.ndarray,
        *,
        channel_names: Sequence[str] | None = None,
        pixel_size_um: float | Sequence[float] = (1.0, 1.0),
        metadata: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
        channel_axis: int | None = None,
        color_order: str = "rgb",
    ) -> CanonicalImage:
        original = np.asarray(data)
        array, resolved_axis = _to_hwc(
            original, channel_axis=channel_axis, channel_names=channel_names
        )
        order = color_order.strip().lower()
        if order not in {
            "rgb",
            "rgba",
            "bgr",
            "bgra",
            "gray",
            "greyscale",
            "grayscale",
        }:
            raise ValueError(f"Unsupported H&E color_order {color_order!r}")

        names = list(channel_names) if channel_names else []
        if array.shape[-1] == 4:
            array = array[..., :3]
            if names:
                names = names[:3]
        if array.shape[-1] == 3:
            if order in {"bgr", "bgra"}:
                array = array[..., ::-1]
                if names:
                    names = names[::-1]
            names = names or ["red", "green", "blue"]
            roles = [
                ChannelRole.BRIGHTFIELD_RED,
                ChannelRole.BRIGHTFIELD_GREEN,
                ChannelRole.BRIGHTFIELD_BLUE,
            ]
        elif array.shape[-1] == 1:
            names = names or ["grayscale"]
            roles = [ChannelRole.MORPHOLOGY]
        else:
            raise ValueError(
                f"H&E input must contain 1, 3, or 4 channels; got {array.shape[-1]}"
            )

        adapted_metadata = _metadata_with_roles(
            metadata, roles, original.shape, resolved_axis
        )
        adapted_metadata["color_order"] = "rgb" if array.shape[-1] == 3 else "grayscale"
        adapted_metadata["raw_intensity_preserved"] = True
        return CanonicalImage(
            data=array,
            modality=self.modality,
            channel_names=names,
            channel_roles=roles,
            pixel_size_um=pixel_size_um,
            metadata=adapted_metadata,
            source_path=source_path,
        )


class COMETAdapter(ModalityAdapter):
    """Resolve COMET DAPI/fluorophore channels by OME-style names."""

    modality = Modality.COMET

    @staticmethod
    def _default_names(n_channels: int) -> list[str]:
        if n_channels == 1:
            return ["DAPI"]
        if n_channels == 5:
            return ["DAPI", "FITC", "TRITC", "Cy5", "Cy7"]
        return [f"channel_{index}" for index in range(n_channels)]

    @staticmethod
    def _role(name: str) -> ChannelRole:
        compact = _normalized_name(name)
        if any(term in compact for term in ("dapi", "hoechst", "nuclear", "nucleus")):
            return ChannelRole.NUCLEAR
        if compact.startswith("af") or any(
            term in compact for term in ("autofluorescence", "autofluor", "background")
        ):
            return ChannelRole.AUTOFLUORESCENCE
        if any(term in compact for term in ("membrane", "wga", "cd298", "b2m")):
            return ChannelRole.MEMBRANE
        if any(
            term in compact
            for term in ("pankeratin", "pancytokeratin", "panck", "cytoplasm")
        ):
            return ChannelRole.CYTOPLASM
        if any(
            term in compact for term in ("fitc", "tritc", "cy3", "cy5", "cy7", "alexa")
        ):
            return ChannelRole.MARKER
        return ChannelRole.MARKER

    def adapt(
        self,
        data: np.ndarray,
        *,
        channel_names: Sequence[str] | None = None,
        pixel_size_um: float | Sequence[float] = (1.0, 1.0),
        metadata: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
        channel_axis: int | None = None,
        color_order: str = "rgb",
    ) -> CanonicalImage:
        del color_order
        original = np.asarray(data)
        array, resolved_axis = _to_hwc(
            original, channel_axis=channel_axis, channel_names=channel_names
        )
        names = (
            list(channel_names)
            if channel_names
            else self._default_names(array.shape[-1])
        )
        roles = [self._role(name) for name in names]
        adapted_metadata = _metadata_with_roles(
            metadata, roles, original.shape, resolved_axis
        )
        adapted_metadata["channel_resolution"] = "semantic_name"
        return CanonicalImage(
            data=array,
            modality=self.modality,
            channel_names=names,
            channel_roles=roles,
            pixel_size_um=pixel_size_um,
            metadata=adapted_metadata,
            source_path=source_path,
        )


class CosMxAdapter(ModalityAdapter):
    """Resolve CosMx morphology channels without hard-coded panel indices."""

    modality = Modality.COSMX

    @staticmethod
    def _default_names(n_channels: int) -> list[str]:
        if n_channels == 1:
            return ["DAPI"]
        if n_channels == 4:
            return ["DAPI", "PanCK", "CD45", "CD298_B2M"]
        if n_channels == 5:
            return ["DAPI", "PanCK", "CD45", "CD298", "B2M"]
        return [f"morphology_{index}" for index in range(n_channels)]

    @staticmethod
    def _role(name: str) -> ChannelRole:
        compact = _normalized_name(name)
        if any(term in compact for term in ("dapi", "hoechst", "nuclear", "nucleus")):
            return ChannelRole.NUCLEAR
        if any(
            term in compact for term in ("cd298", "b2m", "membrane", "wga", "epcam")
        ):
            return ChannelRole.MEMBRANE
        if any(
            term in compact
            for term in ("panck", "pankeratin", "pancytokeratin", "cytokeratin")
        ):
            return ChannelRole.CYTOPLASM
        if any(term in compact for term in ("cd45", "immune", "leukocyte")):
            return ChannelRole.IMMUNE
        if compact.startswith("af") or "autofluor" in compact:
            return ChannelRole.AUTOFLUORESCENCE
        return ChannelRole.MORPHOLOGY

    def adapt(
        self,
        data: np.ndarray,
        *,
        channel_names: Sequence[str] | None = None,
        pixel_size_um: float | Sequence[float] = (1.0, 1.0),
        metadata: Mapping[str, Any] | None = None,
        source_path: str | Path | None = None,
        channel_axis: int | None = None,
        color_order: str = "rgb",
    ) -> CanonicalImage:
        del color_order
        original = np.asarray(data)
        array, resolved_axis = _to_hwc(
            original, channel_axis=channel_axis, channel_names=channel_names
        )
        names = (
            list(channel_names)
            if channel_names
            else self._default_names(array.shape[-1])
        )
        roles = [self._role(name) for name in names]
        adapted_metadata = _metadata_with_roles(
            metadata, roles, original.shape, resolved_axis
        )
        adapted_metadata["channel_resolution"] = "semantic_name"
        return CanonicalImage(
            data=array,
            modality=self.modality,
            channel_names=names,
            channel_roles=roles,
            pixel_size_um=pixel_size_um,
            metadata=adapted_metadata,
            source_path=source_path,
        )


_ADAPTERS: dict[Modality, ModalityAdapter] = {
    Modality.HE: HEAdapter(),
    Modality.COMET: COMETAdapter(),
    Modality.COSMX: CosMxAdapter(),
}


def get_adapter(modality: Modality | str) -> ModalityAdapter:
    """Return the stateless adapter for a modality or accepted alias."""

    return _ADAPTERS[Modality.coerce(modality)]


def adapt_image(
    data: np.ndarray,
    modality: Modality | str,
    *,
    channel_names: Sequence[str] | None = None,
    pixel_size_um: float | Sequence[float] = (1.0, 1.0),
    metadata: Mapping[str, Any] | None = None,
    source_path: str | Path | None = None,
    channel_axis: int | None = None,
    color_order: str = "rgb",
) -> CanonicalImage:
    """Adapt an in-memory native array to the canonical channels-last schema."""

    return get_adapter(modality).adapt(
        data,
        channel_names=channel_names,
        pixel_size_um=pixel_size_um,
        metadata=metadata,
        source_path=source_path,
        channel_axis=channel_axis,
        color_order=color_order,
    )


def load_sample(
    path: str | Path,
    modality: Modality | str,
    *,
    sample_id: str | None = None,
    key: str | None = None,
    channel_names: Sequence[str] | None = None,
    pixel_size_um: float | Sequence[float] = (1.0, 1.0),
    metadata: Mapping[str, Any] | None = None,
    channel_axis: int | None = None,
    color_order: str = "rgb",
) -> QCSample:
    """Read and adapt a file into an unlabeled :class:`QCSample`."""

    image_path = Path(path).expanduser()
    data = read_image(image_path, key=key)
    image = adapt_image(
        data,
        modality,
        channel_names=channel_names,
        pixel_size_um=pixel_size_um,
        metadata=metadata,
        source_path=image_path,
        channel_axis=channel_axis,
        color_order=color_order,
    )
    return QCSample(sample_id=sample_id or image_path.stem, image=image)


__all__ = [
    "COMETAdapter",
    "CosMxAdapter",
    "HEAdapter",
    "ModalityAdapter",
    "adapt_image",
    "get_adapter",
    "load_sample",
    "read_image",
]
