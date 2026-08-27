"""Label-free proxy benchmark for real COMET and CosMx backgrounds.

The benchmark measures recovery of controlled fold/crack *spike-ins* and
metamorphic repeatability.  It deliberately does not treat detector activity
on unannotated images as a false positive and cannot establish performance on
real artifacts.  All learned thresholds are locked on calibration groups
before held-out test groups are scored.

Images use an explicit ``C x Y x X`` in-memory contract. Large TIFFs are
downsampled one plane at a time; memory-mappable planes use bounded-memory exact
area integration so every native pixel contributes without a full-plane copy.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage

from .detectors import (
    CleanReferenceAnomalyDetector,
    classical_candidate_masks,
    tile_scores_to_map,
)
from .features import FeatureTable, extract_patch_feature_table

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
ImageArray = NDArray[np.generic]

COMET_DATASET_NAME = "QUALIFAI public Lunaphore/COMET DAPI recovery"
COMET_SOURCE_URL = "https://zenodo.org/records/12699470"
COSMX_GASTRIC_DATASET_NAME = "CosMx gastric mucosa molecular-imaging data"
COSMX_GASTRIC_SOURCE_URL = "https://zenodo.org/records/8333281"
COSMX_PHGG_DATASET_NAME = (
    "CosMx pediatric high-grade glioma spatial molecular-imaging data"
)
COSMX_PHGG_SOURCE_URL = "https://zenodo.org/records/16877090"


@dataclass(frozen=True)
class MultiplexField:
    """One downsampled, provenance-bearing multiplex image field."""

    source_id: str
    group_id: str
    cohort_id: str
    modality: str
    image: ImageArray
    channel_names: tuple[str, ...]
    source_path: str
    sha256: str
    dataset_name: str
    source_url: str
    source_axes: str
    native_shape: tuple[int, ...]
    native_pixel_size_um: float | None = None
    effective_pixel_size_um: tuple[float, float] | None = None
    pixel_size_source: str = "not_available_in_source"
    downsample_method: str = "none"
    group_level: str = "unspecified_source_group"
    group_independence_declared: bool = False
    group_independence_basis: str = "not_declared"
    lock_manifest_path: str | None = None
    lock_verified: bool = False

    def __post_init__(self) -> None:
        image = np.asarray(self.image).view()
        if image.ndim != 3 or min(image.shape) <= 0:
            raise ValueError("MultiplexField.image must have shape C x Y x X")
        if len(self.channel_names) != image.shape[0]:
            raise ValueError("channel_names must contain one name per channel")
        if not all((self.source_id, self.group_id, self.modality, self.sha256)):
            raise ValueError("source, group, modality, and checksum must be non-empty")
        if not self.group_level or not self.group_independence_basis:
            raise ValueError("group-level provenance must be non-empty")
        if self.modality not in {"comet", "cosmx"}:
            raise ValueError("modality must be 'comet' or 'cosmx'")
        if np.issubdtype(image.dtype, np.floating) and not np.isfinite(image).all():
            raise ValueError("image must not contain NaN or infinity")
        image.setflags(write=False)
        object.__setattr__(self, "image", image)

    def source_record(self, role: str) -> dict[str, Any]:
        """Return JSON-ready source and split provenance."""

        return {
            "source_id": self.source_id,
            "group_id": self.group_id,
            "cohort_id": self.cohort_id,
            "role": role,
            "modality": self.modality,
            "dataset_name": self.dataset_name,
            "source_url": self.source_url,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "source_axes": self.source_axes,
            "native_shape": list(self.native_shape),
            "loaded_shape": list(self.image.shape),
            "loaded_channel_axis": 0,
            "channel_names": list(self.channel_names),
            "dtype": str(self.image.dtype),
            "native_pixel_size_um": self.native_pixel_size_um,
            "effective_pixel_size_um_yx": (
                None
                if self.effective_pixel_size_um is None
                else list(self.effective_pixel_size_um)
            ),
            "pixel_size_source": self.pixel_size_source,
            "downsample_method": self.downsample_method,
            "group_level": self.group_level,
            "group_independence_declared": self.group_independence_declared,
            "group_independence_basis": self.group_independence_basis,
            "lock_manifest_path": self.lock_manifest_path,
            "lock_verified": self.lock_verified,
        }


@dataclass(frozen=True)
class InjectedField:
    """A non-destructive injection with intended and realized pixel support."""

    image: ImageArray
    intended_mask: BoolArray
    effective_changed_mask: BoolArray
    artifact: str
    severity: float
    seed: int
    injection_parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        intended = np.asarray(self.intended_mask, dtype=bool).view()
        effective = np.asarray(self.effective_changed_mask, dtype=bool).view()
        if intended.shape != self.image.shape[1:] or effective.shape != intended.shape:
            raise ValueError("injection masks must match the image spatial shape")
        if np.any(effective & ~intended):
            raise ValueError(
                "effective changed support must be inside intended geometry"
            )
        if not np.any(effective):
            raise ValueError("artifact injection produced no realized pixel changes")
        intended.setflags(write=False)
        effective.setflags(write=False)
        object.__setattr__(self, "intended_mask", intended)
        object.__setattr__(self, "effective_changed_mask", effective)

    @property
    def mask(self) -> BoolArray:
        """Return realized changed support for backward-compatible callers."""

        return self.effective_changed_mask


@dataclass(frozen=True)
class MultiplexProxyConfig:
    """Configuration for a deterministic real-background proxy experiment."""

    seed: int = 29
    severities: tuple[float, ...] = (0.35, 0.65, 1.0)
    patch_size: int = 64
    stride: int = 32
    base_alert_quantile: float = 0.99
    response_threshold_candidates: int = 128
    lesion_dilation_pixels: int = 0
    classical_weight: float = 0.55
    anomaly_weight: float = 0.45
    group_bootstrap_resamples: int = 2_000
    group_bootstrap_seed: int = 20_260_826

    def __post_init__(self) -> None:
        if not self.severities or any(
            not math.isfinite(value) or not 0.0 < value <= 1.0
            for value in self.severities
        ):
            raise ValueError("severities must be finite values in (0, 1]")
        if tuple(sorted(set(self.severities))) != self.severities:
            raise ValueError("severities must be unique and increasing")
        if self.patch_size < 8 or self.stride < 1:
            raise ValueError("patch_size and stride must be positive and practical")
        if not 0.5 < self.base_alert_quantile < 1.0:
            raise ValueError("base_alert_quantile must lie in (0.5, 1)")
        if self.response_threshold_candidates < 8:
            raise ValueError("response_threshold_candidates must be at least 8")
        if self.lesion_dilation_pixels < 0:
            raise ValueError("lesion_dilation_pixels cannot be negative")
        if self.group_bootstrap_resamples <= 0:
            raise ValueError("group_bootstrap_resamples must be positive")
        if self.group_bootstrap_seed < 0:
            raise ValueError("group_bootstrap_seed cannot be negative")
        weights = (self.classical_weight, self.anomaly_weight)
        if min(weights) < 0.0 or sum(weights) <= 0.0:
            raise ValueError("fusion weights must be non-negative with positive sum")


def _sha256(path: Path, chunk_bytes: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _target_shape(shape: tuple[int, int], max_dimension: int) -> tuple[int, int]:
    if max_dimension < 32:
        raise ValueError("max_dimension must be at least 32")
    height, width = shape
    scale = min(1.0, max_dimension / float(max(height, width)))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _resize_plane(
    plane: ArrayLike,
    target: tuple[int, int],
    *,
    memory_mapped: bool,
) -> ImageArray:
    source = np.asarray(plane)
    if source.ndim != 2:
        raise ValueError("TIFF planes must be two-dimensional")
    if source.shape == target:
        return np.array(source, copy=True)
    if min(target) <= 0:
        raise ValueError("target dimensions must be positive")
    if target[0] > source.shape[0] or target[1] > source.shape[1]:
        raise ValueError("area-resize target cannot exceed the source dimensions")
    if not memory_mapped:
        resized = cv2.resize(
            source,
            (target[1], target[0]),
            interpolation=cv2.INTER_AREA,
        )
        return np.asarray(resized, dtype=source.dtype)

    # Exact area integration over a piecewise-constant source plane. Each
    # destination row reads only its contributing source rows, and horizontal
    # interval integrals are vectorized. Unlike point subsampling, every source
    # pixel contributes with its overlap weight, preserving thin signals while
    # keeping peak working memory bounded by roughly one source-row block.
    source_height, source_width = source.shape
    target_height, target_width = target
    scale_y = source_height / float(target_height)
    scale_x = source_width / float(target_width)
    x_edges = np.arange(target_width + 1, dtype=np.float64) * scale_x
    x_edges = np.clip(x_edges, 0.0, float(source_width))
    x_edges[-1] = float(source_width)
    y_edges = np.arange(target_height + 1, dtype=np.float64) * scale_y
    y_edges = np.clip(y_edges, 0.0, float(source_height))
    y_edges[-1] = float(source_height)
    resized_float = np.empty(target, dtype=np.float64)
    for target_y in range(target_height):
        top = float(y_edges[target_y])
        bottom = float(y_edges[target_y + 1])
        first_row = math.floor(top)
        stop_row = min(source_height, math.ceil(bottom))
        row_indices = np.arange(first_row, stop_row, dtype=np.int64)
        row_weights = np.minimum(bottom, row_indices + 1.0) - np.maximum(
            top, row_indices.astype(np.float64)
        )
        source_block = np.asarray(source[first_row:stop_row, :], dtype=np.float64)
        vertical_average = (row_weights @ source_block) / scale_y

        prefix = np.empty(source_width + 1, dtype=np.float64)
        prefix[0] = 0.0
        np.cumsum(vertical_average, dtype=np.float64, out=prefix[1:])
        whole = np.floor(x_edges).astype(np.int64)
        clipped = np.minimum(whole, source_width)
        integrals = prefix[clipped]
        fractional = x_edges - whole
        partial = whole < source_width
        integrals[partial] += fractional[partial] * vertical_average[whole[partial]]
        resized_float[target_y] = np.diff(integrals) / scale_x

    if np.issubdtype(source.dtype, np.integer):
        bounds = np.iinfo(source.dtype)
        resized_float = np.rint(np.clip(resized_float, bounds.min, bounds.max))
    return np.asarray(resized_float, dtype=source.dtype)


def _description_json(description: str | None) -> dict[str, Any]:
    if not description:
        return {}
    try:
        parsed = json.loads(description)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_tiff_cyx(path: Path, max_dimension: int) -> tuple[ImageArray, dict[str, Any]]:
    try:
        import tifffile
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "TIFF loading requires the optional 'wsi' dependencies"
        ) from error

    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        native_shape = tuple(int(value) for value in series.shape)
        axes = str(series.axes).upper()
        description = tif.pages[0].description
        metadata = _description_json(description)
        if len(native_shape) == 2 and axes.endswith("YX"):
            channel_count = 1
            height, width = native_shape
            channel_axis = None
        elif len(native_shape) == 3 and axes.endswith("YX"):
            channel_axis = 0
            channel_count, height, width = native_shape
        else:
            raise ValueError(
                f"Unsupported TIFF axes/shape {axes!r}/{native_shape}; expected YX or CYX/IYX"
            )
        target = _target_shape((height, width), max_dimension)

        try:
            mapped = tifffile.memmap(path, series=0)
        except ValueError:
            mapped = None
        planes: list[ImageArray] = []
        if mapped is not None:
            source_planes = [mapped] if channel_axis is None else list(mapped)
            for plane in source_planes:
                planes.append(_resize_plane(plane, target, memory_mapped=True))
        else:
            if channel_axis is None:
                planes.append(
                    _resize_plane(tif.pages[0].asarray(), target, memory_mapped=False)
                )
            elif len(tif.pages) == channel_count:
                for page_index, page in enumerate(tif.pages):
                    try:
                        decoded = page.asarray()
                    except ValueError as error:
                        # The lean environment intentionally does not require
                        # imagecodecs. OpenCV's libtiff can decode the public
                        # LZW CosMx pages one at a time without a C-stack copy.
                        if "imagecodecs" not in str(error):
                            raise
                        success, decoded_pages = cv2.imreadmulti(
                            str(path),
                            page_index,
                            1,
                            flags=cv2.IMREAD_UNCHANGED,
                        )
                        if not success or len(decoded_pages) != 1:
                            raise RuntimeError(
                                f"Unable to decode compressed TIFF page {page_index}"
                            ) from error
                        decoded = decoded_pages[0]
                    planes.append(_resize_plane(decoded, target, memory_mapped=False))
                    del decoded
            else:
                # Unusual contiguous compressed series: unavoidable fallback,
                # still released before the returned downsampled stack is used.
                full = series.asarray()
                for plane in full:
                    planes.append(_resize_plane(plane, target, memory_mapped=False))
                del full

    stack = np.stack(planes, axis=0)
    scale_y = height / float(target[0])
    scale_x = width / float(target[1])
    return stack, {
        "native_shape": native_shape,
        "source_axes": axes,
        "description_metadata": metadata,
        "scale_yx": (scale_y, scale_x),
        "downsample_method": (
            "none" if (height, width) == target else "all_source_pixels_area_resample"
        ),
    }


def load_comet_dapi_tiff(
    path: str | Path,
    *,
    max_dimension: int = 896,
    native_pixel_size_um: float | None = None,
) -> MultiplexField:
    """Load one real COMET DAPI TIFF with explicit channel/provenance metadata."""

    source = Path(path).expanduser().resolve()
    image, metadata = _read_tiff_cyx(source, max_dimension)
    if image.shape[0] != 1:
        raise ValueError("COMET DAPI input must contain exactly one channel")
    scales = metadata["scale_yx"]
    effective = (
        None
        if native_pixel_size_um is None
        else (
            float(native_pixel_size_um) * scales[0],
            float(native_pixel_size_um) * scales[1],
        )
    )
    return MultiplexField(
        source_id=source.stem,
        group_id=source.stem,
        cohort_id=source.stem,
        modality="comet",
        image=image,
        channel_names=("DAPI",),
        source_path=str(source),
        sha256=_sha256(source),
        dataset_name=COMET_DATASET_NAME,
        source_url=COMET_SOURCE_URL,
        source_axes=metadata["source_axes"],
        native_shape=metadata["native_shape"],
        native_pixel_size_um=native_pixel_size_um,
        effective_pixel_size_um=effective,
        pixel_size_source=(
            "caller_override" if native_pixel_size_um is not None else "not_in_tiff"
        ),
        downsample_method=metadata["downsample_method"],
        group_level="public_field_id",
        group_independence_declared=False,
        group_independence_basis=(
            "QUALIFAI release does not declare higher-level slide/patient mapping"
        ),
    )


def _cosmx_channel_names(metadata: Mapping[str, Any], count: int) -> tuple[str, ...]:
    order = str(metadata.get("ChannelOrder", ""))
    uids = list(order) if len(order) == count else []
    if not uids:
        channels = metadata.get("Channels", [])
        if isinstance(channels, list):
            uids = [str(item.get("UID", index)) for index, item in enumerate(channels)]
    if len(uids) != count:
        uids = [f"channel_{index}" for index in range(count)]
    biological: dict[str, str] = {}
    kit = metadata.get("MorphologyKit", {})
    reagents = kit.get("MorphologyReagents", []) if isinstance(kit, dict) else []
    if isinstance(reagents, list):
        for reagent in reagents:
            if not isinstance(reagent, dict):
                continue
            fluorophore = reagent.get("Fluorophore", {})
            if isinstance(fluorophore, dict) and fluorophore.get("ChannelId"):
                biological[str(fluorophore["ChannelId"])] = str(
                    reagent.get("BiologicalTarget", fluorophore["ChannelId"])
                )
    return tuple(f"{biological.get(uid, uid)}[{uid}]" for uid in uids)


def load_cosmx_morphology_tiff(
    path: str | Path,
    *,
    max_dimension: int = 896,
    channel_names: Sequence[str] | None = None,
    native_pixel_size_um: float | None = None,
    group_id: str | None = None,
    cohort_id: str | None = None,
    dataset_name: str = COSMX_GASTRIC_DATASET_NAME,
    source_url: str = COSMX_GASTRIC_SOURCE_URL,
    group_level: str = "caller_or_filename_inferred_slide_or_run",
    group_independence_declared: bool = False,
    group_independence_basis: str = "not_declared_by_loader_caller",
) -> MultiplexField:
    """Load one CosMx raw morphology FOV, preserving its explicit channel axis."""

    source = Path(path).expanduser().resolve()
    image, details = _read_tiff_cyx(source, max_dimension)
    metadata = details["description_metadata"]
    names = (
        tuple(str(name) for name in channel_names)
        if channel_names is not None
        else _cosmx_channel_names(metadata, image.shape[0])
    )
    if len(names) != image.shape[0]:
        raise ValueError("channel_names must contain one name per CosMx channel")
    inferred_mpp = native_pixel_size_um
    mpp_source = "caller_override"
    if inferred_mpp is None and metadata.get("ImPixelSize_nm") is not None:
        inferred_mpp = float(metadata["ImPixelSize_nm"]) / 1000.0
        mpp_source = "TIFF.ImageDescription.ImPixelSize_nm"
    if inferred_mpp is None:
        mpp_source = "not_available_in_source"
    scales = details["scale_yx"]
    effective = (
        None
        if inferred_mpp is None
        else (inferred_mpp * scales[0], inferred_mpp * scales[1])
    )
    slide_match = re.search(r"_(S\d+)_", source.name, flags=re.IGNORECASE)
    inferred_cohort = slide_match.group(1).upper() if slide_match else source.stem
    resolved_cohort = str(cohort_id or inferred_cohort)
    resolved_group = str(group_id or resolved_cohort)
    return MultiplexField(
        source_id=source.stem,
        group_id=resolved_group,
        cohort_id=resolved_cohort,
        modality="cosmx",
        image=image,
        channel_names=names,
        source_path=str(source),
        sha256=_sha256(source),
        dataset_name=str(dataset_name),
        source_url=str(source_url),
        source_axes=details["source_axes"],
        native_shape=details["native_shape"],
        native_pixel_size_um=inferred_mpp,
        effective_pixel_size_um=effective,
        pixel_size_source=mpp_source,
        downsample_method=details["downsample_method"],
        group_level=str(group_level),
        group_independence_declared=bool(group_independence_declared),
        group_independence_basis=str(group_independence_basis),
    )


def _cosmx_source_identity(path: Path) -> dict[str, Any]:
    """Resolve the two audited public CosMx cohorts without slide-ID collisions."""

    ancestry = {item.name.casefold() for item in (path, *path.parents)}
    slide_match = re.search(r"_(S\d+)_", path.name, flags=re.IGNORECASE)
    slide = slide_match.group(1).upper() if slide_match else path.stem
    if "cosmx_phgg_v1" in ancestry:
        run = path.name.split("__", 1)[0] if "__" in path.name else slide
        return {
            "group_id": f"zenodo16877090:{run}",
            "cohort_id": f"zenodo16877090:{run}",
            "dataset_name": COSMX_PHGG_DATASET_NAME,
            "source_url": COSMX_PHGG_SOURCE_URL,
            "group_level": "slide_or_run_directory",
            "group_independence_declared": False,
            "group_independence_basis": (
                "distinct source slide/run directories are locked, but higher-level "
                "biological independence is not declared by the public release"
            ),
        }
    if "cosmx_gastric_v1" in ancestry:
        return {
            "group_id": f"zenodo8333281:{slide}",
            "cohort_id": f"zenodo8333281:{slide}",
            "dataset_name": COSMX_GASTRIC_DATASET_NAME,
            "source_url": COSMX_GASTRIC_SOURCE_URL,
            "group_level": "source_slide_or_run",
            "group_independence_declared": False,
            "group_independence_basis": (
                "distinct source slide/run identifiers are locked, but higher-level "
                "biological independence is not declared by the public release"
            ),
        }
    raise ValueError(
        "CosMx public provenance is ambiguous; place inputs below "
        "cosmx_gastric_v1/ or cosmx_phgg_v1/, or call "
        "load_cosmx_morphology_tiff with explicit source/group metadata"
    )


def _discover_tiffs(value: str | Path | Sequence[str | Path]) -> list[Path]:
    roots = (
        [Path(value)]
        if isinstance(value, (str, Path))
        else [Path(item) for item in value]
    )
    discovered: set[Path] = set()
    for root in roots:
        if root.is_file() and root.suffix.casefold() in {".tif", ".tiff"}:
            discovered.add(root.resolve())
        elif root.is_dir():
            discovered.update(
                item.resolve()
                for item in root.rglob("*")
                if item.is_file() and item.suffix.casefold() in {".tif", ".tiff"}
            )
    return sorted(discovered)


def _verify_public_field_locks(
    fields: Sequence[MultiplexField], manifest_dir: Path
) -> list[MultiplexField]:
    manifests = sorted(manifest_dir.glob("*.json"))
    if not manifests:
        raise FileNotFoundError(
            f"No public-data lock manifests were found below {manifest_dir}"
        )
    index: dict[tuple[str, str], tuple[Mapping[str, Any], Path]] = {}
    for path in manifests:
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        record_url = str(manifest.get("record_url", "")).rstrip("/")
        files = manifest.get("files", [])
        if not record_url or not isinstance(files, list):
            continue
        for record in files:
            if not isinstance(record, dict) or not record.get("path"):
                continue
            key = (record_url, Path(str(record["path"])).name.casefold())
            if key in index:
                raise ValueError(f"Duplicate public-data lock key: {key}")
            index[key] = (record, path.resolve())

    verified: list[MultiplexField] = []
    for field in fields:
        key = (field.source_url.rstrip("/"), Path(field.source_path).name.casefold())
        if key not in index:
            raise ValueError(
                f"No public-data lock record for {field.source_id!r} at {field.source_url}"
            )
        record, manifest_path = index[key]
        expected_sha256 = str(record.get("sha256", "")).casefold()
        if expected_sha256 != field.sha256.casefold():
            raise ValueError(
                f"SHA-256 lock mismatch for {field.source_id!r}: "
                f"expected={expected_sha256}, observed={field.sha256.casefold()}"
            )
        expected_group = record.get("group_id")
        if expected_group is not None and str(expected_group) != field.group_id:
            raise ValueError(
                f"Group lock mismatch for {field.source_id!r}: "
                f"expected={expected_group!r}, observed={field.group_id!r}"
            )
        expected_shape = record.get("shape_cyx", record.get("shape_yx"))
        if expected_shape is not None and tuple(expected_shape) != field.native_shape:
            raise ValueError(
                f"Native-shape lock mismatch for {field.source_id!r}: "
                f"expected={expected_shape}, observed={field.native_shape}"
            )
        expected_dtype = record.get("dtype")
        if expected_dtype is not None and str(expected_dtype) != str(field.image.dtype):
            raise ValueError(
                f"Dtype lock mismatch for {field.source_id!r}: "
                f"expected={expected_dtype!r}, observed={str(field.image.dtype)!r}"
            )
        verified.append(
            replace(
                field,
                lock_manifest_path=str(manifest_path),
                lock_verified=True,
            )
        )
    return verified


def load_public_multiplex_fields(
    *,
    comet_dir: str | Path,
    cosmx_dir: str | Path | Sequence[str | Path],
    max_dimension: int = 896,
    verify_locks: bool = True,
    lock_manifest_dir: str | Path | None = None,
) -> list[MultiplexField]:
    """Discover, load, and by default verify recovered public TIFF locks."""

    comet_paths = sorted(Path(comet_dir).glob("*.tif*"))
    cosmx_paths = _discover_tiffs(cosmx_dir)
    fields = [
        load_comet_dapi_tiff(path, max_dimension=max_dimension) for path in comet_paths
    ]
    for path in cosmx_paths:
        fields.append(
            load_cosmx_morphology_tiff(
                path,
                max_dimension=max_dimension,
                **_cosmx_source_identity(path),
            )
        )
    if not fields:
        raise ValueError("No COMET or CosMx TIFF files were discovered")
    if not verify_locks:
        return fields
    manifest_dir = (
        Path(lock_manifest_dir)
        if lock_manifest_dir is not None
        else Path(__file__).resolve().parents[2] / "configs" / "public_data"
    )
    return _verify_public_field_locks(fields, manifest_dir)


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def assign_group_splits(
    fields: Sequence[MultiplexField], *, seed: int
) -> dict[str, str]:
    """Assign disjoint independent source groups to fit, calibration, and test."""

    if not fields:
        raise ValueError("At least one field is required")
    _validate_unique_field_identities(fields)
    assignments: dict[str, str] = {}
    for modality in sorted({field.modality for field in fields}):
        groups = sorted(
            {field.group_id for field in fields if field.modality == modality}
        )
        if len(groups) < 3:
            raise ValueError(
                f"{modality} needs at least three source groups for fit/calibration/test"
            )
        ordered = sorted(
            groups, key=lambda group: (_stable_seed(seed, modality, group), group)
        )
        calibration_count = max(1, round(0.2 * len(groups)))
        test_count = max(1, round(0.2 * len(groups)))
        fit_count = len(groups) - calibration_count - test_count
        if fit_count < 1:
            fit_count, calibration_count = 1, 1
            test_count = len(groups) - 2
        roles = (
            ["fit"] * fit_count
            + ["calibration"] * calibration_count
            + ["test"] * test_count
        )
        assignments.update(dict(zip(ordered, roles, strict=True)))
    return assignments


def _validate_unique_field_identities(
    fields: Sequence[MultiplexField],
) -> dict[str, Any]:
    """Reject copied or repeated fields before any group split is constructed."""

    identity_extractors: dict[str, Callable[[MultiplexField], str]] = {
        "source_id": lambda field: field.source_id.casefold(),
        "canonical_source_path": lambda field: str(
            Path(field.source_path).expanduser().resolve()
        ).casefold(),
        "sha256_content": lambda field: field.sha256.casefold(),
    }
    for identity_name, extractor in identity_extractors.items():
        observed: set[str] = set()
        duplicate_found = False
        for field in fields:
            identity = extractor(field)
            if identity in observed:
                duplicate_found = True
            observed.add(identity)
        if duplicate_found:
            raise ValueError(
                f"Duplicate {identity_name} identity across benchmark fields; "
                "each physical/content field must appear exactly once"
            )
    return {
        "field_count": len(fields),
        "source_ids_unique": True,
        "canonical_source_paths_unique": True,
        "sha256_content_digests_unique": True,
        "checked_before_split": True,
    }


def _resolve_role_assignments(
    fields: Sequence[MultiplexField],
    *,
    seed: int,
    assignments: Mapping[str, str] | None,
) -> dict[str, str]:
    _validate_unique_field_identities(fields)
    group_modalities: dict[str, set[str]] = {}
    for field in fields:
        group_modalities.setdefault(field.group_id, set()).add(field.modality)
    collisions = sorted(
        group for group, modalities in group_modalities.items() if len(modalities) > 1
    )
    if collisions:
        raise ValueError(
            "group_id values must be collision-safe across modalities; collisions="
            f"{collisions}"
        )
    for group in sorted(group_modalities):
        records = [field for field in fields if field.group_id == group]
        semantics = {
            (
                field.group_level,
                field.group_independence_declared,
                field.group_independence_basis,
            )
            for field in records
        }
        if len(semantics) != 1:
            raise ValueError(
                f"Inconsistent group-level provenance within group {group!r}"
            )
    if assignments is None:
        return assign_group_splits(fields, seed=seed)
    roles = {str(group): str(role) for group, role in assignments.items()}
    expected = set(group_modalities)
    missing = sorted(expected - set(roles))
    extra = sorted(set(roles) - expected)
    if missing or extra:
        raise ValueError(
            f"Explicit role assignments must exactly cover source groups; missing={missing}, "
            f"extra={extra}"
        )
    invalid = sorted(
        {role for role in roles.values() if role not in {"fit", "calibration", "test"}}
    )
    if invalid:
        raise ValueError(f"Unsupported split roles: {invalid}")
    for modality in sorted({field.modality for field in fields}):
        modality_roles = {
            roles[field.group_id] for field in fields if field.modality == modality
        }
        if modality_roles != {"fit", "calibration", "test"}:
            raise ValueError(
                f"{modality} explicit assignments require fit, calibration, and test; "
                f"received={sorted(modality_roles)}"
            )
    return roles


def _curve_mask(
    shape: tuple[int, int],
    *,
    rng: np.random.Generator,
    radius: int,
    vertical: bool,
) -> BoolArray:
    height, width = shape
    long_length = height if vertical else width
    short_length = width if vertical else height
    position = float(rng.uniform(0.38, 0.62) * short_length)
    amplitude = float(rng.uniform(0.035, 0.09) * short_length)
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    line = np.zeros(shape, dtype=bool)
    start, stop = int(0.08 * long_length), int(0.92 * long_length)
    coordinate = np.arange(start, max(start + 1, stop))
    normalized = coordinate / max(long_length - 1, 1)
    transverse = np.rint(
        position + amplitude * np.sin(2.0 * np.pi * normalized + phase)
    ).astype(int)
    if vertical:
        valid = (transverse >= 0) & (transverse < width)
        line[coordinate[valid], transverse[valid]] = True
    else:
        valid = (transverse >= 0) & (transverse < height)
        line[transverse[valid], coordinate[valid]] = True
    if radius > 0:
        yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        footprint = yy * yy + xx * xx <= radius * radius
        line = ndimage.binary_dilation(line, structure=footprint)
    return np.asarray(line, dtype=bool)


def _cast_like(values: FloatArray, reference: ImageArray) -> ImageArray:
    if np.issubdtype(reference.dtype, np.integer):
        info = np.iinfo(reference.dtype)
        values = np.rint(np.clip(values, info.min, info.max))
    else:
        source_maximum = float(np.max(reference))
        if source_maximum <= 1.0:
            upper = 1.0
        elif source_maximum <= 255.0:
            upper = 255.0
        elif source_maximum <= 65535.0:
            upper = 65535.0
        else:
            upper = max(source_maximum, 1.0)
        values = np.clip(values, 0.0, upper)
    return np.asarray(values, dtype=reference.dtype)


def inject_multiplex_artifact(
    image: ArrayLike,
    *,
    artifact: str,
    severity: float,
    seed: int,
) -> InjectedField:
    """Inject a bright overlap fold or thin signal-loss crack without mutation."""

    source = np.asarray(image)
    if source.ndim != 3 or min(source.shape) <= 0:
        raise ValueError("image must have shape C x Y x X")
    if not 0.0 < severity <= 1.0 or not math.isfinite(severity):
        raise ValueError("severity must lie in (0, 1]")
    normalized_artifact = artifact.casefold()
    if normalized_artifact not in {"fold", "crack"}:
        raise ValueError("artifact must be 'fold' or 'crack'")
    rng = np.random.default_rng(seed)
    min_dimension = min(source.shape[1:])
    if min_dimension < 16:
        raise ValueError("images must be at least 16 x 16 for artifact injection")
    vertical = bool(rng.integers(0, 2))
    if normalized_artifact == "fold":
        radius = max(2, round(min_dimension * 0.022))
    else:
        radius = max(1, round(min_dimension * 0.004))
    mask = _curve_mask(source.shape[1:], rng=rng, radius=radius, vertical=vertical)
    base = source.astype(np.float64, copy=False)
    modified = np.array(base, copy=True)
    if normalized_artifact == "fold":
        offset = max(1, round(radius * 1.1))
        shift = (0, offset) if vertical else (offset, 0)
        shifted = np.stack(
            [
                ndimage.shift(
                    plane,
                    shift=shift,
                    order=1,
                    mode="reflect",
                    prefilter=False,
                )
                for plane in base
            ],
            axis=0,
        )
        gain = 0.35 + 0.85 * severity
        modified[:, mask] = base[:, mask] + gain * shifted[:, mask]
        parameters: dict[str, Any] = {
            "model": "local_shifted_signal_superposition",
            "radius_pixels": radius,
            "offset_yx": list(shift),
            "superposition_gain": gain,
            "orientation": "vertical" if vertical else "horizontal",
        }
    else:
        retained_fraction = 0.30 * (1.0 - severity)
        modified[:, mask] = base[:, mask] * retained_fraction
        parameters = {
            "model": "thin_multichannel_signal_attenuation",
            "radius_pixels": radius,
            "retained_signal_fraction": retained_fraction,
            "orientation": "vertical" if vertical else "horizontal",
        }
    output = _cast_like(modified, source)
    effective_changed_mask = np.any(output != source, axis=0)
    intended_count = int(np.count_nonzero(mask))
    effective_count = int(np.count_nonzero(effective_changed_mask))
    parameters["support_realization"] = {
        "intended_pixel_count": intended_count,
        "effective_changed_pixel_count": effective_count,
        "effective_fraction_of_intended": (
            effective_count / intended_count if intended_count else None
        ),
        "definition": "spatial pixel where at least one output channel differs after clipping and dtype casting",
    }
    return InjectedField(
        image=output,
        intended_mask=mask,
        effective_changed_mask=effective_changed_mask,
        artifact=normalized_artifact,
        severity=float(severity),
        seed=int(seed),
        injection_parameters=parameters,
    )


def incremental_score(injected_score: ArrayLike, base_score: ArrayLike) -> FloatArray:
    """Return paired non-negative response ``max(injected - base, 0)``."""

    injected = np.asarray(injected_score, dtype=np.float64)
    base = np.asarray(base_score, dtype=np.float64)
    if injected.shape != base.shape or injected.ndim != 2:
        raise ValueError("score maps must be matching two-dimensional arrays")
    if not np.isfinite(injected).all() or not np.isfinite(base).all():
        raise ValueError("score maps must be finite")
    return np.maximum(injected - base, 0.0)


def binary_average_precision(labels: ArrayLike, scores: ArrayLike) -> float | None:
    """Tie-invariant average precision; ``None`` when positives are absent."""

    truth = np.asarray(labels, dtype=bool).reshape(-1)
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.size != values.size or not np.isfinite(values).all():
        raise ValueError("labels and finite scores must have equal size")
    positives = int(np.count_nonzero(truth))
    if positives == 0:
        return None
    order = np.argsort(-values, kind="stable")
    sorted_scores = values[order]
    sorted_truth = truth[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), truth.size - 1]
    cumulative_tp = np.cumsum(sorted_truth, dtype=np.int64)[ends]
    retrieved = ends + 1
    recall = cumulative_tp / positives
    precision = cumulative_tp / retrieved
    recall_increment = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_increment * precision))


def _dice(labels: BoolArray, prediction: BoolArray) -> float | None:
    positives = int(np.count_nonzero(labels))
    if positives == 0:
        return None
    intersection = int(np.count_nonzero(labels & prediction))
    denominator = positives + int(np.count_nonzero(prediction))
    return float(2 * intersection / denominator) if denominator else 0.0


def _lesion_hit_rate(labels: BoolArray, prediction: BoolArray) -> float | None:
    components, count = ndimage.label(labels)
    if count == 0:
        return None
    hits = sum(
        bool(np.any(prediction[components == index])) for index in range(1, count + 1)
    )
    return float(hits / count)


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    # ``rankdata`` is intentionally local to keep scipy.stats out of import hot paths.
    from scipy.stats import rankdata

    x_rank = rankdata(np.asarray(x, dtype=np.float64), method="average")
    y_rank = rankdata(np.asarray(y, dtype=np.float64), method="average")
    if np.std(x_rank) <= 1e-12 or np.std(y_rank) <= 1e-12:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


@dataclass(frozen=True)
class _ResponseSample:
    score: FloatArray
    mask: BoolArray


def _select_response_threshold(
    samples: Sequence[_ResponseSample], candidate_count: int
) -> tuple[float, dict[str, Any]]:
    if not samples:
        raise ValueError("Response calibration requires at least one sample")
    sampled_values = np.concatenate(
        [
            sample.score.reshape(-1)[:: max(1, sample.score.size // 100_000)]
            for sample in samples
        ]
    )
    quantiles = np.linspace(0.0, 1.0, candidate_count)
    candidates = np.unique(np.r_[0.0, np.quantile(sampled_values, quantiles)])
    true_positive = np.zeros(len(candidates), dtype=np.int64)
    false_positive = np.zeros(len(candidates), dtype=np.int64)
    false_negative = np.zeros(len(candidates), dtype=np.int64)
    for sample in samples:
        values = sample.score.reshape(-1)
        labels = sample.mask.reshape(-1)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        cumulative_positive = np.r_[
            np.int64(0), np.cumsum(labels[order], dtype=np.int64)
        ]
        indices = np.searchsorted(sorted_values, candidates, side="right")
        positive_count = int(np.count_nonzero(labels))
        predicted_count = len(values) - indices
        sample_true_positive = positive_count - cumulative_positive[indices]
        true_positive += sample_true_positive
        false_positive += predicted_count - sample_true_positive
        false_negative += positive_count - sample_true_positive
    dice = np.divide(
        2 * true_positive,
        2 * true_positive + false_positive + false_negative,
        out=np.zeros(len(candidates), dtype=np.float64),
        where=(2 * true_positive + false_positive + false_negative) > 0,
    )
    precision = np.divide(
        true_positive,
        true_positive + false_positive,
        out=np.zeros(len(candidates), dtype=np.float64),
        where=(true_positive + false_positive) > 0,
    )
    best_index = max(
        range(len(candidates)),
        key=lambda index: (dice[index], precision[index], candidates[index]),
    )
    best_threshold = float(candidates[best_index])
    best_counts = (
        int(true_positive[best_index]),
        int(false_positive[best_index]),
        int(false_negative[best_index]),
    )
    return best_threshold, {
        "objective": (
            "maximum pooled calibration spike-in Dice among the predeclared "
            "deterministic quantile candidate grid"
        ),
        "global_observed_score_optimum_claimed": False,
        "candidate_generation": (
            "quantiles of deterministically stride-sampled calibration response "
            "pixels plus zero"
        ),
        "candidate_source_pixel_count": int(sampled_values.size),
        "calibration_pixel_count_total": int(
            sum(sample.score.size for sample in samples)
        ),
        "tie_break": "precision_then_higher_threshold",
        "comparison": "incremental_score_strictly_greater_than_threshold",
        "candidate_count": len(candidates),
        "calibration_counts": {
            "true_positive_pixels": best_counts[0],
            "activated_outside_mask_pixels": best_counts[1],
            "missed_mask_pixels": best_counts[2],
        },
        "test_labels_used": False,
    }


def _patch_table(field: MultiplexField, config: MultiplexProxyConfig) -> FeatureTable:
    patch = min(config.patch_size, *field.image.shape[1:])
    stride = min(config.stride, patch)
    return extract_patch_feature_table(
        field.image,
        modality=field.modality,
        patch_size=patch,
        stride=stride,
        min_tissue_fraction=0.0,
        channel_axis=0,
        structural_channels=tuple(range(field.image.shape[0])),
    )


def _anomaly_map(
    image: ImageArray,
    *,
    modality: str,
    detector: CleanReferenceAnomalyDetector,
    config: MultiplexProxyConfig,
) -> FloatArray:
    patch = min(config.patch_size, *image.shape[1:])
    stride = min(config.stride, patch)
    table = extract_patch_feature_table(
        image,
        modality=modality,
        patch_size=patch,
        stride=stride,
        min_tissue_fraction=0.0,
        channel_axis=0,
        structural_channels=tuple(range(image.shape[0])),
    )
    if len(table) == 0:
        return np.zeros(image.shape[1:], dtype=np.float64)
    scores = detector.score_samples(table)
    return tile_scores_to_map(scores, table.coordinates, table.image_shape)


@dataclass(frozen=True)
class _ScoreBundle:
    """Compute all three methods with shared classical/anomaly passes."""

    modality: str
    anomaly_detector: CleanReferenceAnomalyDetector
    anomaly_scale: float
    config: MultiplexProxyConfig

    @property
    def methods(self) -> tuple[str, ...]:
        return ("classical", "clean_reference_anomaly", "hybrid")

    def score_all(self, image: ImageArray) -> dict[str, FloatArray]:
        candidates = classical_candidate_masks(
            image,
            modality=self.modality,
            channel_axis=0,
            min_component_size=8,
        )
        classical = np.maximum(candidates.fold_score, candidates.crack_score)
        raw_anomaly = _anomaly_map(
            image,
            modality=self.modality,
            detector=self.anomaly_detector,
            config=self.config,
        )
        anomaly = np.clip(0.5 * raw_anomaly / self.anomaly_scale, 0.0, 1.0)
        total = self.config.classical_weight + self.config.anomaly_weight
        hybrid = np.clip(
            (
                self.config.classical_weight * classical
                + self.config.anomaly_weight * anomaly
            )
            / total,
            0.0,
            1.0,
        )
        return {
            "classical": classical,
            "clean_reference_anomaly": anomaly,
            "hybrid": hybrid,
        }


def _build_score_bundles(
    fields: Sequence[MultiplexField],
    roles: Mapping[str, str],
    config: MultiplexProxyConfig,
) -> tuple[dict[str, _ScoreBundle], dict[str, dict[str, Any]]]:
    bundles: dict[str, _ScoreBundle] = {}
    provenance: dict[str, dict[str, Any]] = {}
    for modality in sorted({field.modality for field in fields}):
        fit_fields = [
            field
            for field in fields
            if field.modality == modality and roles[field.group_id] == "fit"
        ]
        calibration_fields = [
            field
            for field in fields
            if field.modality == modality and roles[field.group_id] == "calibration"
        ]
        fit_tables = [_patch_table(field, config) for field in fit_fields]
        fit_values = [table.values for table in fit_tables if len(table)]
        if not fit_values or sum(len(values) for values in fit_values) < 3:
            raise ValueError(f"{modality} has fewer than three clean-reference patches")
        fit_matrix = np.concatenate(fit_values, axis=0)
        fit_group_ids = sorted(field.group_id for field in fit_fields)
        fit_reference_id = hashlib.sha256("|".join(fit_group_ids).encode()).hexdigest()
        anomaly = CleanReferenceAnomalyDetector().fit(
            fit_matrix, reference_group_id=fit_reference_id
        )

        raw_calibration_maps = [
            _anomaly_map(
                field.image,
                modality=modality,
                detector=anomaly,
                config=config,
            )
            for field in calibration_fields
        ]
        raw_values = np.concatenate(
            [score.reshape(-1) for score in raw_calibration_maps]
        )
        anomaly_scale = max(
            float(np.quantile(raw_values, config.base_alert_quantile)),
            np.finfo(np.float64).eps,
        )

        bundles[modality] = _ScoreBundle(modality, anomaly, anomaly_scale, config)
        total_weight = config.classical_weight + config.anomaly_weight
        provenance[modality] = {
            "fit_group_ids": fit_group_ids,
            "calibration_group_ids": sorted(
                field.group_id for field in calibration_fields
            ),
            "anomaly_fit_patch_count": int(fit_matrix.shape[0]),
            "anomaly_feature_count": int(fit_matrix.shape[1]),
            "anomaly_normalization": {
                "formula": "clip(0.5 * raw_mahalanobis / calibration_quantile, 0, 1)",
                "calibration_quantile": config.base_alert_quantile,
                "raw_quantile": anomaly_scale,
                "test_data_used": False,
            },
            "hybrid_weights": {
                "classical": config.classical_weight / total_weight,
                "clean_reference_anomaly": config.anomaly_weight / total_weight,
            },
        }
    return bundles, provenance


def _base_thresholds(
    fields: Sequence[MultiplexField],
    roles: Mapping[str, str],
    bundles: Mapping[str, _ScoreBundle],
    quantile: float,
) -> dict[tuple[str, str], float]:
    thresholds: dict[tuple[str, str], float] = {}
    for modality, bundle in bundles.items():
        calibration = [
            field
            for field in fields
            if field.modality == modality and roles[field.group_id] == "calibration"
        ]
        scored = [bundle.score_all(field.image) for field in calibration]
        for method in bundle.methods:
            values = np.concatenate([maps[method].reshape(-1) for maps in scored])
            thresholds[(modality, method)] = float(np.quantile(values, quantile))
    return thresholds


def _response_calibration(
    fields: Sequence[MultiplexField],
    roles: Mapping[str, str],
    bundles: Mapping[str, _ScoreBundle],
    config: MultiplexProxyConfig,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], dict[str, Any]]]:
    thresholds: dict[tuple[str, str], float] = {}
    provenance: dict[tuple[str, str], dict[str, Any]] = {}
    for modality, bundle in bundles.items():
        calibration = [
            field
            for field in fields
            if field.modality == modality and roles[field.group_id] == "calibration"
        ]
        samples_by_method: dict[str, list[_ResponseSample]] = {
            method: [] for method in bundle.methods
        }
        support_counts: list[tuple[int, int]] = []
        for field in calibration:
            base_scores = bundle.score_all(field.image)
            for artifact in ("fold", "crack"):
                for severity in config.severities:
                    seed = _stable_seed(config.seed, field.source_id, artifact)
                    injected = inject_multiplex_artifact(
                        field.image,
                        artifact=artifact,
                        severity=severity,
                        seed=seed,
                    )
                    injected_scores = bundle.score_all(injected.image)
                    support_counts.append(
                        (
                            int(np.count_nonzero(injected.intended_mask)),
                            int(np.count_nonzero(injected.effective_changed_mask)),
                        )
                    )
                    for method in bundle.methods:
                        response = incremental_score(
                            injected_scores[method], base_scores[method]
                        )
                        samples_by_method[method].append(
                            _ResponseSample(response, injected.effective_changed_mask)
                        )
        for method, samples in samples_by_method.items():
            threshold, details = _select_response_threshold(
                samples, config.response_threshold_candidates
            )
            calibration_ids = sorted(field.group_id for field in calibration)
            details.update(
                {
                    "threshold": threshold,
                    "role_used": "calibration",
                    "calibration_group_ids": calibration_ids,
                    "fit_groups_used_for_threshold": False,
                    "test_group_ids_used": [],
                    "evaluation_mask_semantics": (
                        "effective changed support after clipping and dtype casting"
                    ),
                    "calibration_injection_support_realization": {
                        "injection_count": len(support_counts),
                        "intended_pixel_count": sum(
                            intended for intended, _ in support_counts
                        ),
                        "effective_changed_pixel_count": sum(
                            effective for _, effective in support_counts
                        ),
                        "effective_fraction_of_intended": (
                            sum(effective for _, effective in support_counts)
                            / sum(intended for intended, _ in support_counts)
                        ),
                    },
                }
            )
            thresholds[(modality, method)] = threshold
            provenance[(modality, method)] = details
    return thresholds, provenance


def _response_metrics(
    response: FloatArray,
    mask: BoolArray,
    threshold: float,
    lesion_dilation: int,
) -> dict[str, Any]:
    evaluation_mask = mask
    if lesion_dilation:
        evaluation_mask = ndimage.binary_dilation(mask, iterations=lesion_dilation)
    prediction = response > threshold
    total_activation = float(np.sum(response))
    outside_activation = float(np.sum(response[~evaluation_mask]))
    outside_pixels = int(np.count_nonzero(~evaluation_mask))
    return {
        "incremental_auprc": binary_average_precision(evaluation_mask, response),
        "calibration_thresholded_dice": _dice(evaluation_mask, prediction),
        "lesion_hit_rate": _lesion_hit_rate(evaluation_mask, prediction),
        "outside_mask_activation_mass_fraction": (
            outside_activation / total_activation if total_activation > 0.0 else None
        ),
        "outside_mask_pixels_activated_fraction": (
            float(np.count_nonzero(prediction & ~evaluation_mask)) / outside_pixels
            if outside_pixels
            else None
        ),
        "mean_incremental_score_inside_mask": float(np.mean(response[evaluation_mask])),
        "mean_incremental_score_outside_mask": (
            float(np.mean(response[~evaluation_mask])) if outside_pixels else None
        ),
        "activated_pixel_count": int(np.count_nonzero(prediction)),
    }


def _flip_metrics(
    base_score: FloatArray,
    restored_flipped_score: FloatArray,
    threshold: float,
) -> dict[str, Any]:
    restored = restored_flipped_score
    first = base_score > threshold
    second = restored > threshold
    union_count = int(np.count_nonzero(first | second))
    intersection = int(np.count_nonzero(first & second))
    return {
        "transform": "horizontal_flip_then_inverse",
        "mean_absolute_score_error": float(np.mean(np.abs(base_score - restored))),
        "alert_mask_dice": (
            float(
                2 * intersection / (np.count_nonzero(first) + np.count_nonzero(second))
            )
            if union_count
            else 1.0
        ),
    }


def _summary(values: Sequence[float | None]) -> dict[str, Any]:
    finite = np.asarray(
        [value for value in values if value is not None and np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"n": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    return {
        "n": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "minimum": float(np.min(finite)),
        "maximum": float(np.max(finite)),
    }


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "incremental_auprc",
        "calibration_thresholded_dice",
        "lesion_hit_rate",
        "outside_mask_activation_mass_fraction",
        "outside_mask_pixels_activated_fraction",
    )
    aggregate: dict[str, Any] = {}
    keys = sorted({(str(row["modality"]), str(row["method"])) for row in rows})
    for modality, method in keys:
        selected = [
            row
            for row in rows
            if row["modality"] == modality and row["method"] == method
        ]
        aggregate[f"{modality}:{method}"] = {
            "test_injection_count": len(selected),
            "macro_metrics": {
                metric: _summary([row["metrics"][metric] for row in selected])
                for metric in metric_names
            },
        }
    return aggregate


def _evaluate_test_fields(
    fields: Sequence[MultiplexField],
    roles: Mapping[str, str],
    bundles: Mapping[str, _ScoreBundle],
    base_thresholds: Mapping[tuple[str, str], float],
    response_thresholds: Mapping[tuple[str, str], float],
    config: MultiplexProxyConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    response_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    severity_groups: dict[
        tuple[str, str, str, str, str, str], list[tuple[float, float, float]]
    ] = {}
    test_fields = [field for field in fields if roles[field.group_id] == "test"]
    for field in test_fields:
        bundle = bundles[field.modality]
        base_started = time.perf_counter()
        base_scores = bundle.score_all(field.image)
        flipped_scores = bundle.score_all(np.flip(field.image, axis=2))
        base_runtime = float(time.perf_counter() - base_started)
        for method in bundle.methods:
            base_score = base_scores[method]
            base_threshold = base_thresholds[(field.modality, method)]
            base_rows.append(
                _base_result_row(
                    field,
                    method,
                    base_score,
                    np.flip(flipped_scores[method], axis=1),
                    base_threshold,
                    base_runtime,
                )
            )
        for artifact in ("fold", "crack"):
            geometry_seed = _stable_seed(config.seed, field.source_id, artifact)
            for severity in config.severities:
                injection_started = time.perf_counter()
                injected = inject_multiplex_artifact(
                    field.image,
                    artifact=artifact,
                    severity=severity,
                    seed=geometry_seed,
                )
                injected_scores = bundle.score_all(injected.image)
                shared_runtime = float(time.perf_counter() - injection_started)
                for method in bundle.methods:
                    response = incremental_score(
                        injected_scores[method], base_scores[method]
                    )
                    metrics = _response_metrics(
                        response,
                        injected.effective_changed_mask,
                        response_thresholds[(field.modality, method)],
                        config.lesion_dilation_pixels,
                    )
                    response_rows.append(
                        _response_result_row(
                            field,
                            method,
                            injected,
                            geometry_seed,
                            response_thresholds[(field.modality, method)],
                            metrics,
                            shared_runtime,
                        )
                    )
                    fixed_geometry_response = float(
                        np.mean(response[injected.intended_mask])
                    )
                    realized_fraction = float(
                        np.count_nonzero(injected.effective_changed_mask)
                        / np.count_nonzero(injected.intended_mask)
                    )
                    severity_groups.setdefault(
                        (
                            field.source_id,
                            field.group_id,
                            field.cohort_id,
                            field.modality,
                            method,
                            artifact,
                        ),
                        [],
                    ).append((severity, fixed_geometry_response, realized_fraction))
    severity_rows = [
        {
            "source_id": source_id,
            "group_id": group_id,
            "cohort_id": cohort_id,
            "modality": modality,
            "method": method,
            "artifact": artifact,
            "severity_response_definition": (
                "mean incremental score over the fixed intended geometry; "
                "all intended pixels remain in the denominator and score response "
                "may include contextual detector effects"
            ),
            "severity_spearman": _spearman(
                [item[0] for item in values], [item[1] for item in values]
            ),
            "realized_support_fraction_by_severity": [
                {"severity": item[0], "fraction": item[2]} for item in values
            ],
            "n_severities": len(values),
        }
        for (
            source_id,
            group_id,
            cohort_id,
            modality,
            method,
            artifact,
        ), values in sorted(severity_groups.items())
    ]
    return response_rows, base_rows, severity_rows


def _base_result_row(
    field: MultiplexField,
    method: str,
    base_score: FloatArray,
    restored_flipped_score: FloatArray,
    threshold: float,
    shared_runtime: float,
) -> dict[str, Any]:
    return {
        "source_id": field.source_id,
        "group_id": field.group_id,
        "cohort_id": field.cohort_id,
        "modality": field.modality,
        "method": method,
        "label": "unmodified_real_field_alert_burden_not_false_positive_rate",
        "alert_threshold": threshold,
        "alert_burden_fraction": float(np.mean(base_score > threshold)),
        "score_summary": {
            "mean": float(np.mean(base_score)),
            "p99": float(np.quantile(base_score, 0.99)),
            "maximum": float(np.max(base_score)),
        },
        "horizontal_flip_consistency": _flip_metrics(
            base_score, restored_flipped_score, threshold
        ),
        "shared_bundle_runtime_seconds": shared_runtime,
    }


def _response_result_row(
    field: MultiplexField,
    method: str,
    injected: InjectedField,
    injection_seed: int,
    threshold: float,
    metrics: Mapping[str, Any],
    shared_runtime: float,
) -> dict[str, Any]:
    return {
        "source_id": field.source_id,
        "group_id": field.group_id,
        "cohort_id": field.cohort_id,
        "modality": field.modality,
        "method": method,
        "artifact": injected.artifact,
        "severity": injected.severity,
        "injection_seed": injection_seed,
        "injection_parameters": dict(injected.injection_parameters),
        "evaluation_mask_semantics": (
            "effective changed support after clipping and dtype casting"
        ),
        "intended_mask_pixel_count": int(np.count_nonzero(injected.intended_mask)),
        "effective_changed_mask_pixel_count": int(
            np.count_nonzero(injected.effective_changed_mask)
        ),
        "effective_changed_fraction_of_intended": (
            float(np.count_nonzero(injected.effective_changed_mask))
            / float(np.count_nonzero(injected.intended_mask))
        ),
        "score_definition": "max(score(injected)-score(base),0)",
        "response_threshold": threshold,
        "metrics": dict(metrics),
        "shared_bundle_runtime_seconds": shared_runtime,
    }


def _split_summary(
    fields: Sequence[MultiplexField], roles: Mapping[str, str]
) -> dict[str, Any]:
    groups = {
        role: sorted(group for group, assigned in roles.items() if assigned == role)
        for role in ("fit", "calibration", "test")
    }
    overlap = {
        "fit_calibration": sorted(set(groups["fit"]) & set(groups["calibration"])),
        "fit_test": sorted(set(groups["fit"]) & set(groups["test"])),
        "calibration_test": sorted(set(groups["calibration"]) & set(groups["test"])),
    }
    cosmx_cohorts = sorted(
        {field.cohort_id for field in fields if field.modality == "cosmx"}
    )
    return {
        "unit": "declared_source_group; public CosMx uses slide/run rather than FOV",
        "assignment": "deterministic_hash_stratified_by_modality",
        "groups": groups,
        "overlap_audit": overlap,
        "all_overlaps_empty": not any(overlap.values()),
        "cosmx_slide_run_groups": cosmx_cohorts,
        "cosmx_limitation": (
            None
            if not cosmx_cohorts or len(cosmx_cohorts) >= 3
            else (
                "Fewer than three independent CosMx slide/run groups are present; "
                "the split should have failed before scoring rather than leak FOVs."
            )
        ),
    }


def _threshold_summary(
    bundles: Mapping[str, _ScoreBundle],
    base_thresholds: Mapping[tuple[str, str], float],
    response_provenance: Mapping[tuple[str, str], Mapping[str, Any]],
    config: MultiplexProxyConfig,
) -> dict[str, Any]:
    return {
        f"{modality}:{method}": {
            "response": dict(response_provenance[(modality, method)]),
            "unmodified_alert_burden": {
                "threshold": base_thresholds[(modality, method)],
                "source": "unmodified_calibration_real_field_score_quantile",
                "quantile": config.base_alert_quantile,
                "semantic_label": "alert_burden_threshold_not_false_positive_threshold",
                "test_data_used": False,
            },
        }
        for modality, bundle in bundles.items()
        for method in bundle.methods
    }


def run_multiplex_proxy_benchmark(
    fields: Sequence[MultiplexField],
    config: MultiplexProxyConfig | None = None,
    *,
    role_assignments: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a deterministic fit/calibration/test proxy benchmark.

    The returned dictionary is JSON-ready.  It must not be presented as
    validation against naturally occurring fold/crack ground truth.
    """

    settings = config or MultiplexProxyConfig()
    if not fields:
        raise ValueError("At least one real-background field is required")
    started = time.perf_counter()
    roles = _resolve_role_assignments(
        fields,
        seed=settings.seed,
        assignments=role_assignments,
    )
    bundles, model_provenance = _build_score_bundles(fields, roles, settings)
    base_thresholds = _base_thresholds(
        fields, roles, bundles, settings.base_alert_quantile
    )
    response_thresholds, threshold_provenance = _response_calibration(
        fields, roles, bundles, settings
    )

    response_rows, base_rows, severity_rows = _evaluate_test_fields(
        fields,
        roles,
        bundles,
        base_thresholds,
        response_thresholds,
        settings,
    )
    source_records = [
        field.source_record(roles[field.group_id])
        for field in sorted(fields, key=lambda value: (value.modality, value.source_id))
    ]
    thresholds = _threshold_summary(
        bundles, base_thresholds, threshold_provenance, settings
    )
    return {
        "schema_version": "multiplex-real-background-proxy-v3",
        "benchmark_kind": "label_free_proxy_not_real_artifact_efficacy",
        "report_eligible": False,
        "scientific_validation_passed": False,
        "claim_boundary": {
            "supported": [
                "software execution on the listed real multiplex backgrounds",
                "recovery of controlled synthetic fold/crack perturbations",
                "paired incremental-score and horizontal-flip consistency proxies",
            ],
            "not_supported": [
                "accuracy, Dice, AUROC, sensitivity, or specificity on naturally occurring artifacts",
                "absence of real artifacts in unmodified images",
                "clinical, regulated, or production readiness",
                "cross-site or cross-panel generalization",
            ],
            "mandatory_language": (
                "Activity on unmodified real fields is alert burden, never a false-positive rate. "
                "Spike-in metrics are generator-conditional proxy evidence."
            ),
        },
        "input_contract": {
            "loaded_axis_order": "CYX",
            "loaded_channel_axis": 0,
            "default_max_dimension": 896,
            "intensity_contract": "source dtype preserved; injection uses clipped float workspace",
            "proxy_mask_contract": (
                "metrics use realized spatial support where at least one channel changed "
                "after clipping and dtype casting; intended geometry is retained only for audit"
            ),
        },
        "config": {
            "seed": settings.seed,
            "severities": list(settings.severities),
            "patch_size": settings.patch_size,
            "stride": settings.stride,
            "base_alert_quantile": settings.base_alert_quantile,
            "response_threshold_candidates": settings.response_threshold_candidates,
            "lesion_dilation_pixels": settings.lesion_dilation_pixels,
            "classical_weight": settings.classical_weight,
            "anomaly_weight": settings.anomaly_weight,
            "group_bootstrap_resamples": settings.group_bootstrap_resamples,
            "group_bootstrap_seed": settings.group_bootstrap_seed,
        },
        "split": _split_summary(fields, roles),
        "input_identity_audit": _validate_unique_field_identities(fields),
        "sources": source_records,
        "model_provenance": model_provenance,
        "thresholds": thresholds,
        "test_response_rows": response_rows,
        "test_unmodified_field_rows": base_rows,
        "test_severity_monotonicity_rows": severity_rows,
        "test_aggregate": _aggregate_rows(response_rows),
        "runtime_seconds": float(time.perf_counter() - started),
    }


def _group_bootstrap_summary(
    group_values: Mapping[str, float | None],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    finite = [
        (group, float(value))
        for group, value in sorted(group_values.items())
        if value is not None and np.isfinite(value)
    ]
    if not finite:
        return {
            "n_groups": 0,
            "mean": None,
            "group_values": [],
            "ci95": {
                "lower": None,
                "upper": None,
                "method": "percentile_group_bootstrap",
                "resamples": resamples,
                "seed": seed,
            },
        }
    values = np.asarray([value for _, value in finite], dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    distribution = np.mean(values[indices], axis=1)
    lower, upper = np.quantile(distribution, (0.025, 0.975))
    return {
        "n_groups": len(finite),
        "mean": float(np.mean(values)),
        "group_values": [
            {"group_id": group, "value": value} for group, value in finite
        ],
        "ci95": {
            "lower": float(lower),
            "upper": float(upper),
            "method": "percentile_group_bootstrap",
            "resamples": resamples,
            "seed": seed,
            "resampling_unit": "declared_source_group_out_of_fold_summary",
        },
    }


def _group_macro_strata(
    rows: Sequence[Mapping[str, Any]],
    *,
    strata: Sequence[str],
    metric_getters: Mapping[str, Callable[[Mapping[str, Any]], float | None]],
    config: MultiplexProxyConfig,
    within_group_reduction: str,
) -> dict[str, Any]:
    keys = sorted({tuple(str(row[name]) for name in strata) for row in rows})
    output: dict[str, Any] = {}
    for key in keys:
        selected = [
            row for row in rows if tuple(str(row[name]) for name in strata) == key
        ]
        groups = sorted({str(row["group_id"]) for row in selected})
        metrics: dict[str, Any] = {}
        for metric, getter in metric_getters.items():
            group_values: dict[str, float | None] = {}
            for group in groups:
                values = np.asarray(
                    [
                        value
                        for row in selected
                        if str(row["group_id"]) == group
                        and (value := getter(row)) is not None
                        and np.isfinite(value)
                    ],
                    dtype=np.float64,
                )
                group_values[group] = float(np.mean(values)) if values.size else None
            bootstrap_seed = _stable_seed(config.group_bootstrap_seed, *key, metric)
            metrics[metric] = _group_bootstrap_summary(
                group_values,
                resamples=config.group_bootstrap_resamples,
                seed=bootstrap_seed,
            )
        output[":".join(key)] = {
            "stratum": dict(zip(strata, key, strict=True)),
            "n_test_rows": len(selected),
            "n_source_groups": len(groups),
            "source_groups": groups,
            "within_group_reduction": within_group_reduction,
            "group_macro_metrics": metrics,
        }
    return output


def _cross_validation_aggregates(
    response_rows: Sequence[Mapping[str, Any]],
    base_rows: Sequence[Mapping[str, Any]],
    severity_rows: Sequence[Mapping[str, Any]],
    config: MultiplexProxyConfig,
) -> dict[str, Any]:
    response_metrics = (
        "incremental_auprc",
        "calibration_thresholded_dice",
        "lesion_hit_rate",
        "outside_mask_activation_mass_fraction",
        "outside_mask_pixels_activated_fraction",
    )
    response_getters = {
        metric: (lambda row, name=metric: row["metrics"].get(name))
        for metric in response_metrics
    }
    base_getters: dict[str, Callable[[Mapping[str, Any]], float | None]] = {
        "unmodified_alert_burden_fraction": lambda row: row.get(
            "alert_burden_fraction"
        ),
        "horizontal_flip_score_mae": lambda row: row["horizontal_flip_consistency"].get(
            "mean_absolute_score_error"
        ),
        "horizontal_flip_alert_dice": lambda row: row[
            "horizontal_flip_consistency"
        ].get("alert_mask_dice"),
    }
    return {
        "response_by_modality_method_artifact": _group_macro_strata(
            response_rows,
            strata=("modality", "method", "artifact"),
            metric_getters=response_getters,
            config=config,
            within_group_reduction=(
                "arithmetic mean across out-of-fold source fields and configured severities"
            ),
        ),
        "unmodified_by_modality_method": _group_macro_strata(
            base_rows,
            strata=("modality", "method"),
            metric_getters=base_getters,
            config=config,
            within_group_reduction="arithmetic mean across out-of-fold source fields",
        ),
        "severity_monotonicity_by_modality_method_artifact": _group_macro_strata(
            severity_rows,
            strata=("modality", "method", "artifact"),
            metric_getters={
                "severity_spearman": lambda row: row.get("severity_spearman")
            },
            config=config,
            within_group_reduction=(
                "arithmetic mean across out-of-fold source-field Spearman values"
            ),
        ),
    }


def _leave_one_group_out_roles(
    groups: Sequence[str],
    *,
    modality: str,
    test_group: str,
    seed: int,
) -> dict[str, str]:
    remaining = [group for group in groups if group != test_group]
    if len(remaining) < 2:
        raise ValueError(
            f"{modality} needs at least three source groups for leave-one-group-out"
        )
    calibration_group = min(
        remaining,
        key=lambda group: (
            _stable_seed(
                seed,
                "leave_one_group_out_calibration",
                modality,
                test_group,
                group,
            ),
            group,
        ),
    )
    return {
        group: (
            "test"
            if group == test_group
            else "calibration"
            if group == calibration_group
            else "fit"
        )
        for group in groups
    }


def _cross_validation_fold_manifest(
    fold_id: str,
    fields: Sequence[MultiplexField],
    roles: Mapping[str, str],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    groups_by_role = {
        role: sorted(group for group, assigned in roles.items() if assigned == role)
        for role in ("fit", "calibration", "test")
    }
    source_ids_by_role = {
        role: sorted(
            field.source_id for field in fields if roles[field.group_id] == role
        )
        for role in ("fit", "calibration", "test")
    }
    overlap = report["split"]["overlap_audit"]
    return {
        "fold_id": fold_id,
        "modality": fields[0].modality,
        "groups_by_role": groups_by_role,
        "source_ids_by_role": source_ids_by_role,
        "role_overlap_audit": overlap,
        "all_role_overlaps_empty": not any(overlap.values()),
        "thresholds": report["thresholds"],
        "model_provenance": report["model_provenance"],
        "test_response_row_count": len(report["test_response_rows"]),
        "test_unmodified_row_count": len(report["test_unmodified_field_rows"]),
        "runtime_seconds": report["runtime_seconds"],
    }


def _group_independence_audit(
    fields: Sequence[MultiplexField],
) -> dict[str, Any]:
    by_modality: dict[str, Any] = {}
    for modality in sorted({field.modality for field in fields}):
        modality_groups: list[dict[str, Any]] = []
        for group in sorted(
            {field.group_id for field in fields if field.modality == modality}
        ):
            representative = next(
                field
                for field in fields
                if field.modality == modality and field.group_id == group
            )
            modality_groups.append(
                {
                    "group_id": group,
                    "group_level": representative.group_level,
                    "independence_declared": (
                        representative.group_independence_declared
                    ),
                    "basis": representative.group_independence_basis,
                }
            )
        declared = all(item["independence_declared"] for item in modality_groups)
        by_modality[modality] = {
            "groups": modality_groups,
            "all_group_independence_declared": declared,
            "interpretation": (
                "source release/locked manifest declares the grouping level"
                if declared
                else (
                    "higher-level independence is unverified; cross-validation and "
                    "bootstrap intervals are descriptive under provisional group IDs"
                )
            ),
        }
    all_declared = all(
        record["all_group_independence_declared"] for record in by_modality.values()
    )
    return {
        "by_modality": by_modality,
        "all_modalities_have_declared_group_independence": all_declared,
    }


def run_multiplex_proxy_cross_validation(
    fields: Sequence[MultiplexField],
    config: MultiplexProxyConfig | None = None,
) -> dict[str, Any]:
    """Run leave-one-declared-source-group-out proxy cross-validation.

    Each modality is processed separately.  Within every fold, one group is
    test, one distinct deterministic group is calibration, and all remaining
    groups are fit references.  Only held-out test rows enter the aggregate.
    """

    settings = config or MultiplexProxyConfig()
    if not fields:
        raise ValueError("At least one real-background field is required")
    _resolve_role_assignments(fields, seed=settings.seed, assignments=None)
    started = time.perf_counter()
    fold_manifests: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    base_rows: list[dict[str, Any]] = []
    severity_rows: list[dict[str, Any]] = []
    expected_test_groups: dict[str, list[str]] = {}
    estimated_bundle_passes = 0
    fit_feature_extractions = 0
    anomaly_only_calibration_passes = 0
    reference_claim_boundary: Mapping[str, Any] | None = None

    for modality in sorted({field.modality for field in fields}):
        modality_fields = [field for field in fields if field.modality == modality]
        groups = sorted({field.group_id for field in modality_fields})
        if len(groups) < 3:
            raise ValueError(
                f"{modality} needs at least three source groups for "
                "leave-one-group-out cross-validation"
            )
        expected_test_groups[modality] = groups
        for test_group in groups:
            roles = _leave_one_group_out_roles(
                groups,
                modality=modality,
                test_group=test_group,
                seed=settings.seed,
            )
            fold_id = f"{modality}:test={test_group}"
            fold_report = run_multiplex_proxy_benchmark(
                modality_fields,
                settings,
                role_assignments=roles,
            )
            reference_claim_boundary = fold_report["claim_boundary"]
            for row in fold_report["test_response_rows"]:
                response_rows.append(
                    {"fold_id": fold_id, "out_of_fold_test": True, **row}
                )
            for row in fold_report["test_unmodified_field_rows"]:
                base_rows.append({"fold_id": fold_id, "out_of_fold_test": True, **row})
            for row in fold_report["test_severity_monotonicity_rows"]:
                severity_rows.append(
                    {"fold_id": fold_id, "out_of_fold_test": True, **row}
                )
            fold_manifests.append(
                _cross_validation_fold_manifest(
                    fold_id, modality_fields, roles, fold_report
                )
            )
            calibration_fields = sum(
                roles[field.group_id] == "calibration" for field in modality_fields
            )
            test_fields = sum(
                roles[field.group_id] == "test" for field in modality_fields
            )
            fit_feature_extractions += sum(
                roles[field.group_id] == "fit" for field in modality_fields
            )
            passes_per_field = 2 + 2 * len(settings.severities)
            estimated_bundle_passes += passes_per_field * (
                calibration_fields + test_fields
            )
            anomaly_only_calibration_passes += calibration_fields

    test_appearances = {
        modality: {
            group: sum(
                group in manifest["groups_by_role"]["test"]
                for manifest in fold_manifests
                if manifest["modality"] == modality
            )
            for group in groups
        }
        for modality, groups in expected_test_groups.items()
    }
    exactly_once = all(
        count == 1
        for modality_counts in test_appearances.values()
        for count in modality_counts.values()
    )
    if not exactly_once:  # defensive audit; construction should make this impossible
        raise RuntimeError("Cross-validation test-group coverage is not exactly once")
    assert reference_claim_boundary is not None
    claim_boundary = dict(reference_claim_boundary)
    independence_audit = _group_independence_audit(fields)
    claim_boundary["cross_validation_warning"] = (
        "Folds are statistically dependent because fit/calibration groups are reused. "
        "Group-bootstrap intervals resample out-of-fold source-group summaries, not folds, "
        "and remain conditional on the synthetic perturbation generator. Higher-level "
        "independence is not declared for every public modality; see group_independence_audit."
    )
    return {
        "schema_version": "multiplex-real-background-proxy-logo-cv-v3",
        "benchmark_kind": "label_free_proxy_cross_validation_not_real_artifact_efficacy",
        "cross_validation": "leave_one_declared_source_group_out",
        "fold_construction": {
            "test": "each modality-specific source group exactly once",
            "calibration": (
                "one distinct group selected by deterministic SHA-256 ordering of "
                "seed, modality, test group, and candidate group"
            ),
            "fit": "all remaining modality-specific groups; at least one required",
            "threshold_rule": "fit/calibration only; held-out test labels never used",
        },
        "report_eligible": False,
        "scientific_validation_passed": False,
        "claim_boundary": claim_boundary,
        "fold_dependence_warning": claim_boundary["cross_validation_warning"],
        "aggregation_contract": {
            "included_rows": "out_of_fold_test_only",
            "primary_unit": "declared_source_group",
            "point_estimate": "equal-weight mean of source-group means",
            "ci95": "deterministic percentile bootstrap over source-group means",
            "not_assumed": "cross-validation folds are independent",
        },
        "config": {
            "seed": settings.seed,
            "severities": list(settings.severities),
            "patch_size": settings.patch_size,
            "stride": settings.stride,
            "base_alert_quantile": settings.base_alert_quantile,
            "response_threshold_candidates": settings.response_threshold_candidates,
            "lesion_dilation_pixels": settings.lesion_dilation_pixels,
            "classical_weight": settings.classical_weight,
            "anomaly_weight": settings.anomaly_weight,
            "group_bootstrap_resamples": settings.group_bootstrap_resamples,
            "group_bootstrap_seed": settings.group_bootstrap_seed,
        },
        "sources": [
            field.source_record("varies_by_cross_validation_fold")
            for field in sorted(
                fields,
                key=lambda value: (value.modality, value.group_id, value.source_id),
            )
        ],
        "group_independence_audit": independence_audit,
        "input_identity_audit": _validate_unique_field_identities(fields),
        "fold_manifests": fold_manifests,
        "test_group_coverage_audit": {
            "appearances": test_appearances,
            "every_group_tested_exactly_once": exactly_once,
        },
        "out_of_fold_test_response_rows": response_rows,
        "out_of_fold_test_unmodified_field_rows": base_rows,
        "out_of_fold_test_severity_monotonicity_rows": severity_rows,
        "out_of_fold_group_macro": _cross_validation_aggregates(
            response_rows, base_rows, severity_rows, settings
        ),
        "computational_plan": {
            "fold_count": len(fold_manifests),
            "shared_classical_anomaly_bundle_passes": estimated_bundle_passes,
            "anomaly_only_calibration_passes": anomaly_only_calibration_passes,
            "fit_feature_table_extractions": fit_feature_extractions,
            "note": (
                "Hybrid maps reuse the classical and anomaly passes; counts exclude "
                "TIFF loading and threshold search."
            ),
        },
        "runtime_seconds": float(time.perf_counter() - started),
    }


def write_multiplex_proxy_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Atomically write a JSON-ready proxy report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(destination)
    return destination


__all__ = [
    "InjectedField",
    "MultiplexField",
    "MultiplexProxyConfig",
    "assign_group_splits",
    "binary_average_precision",
    "incremental_score",
    "inject_multiplex_artifact",
    "load_comet_dapi_tiff",
    "load_cosmx_morphology_tiff",
    "load_public_multiplex_fields",
    "run_multiplex_proxy_benchmark",
    "run_multiplex_proxy_cross_validation",
    "write_multiplex_proxy_report",
]
