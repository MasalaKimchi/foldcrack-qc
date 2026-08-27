"""Manifest-driven H&E frozen-feature anomaly benchmark.

This module is deliberately narrower than the general benchmark machinery.  It
implements one claim-bearing experiment: learn a nominal patch memory bank from
reviewed-clean H&E, select the operating threshold on a separate reviewed-clean
calibration split, and evaluate artifact-*union* localization on a locked test
split.  Anomaly evidence is never presented as semantic fold/crack
classification.

The three manifests are a hard evidence boundary.  They are loaded in strict
mode, checked for cross-manifest identity/content leakage, and rejected when
synthetic provenance is declared.  Reports contain anonymous row keys and
aggregate provenance only; source paths and manifest identifiers are never
serialized.

The implementation accepts only pre-extracted ROI or downsampled raster arrays.
It is not a native streaming WSI reader and enforces a configured raster-size
ceiling before allocating score maps.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .evaluation import build_report, evaluate_sample, write_json_report
from .foundation import (
    DINOv2FeatureExtractor,
    FoundationFeatures,
    PatchKNNAnomalyScorer,
)
from .manifest import GROUP_ID_FIELDS, ManifestValidationError, load_samples
from .schema import ChannelRole, Modality, QCSample

BoolArray = NDArray[np.bool_]

__all__ = [
    "FrozenBenchmarkValidationError",
    "run_frozen_anomaly_benchmark",
]


_RGB_ROLES = (
    ChannelRole.BRIGHTFIELD_RED,
    ChannelRole.BRIGHTFIELD_GREEN,
    ChannelRole.BRIGHTFIELD_BLUE,
)
_EXPECTED_SPLITS: Mapping[str, frozenset[str]] = {
    "fit": frozenset(("development", "train")),
    "calibration": frozenset(("validation",)),
    "locked_test": frozenset(("locked_test",)),
}
_SYNTHETIC_TERMS = frozenset(
    ("synthetic", "simulated", "simulation", "generated", "phantom")
)
_SYNTHETIC_FLAG_KEYS = frozenset(
    (
        "synthetic",
        "is_synthetic",
        "simulated",
        "is_simulated",
        "synthetic_seed",
        "generator",
        "generation_method",
        "artifact_spec",
    )
)
_PROVENANCE_KEYS = frozenset(
    ("source", "source_id", "source_type", "origin", "provenance", "dataset_type")
)
_ADJUDICATION_STATUS_KEYS = frozenset(
    (
        "annotation_status",
        "adjudication_status",
        "ground_truth_status",
        "review_status",
    )
)
_ADJUDICATED_TERMS = frozenset(
    ("adjudicated", "expert_adjudicated", "reviewed_and_adjudicated")
)
_REAL_ORIGIN_TERMS = frozenset(
    ("acquired_real", "instrument_acquired", "specimen_acquired")
)
_APPROVED_PROVENANCE_TERMS = frozenset(("approved", "verified"))


class FrozenBenchmarkValidationError(ValueError):
    """Sanitized benchmark-boundary failure.

    Messages intentionally identify only the manifest role and validation code,
    never the offending path, sample identifier, or grouping identifier.
    """

    def __init__(self, code: str, *, stage: str, record_index: int | None = None):
        self.code = str(code)
        self.stage = str(stage)
        self.record_index = record_index
        location = "" if record_index is None else f" at record {record_index}"
        super().__init__(
            f"Frozen benchmark validation failed [{self.code}] in "
            f"{self.stage}{location}"
        )


@dataclass(frozen=True)
class _Patch:
    top: int
    left: int
    bottom: int
    right: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.bottom - self.top, self.right - self.left


@dataclass(frozen=True)
class _LoadedSplit:
    role: str
    manifest_sha256: str
    samples: tuple[QCSample, ...]


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_array_sha256(array: np.ndarray) -> str:
    candidate = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(candidate.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(item) for item in candidate.shape)).encode("ascii"))
    digest.update(memoryview(candidate).cast("B"))
    return digest.hexdigest()


def _normalize_term(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _manifest_records(path: Path) -> list[Any]:
    """Read records only to enforce provenance fields discarded by QCSample."""

    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, Mapping):
        payload = payload.get("samples", [])
    return list(payload) if isinstance(payload, list) else []


def _truthy_synthetic_flag(value: Any) -> bool:
    if value is None or value is False:
        return False
    return not (
        isinstance(value, str)
        and _normalize_term(value) in {"", "false", "no", "0", "real", "acquired"}
    )


def _declares_synthetic(value: Any, *, parent_key: str = "") -> bool:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = _normalize_term(raw_key)
            if key in _SYNTHETIC_FLAG_KEYS and _truthy_synthetic_flag(item):
                return True
            if key in _PROVENANCE_KEYS and isinstance(item, str):
                normalized = _normalize_term(item)
                if any(term in normalized for term in _SYNTHETIC_TERMS):
                    return True
            if _declares_synthetic(item, parent_key=key):
                return True
        return False
    if isinstance(value, (list, tuple)):
        return any(_declares_synthetic(item, parent_key=parent_key) for item in value)
    if isinstance(value, str) and parent_key in _PROVENANCE_KEYS:
        normalized = _normalize_term(value)
        return any(term in normalized for term in _SYNTHETIC_TERMS)
    return False


def _declares_adjudicated(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("adjudicated") is True:
        return True
    for raw_key, item in value.items():
        key = _normalize_term(raw_key)
        if (
            key in _ADJUDICATION_STATUS_KEYS
            and isinstance(item, str)
            and _normalize_term(item) in _ADJUDICATED_TERMS
        ):
            return True
        if (
            key in {"metadata", "annotation", "ground_truth"}
            and isinstance(item, Mapping)
            and _declares_adjudicated(item)
        ):
            return True
    return False


def _declares_approved_real_acquisition(value: Any) -> bool:
    """Require positive acquisition and governance declarations, not mere absence."""

    if not isinstance(value, Mapping):
        return False
    origin = _normalize_term(value.get("data_origin", value.get("origin", "")))
    provenance = _normalize_term(value.get("provenance_status", ""))
    if origin in _REAL_ORIGIN_TERMS and provenance in _APPROVED_PROVENANCE_TERMS:
        return True
    metadata = value.get("metadata")
    return isinstance(metadata, Mapping) and _declares_approved_real_acquisition(
        metadata
    )


def _load_split(path_value: str | Path, role: str) -> _LoadedSplit:
    path = Path(path_value).expanduser()
    try:
        raw_bytes = path.read_bytes()
        records = _manifest_records(path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise FrozenBenchmarkValidationError(
            "manifest_read_error", stage=role
        ) from None
    if any(_declares_synthetic(record) for record in records):
        raise FrozenBenchmarkValidationError(
            "synthetic_provenance_rejected", stage=role
        )
    if any(not _declares_approved_real_acquisition(record) for record in records):
        raise FrozenBenchmarkValidationError(
            "approved_real_acquisition_provenance_required", stage=role
        )
    if any(not _declares_adjudicated(record) for record in records):
        raise FrozenBenchmarkValidationError(
            "explicit_adjudication_status_required", stage=role
        )
    try:
        samples = tuple(load_samples(path, strict=True))
    except ManifestValidationError as error:
        raise FrozenBenchmarkValidationError(
            error.issue.code,
            stage=role,
            record_index=error.issue.record_index,
        ) from None
    return _LoadedSplit(
        role=role,
        manifest_sha256=_sha256_bytes(raw_bytes),
        samples=samples,
    )


def _valid_mask(sample: QCSample) -> BoolArray:
    if "valid" not in sample.masks and "ignore" not in sample.masks:
        raise FrozenBenchmarkValidationError(
            "explicit_valid_or_ignore_mask_required", stage="sample_contract"
        )
    if "valid" in sample.masks:
        valid = np.asarray(sample.masks["valid"], dtype=bool).copy()
    else:
        valid = ~np.asarray(sample.masks["ignore"], dtype=bool)
    if "ignore" in sample.masks:
        valid &= ~np.asarray(sample.masks["ignore"], dtype=bool)
    if not np.any(valid):
        raise FrozenBenchmarkValidationError(
            "valid_region_empty", stage="sample_contract"
        )
    return np.ascontiguousarray(valid)


def _validate_sample_contract(sample: QCSample, role: str) -> None:
    if sample.modality is not Modality.HE:
        raise FrozenBenchmarkValidationError("he_only", stage=role)
    if sample.image.data.ndim != 3 or sample.image.data.shape[-1] != 3:
        raise FrozenBenchmarkValidationError("native_semantic_rgb_required", stage=role)
    if tuple(sample.image.channel_roles) != _RGB_ROLES:
        raise FrozenBenchmarkValidationError("native_semantic_rgb_required", stage=role)
    input_shape = tuple(sample.image.metadata.get("input_shape", ()))
    input_axis = sample.image.metadata.get("input_channel_axis")
    if (
        len(input_shape) != 3
        or input_axis is None
        or int(input_shape[int(input_axis)]) != 3
        or sample.image.metadata.get("color_order") != "rgb"
    ):
        raise FrozenBenchmarkValidationError("native_semantic_rgb_required", stage=role)
    split = _normalize_term(sample.metadata.get("split", ""))
    if split not in _EXPECTED_SPLITS[role]:
        raise FrozenBenchmarkValidationError("unexpected_split_role", stage=role)
    missing_groups = [
        field
        for field in GROUP_ID_FIELDS
        if not str(sample.metadata.get(field, "")).strip()
    ]
    if missing_groups:
        raise FrozenBenchmarkValidationError(
            "complete_group_hierarchy_required", stage=role
        )
    for mask_name in ("fold", "crack"):
        if mask_name not in sample.masks:
            raise FrozenBenchmarkValidationError(
                "explicit_fold_and_crack_masks_required", stage=role
            )
    _valid_mask(sample)
    if role in {"fit", "calibration"} and (
        np.any(sample.masks["fold"]) or np.any(sample.masks["crack"])
    ):
        raise FrozenBenchmarkValidationError(
            "reviewed_clean_masks_must_be_all_zero", stage=role
        )


def _identity_sets(split: _LoadedSplit) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {
        "sample_id": set(),
        **{field: set() for field in GROUP_ID_FIELDS},
        "source_id": set(),
        "source_path": set(),
        "image_file_sha256": set(),
        "image_content_sha256": set(),
    }
    for sample in split.samples:
        values["sample_id"].add(sample.sample_id)
        for field in GROUP_ID_FIELDS:
            values[field].add(str(sample.metadata[field]))
        source_id = sample.metadata.get("source_id")
        if source_id is not None and str(source_id).strip():
            values["source_id"].add(str(source_id))
        if sample.image.source_path is not None:
            values["source_path"].add(str(Path(sample.image.source_path).resolve()))
        verified = sample.metadata.get("verified_sha256", {})
        if isinstance(verified, Mapping) and verified.get("image_path"):
            values["image_file_sha256"].add(str(verified["image_path"]))
        values["image_content_sha256"].add(_canonical_array_sha256(sample.image.data))
    return values


def _prove_disjointness(
    splits: Sequence[_LoadedSplit],
) -> dict[str, Any]:
    identities = {split.role: _identity_sets(split) for split in splits}
    pairs: list[dict[str, Any]] = []
    for index, first in enumerate(splits):
        for second in splits[index + 1 :]:
            overlaps = {
                key: len(identities[first.role][key] & identities[second.role][key])
                for key in identities[first.role]
            }
            if any(overlaps.values()):
                raise FrozenBenchmarkValidationError(
                    "cross_manifest_split_leakage",
                    stage=f"{first.role}_vs_{second.role}",
                )
            pairs.append(
                {
                    "first_role": first.role,
                    "second_role": second.role,
                    "overlap_counts": overlaps,
                }
            )
    return {
        "status": "no_exact_overlap_detected",
        "exact_disjointness_checks_passed": True,
        "mathematical_independence_proven": False,
        "fields_checked": list(next(iter(identities.values())).keys()),
        "pairwise_checks": pairs,
        "limitation": (
            "Exact identifiers, paths, file hashes, and decoded content are checked; "
            "near-duplicate or spatially overlapping exports require an external "
            "parent-WSI split ledger."
        ),
    }


def _pair(value: float | Sequence[float], name: str) -> tuple[float, float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(float(item) for item in value)
        if len(values) != 2:
            raise ValueError(f"{name} must be a scalar or pair")
    else:
        scalar = float(value)
        values = (scalar, scalar)
    if not all(math.isfinite(item) and item > 0 for item in values):
        raise ValueError(f"{name} values must be finite and positive")
    return values  # type: ignore[return-value]


def _validate_geometry_request(
    *,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
) -> None:
    """Reject caller configuration errors before reading or fitting data."""

    physical = patch_size_um is not None or stride_um is not None
    pixel = patch_size_px is not None or stride_px is not None
    if physical and pixel:
        raise ValueError("physical and pixel patch geometry are mutually exclusive")
    if physical:
        if patch_size_um is None:
            raise ValueError("patch_size_um is required when stride_um is supplied")
        patch = _pair(patch_size_um, "patch_size_um")
        stride = (
            tuple(item / 2.0 for item in patch)
            if stride_um is None
            else _pair(stride_um, "stride_um")
        )
    else:
        patch = _pair(224 if patch_size_px is None else patch_size_px, "patch_size_px")
        stride = (
            tuple(item / 2.0 for item in patch)
            if stride_px is None
            else _pair(stride_px, "stride_px")
        )
        if any(not float(item).is_integer() for item in patch + stride):
            raise ValueError("pixel patch geometry must contain integers")
    if any(step >= size for step, size in zip(stride, patch, strict=True)):
        raise ValueError("stride must be smaller than patch size to guarantee overlap")


def _geometry_evidence(
    *,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
) -> dict[str, Any]:
    if patch_size_um is not None or stride_um is not None:
        assert patch_size_um is not None
        patch = _pair(patch_size_um, "patch_size_um")
        stride = (
            tuple(item / 2.0 for item in patch)
            if stride_um is None
            else _pair(stride_um, "stride_um")
        )
        units = "micrometres"
    else:
        patch = _pair(224 if patch_size_px is None else patch_size_px, "patch_size_px")
        stride = (
            tuple(item / 2.0 for item in patch)
            if stride_px is None
            else _pair(stride_px, "stride_px")
        )
        units = "native_pixels"
    return {
        "patch_size_yx": list(patch),
        "stride_yx": list(stride),
        "units": units,
        "overlap_required": True,
    }


def _pixel_geometry(
    sample: QCSample,
    *,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
) -> tuple[tuple[int, int], tuple[int, int], str]:
    physical = patch_size_um is not None or stride_um is not None
    pixel = patch_size_px is not None or stride_px is not None
    if physical and pixel:
        raise ValueError("physical and pixel patch geometry are mutually exclusive")
    if physical:
        if patch_size_um is None:
            raise ValueError("patch_size_um is required when stride_um is supplied")
        patch_native = _pair(patch_size_um, "patch_size_um")
        stride_native = (
            tuple(item / 2.0 for item in patch_native)
            if stride_um is None
            else _pair(stride_um, "stride_um")
        )
        spacing = sample.image.pixel_size_um
        patch = tuple(
            max(1, round(size / resolution))
            for size, resolution in zip(patch_native, spacing, strict=True)
        )
        stride = tuple(
            max(1, round(size / resolution))
            for size, resolution in zip(stride_native, spacing, strict=True)
        )
        mode = "physical_um"
    else:
        patch_native = _pair(
            224 if patch_size_px is None else patch_size_px, "patch_size_px"
        )
        stride_native = (
            tuple(item / 2.0 for item in patch_native)
            if stride_px is None
            else _pair(stride_px, "stride_px")
        )
        if any(not float(item).is_integer() for item in patch_native + stride_native):
            raise ValueError("pixel patch geometry must contain integers")
        patch = tuple(int(item) for item in patch_native)
        stride = tuple(int(item) for item in stride_native)
        mode = "pixels"
    if any(step >= size for step, size in zip(stride, patch, strict=True)):
        raise ValueError("stride must be smaller than patch size to guarantee overlap")
    return patch, stride, mode  # type: ignore[return-value]


def _axis_starts(length: int, patch: int, stride: int) -> tuple[int, ...]:
    if length <= patch:
        return (0,)
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def _patches(
    shape: tuple[int, int], patch_size: tuple[int, int], stride: tuple[int, int]
) -> tuple[_Patch, ...]:
    height, width = shape
    patch_h, patch_w = min(patch_size[0], height), min(patch_size[1], width)
    top_values = _axis_starts(height, patch_h, min(stride[0], patch_h))
    left_values = _axis_starts(width, patch_w, min(stride[1], patch_w))
    return tuple(
        _Patch(top, left, top + patch_h, left + patch_w)
        for top in top_values
        for left in left_values
    )


def _encode(
    encoder: DINOv2FeatureExtractor,
    images: np.ndarray,
    *,
    batch_size: int,
) -> FoundationFeatures:
    features = encoder.encode(
        images,
        semantic_channels=("red", "green", "blue"),
        batch_size=batch_size,
    )
    if not isinstance(features, FoundationFeatures):
        raise TypeError("encoder.encode must return FoundationFeatures")
    if features.batch_size != int(images.shape[0]):
        raise ValueError("encoder returned a mismatched feature batch")
    if tuple(name.casefold() for name in features.semantic_channels) != (
        "red",
        "green",
        "blue",
    ):
        raise ValueError("encoder did not preserve declared RGB semantics")
    return features


def _token_validity(valid_crops: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    import cv2

    grid_h, grid_w = grid_shape
    coverage = np.empty((valid_crops.shape[0], grid_h, grid_w), dtype=np.float32)
    for index, mask in enumerate(valid_crops):
        coverage[index] = cv2.resize(
            mask.astype(np.float32),
            (grid_w, grid_h),
            interpolation=cv2.INTER_AREA,
        )
    return coverage


def _clean_feature_matrix(
    split: _LoadedSplit,
    encoder: DINOv2FeatureExtractor,
    *,
    batch_size: int,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
    min_valid_token_fraction: float,
    max_tokens: int | None,
) -> tuple[np.ndarray, int, str, int]:
    matrices: list[np.ndarray] = []
    total_patches = 0
    total_available_tokens = 0
    geometry_modes: set[str] = set()
    per_sample_budget = (
        None
        if max_tokens is None
        else max(1, math.ceil(max_tokens / max(len(split.samples), 1)))
    )
    for sample in split.samples:
        sample_matrices: list[np.ndarray] = []
        valid = _valid_mask(sample)
        patch_size, stride, mode = _pixel_geometry(
            sample,
            patch_size_px=patch_size_px,
            stride_px=stride_px,
            patch_size_um=patch_size_um,
            stride_um=stride_um,
        )
        geometry_modes.add(mode)
        coordinates = tuple(
            patch
            for patch in _patches(sample.spatial_shape, patch_size, stride)
            if np.any(valid[patch.top : patch.bottom, patch.left : patch.right])
        )
        batch_count = max(1, math.ceil(len(coordinates) / batch_size))
        per_batch_budget = (
            None
            if per_sample_budget is None
            else max(1, math.ceil(per_sample_budget / batch_count))
        )
        for start in range(0, len(coordinates), batch_size):
            batch_coordinates = coordinates[start : start + batch_size]
            images = np.stack(
                [
                    sample.image.data[
                        patch.top : patch.bottom, patch.left : patch.right, :
                    ]
                    for patch in batch_coordinates
                ],
                axis=0,
            )
            valid_crops = np.stack(
                [
                    valid[patch.top : patch.bottom, patch.left : patch.right]
                    for patch in batch_coordinates
                ],
                axis=0,
            )
            features = _encode(encoder, images, batch_size=batch_size)
            coverage = _token_validity(valid_crops, features.grid_shape)
            selected = features.patch_grid[coverage >= min_valid_token_fraction]
            if selected.size:
                selected = np.asarray(selected, dtype=np.float64)
                total_available_tokens += int(selected.shape[0])
                if (
                    per_batch_budget is not None
                    and selected.shape[0] > per_batch_budget
                ):
                    indices = np.linspace(
                        0,
                        selected.shape[0] - 1,
                        num=per_batch_budget,
                        dtype=np.int64,
                    )
                    selected = selected[indices]
                sample_matrices.append(selected)
            total_patches += len(batch_coordinates)
        if sample_matrices:
            sample_matrix = np.concatenate(sample_matrices, axis=0)
            if (
                per_sample_budget is not None
                and sample_matrix.shape[0] > per_sample_budget
            ):
                indices = np.linspace(
                    0,
                    sample_matrix.shape[0] - 1,
                    num=per_sample_budget,
                    dtype=np.int64,
                )
                sample_matrix = sample_matrix[indices]
            matrices.append(sample_matrix)
    if not matrices:
        raise FrozenBenchmarkValidationError(
            "no_valid_foundation_tokens", stage=split.role
        )
    matrix = np.concatenate(matrices, axis=0)
    if max_tokens is not None and matrix.shape[0] > max_tokens:
        indices = np.linspace(
            0,
            matrix.shape[0] - 1,
            num=max_tokens,
            dtype=np.int64,
        )
        matrix = matrix[indices]
    return matrix, total_patches, geometry_modes.pop(), total_available_tokens


def _score_sample(
    sample: QCSample,
    encoder: DINOv2FeatureExtractor,
    scorer: PatchKNNAnomalyScorer,
    *,
    batch_size: int,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
    min_valid_token_fraction: float,
) -> tuple[np.ndarray, int, str]:
    valid = _valid_mask(sample)
    patch_size, stride, mode = _pixel_geometry(
        sample,
        patch_size_px=patch_size_px,
        stride_px=stride_px,
        patch_size_um=patch_size_um,
        stride_um=stride_um,
    )
    coordinates = tuple(
        patch
        for patch in _patches(sample.spatial_shape, patch_size, stride)
        if np.any(valid[patch.top : patch.bottom, patch.left : patch.right])
    )
    accumulation = np.zeros(sample.spatial_shape, dtype=np.float64)
    weights = np.zeros(sample.spatial_shape, dtype=np.uint32)
    for start in range(0, len(coordinates), batch_size):
        batch_coordinates = coordinates[start : start + batch_size]
        images = np.stack(
            [
                sample.image.data[patch.top : patch.bottom, patch.left : patch.right, :]
                for patch in batch_coordinates
            ],
            axis=0,
        )
        valid_crops = np.stack(
            [
                valid[patch.top : patch.bottom, patch.left : patch.right]
                for patch in batch_coordinates
            ],
            axis=0,
        )
        features = _encode(encoder, images, batch_size=batch_size)
        token_support = (
            _token_validity(valid_crops, features.grid_shape)
            >= min_valid_token_fraction
        )
        raw_token_scores = np.asarray(
            scorer.raw_token_scores(features), dtype=np.float64
        )
        expected_tokens = len(batch_coordinates) * int(np.prod(features.grid_shape))
        if raw_token_scores.shape != (expected_tokens,):
            raise ValueError("scorer returned mismatched raw token scores")
        if not np.isfinite(raw_token_scores).all():
            raise ValueError("scorer returned non-finite raw token scores")
        token_score_maps = raw_token_scores.reshape(
            len(batch_coordinates), *features.grid_shape
        )
        import cv2

        for patch, token_score_map, token_mask, valid_crop in zip(
            batch_coordinates,
            token_score_maps,
            token_support,
            valid_crops,
            strict=True,
        ):
            support_weight = cv2.resize(
                token_mask.astype(np.float64),
                (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            numerator = cv2.resize(
                token_score_map * token_mask,
                (patch.shape[1], patch.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            patch_map = np.divide(
                numerator,
                support_weight,
                out=np.zeros_like(numerator),
                where=support_weight > np.finfo(np.float64).eps,
            )
            supported = support_weight > np.finfo(np.float64).eps
            supported &= valid_crop
            region = np.s_[patch.top : patch.bottom, patch.left : patch.right]
            accumulation[region] += np.where(supported, patch_map, 0.0)
            weights[region] += supported.astype(np.uint32)
    if np.any(valid & (weights == 0)):
        raise ValueError("valid pixels were not covered by extracted patches")
    scores = np.divide(
        accumulation,
        weights,
        out=np.zeros_like(accumulation),
        where=weights > 0,
    )
    return scores, len(coordinates), mode


def _stitched_calibration_scores(
    split: _LoadedSplit,
    encoder: DINOv2FeatureExtractor,
    scorer: PatchKNNAnomalyScorer,
    *,
    batch_size: int,
    patch_size_px: int | Sequence[int] | None,
    stride_px: int | Sequence[int] | None,
    patch_size_um: float | Sequence[float] | None,
    stride_um: float | Sequence[float] | None,
    min_valid_token_fraction: float,
    max_calibration_pixels: int,
) -> tuple[np.ndarray, int, str, int]:
    """Calibrate in the exact native stitched-pixel score domain used at test."""

    values: list[np.ndarray] = []
    total_patches = 0
    total_valid_pixels = 0
    modes: set[str] = set()
    per_sample_cap = max(
        2,
        math.ceil(max_calibration_pixels / max(len(split.samples), 1)),
    )
    for sample in split.samples:
        score_map, patch_count, mode = _score_sample(
            sample,
            encoder,
            scorer,
            batch_size=batch_size,
            patch_size_px=patch_size_px,
            stride_px=stride_px,
            patch_size_um=patch_size_um,
            stride_um=stride_um,
            min_valid_token_fraction=min_valid_token_fraction,
        )
        valid = _valid_mask(sample)
        sample_scores = np.asarray(score_map[valid], dtype=np.float64)
        total_valid_pixels += int(sample_scores.size)
        if sample_scores.size > per_sample_cap:
            indices = np.linspace(
                0,
                sample_scores.size - 1,
                num=per_sample_cap,
                dtype=np.int64,
            )
            sample_scores = sample_scores[indices]
        if sample_scores.size:
            values.append(sample_scores)
        total_patches += patch_count
        modes.add(mode)
    if not values:
        raise FrozenBenchmarkValidationError(
            "no_valid_calibration_pixels", stage="calibration"
        )
    combined = np.concatenate(values)
    if combined.size > max_calibration_pixels:
        indices = np.linspace(
            0,
            combined.size - 1,
            num=max_calibration_pixels,
            dtype=np.int64,
        )
        combined = combined[indices]
    return combined, total_patches, modes.pop(), total_valid_pixels


def _anonymous_group_labels(
    samples: Sequence[QCSample], field: str, prefix: str
) -> dict[str, str]:
    labels: dict[str, str] = {}
    for sample in samples:
        private_value = str(sample.metadata[field])
        if private_value not in labels:
            labels[private_value] = f"{prefix}_{len(labels) + 1:06d}"
    return labels


def _anonymous_correlation_clusters(
    samples: Sequence[QCSample],
) -> tuple[str, ...]:
    """Connect samples sharing any patient/block/slide/run identifier."""

    parents = list(range(len(samples)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        left, right = find(first), find(second)
        if left != right:
            parents[right] = left

    for field in GROUP_ID_FIELDS:
        first_seen: dict[str, int] = {}
        for index, sample in enumerate(samples):
            value = str(sample.metadata[field])
            if value in first_seen:
                union(index, first_seen[value])
            else:
                first_seen[value] = index
    root_labels: dict[int, str] = {}
    labels: list[str] = []
    for index in range(len(samples)):
        root = find(index)
        if root not in root_labels:
            root_labels[root] = f"locked_correlation_cluster_{len(root_labels) + 1:06d}"
        labels.append(root_labels[root])
    return tuple(labels)


def _safe_inference_reason(error: Exception) -> str:
    safe_types = {
        "ArithmeticError",
        "ImportError",
        "MemoryError",
        "RuntimeError",
        "TypeError",
        "ValueError",
    }
    name = type(error).__name__
    return f"inference_{name}" if name in safe_types else "inference_backend_error"


def _reference_support(samples: Sequence[QCSample]) -> dict[str, Any]:
    """Return path/identifier-free composition counts for interpretation gates."""

    def empty_counts() -> dict[str, Any]:
        return {
            "sample_count": 0,
            "positive_sample_count": 0,
            "negative_sample_count": 0,
            "positive_pixel_count": 0,
            "artifact_instance_count": 0,
            "patient_cluster_count": 0,
            "positive_patient_cluster_count": 0,
            "negative_patient_cluster_count": 0,
        }

    overall = empty_counts()
    cohorts: dict[str, dict[str, Any]] = {}
    overall_patients: set[str] = set()
    overall_positive_patients: set[str] = set()
    overall_negative_patients: set[str] = set()
    cohort_patients: dict[str, set[str]] = {}
    cohort_positive_patients: dict[str, set[str]] = {}
    cohort_negative_patients: dict[str, set[str]] = {}
    import cv2

    for sample in samples:
        cohort = str(sample.metadata["cohort"])
        patient = str(sample.metadata["patient_id"])
        valid = _valid_mask(sample)
        target = (
            np.asarray(sample.masks["fold"], dtype=bool)
            | np.asarray(sample.masks["crack"], dtype=bool)
        ) & valid
        positive = bool(np.any(target))
        _, labels = cv2.connectedComponents(target.astype(np.uint8), connectivity=8)
        instance_count = int(np.max(labels)) if labels.size else 0
        counts = cohorts.setdefault(cohort, empty_counts())
        for destination in (overall, counts):
            destination["sample_count"] += 1
            destination[
                "positive_sample_count" if positive else "negative_sample_count"
            ] += 1
            destination["positive_pixel_count"] += int(np.count_nonzero(target))
            destination["artifact_instance_count"] += instance_count
        overall_patients.add(patient)
        cohort_patients.setdefault(cohort, set()).add(patient)
        if positive:
            overall_positive_patients.add(patient)
            cohort_positive_patients.setdefault(cohort, set()).add(patient)
        else:
            overall_negative_patients.add(patient)
            cohort_negative_patients.setdefault(cohort, set()).add(patient)

    overall["patient_cluster_count"] = len(overall_patients)
    overall["positive_patient_cluster_count"] = len(overall_positive_patients)
    overall["negative_patient_cluster_count"] = len(overall_negative_patients)
    for cohort, counts in cohorts.items():
        counts["patient_cluster_count"] = len(cohort_patients.get(cohort, set()))
        counts["positive_patient_cluster_count"] = len(
            cohort_positive_patients.get(cohort, set())
        )
        counts["negative_patient_cluster_count"] = len(
            cohort_negative_patients.get(cohort, set())
        )
    return {"overall": overall, "by_cohort": cohorts}


def run_frozen_anomaly_benchmark(
    fit_manifest: str | Path,
    calibration_manifest: str | Path,
    locked_test_manifest: str | Path,
    *,
    encoder: DINOv2FeatureExtractor,
    scorer: PatchKNNAnomalyScorer | None = None,
    patch_size_px: int | Sequence[int] | None = None,
    stride_px: int | Sequence[int] | None = None,
    patch_size_um: float | Sequence[float] | None = None,
    stride_um: float | Sequence[float] | None = None,
    batch_size: int = 8,
    min_valid_token_fraction: float = 0.5,
    neighbors: int = 1,
    calibration_quantile: float = 0.995,
    max_reference_tokens: int | None = 100_000,
    max_calibration_pixels: int = 1_000_000,
    max_raster_pixels: int = 25_000_000,
    n_resamples: int = 2_000,
    bootstrap_seed: int = 0,
    minimum_positive_test_samples: int = 1,
    minimum_negative_test_samples: int = 1,
    minimum_test_patient_clusters: int = 2,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Run a leakage-resistant, H&E-only frozen anomaly localization benchmark.

    ``fit_manifest`` and ``calibration_manifest`` must contain explicit,
    adjudicated all-zero fold and crack masks.  Missing labels are rejected
    instead of being interpreted as negatives.  ``locked_test_manifest`` must
    contain both masks; evaluation uses their union and therefore supports only
    the claim "artifact anomaly evidence", not fold/crack subtype prediction.

    Patch geometry may be declared in native pixels or physical micrometres.
    Omitting both selects 224-pixel patches with 50% overlap.  The encoder is
    always called in batches no larger than ``batch_size``.
    """

    if encoder is None:
        raise ValueError("encoder must be injected")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    min_fraction = float(min_valid_token_fraction)
    if not math.isfinite(min_fraction) or not 0.0 < min_fraction <= 1.0:
        raise ValueError("min_valid_token_fraction must lie in (0, 1]")
    if int(n_resamples) != n_resamples or n_resamples <= 0:
        raise ValueError("n_resamples must be a positive integer")
    for name, value in (
        ("minimum_positive_test_samples", minimum_positive_test_samples),
        ("minimum_negative_test_samples", minimum_negative_test_samples),
        ("minimum_test_patient_clusters", minimum_test_patient_clusters),
        ("max_calibration_pixels", max_calibration_pixels),
        ("max_raster_pixels", max_raster_pixels),
    ):
        if int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    _validate_geometry_request(
        patch_size_px=patch_size_px,
        stride_px=stride_px,
        patch_size_um=patch_size_um,
        stride_um=stride_um,
    )

    splits = (
        _load_split(fit_manifest, "fit"),
        _load_split(calibration_manifest, "calibration"),
        _load_split(locked_test_manifest, "locked_test"),
    )
    for split in splits:
        for sample in split.samples:
            _validate_sample_contract(sample, split.role)
            if int(np.prod(sample.spatial_shape)) > int(max_raster_pixels):
                raise FrozenBenchmarkValidationError(
                    "preextracted_raster_size_limit_exceeded",
                    stage=split.role,
                )
    disjointness = _prove_disjointness(splits)
    reference_support = _reference_support(splits[2].samples)
    overall_support = reference_support["overall"]
    support_gate = {
        "minimum_positive_test_samples": int(minimum_positive_test_samples),
        "minimum_negative_test_samples": int(minimum_negative_test_samples),
        "minimum_test_patient_clusters": int(minimum_test_patient_clusters),
    }
    support_gate_passed = (
        int(overall_support["positive_sample_count"])
        >= int(minimum_positive_test_samples)
        and int(overall_support["negative_sample_count"])
        >= int(minimum_negative_test_samples)
        and int(overall_support["patient_cluster_count"])
        >= int(minimum_test_patient_clusters)
    )
    geometry_evidence = _geometry_evidence(
        patch_size_px=patch_size_px,
        stride_px=stride_px,
        patch_size_um=patch_size_um,
        stride_um=stride_um,
    )

    active_scorer = scorer or PatchKNNAnomalyScorer(
        neighbors=neighbors,
        calibration_quantile=calibration_quantile,
        max_reference_tokens=max_reference_tokens,
    )
    try:
        (
            fit_features,
            fit_patch_count,
            geometry_mode,
            fit_available_token_count,
        ) = _clean_feature_matrix(
            splits[0],
            encoder,
            batch_size=batch_size,
            patch_size_px=patch_size_px,
            stride_px=stride_px,
            patch_size_um=patch_size_um,
            stride_um=stride_um,
            min_valid_token_fraction=min_fraction,
            max_tokens=max_reference_tokens,
        )
        active_scorer.fit(
            fit_features,
            split_id=f"fit:{splits[0].manifest_sha256}",
        )
        (
            calibration_scores,
            calibration_patch_count,
            calibration_mode,
            calibration_available_pixel_count,
        ) = _stitched_calibration_scores(
            splits[1],
            encoder,
            active_scorer,
            batch_size=batch_size,
            patch_size_px=patch_size_px,
            stride_px=stride_px,
            patch_size_um=patch_size_um,
            stride_um=stride_um,
            min_valid_token_fraction=min_fraction,
            max_calibration_pixels=int(max_calibration_pixels),
        )
        if calibration_mode != geometry_mode:
            raise ValueError("inconsistent geometry mode")
    except FrozenBenchmarkValidationError:
        raise
    # Optional encoder/scorer backends can raise backend-specific exception
    # classes.  Sanitize all of them at this private-data boundary.
    except Exception:  # noqa: BLE001
        raise FrozenBenchmarkValidationError(
            "fit_or_calibration_failure", stage="model_setup"
        ) from None

    scorer_quantile = float(
        getattr(active_scorer, "calibration_quantile", calibration_quantile)
    )
    if not 0.5 < scorer_quantile < 1.0:
        raise FrozenBenchmarkValidationError(
            "calibration_quantile_invalid", stage="calibration"
        )
    if calibration_scores.size < 2:
        raise FrozenBenchmarkValidationError(
            "insufficient_calibration_pixels", stage="calibration"
        )
    threshold = float(np.quantile(calibration_scores, scorer_quantile))
    if not math.isfinite(threshold):
        raise FrozenBenchmarkValidationError(
            "calibration_threshold_invalid", stage="calibration"
        )

    evaluations: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    anonymous_slides = _anonymous_group_labels(
        splits[2].samples, "slide_id", "locked_slide"
    )
    correlation_clusters = _anonymous_correlation_clusters(splits[2].samples)
    for index, sample in enumerate(splits[2].samples, start=1):
        anonymous_key = f"locked_test_{index:06d}"
        anonymous_slide = anonymous_slides[str(sample.metadata["slide_id"])]
        anonymous_cluster = correlation_clusters[index - 1]
        started = perf_counter()
        try:
            score_map, patch_count, sample_mode = _score_sample(
                sample,
                encoder,
                active_scorer,
                batch_size=batch_size,
                patch_size_px=patch_size_px,
                stride_px=stride_px,
                patch_size_um=patch_size_um,
                stride_um=stride_um,
                min_valid_token_fraction=min_fraction,
            )
            if sample_mode != geometry_mode:
                raise ValueError("inconsistent geometry mode")
            target = np.asarray(sample.masks["fold"], dtype=bool) | np.asarray(
                sample.masks["crack"], dtype=bool
            )
            valid = _valid_mask(sample)
            prediction = score_map > threshold
            runtime = perf_counter() - started
            evaluation = evaluate_sample(
                target,
                prediction,
                score_map=score_map,
                threshold=threshold,
                sample_id=anonymous_key,
                slide_id=anonymous_slide,
                modality="he",
                valid_mask=valid,
                spacing=sample.image.pixel_size_um,
                runtime_seconds=runtime,
                metadata={
                    "status": "evaluated",
                    "reference_target": "artifact_union",
                    "prediction_semantics": "anomaly_evidence",
                    "semantic_subtype_claim": False,
                    "patch_count": patch_count,
                    "bootstrap_cluster": anonymous_cluster,
                    "cohort": str(sample.metadata["cohort"]),
                },
            )
            evaluations.append(evaluation)
            outcomes.append(
                {
                    "sample_key": anonymous_key,
                    "status": "evaluated",
                    "reason_code": None,
                    "runtime_seconds": runtime,
                }
            )
        # A locked benchmark must account for every test row even when an
        # optional backend raises an implementation-specific exception.
        except Exception as error:  # noqa: BLE001
            outcomes.append(
                {
                    "sample_key": anonymous_key,
                    "status": "abstained",
                    "reason_code": _safe_inference_reason(error),
                    "runtime_seconds": perf_counter() - started,
                }
            )

    evaluation_report = None
    evaluation_by_cohort: dict[str, Any] = {}
    if evaluations:
        evaluation_report = build_report(
            evaluations,
            n_resamples=int(n_resamples),
            seed=int(bootstrap_seed),
            bootstrap_cluster_key="metadata.bootstrap_cluster",
        )
        cohort_names = sorted({str(item["metadata"]["cohort"]) for item in evaluations})
        for cohort_name in cohort_names:
            cohort_results = [
                item
                for item in evaluations
                if str(item["metadata"]["cohort"]) == cohort_name
            ]
            evaluation_by_cohort[cohort_name] = build_report(
                cohort_results,
                n_resamples=int(n_resamples),
                seed=int(bootstrap_seed),
                bootstrap_cluster_key="metadata.bootstrap_cluster",
            )

    abstained_count = len(outcomes) - len(evaluations)
    run_complete = abstained_count == 0
    metric_report_complete = run_complete and support_gate_passed
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "benchmark": "he_frozen_patch_anomaly_localization",
        "run_status": "complete" if run_complete else "incomplete_abstentions",
        "report_eligible": False,
        "scientific_report_eligible": False,
        "development_metric_report_complete": metric_report_complete,
        "evidence_boundary": {
            "claim_status": "real_data_development_evaluation_not_scientific_benchmark",
            "modality": "he",
            "reference_target": "artifact_union",
            "prediction_semantics": "anomaly_evidence",
            "input_scope": "preextracted_roi_or_downsampled_raster_not_native_wsi",
            "max_raster_pixels": int(max_raster_pixels),
            "semantic_fold_crack_claim": False,
            "synthetic_provenance_rejected": True,
            "positive_real_acquisition_provenance_required": True,
            "strict_manifest_validation": True,
            "private_paths_or_ids_serialized": False,
            "pooled_evaluation_acceptance_eligible": False,
            "primary_stratification": "cohort",
            "bootstrap_cluster_policy": (
                "connected_components_over_patient_block_slide_run"
            ),
            "efficacy_denominator_complete": run_complete,
            "benchmark_contract_consumed": False,
            "model_provenance_locked": False,
            "manifests": {
                split.role: {
                    "sha256": split.manifest_sha256,
                    "sample_count": len(split.samples),
                    "declared_role": split.role,
                    "clean_adjudication_assertion_required": split.role
                    in {"fit", "calibration"},
                }
                for split in splits
            },
            "split_disjointness": disjointness,
        },
        "method": {
            "family": "frozen_foundation_patch_knn",
            "encoder_class": type(encoder).__name__,
            "scorer_class": type(active_scorer).__name__,
            "geometry_mode": geometry_mode,
            "patch_geometry": geometry_evidence,
            "batch_size_bound": int(batch_size),
            "fit_patch_count": fit_patch_count,
            "calibration_patch_count": calibration_patch_count,
            "fit_token_count": int(fit_features.shape[0]),
            "fit_available_token_count_before_deterministic_cap": int(
                fit_available_token_count
            ),
            "calibration_valid_pixel_count": int(calibration_scores.size),
            "calibration_available_valid_pixel_count_before_cap": int(
                calibration_available_pixel_count
            ),
            "calibration_sampling": "deterministic_equal_per_sample_cap",
            "min_valid_token_fraction": min_fraction,
        },
        "calibration": {
            "threshold": threshold,
            "score_domain": "native_stitched_valid_pixel_anomaly_distance",
            "operating_target": "nominal_valid_pixel_score_quantile",
            "threshold_source": (
                "independent_reviewed_clean_calibration_split_identical_stitch_path"
            ),
            "calibration_quantile": scorer_quantile,
            "test_labels_used_for_threshold": False,
        },
        "outcome_summary": {
            "test_sample_count": len(splits[2].samples),
            "evaluated_count": len(evaluations),
            "abstained_count": abstained_count,
        },
        "reference_support": {
            **reference_support,
            "minimum_gate": support_gate,
            "minimum_gate_passed": support_gate_passed,
            "note": (
                "This is a minimal interpretability gate, not a statistical power "
                "or acceptance calculation."
            ),
        },
        "outcomes": outcomes,
        "evaluation": evaluation_report,
        "evaluation_by_cohort": evaluation_by_cohort,
    }
    if output_json is not None:
        write_json_report(report, output_json)
    return report
