"""Leakage-safe benchmark for the public Histology Tissue Fold Dataset v1.0.

The dataset contains real microscope H&E fields labelled ``clean`` or
``tissue_fold`` and pixel masks for the fold class.  It does *not* contain a
crack/tear reference class; this module consequently makes no crack claim.

The runner is intentionally independent of the command-line layer.  A caller
injects an already loaded frozen encoder, making model provenance and device
selection explicit.  Dataset identities are assigned to fit, calibration and
locked-test partitions at the source-slide level.  Thresholds are selected
only from calibration observations and are then applied unchanged to test.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Protocol
from xml.etree import ElementTree

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp, minimize
from scipy.stats import rankdata

from .detectors import classical_fold_candidates
from .foundation import FoundationFeatures, PatchKNNAnomalyScorer

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

__all__ = [
    "PUBLIC_FOLD_METHODS",
    "PublicFoldBenchmarkConfig",
    "PublicFoldDataset",
    "PublicFoldRecord",
    "PublicFoldValidationError",
    "build_public_fold_splits",
    "load_public_fold_dataset",
    "run_public_fold_benchmark",
]


PUBLIC_FOLD_METHODS = (
    "classical_fold",
    "foundation_patchknn",
    "foundation_linear_probe",
    "dinov2_patchknn",
    "dinov2_linear_probe",
)
_DEFAULT_PUBLIC_FOLD_METHODS = (
    "classical_fold",
    "dinov2_patchknn",
    "dinov2_linear_probe",
)
_FOUNDATION_HEADS: Mapping[str, str] = {
    "foundation_patchknn": "patchknn",
    "dinov2_patchknn": "patchknn",
    "foundation_linear_probe": "linear_probe",
    "dinov2_linear_probe": "linear_probe",
}
_FOUNDATION_METHODS = frozenset(_FOUNDATION_HEADS)
_METHOD_SEED_INDEX: Mapping[str, int] = {
    "classical_fold": 0,
    "foundation_patchknn": 1,
    "dinov2_patchknn": 1,
    "foundation_linear_probe": 2,
    "dinov2_linear_probe": 2,
}
_CLASSES = frozenset(("clean", "tissue_fold"))
_PUBLIC_V1_COUNTS: Mapping[tuple[str, str], int] = {
    ("Brain", "clean"): 215,
    ("Brain", "tissue_fold"): 218,
    ("Kidney", "clean"): 181,
    ("Kidney", "tissue_fold"): 285,
    ("Liver", "clean"): 167,
    ("Liver", "tissue_fold"): 270,
    ("Small_Intestine", "clean"): 152,
    ("Small_Intestine", "tissue_fold"): 213,
    ("Testis", "clean"): 184,
    ("Testis", "tissue_fold"): 242,
}
_PUBLIC_V1_RELEASE_IDENTITY_VERSION = "histology-tissue-fold-v1.0-2026-08-26"
_PUBLIC_V1_RELEASE_IDENTITY: Mapping[str, str] = {
    "metadata_sha256": "101ca59ad4505db673253d370698b285f15342c77f590eeee65b0935357b72d4",
    "slide_mapping_sha256": "d3199c431771c8d87ac1d35f178208d1207769a9a048bd748b84808836169a40",
    "license_sha256": "866d89cbf299323640d2ff76a5695e9813fded3a8aeed676c260583763767f17",
    "source_readme_sha256": "6e69e809522c880f093bb8c674351211f939969f594ea8658e64df674371d73f",
    "asset_manifest_sha256": "826202d9951415ea5ffeafe2648b192bccc25f02ad0c3617b3be29bc9a5ab328",
    "localization_exclusion_manifest_sha256": "2002f53e1beb42f8743169d0d023f385b4d7a3cb943d972c5e7a13bb1bf57926",
}
_RUN_PROVENANCE_SCHEMA_VERSION = "public-fold-run-provenance-1.1"
_PUBLIC_FOLD_REPORT_SCHEMA_VERSION = "public-fold-benchmark-1.2"
_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


class PublicFoldValidationError(ValueError):
    """Failure at the public-dataset or benchmark evidence boundary."""

    def __init__(self, code: str, detail: str):
        self.code = str(code)
        self.detail = str(detail)
        super().__init__(f"Public fold benchmark validation failed [{code}]: {detail}")


class FrozenEncoder(Protocol):
    """Minimal injected-encoder interface used by the benchmark."""

    def encode(
        self,
        images: Any,
        *,
        semantic_channels: Sequence[str],
        batch_size: int,
    ) -> FoundationFeatures: ...


@dataclass(frozen=True)
class PublicFoldRecord:
    """One metadata row joined to its source-slide identity and assets."""

    image_filename: str
    organ: str
    class_name: str
    slide_id: str
    image_path: Path
    mask_path: Path | None
    localization_reference_valid: bool = True
    image_sha256: str | None = None
    mask_sha256: str | None = None

    @property
    def is_fold(self) -> bool:
        return self.class_name == "tissue_fold"

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "image_filename": self.image_filename,
            "organ": self.organ,
            "class": self.class_name,
            "slide_id": self.slide_id,
            "localization_reference_valid": self.localization_reference_valid,
            "image_sha256": self.image_sha256,
            "mask_sha256": self.mask_sha256,
        }


@dataclass(frozen=True)
class PublicFoldDataset:
    """Validated records plus immutable provenance evidence."""

    root: Path
    records: tuple[PublicFoldRecord, ...]
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PublicFoldBenchmarkConfig:
    """Reproducible compute, split, calibration and reporting controls."""

    methods: tuple[str, ...] = _DEFAULT_PUBLIC_FOLD_METHODS
    seed: int = 20260826
    fit_fraction: float = 0.60
    calibration_fraction: float = 0.20
    test_fraction: float = 0.20
    max_dimension: int = 896
    tile_size: int = 224
    tile_stride: int = 224
    classical_min_component_size: int = 8
    encoder_batch_size: int = 8
    max_reference_tokens: int = 4096
    max_probe_tokens_per_class: int = 8192
    patchknn_neighbors: int = 3
    patchknn_distance_chunk_size: int = 256
    probe_l2: float = 1e-3
    probe_max_iterations: int = 100
    token_positive_fraction: float = 0.05
    image_score_quantile: float = 0.995
    threshold_candidates: int = 96
    calibration_score_sample: int = 250_000
    bootstrap_resamples: int = 500
    bootstrap_confidence: float = 0.95
    limit_slides_per_stratum_per_split: int | None = None
    strict_public_v1: bool = True
    validate_asset_dimensions: bool = True
    hash_assets: bool = True
    empty_positive_mask_policy: str = "exclude_localization"

    def __post_init__(self) -> None:
        methods = tuple(dict.fromkeys(str(item) for item in self.methods))
        unknown = sorted(set(methods) - set(PUBLIC_FOLD_METHODS))
        if not methods or unknown:
            raise ValueError(
                f"methods must be non-empty and supported; unknown={unknown}"
            )
        fractions = (
            float(self.fit_fraction),
            float(self.calibration_fraction),
            float(self.test_fraction),
        )
        if any(item <= 0 for item in fractions) or not math.isclose(
            sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "fit/calibration/test fractions must be positive and sum to 1"
            )
        positive_integers = {
            "max_dimension": self.max_dimension,
            "tile_size": self.tile_size,
            "tile_stride": self.tile_stride,
            "classical_min_component_size": self.classical_min_component_size,
            "encoder_batch_size": self.encoder_batch_size,
            "max_reference_tokens": self.max_reference_tokens,
            "max_probe_tokens_per_class": self.max_probe_tokens_per_class,
            "patchknn_neighbors": self.patchknn_neighbors,
            "patchknn_distance_chunk_size": self.patchknn_distance_chunk_size,
            "probe_max_iterations": self.probe_max_iterations,
            "threshold_candidates": self.threshold_candidates,
            "calibration_score_sample": self.calibration_score_sample,
        }
        if any(int(value) <= 0 for value in positive_integers.values()):
            raise ValueError(
                "all size, sample, neighbor and iteration controls must be positive"
            )
        if self.tile_stride > self.tile_size:
            raise ValueError(
                "tile_stride cannot exceed tile_size; every pixel needs support"
            )
        if not 0.0 < self.token_positive_fraction <= 1.0:
            raise ValueError("token_positive_fraction must lie in (0,1]")
        if not 0.5 <= self.image_score_quantile <= 1.0:
            raise ValueError("image_score_quantile must lie in [0.5,1]")
        if not math.isfinite(self.probe_l2) or self.probe_l2 < 0:
            raise ValueError("probe_l2 must be finite and non-negative")
        if self.bootstrap_resamples < 0:
            raise ValueError("bootstrap_resamples must be non-negative")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must lie in (0,1)")
        if (
            self.limit_slides_per_stratum_per_split is not None
            and self.limit_slides_per_stratum_per_split <= 0
        ):
            raise ValueError("limit_slides_per_stratum_per_split must be positive")
        if self.empty_positive_mask_policy not in {"error", "exclude_localization"}:
            raise ValueError(
                "empty_positive_mask_policy must be 'error' or 'exclude_localization'"
            )
        object.__setattr__(self, "methods", methods)

    def as_dict(self) -> dict[str, Any]:
        return {
            key: (list(value) if isinstance(value, tuple) else value)
            for key, value in vars(self).items()
        }


@dataclass(frozen=True)
class _LoadedImage:
    image: NDArray[np.uint8]
    target: BoolArray
    original_shape: tuple[int, int]


@dataclass(frozen=True)
class _Scored:
    record: PublicFoldRecord
    score: FloatArray
    valid: BoolArray
    target: BoolArray
    localization_reference_valid: bool
    image_score: float
    runtime_seconds: float


@dataclass(frozen=True)
class _Result:
    field_key: str
    organ: str
    slide_id: str
    label: int
    localization_reference_valid: bool
    tp: int
    fp: int
    fn: int
    tn: int
    n_valid: int
    image_score: float
    image_prediction: int
    runtime_seconds: float


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_hex_digest(value: Any, length: int) -> bool:
    text = str(value or "")
    return len(text) == length and all(
        character in "0123456789abcdef" for character in text
    )


def _verify_public_v1_release_identity(
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare all canonical release components with the audited v1 identity."""

    mismatches = sorted(
        key
        for key, expected in _PUBLIC_V1_RELEASE_IDENTITY.items()
        if observed.get(key) != expected
    )
    if mismatches:
        raise PublicFoldValidationError(
            "public_v1_release_identity_mismatch",
            "canonical release components differ: " + ", ".join(mismatches),
        )
    identity = {key: str(observed[key]) for key in _PUBLIC_V1_RELEASE_IDENTITY}
    return {
        "identity_version": _PUBLIC_V1_RELEASE_IDENTITY_VERSION,
        "verified": True,
        "verified_components": sorted(identity),
        "canonical_identity_sha256": _canonical_sha256(identity),
        "identity": identity,
    }


def _validate_run_provenance(
    value: Mapping[str, Any] | None,
    config: PublicFoldBenchmarkConfig,
) -> dict[str, Any]:
    """Validate caller-captured execution identity before any benchmark scoring.

    Minimal JSON schema (all named fields are required):

    * ``schema_version``: ``public-fold-run-provenance-1.1``;
    * ``capture``: pre-scoring boolean, structural-validation status, and validator ID;
    * ``code``: Git commit plus dirty-diff SHA-256, or packaged-wheel SHA-256;
    * ``environment``: Python/platform and relevant dependency versions;
    * ``method_model``: selected methods, benchmark configuration SHA-256,
      implementation/model/config/weights/loader identities, and frozen plus
      non-transductive assertions; and
    * ``execution``: device and precision.

    Validation is deliberately structural and consistency-based. It does not
    discover Git, packages, devices, or model files after execution; callers
    must capture and approve those identities before invoking the runner.
    """

    audit: dict[str, Any] = {
        "schema_version": _RUN_PROVENANCE_SCHEMA_VERSION,
        "provided": value is not None,
        "validated_before_scoring": True,
        "valid": False,
        "validation_errors": [],
        "identity_sha256": None,
        "value": None,
    }
    if value is None:
        audit["validation_errors"] = ["run_provenance_absent"]
        return audit
    if not isinstance(value, Mapping):
        audit["validation_errors"] = ["run_provenance_not_mapping"]
        return audit
    try:
        normalized = json.loads(
            json.dumps(dict(value), sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError):
        audit["validation_errors"] = ["run_provenance_not_strict_json"]
        return audit
    if not isinstance(normalized, dict):
        audit["validation_errors"] = ["run_provenance_not_object"]
        return audit

    audit["value"] = normalized
    audit["identity_sha256"] = _canonical_sha256(normalized)
    errors: list[str] = []
    if normalized.get("schema_version") != _RUN_PROVENANCE_SCHEMA_VERSION:
        errors.append("schema_version_invalid")

    capture = normalized.get("capture")
    if not isinstance(capture, dict):
        errors.append("capture_missing")
    else:
        if capture.get("captured_before_scoring") is not True:
            errors.append("capture_not_pre_scoring")
        if capture.get("validation_status") != "structurally_validated":
            errors.append("capture_not_structurally_validated")
        if not str(capture.get("validator_id", "")).strip():
            errors.append("capture_validator_id_missing")

    code = normalized.get("code")
    if not isinstance(code, dict):
        errors.append("code_identity_missing")
    else:
        identity_type = code.get("identity_type")
        if identity_type == "git":
            if not _is_hex_digest(code.get("commit"), 40):
                errors.append("code_git_commit_invalid")
            if not _is_hex_digest(code.get("dirty_diff_sha256"), 64):
                errors.append("code_dirty_diff_sha256_invalid")
        elif identity_type == "wheel":
            if not _is_hex_digest(code.get("wheel_sha256"), 64):
                errors.append("code_wheel_sha256_invalid")
        else:
            errors.append("code_identity_type_invalid")

    environment = normalized.get("environment")
    if not isinstance(environment, dict):
        errors.append("environment_identity_missing")
    else:
        if not str(environment.get("python_version", "")).strip():
            errors.append("environment_python_version_missing")
        if not str(environment.get("platform", "")).strip():
            errors.append("environment_platform_missing")
        dependencies = environment.get("dependencies")
        if not isinstance(dependencies, dict):
            errors.append("environment_dependencies_missing")
        else:
            for dependency in ("numpy", "scipy", "opencv"):
                if not str(dependencies.get(dependency, "")).strip():
                    errors.append(f"environment_dependency_{dependency}_missing")
            if any(method != "classical_fold" for method in config.methods):
                for dependency in ("torch", "transformers", "huggingface_hub"):
                    if not str(dependencies.get(dependency, "")).strip():
                        errors.append(f"environment_dependency_{dependency}_missing")

    method_model = normalized.get("method_model")
    if not isinstance(method_model, dict):
        errors.append("method_model_identity_missing")
    else:
        selected_methods = method_model.get("selected_methods")
        if not isinstance(selected_methods, list) or tuple(selected_methods) != tuple(
            config.methods
        ):
            errors.append("method_model_selected_methods_mismatch")
        expected_config_hash = _canonical_sha256(config.as_dict())
        if method_model.get("benchmark_configuration_sha256") != expected_config_hash:
            errors.append("method_model_benchmark_configuration_mismatch")
        for field in ("implementation_id", "model_id", "loader_identity"):
            if not str(method_model.get(field, "")).strip():
                errors.append(f"method_model_{field}_missing")
        if not _is_hex_digest(method_model.get("model_config_sha256"), 64):
            errors.append("method_model_model_config_sha256_invalid")
        if method_model.get("frozen_evaluation") is not True:
            errors.append("method_model_not_frozen_evaluation")
        if method_model.get("transductive_updates") is not False:
            errors.append("method_model_transductive_updates_not_disabled")
        foundation_requested = bool(set(config.methods) & _FOUNDATION_METHODS)
        weights_not_applicable = method_model.get("weights_not_applicable")
        weights_sha256 = method_model.get("weights_sha256")
        if foundation_requested:
            if weights_not_applicable is not False or not _is_hex_digest(
                weights_sha256, 64
            ):
                errors.append("method_model_foundation_weights_identity_invalid")
        elif weights_not_applicable is not True or weights_sha256 not in (None, ""):
            errors.append("method_model_classical_weights_identity_invalid")

    execution = normalized.get("execution")
    if not isinstance(execution, dict):
        errors.append("execution_identity_missing")
    else:
        if not str(execution.get("device", "")).strip():
            errors.append("execution_device_missing")
        if not str(execution.get("precision", "")).strip():
            errors.append("execution_precision_missing")

    audit["validation_errors"] = sorted(set(errors))
    audit["valid"] = not errors
    return audit


def _safe_relative_path(raw: str, *, field: str) -> PurePosixPath:
    value = PurePosixPath(str(raw).strip().replace("\\", "/"))
    if not str(value) or value.is_absolute() or ".." in value.parts:
        raise PublicFoldValidationError("unsafe_relative_path", field)
    return value


def _xlsx_cell_value(cell: ElementTree.Element, shared_strings: Sequence[str]) -> str:
    cell_type = cell.attrib.get("t", "")
    if cell_type == "inlineStr":
        node = cell.find(f"{_XLSX_NS}is/{_XLSX_NS}t")
        return "" if node is None or node.text is None else node.text
    node = cell.find(f"{_XLSX_NS}v")
    value = "" if node is None or node.text is None else node.text
    if cell_type == "s" and value:
        try:
            return shared_strings[int(value)]
        except (IndexError, ValueError):
            raise PublicFoldValidationError(
                "mapping_xlsx_invalid", "shared-string index is invalid"
            ) from None
    return value


def _read_xlsx_rows(path: Path) -> list[dict[str, str]]:
    """Read the simple source mapping workbook without a pandas dependency."""

    try:
        with zipfile.ZipFile(path) as archive:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
                for item in root.findall(f"{_XLSX_NS}si"):
                    shared.append(
                        "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
                    )
            sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    except (OSError, KeyError, zipfile.BadZipFile, ElementTree.ParseError):
        raise PublicFoldValidationError(
            "mapping_xlsx_invalid", "slide_image_mapping.xlsx is unreadable"
        ) from None
    values: list[list[str]] = []
    for row in sheet.findall(f"{_XLSX_NS}sheetData/{_XLSX_NS}row"):
        cells: dict[int, str] = {}
        for cell in row.findall(f"{_XLSX_NS}c"):
            reference = cell.attrib.get("r", "A")
            letters = "".join(
                character for character in reference if character.isalpha()
            )
            column = 0
            for letter in letters.upper():
                column = column * 26 + ord(letter) - ord("A") + 1
            cells[column - 1] = _xlsx_cell_value(cell, shared)
        if cells:
            width = max(cells) + 1
            values.append([cells.get(index, "") for index in range(width)])
    if not values:
        raise PublicFoldValidationError("mapping_xlsx_invalid", "mapping is empty")
    header = [item.strip() for item in values[0]]
    required = ("image_filename", "organ", "class", "slide_id")
    if any(name not in header for name in required):
        raise PublicFoldValidationError(
            "mapping_columns_missing", f"required columns are {required}"
        )
    return [
        {
            name: (row[index].strip() if index < len(row) else "")
            for index, name in enumerate(header)
        }
        for row in values[1:]
        if any(item.strip() for item in row)
    ]


def _load_metadata(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, UnicodeError, csv.Error):
        raise PublicFoldValidationError(
            "metadata_invalid", "metadata.csv is unreadable"
        ) from None
    required = {
        "dataset_version",
        "organ",
        "class",
        "image_filename",
        "image_relative_path",
        "mask_available",
        "mask_filename",
        "mask_relative_path",
        "pairing_status",
    }
    if not rows or not required.issubset(rows[0]):
        raise PublicFoldValidationError(
            "metadata_columns_missing", f"required columns are {sorted(required)}"
        )
    return [
        {str(key): str(value or "").strip() for key, value in row.items()}
        for row in rows
    ]


def _inspect_dimensions(
    image_path: Path, mask_path: Path | None
) -> tuple[int, int, bool]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[-1] != 3:
        raise PublicFoldValidationError("image_decode_failed", image_path.name)
    shape = (int(image.shape[0]), int(image.shape[1]))
    if mask_path is not None:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise PublicFoldValidationError("mask_decode_failed", mask_path.name)
        if tuple(mask.shape) != shape:
            raise PublicFoldValidationError(
                "image_mask_shape_mismatch", image_path.name
            )
        unique = {int(item) for item in np.unique(mask)}
        if not unique.issubset({0, 1, 255}):
            raise PublicFoldValidationError("mask_not_binary", mask_path.name)
        return shape[0], shape[1], bool(np.any(mask > 0))
    return shape[0], shape[1], True


def load_public_fold_dataset(
    root_value: str | Path,
    *,
    strict_public_v1: bool = True,
    validate_asset_dimensions: bool = True,
    hash_assets: bool = True,
    empty_positive_mask_policy: str = "error",
) -> PublicFoldDataset:
    """Validate metadata, source-slide mapping, assets, counts and provenance."""

    if empty_positive_mask_policy not in {"error", "exclude_localization"}:
        raise ValueError(
            "empty_positive_mask_policy must be 'error' or 'exclude_localization'"
        )
    if strict_public_v1 and not validate_asset_dimensions:
        raise PublicFoldValidationError(
            "strict_public_v1_full_validation_required",
            "strict v1 requires image/mask decode, dimension, and binary-mask validation",
        )
    if strict_public_v1 and not hash_assets:
        raise PublicFoldValidationError(
            "strict_public_v1_asset_hashes_required",
            "strict v1 requires SHA-256 for every image and mask asset",
        )
    if strict_public_v1 and empty_positive_mask_policy != "exclude_localization":
        raise PublicFoldValidationError(
            "strict_public_v1_exclusion_manifest_required",
            "strict v1 requires the audited empty-mask localization exclusions",
        )
    root = Path(root_value).expanduser().resolve()
    metadata_path = root / "metadata.csv"
    mapping_path = root / "slide_image_mapping.xlsx"
    license_path = root / "LICENSE.txt"
    readme_candidates = (root / "README.source.md", root / "README.md")
    readme_path = next((item for item in readme_candidates if item.is_file()), None)
    required_files = (metadata_path, mapping_path, license_path)
    if any(not item.is_file() for item in required_files) or readme_path is None:
        raise PublicFoldValidationError(
            "provenance_files_missing",
            "metadata, mapping, license and source README are required",
        )
    if not (root / "images").is_dir() or not (root / "masks").is_dir():
        raise PublicFoldValidationError(
            "asset_directories_missing", "extracted images/ and masks/ are required"
        )
    try:
        license_text = license_path.read_text(encoding="utf-8")
        readme_text = readme_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        raise PublicFoldValidationError(
            "provenance_files_invalid", "license or source README is unreadable"
        ) from None
    provenance_terms = ("H&E", "microscope", "tissue fold")
    if strict_public_v1 and (
        "CC BY 4.0" not in license_text
        or any(
            term.casefold() not in readme_text.casefold() for term in provenance_terms
        )
    ):
        raise PublicFoldValidationError(
            "provenance_declaration_mismatch",
            "source README/license does not declare expected acquisition and license",
        )

    metadata_rows = _load_metadata(metadata_path)
    mapping_rows = _read_xlsx_rows(mapping_path)
    mapping_by_name: dict[str, dict[str, str]] = {}
    for row in mapping_rows:
        filename = row["image_filename"]
        if not filename or filename in mapping_by_name:
            raise PublicFoldValidationError(
                "mapping_duplicate_filename", filename or "blank filename"
            )
        mapping_by_name[filename] = row

    records: list[PublicFoldRecord] = []
    seen_filenames: set[str] = set()
    expected_images: set[Path] = set()
    expected_masks: set[Path] = set()
    counts: Counter[tuple[str, str]] = Counter()
    slide_strata: dict[str, tuple[str, str]] = {}
    shapes: Counter[str] = Counter()
    empty_positive_masks: list[str] = []
    for row in metadata_rows:
        filename = row["image_filename"]
        organ = row["organ"]
        class_name = row["class"]
        if not filename or filename in seen_filenames:
            raise PublicFoldValidationError(
                "metadata_duplicate_filename", filename or "blank filename"
            )
        seen_filenames.add(filename)
        if class_name not in _CLASSES or not organ:
            raise PublicFoldValidationError("metadata_class_or_organ_invalid", filename)
        if strict_public_v1 and row["dataset_version"] != "1.0":
            raise PublicFoldValidationError("dataset_version_mismatch", filename)
        mapping = mapping_by_name.get(filename)
        if mapping is None:
            raise PublicFoldValidationError("mapping_row_missing", filename)
        if mapping["organ"] != organ or mapping["class"] != class_name:
            raise PublicFoldValidationError("mapping_stratum_mismatch", filename)
        slide_id = mapping["slide_id"]
        if not slide_id:
            raise PublicFoldValidationError("mapping_slide_id_missing", filename)
        stratum = (organ, class_name)
        prior = slide_strata.setdefault(slide_id, stratum)
        if prior != stratum:
            raise PublicFoldValidationError(
                "slide_id_crosses_strata", "one slide maps to multiple organ/classes"
            )
        image_relative = _safe_relative_path(row["image_relative_path"], field=filename)
        image_path = root.joinpath(*image_relative.parts)
        if image_path.name != filename or not image_path.is_file():
            raise PublicFoldValidationError("image_pair_missing", filename)
        expected_images.add(image_path.resolve())
        mask_path: Path | None = None
        has_mask = row["mask_available"].casefold() == "yes"
        if class_name == "tissue_fold":
            if not has_mask or row["pairing_status"].casefold() != "matched":
                raise PublicFoldValidationError("fold_mask_not_matched", filename)
            mask_relative = _safe_relative_path(
                row["mask_relative_path"], field=filename
            )
            mask_path = root.joinpath(*mask_relative.parts)
            if mask_path.name != row["mask_filename"] or not mask_path.is_file():
                raise PublicFoldValidationError("fold_mask_missing", filename)
            expected_masks.add(mask_path.resolve())
        elif has_mask or row["mask_relative_path"] or row["mask_filename"]:
            raise PublicFoldValidationError("clean_record_has_mask", filename)
        localization_reference_valid = True
        if validate_asset_dimensions:
            height, width, localization_reference_valid = _inspect_dimensions(
                image_path, mask_path
            )
            shapes[f"{height}x{width}"] += 1
            if class_name == "tissue_fold" and not localization_reference_valid:
                empty_positive_masks.append(row["mask_filename"])
                if empty_positive_mask_policy == "error":
                    raise PublicFoldValidationError(
                        "positive_mask_empty",
                        f"{row['mask_filename']}; choose the explicit audited exclusion policy to continue",
                    )
        record = PublicFoldRecord(
            image_filename=filename,
            organ=organ,
            class_name=class_name,
            slide_id=slide_id,
            image_path=image_path,
            mask_path=mask_path,
            localization_reference_valid=localization_reference_valid,
            image_sha256=_sha256_file(image_path) if hash_assets else None,
            mask_sha256=(
                _sha256_file(mask_path)
                if hash_assets and mask_path is not None
                else None
            ),
        )
        records.append(record)
        counts[stratum] += 1

    if set(mapping_by_name) != seen_filenames:
        raise PublicFoldValidationError(
            "mapping_metadata_set_mismatch", "mapping and metadata filenames differ"
        )
    actual_images = {
        item.resolve()
        for item in (root / "images").rglob("*")
        if item.is_file()
        and item.suffix.casefold() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    }
    actual_masks = {
        item.resolve()
        for item in (root / "masks").rglob("*")
        if item.is_file() and item.suffix.casefold() in {".png", ".tif", ".tiff"}
    }
    if actual_images != expected_images or actual_masks != expected_masks:
        raise PublicFoldValidationError(
            "asset_manifest_set_mismatch",
            "missing or orphan image/mask assets detected",
        )
    if strict_public_v1 and dict(counts) != dict(_PUBLIC_V1_COUNTS):
        raise PublicFoldValidationError(
            "public_v1_count_mismatch", f"observed {sum(counts.values())} records"
        )

    sorted_records = tuple(sorted(records, key=lambda item: item.image_filename))
    observed_release_identity = {
        "metadata_sha256": _sha256_file(metadata_path),
        "slide_mapping_sha256": _sha256_file(mapping_path),
        "license_sha256": _sha256_file(license_path),
        "source_readme_sha256": _sha256_file(readme_path),
        "asset_manifest_sha256": _canonical_sha256(
            [record.manifest_entry() for record in sorted_records]
        ),
        "localization_exclusion_manifest_sha256": _canonical_sha256(
            sorted(empty_positive_masks)
        ),
    }
    release_identity_audit = (
        _verify_public_v1_release_identity(observed_release_identity)
        if strict_public_v1
        else {
            "identity_version": _PUBLIC_V1_RELEASE_IDENTITY_VERSION,
            "verified": False,
            "verified_components": [],
            "canonical_identity_sha256": None,
            "identity": dict(observed_release_identity),
            "status": "not_checked_non_strict",
        }
    )
    audit = {
        "dataset_name": "Histology Tissue Fold Dataset",
        "dataset_version": "1.0",
        "data_origin": "real microscope-acquired H&E teaching-slide fields",
        "license": "CC BY 4.0",
        "citation_status": "associated publication pending according to source README",
        "publication_doi": "10.3390/bioengineering13080937",
        "publication_status_note": (
            "The packaged source README says publication pending, but the current "
            "public repository record links the published Bioengineering article."
        ),
        "annotation": "QuPath manual binary masks for tissue_fold images",
        "claimable_artifacts": ["tissue_fold"],
        "crack_reference_available": False,
        "n_records": len(sorted_records),
        "n_slides": len({record.slide_id for record in sorted_records}),
        "counts": {
            f"{organ}/{class_name}": int(count)
            for (organ, class_name), count in sorted(counts.items())
        },
        "decoded_shape_counts": dict(sorted(shapes.items())),
        **observed_release_identity,
        "release_identity_verified": bool(release_identity_audit["verified"]),
        "release_identity": release_identity_audit,
        "asset_content_hashes_computed": bool(hash_assets),
        "empty_positive_mask_policy": empty_positive_mask_policy,
        "empty_positive_masks": sorted(empty_positive_masks),
        "n_empty_positive_masks": len(empty_positive_masks),
        "validation": {
            "metadata_mapping_exact_pairing": True,
            "image_mask_dimensions_checked": bool(validate_asset_dimensions),
            "binary_mask_values_checked": bool(validate_asset_dimensions),
            "empty_positive_masks_detected": len(empty_positive_masks),
            "empty_positive_masks_excluded_from_localization": (
                len(empty_positive_masks)
                if empty_positive_mask_policy == "exclude_localization"
                else 0
            ),
            "no_orphan_assets": True,
            "slide_strata_consistent": True,
            "public_v1_expected_counts_checked": bool(strict_public_v1),
            "release_identity_verified": bool(release_identity_audit["verified"]),
        },
    }
    return PublicFoldDataset(root=root, records=sorted_records, audit=audit)


def _stable_order_key(seed: int, stratum: tuple[str, str], slide_id: str) -> str:
    value = f"{seed}|{stratum[0]}|{stratum[1]}|{slide_id}".encode()
    return hashlib.sha256(value).hexdigest()


def _split_sizes(
    n_groups: int, config: PublicFoldBenchmarkConfig
) -> tuple[int, int, int]:
    if n_groups < 3:
        raise PublicFoldValidationError(
            "insufficient_slides_per_stratum",
            "each organ/class needs at least three source slides",
        )
    raw = (
        np.asarray(
            [config.fit_fraction, config.calibration_fraction, config.test_fraction]
        )
        * n_groups
    )
    sizes = np.floor(raw).astype(int)
    sizes[:] = np.maximum(sizes, 1)
    while int(sizes.sum()) > n_groups:
        eligible = np.flatnonzero(sizes > 1)
        index = int(eligible[np.argmax(sizes[eligible] - raw[eligible])])
        sizes[index] -= 1
    while int(sizes.sum()) < n_groups:
        index = int(np.argmax(raw - sizes))
        sizes[index] += 1
    return int(sizes[0]), int(sizes[1]), int(sizes[2])


_SPLIT_ROLES = ("fit", "calibration", "locked_test")


def _assignment_objective(
    assignments: Mapping[str, Sequence[str]],
    slide_records: Mapping[str, Sequence[PublicFoldRecord]],
    target_images: Sequence[float],
) -> float:
    return float(
        sum(
            abs(
                sum(len(slide_records[slide_id]) for slide_id in assignments[role])
                - target_images[index]
            )
            for index, role in enumerate(_SPLIT_ROLES)
        )
    )


def _deterministic_swap_fallback(
    slide_ids: Sequence[str],
    slide_records: Mapping[str, Sequence[PublicFoldRecord]],
    role_sizes: Sequence[int],
    target_images: Sequence[float],
) -> dict[str, list[str]]:
    """Deterministic near-exact fallback if the SciPy MILP backend is unavailable."""

    boundaries = np.cumsum([0, *role_sizes])
    assignments = {
        role: list(slide_ids[int(boundaries[index]) : int(boundaries[index + 1])])
        for index, role in enumerate(_SPLIT_ROLES)
    }
    current = _assignment_objective(assignments, slide_records, target_images)
    while True:
        best: tuple[float, str, str, str, str] | None = None
        for left_index, left_role in enumerate(_SPLIT_ROLES):
            for right_role in _SPLIT_ROLES[left_index + 1 :]:
                for left_id in assignments[left_role]:
                    for right_id in assignments[right_role]:
                        candidate = {
                            role: list(values) for role, values in assignments.items()
                        }
                        candidate[left_role].remove(left_id)
                        candidate[right_role].remove(right_id)
                        candidate[left_role].append(right_id)
                        candidate[right_role].append(left_id)
                        value = _assignment_objective(
                            candidate, slide_records, target_images
                        )
                        key = (value, left_role, right_role, left_id, right_id)
                        if value + 1e-12 < current and (best is None or key < best):
                            best = key
        if best is None:
            break
        current, left_role, right_role, left_id, right_id = best
        assignments[left_role].remove(left_id)
        assignments[right_role].remove(right_id)
        assignments[left_role].append(right_id)
        assignments[right_role].append(left_id)
    return assignments


def _balanced_slide_assignments(
    stratum: tuple[str, str],
    slide_records: Mapping[str, Sequence[PublicFoldRecord]],
    config: PublicFoldBenchmarkConfig,
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Minimize field-count imbalance while fixing source-slide group counts."""

    slide_ids = sorted(
        slide_records,
        key=lambda item: _stable_order_key(config.seed, stratum, item),
    )
    role_sizes = _split_sizes(len(slide_ids), config)
    fractions = (config.fit_fraction, config.calibration_fraction, config.test_fraction)
    n_images = np.asarray(
        [len(slide_records[slide_id]) for slide_id in slide_ids], dtype=np.float64
    )
    target_images = np.asarray(fractions, dtype=np.float64) * float(n_images.sum())
    n_slides = len(slide_ids)
    n_assignment = n_slides * len(_SPLIT_ROLES)
    n_variables = n_assignment + 2 * len(_SPLIT_ROLES)

    # Variables are x[slide, role] followed by positive/negative absolute-error
    # slacks.  Stable SHA-ranked tiny costs make tied optima reproducible without
    # changing the integer image-balance objective.
    objective = np.zeros(n_variables, dtype=np.float64)
    tie_epsilon = 1e-7 / max(1, n_slides * n_slides)
    for slide_index in range(n_slides):
        for role_index in range(len(_SPLIT_ROLES)):
            objective[slide_index * len(_SPLIT_ROLES) + role_index] = (
                tie_epsilon * (slide_index + 1) * (role_index + 1)
            )
    objective[n_assignment:] = 1.0

    rows: list[NDArray[np.float64]] = []
    lower: list[float] = []
    upper: list[float] = []
    for slide_index in range(n_slides):
        row = np.zeros(n_variables, dtype=np.float64)
        start = slide_index * len(_SPLIT_ROLES)
        row[start : start + len(_SPLIT_ROLES)] = 1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(1.0)
    for role_index, role_size in enumerate(role_sizes):
        row = np.zeros(n_variables, dtype=np.float64)
        row[role_index : n_assignment : len(_SPLIT_ROLES)] = 1.0
        rows.append(row)
        lower.append(float(role_size))
        upper.append(float(role_size))
    for role_index, target in enumerate(target_images):
        row = np.zeros(n_variables, dtype=np.float64)
        row[role_index : n_assignment : len(_SPLIT_ROLES)] = n_images
        row[n_assignment + 2 * role_index] = -1.0
        row[n_assignment + 2 * role_index + 1] = 1.0
        rows.append(row)
        lower.append(float(target))
        upper.append(float(target))

    algorithm = "scipy.optimize.milp/highs"
    solver_status: dict[str, Any]
    try:
        result = milp(
            objective,
            integrality=np.r_[np.ones(n_assignment), np.zeros(6)],
            bounds=Bounds(
                np.zeros(n_variables),
                np.r_[np.ones(n_assignment), np.full(6, np.inf)],
            ),
            constraints=LinearConstraint(
                np.stack(rows), np.asarray(lower), np.asarray(upper)
            ),
            options={"presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"status={result.status}, message={result.message}")
        matrix = result.x[:n_assignment].reshape(n_slides, len(_SPLIT_ROLES))
        assignment_indices = np.argmax(matrix, axis=1)
        if not np.all(matrix[np.arange(n_slides), assignment_indices] > 0.5):
            raise RuntimeError("MILP returned a non-integral slide assignment")
        assignments = {
            role: [
                slide_ids[index]
                for index in range(n_slides)
                if assignment_indices[index] == role_index
            ]
            for role_index, role in enumerate(_SPLIT_ROLES)
        }
        solver_status = {
            "success": True,
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
        }
    except (RuntimeError, ValueError) as error:
        algorithm = "deterministic_pairwise_swap_fallback"
        assignments = _deterministic_swap_fallback(
            slide_ids, slide_records, role_sizes, target_images
        )
        solver_status = {
            "success": True,
            "status": "fallback",
            "message": str(error),
            "objective": _assignment_objective(
                assignments, slide_records, target_images
            ),
        }

    for role, expected in zip(_SPLIT_ROLES, role_sizes, strict=True):
        assignments[role].sort(
            key=lambda item: _stable_order_key(config.seed, stratum, item)
        )
        if len(assignments[role]) != expected:
            raise PublicFoldValidationError(
                "split_optimizer_group_count_mismatch", f"{stratum}/{role}"
            )
    flattened = [slide_id for role in _SPLIT_ROLES for slide_id in assignments[role]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(slide_ids):
        raise PublicFoldValidationError("split_optimizer_coverage_failed", f"{stratum}")
    actual_images = {
        role: sum(len(slide_records[slide_id]) for slide_id in assignments[role])
        for role in _SPLIT_ROLES
    }
    audit = {
        "organ": stratum[0],
        "class": stratum[1],
        "algorithm": algorithm,
        "solver": solver_status,
        "n_images": int(n_images.sum()),
        "n_slides": n_slides,
        "roles": {
            role: {
                "target_fraction": float(fractions[index]),
                "target_images": float(target_images[index]),
                "actual_images": int(actual_images[role]),
                "absolute_image_deviation": float(
                    abs(actual_images[role] - target_images[index])
                ),
                "target_slides": int(role_sizes[index]),
                "actual_slides": len(assignments[role]),
            }
            for index, role in enumerate(_SPLIT_ROLES)
        },
    }
    return assignments, audit


def _build_public_fold_splits_with_audit(
    records: Sequence[PublicFoldRecord],
    config: PublicFoldBenchmarkConfig,
) -> tuple[dict[str, tuple[PublicFoldRecord, ...]], dict[str, Any]]:
    """Assign complete source slides to deterministic, field-balanced strata."""

    groups: dict[tuple[str, str], dict[str, list[PublicFoldRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    slide_strata: dict[str, tuple[str, str]] = {}
    slide_roles: dict[str, str] = {}
    role_records: dict[str, list[PublicFoldRecord]] = {
        role: [] for role in _SPLIT_ROLES
    }
    if not records:
        raise PublicFoldValidationError("split_input_empty", "no records supplied")
    for record in records:
        stratum = (record.organ, record.class_name)
        prior = slide_strata.setdefault(record.slide_id, stratum)
        if prior != stratum:
            raise PublicFoldValidationError("slide_id_crosses_strata", record.slide_id)
        groups[stratum][record.slide_id].append(record)
    stratum_audits: list[dict[str, Any]] = []
    for stratum, slide_records in sorted(groups.items()):
        assignments, optimizer_audit = _balanced_slide_assignments(
            stratum, slide_records, config
        )
        limit = config.limit_slides_per_stratum_per_split
        if limit is not None:
            assignments = {role: ids[:limit] for role, ids in assignments.items()}
        optimizer_audit["selected_after_smoke_limit"] = {
            role: {
                "n_slides": len(ids),
                "n_images": sum(len(slide_records[slide_id]) for slide_id in ids),
            }
            for role, ids in assignments.items()
        }
        stratum_audits.append(optimizer_audit)
        for role, ids in assignments.items():
            for slide_id in ids:
                if slide_id in slide_roles and slide_roles[slide_id] != role:
                    raise PublicFoldValidationError(
                        "slide_leakage", "source slide assigned to multiple roles"
                    )
                slide_roles[slide_id] = role
                role_records[role].extend(slide_records[slide_id])
    output = {
        role: tuple(sorted(items, key=lambda record: record.image_filename))
        for role, items in role_records.items()
    }
    role_slides = {
        role: {record.slide_id for record in items} for role, items in output.items()
    }
    if (
        role_slides["fit"] & role_slides["calibration"]
        or role_slides["fit"] & role_slides["locked_test"]
        or role_slides["calibration"] & role_slides["locked_test"]
    ):
        raise PublicFoldValidationError("slide_leakage", "split audit failed")
    for role, items in output.items():
        observed = {(record.organ, record.class_name) for record in items}
        if observed != set(groups):
            raise PublicFoldValidationError(
                "split_stratum_missing", f"{role} does not contain every organ/class"
            )
    selected_filenames = [
        record.image_filename for role in _SPLIT_ROLES for record in output[role]
    ]
    if len(selected_filenames) != len(set(selected_filenames)):
        raise PublicFoldValidationError(
            "split_record_overlap", "one field was assigned more than once"
        )
    if config.limit_slides_per_stratum_per_split is None and set(
        selected_filenames
    ) != {record.image_filename for record in records}:
        raise PublicFoldValidationError(
            "split_record_coverage_failed", "full split did not cover every field"
        )
    assignment_manifest: list[dict[str, Any]] = []
    for role in _SPLIT_ROLES:
        by_slide: dict[str, list[PublicFoldRecord]] = defaultdict(list)
        for record in output[role]:
            by_slide[record.slide_id].append(record)
        assignment_manifest.extend(
            {
                "organ": slide_records[0].organ,
                "class": slide_records[0].class_name,
                "slide_id": slide_id,
                "role": role,
                "n_images": len(slide_records),
            }
            for slide_id, slide_records in by_slide.items()
        )
    assignment_manifest.sort(
        key=lambda item: (item["organ"], item["class"], item["slide_id"], item["role"])
    )
    split_audit = {
        "protocol": (
            "organ-by-class stratification; fraction-derived fixed source-slide "
            "group counts; minimum absolute field-count deviation"
        ),
        "group_unit": "provided_source_slide_id",
        "requested_role_fractions": {
            "fit": config.fit_fraction,
            "calibration": config.calibration_fraction,
            "locked_test": config.test_fraction,
        },
        "optimizer_objective": "minimum sum absolute image/field-count deviation",
        "strata": stratum_audits,
        "full_record_coverage": (
            config.limit_slides_per_stratum_per_split is None
            and len(selected_filenames) == len(records)
        ),
        "smoke_limit_applied": config.limit_slides_per_stratum_per_split is not None,
        "assignment_manifest": assignment_manifest,
        "assignment_manifest_sha256": _canonical_sha256(assignment_manifest),
    }
    return output, split_audit


def build_public_fold_splits(
    records: Sequence[PublicFoldRecord],
    config: PublicFoldBenchmarkConfig,
) -> dict[str, tuple[PublicFoldRecord, ...]]:
    """Return deterministic source-slide-disjoint, field-balanced splits."""

    return _build_public_fold_splits_with_audit(records, config)[0]


def _load_image(record: PublicFoldRecord, max_dimension: int) -> _LoadedImage:
    bgr = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise PublicFoldValidationError("image_decode_failed", record.image_filename)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    original_shape = (int(image.shape[0]), int(image.shape[1]))
    if record.mask_path is None:
        target = np.zeros(original_shape, dtype=bool)
    else:
        raw_mask = cv2.imread(str(record.mask_path), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None or tuple(raw_mask.shape) != original_shape:
            raise PublicFoldValidationError(
                "image_mask_shape_mismatch", record.image_filename
            )
        target = raw_mask > 0
        if bool(np.any(target)) != record.localization_reference_valid:
            raise PublicFoldValidationError(
                "localization_reference_state_changed", record.image_filename
            )
    largest = max(original_shape)
    if largest > max_dimension:
        scale = max_dimension / float(largest)
        shape = (
            max(1, round(original_shape[1] * scale)),
            max(1, round(original_shape[0] * scale)),
        )
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
        target = cv2.resize(
            target.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    return _LoadedImage(
        image=np.ascontiguousarray(image),
        target=np.ascontiguousarray(target),
        original_shape=original_shape,
    )


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _tiles(
    image: NDArray[np.uint8], target: BoolArray, config: PublicFoldBenchmarkConfig
) -> Iterable[
    tuple[NDArray[np.uint8], BoolArray, BoolArray, tuple[int, int, int, int]]
]:
    height, width = image.shape[:2]
    for top in _axis_starts(height, config.tile_size, config.tile_stride):
        for left in _axis_starts(width, config.tile_size, config.tile_stride):
            bottom = min(top + config.tile_size, height)
            right = min(left + config.tile_size, width)
            patch = np.full(
                (config.tile_size, config.tile_size, 3), 255, dtype=np.uint8
            )
            patch_target = np.zeros((config.tile_size, config.tile_size), dtype=bool)
            patch_valid = np.zeros_like(patch_target)
            patch[: bottom - top, : right - left] = image[top:bottom, left:right]
            patch_target[: bottom - top, : right - left] = target[
                top:bottom, left:right
            ]
            patch_valid[: bottom - top, : right - left] = True
            yield patch, patch_target, patch_valid, (top, left, bottom, right)


class _PriorityReservoir:
    """Bounded deterministic random-priority sample for streaming matrices."""

    def __init__(self, capacity: int, seed: int):
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.values: NDArray[np.float64] | None = None
        self.keys = np.empty(0, dtype=np.float64)
        self.n_seen = 0

    def add(self, values: Any) -> None:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError("reservoir values must be a finite two-dimensional matrix")
        if matrix.shape[0] == 0:
            return
        new_keys = self.rng.random(matrix.shape[0])
        self.n_seen += int(matrix.shape[0])
        if self.values is None:
            combined_values = matrix
            combined_keys = new_keys
        else:
            combined_values = np.concatenate((self.values, matrix), axis=0)
            combined_keys = np.concatenate((self.keys, new_keys))
        if combined_values.shape[0] > self.capacity:
            indices = np.argpartition(combined_keys, self.capacity - 1)[: self.capacity]
            order = np.argsort(combined_keys[indices], kind="stable")
            indices = indices[order]
            combined_values = combined_values[indices]
            combined_keys = combined_keys[indices]
        self.values = np.ascontiguousarray(combined_values)
        self.keys = np.ascontiguousarray(combined_keys)

    def matrix(self) -> NDArray[np.float64]:
        if self.values is None or self.values.shape[0] == 0:
            raise ValueError("reservoir is empty")
        return self.values


class _LinearTokenProbe:
    """Class-balanced logistic readout over frozen tokens."""

    def __init__(self, l2: float, max_iterations: int):
        self.l2 = float(l2)
        self.max_iterations = int(max_iterations)

    def fit(self, negative: FloatArray, positive: FloatArray) -> _LinearTokenProbe:
        x = np.concatenate((negative, positive), axis=0).astype(np.float64, copy=False)
        y = np.concatenate((np.zeros(len(negative)), np.ones(len(positive)))).astype(
            np.float64
        )
        if x.ndim != 2 or min(x.shape) <= 0 or negative.shape[1] != positive.shape[1]:
            raise ValueError(
                "linear probe needs non-empty aligned positive/negative tokens"
            )
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        x = (x - self.mean_) / self.scale_
        weights = np.where(y > 0, 0.5 / len(positive), 0.5 / len(negative))

        def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
            coefficients = parameters[:-1]
            logits = x @ coefficients + parameters[-1]
            loss = float(
                np.sum(weights * (np.logaddexp(0.0, logits) - y * logits))
                + 0.5 * self.l2 * coefficients @ coefficients
            )
            probability = np.empty_like(logits)
            positive_logits = logits >= 0
            probability[positive_logits] = 1.0 / (
                1.0 + np.exp(-logits[positive_logits])
            )
            exponential = np.exp(logits[~positive_logits])
            probability[~positive_logits] = exponential / (1.0 + exponential)
            residual = weights * (probability - y)
            gradient = np.concatenate(
                (x.T @ residual + self.l2 * coefficients, [residual.sum()])
            )
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(x.shape[1] + 1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iterations, "ftol": 1e-10},
        )
        if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
            raise RuntimeError("linear token probe optimization failed")
        if not result.success:
            raise RuntimeError(
                "linear token probe optimization did not converge: "
                f"status={int(result.status)}, message={result.message}"
            )
        self.coefficients_ = result.x[:-1]
        self.intercept_ = float(result.x[-1])
        self.optimization_ = {
            "success": bool(result.success),
            "status": int(result.status),
            "iterations": int(result.nit),
            "function_evaluations": int(result.nfev),
            "final_loss": float(result.fun),
            "message": str(result.message),
            "gradient_infinity_norm": float(np.max(np.abs(result.jac))),
        }
        return self

    def predict_score(self, matrix: Any) -> FloatArray:
        x = np.asarray(matrix, dtype=np.float64)
        logits = ((x - self.mean_) / self.scale_) @ self.coefficients_ + self.intercept_
        output = np.empty_like(logits)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exponential = np.exp(logits[~positive])
        output[~positive] = exponential / (1.0 + exponential)
        return output


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _encode_batch(
    encoder: FrozenEncoder, patches: Sequence[NDArray[np.uint8]], batch_size: int
) -> FoundationFeatures:
    features = encoder.encode(
        np.stack(patches),
        semantic_channels=("red", "green", "blue"),
        batch_size=batch_size,
    )
    if not isinstance(features, FoundationFeatures):
        raise TypeError("injected encoder must return FoundationFeatures")
    return features


def _token_labels(
    targets: Sequence[BoolArray],
    valids: Sequence[BoolArray],
    grid_shape: tuple[int, int],
    positive_fraction: float,
) -> tuple[BoolArray, BoolArray]:
    labels: list[BoolArray] = []
    supports: list[BoolArray] = []
    for target, valid in zip(targets, valids, strict=True):
        fraction = cv2.resize(
            target.astype(np.float32),
            (grid_shape[1], grid_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        valid_fraction = cv2.resize(
            valid.astype(np.float32),
            (grid_shape[1], grid_shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        labels.append((fraction >= positive_fraction).reshape(-1))
        supports.append((valid_fraction >= 0.5).reshape(-1))
    return np.concatenate(labels), np.concatenate(supports)


def _fit_foundation_models(
    records: Sequence[PublicFoldRecord],
    encoder: FrozenEncoder,
    config: PublicFoldBenchmarkConfig,
) -> tuple[PatchKNNAnomalyScorer | None, _LinearTokenProbe | None, dict[str, Any]]:
    requested_heads = {
        _FOUNDATION_HEADS[method]
        for method in config.methods
        if method in _FOUNDATION_HEADS
    }
    needs_knn = "patchknn" in requested_heads
    needs_probe = "linear_probe" in requested_heads
    reference = _PriorityReservoir(config.max_reference_tokens, config.seed + 11)
    negative = _PriorityReservoir(config.max_probe_tokens_per_class, config.seed + 17)
    positive = _PriorityReservoir(config.max_probe_tokens_per_class, config.seed + 23)
    started = perf_counter()
    for record in records:
        loaded = _load_image(record, config.max_dimension)
        tiled = list(_tiles(loaded.image, loaded.target, config))
        for batch in _batched(tiled, config.encoder_batch_size):
            features = _encode_batch(
                encoder, [item[0] for item in batch], config.encoder_batch_size
            )
            matrix = features.patch_grid.reshape(-1, features.embedding_dim).astype(
                np.float64, copy=False
            )
            labels, supports = _token_labels(
                [item[1] for item in batch],
                [item[2] for item in batch],
                features.grid_shape,
                config.token_positive_fraction,
            )
            if needs_knn and not record.is_fold:
                reference.add(matrix[supports])
            if needs_probe and record.localization_reference_valid:
                negative.add(matrix[supports & ~labels])
                positive.add(matrix[supports & labels])
    knn: PatchKNNAnomalyScorer | None = None
    if needs_knn:
        knn = PatchKNNAnomalyScorer(
            neighbors=config.patchknn_neighbors,
            distance_chunk_size=config.patchknn_distance_chunk_size,
            max_reference_tokens=config.max_reference_tokens,
        ).fit(reference.matrix(), split_id="public_fold_fit_clean")
    probe: _LinearTokenProbe | None = None
    if needs_probe:
        probe = _LinearTokenProbe(config.probe_l2, config.probe_max_iterations).fit(
            negative.matrix(), positive.matrix()
        )
    return (
        knn,
        probe,
        {
            "runtime_seconds": perf_counter() - started,
            "knn_reference_tokens_seen": reference.n_seen if needs_knn else 0,
            "knn_reference_tokens_stored": (
                int(reference.matrix().shape[0]) if needs_knn else 0
            ),
            "probe_negative_tokens_seen": negative.n_seen if needs_probe else 0,
            "probe_positive_tokens_seen": positive.n_seen if needs_probe else 0,
            "probe_tokens_stored_per_class": {
                "negative": int(negative.matrix().shape[0]) if needs_probe else 0,
                "positive": int(positive.matrix().shape[0]) if needs_probe else 0,
            },
            "probe_optimization": None if probe is None else probe.optimization_,
        },
    )


def _image_score(score: FloatArray, valid: BoolArray, quantile: float) -> float:
    values = score[valid]
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("score map must contain finite valid pixels")
    return float(np.quantile(values, quantile))


def _score_classical(
    record: PublicFoldRecord, config: PublicFoldBenchmarkConfig
) -> _Scored:
    started = perf_counter()
    loaded = _load_image(record, config.max_dimension)
    _, score = classical_fold_candidates(
        loaded.image,
        modality="he",
        min_component_size=config.classical_min_component_size,
    )
    valid = np.ones(score.shape, dtype=bool)
    return _Scored(
        record=record,
        score=np.asarray(score, dtype=np.float64),
        valid=valid,
        target=loaded.target,
        localization_reference_valid=record.localization_reference_valid,
        image_score=_image_score(score, valid, config.image_score_quantile),
        runtime_seconds=perf_counter() - started,
    )


def _score_foundation(
    record: PublicFoldRecord,
    encoder: FrozenEncoder,
    config: PublicFoldBenchmarkConfig,
    knn: PatchKNNAnomalyScorer | None,
    probe: _LinearTokenProbe | None,
) -> dict[str, _Scored]:
    started = perf_counter()
    loaded = _load_image(record, config.max_dimension)
    requested = tuple(
        method for method in config.methods if method in _FOUNDATION_METHODS
    )
    requested_heads = tuple(
        dict.fromkeys(_FOUNDATION_HEADS[method] for method in requested)
    )
    accumulators = {
        head: np.zeros(loaded.target.shape, dtype=np.float64)
        for head in requested_heads
    }
    # uint64 keeps every legal high-overlap configuration exact.  A uint16
    # counter can wrap at stride=1 and silently corrupt reconstructed scores.
    coverage = np.zeros(loaded.target.shape, dtype=np.uint64)
    tiled = list(_tiles(loaded.image, loaded.target, config))
    for batch in _batched(tiled, config.encoder_batch_size):
        features = _encode_batch(
            encoder, [item[0] for item in batch], config.encoder_batch_size
        )
        matrix = features.patch_grid.reshape(-1, features.embedding_dim)
        head_token_maps: dict[str, FloatArray] = {}
        if "patchknn" in requested_heads:
            if knn is None:
                raise RuntimeError("requested PatchKNN head was not fitted")
            head_token_maps["patchknn"] = knn.raw_token_scores(matrix).reshape(
                features.batch_size, *features.grid_shape
            )
        if "linear_probe" in requested_heads:
            if probe is None:
                raise RuntimeError("requested linear-probe head was not fitted")
            head_token_maps["linear_probe"] = probe.predict_score(matrix).reshape(
                features.batch_size, *features.grid_shape
            )
        for batch_index, item in enumerate(batch):
            top, left, bottom, right = item[3]
            height, width = bottom - top, right - left
            coverage[top:bottom, left:right] += 1
            for head in requested_heads:
                token_maps = head_token_maps[head]
                reconstructed = cv2.resize(
                    token_maps[batch_index],
                    (config.tile_size, config.tile_size),
                    interpolation=cv2.INTER_LINEAR,
                )
                accumulators[head][top:bottom, left:right] += reconstructed[
                    :height, :width
                ]
    valid = coverage > 0
    if not np.all(valid):
        raise RuntimeError("tile reconstruction left unsupported output pixels")
    head_payloads: dict[str, tuple[FloatArray, float]] = {}
    for head, accumulator in accumulators.items():
        score = np.divide(
            accumulator,
            coverage,
            out=np.zeros_like(accumulator),
            where=coverage > 0,
        )
        head_payloads[head] = (
            score,
            _image_score(score, valid, config.image_score_quantile),
        )
    inclusive_runtime = perf_counter() - started
    scored_by_head: dict[str, _Scored] = {}
    for head, (score, image_score) in head_payloads.items():
        scored_by_head[head] = _Scored(
            record=record,
            score=score,
            valid=valid,
            target=loaded.target,
            localization_reference_valid=record.localization_reference_valid,
            image_score=image_score,
            runtime_seconds=inclusive_runtime,
        )
    # Legacy encoder-specific aliases and their encoder-agnostic names are
    # report identities, not distinct heads.  Sharing the immutable scored
    # object prevents duplicate full-resolution reconstruction and storage.
    return {method: scored_by_head[_FOUNDATION_HEADS[method]] for method in requested}


def _score_reservoir_candidates(
    reservoir: _PriorityReservoir, n_candidates: int
) -> FloatArray:
    values = reservoir.matrix().reshape(-1)
    quantiles = np.linspace(0.0, 1.0, max(3, n_candidates))
    candidates = np.unique(np.quantile(values, quantiles))
    return np.unique(
        np.concatenate(
            (
                [np.nextafter(float(values.min()), -np.inf)],
                candidates,
                [np.nextafter(float(values.max()), np.inf)],
            )
        )
    )


def _reservoir_candidate_audit(
    reservoir: _PriorityReservoir,
    candidates: FloatArray,
    requested_candidates: int,
) -> dict[str, Any]:
    stored = reservoir.matrix().shape[0]
    return {
        "candidate_search": "bounded_priority_reservoir_quantile_grid",
        "requested_quantile_candidate_count": int(requested_candidates),
        "evaluated_unique_candidate_count": len(candidates),
        "reservoir_seed": reservoir.seed,
        "reservoir_capacity": reservoir.capacity,
        "reservoir_values_seen": reservoir.n_seen,
        "reservoir_values_stored": int(stored),
        "global_unique_score_optimum_claimed": False,
    }


def _update_candidate_counts(
    scored: _Scored,
    thresholds: FloatArray,
    totals: dict[str, NDArray[np.int64]],
) -> None:
    if not scored.localization_reference_valid:
        return
    score = scored.score[scored.valid]
    target = scored.target[scored.valid].astype(np.int64)
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    cumulative_positive = np.concatenate(([0], np.cumsum(target[order])))
    indices = np.searchsorted(sorted_score, thresholds, side="left")
    positives = int(target.sum())
    predicted = len(sorted_score) - indices
    tp = positives - cumulative_positive[indices]
    fp = predicted - tp
    fn = positives - tp
    tn = len(sorted_score) - positives - fp
    totals["tp"] += tp
    totals["fp"] += fp
    totals["fn"] += fn
    totals["tn"] += tn


def _select_pixel_threshold(
    thresholds: FloatArray, totals: Mapping[str, NDArray[np.int64]]
) -> tuple[float, dict[str, float]]:
    tp = totals["tp"].astype(np.float64)
    fp = totals["fp"].astype(np.float64)
    fn = totals["fn"].astype(np.float64)
    dice = np.divide(
        2 * tp, 2 * tp + fp + fn, out=np.ones_like(tp), where=(2 * tp + fp + fn) > 0
    )
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    # Conservative tie-break: Dice, precision, then higher threshold.
    best = max(
        range(len(thresholds)),
        key=lambda index: (dice[index], precision[index], thresholds[index]),
    )
    return float(thresholds[best]), {
        "calibration_dice": float(dice[best]),
        "calibration_precision": float(precision[best]),
    }


def _binary_rates(
    labels: NDArray[np.int64], predictions: NDArray[np.int64]
) -> dict[str, float]:
    tp = int(np.count_nonzero((labels == 1) & (predictions == 1)))
    fp = int(np.count_nonzero((labels == 0) & (predictions == 1)))
    fn = int(np.count_nonzero((labels == 1) & (predictions == 0)))
    tn = int(np.count_nonzero((labels == 0) & (predictions == 0)))
    sensitivity = tp / (tp + fn) if tp + fn else 1.0
    specificity = tn / (tn + fp) if tn + fp else 1.0
    return {
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
    }


def _select_image_threshold(
    labels: NDArray[np.int64], scores: FloatArray
) -> tuple[float, dict[str, float]]:
    unique = np.unique(scores)
    candidates = np.unique(
        np.concatenate(
            (
                [np.nextafter(float(unique.min()), -np.inf)],
                unique,
                [np.nextafter(float(unique.max()), np.inf)],
            )
        )
    )
    best_threshold = float(candidates[0])
    best_rates: dict[str, float] | None = None
    best_key = (-np.inf, -np.inf, -np.inf)
    for threshold in candidates:
        rates = _binary_rates(labels, (scores >= threshold).astype(np.int64))
        key = (
            rates["balanced_accuracy"],
            rates["specificity"],
            float(threshold),
        )
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_rates = rates
    assert best_rates is not None
    return best_threshold, best_rates


def _pixel_counts(scored: _Scored, threshold: float) -> tuple[int, int, int, int, int]:
    target = scored.target & scored.valid
    prediction = (scored.score >= threshold) & scored.valid
    tp = int(np.count_nonzero(target & prediction))
    fp = int(np.count_nonzero(~target & prediction & scored.valid))
    fn = int(np.count_nonzero(target & ~prediction & scored.valid))
    tn = int(np.count_nonzero(~target & ~prediction & scored.valid))
    return tp, fp, fn, tn, int(scored.valid.sum())


def _pixel_metrics(results: Sequence[_Result]) -> dict[str, Any]:
    """Pooled pixel micro metrics over the supplied eligible fields."""

    eligible = [item for item in results if item.localization_reference_valid]
    tp = sum(item.tp for item in eligible)
    fp = sum(item.fp for item in eligible)
    fn = sum(item.fn for item in eligible)
    tn = sum(item.tn for item in eligible)
    denominator = 2 * tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_valid": tp + fp + fn + tn,
        "n_localization_images": len(eligible),
        "n_images_excluded_invalid_reference": len(results) - len(eligible),
        "aggregation": "pooled_pixel_micro",
        "dice": float(2 * tp / denominator) if denominator else 1.0,
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "precision": float(tp / (tp + fp))
        if tp + fp
        else (1.0 if tp + fn == 0 else 0.0),
        "recall": float(tp / (tp + fn)) if tp + fn else 1.0,
    }


def _distribution_summary(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
        }
    q1, median, q3 = np.quantile(array, (0.25, 0.5, 0.75))
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": float(array.std(ddof=1)) if array.size > 1 else None,
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
    }


def _single_result_overlap(result: _Result) -> tuple[float, float]:
    dice_denominator = 2 * result.tp + result.fp + result.fn
    union = result.tp + result.fp + result.fn
    return (
        float(2 * result.tp / dice_denominator) if dice_denominator else 1.0,
        float(result.tp / union) if union else 1.0,
    )


def _positive_field_macro(results: Sequence[_Result]) -> dict[str, Any]:
    eligible = [
        item
        for item in results
        if item.label == 1 and item.localization_reference_valid
    ]
    overlaps = [_single_result_overlap(item) for item in eligible]
    return {
        "unit": "positive_localization_field",
        "n_excluded_invalid_reference": sum(
            item.label == 1 and not item.localization_reference_valid
            for item in results
        ),
        "dice": _distribution_summary([item[0] for item in overlaps]),
        "iou": _distribution_summary([item[1] for item in overlaps]),
    }


def _positive_slide_macro(results: Sequence[_Result]) -> dict[str, Any]:
    by_slide: dict[str, list[_Result]] = defaultdict(list)
    for result in results:
        if result.label == 1 and result.localization_reference_valid:
            by_slide[result.slide_id].append(result)
    dice_values: list[float] = []
    iou_values: list[float] = []
    for slide_results in by_slide.values():
        aggregated = replace(
            slide_results[0],
            tp=sum(item.tp for item in slide_results),
            fp=sum(item.fp for item in slide_results),
            fn=sum(item.fn for item in slide_results),
            tn=sum(item.tn for item in slide_results),
            n_valid=sum(item.n_valid for item in slide_results),
        )
        dice, iou = _single_result_overlap(aggregated)
        dice_values.append(dice)
        iou_values.append(iou)
    return {
        "unit": "provided_source_slide_id_with_positive_class",
        "dice": _distribution_summary(dice_values),
        "iou": _distribution_summary(iou_values),
    }


def _ranking_metrics(
    labels: NDArray[np.int64], scores: FloatArray
) -> dict[str, float | None]:
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if positives == 0 or negatives == 0:
        auroc: float | None = None
        auprc: float | None = None
    else:
        ranks = rankdata(scores, method="average")
        auroc = float(
            (ranks[labels == 1].sum() - positives * (positives + 1) / 2)
            / (positives * negatives)
        )
        # Average precision is accumulated at complete score-tie boundaries so
        # file order cannot change the reported result.
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        sorted_labels = labels[order]
        cumulative_tp = np.cumsum(sorted_labels)
        group_ends = np.flatnonzero(
            np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
        )
        group_tp = cumulative_tp[group_ends]
        precisions = group_tp / (group_ends + 1)
        recalls = group_tp / positives
        auprc = float(np.sum(np.diff(np.r_[0.0, recalls]) * precisions))
    return {"auroc": auroc, "auprc": auprc}


def _result_metrics(results: Sequence[_Result]) -> dict[str, Any]:
    labels = np.asarray([item.label for item in results], dtype=np.int64)
    scores = np.asarray([item.image_score for item in results], dtype=np.float64)
    predictions = np.asarray(
        [item.image_prediction for item in results], dtype=np.int64
    )
    image = {
        **_ranking_metrics(labels, scores),
        **_binary_rates(labels, predictions),
        "n_images": len(results),
        "n_positive": int(labels.sum()),
        "n_negative": int(len(labels) - labels.sum()),
    }
    clean = [item for item in results if item.label == 0]
    clean_valid = sum(item.n_valid for item in clean)
    clean_fp = sum(item.fp for item in clean)
    clean_area_fractions = np.asarray(
        [item.fp / item.n_valid for item in clean if item.n_valid > 0],
        dtype=np.float64,
    )
    burden = {
        "n_clean_images": len(clean),
        "false_positive_pixel_fraction": (
            float(clean_fp / clean_valid) if clean_valid else None
        ),
        "false_positive_pixels_per_megapixel": (
            float(clean_fp * 1_000_000 / clean_valid) if clean_valid else None
        ),
        "clean_image_alert_rate": (
            float(sum(item.image_prediction for item in clean) / len(clean))
            if clean
            else None
        ),
        "mean_predicted_area_fraction_per_clean_field": (
            float(clean_area_fractions.mean()) if clean_area_fractions.size else None
        ),
        "clean_pixel_specificity": (
            float(1.0 - clean_fp / clean_valid) if clean_valid else None
        ),
        "fraction_clean_fields_predicted_area_at_least_1_percent": (
            float(np.mean(clean_area_fractions >= 0.01))
            if clean_area_fractions.size
            else None
        ),
        "fraction_clean_fields_predicted_area_at_least_5_percent": (
            float(np.mean(clean_area_fractions >= 0.05))
            if clean_area_fractions.size
            else None
        ),
    }
    all_fields_micro = _pixel_metrics(results)
    positive_fields = [item for item in results if item.label == 1]
    positive_fields_micro = _pixel_metrics(positive_fields)
    return {
        # Backward-compatible alias, now named explicitly alongside it.
        "pixel": dict(all_fields_micro),
        "pixel_all_fields_micro": all_fields_micro,
        "pixel_positive_fields_micro": positive_fields_micro,
        "positive_field_macro": _positive_field_macro(results),
        "positive_slide_macro": _positive_slide_macro(results),
        "image": image,
        "clean_burden": burden,
    }


def _bootstrap(
    results: Sequence[_Result], config: PublicFoldBenchmarkConfig, method_index: int
) -> dict[str, Any]:
    provenance = {
        "cluster_unit": "provided_source_slide_id",
        "stratification": "organ_by_class",
        "sampling": "clusters sampled with replacement within every locked-test stratum",
        "preserves_cluster_count_per_stratum": True,
        "conditional_on": [
            "locked split assignment",
            "frozen fitted model/readout",
            "locked calibration thresholds",
            "observed public dataset",
        ],
    }
    if config.bootstrap_resamples == 0:
        return {"resamples": 0, **provenance, "intervals": {}}
    by_stratum: dict[tuple[str, int], dict[str, list[_Result]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for result in results:
        by_stratum[(result.organ, result.label)][result.slide_id].append(result)
    rng = np.random.default_rng(config.seed + 1009 * (method_index + 1))
    values: dict[str, list[float]] = defaultdict(list)
    for resample_index in range(config.bootstrap_resamples):
        sample: list[_Result] = []
        for stratum, by_slide in sorted(by_stratum.items()):
            slide_ids = sorted(by_slide)
            selected_indices = rng.integers(0, len(slide_ids), size=len(slide_ids))
            for draw_index, selected_index in enumerate(selected_indices):
                slide_id = slide_ids[int(selected_index)]
                replicate_id = (
                    f"{slide_id}#bootstrap-{resample_index}-{stratum[0]}-"
                    f"{stratum[1]}-{draw_index}"
                )
                sample.extend(
                    replace(item, slide_id=replicate_id) for item in by_slide[slide_id]
                )
        metrics = _result_metrics(sample)
        candidates = {
            "pixel.dice": metrics["pixel_all_fields_micro"]["dice"],
            "pixel.iou": metrics["pixel_all_fields_micro"]["iou"],
            "pixel_all_fields_micro.dice": metrics["pixel_all_fields_micro"]["dice"],
            "pixel_all_fields_micro.iou": metrics["pixel_all_fields_micro"]["iou"],
            "pixel_positive_fields_micro.dice": metrics["pixel_positive_fields_micro"][
                "dice"
            ],
            "pixel_positive_fields_micro.iou": metrics["pixel_positive_fields_micro"][
                "iou"
            ],
            "positive_field_macro.dice.mean": metrics["positive_field_macro"]["dice"][
                "mean"
            ],
            "positive_field_macro.iou.mean": metrics["positive_field_macro"]["iou"][
                "mean"
            ],
            "positive_slide_macro.dice.mean": metrics["positive_slide_macro"]["dice"][
                "mean"
            ],
            "positive_slide_macro.iou.mean": metrics["positive_slide_macro"]["iou"][
                "mean"
            ],
            "image.auroc": metrics["image"]["auroc"],
            "image.auprc": metrics["image"]["auprc"],
            "image.sensitivity": metrics["image"]["sensitivity"],
            "image.specificity": metrics["image"]["specificity"],
            "clean_burden.false_positive_pixel_fraction": metrics["clean_burden"][
                "false_positive_pixel_fraction"
            ],
            "clean_burden.mean_predicted_area_fraction_per_clean_field": metrics[
                "clean_burden"
            ]["mean_predicted_area_fraction_per_clean_field"],
            "clean_burden.clean_pixel_specificity": metrics["clean_burden"][
                "clean_pixel_specificity"
            ],
            "clean_burden.fraction_clean_fields_predicted_area_at_least_1_percent": metrics[
                "clean_burden"
            ]["fraction_clean_fields_predicted_area_at_least_1_percent"],
            "clean_burden.fraction_clean_fields_predicted_area_at_least_5_percent": metrics[
                "clean_burden"
            ]["fraction_clean_fields_predicted_area_at_least_5_percent"],
        }
        for name, value in candidates.items():
            if value is not None and math.isfinite(float(value)):
                values[name].append(float(value))
    alpha = (1.0 - config.bootstrap_confidence) / 2.0
    intervals = {}
    for name, samples in sorted(values.items()):
        if samples:
            intervals[name] = {
                "low": float(np.quantile(samples, alpha)),
                "high": float(np.quantile(samples, 1.0 - alpha)),
                "n_valid_resamples": len(samples),
            }
    return {
        "resamples": config.bootstrap_resamples,
        "confidence": config.bootstrap_confidence,
        **provenance,
        "intervals": intervals,
    }


def _split_report(records: Sequence[PublicFoldRecord]) -> dict[str, Any]:
    manifest = [record.manifest_entry() for record in records]
    return {
        "n_images": len(records),
        "n_slides": len({record.slide_id for record in records}),
        "n_localization_references": sum(
            record.localization_reference_valid for record in records
        ),
        "localization_exclusions": sorted(
            record.image_filename
            for record in records
            if not record.localization_reference_valid
        ),
        "counts": dict(
            sorted(
                Counter(f"{item.organ}/{item.class_name}" for item in records).items()
            )
        ),
        "slide_ids": sorted({record.slide_id for record in records}),
        "manifest": manifest,
        "manifest_sha256": _canonical_sha256(manifest),
    }


def _method_identity(method: str) -> dict[str, Any]:
    if method == "classical_fold":
        return {
            "reported_method_id": method,
            "algorithm_family": "classical_fold_candidates",
            "foundation_encoder_required": False,
            "legacy_encoder_specific_alias": False,
        }
    head = _FOUNDATION_HEADS[method]
    return {
        "reported_method_id": method,
        "algorithm_family": head,
        "foundation_encoder_required": True,
        "encoder_identity_location": "top-level model_identity when invoked by CLI",
        "legacy_encoder_specific_alias": method.startswith("dinov2_"),
        "canonical_encoder_agnostic_method_id": f"foundation_{head}",
    }


def _locked_test_result(
    record: PublicFoldRecord,
    scored: _Scored,
    *,
    pixel_threshold: float,
    image_threshold: float,
) -> _Result:
    tp, fp, fn, tn, n_valid = _pixel_counts(scored, pixel_threshold)
    return _Result(
        field_key=record.image_filename,
        organ=record.organ,
        slide_id=record.slide_id,
        label=int(record.is_fold),
        localization_reference_valid=scored.localization_reference_valid,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n_valid=n_valid,
        image_score=scored.image_score,
        image_prediction=int(scored.image_score >= image_threshold),
        runtime_seconds=scored.runtime_seconds,
    )


def _assemble_method_report(
    method: str,
    test_results: Sequence[_Result],
    *,
    pixel_threshold: float,
    pixel_calibration: Mapping[str, float],
    image_threshold: float,
    image_calibration: Mapping[str, float],
    config: PublicFoldBenchmarkConfig,
    calibration_manifest_sha256: str,
    method_index: int,
    calibration_runtime: float,
    test_wall: float,
    foundation_joint_execution: bool = False,
) -> dict[str, Any]:
    """Build one report without coupling execution order to metric semantics."""

    overall = _result_metrics(test_results)
    per_organ = {
        organ: _result_metrics([item for item in test_results if item.organ == organ])
        for organ in sorted({item.organ for item in test_results})
    }
    bootstrap = _bootstrap(test_results, config, method_index)
    intervals = bootstrap["intervals"]
    for family in ("positive_field_macro", "positive_slide_macro"):
        for metric in ("dice", "iou"):
            overall[family][metric]["mean_stratified_cluster_bootstrap_ci"] = (
                intervals.get(f"{family}.{metric}.mean")
            )
    runtime_values = [item.runtime_seconds for item in test_results]
    runtime = {
        "calibration_two_pass_seconds": calibration_runtime,
        "test_wall_seconds": test_wall,
        "test_sum_method_seconds": float(sum(runtime_values)),
        "median_seconds_per_image": float(np.median(runtime_values)),
        "foundation_encoder_time_attribution": (
            "inclusive shared wall time for one encoder traversal and all requested "
            "frozen heads; values must not be summed across method reports"
            if foundation_joint_execution
            else "direct method wall time"
        ),
        "shared_joint_foundation_execution": foundation_joint_execution,
        "additive_across_method_reports": not foundation_joint_execution,
    }
    outcomes = [
        {
            "field_key": item.field_key,
            "organ": item.organ,
            "source_slide_id": item.slide_id,
            "label": item.label,
            "localization_reference_valid": item.localization_reference_valid,
            "tp": item.tp,
            "fp": item.fp,
            "fn": item.fn,
            "tn": item.tn,
            "n_valid": item.n_valid,
            "image_score": item.image_score,
            "image_prediction": item.image_prediction,
            "runtime_seconds": item.runtime_seconds,
        }
        for item in sorted(test_results, key=lambda value: value.field_key)
    ]
    return {
        "method": method,
        "method_identity": _method_identity(method),
        "thresholds": {
            "pixel_localization": {
                "value": pixel_threshold,
                "objective": (
                    "maximum exact pooled calibration Dice among the "
                    "predeclared bounded-reservoir quantile candidates; precision "
                    "and higher-threshold tie-break"
                ),
                "selected_on": "calibration_only",
                "calibration_manifest_sha256": calibration_manifest_sha256,
                **pixel_calibration,
            },
            "image_presence": {
                "value": image_threshold,
                "objective": "maximum calibration balanced accuracy; specificity and higher-threshold tie-break",
                "selected_on": "calibration_only",
                "calibration_manifest_sha256": calibration_manifest_sha256,
                **image_calibration,
            },
            "test_labels_accessed_during_selection": False,
        },
        "locked_test": overall,
        "locked_test_outcomes": outcomes,
        "locked_test_outcomes_sha256": _canonical_sha256(outcomes),
        "per_organ": per_organ,
        "bootstrap_ci": bootstrap,
        "runtime": runtime,
    }


def _execute_method(
    method: str,
    calibration: Sequence[PublicFoldRecord],
    locked_test: Sequence[PublicFoldRecord],
    score_one: Any,
    config: PublicFoldBenchmarkConfig,
    calibration_manifest_sha256: str,
    method_index: int,
) -> dict[str, Any]:
    score_reservoir = _PriorityReservoir(
        config.calibration_score_sample, config.seed + 101 + method_index
    )
    calibration_images: list[tuple[int, float]] = []
    calibration_runtime = 0.0
    for record in calibration:
        scored: _Scored = score_one(record)
        if scored.localization_reference_valid:
            score_reservoir.add(scored.score[scored.valid])
        calibration_images.append((int(record.is_fold), scored.image_score))
        calibration_runtime += scored.runtime_seconds
    thresholds = _score_reservoir_candidates(
        score_reservoir, config.threshold_candidates
    )
    totals = {
        name: np.zeros(len(thresholds), dtype=np.int64)
        for name in ("tp", "fp", "fn", "tn")
    }
    for record in calibration:
        scored = score_one(record)
        _update_candidate_counts(scored, thresholds, totals)
        calibration_runtime += scored.runtime_seconds
    pixel_threshold, pixel_calibration = _select_pixel_threshold(thresholds, totals)
    pixel_calibration.update(
        _reservoir_candidate_audit(
            score_reservoir, thresholds, config.threshold_candidates
        )
    )
    calibration_labels = np.asarray(
        [item[0] for item in calibration_images], dtype=np.int64
    )
    calibration_scores = np.asarray(
        [item[1] for item in calibration_images], dtype=np.float64
    )
    image_threshold, image_calibration = _select_image_threshold(
        calibration_labels, calibration_scores
    )

    test_results: list[_Result] = []
    test_started = perf_counter()
    for record in locked_test:
        scored = score_one(record)
        test_results.append(
            _locked_test_result(
                record,
                scored,
                pixel_threshold=pixel_threshold,
                image_threshold=image_threshold,
            )
        )
    test_wall = perf_counter() - test_started
    return _assemble_method_report(
        method,
        test_results,
        pixel_threshold=pixel_threshold,
        pixel_calibration=pixel_calibration,
        image_threshold=image_threshold,
        image_calibration=image_calibration,
        config=config,
        calibration_manifest_sha256=calibration_manifest_sha256,
        method_index=method_index,
        calibration_runtime=calibration_runtime,
        test_wall=test_wall,
    )


def _execute_foundation_methods(
    calibration: Sequence[PublicFoldRecord],
    locked_test: Sequence[PublicFoldRecord],
    *,
    encoder: FrozenEncoder,
    knn: PatchKNNAnomalyScorer | None,
    probe: _LinearTokenProbe | None,
    config: PublicFoldBenchmarkConfig,
    calibration_manifest_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Calibrate and test every frozen head with shared record traversals.

    Calibration remains deliberately two-pass: the first bounded reservoir
    pass fixes each method's candidate grid, and the second accumulates exact
    confusion counts.  Only scalar/image-level state and bounded reservoirs
    survive between records; full-resolution maps are discarded immediately.
    """

    methods = tuple(
        method for method in config.methods if method in _FOUNDATION_METHODS
    )
    if not methods:
        return {}, {"executed": False}

    score_reservoirs = {
        method: _PriorityReservoir(
            config.calibration_score_sample,
            config.seed + 101 + _METHOD_SEED_INDEX[method],
        )
        for method in methods
    }
    calibration_images: dict[str, list[tuple[int, float]]] = {
        method: [] for method in methods
    }
    calibration_runtime = {method: 0.0 for method in methods}

    calibration_pass_1_started = perf_counter()
    for record in calibration:
        scored_by_method = _score_foundation(record, encoder, config, knn, probe)
        for method in methods:
            scored = scored_by_method[method]
            if scored.localization_reference_valid:
                score_reservoirs[method].add(scored.score[scored.valid])
            calibration_images[method].append((int(record.is_fold), scored.image_score))
            calibration_runtime[method] += scored.runtime_seconds
    calibration_pass_1_wall = perf_counter() - calibration_pass_1_started

    threshold_candidates = {
        method: _score_reservoir_candidates(
            score_reservoirs[method], config.threshold_candidates
        )
        for method in methods
    }
    candidate_totals = {
        method: {
            name: np.zeros(len(threshold_candidates[method]), dtype=np.int64)
            for name in ("tp", "fp", "fn", "tn")
        }
        for method in methods
    }

    calibration_pass_2_started = perf_counter()
    for record in calibration:
        scored_by_method = _score_foundation(record, encoder, config, knn, probe)
        for method in methods:
            scored = scored_by_method[method]
            _update_candidate_counts(
                scored,
                threshold_candidates[method],
                candidate_totals[method],
            )
            calibration_runtime[method] += scored.runtime_seconds
    calibration_pass_2_wall = perf_counter() - calibration_pass_2_started

    pixel_selections: dict[str, tuple[float, dict[str, float]]] = {}
    image_selections: dict[str, tuple[float, dict[str, float]]] = {}
    for method in methods:
        pixel_selections[method] = _select_pixel_threshold(
            threshold_candidates[method], candidate_totals[method]
        )
        pixel_selections[method][1].update(
            _reservoir_candidate_audit(
                score_reservoirs[method],
                threshold_candidates[method],
                config.threshold_candidates,
            )
        )
        image_pairs = calibration_images[method]
        image_selections[method] = _select_image_threshold(
            np.asarray([item[0] for item in image_pairs], dtype=np.int64),
            np.asarray([item[1] for item in image_pairs], dtype=np.float64),
        )

    # All method-specific calibration decisions are locked before this loop.
    test_results: dict[str, list[_Result]] = {method: [] for method in methods}
    test_started = perf_counter()
    for record in locked_test:
        scored_by_method = _score_foundation(record, encoder, config, knn, probe)
        for method in methods:
            pixel_threshold = pixel_selections[method][0]
            image_threshold = image_selections[method][0]
            test_results[method].append(
                _locked_test_result(
                    record,
                    scored_by_method[method],
                    pixel_threshold=pixel_threshold,
                    image_threshold=image_threshold,
                )
            )
    test_wall = perf_counter() - test_started

    reports = {
        method: _assemble_method_report(
            method,
            test_results[method],
            pixel_threshold=pixel_selections[method][0],
            pixel_calibration=pixel_selections[method][1],
            image_threshold=image_selections[method][0],
            image_calibration=image_selections[method][1],
            config=config,
            calibration_manifest_sha256=calibration_manifest_sha256,
            method_index=_METHOD_SEED_INDEX[method],
            calibration_runtime=calibration_runtime[method],
            test_wall=test_wall,
            foundation_joint_execution=True,
        )
        for method in methods
    }
    shared_wall = calibration_pass_1_wall + calibration_pass_2_wall + test_wall
    provenance = {
        "executed": True,
        "execution_strategy": "joint_by_record_and_unique_foundation_head",
        "requested_method_ids": list(methods),
        "unique_heads": list(
            dict.fromkeys(_FOUNDATION_HEADS[method] for method in methods)
        ),
        "calibration_record_passes": 2,
        "locked_test_record_passes": 1,
        "record_scoring_traversals": {
            "calibration": 2 * len(calibration),
            "locked_test": len(locked_test),
        },
        "record_scoring_traversal_definition": (
            "one record scoring traversal; a record may require multiple encoder "
            "batch calls when its tile count exceeds encoder_batch_size"
        ),
        "memory_bound": (
            "full-resolution score maps retained for the current record only; "
            "calibration reservoirs are bounded by calibration_score_sample"
        ),
        "full_resolution_score_maps_retained_between_records": False,
        "threshold_and_bootstrap_state": "independent_per_reported_method_id",
        "shared_record_scoring_wall_seconds": {
            "calibration_pass_1": calibration_pass_1_wall,
            "calibration_pass_2": calibration_pass_2_wall,
            "locked_test": test_wall,
            "total": shared_wall,
        },
        "runtime_scope": (
            "shared calibration/test record-scoring loops; excludes foundation "
            "model fitting and metric/bootstrap report assembly"
        ),
        "method_report_runtime_values_additive": False,
    }
    return reports, provenance


def _report_eligibility(
    dataset: PublicFoldDataset,
    config: PublicFoldBenchmarkConfig,
    split_audit: Mapping[str, Any],
    leakage_audit: Mapping[str, Any],
    training: Mapping[str, Any],
    run_provenance: Mapping[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not config.strict_public_v1:
        reasons.append("strict_public_v1_expected_counts_and_provenance_not_enforced")
    if not config.validate_asset_dimensions:
        reasons.append("image_mask_dimensions_and_binary_masks_not_strictly_validated")
    if not config.hash_assets or not dataset.audit.get("asset_content_hashes_computed"):
        reasons.append("per_asset_content_hashes_not_computed")
    if not dataset.audit.get("release_identity_verified"):
        reasons.append("public_v1_release_identity_not_verified")
    if not run_provenance.get("provided"):
        reasons.append("run_provenance_absent")
    elif not run_provenance.get("valid"):
        reasons.append("run_provenance_invalid")
    if config.limit_slides_per_stratum_per_split is not None:
        reasons.append("cohort_limited_smoke_run")
    if config.empty_positive_mask_policy != "exclude_localization":
        reasons.append("audited_empty_positive_mask_exclusion_policy_not_enabled")
    if not split_audit.get("full_record_coverage"):
        reasons.append("split_does_not_cover_full_validated_cohort")
    if not leakage_audit.get("passed"):
        reasons.append("provided_source_slide_id_overlap_detected")
    validation = dataset.audit.get("validation", {})
    required_validation = (
        "metadata_mapping_exact_pairing",
        "image_mask_dimensions_checked",
        "binary_mask_values_checked",
        "no_orphan_assets",
        "slide_strata_consistent",
        "public_v1_expected_counts_checked",
        "release_identity_verified",
    )
    if any(not validation.get(name) for name in required_validation):
        reasons.append("strict_dataset_integrity_evidence_incomplete")
    probe_optimization = training.get("probe_optimization")
    if probe_optimization is not None and not probe_optimization.get("success"):
        reasons.append("linear_probe_optimizer_not_converged")
    return not reasons, reasons


def run_public_fold_benchmark(
    dataset_root: str | Path,
    *,
    encoder: FrozenEncoder | None = None,
    config: PublicFoldBenchmarkConfig | None = None,
    run_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run calibrated classical/frozen-feature methods on a locked real test set.

    ``run_provenance`` must be captured before this call. The runner validates
    it before reading/scoring data and never discovers or retrofits executable,
    environment, or model identity after the run.
    """

    resolved = PublicFoldBenchmarkConfig() if config is None else config
    provenance_audit = _validate_run_provenance(run_provenance, resolved)
    if set(resolved.methods) & _FOUNDATION_METHODS and encoder is None:
        raise ValueError("foundation methods require an injected frozen encoder")
    dataset = load_public_fold_dataset(
        dataset_root,
        strict_public_v1=resolved.strict_public_v1,
        validate_asset_dimensions=resolved.validate_asset_dimensions,
        hash_assets=resolved.hash_assets,
        empty_positive_mask_policy=resolved.empty_positive_mask_policy,
    )
    splits, split_protocol = _build_public_fold_splits_with_audit(
        dataset.records, resolved
    )
    split_report = {role: _split_report(records) for role, records in splits.items()}
    fit_slides = set(split_report["fit"]["slide_ids"])
    calibration_slides = set(split_report["calibration"]["slide_ids"])
    test_slides = set(split_report["locked_test"]["slide_ids"])
    leakage_audit = {
        "group_unit": "source_slide_id",
        "fit_calibration_overlap": len(fit_slides & calibration_slides),
        "fit_test_overlap": len(fit_slides & test_slides),
        "calibration_test_overlap": len(calibration_slides & test_slides),
        "passed": not (
            fit_slides & calibration_slides
            or fit_slides & test_slides
            or calibration_slides & test_slides
        ),
    }
    knn: PatchKNNAnomalyScorer | None = None
    probe: _LinearTokenProbe | None = None
    training: dict[str, Any] = {"runtime_seconds": 0.0}
    if set(resolved.methods) & _FOUNDATION_METHODS:
        assert encoder is not None
        knn, probe, training = _fit_foundation_models(splits["fit"], encoder, resolved)
    report_eligible, nonreportable_reasons = _report_eligibility(
        dataset,
        resolved,
        split_protocol,
        leakage_audit,
        training,
        provenance_audit,
    )

    unordered_method_reports: dict[str, Any] = {}
    if "classical_fold" in resolved.methods:
        unordered_method_reports["classical_fold"] = _execute_method(
            "classical_fold",
            splits["calibration"],
            splits["locked_test"],
            lambda record: _score_classical(record, resolved),
            resolved,
            split_report["calibration"]["manifest_sha256"],
            _METHOD_SEED_INDEX["classical_fold"],
        )
    foundation_evaluation: dict[str, Any] = {"executed": False}
    if set(resolved.methods) & _FOUNDATION_METHODS:
        assert encoder is not None
        foundation_reports, foundation_evaluation = _execute_foundation_methods(
            splits["calibration"],
            splits["locked_test"],
            encoder=encoder,
            knn=knn,
            probe=probe,
            config=resolved,
            calibration_manifest_sha256=split_report["calibration"]["manifest_sha256"],
        )
        unordered_method_reports.update(foundation_reports)
    method_reports = {
        method: unordered_method_reports[method] for method in resolved.methods
    }

    return {
        "schema_version": _PUBLIC_FOLD_REPORT_SCHEMA_VERSION,
        "status": (
            "complete_reportable_real_public_fold_benchmark"
            if report_eligible
            else "complete_nonreportable_feasibility_run"
        ),
        "execution_status": "complete",
        "report_eligible": report_eligible,
        "nonreportable_reasons": nonreportable_reasons,
        "claim_scope": {
            "modality": "H&E brightfield microscopy",
            "artifact": "tissue_fold",
            "fold_presence": True,
            "fold_localization": True,
            "crack_presence": False,
            "crack_localization": False,
            "cross_modality_generalization": False,
        },
        "dataset": dict(dataset.audit),
        "configuration": resolved.as_dict(),
        "configuration_sha256": _canonical_sha256(resolved.as_dict()),
        "run_provenance": provenance_audit,
        "splits": split_report,
        "split_protocol": split_protocol,
        "leakage_audit": leakage_audit,
        "foundation_training": training,
        "foundation_evaluation": foundation_evaluation,
        "methods": method_reports,
        "limitations": [
            "The dataset contains tissue-fold annotations but no crack/tear reference class; no crack claim is permitted.",
            "Fields are 10x microscope JPG images rather than whole-slide images, so WSI-scale throughput and failure modes remain untested.",
            "The teaching-slide veterinary tissue domain does not establish performance on Merck human, scanner, site, COMET or CosMx distributions.",
            "Evaluation occurs after aspect-preserving downsampling; thin structures below the resulting pixel scale may be lost.",
            "Source slide IDs are provided by the dataset mapping and are the highest available independence unit; patient/block IDs are unavailable.",
            "Thresholds are dataset-specific calibration choices and must not be transferred to another acquisition domain without recalibration.",
        ],
    }
