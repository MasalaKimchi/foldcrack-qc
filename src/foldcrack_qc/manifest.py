"""Validated manifests for internal, multimodal QC evaluation data.

The manifest is the trust boundary between private image exports and the
detector/evaluation code.  It intentionally uses only the standard library,
NumPy, and the project's existing readers/adapters.  Validation diagnostics
never echo paths, patient identifiers, or slide identifiers; ``sample_id`` is
the only record identifier included in an error.

Accepted formats are JSON (a list of records, or ``{"samples": [...]}``) and
JSON Lines/NDJSON (one record per non-empty line).  Relative image and mask
paths are resolved against the directory containing the manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters import adapt_image, read_image
from .schema import ChannelRole, Modality, QCSample


REQUIRED_FIELDS: tuple[str, ...] = ("sample_id", "modality", "image_path", "split")
MASK_PATH_FIELDS: Mapping[str, str] = {
    "fold_mask_path": "fold",
    "crack_mask_path": "crack",
    "tissue_mask_path": "tissue",
    "valid_mask_path": "valid",
    "ignore_mask_path": "ignore",
}
INSTANCE_MASK_PATH_FIELDS: Mapping[str, str] = {
    "fold_instance_mask_path": "fold_instances",
    "crack_instance_mask_path": "crack_instances",
}
GROUP_ID_FIELDS: tuple[str, ...] = ("patient_id", "block_id", "slide_id", "run_id")
ALLOWED_SPLITS: frozenset[str] = frozenset(
    {"development", "train", "validation", "test", "locked_test"}
)
ALLOWED_COHORTS: frozenset[str] = frozenset(
    {
        "development",
        "prevalence",
        "enriched_challenge",
        "external_generalization",
        "missing_degraded_input",
        "downstream_impact",
    }
)


def _checksum_field(path_field: str) -> str:
    return path_field.removesuffix("_path") + "_sha256"


@dataclass(frozen=True)
class ManifestIssue:
    """One sanitized validation finding.

    ``record_index`` is one-based.  For JSONL it is the source line number;
    for JSON it is the position in the record list.  Messages and fields are
    safe for logs: private grouping identifiers and file paths are omitted.
    """

    code: str
    message: str
    record_index: int | None = None
    sample_id: str | None = None
    field: str | None = None
    severity: str = "error"

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.record_index is not None:
            result["record_index"] = self.record_index
        if self.sample_id is not None:
            result["sample_id"] = self.sample_id
        if self.field is not None:
            result["field"] = self.field
        return result


@dataclass(frozen=True)
class ManifestValidation:
    """Structured result returned by :func:`validate_manifest`."""

    record_count: int
    valid_sample_count: int
    issues: tuple[ManifestIssue, ...]
    strict: bool = False

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def is_valid(self) -> bool:
        """Alias that reads naturally in assertions and calling code."""

        return self.valid

    @property
    def error_count(self) -> int:
        return sum(issue.severity == "error" for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(issue.severity == "warning" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "mode": "strict" if self.strict else "exploratory",
            "record_count": self.record_count,
            "valid_sample_count": self.valid_sample_count,
            "issue_count": len(self.issues),
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


class ManifestValidationError(ValueError):
    """Raised by :func:`load_samples` at the first invalid manifest record."""

    def __init__(self, issue: ManifestIssue):
        self.issue = issue
        location = (
            f"record {issue.record_index}"
            if issue.record_index is not None
            else "manifest"
        )
        sample = (
            f", sample_id={issue.sample_id!r}" if issue.sample_id is not None else ""
        )
        super().__init__(
            f"Manifest validation failed [{issue.code}] at {location}{sample}: {issue.message}"
        )


@dataclass(frozen=True)
class _ParsedManifest:
    records: tuple[tuple[int, Any], ...]
    record_count: int
    issues: tuple[ManifestIssue, ...]


@dataclass(frozen=True)
class _ProcessedManifest:
    samples: tuple[QCSample, ...]
    issues: tuple[ManifestIssue, ...]


def _issue(
    code: str,
    message: str,
    *,
    record_index: int | None = None,
    sample_id: str | None = None,
    field: str | None = None,
    severity: str = "error",
) -> ManifestIssue:
    return ManifestIssue(
        code=code,
        message=message,
        record_index=record_index,
        sample_id=sample_id,
        field=field,
        severity=severity,
    )


def _parse_manifest(path: str | Path) -> _ParsedManifest:
    manifest_path = Path(path).expanduser()
    suffix = manifest_path.suffix.lower()
    if suffix not in {".json", ".jsonl", ".ndjson"}:
        issue = _issue(
            "unsupported_manifest_format",
            "Manifest must use .json, .jsonl, or .ndjson format",
        )
        return _ParsedManifest((), 0, (issue,))

    try:
        text = manifest_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        issue = _issue(
            "manifest_read_error",
            f"Manifest could not be read ({type(exc).__name__})",
        )
        return _ParsedManifest((), 0, (issue,))

    if suffix in {".jsonl", ".ndjson"}:
        records: list[tuple[int, Any]] = []
        issues: list[ManifestIssue] = []
        count = 0
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                issues.append(
                    _issue(
                        "json_decode_error",
                        "Record is not valid JSON",
                        record_index=line_number,
                    )
                )
                continue
            records.append((line_number, record))
        return _ParsedManifest(tuple(records), count, tuple(issues))

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        issue = _issue(
            "json_decode_error",
            "Manifest is not valid JSON",
            record_index=max(1, int(exc.lineno)),
        )
        return _ParsedManifest((), 0, (issue,))

    if isinstance(payload, Mapping):
        if "samples" not in payload:
            issue = _issue(
                "invalid_manifest_root",
                "JSON object root must contain a 'samples' list",
            )
            return _ParsedManifest((), 0, (issue,))
        payload = payload["samples"]
    if not isinstance(payload, list):
        issue = _issue(
            "invalid_manifest_root",
            "JSON manifest root must be a record list or an object containing 'samples'",
        )
        return _ParsedManifest((), 0, (issue,))
    return _ParsedManifest(
        tuple((index, record) for index, record in enumerate(payload, start=1)),
        len(payload),
        (),
    )


def _clean_identifier(value: Any) -> str | None:
    if value is None or isinstance(value, (bool, list, tuple, dict)):
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _normalize_term(value: str) -> str:
    return value.strip().casefold().replace("-", "_").replace(" ", "_")


def _resolve_data_path(value: Any, base_directory: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return candidate.resolve()


def _valid_pixel_size(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        values = (float(value),)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        try:
            values = tuple(float(item) for item in value)
        except (TypeError, ValueError):
            return False
        if len(values) != 2:
            return False
    else:
        return False
    return all(math.isfinite(item) and item > 0 for item in values)


def _as_2d(array: np.ndarray) -> np.ndarray | None:
    candidate = np.asarray(array)
    if candidate.ndim == 3 and 1 in candidate.shape:
        candidate = np.squeeze(candidate)
    if candidate.ndim != 2:
        return None
    return candidate


def _binary_mask(array: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Validate a lossless binary label array before boolean conversion."""

    candidate = _as_2d(array)
    if candidate is None:
        return None, "mask_not_2d"
    if not (np.issubdtype(candidate.dtype, np.number) or candidate.dtype == np.bool_):
        return None, "mask_non_numeric"
    if np.issubdtype(candidate.dtype, np.complexfloating):
        return None, "mask_non_binary"
    if not np.all(np.isfinite(candidate)):
        return None, "mask_nonfinite"
    unique = np.unique(candidate)
    if not np.all(np.isin(unique, (0, 1))):
        return None, "mask_non_binary"
    return np.ascontiguousarray(candidate.astype(bool, copy=False)), None


def _instance_mask(array: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    """Validate an optional non-negative integer instance-label raster."""

    candidate = _as_2d(array)
    if candidate is None:
        return None, "instance_mask_not_2d"
    if not np.issubdtype(candidate.dtype, np.integer) or candidate.dtype == np.bool_:
        return None, "instance_mask_not_integer"
    if np.any(candidate < 0):
        return None, "instance_mask_negative"
    return np.ascontiguousarray(candidate), None


def _normalize_sha256(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized.startswith("sha256:"):
        normalized = normalized[7:]
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return None
    return normalized


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    candidate = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(candidate.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(item) for item in candidate.shape)).encode("ascii"))
    digest.update(memoryview(candidate).cast("B"))
    return digest.hexdigest()


def _process_records(
    parsed: _ParsedManifest,
    *,
    base_directory: Path,
    fail_fast: bool,
    strict: bool,
) -> _ProcessedManifest:
    issues: list[ManifestIssue] = list(parsed.issues)
    samples: list[QCSample] = []

    def emit(issue: ManifestIssue) -> None:
        if fail_fast and issue.severity == "error":
            raise ManifestValidationError(issue)
        issues.append(issue)

    if fail_fast and parsed.issues:
        raise ManifestValidationError(parsed.issues[0])
    if parsed.record_count == 0 and not parsed.issues:
        empty_issue = _issue(
            "empty_manifest", "Manifest must contain at least one record"
        )
        if fail_fast:
            raise ManifestValidationError(empty_issue)
        issues.append(empty_issue)

    seen_sample_ids: set[str] = set()
    group_splits: dict[str, dict[str, str]] = {field: {} for field in GROUP_ID_FIELDS}
    source_id_splits: dict[str, str] = {}
    source_path_splits: dict[str, str] = {}
    file_digest_splits: dict[str, str] = {}
    content_digest_splits: dict[str, str] = {}
    digest_cache: dict[Path, str] = {}

    for record_index, raw_record in parsed.records:
        if not isinstance(raw_record, Mapping):
            emit(
                _issue(
                    "record_not_object",
                    "Each manifest record must be a JSON object",
                    record_index=record_index,
                )
            )
            continue

        sample_id = _clean_identifier(raw_record.get("sample_id"))
        state = {"ok": True}

        def add(issue: ManifestIssue) -> None:
            if issue.severity == "error":
                state["ok"] = False
            emit(issue)

        def policy(code: str, message: str, field: str) -> None:
            add(
                _issue(
                    code,
                    message,
                    record_index=record_index,
                    sample_id=sample_id,
                    field=field,
                    severity="error" if strict else "warning",
                )
            )

        for field in REQUIRED_FIELDS:
            value = raw_record.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                add(
                    _issue(
                        "missing_required_field",
                        f"Required field {field!r} is missing or empty",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=field,
                    )
                )

        if sample_id is None and "sample_id" in raw_record:
            add(
                _issue(
                    "invalid_sample_id",
                    "sample_id must be a non-empty scalar value",
                    record_index=record_index,
                    field="sample_id",
                )
            )

        group_ids: dict[str, str] = {}
        for field in GROUP_ID_FIELDS:
            raw_identifier = raw_record.get(field)
            identifier = _clean_identifier(raw_identifier)
            if raw_identifier is not None and identifier is None:
                add(
                    _issue(
                        "invalid_group_id",
                        f"{field} must be a non-empty scalar value when supplied",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=field,
                    )
                )
            elif identifier is not None:
                group_ids[field] = identifier
        if not group_ids:
            add(
                _issue(
                    "missing_group_id",
                    "At least one patient/block/slide/run identifier is required for split isolation",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="patient_id/block_id/slide_id/run_id",
                )
            )

        split = _clean_identifier(raw_record.get("split"))
        normalized_split = _normalize_term(split) if split is not None else None
        raw_split = raw_record.get("split")
        if (
            split is None
            and raw_split is not None
            and not (isinstance(raw_split, str) and not raw_split.strip())
        ):
            add(
                _issue(
                    "invalid_split",
                    "split must be a non-empty scalar vocabulary term",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="split",
                )
            )
        if split is not None and normalized_split not in ALLOWED_SPLITS:
            policy(
                "uncontrolled_split",
                "split is outside the controlled vocabulary",
                "split",
            )

        raw_cohort = raw_record.get("cohort", raw_record.get("cohort_role"))
        cohort = _clean_identifier(raw_cohort)
        normalized_cohort = _normalize_term(cohort) if cohort is not None else None
        if (
            cohort is None
            and raw_cohort is not None
            and not (isinstance(raw_cohort, str) and not raw_cohort.strip())
        ):
            add(
                _issue(
                    "invalid_cohort",
                    "cohort must be a non-empty scalar vocabulary term",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="cohort",
                )
            )
        elif cohort is None:
            policy(
                "missing_cohort",
                "A locked cohort term is required for claim-bearing evaluation",
                "cohort",
            )
        elif normalized_cohort not in ALLOWED_COHORTS:
            policy(
                "uncontrolled_cohort",
                "cohort is outside the controlled vocabulary",
                "cohort",
            )

        raw_modality = raw_record.get("modality")
        modality: Modality | None = None
        if raw_modality is not None:
            try:
                modality = Modality.coerce(raw_modality)
            except (TypeError, ValueError):
                add(
                    _issue(
                        "unsupported_modality",
                        "modality must identify H&E, COMET, or CosMx",
                        record_index=record_index,
                        sample_id=sample_id,
                        field="modality",
                    )
                )

        try:
            image_path = _resolve_data_path(
                raw_record.get("image_path"), base_directory
            )
        except (OSError, RuntimeError):
            image_path = None
            add(
                _issue(
                    "invalid_path_field",
                    "image_path could not be resolved safely",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="image_path",
                )
            )
        if image_path is None:
            if raw_record.get("image_path") is not None:
                add(
                    _issue(
                        "invalid_path_field",
                        "image_path must be a non-empty string",
                        record_index=record_index,
                        sample_id=sample_id,
                        field="image_path",
                    )
                )

        raw_channel_names = raw_record.get("channel_names")
        channel_names: tuple[str, ...] | None = None
        if raw_channel_names is not None:
            if (
                not isinstance(raw_channel_names, list)
                or not raw_channel_names
                or any(
                    not isinstance(item, str) or not item.strip()
                    for item in raw_channel_names
                )
            ):
                add(
                    _issue(
                        "invalid_channel_names",
                        "channel_names must be a non-empty list of non-empty strings",
                        record_index=record_index,
                        sample_id=sample_id,
                        field="channel_names",
                    )
                )
            else:
                channel_names = tuple(item.strip() for item in raw_channel_names)
                if len({name.casefold() for name in channel_names}) != len(
                    channel_names
                ):
                    add(
                        _issue(
                            "duplicate_channel_name",
                            "channel_names must be unique under case-insensitive comparison",
                            record_index=record_index,
                            sample_id=sample_id,
                            field="channel_names",
                        )
                    )

        if modality in {Modality.COMET, Modality.COSMX} and channel_names is None:
            policy(
                "missing_fluorescence_channel_names",
                "Fluorescence channel names must be explicit for locked evaluation",
                "channel_names",
            )

        raw_channel_axis = raw_record.get("channel_axis")
        channel_axis: int | None = None
        if raw_channel_axis is not None:
            if isinstance(raw_channel_axis, bool) or not isinstance(
                raw_channel_axis, int
            ):
                add(
                    _issue(
                        "invalid_channel_axis",
                        "channel_axis must be an integer or null",
                        record_index=record_index,
                        sample_id=sample_id,
                        field="channel_axis",
                    )
                )
            else:
                channel_axis = int(raw_channel_axis)

        pixel_size_supplied = "pixel_size_um" in raw_record
        pixel_size_um = raw_record.get("pixel_size_um", 1.0)
        if not pixel_size_supplied:
            policy(
                "missing_pixel_size",
                "pixel_size_um must be explicit for physical-scale evaluation",
                "pixel_size_um",
            )
        if not _valid_pixel_size(pixel_size_um):
            add(
                _issue(
                    "invalid_pixel_size",
                    "pixel_size_um must be a positive finite scalar or [y_um, x_um]",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="pixel_size_um",
                )
            )

        color_order = raw_record.get("color_order", "rgb")
        if not isinstance(color_order, str):
            add(
                _issue(
                    "invalid_color_order",
                    "color_order must be a string",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="color_order",
                )
            )

        mask_paths: dict[str, Path] = {}
        mask_path_fields: dict[str, str] = {}
        for path_field, mask_name in MASK_PATH_FIELDS.items():
            raw_path = raw_record.get(path_field)
            if raw_path is None:
                continue
            try:
                resolved = _resolve_data_path(raw_path, base_directory)
            except (OSError, RuntimeError):
                resolved = None
            if resolved is None:
                add(
                    _issue(
                        "invalid_path_field",
                        f"{path_field} must be a non-empty string",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
            else:
                mask_paths[mask_name] = resolved
                mask_path_fields[mask_name] = path_field

        instance_paths: dict[str, Path] = {}
        instance_path_fields: dict[str, str] = {}
        for path_field, mask_name in INSTANCE_MASK_PATH_FIELDS.items():
            raw_path = raw_record.get(path_field)
            if raw_path is None:
                continue
            try:
                resolved = _resolve_data_path(raw_path, base_directory)
            except (OSError, RuntimeError):
                resolved = None
            if resolved is None:
                add(
                    _issue(
                        "invalid_path_field",
                        f"{path_field} must be a non-empty string",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
            else:
                instance_paths[mask_name] = resolved
                instance_path_fields[mask_name] = path_field

        if "valid" not in mask_paths and "ignore" not in mask_paths:
            policy(
                "missing_valid_region",
                "A valid_mask_path or ignore_mask_path is required for locked evaluation",
                "valid_mask_path/ignore_mask_path",
            )

        source_id = _clean_identifier(raw_record.get("source_id"))
        if raw_record.get("source_id") is not None and source_id is None:
            add(
                _issue(
                    "invalid_source_id",
                    "source_id must be a non-empty scalar value when supplied",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="source_id",
                )
            )

        paths_by_field: dict[str, Path] = {}
        if image_path is not None:
            paths_by_field["image_path"] = image_path
        paths_by_field.update(
            {mask_path_fields[name]: path for name, path in mask_paths.items()}
        )
        paths_by_field.update(
            {instance_path_fields[name]: path for name, path in instance_paths.items()}
        )
        expected_digests: dict[str, str] = {}
        for path_field in paths_by_field:
            checksum_field = _checksum_field(path_field)
            raw_checksum = raw_record.get(checksum_field)
            if raw_checksum is None:
                policy(
                    "missing_checksum",
                    f"{checksum_field} is required to lock input provenance",
                    checksum_field,
                )
                continue
            checksum = _normalize_sha256(raw_checksum)
            if checksum is None:
                add(
                    _issue(
                        "invalid_checksum",
                        f"{checksum_field} must contain a SHA-256 digest",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=checksum_field,
                    )
                )
            else:
                expected_digests[path_field] = checksum

        # Structural problems should not trigger noisy secondary I/O failures.
        if (
            not state["ok"]
            or modality is None
            or image_path is None
            or sample_id is None
        ):
            continue

        try:
            raw_image = read_image(image_path)
        except (
            Exception
        ) as exc:  # readers expose several backend-specific exception types
            add(
                _issue(
                    "image_load_error",
                    f"image_path could not be loaded ({type(exc).__name__})",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="image_path",
                )
            )
            continue

        image_array = np.asarray(raw_image)
        if not (
            np.issubdtype(image_array.dtype, np.number) or image_array.dtype == np.bool_
        ):
            add(
                _issue(
                    "image_non_numeric",
                    "Decoded image must contain numeric values",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="image_path",
                )
            )
        elif np.issubdtype(image_array.dtype, np.complexfloating) or not np.all(
            np.isfinite(image_array)
        ):
            add(
                _issue(
                    "image_nonfinite",
                    "Decoded image contains complex, NaN, or infinite values",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="image_path",
                )
            )
        if not state["ok"]:
            continue

        if modality in {Modality.COMET, Modality.COSMX} and image_array.ndim == 3:
            if raw_channel_axis is None:
                policy(
                    "implicit_channel_axis",
                    "channel_axis must be explicit for locked fluorescence evaluation",
                    "channel_axis",
                )
            if not state["ok"]:
                continue

        image_metadata = {
            "manifest_record_index": record_index,
            "split": split,
            "cohort": cohort,
            "strict_manifest": strict,
        }
        try:
            image = adapt_image(
                raw_image,
                modality,
                channel_names=channel_names,
                pixel_size_um=pixel_size_um,
                metadata=image_metadata,
                source_path=image_path,
                channel_axis=channel_axis,
                color_order=color_order,
            )
        except Exception as exc:
            add(
                _issue(
                    "image_adaptation_error",
                    f"Image metadata or array layout is invalid ({type(exc).__name__})",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="image_path",
                )
            )
            continue

        if modality in {Modality.COMET, Modality.COSMX}:
            roles = tuple(image.channel_roles)
            if ChannelRole.NUCLEAR not in roles:
                policy(
                    "missing_nuclear_role",
                    "Fluorescence channel names do not resolve a nuclear channel",
                    "channel_names",
                )
            if ChannelRole.UNKNOWN in roles:
                policy(
                    "unresolved_channel_role",
                    "One or more fluorescence channels have unresolved semantic roles",
                    "channel_names",
                )
        if not state["ok"]:
            continue

        actual_digests: dict[str, str] = {}

        def verify_digest(path_field: str, path: Path) -> None:
            expected = expected_digests.get(path_field)
            if expected is None and not strict:
                return
            try:
                actual = digest_cache.get(path)
                if actual is None:
                    actual = _file_sha256(path)
                    digest_cache[path] = actual
            except OSError as exc:
                add(
                    _issue(
                        "checksum_read_error",
                        f"{_checksum_field(path_field)} could not be verified ({type(exc).__name__})",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=_checksum_field(path_field),
                    )
                )
                return
            actual_digests[path_field] = actual
            if expected is not None and actual != expected:
                add(
                    _issue(
                        "checksum_mismatch",
                        f"{_checksum_field(path_field)} does not match the referenced file",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=_checksum_field(path_field),
                    )
                )

        verify_digest("image_path", image_path)

        masks: dict[str, np.ndarray] = {}
        for mask_name, mask_path in mask_paths.items():
            path_field = mask_path_fields[mask_name]
            try:
                raw_mask = read_image(mask_path)
            except Exception as exc:
                add(
                    _issue(
                        "mask_load_error",
                        f"{path_field} could not be loaded ({type(exc).__name__})",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            mask, mask_error = _binary_mask(raw_mask)
            if mask_error is not None or mask is None:
                messages = {
                    "mask_not_2d": "must contain one 2-D mask",
                    "mask_non_numeric": "must contain numeric or boolean labels",
                    "mask_nonfinite": "contains NaN or infinite labels",
                    "mask_non_binary": "must contain only binary 0/1 labels",
                }
                add(
                    _issue(
                        mask_error or "mask_invalid",
                        f"{path_field} {messages.get(mask_error, 'is invalid')}",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            if tuple(mask.shape) != image.spatial_shape:
                add(
                    _issue(
                        "mask_shape_mismatch",
                        f"{path_field} shape {tuple(mask.shape)} does not match image shape {image.spatial_shape}",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            masks[mask_name] = mask
            verify_digest(path_field, mask_path)

        instance_masks: dict[str, np.ndarray] = {}
        for mask_name, mask_path in instance_paths.items():
            path_field = instance_path_fields[mask_name]
            try:
                raw_mask = read_image(mask_path)
            except Exception as exc:
                add(
                    _issue(
                        "mask_load_error",
                        f"{path_field} could not be loaded ({type(exc).__name__})",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            instance_mask, mask_error = _instance_mask(raw_mask)
            if mask_error is not None or instance_mask is None:
                messages = {
                    "instance_mask_not_2d": "must contain one 2-D label raster",
                    "instance_mask_not_integer": "must use an integer dtype",
                    "instance_mask_negative": "must contain non-negative instance IDs",
                }
                add(
                    _issue(
                        mask_error or "instance_mask_invalid",
                        f"{path_field} {messages.get(mask_error, 'is invalid')}",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            if tuple(instance_mask.shape) != image.spatial_shape:
                add(
                    _issue(
                        "mask_shape_mismatch",
                        f"{path_field} shape {tuple(instance_mask.shape)} does not match image shape {image.spatial_shape}",
                        record_index=record_index,
                        sample_id=sample_id,
                        field=path_field,
                    )
                )
                continue
            instance_masks[mask_name] = instance_mask
            verify_digest(path_field, mask_path)

        if (
            "valid" in masks
            and "ignore" in masks
            and np.any(masks["valid"] & masks["ignore"])
        ):
            add(
                _issue(
                    "valid_ignore_overlap",
                    "valid and ignore masks must not overlap",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="valid_mask_path/ignore_mask_path",
                )
            )
        if "valid" not in masks and "ignore" in masks:
            masks["valid"] = np.ascontiguousarray(~masks["ignore"])

        if not state["ok"]:
            continue

        # Hash canonical HWC content so equivalent CxHxW/HxWxC exports cannot
        # bypass duplicate detection merely by changing storage layout.
        content_digest = _array_sha256(image.data) if strict else None

        # Only fully valid records influence duplicate/leakage state.  This
        # prevents a malformed early record from poisoning later valid rows.
        if sample_id in seen_sample_ids:
            add(
                _issue(
                    "duplicate_sample_id",
                    "sample_id occurs more than once",
                    record_index=record_index,
                    sample_id=sample_id,
                    field="sample_id",
                )
            )
        if normalized_split is not None:
            for field, identifier in group_ids.items():
                prior = group_splits[field].get(identifier)
                if prior is not None and prior != normalized_split:
                    add(
                        _issue(
                            f"{field.removesuffix('_id')}_split_leakage",
                            f"{field} occurs in more than one split",
                            record_index=record_index,
                            sample_id=sample_id,
                            field=field,
                        )
                    )
            for code, field, value, mapping in (
                ("source_split_leakage", "source_id", source_id, source_id_splits),
                (
                    "source_path_split_leakage",
                    "image_path",
                    str(image_path),
                    source_path_splits,
                ),
                (
                    "file_checksum_split_leakage",
                    "image_sha256",
                    actual_digests.get("image_path"),
                    file_digest_splits,
                ),
                (
                    "image_content_split_leakage",
                    "image_path",
                    content_digest,
                    content_digest_splits,
                ),
            ):
                if value is not None and mapping.get(value) not in {
                    None,
                    normalized_split,
                }:
                    add(
                        _issue(
                            code,
                            "The same source or content occurs in more than one split",
                            record_index=record_index,
                            sample_id=sample_id,
                            field=field,
                        )
                    )

        if not state["ok"]:
            continue

        seen_sample_ids.add(sample_id)
        if normalized_split is not None:
            for field, identifier in group_ids.items():
                group_splits[field].setdefault(identifier, normalized_split)
            for value, mapping in (
                (source_id, source_id_splits),
                (str(image_path), source_path_splits),
                (actual_digests.get("image_path"), file_digest_splits),
                (content_digest, content_digest_splits),
            ):
                if value is not None:
                    mapping.setdefault(value, normalized_split)

        sample_metadata: dict[str, Any] = {
            "split": split,
            "cohort": cohort,
            "manifest_record_index": record_index,
            "mask_source_paths": {name: str(path) for name, path in mask_paths.items()},
            "instance_mask_source_paths": {
                name: str(path) for name, path in instance_paths.items()
            },
            "instance_masks": instance_masks,
            "verified_sha256": actual_digests,
            "strict_manifest": strict,
        }
        sample_metadata.update(group_ids)
        if source_id is not None:
            sample_metadata["source_id"] = source_id
        samples.append(
            QCSample(
                sample_id=sample_id, image=image, masks=masks, metadata=sample_metadata
            )
        )

    return _ProcessedManifest(tuple(samples), tuple(issues))


def validate_manifest(path: str | Path, *, strict: bool = False) -> ManifestValidation:
    """Validate syntax, records, data readability, shapes, and split isolation.

    Validation failures are returned as structured issues instead of raised.
    Exploratory mode keeps loading when lock-readiness fields are absent but
    emits warnings.  Strict mode promotes those warnings to errors and verifies
    declared SHA-256 checksums.  The result contains no private identifiers
    other than ``sample_id``.
    """

    manifest_path = Path(path).expanduser()
    parsed = _parse_manifest(manifest_path)
    processed = _process_records(
        parsed,
        base_directory=manifest_path.resolve().parent,
        fail_fast=False,
        strict=bool(strict),
    )
    return ManifestValidation(
        record_count=parsed.record_count,
        valid_sample_count=len(processed.samples),
        issues=processed.issues,
        strict=bool(strict),
    )


def load_samples(path: str | Path, *, strict: bool = False) -> list[QCSample]:
    """Load every record as a :class:`QCSample`, raising on the first error."""

    manifest_path = Path(path).expanduser()
    parsed = _parse_manifest(manifest_path)
    processed = _process_records(
        parsed,
        base_directory=manifest_path.resolve().parent,
        fail_fast=True,
        strict=bool(strict),
    )
    return list(processed.samples)


__all__ = [
    "MASK_PATH_FIELDS",
    "INSTANCE_MASK_PATH_FIELDS",
    "REQUIRED_FIELDS",
    "ALLOWED_COHORTS",
    "ALLOWED_SPLITS",
    "GROUP_ID_FIELDS",
    "ManifestIssue",
    "ManifestValidation",
    "ManifestValidationError",
    "load_samples",
    "validate_manifest",
]
