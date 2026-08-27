"""Exploratory paired comparison of hardened public H&E fold reports.

This comparator deliberately has a narrow evidence boundary.  It accepts only
report-eligible ``public-fold-benchmark-1.2`` artifacts whose release identity,
locked-test manifest, provenance identity, and per-method outcome-table hashes
can all be verified from the artifact itself.  It estimates descriptive paired
differences in positive-field Dice by resampling source-slide clusters within
organ.  It does not compute p-values or make superiority/noninferiority claims.

Example::

    python scripts/compare_hardened_public_fold.py \
      --report classical=artifacts/public_fold/classical_hardened_v1_2.json \
      --method classical=classical_fold \
      --report siglip2=artifacts/public_fold/siglip2_hardened_v1_2.json \
      --method siglip2=foundation_linear_probe \
      --output artifacts/public_fold/paired_exploratory.json

``--method NAME=METHOD_ID`` may be omitted only when that report contains one
method.  The output is byte-deterministic for identical inputs and options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "public-fold-benchmark-1.2"
COMPARISON_SCHEMA = "public-fold-paired-comparison-1.0"
DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 20260826
DEFAULT_CONFIDENCE = 0.95
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SPLIT_ROLES = ("fit", "calibration", "locked_test")
_CANONICAL_STATUS = "complete_reportable_real_public_fold_benchmark"
_CANONICAL_CLAIM_SCOPE: Mapping[str, Any] = {
    "artifact": "tissue_fold",
    "crack_localization": False,
    "crack_presence": False,
    "cross_modality_generalization": False,
    "fold_localization": True,
    "fold_presence": True,
    "modality": "H&E brightfield microscopy",
}
_CANONICAL_RELEASE_IDENTITY_VERSION = "histology-tissue-fold-v1.0-2026-08-26"
_CANONICAL_RELEASE_IDENTITY: Mapping[str, str] = {
    "asset_manifest_sha256": "826202d9951415ea5ffeafe2648b192bccc25f02ad0c3617b3be29bc9a5ab328",
    "license_sha256": "866d89cbf299323640d2ff76a5695e9813fded3a8aeed676c260583763767f17",
    "localization_exclusion_manifest_sha256": "2002f53e1beb42f8743169d0d023f385b4d7a3cb943d972c5e7a13bb1bf57926",
    "metadata_sha256": "101ca59ad4505db673253d370698b285f15342c77f590eeee65b0935357b72d4",
    "slide_mapping_sha256": "d3199c431771c8d87ac1d35f178208d1207769a9a048bd748b84808836169a40",
    "source_readme_sha256": "6e69e809522c880f093bb8c674351211f939969f594ea8658e64df674371d73f",
}
_EVALUATION_CONTRACT_CONFIGURATION_FIELDS = (
    "max_dimension",
    "tile_size",
    "tile_stride",
    "seed",
    "fit_fraction",
    "calibration_fraction",
    "test_fraction",
    "empty_positive_mask_policy",
    "limit_slides_per_stratum_per_split",
    "strict_public_v1",
    "validate_asset_dimensions",
    "hash_assets",
    "calibration_score_sample",
    "image_score_quantile",
    "threshold_candidates",
)
_METHOD_IDS = frozenset(
    {
        "classical_fold",
        "foundation_patchknn",
        "foundation_linear_probe",
        "dinov2_patchknn",
        "dinov2_linear_probe",
    }
)


class ComparisonValidationError(ValueError):
    """An input failed the comparator's closed evidence boundary."""


@dataclass(frozen=True)
class _ValidatedReport:
    label: str
    path: Path
    artifact_sha256: str
    selected_method: str
    method_identity: Mapping[str, Any]
    method_identity_sha256: str
    model_identity_sha256: str | None
    provenance_identity_sha256: str
    provenance_schema_version: str
    execution_identity: Mapping[str, Any]
    claim_scope: Mapping[str, Any]
    evaluation_contract: Mapping[str, Any]
    evaluation_contract_sha256: str
    dataset_identity: Mapping[str, Any]
    dataset_identity_sha256: str
    locked_test_manifest: Sequence[Mapping[str, Any]]
    locked_test_manifest_sha256: str
    positive_dice: Mapping[str, float]
    positive_metadata: Mapping[str, tuple[str, str]]
    evaluation_domain: Mapping[str, tuple[int, int, int]]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ComparisonValidationError(f"duplicate JSON object key: {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ComparisonValidationError(f"non-finite JSON number is forbidden: {value}")


def _load_strict_json(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise ComparisonValidationError(f"report is not a regular file: {path}")
    try:
        payload = path.read_bytes()
        artifact_sha256 = hashlib.sha256(payload).hexdigest()
        raw = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except ComparisonValidationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ComparisonValidationError(
            f"cannot read strict JSON {path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ComparisonValidationError(f"report root must be a JSON object: {path}")
    return raw, artifact_sha256


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ComparisonValidationError(f"{context} must be an object")
    return value


def _require_list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ComparisonValidationError(f"{context} must be an array")
    return value


def _validate_dataset_identity(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    dataset = _require_mapping(report.get("dataset"), f"{label}.dataset")
    release = _require_mapping(
        dataset.get("release_identity"), f"{label}.dataset.release_identity"
    )
    identity = _require_mapping(
        release.get("identity"), f"{label}.dataset.release_identity.identity"
    )
    if dict(identity) != dict(_CANONICAL_RELEASE_IDENTITY):
        raise ComparisonValidationError(
            f"{label}: dataset release identity is not the canonical public H&E fold release"
        )
    observed_hash = release.get("canonical_identity_sha256")
    if not _is_sha256(observed_hash) or observed_hash != _canonical_sha256(identity):
        raise ComparisonValidationError(
            f"{label}: dataset canonical release identity hash does not recompute"
        )
    if release.get("verified") is not True:
        raise ComparisonValidationError(
            f"{label}: dataset release identity is not verified"
        )
    if dataset.get("release_identity_verified") is not True:
        raise ComparisonValidationError(
            f"{label}: dataset identity verification flag is false"
        )
    if dataset.get("asset_content_hashes_computed") is not True:
        raise ComparisonValidationError(
            f"{label}: dataset asset hashes were not computed"
        )
    verified_components = release.get("verified_components")
    if verified_components != sorted(identity):
        raise ComparisonValidationError(
            f"{label}: verified dataset components do not exactly match identity"
        )
    for key, expected in identity.items():
        if dataset.get(key) != expected:
            raise ComparisonValidationError(
                f"{label}: dataset field {key!r} disagrees with release identity"
            )
    canonical_fields = {
        "dataset_name": "Histology Tissue Fold Dataset",
        "dataset_version": "1.0",
        "license": "CC BY 4.0",
        "claimable_artifacts": ["tissue_fold"],
        "crack_reference_available": False,
        "data_origin": "real microscope-acquired H&E teaching-slide fields",
        "empty_positive_mask_policy": "exclude_localization",
        "n_records": 2127,
        "n_slides": 283,
    }
    for field, expected in canonical_fields.items():
        if dataset.get(field) != expected:
            raise ComparisonValidationError(
                f"{label}: dataset {field!r} is not canonical"
            )
    if release.get("identity_version") != _CANONICAL_RELEASE_IDENTITY_VERSION:
        raise ComparisonValidationError(
            f"{label}: dataset release identity version is not canonical"
        )
    identity_contract = {
        **canonical_fields,
        "release_identity": {
            "identity_version": release.get("identity_version"),
            "canonical_identity_sha256": observed_hash,
            "identity": dict(identity),
        },
    }
    return identity_contract


def _validate_split_manifest(
    split_report: Mapping[str, Any], label: str, role: str
) -> tuple[list[Mapping[str, Any]], str]:
    manifest_raw = _require_list(
        split_report.get("manifest"), f"{label}.splits.{role}.manifest"
    )
    manifest_hash = split_report.get("manifest_sha256")
    if not _is_sha256(manifest_hash) or manifest_hash != _canonical_sha256(
        manifest_raw
    ):
        raise ComparisonValidationError(
            f"{label}: {role} manifest SHA-256 does not recompute"
        )
    if not manifest_raw:
        raise ComparisonValidationError(f"{label}: {role} manifest is empty")

    manifest: list[Mapping[str, Any]] = []
    filenames: set[str] = set()
    slides: set[str] = set()
    counts: Counter[str] = Counter()
    localization_references = 0
    localization_exclusions: list[str] = []
    for index, raw_item in enumerate(manifest_raw):
        item = _require_mapping(raw_item, f"{label}.{role}.manifest[{index}]")
        for field in ("image_filename", "organ", "class", "slide_id"):
            if not isinstance(item.get(field), str) or not item[field]:
                raise ComparisonValidationError(
                    f"{label}: manifest item {index} has invalid {field}"
                )
        filename = item["image_filename"]
        if filename in filenames:
            raise ComparisonValidationError(
                f"{label}: duplicate {role} field {filename!r}"
            )
        filenames.add(filename)
        class_name = item["class"]
        if class_name not in {"clean", "tissue_fold"}:
            raise ComparisonValidationError(
                f"{label}: unsupported locked-test class {class_name!r}"
            )
        reference_valid = item.get("localization_reference_valid")
        if not isinstance(reference_valid, bool):
            raise ComparisonValidationError(
                f"{label}: manifest localization_reference_valid must be boolean"
            )
        image_sha256 = item.get("image_sha256")
        mask_sha256 = item.get("mask_sha256")
        if not _is_sha256(image_sha256):
            raise ComparisonValidationError(
                f"{label}: invalid image SHA-256 for {filename!r}"
            )
        if mask_sha256 is not None and not _is_sha256(mask_sha256):
            raise ComparisonValidationError(
                f"{label}: invalid mask SHA-256 for {filename!r}"
            )
        if class_name == "clean" and mask_sha256 is not None:
            raise ComparisonValidationError(
                f"{label}: clean field {filename!r} unexpectedly has a mask hash"
            )
        if class_name == "tissue_fold" and reference_valid and mask_sha256 is None:
            raise ComparisonValidationError(
                f"{label}: valid positive field {filename!r} lacks a mask hash"
            )
        slides.add(item["slide_id"])
        counts[f"{item['organ']}/{class_name}"] += 1
        localization_references += int(reference_valid)
        if not reference_valid:
            localization_exclusions.append(str(filename))
        manifest.append(item)

    expected_slide_ids = sorted(slides)
    expected_counts = dict(sorted(counts.items()))
    derived_checks = {
        "n_images": len(manifest),
        "n_slides": len(slides),
        "n_localization_references": localization_references,
        "slide_ids": expected_slide_ids,
        "counts": expected_counts,
        "localization_exclusions": sorted(localization_exclusions),
    }
    for field, expected in derived_checks.items():
        if split_report.get(field) != expected:
            raise ComparisonValidationError(
                f"{label}: {role} {field} disagrees with its manifest"
            )
    return manifest, manifest_hash


def _validate_splits_and_leakage(
    report: Mapping[str, Any], label: str, dataset_identity: Mapping[str, Any]
) -> tuple[dict[str, list[Mapping[str, Any]]], dict[str, str], dict[str, Any]]:
    splits = _require_mapping(report.get("splits"), f"{label}.splits")
    if set(splits) != set(_SPLIT_ROLES):
        raise ComparisonValidationError(
            f"{label}: splits must contain exactly {', '.join(_SPLIT_ROLES)}"
        )
    manifests: dict[str, list[Mapping[str, Any]]] = {}
    manifest_hashes: dict[str, str] = {}
    for role in _SPLIT_ROLES:
        split_report = _require_mapping(splits.get(role), f"{label}.splits.{role}")
        manifest, manifest_hash = _validate_split_manifest(split_report, label, role)
        manifests[role] = manifest
        manifest_hashes[role] = manifest_hash

    field_roles: dict[str, str] = {}
    slide_contract: dict[str, tuple[str, str, str]] = {}
    derived_assignment: list[dict[str, Any]] = []
    for role in _SPLIT_ROLES:
        by_slide: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for item in manifests[role]:
            filename = str(item["image_filename"])
            if filename in field_roles:
                raise ComparisonValidationError(
                    f"{label}: field {filename!r} occurs in multiple splits"
                )
            field_roles[filename] = role
            slide_id = str(item["slide_id"])
            stratum = (str(item["organ"]), str(item["class"]), role)
            prior = slide_contract.get(slide_id)
            if prior is not None and prior != stratum:
                raise ComparisonValidationError(
                    f"{label}: slide {slide_id!r} maps to multiple organ/class/split strata"
                )
            slide_contract[slide_id] = stratum
            by_slide[slide_id].append(item)
        derived_assignment.extend(
            {
                "organ": records[0]["organ"],
                "class": records[0]["class"],
                "slide_id": slide_id,
                "role": role,
                "n_images": len(records),
            }
            for slide_id, records in by_slide.items()
        )
    derived_assignment.sort(
        key=lambda item: (item["organ"], item["class"], item["slide_id"], item["role"])
    )
    if len(field_roles) != dataset_identity["n_records"]:
        raise ComparisonValidationError(
            f"{label}: split manifests do not cover the canonical dataset record count"
        )
    if len(slide_contract) != dataset_identity["n_slides"]:
        raise ComparisonValidationError(
            f"{label}: split manifests do not cover the canonical dataset slide count"
        )

    slide_sets = {
        role: {str(item["slide_id"]) for item in manifests[role]}
        for role in _SPLIT_ROLES
    }
    recomputed_leakage = {
        "group_unit": "source_slide_id",
        "fit_calibration_overlap": len(slide_sets["fit"] & slide_sets["calibration"]),
        "fit_test_overlap": len(slide_sets["fit"] & slide_sets["locked_test"]),
        "calibration_test_overlap": len(
            slide_sets["calibration"] & slide_sets["locked_test"]
        ),
        "passed": not (
            slide_sets["fit"] & slide_sets["calibration"]
            or slide_sets["fit"] & slide_sets["locked_test"]
            or slide_sets["calibration"] & slide_sets["locked_test"]
        ),
    }
    leakage = _require_mapping(report.get("leakage_audit"), f"{label}.leakage_audit")
    if dict(leakage) != recomputed_leakage or not recomputed_leakage["passed"]:
        raise ComparisonValidationError(
            f"{label}: split leakage audit does not recompute from manifests"
        )

    protocol = _require_mapping(report.get("split_protocol"), f"{label}.split_protocol")
    assignment = _require_list(
        protocol.get("assignment_manifest"),
        f"{label}.split_protocol.assignment_manifest",
    )
    assignment_hash = protocol.get("assignment_manifest_sha256")
    if (
        not _is_sha256(assignment_hash)
        or assignment_hash != _canonical_sha256(assignment)
        or assignment != derived_assignment
    ):
        raise ComparisonValidationError(
            f"{label}: split assignment manifest/hash does not recompute"
        )
    if protocol.get("group_unit") != "provided_source_slide_id":
        raise ComparisonValidationError(f"{label}: split grouping unit is invalid")
    if protocol.get("full_record_coverage") is not True:
        raise ComparisonValidationError(
            f"{label}: split does not have full record coverage"
        )
    if protocol.get("smoke_limit_applied") is not False:
        raise ComparisonValidationError(
            f"{label}: smoke-limited split is not comparable"
        )
    if not isinstance(protocol.get("protocol"), str) or not protocol["protocol"]:
        raise ComparisonValidationError(f"{label}: split protocol identity is missing")
    split_contract = {
        "protocol": protocol["protocol"],
        "group_unit": protocol["group_unit"],
        "requested_role_fractions": protocol.get("requested_role_fractions"),
        "full_record_coverage": True,
        "smoke_limit_applied": False,
        "assignment_manifest_sha256": assignment_hash,
        "manifest_sha256_by_role": manifest_hashes,
    }
    return manifests, manifest_hashes, split_contract


def _validate_claim_scope(report: Mapping[str, Any], label: str) -> dict[str, Any]:
    if report.get("status") != _CANONICAL_STATUS:
        raise ComparisonValidationError(
            f"{label}: status is not the canonical reportable public benchmark status"
        )
    claim_scope = _require_mapping(report.get("claim_scope"), f"{label}.claim_scope")
    if dict(claim_scope) != dict(_CANONICAL_CLAIM_SCOPE):
        raise ComparisonValidationError(
            f"{label}: claim scope is not the canonical public H&E fold-only scope"
        )
    return dict(claim_scope)


def _validate_configuration(
    report: Mapping[str, Any],
    label: str,
    method_ids: Sequence[str],
    split_contract: Mapping[str, Any],
    dataset_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], str, list[str]]:
    configuration = _require_mapping(
        report.get("configuration"), f"{label}.configuration"
    )
    configuration_sha256 = report.get("configuration_sha256")
    if not _is_sha256(
        configuration_sha256
    ) or configuration_sha256 != _canonical_sha256(configuration):
        raise ComparisonValidationError(
            f"{label}: top-level configuration SHA-256 does not recompute"
        )
    missing = set(_EVALUATION_CONTRACT_CONFIGURATION_FIELDS) - set(configuration)
    if missing:
        raise ComparisonValidationError(
            f"{label}: comparison-critical configuration fields missing: "
            + ", ".join(sorted(missing))
        )
    configured_methods = configuration.get("methods")
    if (
        not isinstance(configured_methods, list)
        or len(configured_methods) != len(set(configured_methods))
        or set(configured_methods) != set(method_ids)
        or any(method not in _METHOD_IDS for method in configured_methods)
    ):
        raise ComparisonValidationError(
            f"{label}: configuration method set is invalid or inconsistent"
        )
    for field in ("max_dimension", "tile_size", "tile_stride"):
        value = configuration.get(field)
        if not _is_nonnegative_int(value) or value == 0:
            raise ComparisonValidationError(
                f"{label}: configuration {field} must be a positive integer"
            )
    if not isinstance(configuration.get("seed"), int) or isinstance(
        configuration.get("seed"), bool
    ):
        raise ComparisonValidationError(f"{label}: split seed must be an integer")
    fractions: dict[str, float] = {}
    for role, field in (
        ("fit", "fit_fraction"),
        ("calibration", "calibration_fraction"),
        ("locked_test", "test_fraction"),
    ):
        value = configuration.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0.0 < float(value) < 1.0
        ):
            raise ComparisonValidationError(f"{label}: invalid split fraction {field}")
        fractions[role] = float(value)
    if not math.isclose(math.fsum(fractions.values()), 1.0, abs_tol=1e-12):
        raise ComparisonValidationError(f"{label}: split fractions do not sum to one")
    if split_contract.get("requested_role_fractions") != fractions:
        raise ComparisonValidationError(
            f"{label}: split protocol fractions disagree with configuration"
        )
    required_true = ("strict_public_v1", "validate_asset_dimensions", "hash_assets")
    if any(configuration.get(field) is not True for field in required_true):
        raise ComparisonValidationError(
            f"{label}: strict dataset validation/hash configuration is disabled"
        )
    if configuration.get("limit_slides_per_stratum_per_split") is not None:
        raise ComparisonValidationError(
            f"{label}: smoke-limited configuration is invalid"
        )
    if (
        configuration.get("empty_positive_mask_policy") != "exclude_localization"
        or configuration.get("empty_positive_mask_policy")
        != dataset_identity["empty_positive_mask_policy"]
    ):
        raise ComparisonValidationError(
            f"{label}: empty-positive mask policy is inconsistent"
        )
    for field in ("calibration_score_sample", "threshold_candidates"):
        value = configuration.get(field)
        if not _is_nonnegative_int(value) or value == 0:
            raise ComparisonValidationError(
                f"{label}: configuration {field} must be a positive integer"
            )
    image_quantile = configuration.get("image_score_quantile")
    if (
        not isinstance(image_quantile, (int, float))
        or isinstance(image_quantile, bool)
        or not 0.0 < float(image_quantile) < 1.0
    ):
        raise ComparisonValidationError(
            f"{label}: image_score_quantile must lie strictly between zero and one"
        )
    contract = {
        "configuration": {
            field: configuration[field]
            for field in _EVALUATION_CONTRACT_CONFIGURATION_FIELDS
        },
        "split": dict(split_contract),
        "reference": {
            "dataset_release_identity_sha256": dataset_identity["release_identity"][
                "canonical_identity_sha256"
            ],
            "localization_exclusion_manifest_sha256": dataset_identity[
                "release_identity"
            ]["identity"]["localization_exclusion_manifest_sha256"],
            "empty_positive_mask_policy": dataset_identity[
                "empty_positive_mask_policy"
            ],
        },
    }
    return contract, configuration_sha256, list(configured_methods)


def _expected_method_identity(method_id: str) -> dict[str, Any]:
    if method_id == "classical_fold":
        return {
            "reported_method_id": method_id,
            "algorithm_family": "classical_fold_candidates",
            "foundation_encoder_required": False,
            "legacy_encoder_specific_alias": False,
        }
    if method_id not in _METHOD_IDS:
        raise ComparisonValidationError(f"unsupported method identity: {method_id!r}")
    head = "patchknn" if method_id.endswith("patchknn") else "linear_probe"
    return {
        "reported_method_id": method_id,
        "algorithm_family": head,
        "foundation_encoder_required": True,
        "encoder_identity_location": "top-level model_identity when invoked by CLI",
        "legacy_encoder_specific_alias": method_id.startswith("dinov2_"),
        "canonical_encoder_agnostic_method_id": f"foundation_{head}",
    }


def _validate_model_identity(
    report: Mapping[str, Any],
    label: str,
    *,
    foundation_requested: bool,
    method_model: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> str | None:
    raw_model_identity = report.get("model_identity")
    if not foundation_requested:
        if raw_model_identity is not None:
            raise ComparisonValidationError(
                f"{label}: classical-only report unexpectedly has model_identity"
            )
        classical_identity = {
            "id": method_model.get("model_id"),
            "loader": method_model.get("loader_identity"),
        }
        if method_model.get("model_config_sha256") != _canonical_sha256(
            classical_identity
        ):
            raise ComparisonValidationError(
                f"{label}: classical model configuration identity does not recompute"
            )
        return None
    model_identity = _require_mapping(raw_model_identity, f"{label}.model_identity")
    if model_identity.get("id") != method_model.get("model_id"):
        raise ComparisonValidationError(
            f"{label}: top-level and provenance model identifiers disagree"
        )
    declared_loader = model_identity.get("loader")
    if declared_loader is None:
        required_hf_fields = (
            "configuration_files",
            "weight_files",
            "requested_revision",
            "resolved_revision",
            "token_used",
        )
        if any(field not in model_identity for field in required_hf_fields):
            raise ComparisonValidationError(
                f"{label}: loader-less model identity is not the canonical Transformers schema"
            )
        requested_revision = model_identity.get("requested_revision")
        resolved_revision = model_identity.get("resolved_revision")
        if (
            not isinstance(requested_revision, str)
            or len(requested_revision) != 40
            or any(
                character not in "0123456789abcdef" for character in requested_revision
            )
            or resolved_revision != requested_revision
            or model_identity.get("trust_remote_code") is not False
            or model_identity.get("network_access_allowed") is not False
            or model_identity.get("token_used") is not False
        ):
            raise ComparisonValidationError(
                f"{label}: loader-less Transformers identity is not exact, offline, and remote-code-disabled"
            )
        expected_loader = "transformers_pretrained_trust_remote_code_false"
    elif isinstance(declared_loader, str) and declared_loader:
        expected_loader = declared_loader
    else:
        raise ComparisonValidationError(
            f"{label}: explicit top-level model loader identity is invalid"
        )
    if expected_loader != method_model.get("loader_identity"):
        raise ComparisonValidationError(
            f"{label}: top-level and provenance model loaders disagree"
        )
    if model_identity.get("resolved_device") != execution.get("device"):
        raise ComparisonValidationError(
            f"{label}: model and provenance execution devices disagree"
        )
    if model_identity.get("trust_remote_code") is not False or not isinstance(
        model_identity.get("network_access_allowed"), bool
    ):
        raise ComparisonValidationError(
            f"{label}: foundation model remote-code/network identity is invalid"
        )
    if "token_used" in model_identity and model_identity.get("token_used") is not False:
        raise ComparisonValidationError(f"{label}: model token-use identity is invalid")
    if not isinstance(model_identity.get("input"), dict) or not model_identity["input"]:
        raise ComparisonValidationError(f"{label}: model input identity is missing")

    def validate_asset(raw_asset: Any, context: str) -> str:
        asset = _require_mapping(raw_asset, context)
        location = asset.get("path", asset.get("filename"))
        if not isinstance(location, str) or not location:
            raise ComparisonValidationError(
                f"{label}: model asset path/filename is missing"
            )
        if not _is_sha256(asset.get("sha256")):
            raise ComparisonValidationError(f"{label}: model asset SHA-256 is invalid")
        if not _is_nonnegative_int(asset.get("size_bytes")) or asset["size_bytes"] == 0:
            raise ComparisonValidationError(f"{label}: model asset size is invalid")
        return str(asset["sha256"])

    weight_candidates: list[str] = []
    if "weights" in model_identity:
        weight_candidates.append(
            validate_asset(model_identity["weights"], f"{label}.model_identity.weights")
        )
    if "weight_files" in model_identity:
        weight_files = _require_list(
            model_identity["weight_files"], f"{label}.model_identity.weight_files"
        )
        weight_candidates.extend(
            validate_asset(item, f"{label}.model_identity.weight_files[{index}]")
            for index, item in enumerate(weight_files)
        )
    if "configuration_files" in model_identity:
        configuration_files = _require_list(
            model_identity["configuration_files"],
            f"{label}.model_identity.configuration_files",
        )
        for index, item in enumerate(configuration_files):
            validate_asset(item, f"{label}.model_identity.configuration_files[{index}]")
    if "assets" in model_identity:
        assets = _require_mapping(
            model_identity["assets"], f"{label}.model_identity.assets"
        )
        if not assets:
            raise ComparisonValidationError(
                f"{label}: foundation model asset identity is empty"
            )
        validated_assets = {
            str(name): validate_asset(
                raw_asset, f"{label}.model_identity.assets.{name}"
            )
            for name, raw_asset in assets.items()
        }
        if "model.safetensors" in validated_assets:
            weight_candidates.append(validated_assets["model.safetensors"])
    unique_weights = sorted(set(weight_candidates))
    if not unique_weights:
        raise ComparisonValidationError(
            f"{label}: foundation model has no exact weight asset identity"
        )
    expected_weights_sha256 = (
        unique_weights[0]
        if len(unique_weights) == 1
        else _canonical_sha256(unique_weights)
    )
    if method_model.get("weights_sha256") != expected_weights_sha256:
        raise ComparisonValidationError(
            f"{label}: provenance weight SHA-256 disagrees with model assets"
        )
    model_identity_sha256 = _canonical_sha256(model_identity)
    if method_model.get("model_config_sha256") != model_identity_sha256:
        raise ComparisonValidationError(
            f"{label}: top-level model configuration identity does not recompute"
        )
    return model_identity_sha256


def _validate_provenance(
    report: Mapping[str, Any],
    label: str,
    *,
    configuration_sha256: str,
    configured_methods: Sequence[str],
) -> tuple[str, str, dict[str, Any], str | None]:
    provenance = _require_mapping(
        report.get("run_provenance"), f"{label}.run_provenance"
    )
    if provenance.get("provided") is not True or provenance.get("valid") is not True:
        raise ComparisonValidationError(f"{label}: run provenance is not valid")
    if provenance.get("validated_before_scoring") is not True:
        raise ComparisonValidationError(
            f"{label}: run provenance was not validated before scoring"
        )
    if provenance.get("validation_errors") != []:
        raise ComparisonValidationError(
            f"{label}: run provenance contains validation errors"
        )
    value = _require_mapping(provenance.get("value"), f"{label}.run_provenance.value")
    identity_sha256 = provenance.get("identity_sha256")
    if not _is_sha256(identity_sha256) or identity_sha256 != _canonical_sha256(value):
        raise ComparisonValidationError(
            f"{label}: run provenance identity SHA-256 does not recompute"
        )
    schema_version = provenance.get("schema_version")
    if (
        schema_version != "public-fold-run-provenance-1.1"
        or value.get("schema_version") != schema_version
    ):
        raise ComparisonValidationError(
            f"{label}: run provenance schema is inconsistent"
        )
    capture = _require_mapping(value.get("capture"), f"{label}.provenance.capture")
    if (
        capture.get("captured_before_scoring") is not True
        or capture.get("validation_status") != "structurally_validated"
        or not isinstance(capture.get("validator_id"), str)
        or not capture["validator_id"]
        or not isinstance(capture.get("approval_scope"), str)
        or not capture["approval_scope"]
    ):
        raise ComparisonValidationError(
            f"{label}: provenance capture identity is invalid"
        )

    code = _require_mapping(value.get("code"), f"{label}.provenance.code")
    identity_type = code.get("identity_type")
    if identity_type == "git":
        commit = code.get("commit")
        if (
            not isinstance(commit, str)
            or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit)
            or not _is_sha256(code.get("dirty_diff_sha256"))
            or not isinstance(code.get("dirty_diff_capture"), str)
            or not code["dirty_diff_capture"]
        ):
            raise ComparisonValidationError(f"{label}: Git code identity is invalid")
        untracked = code.get("untracked_runtime_sources", [])
        if not isinstance(untracked, list) or any(
            not isinstance(item, str) or not item for item in untracked
        ):
            raise ComparisonValidationError(
                f"{label}: untracked runtime-source identity is invalid"
            )
    elif identity_type == "wheel":
        if not _is_sha256(code.get("wheel_sha256")):
            raise ComparisonValidationError(f"{label}: wheel code identity is invalid")
    else:
        raise ComparisonValidationError(f"{label}: code identity type is invalid")

    environment = _require_mapping(
        value.get("environment"), f"{label}.provenance.environment"
    )
    if any(
        not isinstance(environment.get(field), str) or not environment[field]
        for field in ("python_version", "platform")
    ):
        raise ComparisonValidationError(f"{label}: environment identity is incomplete")
    dependencies = _require_mapping(
        environment.get("dependencies"), f"{label}.provenance.environment.dependencies"
    )
    for dependency in ("numpy", "scipy", "opencv"):
        if (
            not isinstance(dependencies.get(dependency), str)
            or not dependencies[dependency]
        ):
            raise ComparisonValidationError(
                f"{label}: dependency identity {dependency!r} is missing"
            )

    method_model = _require_mapping(
        value.get("method_model"), f"{label}.provenance.method_model"
    )
    if method_model.get("selected_methods") != list(configured_methods):
        raise ComparisonValidationError(
            f"{label}: provenance selected methods disagree with configuration"
        )
    if method_model.get("benchmark_configuration_sha256") != configuration_sha256:
        raise ComparisonValidationError(
            f"{label}: provenance configuration identity disagrees with report"
        )
    if (
        method_model.get("implementation_id")
        != "foldcrack_qc.public_fold_benchmark:v1.2"
    ):
        raise ComparisonValidationError(
            f"{label}: benchmark implementation identity is invalid"
        )
    for field in ("model_id", "loader_identity"):
        if not isinstance(method_model.get(field), str) or not method_model[field]:
            raise ComparisonValidationError(
                f"{label}: provenance model field {field!r} is missing"
            )
    if not _is_sha256(method_model.get("model_config_sha256")):
        raise ComparisonValidationError(
            f"{label}: model configuration identity is invalid"
        )
    if (
        method_model.get("frozen_evaluation") is not True
        or method_model.get("transductive_updates") is not False
    ):
        raise ComparisonValidationError(
            f"{label}: provenance does not describe frozen non-transductive evaluation"
        )
    foundation_requested = any(
        method != "classical_fold" for method in configured_methods
    )
    if foundation_requested:
        for dependency in ("torch", "transformers", "huggingface_hub"):
            if (
                not isinstance(dependencies.get(dependency), str)
                or not dependencies[dependency]
            ):
                raise ComparisonValidationError(
                    f"{label}: foundation dependency identity {dependency!r} is missing"
                )
        if method_model.get("weights_not_applicable") is not False or not _is_sha256(
            method_model.get("weights_sha256")
        ):
            raise ComparisonValidationError(
                f"{label}: foundation weight identity is invalid"
            )
    elif method_model.get("weights_not_applicable") is not True or method_model.get(
        "weights_sha256"
    ) not in (None, ""):
        raise ComparisonValidationError(
            f"{label}: classical weight identity is invalid"
        )

    execution = _require_mapping(
        value.get("execution"), f"{label}.provenance.execution"
    )
    if any(
        not isinstance(execution.get(field), str) or not execution[field]
        for field in ("device", "precision")
    ):
        raise ComparisonValidationError(f"{label}: execution identity is incomplete")
    model_identity_sha256 = _validate_model_identity(
        report,
        label,
        foundation_requested=foundation_requested,
        method_model=method_model,
        execution=execution,
    )
    execution_identity = {
        "code_sha256": _canonical_sha256(code),
        "environment_sha256": _canonical_sha256(environment),
        "method_model_sha256": _canonical_sha256(method_model),
        "execution": dict(execution),
    }
    return identity_sha256, schema_version, execution_identity, model_identity_sha256


def _validate_outcome_table(
    *,
    label: str,
    method_id: str,
    method_report: Mapping[str, Any],
    manifest_by_field: Mapping[str, Mapping[str, Any]],
) -> tuple[
    dict[str, float],
    dict[str, tuple[str, str]],
    dict[str, tuple[int, int, int]],
]:
    outcomes = _require_list(
        method_report.get("locked_test_outcomes"),
        f"{label}.methods.{method_id}.locked_test_outcomes",
    )
    outcome_hash = method_report.get("locked_test_outcomes_sha256")
    if not _is_sha256(outcome_hash) or outcome_hash != _canonical_sha256(outcomes):
        raise ComparisonValidationError(
            f"{label}/{method_id}: locked-test outcome SHA-256 does not recompute"
        )
    if method_report.get("method") != method_id:
        raise ComparisonValidationError(
            f"{label}/{method_id}: embedded method identifier is inconsistent"
        )
    if len(outcomes) != len(manifest_by_field):
        raise ComparisonValidationError(
            f"{label}/{method_id}: outcome table does not cover locked test"
        )

    seen: set[str] = set()
    positive_dice: dict[str, float] = {}
    positive_metadata: dict[str, tuple[str, str]] = {}
    evaluation_domain: dict[str, tuple[int, int, int]] = {}
    for index, raw_outcome in enumerate(outcomes):
        outcome = _require_mapping(
            raw_outcome, f"{label}.{method_id}.outcomes[{index}]"
        )
        field_key = outcome.get("field_key")
        if not isinstance(field_key, str) or field_key not in manifest_by_field:
            raise ComparisonValidationError(
                f"{label}/{method_id}: unknown outcome field {field_key!r}"
            )
        if field_key in seen:
            raise ComparisonValidationError(
                f"{label}/{method_id}: duplicate outcome field {field_key!r}"
            )
        seen.add(field_key)
        manifest_item = manifest_by_field[field_key]
        expected_label = int(manifest_item["class"] == "tissue_fold")
        metadata_checks = {
            "organ": manifest_item["organ"],
            "source_slide_id": manifest_item["slide_id"],
            "label": expected_label,
            "localization_reference_valid": manifest_item[
                "localization_reference_valid"
            ],
        }
        if not _is_nonnegative_int(outcome.get("label")) or outcome["label"] not in (
            0,
            1,
        ):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} has invalid label type/value"
            )
        if not isinstance(outcome.get("localization_reference_valid"), bool):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} has invalid reference-valid flag"
            )
        for field, expected in metadata_checks.items():
            if outcome.get(field) != expected:
                raise ComparisonValidationError(
                    f"{label}/{method_id}: {field_key!r} has inconsistent {field}"
                )
        confusion: dict[str, int] = {}
        for field in ("tp", "fp", "fn", "tn", "n_valid"):
            value = outcome.get(field)
            if not _is_nonnegative_int(value):
                raise ComparisonValidationError(
                    f"{label}/{method_id}: {field_key!r} has invalid {field}"
                )
            confusion[field] = value
        if (
            sum(confusion[field] for field in ("tp", "fp", "fn", "tn"))
            != confusion["n_valid"]
        ):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} confusion counts do not sum"
            )
        evaluation_domain[field_key] = (
            confusion["n_valid"],
            confusion["tp"] + confusion["fn"],
            confusion["fp"] + confusion["tn"],
        )
        image_prediction = outcome.get("image_prediction")
        image_score = outcome.get("image_score")
        runtime_seconds = outcome.get("runtime_seconds")
        if not _is_nonnegative_int(image_prediction) or image_prediction not in (0, 1):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} has invalid image prediction"
            )
        if (
            not isinstance(image_score, (int, float))
            or isinstance(image_score, bool)
            or not math.isfinite(float(image_score))
        ):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} has invalid image score"
            )
        if (
            not isinstance(runtime_seconds, (int, float))
            or isinstance(runtime_seconds, bool)
            or not math.isfinite(float(runtime_seconds))
            or runtime_seconds < 0
        ):
            raise ComparisonValidationError(
                f"{label}/{method_id}: {field_key!r} has invalid runtime"
            )
        if expected_label == 1 and manifest_item["localization_reference_valid"]:
            denominator = 2 * confusion["tp"] + confusion["fp"] + confusion["fn"]
            if confusion["tp"] + confusion["fn"] <= 0 or denominator <= 0:
                raise ComparisonValidationError(
                    f"{label}/{method_id}: positive field {field_key!r} has no reference pixels"
                )
            positive_dice[field_key] = 2 * confusion["tp"] / denominator
            positive_metadata[field_key] = (
                str(manifest_item["organ"]),
                str(manifest_item["slide_id"]),
            )

    if seen != set(manifest_by_field):
        raise ComparisonValidationError(
            f"{label}/{method_id}: outcome field set does not equal locked test"
        )
    if not positive_dice:
        raise ComparisonValidationError(
            f"{label}/{method_id}: no valid positive localization fields"
        )
    locked_test = _require_mapping(
        method_report.get("locked_test"), f"{label}.{method_id}.locked_test"
    )
    positive_macro = _require_mapping(
        locked_test.get("positive_field_macro"),
        f"{label}.{method_id}.locked_test.positive_field_macro",
    )
    reported_dice = _require_mapping(
        positive_macro.get("dice"), f"{label}.{method_id}.positive_field_macro.dice"
    )
    point_mean = math.fsum(positive_dice.values()) / len(positive_dice)
    reported_mean = reported_dice.get("mean")
    if (
        not isinstance(reported_mean, (int, float))
        or isinstance(reported_mean, bool)
        or not math.isclose(float(reported_mean), point_mean, abs_tol=1e-12)
    ):
        raise ComparisonValidationError(
            f"{label}/{method_id}: reported positive-field Dice mean does not recompute"
        )
    if reported_dice.get("n") != len(positive_dice):
        raise ComparisonValidationError(
            f"{label}/{method_id}: reported positive-field count does not recompute"
        )
    return positive_dice, positive_metadata, evaluation_domain


def _validate_report(
    label: str, path: Path, selected_method: str | None
) -> _ValidatedReport:
    report, artifact_sha256 = _load_strict_json(path)
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ComparisonValidationError(
            f"{label}: expected schema {REPORT_SCHEMA!r}, got {report.get('schema_version')!r}"
        )
    if report.get("report_eligible") is not True:
        raise ComparisonValidationError(f"{label}: report_eligible is not true")
    if report.get("execution_status") != "complete":
        raise ComparisonValidationError(f"{label}: execution is not complete")
    if report.get("nonreportable_reasons") != []:
        raise ComparisonValidationError(f"{label}: nonreportable reasons are present")
    claim_scope = _validate_claim_scope(report, label)
    dataset_identity = _validate_dataset_identity(report, label)
    manifests, manifest_hashes, split_contract = _validate_splits_and_leakage(
        report, label, dataset_identity
    )
    manifest = manifests["locked_test"]
    manifest_sha256 = manifest_hashes["locked_test"]
    methods = _require_mapping(report.get("methods"), f"{label}.methods")
    if not methods:
        raise ComparisonValidationError(f"{label}: report contains no methods")
    if any(not isinstance(method_id, str) for method_id in methods):
        raise ComparisonValidationError(f"{label}: method identifiers must be strings")
    evaluation_contract, configuration_sha256, configured_methods = (
        _validate_configuration(
            report,
            label,
            list(methods),
            split_contract,
            dataset_identity,
        )
    )
    (
        provenance_sha256,
        provenance_schema,
        execution_identity,
        model_identity_sha256,
    ) = _validate_provenance(
        report,
        label,
        configuration_sha256=configuration_sha256,
        configured_methods=configured_methods,
    )
    if selected_method is None:
        if len(methods) != 1:
            raise ComparisonValidationError(
                f"{label}: --method {label}=METHOD_ID is required for a multi-method report"
            )
        selected_method = next(iter(methods))
    if selected_method not in methods:
        raise ComparisonValidationError(
            f"{label}: selected method {selected_method!r} does not exist"
        )

    manifest_by_field = {str(item["image_filename"]): item for item in manifest}
    selected_dice: dict[str, float] | None = None
    selected_metadata: dict[str, tuple[str, str]] | None = None
    selected_domain: dict[str, tuple[int, int, int]] | None = None
    reference_domain: dict[str, tuple[int, int, int]] | None = None
    for method_id, raw_method_report in methods.items():
        if not isinstance(method_id, str) or method_id not in _METHOD_IDS:
            raise ComparisonValidationError(f"{label}: invalid method key")
        method_report = _require_mapping(
            raw_method_report, f"{label}.methods.{method_id}"
        )
        method_identity = _require_mapping(
            method_report.get("method_identity"),
            f"{label}.methods.{method_id}.method_identity",
        )
        if dict(method_identity) != _expected_method_identity(method_id):
            raise ComparisonValidationError(
                f"{label}/{method_id}: selected method identity is not exact"
            )
        dice, metadata, evaluation_domain = _validate_outcome_table(
            label=label,
            method_id=method_id,
            method_report=method_report,
            manifest_by_field=manifest_by_field,
        )
        if reference_domain is None:
            reference_domain = evaluation_domain
        elif evaluation_domain != reference_domain:
            raise ComparisonValidationError(
                f"{label}: per-field reference/evaluation domain differs across methods"
            )
        if method_id == selected_method:
            selected_dice = dice
            selected_metadata = metadata
            selected_domain = evaluation_domain
    assert (
        selected_dice is not None
        and selected_metadata is not None
        and selected_domain is not None
    )

    positive_slides: dict[str, set[str]] = defaultdict(set)
    for organ, slide_id in selected_metadata.values():
        positive_slides[organ].add(slide_id)
    insufficient = {
        organ: len(slides)
        for organ, slides in positive_slides.items()
        if len(slides) < 2
    }
    if not positive_slides or insufficient:
        detail = ", ".join(
            f"{organ}={count}" for organ, count in sorted(insufficient.items())
        )
        raise ComparisonValidationError(
            f"{label}: at least two positive source slides per organ are required"
            + (f" ({detail})" if detail else "")
        )

    selected_report = _require_mapping(
        methods[selected_method], f"{label}.methods.{selected_method}"
    )
    method_identity = _require_mapping(
        selected_report.get("method_identity"),
        f"{label}.methods.{selected_method}.method_identity",
    )
    return _ValidatedReport(
        label=label,
        path=path.resolve(),
        artifact_sha256=artifact_sha256,
        selected_method=selected_method,
        method_identity=method_identity,
        method_identity_sha256=_canonical_sha256(method_identity),
        model_identity_sha256=model_identity_sha256,
        provenance_identity_sha256=provenance_sha256,
        provenance_schema_version=provenance_schema,
        execution_identity=execution_identity,
        claim_scope=claim_scope,
        evaluation_contract=evaluation_contract,
        evaluation_contract_sha256=_canonical_sha256(evaluation_contract),
        dataset_identity=dataset_identity,
        dataset_identity_sha256=_canonical_sha256(dataset_identity),
        locked_test_manifest=manifest,
        locked_test_manifest_sha256=manifest_sha256,
        positive_dice=selected_dice,
        positive_metadata=selected_metadata,
        evaluation_domain=selected_domain,
    )


def _parse_named_values(specs: Sequence[str], kind: str) -> dict[str, str]:
    output: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ComparisonValidationError(
                f"{kind} specification must be NAME=VALUE, got {spec!r}"
            )
        label, value = spec.split("=", 1)
        if not _LABEL_PATTERN.fullmatch(label):
            raise ComparisonValidationError(f"invalid report label: {label!r}")
        if not value:
            raise ComparisonValidationError(f"empty {kind} value for {label!r}")
        if label in output:
            raise ComparisonValidationError(f"duplicate {kind} label: {label!r}")
        output[label] = value
    return output


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ComparisonValidationError("cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _interval(values: Sequence[float], confidence: float) -> dict[str, Any]:
    tail = (1.0 - confidence) / 2.0
    return {
        "confidence": confidence,
        "lower": _percentile(values, tail),
        "upper": _percentile(values, 1.0 - tail),
        "method": "stratified_source_slide_cluster_percentile",
        "interpretation": "descriptive_exploratory_only",
    }


def _comparator_runtime_identity() -> dict[str, Any]:
    code_path = Path(__file__).resolve()
    try:
        code_payload = code_path.read_bytes()
    except OSError as error:
        raise ComparisonValidationError(
            f"cannot capture comparator code identity: {error}"
        ) from error
    return {
        "implementation_id": "scripts.compare_hardened_public_fold:v1.1",
        "code_path": str(code_path),
        "code_sha256": hashlib.sha256(code_payload).hexdigest(),
        "runtime": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "byteorder": sys.byteorder,
            "random_generator": "python.random.Random_MT19937_randrange",
            "percentile_interpolation": "linear_(n_minus_1)_order_statistic",
        },
    }


def _bootstrap_means(
    reports: Sequence[_ValidatedReport], *, resamples: int, seed: int
) -> dict[str, list[float]]:
    reference = reports[0]
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for field_key, (organ, slide_id) in reference.positive_metadata.items():
        groups[organ][slide_id].append(field_key)
    ordered_groups = {
        organ: {slide_id: sorted(fields) for slide_id, fields in sorted(slides.items())}
        for organ, slides in sorted(groups.items())
    }
    if not ordered_groups or any(not slides for slides in ordered_groups.values()):
        raise ComparisonValidationError("positive-field cluster structure is empty")

    rng = random.Random(seed)
    values: dict[str, list[float]] = {report.label: [] for report in reports}
    for _ in range(resamples):
        selected_fields: list[str] = []
        for slides in ordered_groups.values():
            slide_ids = list(slides)
            for _ in slide_ids:
                sampled_slide = slide_ids[rng.randrange(len(slide_ids))]
                selected_fields.extend(slides[sampled_slide])
        denominator = len(selected_fields)
        if denominator == 0:
            raise ComparisonValidationError(
                "a cluster bootstrap draw contained no fields"
            )
        for report in reports:
            values[report.label].append(
                math.fsum(report.positive_dice[field] for field in selected_fields)
                / denominator
            )
    return values


def compare_reports(
    report_specs: Sequence[str],
    method_specs: Sequence[str] = (),
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> dict[str, Any]:
    """Validate reports and return a deterministic paired descriptive comparison."""

    if resamples <= 0:
        raise ComparisonValidationError("resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ComparisonValidationError(
            "confidence must be strictly between zero and one"
        )
    named_reports = _parse_named_values(report_specs, "report")
    if len(named_reports) < 2:
        raise ComparisonValidationError("at least two labeled reports are required")
    selected_methods = _parse_named_values(method_specs, "method")
    unknown_method_labels = set(selected_methods) - set(named_reports)
    if unknown_method_labels:
        raise ComparisonValidationError(
            "method selection references unknown reports: "
            + ", ".join(sorted(unknown_method_labels))
        )
    reports = [
        _validate_report(
            label,
            Path(path),
            selected_methods.get(label),
        )
        for label, path in named_reports.items()
    ]

    reference = reports[0]
    for report in reports[1:]:
        if report.dataset_identity != reference.dataset_identity:
            raise ComparisonValidationError(
                f"{report.label}: dataset identity does not exactly match {reference.label}"
            )
        if report.locked_test_manifest != reference.locked_test_manifest:
            raise ComparisonValidationError(
                f"{report.label}: locked-test manifest does not exactly match {reference.label}"
            )
        if report.claim_scope != reference.claim_scope:
            raise ComparisonValidationError(
                f"{report.label}: claim scope does not exactly match {reference.label}"
            )
        if report.evaluation_contract != reference.evaluation_contract:
            raise ComparisonValidationError(
                f"{report.label}: comparison-critical evaluation contract does not exactly match {reference.label}"
            )
        if report.positive_metadata != reference.positive_metadata:
            raise ComparisonValidationError(
                f"{report.label}: positive-field pairing metadata does not match {reference.label}"
            )
        if set(report.positive_dice) != set(reference.positive_dice):
            raise ComparisonValidationError(
                f"{report.label}: positive-field pairing keys do not match {reference.label}"
            )
        if report.evaluation_domain != reference.evaluation_domain:
            raise ComparisonValidationError(
                f"{report.label}: per-field reference/evaluation domain does not exactly match {reference.label}"
            )

    bootstrap = _bootstrap_means(reports, resamples=resamples, seed=seed)
    point_means = {
        report.label: math.fsum(report.positive_dice.values())
        / len(report.positive_dice)
        for report in reports
    }
    report_output: dict[str, Any] = {}
    for report in reports:
        metadata = list(report.positive_metadata.values())
        report_output[report.label] = {
            "path": str(report.path),
            "artifact_sha256": report.artifact_sha256,
            "selected_method": report.selected_method,
            "method_identity": dict(report.method_identity),
            "method_identity_sha256": report.method_identity_sha256,
            "model_identity_sha256": report.model_identity_sha256,
            "run_provenance": {
                "schema_version": report.provenance_schema_version,
                "identity_sha256": report.provenance_identity_sha256,
                "execution_identity": dict(report.execution_identity),
            },
            "positive_field_macro_dice": {
                "point_mean": point_means[report.label],
                "descriptive_bootstrap_ci": _interval(
                    bootstrap[report.label], confidence
                ),
                "n_fields": len(report.positive_dice),
                "n_source_slides": len({slide for _, slide in metadata}),
            },
        }

    paired_differences: list[dict[str, Any]] = []
    for left_index, left in enumerate(reports):
        for right in reports[left_index + 1 :]:
            replicate_differences = [
                left_value - right_value
                for left_value, right_value in zip(
                    bootstrap[left.label], bootstrap[right.label], strict=True
                )
            ]
            paired_differences.append(
                {
                    "contrast": f"{left.label}_minus_{right.label}",
                    "left": left.label,
                    "right": right.label,
                    "point_difference": point_means[left.label]
                    - point_means[right.label],
                    "descriptive_bootstrap_ci": _interval(
                        replicate_differences, confidence
                    ),
                    "same_cluster_draw_used_for_both_methods": True,
                }
            )

    organs = sorted({organ for organ, _ in reference.positive_metadata.values()})
    positive_slides_by_organ = {
        organ: len(
            {
                slide
                for item_organ, slide in reference.positive_metadata.values()
                if item_organ == organ
            }
        )
        for organ in organs
    }
    return {
        "schema_version": COMPARISON_SCHEMA,
        "status": "complete_exploratory_descriptive_paired_comparison",
        "claim_scope": {
            "metric": "positive_localization_field_dice",
            "validated_source_benchmark_claim_scope": dict(reference.claim_scope),
            "descriptive_exploratory_only": True,
            "superiority_claim_made": False,
            "noninferiority_claim_made": False,
            "p_values_computed": False,
        },
        "configuration": {
            "bootstrap_resamples": resamples,
            "seed": seed,
            "confidence": confidence,
            "cluster_unit": "source_slide_id",
            "stratification": "organ",
            "sampling": "clusters_with_replacement_within_each_organ",
            "estimand": "positive_field_macro_mean_dice",
            "draw_pairing": "identical_cluster_draw_for_every_selected_method",
        },
        "comparator_identity": _comparator_runtime_identity(),
        "shared_evidence": {
            "dataset_identity": dict(reference.dataset_identity),
            "dataset_identity_sha256": reference.dataset_identity_sha256,
            "locked_test_manifest_sha256": reference.locked_test_manifest_sha256,
            "evaluation_contract": dict(reference.evaluation_contract),
            "evaluation_contract_sha256": reference.evaluation_contract_sha256,
            "per_field_evaluation_domain_sha256": _canonical_sha256(
                reference.evaluation_domain
            ),
            "n_locked_test_fields": len(reference.locked_test_manifest),
            "n_positive_localization_fields": len(reference.positive_dice),
            "positive_source_slides_by_organ": positive_slides_by_organ,
        },
        "reports": report_output,
        "paired_differences": paired_differences,
        "limitations": [
            "Intervals are exploratory descriptive uncertainty summaries, not hypothesis tests.",
            "No p-values, multiplicity adjustment, superiority claim, or noninferiority claim is provided.",
            "Inference is limited to the exact locked public H&E fold cohort and does not establish COMET or CosMx performance.",
        ],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="labeled hardened report; provide at least twice",
    )
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        metavar="NAME=METHOD_ID",
        help="selected method for a labeled report; required when it has multiple methods",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        named_reports = _parse_named_values(args.report, "report")
        output = args.output.resolve()
        if output in {Path(path).resolve() for path in named_reports.values()}:
            raise ComparisonValidationError(
                "output path must not overwrite an input report"
            )
        result = compare_reports(
            args.report,
            args.method,
            resamples=args.resamples,
            seed=args.seed,
            confidence=args.confidence,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (ComparisonValidationError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
