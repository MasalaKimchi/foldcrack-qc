"""Scientific benchmark contract for real-data fold/crack evaluation.

This module intentionally uses only the Python standard library.  It validates
the *design* of a benchmark before image loading or model execution begins.  In
particular, it separates configuration validity from scientific report
eligibility: a planned benchmark may be structurally valid while remaining
blocked on data, annotations, licenses, or implementations.

Synthetic samples remain useful for unit and integration tests, but this
contract never permits them to support an efficacy or acceptance claim.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_MODALITIES: tuple[str, ...] = ("he", "comet", "cosmx")
REQUIRED_ARTIFACTS: tuple[str, ...] = ("fold", "crack")
ALLOWED_ARTIFACTS: tuple[str, ...] = (*REQUIRED_ARTIFACTS, "artifact_union")
REQUIRED_COHORTS: tuple[str, ...] = ("fit", "calibration", "test")
REQUIRED_SPLIT_KEYS: tuple[str, ...] = (
    "patient_id",
    "block_id",
    "slide_id",
    "run_id",
    "content_id",
)
REQUIRED_REGIMES: tuple[str, ...] = (
    "native_zero_shot",
    "calibrated",
    "clean_reference_anomaly",
    "shallow_adaptation",
    "lora",
    "full_finetune",
)

EXPECTED_REGIME_COHORTS: Mapping[str, frozenset[str]] = {
    "native_zero_shot": frozenset({"test"}),
    "calibrated": frozenset({"calibration", "test"}),
    "clean_reference_anomaly": frozenset({"fit", "calibration", "test"}),
    "shallow_adaptation": frozenset({"fit", "calibration", "test"}),
    "lora": frozenset({"fit", "calibration", "test"}),
    "full_finetune": frozenset({"fit", "calibration", "test"}),
}

TASK_OUTPUT_TYPES: Mapping[str, frozenset[str]] = {
    "presence": frozenset({"slide_score", "slide_label"}),
    "localization": frozenset({"pixel_mask", "dense_score_map", "centerline_mask"}),
    "patch_classification": frozenset({"patch_score", "patch_label"}),
}

FOLD_OUTPUT_TYPES: Mapping[str, frozenset[str]] = {
    "presence": TASK_OUTPUT_TYPES["presence"],
    "localization": frozenset({"pixel_mask", "dense_score_map"}),
    "patch_classification": TASK_OUTPUT_TYPES["patch_classification"],
}

CRACK_OUTPUT_TYPES: Mapping[str, frozenset[str]] = {
    "presence": TASK_OUTPUT_TYPES["presence"],
    "localization": frozenset({"pixel_mask", "dense_score_map", "centerline_mask"}),
    "patch_classification": TASK_OUTPUT_TYPES["patch_classification"],
}

UNION_OUTPUT_TYPES: Mapping[str, frozenset[str]] = {
    "presence": TASK_OUTPUT_TYPES["presence"],
    "localization": frozenset({"pixel_mask", "dense_score_map"}),
    "patch_classification": TASK_OUTPUT_TYPES["patch_classification"],
}

RESOURCE_AVAILABILITY = frozenset({"available", "planned", "unavailable", "restricted"})
LICENSE_STATUSES = frozenset({"approved", "pending", "unknown", "restricted"})
METHOD_AVAILABILITY = frozenset({"ready", "planned", "gated", "unavailable"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_APPROVAL_AREAS = ("data_use", "privacy", "method_assets", "scientific_owner")


class BenchmarkContractError(ValueError):
    """Raised when a contract cannot be loaded or is not eligible as requested."""


@dataclass(frozen=True)
class ContractIssue:
    """One structured contract finding.

    ``error`` denotes an internally invalid contract. ``blocker`` denotes a
    valid plan that cannot yet support a scientific report. ``warning`` records
    an intentional exclusion or non-fatal limitation.
    """

    code: str
    message: str
    path: str = "$"
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class BenchmarkContractReport:
    """Validation result and scientific-report eligibility decision."""

    issues: tuple[ContractIssue, ...]
    eligible_method_ids: tuple[str, ...] = ()
    gated_method_ids: tuple[str, ...] = ()
    cohort_record_counts: tuple[tuple[str, int], ...] = ()

    @property
    def errors(self) -> tuple[ContractIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def blockers(self) -> tuple[ContractIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "blocker")

    @property
    def warnings(self) -> tuple[ContractIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def configuration_valid(self) -> bool:
        return not self.errors

    @property
    def valid(self) -> bool:
        """Alias for structural/configuration validity."""

        return self.configuration_valid

    @property
    def is_valid(self) -> bool:
        return self.configuration_valid

    @property
    def report_eligible(self) -> bool:
        return self.configuration_valid and not self.blockers

    @property
    def status(self) -> str:
        if not self.configuration_valid:
            return "invalid"
        if not self.report_eligible:
            return "configuration_valid_report_blocked"
        return "scientific_report_eligible"

    @property
    def cohort_counts(self) -> dict[str, int]:
        return dict(self.cohort_record_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "configuration_valid": self.configuration_valid,
            "report_eligible": self.report_eligible,
            "eligible_method_ids": list(self.eligible_method_ids),
            "gated_method_ids": list(self.gated_method_ids),
            "cohort_record_counts": self.cohort_counts,
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def raise_for_errors(self) -> None:
        if self.errors:
            first = self.errors[0]
            raise BenchmarkContractError(
                f"Benchmark contract is invalid: {first.code} at {first.path}: "
                f"{first.message}"
            )

    def raise_if_not_report_eligible(self) -> None:
        if not self.report_eligible:
            findings = self.errors or self.blockers
            first = findings[0]
            raise BenchmarkContractError(
                f"Benchmark is not scientifically report-eligible: {first.code} "
                f"at {first.path}: {first.message}"
            )


@dataclass(frozen=True)
class BenchmarkContract:
    """A loaded JSON contract paired with its validation report."""

    data: Mapping[str, Any]
    report: BenchmarkContractReport
    source_path: str | None = None

    @property
    def method_ids(self) -> tuple[str, ...]:
        methods = self.data.get("methods", [])
        if not isinstance(methods, Sequence) or isinstance(methods, (str, bytes)):
            return ()
        return tuple(
            str(method.get("id"))
            for method in methods
            if isinstance(method, Mapping) and method.get("id")
        )

    @property
    def report_eligible(self) -> bool:
        return self.report.report_eligible

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(dict(self.data))


class _Collector:
    def __init__(self) -> None:
        self.issues: list[ContractIssue] = []

    def add(
        self,
        code: str,
        message: str,
        path: str = "$",
        severity: str = "error",
    ) -> None:
        self.issues.append(
            ContractIssue(code=code, message=message, path=path, severity=severity)
        )


def _is_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_present(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (Mapping, Sequence)):
        return bool(value)
    return True


def _read_contract_source(
    source: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    if isinstance(source, Mapping):
        return deepcopy(dict(source)), None
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"Could not load benchmark contract {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkContractError("Benchmark contract JSON root must be an object")
    return payload, str(path.resolve())


def _as_named_mapping(
    value: object,
    *,
    key_name: str,
    collector: _Collector,
    path: str,
) -> dict[str, Mapping[str, Any]]:
    """Normalize either an object keyed by id or a list of objects with an id."""

    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(value, Mapping):
        iterable = []
        for name, item in value.items():
            if isinstance(item, Mapping):
                materialized = dict(item)
                materialized.setdefault(key_name, str(name))
                iterable.append(materialized)
            else:
                iterable.append(item)
    elif _is_sequence(value):
        iterable = list(value)
    else:
        collector.add("expected_collection", "Expected an object or array", path)
        return result

    for index, item in enumerate(iterable):
        item_path = f"{path}[{index}]"
        if not isinstance(item, Mapping):
            collector.add("expected_object", "Entry must be an object", item_path)
            continue
        raw_name = item.get(key_name)
        if not isinstance(raw_name, str) or not raw_name.strip():
            collector.add("missing_id", f"Entry requires non-empty {key_name!r}", item_path)
            continue
        name = raw_name.strip()
        if name in result:
            collector.add("duplicate_id", f"Duplicate identifier {name!r}", item_path)
            continue
        result[name] = item
    return result


def _validate_acceptance(config: Mapping[str, Any], collector: _Collector) -> None:
    acceptance = config.get("scientific_acceptance")
    if not isinstance(acceptance, Mapping):
        collector.add(
            "missing_scientific_acceptance",
            "scientific_acceptance must be an object",
            "$.scientific_acceptance",
        )
        return
    policy_flags = {
        "require_real_data": True,
        "require_artifact_annotations": True,
        "require_ignore_masks": True,
        "allow_synthetic_for_scientific_acceptance": False,
        "require_regime_stratification": True,
    }
    for field, expected in policy_flags.items():
        if acceptance.get(field) is not expected:
            code = (
                "synthetic_acceptance_forbidden"
                if field == "allow_synthetic_for_scientific_acceptance"
                else "unsafe_acceptance_policy"
            )
            collector.add(
                code,
                f"{field} must be {expected!r}; this safety rule cannot be relaxed",
                f"$.scientific_acceptance.{field}",
            )
    statuses = acceptance.get("accepted_annotation_statuses")
    normalized = (
        {str(item).strip().casefold() for item in statuses}
        if _is_sequence(statuses)
        else set()
    )
    if normalized != {"adjudicated"}:
        collector.add(
            "adjudication_requirement_cannot_be_relaxed",
            "accepted_annotation_statuses must be exactly ['adjudicated']",
            "$.scientific_acceptance.accepted_annotation_statuses",
        )


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.strip().lower()
    return bool(_SHA256.fullmatch(normalized)) and normalized != "0" * 64


def _validate_ontology_and_governance(
    config: Mapping[str, Any], collector: _Collector
) -> None:
    ontology = config.get("ontology")
    if not isinstance(ontology, Mapping):
        collector.add(
            "missing_ontology_contract",
            "A versioned, approved crack/fold ontology is required",
            "$.ontology",
        )
    else:
        for field in ("version", "crack_definition", "approval_reference"):
            if not _is_present(ontology.get(field)):
                collector.add(
                    "ontology_approval_incomplete",
                    f"Ontology field {field!r} is required",
                    f"$.ontology.{field}",
                    "blocker",
                )
        if ontology.get("stakeholder_approved") is not True:
            collector.add(
                "ontology_not_approved",
                "The operational crack definition must be stakeholder-approved",
                "$.ontology.stakeholder_approved",
                "blocker",
            )

    governance = config.get("governance_approvals")
    if not isinstance(governance, Mapping):
        collector.add(
            "missing_governance_approvals",
            "Data, privacy, method-asset, and scientific-owner approvals are required",
            "$.governance_approvals",
            "blocker",
        )
        return
    for area in _APPROVAL_AREAS:
        approval = governance.get(area)
        path = f"$.governance_approvals.{area}"
        if not isinstance(approval, Mapping):
            collector.add(
                "governance_approval_missing",
                f"Governance approval {area!r} is missing",
                path,
                "blocker",
            )
            continue
        if approval.get("status") != "approved":
            collector.add(
                "governance_approval_pending",
                f"Governance approval {area!r} is not approved",
                f"{path}.status",
                "blocker",
            )
        if not _is_present(approval.get("reference")) or not _is_sha256(
            approval.get("evidence_sha256")
        ):
            collector.add(
                "governance_evidence_unverified",
                "Approval reference and evidence SHA-256 are required",
                path,
                "blocker",
            )


def _validate_modalities(
    config: Mapping[str, Any], collector: _Collector
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    modalities = _as_named_mapping(
        config.get("modalities"),
        key_name="id",
        collector=collector,
        path="$.modalities",
    )
    for modality in REQUIRED_MODALITIES:
        if modality not in modalities:
            collector.add(
                "missing_required_modality",
                f"Benchmark must define modality {modality!r}",
                "$.modalities",
            )

    variant_modalities: dict[str, str] = {}
    for modality_id, modality in modalities.items():
        path = f"$.modalities.{modality_id}"
        variants = modality.get("input_variants")
        if not _is_sequence(variants) or not variants:
            collector.add(
                "missing_input_variants",
                "Each modality needs at least one semantic-channel input variant",
                f"{path}.input_variants",
            )
            continue
        for index, variant in enumerate(variants):
            variant_path = f"{path}.input_variants[{index}]"
            if not isinstance(variant, Mapping):
                collector.add("expected_object", "Input variant must be an object", variant_path)
                continue
            variant_id = variant.get("id")
            if not isinstance(variant_id, str) or not variant_id.strip():
                collector.add("missing_id", "Input variant requires an id", variant_path)
                continue
            if variant_id in variant_modalities:
                collector.add("duplicate_id", f"Duplicate input variant {variant_id!r}", variant_path)
            variant_modalities[str(variant_id)] = modality_id
            if variant.get("channel_selection") != "semantic_role":
                collector.add(
                    "positional_channel_selection_forbidden",
                    "Channels must be selected by semantic role, never by array position",
                    f"{variant_path}.channel_selection",
                )
            roles = variant.get("semantic_channels")
            if not _is_sequence(roles) or not roles or any(
                not isinstance(role, str) or not role.strip() for role in roles
            ):
                collector.add(
                    "missing_semantic_channels",
                    "semantic_channels must be a non-empty array of role names",
                    f"{variant_path}.semantic_channels",
                )
            if "channel_indices" in variant:
                collector.add(
                    "positional_channel_selection_forbidden",
                    "channel_indices are not allowed in a scientific benchmark contract",
                    f"{variant_path}.channel_indices",
                )
    return modalities, variant_modalities


def _validate_regimes(
    config: Mapping[str, Any], collector: _Collector
) -> dict[str, Mapping[str, Any]]:
    regimes = _as_named_mapping(
        config.get("regimes"),
        key_name="id",
        collector=collector,
        path="$.regimes",
    )
    for regime_id in REQUIRED_REGIMES:
        if regime_id not in regimes:
            collector.add(
                "missing_required_regime",
                f"Benchmark must define regime {regime_id!r}",
                "$.regimes",
            )
            continue
        required = regimes[regime_id].get("required_cohorts")
        required_set = (
            {str(item) for item in required}
            if _is_sequence(required)
            else set()
        )
        if required_set != set(EXPECTED_REGIME_COHORTS[regime_id]):
            collector.add(
                "invalid_regime_cohorts",
                f"{regime_id} must use exactly "
                f"{sorted(EXPECTED_REGIME_COHORTS[regime_id])}",
                f"$.regimes.{regime_id}.required_cohorts",
            )
    return regimes


def _validate_resources(
    config: Mapping[str, Any], collector: _Collector
) -> tuple[dict[str, Mapping[str, Any]], dict[str, bool]]:
    resources = _as_named_mapping(
        config.get("resources", []),
        key_name="id",
        collector=collector,
        path="$.resources",
    )
    eligible: dict[str, bool] = {}
    for resource_id, resource in resources.items():
        path = f"$.resources.{resource_id}"
        availability = resource.get("availability_status")
        license_status = resource.get("license_status")
        commercial = resource.get("commercial_use_approved")
        if availability not in RESOURCE_AVAILABILITY:
            collector.add(
                "invalid_resource_availability",
                f"availability_status must be one of {sorted(RESOURCE_AVAILABILITY)}",
                f"{path}.availability_status",
            )
        if license_status not in LICENSE_STATUSES:
            collector.add(
                "invalid_license_status",
                f"license_status must be one of {sorted(LICENSE_STATUSES)}",
                f"{path}.license_status",
            )
        if not isinstance(commercial, bool):
            collector.add(
                "missing_commercial_license_decision",
                "commercial_use_approved must be explicitly true or false",
                f"{path}.commercial_use_approved",
            )
        eligible[resource_id] = (
            availability == "available"
            and license_status == "approved"
            and commercial is True
        )
    return resources, eligible


def _validate_capabilities(
    method: Mapping[str, Any], collector: _Collector, path: str
) -> dict[tuple[str, str], str]:
    capabilities = method.get("capabilities")
    result: dict[tuple[str, str], str] = {}
    if not _is_sequence(capabilities) or not capabilities:
        collector.add(
            "missing_method_capabilities",
            "Method must declare at least one artifact/task/output capability",
            f"{path}.capabilities",
        )
        return result
    for index, capability in enumerate(capabilities):
        item_path = f"{path}.capabilities[{index}]"
        if not isinstance(capability, Mapping):
            collector.add("expected_object", "Capability must be an object", item_path)
            continue
        artifact = capability.get("artifact")
        task = capability.get("task")
        output = capability.get("output_type")
        if artifact not in ALLOWED_ARTIFACTS:
            collector.add(
                "invalid_artifact", f"artifact must be one of {ALLOWED_ARTIFACTS}", item_path
            )
            continue
        allowed = (
            FOLD_OUTPUT_TYPES
            if artifact == "fold"
            else CRACK_OUTPUT_TYPES
            if artifact == "crack"
            else UNION_OUTPUT_TYPES
        )
        if task not in allowed:
            collector.add(
                "invalid_task_type",
                f"task must be one of {sorted(allowed)}",
                f"{item_path}.task",
            )
            continue
        if output not in allowed[task]:
            collector.add(
                "task_output_incompatible",
                f"Output {output!r} cannot be evaluated as {artifact} {task}",
                f"{item_path}.output_type",
            )
            continue
        key = (artifact, task)
        if key in result:
            collector.add(
                "duplicate_method_capability",
                f"Method declares {artifact} {task} more than once",
                item_path,
            )
            continue
        result[key] = str(output)
    return result


def _validate_methods(
    config: Mapping[str, Any],
    collector: _Collector,
    modalities: Mapping[str, Mapping[str, Any]],
    variant_modalities: Mapping[str, str],
    regimes: Mapping[str, Mapping[str, Any]],
    resources: Mapping[str, Mapping[str, Any]],
    resource_eligible: Mapping[str, bool],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[str, dict[tuple[str, str], str]],
    tuple[str, ...],
    tuple[str, ...],
]:
    methods = _as_named_mapping(
        config.get("methods"),
        key_name="id",
        collector=collector,
        path="$.methods",
    )
    capabilities: dict[str, dict[tuple[str, str], str]] = {}
    eligible: list[str] = []
    gated: list[str] = []

    for method_id, method in methods.items():
        path = f"$.methods.{method_id}"
        method_modalities = method.get("modalities")
        if not _is_sequence(method_modalities) or not method_modalities:
            collector.add(
                "missing_method_modalities",
                "Method must declare supported modalities",
                f"{path}.modalities",
            )
            modality_values: set[str] = set()
        else:
            modality_values = {str(item) for item in method_modalities}
            unknown = modality_values.difference(modalities)
            if unknown:
                collector.add(
                    "unknown_method_modality",
                    f"Unknown modalities: {sorted(unknown)}",
                    f"{path}.modalities",
                )

        method_variants = method.get("input_variants")
        if not _is_sequence(method_variants) or not method_variants:
            collector.add(
                "missing_method_input_variants",
                "Method must reference at least one semantic input variant",
                f"{path}.input_variants",
            )
        else:
            normalized_variants = {str(item) for item in method_variants}
            unknown_variants = normalized_variants.difference(variant_modalities)
            if unknown_variants:
                collector.add(
                    "unknown_input_variant",
                    f"Unknown input variants: {sorted(unknown_variants)}",
                    f"{path}.input_variants",
                )
            mismatched_variants = {
                variant
                for variant in normalized_variants.difference(unknown_variants)
                if variant_modalities[variant] not in modality_values
            }
            if mismatched_variants:
                collector.add(
                    "method_input_variant_modality_mismatch",
                    "Input variants must belong to one of the method's declared modalities: "
                    f"{sorted(mismatched_variants)}",
                    f"{path}.input_variants",
                )

        method_regimes = method.get("regimes")
        if not _is_sequence(method_regimes) or not method_regimes:
            collector.add(
                "missing_method_regimes",
                "Method must declare at least one evaluation regime",
                f"{path}.regimes",
            )
        else:
            unknown_regimes = {str(item) for item in method_regimes}.difference(regimes)
            if unknown_regimes:
                collector.add(
                    "unknown_method_regime",
                    f"Unknown regimes: {sorted(unknown_regimes)}",
                    f"{path}.regimes",
                )

        capabilities[method_id] = _validate_capabilities(method, collector, path)

        availability = method.get("availability_status")
        if availability not in METHOD_AVAILABILITY:
            collector.add(
                "invalid_method_availability",
                f"availability_status must be one of {sorted(METHOD_AVAILABILITY)}",
                f"{path}.availability_status",
            )
        enabled = method.get("enabled")
        required = method.get("required_for_acceptance", False)
        if not isinstance(enabled, bool):
            collector.add(
                "missing_method_enabled_flag",
                "enabled must be explicitly true or false",
                f"{path}.enabled",
            )
            enabled = False
        if not isinstance(required, bool):
            collector.add(
                "invalid_required_flag",
                "required_for_acceptance must be boolean",
                f"{path}.required_for_acceptance",
            )
            required = False

        resource_ids = method.get("resource_ids", [])
        if not _is_sequence(resource_ids):
            collector.add(
                "invalid_method_resources",
                "resource_ids must be an array",
                f"{path}.resource_ids",
            )
            resource_ids = []
        missing_resources = {str(item) for item in resource_ids}.difference(resources)
        if missing_resources:
            collector.add(
                "unknown_resource",
                f"Unknown resources: {sorted(missing_resources)}",
                f"{path}.resource_ids",
            )

        resource_gate = any(
            not resource_eligible.get(str(resource_id), False)
            for resource_id in resource_ids
        )
        is_eligible = bool(enabled) and availability == "ready" and not resource_gate
        if is_eligible:
            eligible.append(method_id)
        else:
            gated.append(method_id)
            if required:
                collector.add(
                    "required_method_unavailable",
                    "Required method is disabled, not ready, or lacks an approved resource/license",
                    path,
                    "blocker",
                )
            elif enabled:
                collector.add(
                    "method_gated",
                    "Enabled method is excluded until implementation, resource, and license gates pass",
                    path,
                    "warning",
                )

    if not eligible:
        collector.add(
            "no_eligible_methods",
            "At least one enabled, ready, license-approved method is required",
            "$.methods",
            "blocker",
        )
    return methods, capabilities, tuple(sorted(eligible)), tuple(sorted(gated))


def _validate_metrics(
    config: Mapping[str, Any], collector: _Collector
) -> dict[tuple[str, str], tuple[str, set[str]]]:
    metrics = config.get("metrics")
    result: dict[tuple[str, str], tuple[str, set[str]]] = {}
    if not isinstance(metrics, Mapping):
        collector.add("missing_metrics", "metrics must be an object", "$.metrics")
        return result
    required_tasks = ("presence", "localization", "patch_classification")
    for artifact in ALLOWED_ARTIFACTS:
        artifact_metrics = metrics.get(artifact)
        if not isinstance(artifact_metrics, Mapping):
            collector.add(
                "missing_artifact_metrics",
                f"Metrics for {artifact} are required",
                f"$.metrics.{artifact}",
            )
            continue
        for task in required_tasks:
            group = artifact_metrics.get(task)
            path = f"$.metrics.{artifact}.{task}"
            if not isinstance(group, Mapping):
                collector.add(
                    "missing_task_metrics",
                    f"Metrics for {artifact} {task} are required",
                    path,
                )
                continue
            output = group.get("comparison_output_type")
            allowed = (
                FOLD_OUTPUT_TYPES
                if artifact == "fold"
                else CRACK_OUTPUT_TYPES
                if artifact == "crack"
                else UNION_OUTPUT_TYPES
            )
            if output not in allowed[task]:
                collector.add(
                    "task_output_incompatible",
                    f"Output {output!r} cannot be used for {artifact} {task}",
                    f"{path}.comparison_output_type",
                )
            metric_ids = group.get("metric_ids")
            if not _is_sequence(metric_ids) or not metric_ids:
                collector.add(
                    "missing_metric_ids",
                    "metric_ids must be a non-empty array",
                    f"{path}.metric_ids",
                )
                ids: set[str] = set()
            else:
                ids = {str(item) for item in metric_ids}
                if len(ids) != len(metric_ids):
                    collector.add(
                        "duplicate_metric_id",
                        "metric_ids cannot contain duplicates",
                        f"{path}.metric_ids",
                    )
            if task == "presence" and ids.intersection(
                {"brier_score", "expected_calibration_error"}
            ) and group.get("score_semantics") != "calibrated_probability":
                collector.add(
                    "probability_metric_semantics_missing",
                    "Brier score and ECE require calibrated_probability semantics",
                    f"{path}.score_semantics",
                )
            if task == "localization" and "surface_dice" in ids:
                tolerance = group.get("surface_tolerance_um")
                if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                    collector.add(
                        "physical_metric_tolerance_missing",
                        "surface_dice requires a positive surface_tolerance_um",
                        f"{path}.surface_tolerance_um",
                    )
            if task == "localization" and ids.intersection(
                {"cldice", "centerline_f1_tolerance"}
            ):
                tolerance = group.get("centerline_tolerance_um")
                if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                    collector.add(
                        "physical_metric_tolerance_missing",
                        "Crack centerline metrics require centerline_tolerance_um",
                        f"{path}.centerline_tolerance_um",
                    )
            result[(artifact, task)] = (str(output), ids)

    # Guardrails ensure the main topology-sensitive metrics cannot disappear.
    required_metric_ids = {
        ("fold", "presence"): {"auprc", "sensitivity_at_fixed_false_positive_rate"},
        ("fold", "localization"): {"dice", "iou", "surface_dice"},
        ("crack", "presence"): {"auprc", "sensitivity_at_fixed_false_positive_rate"},
        ("crack", "localization"): {"cldice", "centerline_f1_tolerance"},
        ("artifact_union", "presence"): {
            "auprc",
            "sensitivity_at_fixed_false_positive_rate",
        },
        ("artifact_union", "localization"): {"dice", "iou", "surface_dice"},
    }
    for key, required_ids in required_metric_ids.items():
        configured = result.get(key, ("", set()))[1]
        missing = required_ids.difference(configured)
        if missing:
            collector.add(
                "missing_required_metric",
                f"Required metrics are missing: {sorted(missing)}",
                f"$.metrics.{key[0]}.{key[1]}.metric_ids",
            )
    return result


def _validate_reporting(config: Mapping[str, Any], collector: _Collector) -> None:
    reporting = config.get("reporting")
    if not isinstance(reporting, Mapping):
        collector.add(
            "missing_reporting_contract",
            "A locked reporting and stratification contract is required",
            "$.reporting",
        )
        return
    required_strata = {
        "modality",
        "artifact",
        "task",
        "regime",
        "input_variant",
        "site",
        "scanner_or_instrument",
        "tissue_type",
        "severity",
    }
    raw_strata = reporting.get("stratify_by")
    strata = {str(item) for item in raw_strata} if _is_sequence(raw_strata) else set()
    missing = required_strata.difference(strata)
    if missing:
        collector.add(
            "incomplete_reporting_strata",
            f"Required reporting strata are missing: {sorted(missing)}",
            "$.reporting.stratify_by",
        )
    for field in (
        "forbid_pooled_primary_score_across_modalities",
        "forbid_ranking_patch_classifiers_by_pixel_metrics",
        "require_runtime_memory_failure_and_abstention_reporting",
    ):
        if reporting.get(field) is not True:
            collector.add(
                "unsafe_reporting_policy",
                f"{field} must be true",
                f"$.reporting.{field}",
            )
    if reporting.get("confidence_intervals") != (
        "cluster_bootstrap_at_highest_available_patient_or_block_unit"
    ):
        collector.add(
            "invalid_confidence_interval_contract",
            "Confidence intervals must use the locked highest-unit cluster bootstrap",
            "$.reporting.confidence_intervals",
        )


def _validate_comparisons(
    config: Mapping[str, Any],
    collector: _Collector,
    methods: Mapping[str, Mapping[str, Any]],
    capabilities: Mapping[str, Mapping[tuple[str, str], str]],
    metrics: Mapping[tuple[str, str], tuple[str, set[str]]],
    eligible_methods: Sequence[str],
    regimes: Mapping[str, Mapping[str, Any]],
    variant_modalities: Mapping[str, str],
) -> None:
    comparisons = config.get("comparisons")
    if not _is_sequence(comparisons) or not comparisons:
        collector.add(
            "missing_comparisons",
            "At least one task-compatible comparison must be declared",
            "$.comparisons",
        )
        return
    seen: set[str] = set()
    eligible_set = set(eligible_methods)
    covered_cells: set[tuple[str, str, str]] = set()
    for index, comparison in enumerate(comparisons):
        path = f"$.comparisons[{index}]"
        if not isinstance(comparison, Mapping):
            collector.add("expected_object", "Comparison must be an object", path)
            continue
        comparison_id = comparison.get("id")
        if not isinstance(comparison_id, str) or not comparison_id.strip():
            collector.add("missing_id", "Comparison requires an id", path)
        elif comparison_id in seen:
            collector.add("duplicate_id", f"Duplicate comparison {comparison_id!r}", path)
        else:
            seen.add(comparison_id)
        artifact = comparison.get("artifact")
        task = comparison.get("task")
        modality = comparison.get("modality")
        regime = comparison.get("regime")
        input_variant = comparison.get("input_variant")
        output = comparison.get("comparison_output_type")
        if artifact not in ALLOWED_ARTIFACTS or task not in TASK_OUTPUT_TYPES:
            collector.add(
                "invalid_comparison_task",
                "Comparison must declare a supported artifact and task",
                path,
            )
            continue
        metric_group = metrics.get((str(artifact), str(task)))
        if metric_group is None:
            collector.add("unknown_metric_group", "No metric group exists for comparison", path)
        elif output != metric_group[0]:
            collector.add(
                "comparison_metric_output_mismatch",
                "Comparison output must exactly match its metric group's output type",
                f"{path}.comparison_output_type",
            )
        if modality not in REQUIRED_MODALITIES:
            collector.add(
                "invalid_comparison_modality",
                f"modality must be one of {REQUIRED_MODALITIES}",
                f"{path}.modality",
            )
        if regime not in regimes:
            collector.add(
                "invalid_comparison_regime",
                "Comparison must lock one declared adaptation regime",
                f"{path}.regime",
            )
        if input_variant not in variant_modalities:
            collector.add(
                "invalid_comparison_input_variant",
                "Comparison must lock one declared semantic input variant",
                f"{path}.input_variant",
            )
        elif variant_modalities[str(input_variant)] != modality:
            collector.add(
                "comparison_input_variant_modality_mismatch",
                "Comparison input variant does not belong to its modality",
                f"{path}.input_variant",
            )
        if modality in REQUIRED_MODALITIES:
            covered_cells.add((str(modality), str(artifact), str(task)))

        if comparison.get("stratify_by_regime") is not True:
            collector.add(
                "regime_pooling_forbidden",
                "Comparisons must report each adaptation regime separately",
                f"{path}.stratify_by_regime",
            )

        method_ids = comparison.get("method_ids")
        if not _is_sequence(method_ids) or not method_ids:
            collector.add(
                "missing_comparison_methods",
                "Comparison requires method_ids",
                f"{path}.method_ids",
            )
            continue
        normalized_ids = [str(item) for item in method_ids]
        eligible_compatible_ids: set[str] = set()
        for method_id in normalized_ids:
            if method_id not in methods:
                collector.add(
                    "unknown_comparison_method",
                    f"Unknown comparison method {method_id!r}",
                    f"{path}.method_ids",
                )
                continue
            if modality not in {str(item) for item in methods[method_id].get("modalities", [])}:
                collector.add(
                    "method_modality_incompatible",
                    f"Method {method_id!r} does not support modality {modality!r}",
                    f"{path}.method_ids",
                )
            method_regimes = {
                str(item) for item in methods[method_id].get("regimes", [])
            }
            if regime not in method_regimes:
                collector.add(
                    "method_regime_incompatible",
                    f"Method {method_id!r} is not declared for regime {regime!r}",
                    f"{path}.method_ids",
                )
            method_variants = {
                str(item) for item in methods[method_id].get("input_variants", [])
            }
            if input_variant not in method_variants:
                collector.add(
                    "method_input_variant_incompatible",
                    f"Method {method_id!r} does not consume {input_variant!r}",
                    f"{path}.method_ids",
                )
            method_output = capabilities.get(method_id, {}).get((str(artifact), str(task)))
            if method_output is None:
                collector.add(
                    "method_task_incompatible",
                    f"Method {method_id!r} does not support {artifact} {task}",
                    f"{path}.method_ids",
                )
            elif method_output != output:
                collector.add(
                    "method_output_incompatible",
                    f"Method {method_id!r} produces {method_output!r}, not {output!r}",
                    f"{path}.method_ids",
                )
            if (
                method_id in eligible_set
                and modality in {str(item) for item in methods[method_id].get("modalities", [])}
                and regime in method_regimes
                and input_variant in method_variants
                and method_output == output
            ):
                eligible_compatible_ids.add(method_id)

        metric_ids = comparison.get("metric_ids")
        if not _is_sequence(metric_ids) or not metric_ids:
            collector.add(
                "missing_comparison_metrics",
                "Comparison requires metric_ids",
                f"{path}.metric_ids",
            )
        elif metric_group is not None:
            unknown = {str(item) for item in metric_ids}.difference(metric_group[1])
            if unknown:
                collector.add(
                    "comparison_metric_incompatible",
                    f"Metrics do not belong to this task: {sorted(unknown)}",
                    f"{path}.metric_ids",
                )

        required = comparison.get("required_for_acceptance", False)
        minimum = comparison.get("minimum_eligible_methods", 2)
        if not isinstance(required, bool) or not isinstance(minimum, int) or minimum < 1:
            collector.add(
                "invalid_comparison_acceptance_rule",
                "required_for_acceptance must be boolean and minimum_eligible_methods positive",
                path,
            )
        elif required:
            count = len(eligible_compatible_ids)
            if count < minimum:
                collector.add(
                    "insufficient_eligible_comparison_methods",
                    f"Comparison needs {minimum} eligible methods but currently has {count}",
                    path,
                    "blocker",
                )

    required_cells = {
        (modality, artifact, task)
        for modality in REQUIRED_MODALITIES
        for artifact in ALLOWED_ARTIFACTS
        for task in ("presence", "localization")
    }
    missing_cells = sorted(required_cells.difference(covered_cells))
    if missing_cells:
        collector.add(
            "incomplete_comparison_coverage",
            "Every modality requires fold, crack, and artifact-union presence and "
            f"localization comparisons; missing {missing_cells}",
            "$.comparisons",
        )


def _normalize_cohort_records(
    value: object, collector: _Collector
) -> dict[str, list[Mapping[str, Any]]]:
    records = {role: [] for role in REQUIRED_COHORTS}
    if value is None:
        return records
    if isinstance(value, Mapping):
        for role, items in value.items():
            if role not in REQUIRED_COHORTS:
                collector.add(
                    "unknown_cohort_role",
                    f"Unknown cohort role {role!r}",
                    "$.cohort_records",
                )
                continue
            if not _is_sequence(items):
                collector.add(
                    "invalid_cohort_records",
                    "Cohort records must be an array",
                    f"$.cohort_records.{role}",
                )
                continue
            records[role] = [item for item in items if isinstance(item, Mapping)]
            if len(records[role]) != len(items):
                collector.add(
                    "invalid_cohort_record",
                    "Every cohort record must be an object",
                    f"$.cohort_records.{role}",
                )
        return records
    if _is_sequence(value):
        for index, item in enumerate(value):
            if not isinstance(item, Mapping):
                collector.add(
                    "invalid_cohort_record",
                    "Every cohort record must be an object",
                    f"$.cohort_records[{index}]",
                )
                continue
            role = item.get("cohort_role", item.get("split"))
            if role not in REQUIRED_COHORTS:
                collector.add(
                    "unknown_cohort_role",
                    "Record requires cohort_role fit, calibration, or test",
                    f"$.cohort_records[{index}]",
                )
                continue
            records[str(role)].append(item)
        return records
    collector.add(
        "invalid_cohort_records",
        "cohort_records must be an object keyed by role or an array",
        "$.cohort_records",
    )
    return records


def _validate_cohorts_and_records(
    config: Mapping[str, Any],
    collector: _Collector,
    supplied_records: object,
) -> tuple[tuple[str, int], ...]:
    split_policy = config.get("split_policy")
    if not isinstance(split_policy, Mapping):
        collector.add("missing_split_policy", "split_policy must be an object", "$.split_policy")
        unit_keys: list[str] = []
    else:
        raw_keys = split_policy.get("unit_keys")
        unit_keys = [str(item) for item in raw_keys] if _is_sequence(raw_keys) else []
        missing_keys = set(REQUIRED_SPLIT_KEYS).difference(unit_keys)
        if missing_keys:
            collector.add(
                "missing_split_unit_keys",
                f"Split policy is missing leakage keys: {sorted(missing_keys)}",
                "$.split_policy.unit_keys",
            )
        if len(unit_keys) != len(set(unit_keys)):
            collector.add(
                "duplicate_split_unit_key",
                "split unit keys must be unique",
                "$.split_policy.unit_keys",
            )

    cohorts = _as_named_mapping(
        config.get("cohorts"),
        key_name="role",
        collector=collector,
        path="$.cohorts",
    )
    for role in REQUIRED_COHORTS:
        cohort = cohorts.get(role)
        if cohort is None:
            collector.add(
                "missing_required_cohort",
                f"Independent {role} cohort is required",
                "$.cohorts",
            )
            continue
        path = f"$.cohorts.{role}"
        if cohort.get("data_origin") != "real":
            collector.add(
                "synthetic_efficacy_data_forbidden",
                "Scientific fit/calibration/test cohorts must contain real acquired data",
                f"{path}.data_origin",
            )
        if not _is_present(cohort.get("manifest_path")):
            collector.add(
                "missing_cohort_manifest",
                "Cohort must reference a versioned manifest",
                f"{path}.manifest_path",
            )
        artifacts = cohort.get("annotated_artifacts")
        artifact_set = {str(item) for item in artifacts} if _is_sequence(artifacts) else set()
        if not set(REQUIRED_ARTIFACTS).issubset(artifact_set):
            collector.add(
                "missing_artifact_annotations",
                "Cohort declaration must cover separate fold and crack annotations",
                f"{path}.annotated_artifacts",
                "blocker",
            )
        if cohort.get("ignore_masks") is not True:
            collector.add(
                "missing_ignore_masks",
                "Cohort declaration must require ignore/uncertain-region masks",
                f"{path}.ignore_masks",
                "blocker",
            )

    records = _normalize_cohort_records(supplied_records, collector)
    counts = tuple((role, len(records[role])) for role in REQUIRED_COHORTS)
    accepted = {"adjudicated"}
    attestations = config.get("cohort_evidence_attestations")
    if not isinstance(attestations, Mapping):
        attestations = {}
    seen_by_key: dict[str, dict[str, set[str]]] = {key: {} for key in unit_keys}
    test_support: dict[str, dict[str, int]] = {
        modality: {"fold_positive": 0, "crack_positive": 0, "union_negative": 0}
        for modality in REQUIRED_MODALITIES
    }

    for role in REQUIRED_COHORTS:
        role_records = records[role]
        attestation = attestations.get(role)
        attestation_path = f"$.cohort_evidence_attestations.{role}"
        if not isinstance(attestation, Mapping):
            collector.add(
                "strict_manifest_attestation_missing",
                f"A strict validation attestation is required for {role}",
                attestation_path,
                "blocker",
            )
        else:
            if attestation.get("strict_manifest_validation_passed") is not True:
                collector.add(
                    "strict_manifest_validation_unverified",
                    f"Strict manifest validation has not passed for {role}",
                    attestation_path,
                    "blocker",
                )
            if not _is_sha256(attestation.get("manifest_sha256")) or not _is_sha256(
                attestation.get("validation_report_sha256")
            ):
                collector.add(
                    "strict_manifest_attestation_checksum_missing",
                    "Manifest and validation-report SHA-256 values are required",
                    attestation_path,
                    "blocker",
                )
            if attestation.get("record_count") != len(role_records):
                collector.add(
                    "strict_manifest_attestation_count_mismatch",
                    "Attested record_count must equal the supplied cohort records",
                    f"{attestation_path}.record_count",
                    "blocker",
                )
        if not role_records:
            collector.add(
                "cohort_records_unverified",
                f"No realized {role} records were supplied; declarations alone cannot prove eligibility",
                f"$.cohort_records.{role}",
                "blocker",
            )
            continue
        cohort_defaults = cohorts.get(role, {})
        present_modalities = {
            str(record.get("modality"))
            for record in role_records
            if record.get("modality") in REQUIRED_MODALITIES
        }
        missing_modalities = set(REQUIRED_MODALITIES).difference(present_modalities)
        if missing_modalities:
            collector.add(
                "incomplete_cohort_modality_coverage",
                f"{role} cohort has no realized records for {sorted(missing_modalities)}",
                f"$.cohort_records.{role}",
                "blocker",
            )
        for index, record in enumerate(role_records):
            path = f"$.cohort_records.{role}[{index}]"
            origin = record.get("data_origin", cohort_defaults.get("data_origin"))
            if origin != "acquired_real" or record.get("is_synthetic") is True:
                collector.add(
                    "synthetic_efficacy_data_forbidden",
                    "Records must positively declare acquired_real provenance",
                    path,
                )
            if record.get("provenance_status") != "approved":
                collector.add(
                    "record_provenance_unapproved",
                    "Each record requires approved acquisition provenance",
                    f"{path}.provenance_status",
                    "blocker",
                )
            if record.get("strict_manifest_validated") is not True:
                collector.add(
                    "record_strict_validation_unverified",
                    "Each realized record must come from strict manifest validation",
                    f"{path}.strict_manifest_validated",
                    "blocker",
                )
            if not _is_sha256(record.get("image_sha256")):
                collector.add(
                    "record_image_checksum_missing",
                    "A verified image SHA-256 is required",
                    f"{path}.image_sha256",
                    "blocker",
                )
            modality = record.get("modality")
            if modality not in REQUIRED_MODALITIES:
                collector.add(
                    "invalid_record_modality",
                    f"Record modality must be one of {REQUIRED_MODALITIES}",
                    f"{path}.modality",
                )
            status = record.get("annotation_status", cohort_defaults.get("annotation_status"))
            if status not in accepted:
                collector.add(
                    "unaccepted_annotation_status",
                    f"Annotation status must be one of {sorted(accepted)}",
                    f"{path}.annotation_status",
                    "blocker",
                )
            annotations = record.get("annotations")
            annotation_sha256 = record.get("annotation_sha256")
            reference_positive = record.get("reference_positive")
            for artifact in REQUIRED_ARTIFACTS:
                if not isinstance(annotations, Mapping) or not _is_present(
                    annotations.get(artifact)
                ):
                    collector.add(
                        "missing_real_annotation",
                        f"Real {artifact} ground truth is required for efficacy",
                        f"{path}.annotations.{artifact}",
                        "blocker",
                    )
                if not isinstance(annotation_sha256, Mapping) or not _is_sha256(
                    annotation_sha256.get(artifact)
                ):
                    collector.add(
                        "annotation_checksum_missing",
                        f"A verified {artifact} mask SHA-256 is required",
                        f"{path}.annotation_sha256.{artifact}",
                        "blocker",
                    )
                if not isinstance(reference_positive, Mapping) or not isinstance(
                    reference_positive.get(artifact), bool
                ):
                    collector.add(
                        "reference_class_support_missing",
                        f"Explicit {artifact} positive/negative support is required",
                        f"{path}.reference_positive.{artifact}",
                        "blocker",
                    )
            ignore_mask = record.get("ignore_mask", record.get("ignore_mask_path"))
            if not _is_present(ignore_mask):
                collector.add(
                    "missing_record_ignore_mask",
                    "Every efficacy record requires an ignore/uncertain-region mask",
                    f"{path}.ignore_mask",
                    "blocker",
                )
            if not _is_sha256(record.get("ignore_mask_sha256")):
                collector.add(
                    "ignore_mask_checksum_missing",
                    "A verified ignore-mask SHA-256 is required",
                    f"{path}.ignore_mask_sha256",
                    "blocker",
                )
            if (
                role == "test"
                and modality in REQUIRED_MODALITIES
                and isinstance(reference_positive, Mapping)
            ):
                fold_positive = reference_positive.get("fold") is True
                crack_positive = reference_positive.get("crack") is True
                test_support[str(modality)]["fold_positive"] += int(fold_positive)
                test_support[str(modality)]["crack_positive"] += int(crack_positive)
                test_support[str(modality)]["union_negative"] += int(
                    not fold_positive and not crack_positive
                )
            for key in unit_keys:
                value = record.get(key)
                if not _is_present(value):
                    collector.add(
                        "missing_split_identifier",
                        f"Record is missing required split identifier {key!r}",
                        f"{path}.{key}",
                        "blocker",
                    )
                    continue
                canonical = str(value)
                seen_by_key[key].setdefault(canonical, set()).add(role)

    # Do not echo identifier values: they may contain sensitive information.
    for key, values in seen_by_key.items():
        overlap_count = sum(1 for roles in values.values() if len(roles) > 1)
        if overlap_count:
            collector.add(
                "cohort_identifier_overlap",
                f"Detected {overlap_count} {key} value(s) shared across fit/calibration/test",
                f"$.split_policy.unit_keys.{key}",
            )
    for modality, support in test_support.items():
        for category, count in support.items():
            if count < 1:
                collector.add(
                    "insufficient_test_class_support",
                    f"Test modality {modality!r} requires at least one {category} record",
                    f"$.cohort_records.test.{modality}.{category}",
                    "blocker",
                )
    return counts


def validate_benchmark_contract(
    source: str | Path | Mapping[str, Any],
    *,
    cohort_records: object | None = None,
) -> BenchmarkContractReport:
    """Validate a benchmark contract and decide scientific report eligibility.

    Parameters
    ----------
    source:
        JSON path or already-loaded mapping.
    cohort_records:
        Optional realized records, either keyed by ``fit``/``calibration``/``test``
        or a flat list with ``cohort_role``.  When omitted, embedded
        ``cohort_records`` are used.  Empty or absent records block report
        eligibility because manifest declarations cannot prove non-overlap.
    """

    config, _ = _read_contract_source(source)
    collector = _Collector()
    if config.get("schema_version") != "1.0":
        collector.add(
            "unsupported_schema_version",
            "schema_version must be '1.0'",
            "$.schema_version",
        )
    if not _is_present(config.get("benchmark_id")):
        collector.add("missing_benchmark_id", "benchmark_id is required", "$.benchmark_id")

    _validate_acceptance(config, collector)
    _validate_ontology_and_governance(config, collector)
    modalities, variant_modalities = _validate_modalities(config, collector)
    regimes = _validate_regimes(config, collector)
    resources, resource_eligible = _validate_resources(config, collector)
    methods, capabilities, eligible, gated = _validate_methods(
        config,
        collector,
        modalities,
        variant_modalities,
        regimes,
        resources,
        resource_eligible,
    )
    metrics = _validate_metrics(config, collector)
    _validate_comparisons(
        config,
        collector,
        methods,
        capabilities,
        metrics,
        eligible,
        regimes,
        variant_modalities,
    )
    _validate_reporting(config, collector)
    realized_records = config.get("cohort_records") if cohort_records is None else cohort_records
    counts = _validate_cohorts_and_records(config, collector, realized_records)

    return BenchmarkContractReport(
        issues=tuple(collector.issues),
        eligible_method_ids=eligible,
        gated_method_ids=gated,
        cohort_record_counts=counts,
    )


def load_benchmark_contract(
    source: str | Path | Mapping[str, Any],
    *,
    cohort_records: object | None = None,
    require_report_eligible: bool = False,
) -> BenchmarkContract:
    """Load a contract, rejecting structural errors by default.

    Set ``require_report_eligible=True`` at the boundary of any scientific
    report or acceptance pipeline.  Planned example contracts may otherwise be
    loaded while their data/resource blockers remain visible in ``report``.
    """

    config, source_path = _read_contract_source(source)
    report = validate_benchmark_contract(config, cohort_records=cohort_records)
    report.raise_for_errors()
    if require_report_eligible:
        report.raise_if_not_report_eligible()
    return BenchmarkContract(data=config, report=report, source_path=source_path)


__all__ = [
    "ALLOWED_ARTIFACTS",
    "REQUIRED_ARTIFACTS",
    "REQUIRED_COHORTS",
    "REQUIRED_MODALITIES",
    "REQUIRED_REGIMES",
    "REQUIRED_SPLIT_KEYS",
    "BenchmarkContract",
    "BenchmarkContractError",
    "BenchmarkContractReport",
    "ContractIssue",
    "load_benchmark_contract",
    "validate_benchmark_contract",
]
