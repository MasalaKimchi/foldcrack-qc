"""Operational PASS/REVIEW/FAIL evaluation and acceptance-gate checks.

Localization metrics answer whether a mask overlaps an annotation.  This module
answers the separate safety/workflow question: did consequential cases avoid an
automatic pass, how many cases were referred, and was useful tissue overmasked?
Synthetic records are deliberately ineligible for acceptance regardless of their
numbers.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .schema import Modality

DECISIONS = frozenset({"PASS", "REVIEW", "FAIL"})


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be boolean")
    return bool(value)


def _optional_fraction(value: Any, name: str) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite fraction in [0, 1]")
    return number


def _wilson(
    successes: int, total: int, confidence: float
) -> dict[str, float | int | None]:
    if total == 0:
        return {
            "successes": 0,
            "total": 0,
            "estimate": None,
            "lower": None,
            "upper": None,
        }
    # Two-sided normal approximation. Gate direction is explicit in the report;
    # a locked study may substitute an approved exact or stratified method.
    from scipy.stats import norm

    z = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return {
        "successes": successes,
        "total": total,
        "estimate": float(proportion),
        "lower": float(max(0.0, center - radius)),
        "upper": float(min(1.0, center + radius)),
    }


def _bootstrap_mean(
    values: Sequence[float], *, confidence: float, n_resamples: int, seed: int
) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "estimate": None, "lower": None, "upper": None}
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(n_resamples, len(array)), replace=True).mean(
        axis=1
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "n": len(array),
        "estimate": float(array.mean()),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1.0 - alpha)),
    }


def _gate(
    *,
    value: float | None,
    threshold: float,
    direction: str,
    evidence_count: int,
    minimum_count: int,
    eligible: bool,
) -> dict[str, Any]:
    if not eligible:
        status = "NOT_EVALUATED"
    elif evidence_count < minimum_count or value is None:
        status = "INSUFFICIENT_EVIDENCE"
    elif direction == "minimum":
        status = "PASS" if value >= threshold else "FAIL"
    elif direction == "maximum":
        status = "PASS" if value <= threshold else "FAIL"
    else:
        raise ValueError("gate direction must be minimum or maximum")
    return {
        "status": status,
        "value": value,
        "threshold": threshold,
        "direction": direction,
        "evidence_count": evidence_count,
        "minimum_evidence_count": minimum_count,
    }


def evaluate_operational_decisions(
    records: Sequence[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    synthetic: bool = False,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate operational safety metrics separately for every modality.

    Required record fields are ``modality``, ``decision``,
    ``reference_actionable_artifact``, and ``reference_severe``. Optional
    sample-level fractions are ``high_confidence_mask_precision`` and
    ``valid_tissue_overmask_fraction``; ``technical_abstention`` defaults false.
    ``REVIEW`` is retained as its own workflow state and counts as a referral,
    never silently as a positive or negative class.
    """

    if not records:
        raise ValueError("records must not be empty")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        modality = Modality.coerce(item.get("modality")).value
        decision = str(item.get("decision", "")).strip().upper()
        if decision not in DECISIONS:
            raise ValueError(f"decision must be one of {sorted(DECISIONS)}")
        normalized = {
            "decision": decision,
            "actionable": _as_bool(
                item.get("reference_actionable_artifact"),
                "reference_actionable_artifact",
            ),
            "severe": _as_bool(item.get("reference_severe"), "reference_severe"),
            "abstention": _as_bool(
                item.get("technical_abstention", False), "technical_abstention"
            ),
            "mask_precision": _optional_fraction(
                item.get("high_confidence_mask_precision"),
                "high_confidence_mask_precision",
            ),
            "overmask": _optional_fraction(
                item.get("valid_tissue_overmask_fraction"),
                "valid_tissue_overmask_fraction",
            ),
        }
        if normalized["abstention"] and decision == "PASS":
            raise ValueError("technical_abstention cannot result in PASS")
        if normalized["severe"] and not normalized["actionable"]:
            raise ValueError("reference_severe requires reference_actionable_artifact")
        grouped[modality].append(normalized)

    eligible = not synthetic
    required_modalities = tuple(
        Modality.coerce(value).value
        for value in acceptance.get("required_modalities", tuple(sorted(grouped)))
    )
    missing_required_modalities = sorted(set(required_modalities) - set(grouped))
    minimum_severe = int(acceptance.get("minimum_severe_positive_samples", 1))
    minimum_auto_pass = int(acceptance.get("minimum_auto_pass_samples", 1))
    minimum_mask = int(acceptance.get("minimum_mask_evaluated_samples", 1))
    minimum_prevalence = int(acceptance.get("minimum_prevalence_samples", 1))
    if min(minimum_severe, minimum_auto_pass, minimum_mask, minimum_prevalence) < 1:
        raise ValueError("minimum evidence counts must be positive")

    modality_reports: list[dict[str, Any]] = []
    for modality_index, (modality, items) in enumerate(sorted(grouped.items())):
        severe = [item for item in items if item["severe"]]
        severe_flagged = sum(item["decision"] != "PASS" for item in severe)
        severe_sensitivity = _wilson(severe_flagged, len(severe), confidence)

        auto_pass = [item for item in items if item["decision"] == "PASS"]
        true_auto_pass = sum(not item["actionable"] for item in auto_pass)
        auto_pass_npv = _wilson(true_auto_pass, len(auto_pass), confidence)

        mask_values = [
            item["mask_precision"]
            for item in items
            if item["mask_precision"] is not None
        ]
        mask_precision = _bootstrap_mean(
            mask_values,
            confidence=confidence,
            n_resamples=int(n_resamples),
            seed=seed + modality_index,
        )
        overmask_values = [
            item["overmask"] for item in items if item["overmask"] is not None
        ]
        mean_overmask = float(np.mean(overmask_values)) if overmask_values else None
        referral_rate = float(
            sum(item["decision"] == "REVIEW" for item in items) / len(items)
        )
        fail_rate = float(
            sum(item["decision"] == "FAIL" for item in items) / len(items)
        )
        abstention_rate = float(sum(item["abstention"] for item in items) / len(items))

        gates = {
            "severe_artifact_sensitivity_lcb": _gate(
                value=severe_sensitivity["lower"],
                threshold=float(acceptance["severe_artifact_sensitivity_lcb"]),
                direction="minimum",
                evidence_count=len(severe),
                minimum_count=minimum_severe,
                eligible=eligible,
            ),
            "auto_pass_npv": _gate(
                value=auto_pass_npv["estimate"],
                threshold=float(acceptance["auto_pass_npv"]),
                direction="minimum",
                evidence_count=len(auto_pass),
                minimum_count=minimum_auto_pass,
                eligible=eligible,
            ),
            "high_confidence_mask_precision_lcb": _gate(
                value=mask_precision["lower"],
                threshold=float(acceptance["high_confidence_mask_precision_lcb"]),
                direction="minimum",
                evidence_count=len(mask_values),
                minimum_count=minimum_mask,
                eligible=eligible,
            ),
            "valid_tissue_overmask_rate_max": _gate(
                value=mean_overmask,
                threshold=float(acceptance["valid_tissue_overmask_rate_max"]),
                direction="maximum",
                evidence_count=len(overmask_values),
                minimum_count=minimum_prevalence,
                eligible=eligible,
            ),
            "review_referral_rate_max": _gate(
                value=referral_rate,
                threshold=float(acceptance["review_referral_rate_max"]),
                direction="maximum",
                evidence_count=len(items),
                minimum_count=minimum_prevalence,
                eligible=eligible,
            ),
        }
        statuses = {gate["status"] for gate in gates.values()}
        if not eligible:
            status = "NOT_EVALUATED_SYNTHETIC"
        elif "FAIL" in statuses:
            status = "FAIL"
        elif "INSUFFICIENT_EVIDENCE" in statuses:
            status = "INSUFFICIENT_EVIDENCE"
        else:
            status = "PASS"
        modality_reports.append(
            {
                "modality": modality,
                "status": status,
                "n_samples": len(items),
                "metrics": {
                    "severe_artifact_sensitivity": severe_sensitivity,
                    "auto_pass_npv": auto_pass_npv,
                    "high_confidence_mask_precision": mask_precision,
                    "mean_valid_tissue_overmask_fraction": mean_overmask,
                    "review_referral_rate": referral_rate,
                    "fail_rate": fail_rate,
                    "technical_abstention_rate": abstention_rate,
                },
                "gates": gates,
            }
        )

    statuses = {report["status"] for report in modality_reports}
    if not eligible:
        overall = "NOT_EVALUATED_SYNTHETIC"
    elif "FAIL" in statuses:
        overall = "FAIL"
    elif "INSUFFICIENT_EVIDENCE" in statuses or missing_required_modalities:
        overall = "INSUFFICIENT_EVIDENCE"
    else:
        overall = "PASS"
    return {
        "schema_version": "1.0",
        "acceptance_eligible": eligible,
        "overall_status": overall,
        "confidence": confidence,
        "required_modalities": list(required_modalities),
        "missing_required_modalities": missing_required_modalities,
        "modalities": modality_reports,
        "limitations": (
            "Synthetic data cannot satisfy an operational acceptance gate."
            if synthetic
            else "Acceptance applies only to the locked cohorts, ontology, thresholds, and actions represented."
        ),
    }


__all__ = ["DECISIONS", "evaluate_operational_decisions"]
