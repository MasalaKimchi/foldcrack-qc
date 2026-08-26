from __future__ import annotations

import unittest

from foldcrack_qc.operational import evaluate_operational_decisions


def acceptance(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "severe_artifact_sensitivity_lcb": 0.2,
        "auto_pass_npv": 0.9,
        "high_confidence_mask_precision_lcb": 0.8,
        "valid_tissue_overmask_rate_max": 0.01,
        "review_referral_rate_max": 0.6,
        "minimum_severe_positive_samples": 2,
        "minimum_auto_pass_samples": 2,
        "minimum_mask_evaluated_samples": 2,
        "minimum_prevalence_samples": 2,
    }
    values.update(overrides)
    return values


def good_records() -> list[dict[str, object]]:
    return [
        {
            "modality": "he",
            "decision": "REVIEW",
            "reference_actionable_artifact": True,
            "reference_severe": True,
            "high_confidence_mask_precision": 0.96,
            "valid_tissue_overmask_fraction": 0.003,
        },
        {
            "modality": "he",
            "decision": "FAIL",
            "reference_actionable_artifact": True,
            "reference_severe": True,
            "high_confidence_mask_precision": 0.98,
            "valid_tissue_overmask_fraction": 0.004,
        },
        {
            "modality": "he",
            "decision": "PASS",
            "reference_actionable_artifact": False,
            "reference_severe": False,
            "valid_tissue_overmask_fraction": 0.001,
        },
        {
            "modality": "he",
            "decision": "PASS",
            "reference_actionable_artifact": False,
            "reference_severe": False,
            "valid_tissue_overmask_fraction": 0.002,
        },
    ]


class OperationalEvaluationTests(unittest.TestCase):
    def test_synthetic_records_are_never_acceptance_eligible(self) -> None:
        report = evaluate_operational_decisions(
            good_records(), acceptance(), synthetic=True, n_resamples=30
        )
        self.assertFalse(report["acceptance_eligible"])
        self.assertEqual(report["overall_status"], "NOT_EVALUATED_SYNTHETIC")
        statuses = {
            gate["status"]
            for modality in report["modalities"]
            for gate in modality["gates"].values()
        }
        self.assertEqual(statuses, {"NOT_EVALUATED"})

    def test_complete_safe_evidence_can_pass_illustrative_gates(self) -> None:
        report = evaluate_operational_decisions(
            good_records(), acceptance(), n_resamples=50, seed=3
        )
        self.assertEqual(report["overall_status"], "PASS")
        metrics = report["modalities"][0]["metrics"]
        self.assertEqual(metrics["severe_artifact_sensitivity"]["estimate"], 1.0)
        self.assertEqual(metrics["auto_pass_npv"]["estimate"], 1.0)
        self.assertAlmostEqual(metrics["review_referral_rate"], 0.25)

    def test_missing_required_stratum_is_insufficient_not_pass(self) -> None:
        records = [item for item in good_records() if not item["reference_severe"]]
        report = evaluate_operational_decisions(
            records,
            acceptance(minimum_prevalence_samples=1),
            n_resamples=20,
        )
        self.assertEqual(report["overall_status"], "INSUFFICIENT_EVIDENCE")

    def test_silent_severe_miss_fails(self) -> None:
        records = good_records()
        records[0] = {**records[0], "decision": "PASS"}
        report = evaluate_operational_decisions(
            records,
            acceptance(severe_artifact_sensitivity_lcb=0.25),
            n_resamples=20,
        )
        self.assertEqual(report["overall_status"], "FAIL")

    def test_required_modality_cannot_be_hidden_by_available_modality(self) -> None:
        report = evaluate_operational_decisions(
            good_records(),
            acceptance(required_modalities=["he", "comet"]),
            n_resamples=20,
        )
        self.assertEqual(report["overall_status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(report["missing_required_modalities"], ["comet"])

    def test_abstention_cannot_silently_pass(self) -> None:
        records = good_records()
        records[2] = {**records[2], "technical_abstention": True}
        with self.assertRaisesRegex(ValueError, "cannot result in PASS"):
            evaluate_operational_decisions(records, acceptance(), n_resamples=20)


if __name__ == "__main__":
    unittest.main()
