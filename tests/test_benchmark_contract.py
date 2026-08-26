from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from foldcrack_qc.benchmark_contract import (
    REQUIRED_REGIMES,
    REQUIRED_SPLIT_KEYS,
    BenchmarkContractError,
    load_benchmark_contract,
    validate_benchmark_contract,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_PATH = ROOT / "configs" / "benchmark.real.example.json"


class RealBenchmarkContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.example = json.loads(EXAMPLE_PATH.read_text(encoding="utf-8"))

    @staticmethod
    def _real_records() -> dict[str, list[dict[str, object]]]:
        result: dict[str, list[dict[str, object]]] = {}
        for role in ("fit", "calibration", "test"):
            result[role] = []
            for modality in ("he", "comet", "cosmx"):
                for case_index, case_type in enumerate(
                    ("fold_positive", "crack_positive", "negative")
                ):
                    prefix = f"{role}-{modality}-{case_index}"

                    def digest(suffix: str, *, record_prefix: str = prefix) -> str:
                        return hashlib.sha256(
                            f"{record_prefix}-{suffix}".encode()
                        ).hexdigest()

                    result[role].append({
                        "sample_id": f"sample-{prefix}",
                        "modality": modality,
                        "data_origin": "acquired_real",
                        "provenance_status": "approved",
                        "strict_manifest_validated": True,
                        "image_sha256": digest("image"),
                        "annotation_status": "adjudicated",
                        "annotations": {
                            "fold": f"annotations/{prefix}-fold.npy",
                            "crack": f"annotations/{prefix}-crack.npy",
                        },
                        "annotation_sha256": {
                            "fold": digest("fold"),
                            "crack": digest("crack"),
                        },
                        "reference_positive": {
                            "fold": case_type == "fold_positive",
                            "crack": case_type == "crack_positive",
                        },
                        "ignore_mask": f"annotations/{prefix}-ignore.npy",
                        "ignore_mask_sha256": digest("ignore"),
                        "patient_id": f"patient-{prefix}",
                        "block_id": f"block-{prefix}",
                        "slide_id": f"slide-{prefix}",
                        "run_id": f"run-{prefix}",
                        "content_id": f"sha256-{prefix}",
                    })
        return result

    def _eligible_config(self) -> dict[str, object]:
        config = deepcopy(self.example)
        config["ontology"].update(
            {
                "version": "foldcrack-ontology-v1",
                "crack_definition": "tissue_tear",
                "stakeholder_approved": True,
                "approval_reference": "TEST-ONTOLOGY-APPROVAL",
            }
        )
        for approval in config["governance_approvals"].values():
            approval.update(
                {
                    "status": "approved",
                    "reference": "TEST-APPROVAL",
                    "evidence_sha256": "d" * 64,
                }
            )
        # The example intentionally gates every advanced implementation. Simulate
        # explicit readiness only for two supervised and two clean-reference
        # methods so each required cell has a true same-regime head-to-head pair.
        for method in config["methods"]:
            if method["id"] in {
                "patchcore_dinov2",
                "padim_dinov2",
                "unet_supervised",
                "segformer_supervised",
            }:
                method["availability_status"] = "ready"
            elif method["id"] != "classical_custom":
                method["enabled"] = False
        dinov2_resource = next(
            resource
            for resource in config["resources"]
            if resource["id"] == "dinov2_apache_weights"
        )
        # Scientific eligibility is tested only after simulating an explicit
        # corporate approval; the checked-in example correctly remains pending.
        dinov2_resource["license_status"] = "approved"
        dinov2_resource["commercial_use_approved"] = True
        config["cohort_records"] = self._real_records()
        for role, records in config["cohort_records"].items():
            config["cohort_evidence_attestations"][role] = {
                "strict_manifest_validation_passed": True,
                "manifest_sha256": "e" * 64,
                "validation_report_sha256": "f" * 64,
                "record_count": len(records),
            }
        return config

    def test_example_is_loadable_but_truthfully_report_blocked(self) -> None:
        contract = load_benchmark_contract(EXAMPLE_PATH)
        self.assertTrue(contract.report.configuration_valid, contract.report.to_dict())
        self.assertFalse(contract.report_eligible)
        self.assertEqual(contract.report.status, "configuration_valid_report_blocked")
        self.assertEqual(
            contract.report.cohort_counts, {"fit": 0, "calibration": 0, "test": 0}
        )
        expected_methods = {
            "classical_custom",
            "histoqc",
            "grandqc",
            "qualifai",
            "patchcore_dinov2",
            "padim_dinov2",
            "dinov2_linear_probe",
            "dinov2_decoder",
            "dinov2_lora",
            "unet_supervised",
            "segformer_supervised",
            "diffusionqc",
        }
        self.assertSetEqual(set(contract.method_ids), expected_methods)
        self.assertIn(
            "cohort_records_unverified",
            {issue.code for issue in contract.report.blockers},
        )
        with self.assertRaises(BenchmarkContractError):
            load_benchmark_contract(EXAMPLE_PATH, require_report_eligible=True)

    def test_fully_real_nonoverlapping_contract_is_report_eligible(self) -> None:
        report = validate_benchmark_contract(self._eligible_config())
        self.assertTrue(report.configuration_valid, report.to_dict())
        self.assertTrue(report.report_eligible, report.to_dict())
        self.assertEqual(report.status, "scientific_report_eligible")
        self.assertSetEqual(
            set(report.eligible_method_ids),
            {
                "classical_custom",
                "patchcore_dinov2",
                "padim_dinov2",
                "unet_supervised",
                "segformer_supervised",
            },
        )
        self.assertEqual(report.cohort_counts, {"fit": 9, "calibration": 9, "test": 9})

    def test_synthetic_data_cannot_support_scientific_acceptance(self) -> None:
        config = self._eligible_config()
        config["cohort_records"]["test"][0]["data_origin"] = "synthetic"
        config["cohort_records"]["test"][0]["is_synthetic"] = True
        report = validate_benchmark_contract(config)
        self.assertFalse(report.configuration_valid)
        self.assertFalse(report.report_eligible)
        self.assertIn(
            "synthetic_efficacy_data_forbidden",
            {issue.code for issue in report.errors},
        )

    def test_policy_cannot_opt_synthetic_data_into_acceptance(self) -> None:
        config = self._eligible_config()
        config["scientific_acceptance"][
            "allow_synthetic_for_scientific_acceptance"
        ] = True
        report = validate_benchmark_contract(config)
        self.assertIn(
            "synthetic_acceptance_forbidden", {issue.code for issue in report.errors}
        )

    def test_every_split_identifier_prevents_cross_cohort_leakage(self) -> None:
        for key in REQUIRED_SPLIT_KEYS:
            with self.subTest(key=key):
                config = self._eligible_config()
                private_value = f"PRIVATE-SENSITIVE-{key}-1293"
                config["cohort_records"]["fit"][0][key] = private_value
                config["cohort_records"]["test"][0][key] = private_value
                report = validate_benchmark_contract(config)
                issues = [
                    issue
                    for issue in report.errors
                    if issue.code == "cohort_identifier_overlap"
                    and issue.path.endswith(key)
                ]
                self.assertEqual(len(issues), 1, report.to_dict())
                self.assertNotIn(private_value, json.dumps(report.to_dict()))

    def test_missing_split_identifier_blocks_unverifiable_data(self) -> None:
        config = self._eligible_config()
        config["cohort_records"]["calibration"][1].pop("content_id")
        report = validate_benchmark_contract(config)
        self.assertTrue(report.configuration_valid)
        self.assertFalse(report.report_eligible)
        self.assertIn(
            "missing_split_identifier", {issue.code for issue in report.blockers}
        )

    def test_real_annotations_and_ignore_masks_are_required_per_record(self) -> None:
        config = self._eligible_config()
        record = config["cohort_records"]["test"][0]
        record["annotations"].pop("crack")
        record.pop("ignore_mask")
        report = validate_benchmark_contract(config)
        blocker_codes = {issue.code for issue in report.blockers}
        self.assertIn("missing_real_annotation", blocker_codes)
        self.assertIn("missing_record_ignore_mask", blocker_codes)
        self.assertFalse(report.report_eligible)

    def test_unadjudicated_ground_truth_is_not_eligible(self) -> None:
        config = self._eligible_config()
        config["cohort_records"]["fit"][0]["annotation_status"] = "single_reviewer"
        report = validate_benchmark_contract(config)
        self.assertIn(
            "unaccepted_annotation_status", {issue.code for issue in report.blockers}
        )

    def test_adjudication_requirement_cannot_be_configured_away(self) -> None:
        config = self._eligible_config()
        config["scientific_acceptance"]["accepted_annotation_statuses"] = [
            "unreviewed"
        ]
        for records in config["cohort_records"].values():
            for record in records:
                record["annotation_status"] = "unreviewed"
        report = validate_benchmark_contract(config)
        self.assertFalse(report.configuration_valid)
        self.assertIn(
            "adjudication_requirement_cannot_be_relaxed",
            {issue.code for issue in report.errors},
        )

    def test_strict_evidence_attestation_and_checksums_are_required(self) -> None:
        config = self._eligible_config()
        config["cohort_evidence_attestations"]["test"][
            "strict_manifest_validation_passed"
        ] = False
        config["cohort_records"]["test"][0]["image_sha256"] = "0" * 64
        report = validate_benchmark_contract(config)
        blocker_codes = {issue.code for issue in report.blockers}
        self.assertIn("strict_manifest_validation_unverified", blocker_codes)
        self.assertIn("record_image_checksum_missing", blocker_codes)
        self.assertFalse(report.report_eligible)

    def test_ontology_and_governance_approval_are_hard_report_gates(self) -> None:
        config = self._eligible_config()
        config["ontology"]["stakeholder_approved"] = False
        config["governance_approvals"]["method_assets"]["status"] = "pending"
        report = validate_benchmark_contract(config)
        blocker_codes = {issue.code for issue in report.blockers}
        self.assertIn("ontology_not_approved", blocker_codes)
        self.assertIn("governance_approval_pending", blocker_codes)

    def test_reporting_and_full_multimodal_task_coverage_are_required(self) -> None:
        config = self._eligible_config()
        config.pop("reporting")
        config["comparisons"] = [config["comparisons"][0]]
        report = validate_benchmark_contract(config)
        error_codes = {issue.code for issue in report.errors}
        self.assertIn("missing_reporting_contract", error_codes)
        self.assertIn("incomplete_comparison_coverage", error_codes)

    def test_comparisons_require_same_regime_and_modality_input_variant(self) -> None:
        config = self._eligible_config()
        comparison = config["comparisons"][0]
        comparison["regime"] = "clean_reference_anomaly"
        comparison["input_variant"] = "comet.structural_rgb"
        report = validate_benchmark_contract(config)
        codes = {issue.code for issue in report.errors}
        self.assertIn("comparison_input_variant_modality_mismatch", codes)
        self.assertIn("method_regime_incompatible", codes)
        self.assertIn("method_input_variant_incompatible", codes)

    def test_all_modalities_need_realized_records_in_every_cohort(self) -> None:
        config = self._eligible_config()
        config["cohort_records"]["calibration"] = [
            record
            for record in config["cohort_records"]["calibration"]
            if record["modality"] != "cosmx"
        ]
        report = validate_benchmark_contract(config)
        findings = [
            issue
            for issue in report.blockers
            if issue.code == "incomplete_cohort_modality_coverage"
        ]
        self.assertEqual(len(findings), 1)
        self.assertIn("cosmx", findings[0].message)

    def test_patch_classifier_cannot_enter_pixel_localization_comparison(self) -> None:
        config = self._eligible_config()
        comparison = config["comparisons"][0]
        comparison["method_ids"].append("dinov2_linear_probe")
        report = validate_benchmark_contract(config)
        self.assertIn(
            "method_task_incompatible", {issue.code for issue in report.errors}
        )
        with self.assertRaises(BenchmarkContractError):
            load_benchmark_contract(config)

    def test_generic_anomaly_method_cannot_claim_fold_or_crack_semantics(self) -> None:
        config = self._eligible_config()
        fold_comparison = config["comparisons"][0]
        fold_comparison["method_ids"].append("patchcore_dinov2")
        report = validate_benchmark_contract(config)
        self.assertIn(
            "method_task_incompatible", {issue.code for issue in report.errors}
        )

    def test_output_and_metric_mismatches_are_rejected(self) -> None:
        config = self._eligible_config()
        comparison = config["comparisons"][0]
        comparison["comparison_output_type"] = "slide_score"
        comparison["metric_ids"] = ["auprc"]
        report = validate_benchmark_contract(config)
        codes = {issue.code for issue in report.errors}
        self.assertIn("comparison_metric_output_mismatch", codes)
        self.assertIn("method_output_incompatible", codes)
        self.assertIn("comparison_metric_incompatible", codes)

    def test_regime_stratification_cannot_be_disabled(self) -> None:
        config = self._eligible_config()
        config["comparisons"][0]["stratify_by_regime"] = False
        report = validate_benchmark_contract(config)
        self.assertIn("regime_pooling_forbidden", {issue.code for issue in report.errors})

    def test_all_five_regime_contracts_and_cohort_semantics_are_required(self) -> None:
        config = self._eligible_config()
        config["regimes"] = [
            regime for regime in config["regimes"] if regime["id"] != "lora"
        ]
        report = validate_benchmark_contract(config)
        self.assertIn("missing_required_regime", {issue.code for issue in report.errors})

        config = self._eligible_config()
        lora = next(regime for regime in config["regimes"] if regime["id"] == "lora")
        lora["required_cohorts"] = ["fit", "test"]
        report = validate_benchmark_contract(config)
        self.assertIn("invalid_regime_cohorts", {issue.code for issue in report.errors})
        self.assertTupleEqual(REQUIRED_REGIMES, tuple(EXPECTED_REGIME_IDS))

    def test_semantic_channels_are_required_and_positional_indices_forbidden(self) -> None:
        config = self._eligible_config()
        variant = config["modalities"]["comet"]["input_variants"][0]
        variant["channel_selection"] = "position"
        variant["channel_indices"] = [0]
        report = validate_benchmark_contract(config)
        positional = [
            issue for issue in report.errors if issue.code == "positional_channel_selection_forbidden"
        ]
        self.assertGreaterEqual(len(positional), 2)

    def test_license_and_resource_gates_exclude_required_methods(self) -> None:
        config = self._eligible_config()
        patchcore = next(
            method for method in config["methods"] if method["id"] == "patchcore_dinov2"
        )
        patchcore["required_for_acceptance"] = True
        dinov2 = next(
            resource
            for resource in config["resources"]
            if resource["id"] == "dinov2_apache_weights"
        )
        dinov2["commercial_use_approved"] = False
        report = validate_benchmark_contract(config)
        self.assertNotIn("patchcore_dinov2", report.eligible_method_ids)
        self.assertIn("patchcore_dinov2", report.gated_method_ids)
        self.assertIn(
            "required_method_unavailable", {issue.code for issue in report.blockers}
        )
        self.assertIn(
            "insufficient_eligible_comparison_methods",
            {issue.code for issue in report.blockers},
        )

    def test_external_cohort_records_override_planning_placeholder(self) -> None:
        config = self._eligible_config()
        config.pop("cohort_records")
        records = self._real_records()
        report = validate_benchmark_contract(config, cohort_records=records)
        self.assertTrue(report.report_eligible, report.to_dict())

    def test_flat_cohort_record_list_is_supported(self) -> None:
        config = self._eligible_config()
        records = []
        for role, items in config.pop("cohort_records").items():
            for item in items:
                item["cohort_role"] = role
                records.append(item)
        report = validate_benchmark_contract(config, cohort_records=records)
        self.assertTrue(report.report_eligible, report.to_dict())

    def test_report_serialization_and_raise_helpers(self) -> None:
        report = validate_benchmark_contract(self.example)
        payload = report.to_dict()
        self.assertEqual(payload["status"], "configuration_valid_report_blocked")
        self.assertFalse(payload["report_eligible"])
        report.raise_for_errors()
        with self.assertRaises(BenchmarkContractError):
            report.raise_if_not_report_eligible()

        invalid = deepcopy(self.example)
        invalid["schema_version"] = "999"
        invalid_report = validate_benchmark_contract(invalid)
        with self.assertRaises(BenchmarkContractError):
            invalid_report.raise_for_errors()

    def test_non_object_json_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "contract.json"
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(BenchmarkContractError):
                load_benchmark_contract(path)


# Keeping the expected sequence local makes accidental renaming of a reporting
# regime a deliberate test update rather than a silent compatibility break.
EXPECTED_REGIME_IDS = (
    "native_zero_shot",
    "calibrated",
    "clean_reference_anomaly",
    "shallow_adaptation",
    "lora",
    "full_finetune",
)


if __name__ == "__main__":
    unittest.main()
