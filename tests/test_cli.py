from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from foldcrack_qc.cli import _clean_generated_output, entrypoint, main


class CleanCommandTests(unittest.TestCase):
    @staticmethod
    def _write_marker(directory: Path, **updates: object) -> None:
        payload: dict[str, object] = {
            "kind": "foldcrack_qc_generated_output",
            "schema_version": 1,
            "status": "complete",
        }
        payload.update(updates)
        (directory / "RUN_MANIFEST.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_clean_removes_only_versioned_output_below_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            output = root / "run-001"
            output.mkdir(parents=True)
            self._write_marker(output)
            (output / "result.json").write_text("{}", encoding="utf-8")

            with redirect_stdout(StringIO()):
                result = _clean_generated_output(output, approved_roots=(root,))
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertTrue(root.exists())

    def test_clean_refuses_root_unknown_marker_and_running_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            output = root / "run-001"
            output.mkdir(parents=True)

            with self.assertRaises(ValueError):
                _clean_generated_output(root, approved_roots=(root,))

            self._write_marker(output, kind="unrecognized")
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))
            self.assertTrue(output.exists())

            self._write_marker(output, schema_version=2)
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))

            self._write_marker(output, status="running")
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))

    def test_clean_refuses_unapproved_and_symlinked_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            approved = directory / "approved"
            outside = directory / "outside" / "run"
            outside.mkdir(parents=True)
            approved.mkdir()
            self._write_marker(outside)
            with self.assertRaises(ValueError):
                _clean_generated_output(outside, approved_roots=(approved,))
            self.assertTrue(outside.exists())

            real = approved / "real-run"
            real.mkdir()
            self._write_marker(real)
            linked = approved / "linked-run"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                _clean_generated_output(linked, approved_roots=(approved,))
            self.assertTrue(real.exists())


class EvaluationCommandTests(unittest.TestCase):
    def test_foundation_smoke_cli_wires_auditable_configuration(self) -> None:
        revision = "a" * 40
        captured = StringIO()
        with patch(
            "foldcrack_qc.foundation_smoke.run_foundation_smoke",
            return_value={
                "status": "passed",
                "result_type": "engineering_foundation_smoke_only",
            },
        ) as runner, redirect_stdout(captured):
            status = main(
                [
                    "foundation-smoke",
                    "--revision",
                    revision,
                    "--cache-dir",
                    "model-cache",
                    "--device",
                    "mps",
                    "--steady-runs",
                    "2",
                    "--lora-rank",
                    "4",
                    "--output-json",
                    "smoke.json",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(captured.getvalue())["status"], "passed")
        config = runner.call_args.args[0]
        self.assertEqual(config.revision, revision)
        self.assertEqual(config.device, "mps")
        self.assertEqual(config.steady_runs, 2)
        self.assertEqual(config.lora_rank, 4)
        self.assertFalse(config.allow_download)
        self.assertEqual(runner.call_args.kwargs["output_json"], Path("smoke.json"))

    def test_frozen_feature_cli_locks_model_provenance_and_physical_scale(
        self,
    ) -> None:
        revision = "b" * 40
        report = {
            "method": {},
            "evidence_boundary": {},
            "outcome_summary": {
                "test_sample_count": 2,
                "evaluated_count": 2,
                "abstained_count": 0,
            },
        }
        loaded = SimpleNamespace(
            model=object(),
            resolved_revision=revision,
            weight_digests=(
                SimpleNamespace(
                    as_dict=lambda: {
                        "filename": "model.safetensors",
                        "sha256": "c" * 64,
                        "size_bytes": 123,
                    }
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "benchmark.json"
            with (
                patch(
                    "foldcrack_qc.foundation_smoke.load_huggingface_model",
                    return_value=loaded,
                ) as loader,
                patch(
                    "foldcrack_qc.foundation_smoke.dinov2_model_geometry",
                    return_value=((14, 14), 1),
                ),
                patch(
                    "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                    return_value=object(),
                ),
                patch(
                    "foldcrack_qc.frozen_benchmark.run_frozen_anomaly_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "frozen-feature-benchmark",
                        "--fit-manifest",
                        "fit.json",
                        "--calibration-manifest",
                        "calibration.json",
                        "--locked-test-manifest",
                        "test.json",
                        "--revision",
                        revision,
                        "--cache-dir",
                        "model-cache",
                        "--device",
                        "mps",
                        "--patch-size-um",
                        "112",
                        "--stride-um",
                        "56",
                        "--n-resamples",
                        "10",
                        "--output-json",
                        str(output_path),
                    ]
                )

            self.assertEqual(status, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["evidence_boundary"]["model_identity_locked"])
            model = payload["method"]["model_identity"]
            self.assertEqual(model["requested_revision"], revision)
            self.assertEqual(model["resolved_revision"], revision)
            self.assertEqual(model["weight_files"][0]["sha256"], "c" * 64)
            self.assertFalse(loader.call_args.args[0].allow_download)
            self.assertEqual(runner.call_args.kwargs["patch_size_um"], 112.0)
            self.assertEqual(runner.call_args.kwargs["stride_um"], 56.0)

    def test_validate_benchmark_distinguishes_valid_plan_from_report_ready(self) -> None:
        root = Path(__file__).resolve().parents[1]
        contract = root / "configs" / "benchmark.real.example.json"
        captured = StringIO()
        with redirect_stdout(captured):
            status = main(["validate-benchmark", str(contract), "--json"])
        payload = json.loads(captured.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(payload["configuration_valid"])
        self.assertFalse(payload["report_eligible"])

        with redirect_stdout(StringIO()):
            strict_status = main(
                [
                    "validate-benchmark",
                    str(contract),
                    "--require-report-eligible",
                ]
            )
        self.assertEqual(strict_status, 3)

    def test_shared_entrypoint_converts_validation_error_to_exit_code(self) -> None:
        errors = StringIO()
        with redirect_stderr(errors):
            status = entrypoint(["validate-benchmark", "missing.json"])
        self.assertEqual(status, 2)
        self.assertIn("error:", errors.getvalue())

    def test_validate_manifest_cli_json_and_strict_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "image.npy", np.zeros((16, 18, 3), dtype=np.uint8))
            record = {
                "sample_id": "sample-opaque",
                "patient_id": "patient-opaque",
                "modality": "he",
                "image_path": "image.npy",
                "split": "development",
                "pixel_size_um": 0.5,
            }
            path = directory / "manifest.json"
            path.write_text(json.dumps([record]), encoding="utf-8")

            output = StringIO()
            with redirect_stdout(output):
                exploratory_status = main(["validate-manifest", str(path), "--json"])
            payload = json.loads(output.getvalue())
            self.assertEqual(exploratory_status, 0)
            self.assertTrue(payload["valid"])
            self.assertGreater(payload["warning_count"], 0)

            with redirect_stdout(StringIO()):
                strict_status = main(["validate-manifest", str(path), "--strict"])
            self.assertEqual(strict_status, 2)

    def test_operational_eval_preserves_synthetic_not_evaluated_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [
                {
                    "modality": "he",
                    "decision": "REVIEW",
                    "reference_actionable_artifact": True,
                    "reference_severe": True,
                    "technical_abstention": False,
                    "high_confidence_mask_precision": 1.0,
                    "valid_tissue_overmask_fraction": 0.0,
                }
            ]
            acceptance = {
                "severe_artifact_sensitivity_lcb": 0.0,
                "auto_pass_npv": 0.0,
                "high_confidence_mask_precision_lcb": 0.0,
                "valid_tissue_overmask_rate_max": 1.0,
                "review_referral_rate_max": 1.0,
                "minimum_severe_positive_samples": 1,
                "minimum_auto_pass_samples": 1,
                "minimum_mask_evaluated_samples": 1,
                "minimum_prevalence_samples": 1,
            }
            records_path = directory / "records.json"
            acceptance_path = directory / "acceptance.json"
            output_path = directory / "report.json"
            records_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

            captured = StringIO()
            with redirect_stdout(captured):
                status = main(
                    [
                        "operational-eval",
                        "--records",
                        str(records_path),
                        "--acceptance",
                        str(acceptance_path),
                        "--synthetic",
                        "--output",
                        str(output_path),
                    ]
                )
            stdout_payload = json.loads(captured.getvalue())
            file_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(stdout_payload, file_payload)
            self.assertEqual(
                stdout_payload["overall_status"], "NOT_EVALUATED_SYNTHETIC"
            )
            self.assertFalse(stdout_payload["acceptance_eligible"])

            with redirect_stdout(StringIO()):
                locked_status = main(
                    [
                        "operational-eval",
                        "--records",
                        str(records_path),
                        "--acceptance",
                        str(acceptance_path),
                    ]
                )
            self.assertEqual(locked_status, 3)


if __name__ == "__main__":
    unittest.main()
