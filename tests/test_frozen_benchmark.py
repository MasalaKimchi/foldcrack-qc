from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from foldcrack_qc.foundation import FoundationFeatures
from foldcrack_qc.frozen_benchmark import (
    FrozenBenchmarkValidationError,
    run_frozen_anomaly_benchmark,
)


class _FakeEncoder:
    """Tiny deterministic spatial encoder with no optional runtime dependency."""

    def __init__(self, *, fail_on_bright: bool = False) -> None:
        self.batch_sizes: list[int] = []
        self.fail_on_bright = fail_on_bright

    def encode(
        self,
        images: np.ndarray,
        *,
        semantic_channels: tuple[str, str, str],
        batch_size: int,
    ) -> FoundationFeatures:
        array = np.asarray(images)
        self.batch_sizes.append(int(array.shape[0]))
        if self.fail_on_bright and float(array.max()) > 245.0:
            raise RuntimeError("PRIVATE/path/that/must/not/be/reported")
        normalized = array.astype(np.float32) / 255.0
        pooled = np.empty((array.shape[0], 2, 2), dtype=np.float32)
        for index, image in enumerate(normalized):
            pooled[index] = cv2.resize(
                image[..., 0], (2, 2), interpolation=cv2.INTER_AREA
            )
        grid = np.stack((pooled, pooled * pooled + 0.01), axis=-1)
        cls = grid.mean(axis=(1, 2))
        return FoundationFeatures(
            cls_embedding=cls,
            patch_grid=grid,
            input_size=(8, 8),
            patch_size=(4, 4),
            semantic_channels=semantic_channels,
        )


class _FakeCalibrationScorer:
    calibration_quantile = 0.75

    def __init__(self) -> None:
        self.fit_split_id: str | None = None
        self.calibration_split_id: str | None = None
        self.fit_values: np.ndarray | None = None
        self.calibration_values: np.ndarray | None = None

    def fit(self, features: np.ndarray, *, split_id: str) -> _FakeCalibrationScorer:
        self.fit_values = np.asarray(features).copy()
        self.fit_split_id = split_id
        return self

    def calibrate(
        self, features: np.ndarray, *, split_id: str
    ) -> _FakeCalibrationScorer:
        self.calibration_values = np.asarray(features).copy()
        self.calibration_split_id = split_id
        self.threshold_ = float(
            np.quantile(self.calibration_values[:, 0], self.calibration_quantile)
        )
        return self

    def score_heatmaps(
        self,
        features: FoundationFeatures,
        *,
        output_shape: tuple[int, int],
        calibrated: bool,
    ) -> np.ndarray:
        if calibrated:
            raise AssertionError("benchmark must retain the raw calibration threshold")
        maps = np.empty((features.batch_size, *output_shape), dtype=np.float64)
        for index, grid in enumerate(features.patch_grid[..., 0]):
            maps[index] = cv2.resize(
                grid.astype(np.float64),
                (output_shape[1], output_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return maps

    def raw_token_scores(self, features: FoundationFeatures) -> np.ndarray:
        return np.asarray(features.patch_grid[..., 0], dtype=np.float64).reshape(-1)


class FrozenBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _record(
        self,
        directory: Path,
        *,
        role: str,
        index: int,
        value: int,
        artifact: bool = False,
        bright_failure: bool = False,
    ) -> dict[str, object]:
        image = np.full((12, 12, 3), value, dtype=np.uint8)
        fold = np.zeros((12, 12), dtype=np.uint8)
        crack = np.zeros((12, 12), dtype=np.uint8)
        if artifact:
            image[8:, 8:] = 240
            fold[8:, 8:] = 1
        if bright_failure:
            image[0, 0] = 255
        valid = np.ones((12, 12), dtype=np.uint8)

        prefix = f"PRIVATE-{role}-{index}"
        arrays = {
            "image_path": image,
            "fold_mask_path": fold,
            "crack_mask_path": crack,
            "valid_mask_path": valid,
        }
        record: dict[str, object] = {
            "sample_id": f"PRIVATE-SAMPLE-{role}-{index}",
            "patient_id": f"PRIVATE-PATIENT-{role}-{index}",
            "block_id": f"PRIVATE-BLOCK-{role}-{index}",
            "slide_id": f"PRIVATE-SLIDE-{role}-{index}",
            "run_id": f"PRIVATE-RUN-{role}-{index}",
            "source_id": f"PRIVATE-SOURCE-{role}-{index}",
            "modality": "he",
            "channel_names": ["red", "green", "blue"],
            "channel_axis": 2,
            "color_order": "rgb",
            "pixel_size_um": [0.5, 0.5],
            "data_origin": "acquired_real",
            "provenance_status": "approved",
            "adjudication_status": "adjudicated",
            "split": {
                "fit": "train",
                "calibration": "validation",
                "locked_test": "locked_test",
            }[role],
            "cohort": "prevalence" if role == "locked_test" else "development",
        }
        for field, array in arrays.items():
            filename = f"{prefix}-{field}.npy"
            path = directory / filename
            np.save(path, array)
            record[field] = filename
            record[field.removesuffix("_path") + "_sha256"] = self._sha256(path)
        return record

    @staticmethod
    def _write_manifest(
        directory: Path, role: str, records: list[dict[str, object]]
    ) -> Path:
        path = directory / f"{role}.json"
        path.write_text(json.dumps({"samples": records}), encoding="utf-8")
        return path

    def _manifests(
        self,
        directory: Path,
        *,
        artifact_test: bool = True,
        bright_failure: bool = False,
    ) -> tuple[Path, Path, Path]:
        fit = [
            self._record(directory, role="fit", index=0, value=10),
            self._record(directory, role="fit", index=1, value=20),
        ]
        calibration = [
            self._record(directory, role="calibration", index=0, value=30),
            self._record(directory, role="calibration", index=1, value=40),
        ]
        test = [
            self._record(
                directory,
                role="locked_test",
                index=0,
                value=25,
                artifact=artifact_test,
                bright_failure=bright_failure,
            ),
            self._record(
                directory,
                role="locked_test",
                index=1,
                value=28,
            ),
        ]
        return (
            self._write_manifest(directory, "fit", fit),
            self._write_manifest(directory, "calibration", calibration),
            self._write_manifest(directory, "locked_test", test),
        )

    def test_runs_calibration_locked_union_evaluation_and_redacts_private_data(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = self._manifests(directory)
            output_path = directory / "report.json"
            encoder = _FakeEncoder()
            scorer = _FakeCalibrationScorer()

            report = run_frozen_anomaly_benchmark(
                *manifests,
                encoder=encoder,  # type: ignore[arg-type]
                scorer=scorer,  # type: ignore[arg-type]
                patch_size_px=8,
                stride_px=4,
                batch_size=2,
                n_resamples=8,
                output_json=output_path,
            )

            disjointness = report["evidence_boundary"]["split_disjointness"]
            self.assertTrue(disjointness["exact_disjointness_checks_passed"])
            self.assertFalse(disjointness["mathematical_independence_proven"])
            self.assertEqual(report["outcome_summary"]["evaluated_count"], 2)
            self.assertEqual(report["outcome_summary"]["abstained_count"], 0)
            self.assertEqual(report["run_status"], "complete")
            self.assertFalse(report["scientific_report_eligible"])
            self.assertTrue(report["development_metric_report_complete"])
            self.assertTrue(report["reference_support"]["minimum_gate_passed"])
            self.assertIn("prevalence", report["evaluation_by_cohort"])
            self.assertEqual(
                report["evaluation"]["samples"][0]["metadata"]["reference_target"],
                "artifact_union",
            )
            self.assertFalse(
                report["evaluation"]["samples"][0]["metadata"][
                    "semantic_subtype_claim"
                ]
            )
            self.assertTrue(np.isfinite(report["calibration"]["threshold"]))
            self.assertEqual(
                report["calibration"]["score_domain"],
                "native_stitched_valid_pixel_anomaly_distance",
            )
            self.assertIn("identical_stitch_path", report["calibration"]["threshold_source"])
            self.assertFalse(report["calibration"]["test_labels_used_for_threshold"])
            self.assertEqual(
                report["method"]["patch_geometry"],
                {
                    "patch_size_yx": [8.0, 8.0],
                    "stride_yx": [4.0, 4.0],
                    "units": "native_pixels",
                    "overlap_required": True,
                },
            )
            first_bootstrap = next(iter(report["evaluation"]["bootstrap"].values()))
            self.assertEqual(
                first_bootstrap["cluster_key"], "metadata.bootstrap_cluster"
            )
            self.assertIsNotNone(scorer.fit_split_id)
            self.assertIsNone(scorer.calibration_split_id)
            self.assertLessEqual(max(encoder.batch_sizes), 2)
            self.assertTrue(output_path.is_file())

            serialized = json.dumps(report)
            serialized_file = output_path.read_text(encoding="utf-8")
            for private_value in (
                "PRIVATE-SAMPLE",
                "PRIVATE-PATIENT",
                "PRIVATE-SLIDE",
                "PRIVATE-SOURCE",
                str(directory),
            ):
                self.assertNotIn(private_value, serialized)
                self.assertNotIn(private_value, serialized_file)

    def test_physical_patch_geometry_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifests = self._manifests(Path(temporary))
            report = run_frozen_anomaly_benchmark(
                *manifests,
                encoder=_FakeEncoder(),  # type: ignore[arg-type]
                scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                patch_size_um=4.0,
                stride_um=2.0,
                batch_size=3,
                n_resamples=4,
            )
            self.assertEqual(report["method"]["geometry_mode"], "physical_um")

    def test_missing_clean_mask_is_unlabeled_not_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["samples"][0].pop("crack_mask_path")
            payload["samples"][0].pop("crack_mask_sha256")
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(caught.exception.code, "explicit_fold_and_crack_masks_required")

    def test_positive_clean_mask_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[1].read_text(encoding="utf-8"))
            record = payload["samples"][0]
            mask_path = directory / record["fold_mask_path"]
            mask = np.load(mask_path, allow_pickle=False)
            mask[2:4, 2:4] = 1
            np.save(mask_path, mask)
            record["fold_mask_sha256"] = self._sha256(mask_path)
            manifests[1].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(
                caught.exception.code, "reviewed_clean_masks_must_be_all_zero"
            )

    def test_clean_negative_requires_explicit_adjudication_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["samples"][0].pop("adjudication_status")
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(
                caught.exception.code, "explicit_adjudication_status_required"
            )

    def test_locked_test_also_requires_explicit_adjudication_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[2].read_text(encoding="utf-8"))
            payload["samples"][0].pop("adjudication_status")
            manifests[2].write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(
                caught.exception.code, "explicit_adjudication_status_required"
            )

    def test_synthetic_provenance_is_rejected_before_model_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["samples"][0]["metadata"] = {"is_synthetic": True}
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")
            encoder = _FakeEncoder()
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=encoder,  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(caught.exception.code, "synthetic_provenance_rejected")
            self.assertEqual(encoder.batch_sizes, [])

    def test_absent_positive_real_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            payload["samples"][0].pop("data_origin")
            payload["samples"][0].pop("provenance_status")
            manifests[0].write_text(json.dumps(payload), encoding="utf-8")
            encoder = _FakeEncoder()
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=encoder,  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(
                caught.exception.code,
                "approved_real_acquisition_provenance_required",
            )
            self.assertEqual(encoder.batch_sizes, [])

    def test_cross_manifest_group_leakage_is_rejected_and_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifests = list(self._manifests(directory))
            fit_payload = json.loads(manifests[0].read_text(encoding="utf-8"))
            test_payload = json.loads(manifests[2].read_text(encoding="utf-8"))
            secret = fit_payload["samples"][0]["patient_id"]
            test_payload["samples"][0]["patient_id"] = secret
            manifests[2].write_text(json.dumps(test_payload), encoding="utf-8")
            with self.assertRaises(FrozenBenchmarkValidationError) as caught:
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    n_resamples=2,
                )
            self.assertEqual(caught.exception.code, "cross_manifest_split_leakage")
            self.assertNotIn(secret, str(caught.exception))

    def test_test_inference_failure_is_retained_as_anonymous_abstention(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifests = self._manifests(
                Path(temporary), artifact_test=False, bright_failure=True
            )
            report = run_frozen_anomaly_benchmark(
                *manifests,
                encoder=_FakeEncoder(fail_on_bright=True),  # type: ignore[arg-type]
                scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                patch_size_px=8,
                stride_px=4,
                batch_size=2,
                n_resamples=4,
            )
            self.assertEqual(report["outcome_summary"]["evaluated_count"], 1)
            self.assertEqual(report["outcome_summary"]["abstained_count"], 1)
            self.assertEqual(report["run_status"], "incomplete_abstentions")
            self.assertFalse(report["development_metric_report_complete"])
            self.assertFalse(report["scientific_report_eligible"])
            abstained = [
                item for item in report["outcomes"] if item["status"] == "abstained"
            ]
            self.assertEqual(len(abstained), 1)
            self.assertEqual(abstained[0]["reason_code"], "inference_RuntimeError")
            self.assertNotIn("PRIVATE/path", json.dumps(report))

    def test_all_negative_test_is_not_a_complete_metric_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifests = self._manifests(Path(temporary), artifact_test=False)
            report = run_frozen_anomaly_benchmark(
                *manifests,
                encoder=_FakeEncoder(),  # type: ignore[arg-type]
                scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                patch_size_px=8,
                stride_px=4,
                n_resamples=4,
            )
            support = report["reference_support"]["overall"]
            self.assertEqual(support["positive_sample_count"], 0)
            self.assertEqual(support["negative_sample_count"], 2)
            self.assertFalse(report["reference_support"]["minimum_gate_passed"])
            self.assertFalse(report["development_metric_report_complete"])
            self.assertFalse(report["scientific_report_eligible"])

    def test_pixel_and_physical_geometry_cannot_be_mixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifests = self._manifests(Path(temporary))
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                run_frozen_anomaly_benchmark(
                    *manifests,
                    encoder=_FakeEncoder(),  # type: ignore[arg-type]
                    scorer=_FakeCalibrationScorer(),  # type: ignore[arg-type]
                    patch_size_px=8,
                    stride_px=4,
                    patch_size_um=4.0,
                    n_resamples=2,
                )


if __name__ == "__main__":
    unittest.main()
