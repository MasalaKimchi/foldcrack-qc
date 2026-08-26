from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from foldcrack_qc.evaluation import (  # noqa: E402
    aggregate_by_slide,
    aggregate_results,
    bootstrap_ci_by_cluster,
    bootstrap_ci_by_sample,
    boundary_metrics,
    build_report,
    burden_metrics,
    centerline_metrics,
    confusion_counts,
    evaluate_sample,
    froc_counts,
    instance_metrics,
    pixel_metrics,
    report_to_markdown,
    runtime_summary,
    skeletonize_binary,
    write_csv_report,
    write_json_report,
    write_markdown_report,
)


class PixelMetricTests(unittest.TestCase):
    def test_known_confusion_and_pixel_metrics(self) -> None:
        target = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        prediction = np.array([[1, 0], [1, 0]], dtype=np.uint8)

        counts = confusion_counts(target, prediction)
        self.assertEqual(counts, {"tp": 1, "fp": 1, "fn": 1, "tn": 1, "n_valid": 4})
        metrics = pixel_metrics(target, prediction)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mcc"], 0.0)

    def test_valid_mask_excludes_ignored_error(self) -> None:
        target = np.zeros((2, 2), dtype=bool)
        prediction = np.zeros((2, 2), dtype=bool)
        prediction[0, 0] = True
        valid = np.ones((2, 2), dtype=bool)
        valid[0, 0] = False

        metrics = pixel_metrics(target, prediction, valid)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["n_valid"], 3)
        self.assertEqual(metrics["dice"], 1.0)

    def test_empty_masks_are_correct_negative(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        metrics = pixel_metrics(empty, empty)
        self.assertEqual(metrics["dice"], 1.0)
        self.assertEqual(metrics["iou"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)


class GeometryMetricTests(unittest.TestCase):
    def test_boundary_tolerance_accepts_one_pixel_shift(self) -> None:
        target = np.zeros((9, 9), dtype=bool)
        prediction = np.zeros_like(target)
        target[2:7, 3] = True
        prediction[2:7, 4] = True

        strict = boundary_metrics(target, prediction, tolerance=0)
        tolerant = boundary_metrics(target, prediction, tolerance=1)
        self.assertEqual(strict["surface_dice"], 0.0)
        self.assertEqual(tolerant["surface_dice"], 1.0)
        self.assertAlmostEqual(tolerant["hd95"], 1.0)

    def test_centerline_metric_handles_width_disagreement(self) -> None:
        target = np.zeros((11, 11), dtype=bool)
        prediction = np.zeros_like(target)
        target[2:9, 5] = True
        prediction[2:9, 4:7] = True

        skeleton = skeletonize_binary(prediction)
        self.assertGreater(int(skeleton.sum()), 0)
        metrics = centerline_metrics(target, prediction, tolerance=1)
        self.assertGreaterEqual(metrics["centerline_f1"], 0.8)
        self.assertGreaterEqual(metrics["cldice"], 0.8)

    def test_instance_matching_counts_tp_fp_fn(self) -> None:
        target = np.zeros((10, 10), dtype=bool)
        prediction = np.zeros_like(target)
        target[1:3, 1:3] = True
        target[6:8, 6:8] = True
        prediction[1:3, 1:3] = True
        prediction[1:3, 6:8] = True

        metrics = instance_metrics(target, prediction, iou_threshold=0.5)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertAlmostEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["matches"][0]["iou"], 1.0)

    def test_zero_iou_threshold_does_not_match_disjoint_components(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        prediction = np.zeros_like(target)
        target[1:3, 1:3] = True
        prediction[5:7, 5:7] = True

        metrics = instance_metrics(target, prediction, iou_threshold=0.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)

    def test_zero_iou_threshold_does_not_match_disconnected_objects(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        prediction = np.zeros_like(target)
        target[1:3, 1:3] = True
        prediction[5:7, 5:7] = True

        metrics = instance_metrics(target, prediction, iou_threshold=0.0)
        self.assertEqual(metrics["tp"], 0)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)

    def test_froc_counts_at_multiple_thresholds(self) -> None:
        target = np.zeros((8, 8), dtype=bool)
        target[1:3, 1:3] = True
        scores = np.zeros((8, 8), dtype=float)
        scores[1:3, 1:3] = 0.8
        scores[5:7, 5:7] = 0.6

        rows = froc_counts(target, scores, [0.5, 0.7], iou_threshold=0.5)
        self.assertEqual(rows[0]["tp"], 1)
        self.assertEqual(rows[0]["fp"], 1)
        self.assertEqual(rows[1]["tp"], 1)
        self.assertEqual(rows[1]["fp"], 0)


class BurdenAggregationAndReportTests(unittest.TestCase):
    def _sample_results(self) -> list[dict]:
        positive = np.zeros((6, 6), dtype=bool)
        positive[2:4, 2:4] = True
        false_positive = np.zeros((6, 6), dtype=bool)
        false_positive[0:2, 0:2] = True
        empty = np.zeros((6, 6), dtype=bool)
        return [
            evaluate_sample(
                positive,
                positive,
                sample_id="a",
                slide_id="slide-1",
                modality="H&E",
                runtime_seconds=1.0,
            ),
            evaluate_sample(
                empty,
                false_positive,
                sample_id="b",
                slide_id="slide-1",
                modality="H&E",
                runtime_seconds=3.0,
            ),
        ]

    def test_burden_fraction_and_area_errors(self) -> None:
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        prediction = np.array([[1, 1], [1, 0]], dtype=bool)
        metrics = burden_metrics(target, prediction, pixel_area=2.0)
        self.assertEqual(metrics["true_fraction"], 0.5)
        self.assertEqual(metrics["predicted_fraction"], 0.75)
        self.assertEqual(metrics["absolute_fraction_error"], 0.25)
        self.assertEqual(metrics["true_area"], 4.0)
        self.assertEqual(metrics["absolute_area_error"], 2.0)

    def test_sample_evaluation_and_slide_aggregation(self) -> None:
        results = self._sample_results()
        summary = aggregate_results(results)
        self.assertEqual(summary["n_samples"], 2)
        self.assertEqual(summary["n_slides"], 1)
        self.assertAlmostEqual(summary["runtime"]["mean_seconds"], 2.0)
        self.assertEqual(summary["instance"]["tp"], 1)
        self.assertEqual(summary["instance"]["fp"], 1)

        slides = aggregate_by_slide(results)
        self.assertEqual(len(slides), 1)
        self.assertEqual(slides[0]["group"]["slide_id"], "slide-1")

    def test_physical_minimum_instance_area_converts_with_spacing(self) -> None:
        coarse = np.zeros((8, 8), dtype=bool)
        coarse[2:4, 2:4] = True
        fine = np.zeros((16, 16), dtype=bool)
        fine[4:8, 4:8] = True

        coarse_result = evaluate_sample(
            coarse,
            coarse,
            spacing=(1.0, 1.0),
            min_instance_area_physical=3.0,
        )
        fine_result = evaluate_sample(
            fine,
            fine,
            spacing=(0.5, 0.5),
            min_instance_area_physical=3.0,
        )
        self.assertEqual(coarse_result["config"]["min_instance_area"], 3)
        self.assertEqual(fine_result["config"]["min_instance_area"], 12)
        self.assertEqual(coarse_result["instance"]["tp"], 1)
        self.assertEqual(fine_result["instance"]["tp"], 1)

    def test_bootstrap_ci_is_seeded_and_sample_based(self) -> None:
        results = self._sample_results()
        first = bootstrap_ci_by_sample(results, "pixel.dice", n_resamples=200, seed=11)
        second = bootstrap_ci_by_sample(results, "pixel.dice", n_resamples=200, seed=11)
        self.assertEqual(first, second)
        self.assertEqual(first["estimate"], 0.5)
        self.assertLessEqual(first["lower"], first["estimate"])
        self.assertGreaterEqual(first["upper"], first["estimate"])

    def test_cluster_bootstrap_targets_displayed_pooled_metric(self) -> None:
        results = self._sample_results()
        interval = bootstrap_ci_by_cluster(
            results, "pixel.dice", n_resamples=100, seed=5
        )
        summary = aggregate_results(results)
        self.assertAlmostEqual(interval["estimate"], summary["pixel"]["dice"])
        self.assertAlmostEqual(interval["estimate"], 2.0 / 3.0)
        # Both sample evaluations deliberately share slide-1.
        self.assertEqual(interval["n_clusters"], 1)

    def test_runtime_summary_accepts_seconds_or_results(self) -> None:
        direct = runtime_summary([1.0, 2.0, 3.0])
        from_results = runtime_summary(self._sample_results())
        self.assertEqual(direct["total_seconds"], 6.0)
        self.assertEqual(from_results["total_seconds"], 4.0)
        self.assertEqual(from_results["median_seconds"], 2.0)

    def test_json_csv_and_markdown_reports(self) -> None:
        results = self._sample_results()
        report = build_report(
            results,
            bootstrap_metrics=("pixel.dice",),
            n_resamples=100,
            seed=7,
        )
        markdown = report_to_markdown(report)
        self.assertIn("# Fold/Crack QC Evaluation", markdown)
        self.assertIn("H&E", markdown)

        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            json_path = write_json_report(report, output / "report.json")
            csv_path = write_csv_report(results, output / "samples.csv")
            markdown_path = write_markdown_report(report, output / "report.md")

            parsed = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["schema_version"], "1.0")
            with csv_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("pixel.dice", rows[0])
            self.assertEqual(markdown_path.read_text(encoding="utf-8"), markdown)


class ValidationTests(unittest.TestCase):
    def test_shape_and_parameter_validation(self) -> None:
        with self.assertRaises(ValueError):
            pixel_metrics(np.zeros((2, 2)), np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            boundary_metrics(np.zeros((2, 2)), np.zeros((2, 2)), tolerance=-1)
        with self.assertRaises(ValueError):
            instance_metrics(np.zeros((2, 2)), np.zeros((2, 2)), connectivity=6)
        with self.assertRaises(ValueError):
            evaluate_sample(np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pixel_metrics(
                np.array([[0.0, np.nan], [0.0, 1.0]]),
                np.zeros((2, 2)),
            )
        with self.assertRaisesRegex(ValueError, "binary"):
            pixel_metrics(
                np.array([[0, 1], [2, 0]], dtype=np.uint8),
                np.zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
