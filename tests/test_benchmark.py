from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from foldcrack_qc.benchmark import (
    BenchmarkConfig,
    StructuralViewError,
    _binary_average_precision,
    _view,
    run_feasibility,
)
from foldcrack_qc.schema import CanonicalImage, ChannelRole, QCSample
from foldcrack_qc.synthetic import generate_synthetic_sample


def _replace_image(
    sample: QCSample,
    data: np.ndarray,
    *,
    names: list[str] | tuple[str, ...] | None = None,
    roles: list[ChannelRole] | tuple[ChannelRole, ...] | None = None,
) -> QCSample:
    image = CanonicalImage(
        data=data,
        modality=sample.modality,
        channel_names=names or sample.image.channel_names,
        channel_roles=roles or sample.image.channel_roles,
        pixel_size_um=sample.image.pixel_size_um,
        metadata=sample.image.metadata,
    )
    return QCSample(
        sample.sample_id, image, masks=sample.masks, metadata=sample.metadata
    )


class SemanticStructuralViewTests(unittest.TestCase):
    def test_comet_full_view_excludes_marker_channels_and_is_permutation_invariant(
        self,
    ) -> None:
        sample = generate_synthetic_sample("comet", seed=12, size=(128, 128))
        baseline = _view(sample, "full_structural")
        self.assertEqual(baseline.shape[-1], 3)

        marker_indices = [
            index
            for index, role in enumerate(sample.image.channel_roles)
            if role is ChannelRole.MARKER
        ]
        modified = sample.image.data.copy()
        modified[..., marker_indices] = 10_000.0
        marker_changed = _replace_image(sample, modified)
        np.testing.assert_allclose(_view(marker_changed, "full_structural"), baseline)

        permutation = np.asarray([4, 2, 0, 1, 3])
        permuted = _replace_image(
            sample,
            sample.image.data[..., permutation],
            names=[sample.image.channel_names[index] for index in permutation],
            roles=[sample.image.channel_roles[index] for index in permutation],
        )
        np.testing.assert_allclose(_view(permuted, "full_structural"), baseline)

    def test_required_role_coverage_and_missing_role_abstention(self) -> None:
        he = generate_synthetic_sample("he", seed=2, size=(128, 128))
        comet = generate_synthetic_sample("comet", seed=2, size=(128, 128))
        cosmx = generate_synthetic_sample("cosmx", seed=2, size=(128, 128))
        self.assertEqual(_view(he, "full_structural").shape[-1], 3)
        self.assertEqual(_view(comet, "minimal_structural").shape[-1], 1)
        self.assertEqual(_view(cosmx, "full_structural").shape[-1], 4)
        self.assertEqual(_view(cosmx, "minimal_structural").shape[-1], 1)

        missing_nuclear = _replace_image(
            comet,
            comet.image.data,
            roles=[ChannelRole.MARKER] * comet.image.n_channels,
        )
        with self.assertRaisesRegex(StructuralViewError, "abstain.*nuclear"):
            _view(missing_nuclear, "full_structural")

    def test_he_rgb_role_order_is_permutation_invariant(self) -> None:
        sample = generate_synthetic_sample("he", seed=8, size=(128, 128))
        baseline = _view(sample, "full_structural")
        permutation = np.asarray([2, 0, 1])
        permuted = _replace_image(
            sample,
            sample.image.data[..., permutation],
            names=[sample.image.channel_names[index] for index in permutation],
            roles=[sample.image.channel_roles[index] for index in permutation],
        )
        np.testing.assert_array_equal(_view(permuted, "full_structural"), baseline)


class FeasibilityBenchmarkTests(unittest.TestCase):
    def test_pixel_average_precision_is_tie_invariant(self) -> None:
        labels = np.asarray([True, False, True, False])
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(_binary_average_precision(labels, scores), 5.0 / 6.0)
        self.assertAlmostEqual(
            _binary_average_precision(
                np.asarray([True, False]),
                np.asarray([0.5, 0.5]),
            ),
            0.5,
        )

    def test_end_to_end_benchmark_has_factorial_and_validation_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "feasibility"
            outcome = run_feasibility(
                BenchmarkConfig(
                    output_dir=output,
                    samples_per_modality=5,
                    clean_samples_per_modality=1,
                    image_size=(128, 128),
                    patch_size=32,
                    overlays_per_modality=2,
                    bootstrap_resamples=12,
                    # One fit image is intentionally tiny for test speed; keep
                    # the production/default gate at 5% and allow this unit
                    # fixture's higher finite-sample variation explicitly.
                    max_held_out_clean_fpr=0.20,
                    seed=101,
                )
            )

            self.assertTrue(outcome["engineering_smoke_test_passed"])
            self.assertEqual(outcome["unique_image_count"], 15)
            self.assertEqual(outcome["result_count"], 105)
            self.assertTrue(all(outcome["engineering_checks"].values()))

            manifest = json.loads((output / "RUN_MANIFEST.json").read_text())
            self.assertEqual(manifest["schema_version"], 1)
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue(manifest["engineering_smoke_test_passed"])
            self.assertFalse(manifest["scientific_validation_passed"])
            self.assertIn("Synthetic", manifest["scientific_validation_reason"])
            self.assertTrue(manifest["metamorphic_diagnostics"]["passed"])
            self.assertEqual(manifest["config"]["patch_size"], 32)
            provenance = manifest["anomaly_threshold_provenance"]
            self.assertEqual(len(provenance), 6)
            for record in provenance.values():
                self.assertTrue(record["locked"])
                self.assertFalse(record["fit_calibration_exact_overlap"])
                self.assertEqual(record["normalized_decision_threshold"], 0.5)
                self.assertEqual(
                    record["score_domain"],
                    "raw_mahalanobis_after_mean_overlap_stitching_valid_pixels",
                )
                self.assertNotIn(
                    record["fit_reference_group_id"],
                    record["calibration_group_ids"],
                )
            anomaly_regression = manifest["anomaly_regression_diagnostics"]
            self.assertTrue(anomaly_regression["held_out_clean_fpr_passed"])
            self.assertTrue(anomaly_regression["positive_ranking_passed"])
            self.assertEqual(len(anomaly_regression["groups"]), 6)
            self.assertTrue(
                all(
                    group["positive_pixel_auprc"] > group["positive_pixel_prevalence"]
                    for group in anomaly_regression["groups"].values()
                )
            )

            report = json.loads((output / "evaluation_report.json").read_text())
            self.assertEqual(report["unique_image_count"], 15)
            self.assertEqual(report["prediction_count"], 105)
            self.assertEqual(report["output_group_count"], 21)
            self.assertNotIn("summary", report)
            expected_scenarios = {
                "clean",
                "hard_negative_only",
                "fold_only",
                "crack_only",
                "both",
            }
            for group in report["groups"]:
                self.assertEqual(set(group["scenario_counts"]), expected_scenarios)
                self.assertEqual(set(group["scenario_counts"].values()), {1})
                self.assertEqual(group["unique_image_count"], 5)
                self.assertEqual(group["prediction_count"], 5)

            with (output / "comparison.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 21)
            self.assertEqual(
                {row["method"] for row in rows},
                {"classical", "clean_reference_anomaly", "hybrid"},
            )
            self.assertEqual(
                {row["target"] for row in rows}, {"fold", "crack", "artifact"}
            )
            self.assertTrue(all(row["decision_rule"] for row in rows))
            self.assertTrue(all(row["runtime_semantics"] for row in rows))
            anomaly_rows = [
                row for row in rows if row["method"] == "clean_reference_anomaly"
            ]
            self.assertEqual(
                {row["decision_threshold"] for row in anomaly_rows}, {"0.5"}
            )

            with (output / "per_sample_results.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                per_sample = list(csv.DictReader(handle))
            self.assertEqual(len(per_sample), 105)
            self.assertEqual(
                {row["metadata.scenario"] for row in per_sample}, expected_scenarios
            )

            acceptance = json.loads(
                (output / "operational_acceptance.json").read_text()
            )
            self.assertFalse(acceptance["acceptance_eligible"])
            self.assertEqual(acceptance["overall_status"], "NOT_EVALUATED_SYNTHETIC")
            self.assertFalse(acceptance["decision_records_created"])

            for modality in ("he", "comet", "cosmx"):
                overlays = list((output / "overlays" / modality).glob("*.png"))
                self.assertEqual(len(overlays), 2)

            feasibility_text = (output / "FEASIBILITY_REPORT.md").read_text()
            self.assertIn("Engineering smoke test only", feasibility_text)
            self.assertIn("fold, crack, and artifact union", feasibility_text)
            self.assertIn(
                "What is still required for a performance claim", feasibility_text
            )
            self.assertIn(
                "Clean-reference anomaly calibration diagnostics", feasibility_text
            )
            self.assertIn("no second post-stitch threshold", feasibility_text)

    def test_default_patch_context_preserves_thin_crack_resolution(self) -> None:
        config = BenchmarkConfig()
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.patch_size * 0.5, 16.0)
        self.assertEqual(config.max_held_out_clean_fpr, 0.05)

    def test_requires_every_factorial_scenario(self) -> None:
        with self.assertRaisesRegex(ValueError, "every synthetic scenario"):
            BenchmarkConfig(samples_per_modality=4)

    def test_refuses_nonempty_unowned_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "foreign"
            output.mkdir()
            (output / "user-file.txt").write_text("preserve me", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "nonempty unowned"):
                run_feasibility(
                    BenchmarkConfig(
                        output_dir=output,
                        samples_per_modality=5,
                        clean_samples_per_modality=1,
                        image_size=(128, 128),
                        patch_size=32,
                        bootstrap_resamples=2,
                    )
                )
            self.assertEqual((output / "user-file.txt").read_text(), "preserve me")

    def test_refuses_reuse_of_owned_output_to_prevent_stale_overlays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "previous-run"
            output.mkdir()
            (output / "RUN_MANIFEST.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "foldcrack_qc_generated_output",
                        "status": "complete",
                    }
                ),
                encoding="utf-8",
            )
            (output / "overlays").mkdir()
            with self.assertRaisesRegex(FileExistsError, "existing benchmark output"):
                run_feasibility(
                    BenchmarkConfig(
                        output_dir=output,
                        samples_per_modality=5,
                        clean_samples_per_modality=1,
                        image_size=(128, 128),
                        patch_size=32,
                        bootstrap_resamples=2,
                    )
                )


if __name__ == "__main__":
    unittest.main()
