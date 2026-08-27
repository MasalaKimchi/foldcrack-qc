from __future__ import annotations

import builtins
import unittest
from unittest import mock

import numpy as np

from foldcrack_qc.detectors import (
    CandidateMasks,
    CleanReferenceAnomalyDetector,
    FrozenDINOv2Encoder,
    HybridQCDetector,
    classical_candidate_masks,
    connected_component_cleanup,
    fuse_score_maps,
    tile_scores_to_map,
)
from foldcrack_qc.features import (
    FLUORESCENCE_FEATURE_NAMES,
    HE_FEATURE_NAMES,
    extract_patch_feature_table,
    fluorescence_patch_features,
    he_patch_features,
    tissue_mask,
)
from foldcrack_qc.schema import Modality
from foldcrack_qc.synthetic import generate_synthetic_sample


def synthetic_he(*, include_artifacts: bool = True, seed: int = 13) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.full((128, 128, 3), 255, dtype=np.uint8)
    tissue = np.asarray([225.0, 175.0, 195.0]) + rng.normal(0.0, 1.0, (96, 96, 3))
    image[16:112, 16:112] = np.clip(tissue, 0, 255).astype(np.uint8)
    if include_artifacts:
        image[45:75, 55:90] = (105, 40, 85)  # dark/saturated fold surrogate
        image[20:108, 98:100] = 255  # thin internal tear surrogate
    return image


def synthetic_fluorescence(
    *, include_artifacts: bool = True, seed: int = 21
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image = np.zeros((3, 128, 128), dtype=np.uint16)
    tissue = np.clip(1800.0 + rng.normal(0.0, 100.0, (3, 96, 96)), 0, 65535)
    image[:, 16:112, 16:112] = tissue.astype(np.uint16)
    if include_artifacts:
        fold = np.clip(8000.0 + rng.normal(0.0, 300.0, (3, 30, 35)), 0, 65535)
        image[:, 45:75, 55:90] = fold.astype(np.uint16)
        image[:, 20:108, 98:100] = 0
    return image


class FeatureTests(unittest.TestCase):
    def test_he_tissue_mask_ignores_white_background(self) -> None:
        image = synthetic_he(include_artifacts=False)
        mask = tissue_mask(image, modality="he", min_component_size=16)
        self.assertGreater(mask[30:100, 30:100].mean(), 0.99)
        self.assertEqual(int(mask[:8].sum()), 0)
        self.assertAlmostEqual(float(mask.mean()), 96 * 96 / (128 * 128), places=2)

    def test_modality_feature_vectors_are_finite_and_fixed_length(self) -> None:
        he = he_patch_features(synthetic_he())
        fluorescence = fluorescence_patch_features(
            synthetic_fluorescence(), channel_axis=0, structural_channels=(0, 1)
        )
        self.assertEqual(tuple(he), HE_FEATURE_NAMES)
        self.assertEqual(tuple(fluorescence), FLUORESCENCE_FEATURE_NAMES)
        self.assertTrue(np.isfinite(tuple(he.values())).all())
        self.assertTrue(np.isfinite(tuple(fluorescence.values())).all())

    def test_patch_table_preserves_coordinates_and_is_deterministic(self) -> None:
        image = synthetic_he(include_artifacts=False)
        first = extract_patch_feature_table(image, patch_size=32, stride=32)
        second = extract_patch_feature_table(image, patch_size=32, stride=32)
        self.assertEqual(first.values.shape, (16, len(HE_FEATURE_NAMES)))
        np.testing.assert_array_equal(first.coordinates[0], (0, 0, 32, 32))
        np.testing.assert_array_equal(first.coordinates[-1], (96, 96, 128, 128))
        np.testing.assert_allclose(first.values, second.values, rtol=0.0, atol=0.0)

    def test_nonfinite_image_is_rejected_not_converted_to_dropout(self) -> None:
        image = synthetic_he().astype(np.float32)
        image[20, 20, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            tissue_mask(image, modality="he")


class ClassicalDetectorTests(unittest.TestCase):
    def test_he_fold_and_crack_candidates_localize_surrogates(self) -> None:
        image = synthetic_he()
        result = classical_candidate_masks(image, modality="he", min_component_size=8)

        fold_core = result.fold[47:73, 57:88]
        crack_core = result.crack[24:104, 98:100]
        self.assertGreater(float(fold_core.mean()), 0.95)
        self.assertGreater(float(crack_core.mean()), 0.95)
        self.assertGreater(float(result.fold_score[47:73, 57:88].mean()), 0.75)
        self.assertGreater(float(result.crack_score[24:104, 98:100].mean()), 0.65)

        # A discontinuity can be proposed without tracing the external specimen
        # perimeter as a crack.
        perimeter_band = np.zeros((128, 128), dtype=bool)
        perimeter_band[15:17, 16:112] = True
        perimeter_band[111:113, 16:112] = True
        perimeter_band[16:112, 15:17] = True
        perimeter_band[16:112, 111:113] = True
        self.assertLess(int(np.count_nonzero(result.crack & perimeter_band)), 8)

    def test_fluorescence_support_and_candidates_accept_channels_first(self) -> None:
        image = synthetic_fluorescence()
        result = classical_candidate_masks(
            image,
            modality="comet",
            channel_axis=0,
            min_component_size=8,
        )
        self.assertGreater(float(result.tissue[30:100, 30:90].mean()), 0.95)
        self.assertGreater(float(result.fold[47:73, 57:88].mean()), 0.90)
        self.assertGreater(float(result.crack[24:104, 98:100].mean()), 0.90)
        negative_axis_result = classical_candidate_masks(
            image,
            modality="comet",
            channel_axis=-3,
            min_component_size=8,
        )
        np.testing.assert_array_equal(result.fold, negative_axis_result.fold)
        np.testing.assert_array_equal(result.crack, negative_axis_result.crack)

    def test_connected_component_cleanup_filters_specks_and_fills(self) -> None:
        mask = np.zeros((30, 30), dtype=bool)
        mask[2, 2] = True
        mask[10:20, 10:20] = True
        mask[14:16, 14:16] = False
        cleaned = connected_component_cleanup(mask, min_area=20, fill_holes=True)
        self.assertFalse(cleaned[2, 2])
        self.assertTrue(cleaned[14:16, 14:16].all())
        self.assertEqual(int(cleaned.sum()), 100)

    def test_curved_he_crack_has_non_degenerate_multiscale_recall_and_guards(
        self,
    ) -> None:
        metrics: list[tuple[float, float, float]] = []
        for size in (128, 192, 384):
            sample = generate_synthetic_sample(Modality.HE, seed=11, size=(size, size))
            result = classical_candidate_masks(
                sample.image.data,
                modality="he",
                min_component_size=max(4, size // 32),
            )
            truth = sample.mask("crack", required=True)
            hard_negative = sample.mask("hard_negative", required=True)
            true_positive = int(np.count_nonzero(result.crack & truth))
            precision = true_positive / max(1, int(result.crack.sum()))
            recall = true_positive / max(1, int(truth.sum()))
            hard_negative_rate = int(
                np.count_nonzero(result.crack & hard_negative)
            ) / max(1, int(hard_negative.sum()))
            metrics.append((precision, recall, hard_negative_rate))

            # This is an engineering phantom check, not a clinical performance
            # claim: each scale must remain useful and avoid most labeled
            # lumen/cleft mimics.
            self.assertGreaterEqual(precision, 0.65)
            self.assertGreaterEqual(recall, 0.88)
            self.assertLessEqual(hard_negative_rate, 0.35)

            perimeter = np.zeros((size, size), dtype=bool)
            perimeter[[0, 1, -2, -1], :] = True
            perimeter[:, [0, 1, -2, -1]] = True
            self.assertEqual(int(np.count_nonzero(result.crack & perimeter)), 0)

        # Keep the measured values visible when a test fails/regresses.
        self.assertEqual(len(metrics), 3, msg=f"engineering metrics={metrics!r}")

    def test_physical_area_and_radius_are_resolution_aware(self) -> None:
        coarse = np.zeros((32, 32), dtype=bool)
        coarse[4:6, 4:6] = True  # 4 um^2: reject
        coarse[15:19, 15:19] = True  # 16 um^2: retain
        fine = np.zeros((64, 64), dtype=bool)
        fine[8:12, 8:12] = True  # same 4 um^2 object
        fine[30:38, 30:38] = True  # same 16 um^2 object

        coarse_clean = connected_component_cleanup(
            coarse,
            pixel_size_um=1.0,
            min_area_um2=12.0,
            closing_radius_um=1.0,
        )
        fine_clean = connected_component_cleanup(
            fine,
            pixel_size_um=0.5,
            min_area_um2=12.0,
            closing_radius_um=1.0,
        )
        self.assertEqual(int(coarse_clean[4:6, 4:6].sum()), 0)
        self.assertEqual(int(fine_clean[8:12, 8:12].sum()), 0)
        self.assertGreater(int(coarse_clean[15:19, 15:19].sum()), 0)
        self.assertGreater(int(fine_clean[30:38, 30:38].sum()), 0)

        with self.assertRaisesRegex(ValueError, "pixel_size_um is required"):
            connected_component_cleanup(coarse, min_area_um2=12.0)

    def test_physical_crack_geometry_is_resampling_stable(self) -> None:
        recalls: list[float] = []
        burden_multipliers: list[float] = []
        for size in (128, 192, 384):
            spacing = 128.0 / size
            sample = generate_synthetic_sample(
                Modality.HE,
                seed=11,
                size=(size, size),
                pixel_size_um=spacing,
            )
            result = classical_candidate_masks(
                sample.image.data,
                modality="he",
                min_component_size=4,
                pixel_size_um=spacing,
                min_component_area_um2=8.0,
                crack_neighborhood_radius_um=4.0,
                fold_morphology_radius_um=0.75,
            )
            truth = sample.mask("crack", required=True)
            true_positive = int(np.count_nonzero(result.crack & truth))
            recalls.append(true_positive / max(1, int(truth.sum())))
            predicted_burden_um2 = float(result.crack.sum()) * spacing * spacing
            truth_burden_um2 = float(truth.sum()) * spacing * spacing
            burden_multipliers.append(predicted_burden_um2 / truth_burden_um2)

        self.assertGreaterEqual(min(recalls), 0.90, msg=f"recalls={recalls!r}")
        self.assertLessEqual(
            max(recalls) - min(recalls), 0.08, msg=f"recalls={recalls!r}"
        )
        # Rasterized ground-truth line burden itself changes at coarse scale;
        # compare the prediction/truth physical-burden ratio at each scale.
        burden_ratio = max(burden_multipliers) / min(burden_multipliers)
        self.assertLessEqual(
            burden_ratio,
            1.08,
            msg=f"burden_multipliers={burden_multipliers!r}",
        )


class CleanReferenceTests(unittest.TestCase):
    def test_robust_mahalanobis_separates_shifted_features(self) -> None:
        rng = np.random.default_rng(7)
        clean = rng.normal(0.0, 1.0, (300, 6))
        calibration = rng.normal(0.0, 1.0, (100, 6))
        nominal = rng.normal(0.0, 1.0, (40, 6))
        anomalous = rng.normal(5.0, 1.0, (40, 6))
        detector = CleanReferenceAnomalyDetector(calibration_quantile=0.99).fit(
            clean,
            reference_group_id="fit-patients",
        )
        detector.calibrate_stitched_maps(
            [detector.score_samples(calibration).reshape(10, 10)],
            calibration_group_ids=["calibration-patients"],
        )

        nominal_scores = detector.score_samples(nominal)
        anomalous_scores = detector.score_samples(anomalous)
        self.assertGreater(
            float(np.median(anomalous_scores)), 4.0 * float(np.median(nominal_scores))
        )
        self.assertGreater(float(detector.predict(anomalous).mean()), 0.95)
        self.assertLess(float(detector.predict(nominal).mean()), 0.15)
        self.assertTrue(np.all(detector.calibrated_scores(anomalous) >= 0.0))
        self.assertTrue(np.all(detector.calibrated_scores(anomalous) <= 1.0))

    def test_threshold_is_independent_stitched_domain_and_locked(self) -> None:
        rng = np.random.default_rng(31)
        fit = rng.normal(size=(80, 4))
        detector = CleanReferenceAnomalyDetector(calibration_quantile=0.75).fit(
            fit,
            reference_group_id="fit-group",
        )
        with self.assertRaisesRegex(RuntimeError, "independent stitched clean maps"):
            detector.calibrated_scores(fit[:2])

        stitched_map = np.asarray([[1.0, 3.0], [5.0, 100.0]])
        support = np.asarray([[True, True], [True, False]])
        detector.calibrate_stitched_maps(
            [stitched_map],
            support_masks=[support],
            calibration_group_ids=["calibration-group"],
        )
        self.assertAlmostEqual(detector.threshold_, 4.0)
        self.assertAlmostEqual(float(detector.normalize_raw_scores([4.0])[0]), 0.5)
        provenance = detector.locked_threshold_provenance
        self.assertTrue(provenance["locked"])
        self.assertEqual(
            provenance["score_domain"],
            "raw_mahalanobis_after_mean_overlap_stitching_valid_pixels",
        )
        self.assertFalse(provenance["fit_calibration_exact_overlap"])
        provenance["calibration_group_ids"].append("mutation-attempt")
        self.assertNotIn(
            "mutation-attempt",
            detector.locked_threshold_provenance["calibration_group_ids"],
        )
        with self.assertRaisesRegex(RuntimeError, "already locked"):
            detector.calibrate_stitched_maps(
                [stitched_map],
                calibration_group_ids=["different-calibration-group"],
            )

    def test_fit_and_calibration_groups_cannot_overlap(self) -> None:
        rng = np.random.default_rng(4)
        unspecified = CleanReferenceAnomalyDetector().fit(rng.normal(size=(30, 3)))
        with self.assertRaisesRegex(ValueError, "explicit fit reference_group_id"):
            unspecified.calibrate_stitched_maps(
                [np.ones((4, 4))],
                calibration_group_ids=["calibration-group"],
            )
        detector = CleanReferenceAnomalyDetector().fit(
            rng.normal(size=(30, 3)),
            reference_group_id="same-group",
        )
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            detector.calibrate_stitched_maps(
                [np.ones((4, 4))],
                calibration_group_ids=["same-group"],
            )

    def test_constant_and_missing_clean_features_are_stable(self) -> None:
        clean = np.column_stack((np.ones(20), np.linspace(0.0, 1.0, 20)))
        clean[3, 1] = np.nan
        detector = CleanReferenceAnomalyDetector().fit(clean)
        scores = detector.score_samples([[1.0, np.nan], [1.0, 5.0]])
        self.assertTrue(np.isfinite(scores).all())
        self.assertGreater(scores[1], scores[0])

    def test_score_map_projection_and_fusion(self) -> None:
        coordinates = np.asarray([[0, 0, 4, 4], [2, 2, 6, 6]])
        score_map, coverage = tile_scores_to_map(
            [0.2, 0.8],
            coordinates,
            (7, 7),
            return_coverage=True,
        )
        self.assertAlmostEqual(float(score_map[0, 0]), 0.2)
        self.assertAlmostEqual(float(score_map[3, 3]), 0.5)
        self.assertAlmostEqual(float(score_map[5, 5]), 0.8)
        self.assertTrue(coverage[0, 0])
        self.assertFalse(coverage[6, 6])
        self.assertEqual(float(score_map[6, 6]), 0.0)

        score_map = score_map[:6, :6]
        support = np.ones((6, 6), dtype=bool)
        support[0, 0] = False
        fused = fuse_score_maps(
            (score_map, np.ones((6, 6))), weights=(1.0, 3.0), tissue=support
        )
        self.assertEqual(float(fused[0, 0]), 0.0)
        self.assertAlmostEqual(float(fused[5, 5]), 0.95)

    def test_hybrid_end_to_end_uses_clean_patch_bank(self) -> None:
        clean_tables = [
            extract_patch_feature_table(
                synthetic_he(include_artifacts=False, seed=seed),
                patch_size=32,
                stride=32,
            )
            for seed in range(6)
        ]
        reference = np.vstack([table.values for table in clean_tables])
        detector = HybridQCDetector().fit(
            reference,
            reference_group_id="fit-clean-images",
        )
        calibration_table = extract_patch_feature_table(
            synthetic_he(include_artifacts=False, seed=999),
            patch_size=32,
            stride=32,
        )
        calibration_raw_map = tile_scores_to_map(
            detector.anomaly_detector.score_samples(calibration_table),
            calibration_table.coordinates,
            calibration_table.image_shape,
        )
        detector.anomaly_detector.calibrate_stitched_maps(
            [calibration_raw_map],
            calibration_group_ids=["calibration-clean-images"],
        )
        result = detector.score(synthetic_he(seed=101), patch_size=32, stride=32)

        self.assertEqual(result.anomaly_score.shape, (128, 128))
        self.assertEqual(result.anomaly_coverage.shape, (128, 128))
        self.assertTrue(np.any(result.anomaly_coverage))
        self.assertEqual(result.fused_score.shape, (128, 128))
        self.assertTrue(np.isfinite(result.fused_score).all())
        self.assertTrue(
            np.all((result.fused_score >= 0.0) & (result.fused_score <= 1.0))
        )
        self.assertGreater(float(result.candidates.fold[47:73, 57:88].mean()), 0.95)
        self.assertGreater(float(result.fused_score[47:73, 57:88].mean()), 0.50)

    def test_anomaly_only_branch_can_alert_when_weighted_fusion_cannot(self) -> None:
        class AlwaysAnomalous:
            is_fitted = True

            def calibrated_scores(self, features: object) -> np.ndarray:
                return np.ones(len(features), dtype=np.float64)  # type: ignore[arg-type]

        image = synthetic_he(include_artifacts=False)
        support = np.ones(image.shape[:2], dtype=bool)
        empty = np.zeros(image.shape[:2], dtype=bool)
        zero_score = np.zeros(image.shape[:2], dtype=np.float64)
        candidates = CandidateMasks(
            support,
            empty,
            empty,
            zero_score,
            zero_score,
            review_support=support,
        )
        detector = HybridQCDetector(
            anomaly_detector=AlwaysAnomalous(),  # type: ignore[arg-type]
            classical_weight=0.99,
            anomaly_weight=0.01,
            decision_threshold=0.50,
        )
        with mock.patch(
            "foldcrack_qc.detectors.classical_candidate_masks",
            return_value=candidates,
        ):
            result = detector.score(
                image,
                modality="he",
                patch_size=32,
                stride=32,
                tissue=support,
            )

        self.assertLess(float(result.fused_score.max()), detector.decision_threshold)
        self.assertEqual(detector.anomaly_decision_threshold, 0.5)
        self.assertGreaterEqual(float(result.anomaly_score.min()), 0.99)
        self.assertGreater(float(result.predicted_mask[2:-2, 2:-2].mean()), 0.99)

    def test_hybrid_retains_crack_proposals_outside_inferred_tissue(self) -> None:
        class NominalReference:
            is_fitted = True

            def calibrated_scores(self, features: object) -> np.ndarray:
                return np.zeros(len(features), dtype=np.float64)  # type: ignore[arg-type]

        sample = generate_synthetic_sample(Modality.HE, seed=11, size=(192, 192))
        detector = HybridQCDetector(
            anomaly_detector=NominalReference(),  # type: ignore[arg-type]
            decision_threshold=0.99,
        )
        result = detector.score(
            sample.image.data,
            modality="he",
            patch_size=32,
            stride=32,
        )
        outside_tissue_crack = result.candidates.crack & ~result.candidates.tissue

        self.assertGreater(int(outside_tissue_crack.sum()), 100)
        self.assertTrue(result.candidates.review_support[outside_tissue_crack].all())
        self.assertGreater(
            float(result.predicted_mask[outside_tissue_crack].mean()), 0.99
        )

    def test_hybrid_physical_patch_context_overrides_pixel_context(self) -> None:
        class NominalReference:
            is_fitted = True

            def calibrated_scores(self, features: object) -> np.ndarray:
                return np.zeros(len(features), dtype=np.float64)  # type: ignore[arg-type]

        detector = HybridQCDetector(
            anomaly_detector=NominalReference(),  # type: ignore[arg-type]
        )
        result = detector.score(
            synthetic_he(include_artifacts=False),
            modality="he",
            patch_size=7,
            stride=7,
            patch_size_um=(32.0, 32.0),
            stride_um=(16.0, 16.0),
            pixel_size_um=(1.0, 0.5),
        )

        self.assertEqual(result.feature_table.patch_size, (32, 64))
        np.testing.assert_array_equal(
            result.feature_table.coordinates[0], (0, 0, 32, 64)
        )
        np.testing.assert_array_equal(
            result.feature_table.coordinates[1], (0, 32, 32, 96)
        )
        with self.assertRaisesRegex(ValueError, "pixel_size_um is required"):
            detector.score(
                synthetic_he(include_artifacts=False),
                patch_size_um=32.0,
            )


class OptionalEncoderTests(unittest.TestCase):
    def test_dinov2_legacy_wrapper_never_executes_unpinned_torch_hub(self) -> None:
        with (
            mock.patch("torch.hub.load") as hub_load,
            self.assertRaisesRegex(RuntimeError, "no longer permits unpinned"),
        ):
            FrozenDINOv2Encoder(allow_download=True)
        hub_load.assert_not_called()

    def test_dinov2_rejects_implicit_multiplex_channel_truncation(self) -> None:
        # Bypass dependency-bearing initialization: channel-safety validation
        # must happen before torch/model access.
        encoder = object.__new__(FrozenDINOv2Encoder)
        multiplex = np.zeros((2, 16, 16, 4), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "explicit semantic RGB projection"):
            encoder.encode(multiplex)

    def test_dinov2_dependency_failure_is_actionable(self) -> None:
        real_import = builtins.__import__

        def reject_torch(name: str, *args: object, **kwargs: object) -> object:
            if name == "torch":
                raise ImportError("simulated missing torch")
            return real_import(name, *args, **kwargs)

        with (
            mock.patch("builtins.__import__", side_effect=reject_torch),
            self.assertRaisesRegex(ImportError, "requires PyTorch"),
        ):
            FrozenDINOv2Encoder()


if __name__ == "__main__":
    unittest.main()
