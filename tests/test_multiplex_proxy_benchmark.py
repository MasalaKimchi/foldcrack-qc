from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

import foldcrack_qc.multiplex_proxy_benchmark as proxy_module
from foldcrack_qc.multiplex_proxy_benchmark import (
    MultiplexField,
    MultiplexProxyConfig,
    assign_group_splits,
    binary_average_precision,
    incremental_score,
    inject_multiplex_artifact,
    load_comet_dapi_tiff,
    load_cosmx_morphology_tiff,
    load_public_multiplex_fields,
    run_multiplex_proxy_benchmark,
    run_multiplex_proxy_cross_validation,
)


def _field(
    source_id: str,
    seed: int,
    *,
    modality: str = "comet",
    group_id: str | None = None,
) -> MultiplexField:
    rng = np.random.default_rng(seed)
    image = np.clip(1400.0 + rng.normal(0.0, 180.0, (2, 64, 72)), 0, 65535).astype(
        np.uint16
    )
    yy, xx = np.ogrid[:64, :72]
    for y, x in ((18, 20), (31, 48), (49, 31)):
        nucleus = (yy - y) ** 2 + (xx - x) ** 2 <= 6**2
        image[0, nucleus] = np.clip(image[0, nucleus] + 5000, 0, 65535)
    return MultiplexField(
        source_id=source_id,
        group_id=group_id or source_id,
        cohort_id=f"cohort-{group_id or source_id}",
        modality=modality,
        image=image,
        channel_names=("DAPI", "morphology"),
        source_path=f"/fixture/{source_id}.tif",
        sha256=f"{seed:064x}",
        dataset_name="generated unit-test fixture",
        source_url="https://example.invalid/fixture",
        source_axes="CYX",
        native_shape=image.shape,
        native_pixel_size_um=0.5,
        effective_pixel_size_um=(0.5, 0.5),
        pixel_size_source="unit_test",
        group_level="unit_test_source_group",
        group_independence_declared=True,
        group_independence_basis="unit_test_construction",
    )


class LoaderTests(unittest.TestCase):
    def test_streaming_area_resize_matches_full_area_and_preserves_thin_lines(
        self,
    ) -> None:
        import cv2

        rng = np.random.default_rng(42)
        array = rng.integers(0, 65_536, size=(83, 97), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            mapped = np.memmap(
                Path(directory) / "plane.bin",
                mode="w+",
                dtype=array.dtype,
                shape=array.shape,
            )
            mapped[:] = array
            mapped.flush()
            actual = proxy_module._resize_plane(mapped, (17, 19), memory_mapped=True)
            expected = cv2.resize(array, (19, 17), interpolation=cv2.INTER_AREA)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0)

            phantom = np.zeros((104, 104), dtype=np.uint16)
            phantom[5, :] = np.iinfo(np.uint16).max
            mapped_phantom = np.memmap(
                Path(directory) / "phantom.bin",
                mode="w+",
                dtype=phantom.dtype,
                shape=phantom.shape,
            )
            mapped_phantom[:] = phantom
            mapped_phantom.flush()
            thin_actual = proxy_module._resize_plane(
                mapped_phantom, (8, 8), memory_mapped=True
            )
            thin_expected = cv2.resize(phantom, (8, 8), interpolation=cv2.INTER_AREA)
            self.assertGreater(int(np.max(thin_actual)), 0)
            np.testing.assert_allclose(thin_actual, thin_expected, rtol=0.0, atol=1.0)

            for source_size, target_size in ((21, 19), (25, 11)):
                edge_case = np.arange(
                    source_size * source_size, dtype=np.uint16
                ).reshape(source_size, source_size)
                mapped_edge = np.memmap(
                    Path(directory) / f"edge-{source_size}.bin",
                    mode="w+",
                    dtype=edge_case.dtype,
                    shape=edge_case.shape,
                )
                mapped_edge[:] = edge_case
                mapped_edge.flush()
                edge_actual = proxy_module._resize_plane(
                    mapped_edge,
                    (target_size, target_size),
                    memory_mapped=True,
                )
                edge_expected = cv2.resize(
                    edge_case,
                    (target_size, target_size),
                    interpolation=cv2.INTER_AREA,
                )
                np.testing.assert_allclose(
                    edge_actual, edge_expected, rtol=0.0, atol=1.0
                )

    def test_comet_and_cosmx_loaders_preserve_explicit_cyx_contract(self) -> None:
        try:
            import tifffile
        except ImportError:
            self.skipTest("optional tifffile dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comet_path = root / "COMET_fixture_DAPI.tif"
            comet = np.arange(80 * 64, dtype=np.uint16).reshape(80, 64)
            tifffile.imwrite(comet_path, comet, photometric="minisblack")

            cosmx_path = root / "fixture_S2_F001.TIF"
            cosmx = np.stack([comet, comet + 100, comet + 200], axis=0)
            description = json.dumps(
                {
                    "ChannelOrder": "BGU",
                    "ImPixelSize_nm": 180,
                    "MorphologyKit": {
                        "MorphologyReagents": [
                            {
                                "Fluorophore": {"ChannelId": "B"},
                                "BiologicalTarget": "CD298",
                            },
                            {
                                "Fluorophore": {"ChannelId": "G"},
                                "BiologicalTarget": "PanCK",
                            },
                            {
                                "Fluorophore": {"ChannelId": "U"},
                                "BiologicalTarget": "DNA",
                            },
                        ]
                    },
                }
            )
            tifffile.imwrite(
                cosmx_path,
                cosmx,
                photometric="minisblack",
                metadata=None,
                description=description,
            )

            comet_field = load_comet_dapi_tiff(comet_path, max_dimension=48)
            cosmx_field = load_cosmx_morphology_tiff(cosmx_path, max_dimension=48)

        self.assertEqual(comet_field.image.shape[0], 1)
        self.assertLessEqual(max(comet_field.image.shape[1:]), 48)
        self.assertEqual(comet_field.channel_names, ("DAPI",))
        self.assertFalse(comet_field.group_independence_declared)
        self.assertEqual(cosmx_field.image.shape[0], 3)
        self.assertLessEqual(max(cosmx_field.image.shape[1:]), 48)
        self.assertEqual(cosmx_field.channel_names, ("CD298[B]", "PanCK[G]", "DNA[U]"))
        self.assertAlmostEqual(cosmx_field.native_pixel_size_um or 0.0, 0.18)
        self.assertEqual(cosmx_field.cohort_id, "S2")
        self.assertEqual(cosmx_field.group_id, "S2")
        self.assertEqual(len(comet_field.sha256), 64)

    def test_public_cosmx_cohorts_receive_collision_safe_run_groups(self) -> None:
        try:
            import tifffile
        except ImportError:
            self.skipTest("optional tifffile dependency is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            comet = root / "empty_comet"
            comet.mkdir()
            gastric = root / "cosmx_gastric_v1" / "raw_morphology"
            phgg = root / "cosmx_phgg_v1" / "raw_morphology"
            gastric.mkdir(parents=True)
            phgg.mkdir(parents=True)
            image = np.zeros((5, 32, 36), dtype=np.uint16)
            description = json.dumps({"ChannelOrder": "BGYRU", "ImPixelSize_nm": 180})
            tifffile.imwrite(
                gastric / "20221003_fixture_S2_F001.TIF",
                image,
                photometric="minisblack",
                metadata=None,
                description=description,
            )
            tifffile.imwrite(
                phgg / "R5779_TMA2-S6__20230505_fixture_S2_F001.TIF",
                image,
                photometric="minisblack",
                metadata=None,
                description=description,
            )
            fields = load_public_multiplex_fields(
                comet_dir=comet,
                cosmx_dir=(gastric, phgg),
                max_dimension=32,
                verify_locks=False,
            )
            lock_dir = root / "locks"
            lock_dir.mkdir()
            for index, field in enumerate(fields):
                lock = {
                    "record_url": field.source_url,
                    "files": [
                        {
                            "path": Path(field.source_path).name,
                            "group_id": field.group_id,
                            "shape_cyx": list(field.native_shape),
                            "sha256": field.sha256,
                        }
                    ],
                }
                (lock_dir / f"lock-{index}.json").write_text(
                    json.dumps(lock), encoding="utf-8"
                )
            verified_fields = load_public_multiplex_fields(
                comet_dir=comet,
                cosmx_dir=(gastric, phgg),
                max_dimension=32,
                lock_manifest_dir=lock_dir,
            )

        self.assertEqual(len(fields), 2)
        self.assertEqual(
            {field.group_id for field in fields},
            {"zenodo8333281:S2", "zenodo16877090:R5779_TMA2-S6"},
        )
        self.assertEqual(len({field.source_url for field in fields}), 2)
        self.assertTrue(all(not field.group_independence_declared for field in fields))
        self.assertTrue(
            all(not field.group_independence_declared for field in verified_fields)
        )
        self.assertTrue(all(field.lock_verified for field in verified_fields))
        self.assertTrue(all(field.lock_manifest_path for field in verified_fields))


class InjectionAndMetricTests(unittest.TestCase):
    def test_injection_is_deterministic_non_destructive_and_dtype_safe(self) -> None:
        field = _field("field-a", 1)
        source = np.array(field.image, copy=True)
        first = inject_multiplex_artifact(
            field.image, artifact="fold", severity=0.7, seed=91
        )
        second = inject_multiplex_artifact(
            field.image, artifact="fold", severity=0.7, seed=91
        )
        crack = inject_multiplex_artifact(
            field.image, artifact="crack", severity=0.7, seed=91
        )
        low = inject_multiplex_artifact(
            field.image, artifact="fold", severity=0.3, seed=91
        )

        np.testing.assert_array_equal(field.image, source)
        np.testing.assert_array_equal(first.image, second.image)
        np.testing.assert_array_equal(first.intended_mask, second.intended_mask)
        np.testing.assert_array_equal(
            first.effective_changed_mask, second.effective_changed_mask
        )
        np.testing.assert_array_equal(
            first.image[:, ~first.effective_changed_mask],
            source[:, ~first.effective_changed_mask],
        )
        np.testing.assert_array_equal(
            crack.image[:, ~crack.effective_changed_mask],
            source[:, ~crack.effective_changed_mask],
        )
        self.assertEqual(first.image.dtype, source.dtype)
        self.assertGreater(int(np.count_nonzero(first.effective_changed_mask)), 0)
        self.assertGreater(int(np.count_nonzero(crack.effective_changed_mask)), 0)
        self.assertTrue(
            np.all(
                np.any(
                    first.image[:, first.effective_changed_mask]
                    != source[:, first.effective_changed_mask],
                    axis=0,
                )
            )
        )
        self.assertTrue(
            np.all(
                np.any(
                    crack.image[:, crack.effective_changed_mask]
                    != source[:, crack.effective_changed_mask],
                    axis=0,
                )
            )
        )
        np.testing.assert_array_equal(first.intended_mask, low.intended_mask)
        self.assertTrue(np.all(first.effective_changed_mask <= first.intended_mask))
        self.assertGreaterEqual(
            int(np.sum(first.image.astype(np.int64) - source.astype(np.int64))),
            int(np.sum(low.image.astype(np.int64) - source.astype(np.int64))),
        )

    def test_effective_mask_excludes_clipped_or_zero_signal_pixels(self) -> None:
        image = np.full((2, 64, 64), 1_000, dtype=np.uint16)
        image[:, 20:32, 20:32] = 0
        image[:, 32:44, 32:44] = np.iinfo(np.uint16).max
        injected = inject_multiplex_artifact(
            image, artifact="fold", severity=1.0, seed=91
        )
        expected = np.any(injected.image != image, axis=0)
        np.testing.assert_array_equal(injected.effective_changed_mask, expected)
        self.assertTrue(
            np.all(injected.effective_changed_mask <= injected.intended_mask)
        )
        self.assertLessEqual(
            int(np.count_nonzero(injected.effective_changed_mask)),
            int(np.count_nonzero(injected.intended_mask)),
        )

    def test_incremental_score_clips_negative_changes_and_rejects_shape_mismatch(
        self,
    ) -> None:
        base = np.asarray([[0.2, 0.8], [0.5, 0.1]])
        injected = np.asarray([[0.5, 0.4], [0.5, 0.9]])
        np.testing.assert_allclose(
            incremental_score(injected, base), [[0.3, 0.0], [0.0, 0.8]]
        )
        with self.assertRaisesRegex(ValueError, "matching"):
            incremental_score(np.zeros((2, 3)), base)

    def test_average_precision_is_tie_invariant_and_empty_positive_is_explicit(
        self,
    ) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=bool)
        scores = np.asarray([0.9, 0.8, 0.7, 0.6])
        self.assertAlmostEqual(
            binary_average_precision(labels, scores) or 0.0, 5.0 / 6.0
        )
        self.assertAlmostEqual(
            binary_average_precision(
                np.asarray([1, 0], dtype=bool), np.asarray([0.5, 0.5])
            )
            or 0.0,
            0.5,
        )
        self.assertIsNone(binary_average_precision(np.zeros(4, dtype=bool), scores))


class SplitAndBenchmarkTests(unittest.TestCase):
    def test_duplicate_content_is_rejected_before_group_splitting(self) -> None:
        original = _field("field-a", 1, group_id="group-a")
        copied = replace(
            original,
            source_id="field-b",
            group_id="group-b",
            cohort_id="cohort-group-b",
            source_path="/fixture/field-b.tif",
        )
        third = _field("field-c", 3, group_id="group-c")
        with self.assertRaisesRegex(ValueError, "sha256_content"):
            assign_group_splits([original, copied, third], seed=13)

    def test_group_split_is_deterministic_and_disjoint(self) -> None:
        fields = [_field(f"field-{index}", index + 1) for index in range(5)]
        first = assign_group_splits(fields, seed=13)
        second = assign_group_splits(fields, seed=13)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"fit", "calibration", "test"})
        role_groups = {
            role: {group for group, value in first.items() if value == role}
            for role in ("fit", "calibration", "test")
        }
        self.assertFalse(role_groups["fit"] & role_groups["calibration"])
        self.assertFalse(role_groups["fit"] & role_groups["test"])
        self.assertFalse(role_groups["calibration"] & role_groups["test"])
        with self.assertRaisesRegex(ValueError, "at least three"):
            assign_group_splits(fields[:2], seed=13)

    def test_proxy_report_enforces_claim_and_threshold_boundaries(self) -> None:
        fields = [_field(f"field-{index}", index + 10) for index in range(3)]
        config = MultiplexProxyConfig(
            seed=7,
            severities=(0.45, 0.9),
            patch_size=32,
            stride=16,
            response_threshold_candidates=16,
        )
        report = run_multiplex_proxy_benchmark(fields, config)

        self.assertFalse(report["report_eligible"])
        self.assertFalse(report["scientific_validation_passed"])
        self.assertEqual(
            report["benchmark_kind"], "label_free_proxy_not_real_artifact_efficacy"
        )
        self.assertTrue(report["split"]["all_overlaps_empty"])
        test_groups = set(report["split"]["groups"]["test"])
        for threshold in report["thresholds"].values():
            response = threshold["response"]
            self.assertFalse(response["test_labels_used"])
            self.assertEqual(response["test_group_ids_used"], [])
            self.assertFalse(test_groups & set(response["calibration_group_ids"]))
            self.assertFalse(threshold["unmodified_alert_burden"]["test_data_used"])
        self.assertTrue(report["test_response_rows"])
        self.assertTrue(report["test_unmodified_field_rows"])
        self.assertTrue(
            all(
                "not_false_positive_rate" in row["label"]
                for row in report["test_unmodified_field_rows"]
            )
        )
        json.dumps(report, allow_nan=False)

    def test_empty_benchmark_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "At least one"):
            run_multiplex_proxy_benchmark([])


class CrossValidationTests(unittest.TestCase):
    def test_leave_one_group_out_is_disjoint_oof_and_group_macro(self) -> None:
        fields = [
            _field(
                f"{modality}-field-{index}",
                seed=100 * modality_index + index + 1,
                modality=modality,
                group_id=f"{modality}-group-{index}",
            )
            for modality_index, modality in enumerate(("comet", "cosmx"))
            for index in range(3)
        ]
        config = MultiplexProxyConfig(
            seed=19,
            severities=(0.6,),
            patch_size=32,
            stride=16,
            response_threshold_candidates=8,
            group_bootstrap_resamples=32,
            group_bootstrap_seed=41,
        )
        report = run_multiplex_proxy_cross_validation(fields, config)

        self.assertFalse(report["report_eligible"])
        self.assertFalse(report["scientific_validation_passed"])
        self.assertEqual(len(report["fold_manifests"]), 6)
        self.assertTrue(
            report["test_group_coverage_audit"]["every_group_tested_exactly_once"]
        )
        self.assertIn("statistically dependent", report["fold_dependence_warning"])
        self.assertTrue(
            report["group_independence_audit"][
                "all_modalities_have_declared_group_independence"
            ]
        )
        self.assertEqual(
            report["aggregation_contract"]["included_rows"], "out_of_fold_test_only"
        )

        manifest_by_id = {
            manifest["fold_id"]: manifest for manifest in report["fold_manifests"]
        }
        for manifest in report["fold_manifests"]:
            roles = manifest["groups_by_role"]
            self.assertEqual(len(roles["test"]), 1)
            self.assertEqual(len(roles["calibration"]), 1)
            self.assertEqual(len(roles["fit"]), 1)
            self.assertTrue(manifest["all_role_overlaps_empty"])
            self.assertFalse(set(roles["test"]) & set(roles["calibration"]))
            for threshold in manifest["thresholds"].values():
                provenance = threshold["response"]
                self.assertEqual(
                    provenance["calibration_group_ids"], roles["calibration"]
                )
                self.assertEqual(provenance["test_group_ids_used"], [])

        for row in report["out_of_fold_test_response_rows"]:
            manifest = manifest_by_id[row["fold_id"]]
            self.assertTrue(row["out_of_fold_test"])
            self.assertIn(row["group_id"], manifest["groups_by_role"]["test"])
        tested_groups = {
            row["group_id"] for row in report["out_of_fold_test_response_rows"]
        }
        self.assertEqual(tested_groups, {field.group_id for field in fields})

        response_aggregate = report["out_of_fold_group_macro"][
            "response_by_modality_method_artifact"
        ]
        self.assertIn("comet:classical:fold", response_aggregate)
        self.assertIn("cosmx:hybrid:crack", response_aggregate)
        for stratum in response_aggregate.values():
            self.assertEqual(stratum["n_source_groups"], 3)
            for metric in stratum["group_macro_metrics"].values():
                self.assertEqual(metric["n_groups"], 3)
                self.assertEqual(metric["ci95"]["resamples"], 32)
                self.assertLessEqual(metric["ci95"]["lower"], metric["mean"])
                self.assertGreaterEqual(metric["ci95"]["upper"], metric["mean"])
        untouched = report["out_of_fold_group_macro"]["unmodified_by_modality_method"]
        self.assertIn("comet:classical", untouched)
        self.assertIn(
            "unmodified_alert_burden_fraction",
            untouched["comet:classical"]["group_macro_metrics"],
        )
        json.dumps(report, allow_nan=False)

    def test_group_bootstrap_is_seed_deterministic(self) -> None:
        values = {"group-a": 0.1, "group-b": 0.4, "group-c": 0.9}
        first = proxy_module._group_bootstrap_summary(values, resamples=64, seed=71)
        second = proxy_module._group_bootstrap_summary(values, resamples=64, seed=71)
        self.assertEqual(first, second)
        self.assertEqual(first["n_groups"], 3)
        self.assertEqual(
            first["ci95"]["resampling_unit"],
            "declared_source_group_out_of_fold_summary",
        )


if __name__ == "__main__":
    unittest.main()
