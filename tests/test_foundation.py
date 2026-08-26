from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from foldcrack_qc.foundation import (
    DINOv2FeatureExtractor,
    FoundationFeatures,
    PatchKNNAnomalyScorer,
    foundation_runtime_diagnostics,
    preprocess_dinov2_rgb,
    reconstruct_anomaly_heatmaps,
    select_torch_device,
    validate_semantic_rgb,
)


class SemanticRGBTests(unittest.TestCase):
    def test_rejects_implicit_multiplex_truncation(self) -> None:
        multiplex = np.zeros((2, 24, 24, 5), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "semantic RGB projection"):
            validate_semantic_rgb(
                multiplex,
                semantic_channels=("DAPI", "panCK", "CD45"),
            )

    def test_requires_unit_interval_for_float_input(self) -> None:
        rgb = np.full((8, 8, 3), 128.0, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            validate_semantic_rgb(rgb)

    def test_uint8_scaling_is_explicit_and_deterministic(self) -> None:
        rgb = np.full((2, 9, 7, 3), 255, dtype=np.uint8)
        first = validate_semantic_rgb(rgb)
        second = validate_semantic_rgb(rgb)
        self.assertEqual(first.shape, (2, 9, 7, 3))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, np.ones_like(first))

    def test_preprocessing_returns_deterministic_nchw(self) -> None:
        y, x = np.mgrid[:18, :22]
        image = np.stack(
            (y / 17.0, x / 21.0, (x + y) / 38.0),
            axis=-1,
        ).astype(np.float32)
        first = preprocess_dinov2_rgb(image, image_size=224)
        second = preprocess_dinov2_rgb(image, image_size=(224, 224))
        self.assertEqual(first.shape, (1, 3, 224, 224))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_semantic_channel_names_are_auditable(self) -> None:
        rgb = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_semantic_rgb(
                rgb,
                semantic_channels=("DAPI", "DAPI", "autofluorescence"),
            )


class FeatureContainerTests(unittest.TestCase):
    def test_container_exposes_cls_and_spatial_grid(self) -> None:
        features = FoundationFeatures(
            cls_embedding=np.zeros((2, 6), dtype=np.float32),
            patch_grid=np.zeros((2, 4, 5, 6), dtype=np.float32),
            input_size=(224, 224),
            patch_size=(14, 14),
            semantic_channels=("DAPI", "panCK", "autofluorescence"),
        )
        self.assertEqual(features.batch_size, 2)
        self.assertEqual(features.grid_shape, (4, 5))
        self.assertEqual(features.embedding_dim, 6)

    def test_container_rejects_mismatched_embedding_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "share batch and embedding"):
            FoundationFeatures(
                cls_embedding=np.zeros((2, 7), dtype=np.float32),
                patch_grid=np.zeros((2, 4, 4, 6), dtype=np.float32),
                input_size=(224, 224),
                patch_size=(14, 14),
                semantic_channels=("red", "green", "blue"),
            )


class PatchKNNTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = np.asarray(
            (
                (1.00, 0.00),
                (0.99, 0.01),
                (0.98, -0.01),
                (0.97, 0.02),
            ),
            dtype=np.float64,
        )
        self.calibration = np.asarray(
            (
                (0.96, 0.03),
                (0.95, -0.02),
                (0.94, 0.04),
                (0.93, -0.03),
            ),
            dtype=np.float64,
        )

    def test_fit_and_calibration_are_separate(self) -> None:
        scorer = PatchKNNAnomalyScorer(calibration_quantile=0.9)
        scorer.fit(self.reference, split_id="fit-slides")
        with self.assertRaisesRegex(RuntimeError, "calibrate"):
            scorer.calibrated_token_scores(self.calibration)
        scorer.calibrate(self.calibration, split_id="calibration-slides")
        scores = scorer.calibrated_token_scores(
            np.asarray(((0.95, 0.01), (0.0, 1.0)), dtype=np.float64)
        )
        self.assertLess(float(scores[0]), float(scores[1]))
        self.assertEqual(float(scores[1]), 1.0)

    def test_exact_copy_of_fit_bank_cannot_be_used_for_calibration(self) -> None:
        scorer = PatchKNNAnomalyScorer().fit(self.reference)
        with self.assertRaisesRegex(ValueError, "duplicate the fit bank"):
            scorer.calibrate(self.reference.copy())

    def test_split_identifier_prevents_regenerated_split_leakage(self) -> None:
        scorer = PatchKNNAnomalyScorer().fit(
            self.reference,
            split_id="same-slides",
        )
        with self.assertRaisesRegex(ValueError, "split_id matches"):
            scorer.calibrate(
                self.calibration,
                split_id="same-slides",
            )

    def test_spatial_tokens_reconstruct_localized_heatmap(self) -> None:
        scorer = PatchKNNAnomalyScorer(calibration_quantile=0.9)
        scorer.fit(self.reference, split_id="fit")
        scorer.calibrate(self.calibration, split_id="calibration")
        query = np.asarray(
            [
                [
                    [[1.0, 0.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ]
            ],
            dtype=np.float64,
        )
        heatmaps = scorer.score_heatmaps(query, output_shape=(16, 16))
        self.assertEqual(heatmaps.shape, (1, 16, 16))
        self.assertGreater(float(heatmaps[0, -1, -1]), float(heatmaps[0, 0, 0]))

    def test_heatmap_requires_spatial_tokens(self) -> None:
        scorer = PatchKNNAnomalyScorer().fit(self.reference)
        scorer.calibrate(self.calibration)
        with self.assertRaisesRegex(ValueError, "requires spatial"):
            scorer.score_heatmaps(self.calibration)

    def test_standalone_heatmap_preserves_single_map_rank(self) -> None:
        token_scores = np.asarray(((0.0, 0.2), (0.5, 1.0)))
        heatmap = reconstruct_anomaly_heatmaps(token_scores, (10, 12))
        self.assertEqual(heatmap.shape, (10, 12))
        self.assertAlmostEqual(float(heatmap[-1, -1]), 1.0)


class _FakeMPSBackend:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_built() -> bool:
        return True


class _FakeMPSMemory:
    @staticmethod
    def current_allocated_memory() -> int:
        return 1_024

    @staticmethod
    def driver_allocated_memory() -> int:
        return 2_048

    @staticmethod
    def recommended_max_memory() -> int:
        return 16_384


class _FakeTorchRuntime:
    __version__ = "test"
    backends = SimpleNamespace(mps=_FakeMPSBackend())
    mps = _FakeMPSMemory()


class RuntimeDiagnosticTests(unittest.TestCase):
    def test_auto_prefers_available_mps(self) -> None:
        self.assertEqual(
            select_torch_device("auto", torch_module=_FakeTorchRuntime()),
            "mps",
        )

    def test_mps_memory_is_reported_when_available(self) -> None:
        diagnostics = foundation_runtime_diagnostics(
            torch_module=_FakeTorchRuntime()
        )
        self.assertTrue(diagnostics.torch_available)
        self.assertEqual(diagnostics.device, "mps")
        self.assertEqual(diagnostics.mps_current_allocated_bytes, 1_024)
        self.assertEqual(diagnostics.mps_driver_allocated_bytes, 2_048)
        self.assertEqual(diagnostics.mps_recommended_max_memory_bytes, 16_384)

    def test_missing_torch_is_a_diagnostic_not_an_import_failure(self) -> None:
        with mock.patch(
            "foldcrack_qc.foundation._import_torch",
            side_effect=ImportError("simulated missing torch"),
        ):
            diagnostics = foundation_runtime_diagnostics()
        self.assertFalse(diagnostics.torch_available)
        self.assertIn("simulated missing torch", diagnostics.error or "")


@unittest.skipUnless(
    importlib.util.find_spec("torch") is not None,
    "optional PyTorch dependency is not installed",
)
class TorchExtractorTests(unittest.TestCase):
    def test_hugging_face_style_fake_model_returns_cls_and_patch_grid(self) -> None:
        import torch

        class FakeHFModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.batch_sizes: list[int] = []

            def forward(self, *, pixel_values: object) -> object:
                assert torch.is_tensor(pixel_values)
                self.batch_sizes.append(int(pixel_values.shape[0]))
                patch = torch.nn.functional.adaptive_avg_pool2d(
                    pixel_values,
                    (16, 16),
                )
                patch = patch.flatten(2).transpose(1, 2) * self.scale
                cls = torch.mean(patch, dim=1, keepdim=True)
                return SimpleNamespace(
                    last_hidden_state=torch.cat((cls, patch), dim=1)
                )

        model = FakeHFModel()
        extractor = DINOv2FeatureExtractor(
            model,
            device="cpu",
            image_size=224,
            patch_size=14,
        )
        images = np.linspace(
            0.0,
            1.0,
            num=3 * 20 * 18 * 3,
            dtype=np.float32,
        ).reshape(3, 20, 18, 3)
        features = extractor.encode(
            images,
            semantic_channels=("DAPI", "panCK", "autofluorescence"),
            batch_size=2,
        )
        self.assertEqual(features.cls_embedding.shape, (3, 3))
        self.assertEqual(features.patch_grid.shape, (3, 16, 16, 3))
        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertFalse(model.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
