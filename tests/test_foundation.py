from __future__ import annotations

import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from foldcrack_qc.foundation import (
    HIBOU_B_MEAN,
    HIBOU_B_STD,
    SIGLIP2_BASE_MEAN,
    SIGLIP2_BASE_STD,
    DINOv2FeatureExtractor,
    FoundationFeatures,
    PatchKNNAnomalyScorer,
    foundation_runtime_diagnostics,
    load_local_hibou_b,
    load_local_siglip2_base_vision,
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

    def test_preprocessing_accepts_published_pathology_normalization(self) -> None:
        image = np.broadcast_to(
            np.asarray(HIBOU_B_MEAN, dtype=np.float32), (9, 11, 3)
        ).copy()
        normalized = preprocess_dinov2_rgb(
            image,
            image_size=(8, 10),
            normalization_mean=HIBOU_B_MEAN,
            normalization_std=HIBOU_B_STD,
        )
        self.assertEqual(normalized.shape, (1, 3, 8, 10))
        np.testing.assert_allclose(normalized, 0.0, atol=2e-6)

    def test_preprocessing_rejects_invalid_normalization(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "three values"):
            preprocess_dinov2_rgb(
                image,
                normalization_mean=(0.5, 0.5),
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            preprocess_dinov2_rgb(
                image,
                normalization_std=(1.0, 0.0, 1.0),
            )

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
        diagnostics = foundation_runtime_diagnostics(torch_module=_FakeTorchRuntime())
        self.assertTrue(diagnostics.torch_available)
        self.assertEqual(diagnostics.device, "mps")
        self.assertEqual(diagnostics.mps_current_allocated_bytes, 1_024)
        self.assertEqual(diagnostics.mps_driver_allocated_bytes, 2_048)
        self.assertEqual(diagnostics.mps_recommended_max_memory_bytes, 16_384)
        self.assertEqual(
            set(diagnostics.as_dict()),
            {
                "torch_available",
                "torch_version",
                "device",
                "mps_built",
                "mps_available",
                "mps_current_allocated_bytes",
                "mps_driver_allocated_bytes",
                "mps_recommended_max_memory_bytes",
                "error",
            },
        )

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
                return SimpleNamespace(last_hidden_state=torch.cat((cls, patch), dim=1))

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
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.parameters())
        )

    def test_positional_model_prefers_mapping_style_forward_features(self) -> None:
        import torch

        class FakeHibou(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
                self.forward_features_called = False

            def forward(self, tensor: object) -> object:
                raise AssertionError("CLS-only forward must not be used")

            def forward_features(self, tensor: object) -> object:
                assert torch.is_tensor(tensor)
                self.forward_features_called = True
                patch = torch.nn.functional.adaptive_avg_pool2d(tensor, (16, 16))
                patch = patch.flatten(2).transpose(1, 2) * self.scale
                return {
                    "x_norm_clstoken": patch.mean(dim=1),
                    "x_norm_regtokens": patch[:, :4],
                    "x_norm_patchtokens": patch,
                }

        model = FakeHibou()
        extractor = DINOv2FeatureExtractor(
            model,
            device="cpu",
            image_size=224,
            patch_size=14,
            prefix_tokens=5,
            model_input_name=None,
            normalization_mean=HIBOU_B_MEAN,
            normalization_std=HIBOU_B_STD,
        )
        features = extractor.encode(
            np.full((1, 20, 20, 3), 128, dtype=np.uint8),
            batch_size=1,
        )
        self.assertTrue(model.forward_features_called)
        self.assertEqual(features.patch_grid.shape, (1, 16, 16, 3))
        np.testing.assert_allclose(extractor.normalization_mean, HIBOU_B_MEAN)
        np.testing.assert_allclose(extractor.normalization_std, HIBOU_B_STD)

    def test_zero_prefix_model_uses_explicit_pooler_and_all_patch_tokens(self) -> None:
        import torch

        class FakeSiglipVision(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))

            def forward(self, *, pixel_values: object) -> object:
                assert torch.is_tensor(pixel_values)
                patch = torch.nn.functional.adaptive_avg_pool2d(pixel_values, (14, 14))
                patch = patch.flatten(2).transpose(1, 2) * self.scale
                return SimpleNamespace(
                    last_hidden_state=patch,
                    pooler_output=patch.mean(dim=1),
                )

        extractor = DINOv2FeatureExtractor(
            FakeSiglipVision(),
            device="cpu",
            image_size=224,
            patch_size=16,
            prefix_tokens=0,
            global_embedding_name="pooler_output",
            normalization_mean=SIGLIP2_BASE_MEAN,
            normalization_std=SIGLIP2_BASE_STD,
        )
        features = extractor.encode(
            np.full((2, 20, 20, 3), 128, dtype=np.uint8),
            batch_size=2,
        )
        self.assertEqual(features.cls_embedding.shape, (2, 3))
        self.assertEqual(features.patch_grid.shape, (2, 14, 14, 3))
        np.testing.assert_allclose(
            features.cls_embedding,
            features.patch_grid.mean(axis=(1, 2)),
            atol=1e-6,
        )

    def test_zero_prefix_model_requires_named_global_embedding(self) -> None:
        import torch

        with self.assertRaisesRegex(ValueError, "global_embedding_name"):
            DINOv2FeatureExtractor(
                torch.nn.Identity(),
                device="cpu",
                image_size=224,
                patch_size=16,
                prefix_tokens=0,
                model_input_name=None,
            )

    def test_zero_prefix_model_rejects_missing_named_global_output(self) -> None:
        import torch

        class MissingPooler(torch.nn.Module):
            def forward(self, *, pixel_values: object) -> object:
                assert torch.is_tensor(pixel_values)
                return SimpleNamespace(
                    last_hidden_state=torch.zeros(
                        (pixel_values.shape[0], 196, 4),
                        device=pixel_values.device,
                    )
                )

        extractor = DINOv2FeatureExtractor(
            MissingPooler(),
            device="cpu",
            image_size=224,
            patch_size=16,
            prefix_tokens=0,
            global_embedding_name="pooler_output",
        )
        with self.assertRaisesRegex(TypeError, "configured global embedding"):
            extractor.encode(np.zeros((1, 16, 16, 3), dtype=np.float32))


class LocalSiglip2LoaderTests(unittest.TestCase):
    @staticmethod
    def _assets(root: Path) -> dict[str, str]:
        files = {
            "model.safetensors": b"locked siglip2 test weights",
            "config.json": (
                b'{"model_type":"siglip","vision_config":'
                b'{"model_type":"siglip_vision_model"}}'
            ),
            "preprocessor_config.json": (
                b'{"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],'
                b'"size":{"height":224,"width":224},"resample":2,'
                b'"do_resize":true,"do_rescale":true,"do_normalize":true,'
                b'"rescale_factor":0.00392156862745098}'
            ),
            "README.md": b"---\nlicense: apache-2.0\n---\n",
        }
        for name, content in files.items():
            (root / name).write_bytes(content)
        return {
            name: hashlib.sha256(content).hexdigest() for name, content in files.items()
        }

    def test_local_loader_locks_assets_license_geometry_and_network_policy(
        self,
    ) -> None:
        fake_model = SimpleNamespace(
            config=SimpleNamespace(image_size=224, patch_size=16, hidden_size=768)
        )
        fake_class = SimpleNamespace(from_pretrained=mock.Mock(return_value=fake_model))
        fake_processor = mock.Mock(
            return_value={"pixel_values": np.zeros((1, 3, 224, 224), dtype=np.float32)}
        )
        fake_processor_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=fake_processor)
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digests = self._assets(root)
            with (
                mock.patch.dict(
                    "foldcrack_qc.foundation._SIGLIP2_BASE_ASSET_SHA256",
                    digests,
                    clear=True,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._import_siglip_vision_model",
                    return_value=fake_class,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._import_siglip_image_processor",
                    return_value=fake_processor_class,
                ),
            ):
                loaded = load_local_siglip2_base_vision(root)

        self.assertIs(loaded.model, fake_model)
        fake_class.from_pretrained.assert_called_once_with(
            str(root.resolve()),
            local_files_only=True,
            trust_remote_code=False,
            token=False,
            use_safetensors=True,
        )
        fake_processor_class.from_pretrained.assert_called_once_with(
            str(root.resolve()),
            local_files_only=True,
            token=False,
        )
        self.assertIsNotNone(loaded.preprocessor)
        assert loaded.preprocessor is not None
        output = loaded.preprocessor(
            np.zeros((1, 8, 8, 3), dtype=np.uint8),
            semantic_channels=("red", "green", "blue"),
        )
        self.assertEqual(output.shape, (1, 3, 224, 224))
        fake_processor.assert_called_once()
        self.assertEqual(
            loaded.provenance["source"]["revision"],
            "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2",
        )
        self.assertEqual(loaded.provenance["license"]["spdx"], "Apache-2.0")
        self.assertEqual(loaded.provenance["output"]["prefix_tokens"], 0)
        self.assertFalse(loaded.provenance["trust_remote_code"])
        self.assertFalse(loaded.provenance["token_used"])
        self.assertFalse(loaded.provenance["network_access_allowed"])

    def test_local_loader_rejects_tampered_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            digests = self._assets(root)
            (root / "model.safetensors").write_bytes(b"tampered")
            with (
                mock.patch.dict(
                    "foldcrack_qc.foundation._SIGLIP2_BASE_ASSET_SHA256",
                    digests,
                    clear=True,
                ),
                self.assertRaisesRegex(ValueError, "model.safetensors"),
            ):
                load_local_siglip2_base_vision(root)


class LocalHibouLoaderTests(unittest.TestCase):
    @staticmethod
    def _assets(root: Path) -> tuple[Path, Path]:
        weights = root / "hibou-b.pth"
        weights.write_bytes(b"locked test weights")
        source = root / "source"
        (source / "hibou" / "models").mkdir(parents=True)
        (source / "hibou" / "__init__.py").write_text("", encoding="utf-8")
        (source / "hibou" / "models" / "__init__.py").write_text(
            "# official build module fixture\n", encoding="utf-8"
        )
        (source / "hibou" / "models" / "vision_transformer.py").write_text(
            "# official vision fixture\n", encoding="utf-8"
        )
        (source / "README.md").write_text("Hibou-B", encoding="utf-8")
        (source / "LICENSE").write_text(
            "Apache License\nVersion 2.0\n", encoding="utf-8"
        )
        return weights, source

    @staticmethod
    def _git_output(source: Path, *arguments: str) -> str:
        del source
        if arguments == ("rev-parse", "HEAD"):
            return "a" * 40
        if arguments == ("status", "--porcelain", "--untracked-files=all"):
            return ""
        if arguments == ("config", "--get", "remote.origin.url"):
            return "https://github.com/HistAI/hibou.git"
        raise AssertionError(arguments)

    @staticmethod
    def _approved_release(weights: Path, source: Path) -> dict[str, object]:
        return {
            "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
            "source_sha256": {
                relative: hashlib.sha256((source / relative).read_bytes()).hexdigest()
                for relative in (
                    "hibou/models/__init__.py",
                    "hibou/models/vision_transformer.py",
                    "README.md",
                    "LICENSE",
                )
            },
        }

    def test_local_loader_records_hash_commit_license_and_geometry(self) -> None:
        class FakeModel:
            patch_size = 14
            num_register_tokens = 4

            def forward_features(self, tensor: object) -> dict[str, object]:
                return {"tensor": tensor}

        with tempfile.TemporaryDirectory() as temporary:
            weights, source = self._assets(Path(temporary))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            builder = mock.Mock(return_value=FakeModel())
            with (
                mock.patch.dict(
                    "foldcrack_qc.foundation._HIBOU_B_APPROVED_RELEASES",
                    {"a" * 40: self._approved_release(weights, source)},
                    clear=True,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._git_output",
                    side_effect=self._git_output,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._import_hibou_build_model",
                    return_value=builder,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._load_strict_torch_state_dict",
                ) as strict_loader,
            ):
                loaded = load_local_hibou_b(
                    weights,
                    source,
                    expected_weights_sha256=digest,
                    expected_source_commit="a" * 40,
                )

        self.assertIsInstance(loaded.model, FakeModel)
        builder.assert_called_once_with(
            None,
            img_size=224,
            arch="vit_base",
            patch_size=14,
            num_register_tokens=4,
        )
        strict_loader.assert_called_once_with(loaded.model, weights.resolve())
        self.assertEqual(loaded.provenance["weights"]["sha256"], digest)
        self.assertEqual(loaded.provenance["source"]["commit"], "a" * 40)
        self.assertEqual(loaded.provenance["license"]["spdx"], "Apache-2.0")
        self.assertFalse(loaded.provenance["trust_remote_code"])
        self.assertFalse(loaded.provenance["network_access_allowed"])

    def test_approved_release_locks_full_imported_python_closure(self) -> None:
        from foldcrack_qc import foundation as module

        release = module._HIBOU_B_APPROVED_RELEASES[
            "c453bbe4dab0fec6f7df343b09ea87048629c58d"
        ]
        source_hashes = release["source_sha256"]
        self.assertIsInstance(source_hashes, dict)
        self.assertEqual(
            {name for name in source_hashes if str(name).endswith(".py")},
            {
                "hibou/models/__init__.py",
                "hibou/models/vision_transformer.py",
                "hibou/models/layers/__init__.py",
                "hibou/models/layers/attention.py",
                "hibou/models/layers/block.py",
                "hibou/models/layers/dino_head.py",
                "hibou/models/layers/drop_path.py",
                "hibou/models/layers/layer_scale.py",
                "hibou/models/layers/mlp.py",
                "hibou/models/layers/patch_embed.py",
                "hibou/models/layers/swiglu_ffn.py",
            },
        )

    def test_local_loader_rejects_untracked_source_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            weights, source = self._assets(Path(temporary))
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()

            def dirty_git_output(path: Path, *arguments: str) -> str:
                if arguments == ("status", "--porcelain", "--untracked-files=all"):
                    return "?? hibou/models/layers/payload.py"
                return self._git_output(path, *arguments)

            with (
                mock.patch.dict(
                    "foldcrack_qc.foundation._HIBOU_B_APPROVED_RELEASES",
                    {"a" * 40: self._approved_release(weights, source)},
                    clear=True,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._git_output",
                    side_effect=dirty_git_output,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._import_hibou_build_model"
                ) as importer,
                self.assertRaisesRegex(ValueError, "tracked or untracked"),
            ):
                load_local_hibou_b(
                    weights,
                    source,
                    expected_weights_sha256=digest,
                    expected_source_commit="a" * 40,
                )
        importer.assert_not_called()

    def test_local_loader_rejects_hash_and_geometry_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            weights, source = self._assets(Path(temporary))
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_local_hibou_b(
                    weights,
                    source,
                    expected_weights_sha256="0" * 64,
                    expected_source_commit="a" * 40,
                )

            bad_model = SimpleNamespace(
                patch_size=16,
                num_register_tokens=4,
                forward_features=lambda tensor: tensor,
            )
            with (
                mock.patch.dict(
                    "foldcrack_qc.foundation._HIBOU_B_APPROVED_RELEASES",
                    {"a" * 40: self._approved_release(weights, source)},
                    clear=True,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._git_output",
                    side_effect=self._git_output,
                ),
                mock.patch(
                    "foldcrack_qc.foundation._import_hibou_build_model",
                    return_value=mock.Mock(return_value=bad_model),
                ),
                mock.patch(
                    "foldcrack_qc.foundation._load_strict_torch_state_dict",
                ),
                self.assertRaisesRegex(ValueError, "patch14"),
            ):
                load_local_hibou_b(
                    weights,
                    source,
                    expected_weights_sha256=hashlib.sha256(
                        weights.read_bytes()
                    ).hexdigest(),
                    expected_source_commit="a" * 40,
                )

    def test_local_loader_rejects_missing_locks_before_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            weights, source = self._assets(Path(temporary))
            with (
                mock.patch(
                    "foldcrack_qc.foundation._import_hibou_build_model"
                ) as importer,
                self.assertRaisesRegex(ValueError, "requires both an approved"),
            ):
                load_local_hibou_b(weights, source)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
