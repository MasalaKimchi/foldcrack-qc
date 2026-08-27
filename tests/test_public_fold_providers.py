from __future__ import annotations

import os
import subprocess
import sys
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from foldcrack_qc.public_fold_providers import (
    PUBLIC_FOLD_ENCODER_NAMES,
    PUBLIC_FOLD_ENCODER_PROVIDERS,
    BuiltPublicFoldProvider,
    build_public_fold_encoder,
)


def _arguments(**updates: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "allow_download": False,
        "cache_dir": Path("model-cache"),
        "device": "cpu",
        "hibou_source": Path("models/hibou/source"),
        "hibou_source_commit": "b" * 40,
        "hibou_weights": Path("models/hibou/model.pth"),
        "hibou_weights_sha256": "a" * 64,
        "model_id": "facebook/dinov2-small",
        "revision": "d" * 40,
        "siglip2_snapshot": Path("models/siglip2"),
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _encoder(device: str) -> SimpleNamespace:
    return SimpleNamespace(device=device, encode=Mock())


class PublicFoldProviderRegistryTests(unittest.TestCase):
    def test_registry_is_complete_ordered_and_immutable(self) -> None:
        self.assertEqual(
            PUBLIC_FOLD_ENCODER_NAMES,
            ("dinov2-hf", "hibou-b-local", "siglip2-base-local"),
        )
        self.assertEqual(
            tuple(PUBLIC_FOLD_ENCODER_PROVIDERS), PUBLIC_FOLD_ENCODER_NAMES
        )
        self.assertTrue(
            all(
                callable(spec.builder)
                for spec in PUBLIC_FOLD_ENCODER_PROVIDERS.values()
            )
        )
        self.assertTrue(
            PUBLIC_FOLD_ENCODER_PROVIDERS["dinov2-hf"].allows_dinov2_legacy_aliases
        )
        self.assertFalse(
            PUBLIC_FOLD_ENCODER_PROVIDERS["hibou-b-local"].allows_dinov2_legacy_aliases
        )
        self.assertFalse(
            PUBLIC_FOLD_ENCODER_PROVIDERS[
                "siglip2-base-local"
            ].allows_dinov2_legacy_aliases
        )

        with self.assertRaises(TypeError):
            PUBLIC_FOLD_ENCODER_PROVIDERS["unapproved"] = PUBLIC_FOLD_ENCODER_PROVIDERS[
                "dinov2-hf"
            ]

    def test_built_provider_is_frozen_and_copies_validated_identity(self) -> None:
        source = {
            "id": "example",
            "requested_device": "cpu",
            "resolved_device": "cpu",
            "input": {"image_size": [224, 224]},
        }
        encoder = _encoder("cpu")

        built = BuiltPublicFoldProvider(encoder, source)
        source["id"] = "mutated"
        source["input"]["image_size"][0] = 999

        self.assertIs(built.encoder, encoder)
        self.assertEqual(built.model_identity["id"], "example")
        self.assertEqual(built.model_identity["input"], {"image_size": [224, 224]})
        with self.assertRaises(TypeError):
            built.model_identity["new"] = "value"
        with self.assertRaises(FrozenInstanceError):
            built.encoder = _encoder("mps")

    def test_built_provider_rejects_malformed_encoder_or_identity(self) -> None:
        valid_identity = {
            "id": "example",
            "requested_device": "cpu",
            "resolved_device": "cpu",
        }
        with self.assertRaisesRegex(TypeError, "callable encode"):
            BuiltPublicFoldProvider(SimpleNamespace(device="cpu"), valid_identity)
        with self.assertRaisesRegex(ValueError, "must be non-empty"):
            BuiltPublicFoldProvider(_encoder("cpu"), {})
        for missing in ("requested_device", "resolved_device"):
            with self.subTest(missing=missing):
                identity = dict(valid_identity)
                del identity[missing]
                with self.assertRaisesRegex(ValueError, repr(missing)):
                    BuiltPublicFoldProvider(_encoder("cpu"), identity)

    def test_registry_import_does_not_load_model_implementations(self) -> None:
        root = Path(__file__).resolve().parents[1]
        environment = dict(os.environ)
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(root / "src")
            if not existing_pythonpath
            else os.pathsep.join((str(root / "src"), existing_pythonpath))
        )
        script = (
            "import sys; "
            "import foldcrack_qc.public_fold_providers; "
            "assert 'foldcrack_qc.foundation' not in sys.modules; "
            "assert 'foldcrack_qc.foundation_smoke' not in sys.modules"
        )

        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_unknown_provider_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown public-fold foundation"):
            build_public_fold_encoder(
                "unapproved",
                _arguments(),
                ("foundation_patchknn",),
            )

    def test_every_non_dino_provider_centrally_rejects_legacy_aliases(self) -> None:
        expected = {
            "hibou-b-local": (
                "Hibou-B must use encoder-agnostic foundation_patchknn and/or "
                "foundation_linear_probe, not DINOv2 aliases: "
                "['dinov2_linear_probe', 'dinov2_patchknn']"
            ),
            "siglip2-base-local": (
                "SigLIP2 Base must use encoder-agnostic foundation_patchknn and/or "
                "foundation_linear_probe, not DINOv2 aliases: "
                "['dinov2_linear_probe', 'dinov2_patchknn']"
            ),
        }
        non_dino = {
            name
            for name, spec in PUBLIC_FOLD_ENCODER_PROVIDERS.items()
            if not spec.allows_dinov2_legacy_aliases
        }
        self.assertEqual(non_dino, set(expected))

        for name, message in expected.items():
            with self.subTest(name=name), self.assertRaises(ValueError) as caught:
                build_public_fold_encoder(
                    name,
                    _arguments(allow_download=True),
                    ("dinov2_patchknn", "dinov2_linear_probe"),
                )
            self.assertEqual(str(caught.exception), message)

    def test_dinov2_builder_preserves_loader_and_identity_contract(self) -> None:
        weight_digest = SimpleNamespace(
            as_dict=lambda: {
                "filename": "model.safetensors",
                "sha256": "e" * 64,
                "size_bytes": 456,
            }
        )
        config_digest = SimpleNamespace(
            as_dict=lambda: {
                "filename": "config.json",
                "sha256": "f" * 64,
                "size_bytes": 123,
            }
        )
        loaded = SimpleNamespace(
            model=object(),
            resolved_revision="c" * 40,
            weight_digests=(weight_digest,),
            configuration_digests=(config_digest,),
        )
        encoder = _encoder("cpu")
        args = _arguments(allow_download=True)
        with (
            patch(
                "foldcrack_qc.foundation_smoke.load_huggingface_model",
                return_value=loaded,
            ) as loader,
            patch(
                "foldcrack_qc.foundation_smoke.dinov2_model_geometry",
                return_value=((14, 14), 1),
            ) as geometry,
            patch(
                "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                return_value=encoder,
            ) as extractor,
        ):
            built = build_public_fold_encoder(
                "dinov2-hf",
                args,
                ("dinov2_patchknn",),
            )

        self.assertIs(built.encoder, encoder)
        identity = dict(built.model_identity)
        model_config = loader.call_args.args[0]
        self.assertEqual(model_config.revision, "d" * 40)
        self.assertEqual(model_config.model_id, "facebook/dinov2-small")
        self.assertEqual(model_config.cache_dir, Path("model-cache"))
        self.assertTrue(model_config.allow_download)
        geometry.assert_called_once_with(loaded.model, 224)
        extractor.assert_called_once_with(
            loaded.model,
            device="cpu",
            image_size=224,
            patch_size=(14, 14),
            prefix_tokens=1,
            model_input_name="pixel_values",
        )
        self.assertEqual(
            identity,
            {
                "id": "facebook/dinov2-small",
                "requested_revision": "d" * 40,
                "resolved_revision": "c" * 40,
                "weight_files": [weight_digest.as_dict()],
                "configuration_files": [config_digest.as_dict()],
                "requested_device": "cpu",
                "resolved_device": "cpu",
                "trust_remote_code": False,
                "token_used": False,
                "network_access_allowed": True,
                "input": {
                    "normalization": "ImageNet",
                    "image_size": [224, 224],
                    "patch_size": [14, 14],
                    "prefix_tokens": 1,
                },
            },
        )

    def test_hibou_builder_preserves_validation_loader_and_identity(self) -> None:
        provenance = {
            "id": "HistAI/Hibou-B",
            "weights": {"sha256": "a" * 64},
            "source": {"commit": "b" * 40},
            "trust_remote_code": False,
            "network_access_allowed": False,
        }
        local = SimpleNamespace(model=object(), provenance=provenance)
        encoder = _encoder("mps")
        args = _arguments(device="mps")
        with (
            patch(
                "foldcrack_qc.foundation.load_local_hibou_b",
                return_value=local,
            ) as loader,
            patch(
                "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                return_value=encoder,
            ) as extractor,
        ):
            built = build_public_fold_encoder(
                "hibou-b-local",
                args,
                ("foundation_linear_probe",),
            )

        self.assertIs(built.encoder, encoder)
        identity = dict(built.model_identity)
        loader.assert_called_once_with(
            Path("models/hibou/model.pth"),
            Path("models/hibou/source"),
            expected_weights_sha256="a" * 64,
            expected_source_commit="b" * 40,
        )
        extractor.assert_called_once_with(
            local.model,
            device="mps",
            image_size=224,
            patch_size=14,
            prefix_tokens=5,
            model_input_name=None,
            normalization_mean=(0.7068, 0.5755, 0.722),
            normalization_std=(0.195, 0.2316, 0.1816),
        )
        self.assertEqual(
            identity,
            {
                **provenance,
                "requested_device": "mps",
                "resolved_device": "mps",
                "output_contract": {
                    "type": "mapping",
                    "cls_key": "x_norm_clstoken",
                    "patch_key": "x_norm_patchtokens",
                    "prefix_tokens": 5,
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "not DINOv2 aliases"):
            build_public_fold_encoder(
                "hibou-b-local",
                _arguments(allow_download=True, hibou_weights=None),
                ("dinov2_patchknn",),
            )

    def test_siglip2_builder_preserves_validation_loader_and_identity(self) -> None:
        provenance = {
            "id": "google/siglip2-base-patch16-224",
            "assets": {"model.safetensors": {"sha256": "b" * 64}},
            "trust_remote_code": False,
            "network_access_allowed": False,
        }
        local = SimpleNamespace(
            model=object(),
            preprocessor=object(),
            provenance=provenance,
        )
        encoder = _encoder("mps")
        args = _arguments(device="mps")
        with (
            patch(
                "foldcrack_qc.foundation.load_local_siglip2_base_vision",
                return_value=local,
            ) as loader,
            patch(
                "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                return_value=encoder,
            ) as extractor,
        ):
            built = build_public_fold_encoder(
                "siglip2-base-local",
                args,
                ("foundation_patchknn",),
            )

        self.assertIs(built.encoder, encoder)
        identity = dict(built.model_identity)
        loader.assert_called_once_with(Path("models/siglip2"))
        extractor.assert_called_once_with(
            local.model,
            device="mps",
            image_size=224,
            patch_size=16,
            prefix_tokens=0,
            model_input_name="pixel_values",
            global_embedding_name="pooler_output",
            normalization_mean=(0.5, 0.5, 0.5),
            normalization_std=(0.5, 0.5, 0.5),
            preprocessor=local.preprocessor,
        )
        self.assertEqual(
            identity,
            {
                **provenance,
                "requested_device": "mps",
                "resolved_device": "mps",
                "output_contract": {
                    "type": "object",
                    "global_key": "pooler_output",
                    "patch_key": "last_hidden_state",
                    "prefix_tokens": 0,
                },
            },
        )

        with self.assertRaisesRegex(ValueError, "local-only SigLIP2 Base loader"):
            build_public_fold_encoder(
                "siglip2-base-local",
                _arguments(allow_download=True, siglip2_snapshot=None),
                ("foundation_patchknn",),
            )


if __name__ == "__main__":
    unittest.main()
