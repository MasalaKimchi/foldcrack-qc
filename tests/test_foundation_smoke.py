from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from foldcrack_qc.foundation_smoke import (
    FoundationSmokeConfig,
    LoadedFoundationModel,
    WeightDigest,
    _device_agreement_gate,
    deterministic_smoke_patches,
    load_huggingface_model,
    run_foundation_smoke,
)

REVISION = "1234567890abcdef1234567890abcdef12345678"


class ConfigAndInputTests(unittest.TestCase):
    def test_defaults_are_offline_and_revision_is_immutable(self) -> None:
        config = FoundationSmokeConfig(revision=REVISION)
        self.assertFalse(config.allow_download)
        self.assertTrue(config.as_dict()["offline"])
        with self.assertRaisesRegex(ValueError, "exact 40-character"):
            FoundationSmokeConfig(revision="main")

    def test_lora_contract_accepts_only_rank_four_or_eight(self) -> None:
        for rank in (4, 8):
            self.assertEqual(
                FoundationSmokeConfig(revision=REVISION, lora_rank=rank).lora_rank,
                rank,
            )
        with self.assertRaisesRegex(ValueError, "None, 4, or 8"):
            FoundationSmokeConfig(revision=REVISION, lora_rank=16)

    def test_device_agreement_is_a_real_pass_fail_gate(self) -> None:
        config = FoundationSmokeConfig(revision=REVISION)
        passing = {
            "max_abs_error": 5e-4,
            "cls_cosine_similarity": 0.99999,
            "spatial_cosine_similarity": 0.99995,
        }
        failing = {**passing, "max_abs_error": 0.01}
        self.assertTrue(_device_agreement_gate(passing, config)["passed"])
        self.assertFalse(_device_agreement_gate(failing, config)["passed"])
        with self.assertRaisesRegex(ValueError, "approved smoke allowlist"):
            FoundationSmokeConfig(revision=REVISION, model_id="unknown/model")

    def test_two_patch_input_is_exactly_reproducible_and_typed(self) -> None:
        first = deterministic_smoke_patches(56)
        second = deterministic_smoke_patches(56)
        self.assertEqual(first.shape, (2, 56, 56, 3))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertFalse(np.array_equal(first[0], first[1]))
        np.testing.assert_array_equal(first, second)


class _FakeHub:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = snapshot
        self.calls: list[dict[str, object]] = []

    def snapshot_download(self, **kwargs: object) -> str:
        self.calls.append(dict(kwargs))
        return str(self.snapshot)


class _FakeAutoModel:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []
    model = object()

    @classmethod
    def from_pretrained(cls, model_id: str, **kwargs: object) -> object:
        cls.calls.append((model_id, dict(kwargs)))
        return cls.model


class LoadingPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeAutoModel.calls = []

    def _snapshot(self, root: Path) -> tuple[Path, bytes]:
        snapshot = root / REVISION
        snapshot.mkdir(parents=True)
        contents = b"deterministic fake safetensors payload"
        (snapshot / "model.safetensors").write_bytes(contents)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        return snapshot, contents

    def test_offline_and_download_modes_never_use_token_or_remote_code(self) -> None:
        for allow_download in (False, True):
            with (
                self.subTest(allow_download=allow_download),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                snapshot, contents = self._snapshot(root)
                hub = _FakeHub(snapshot)
                config = FoundationSmokeConfig(
                    revision=REVISION,
                    cache_dir=root / "cache",
                    allow_download=allow_download,
                )
                loaded = load_huggingface_model(
                    config,
                    transformers_module=SimpleNamespace(AutoModel=_FakeAutoModel),
                    hub_module=hub,
                )

                self.assertEqual(hub.calls[0]["token"], False)
                self.assertEqual(
                    hub.calls[0]["local_files_only"],
                    not allow_download,
                )
                model_id, model_kwargs = _FakeAutoModel.calls[-1]
                self.assertEqual(model_id, "facebook/dinov2-small")
                self.assertEqual(model_kwargs["revision"], REVISION)
                self.assertEqual(model_kwargs["token"], False)
                self.assertEqual(model_kwargs["trust_remote_code"], False)
                self.assertEqual(model_kwargs["local_files_only"], True)
                self.assertEqual(model_kwargs["cache_dir"], str(root / "cache"))
                self.assertEqual(loaded.resolved_revision, REVISION)
                self.assertEqual(
                    loaded.weight_digests[0].sha256,
                    hashlib.sha256(contents).hexdigest(),
                )
                self.assertEqual(
                    loaded.configuration_digests[0].filename, "config.json"
                )

    def test_snapshot_revision_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            mismatch = root / ("a" * 40)
            mismatch.mkdir()
            (mismatch / "model.safetensors").write_bytes(b"weights")
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                load_huggingface_model(
                    FoundationSmokeConfig(revision=REVISION, cache_dir=root / "cache"),
                    transformers_module=SimpleNamespace(AutoModel=_FakeAutoModel),
                    hub_module=_FakeHub(mismatch),
                )

    def test_snapshot_without_weights_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            snapshot = root / REVISION
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "No model weight files"):
                load_huggingface_model(
                    FoundationSmokeConfig(revision=REVISION, cache_dir=root / "cache"),
                    transformers_module=SimpleNamespace(AutoModel=_FakeAutoModel),
                    hub_module=_FakeHub(snapshot),
                )


class ReportBoundaryTests(unittest.TestCase):
    def test_report_cannot_claim_scientific_validation(self) -> None:
        contents = b"fake-weights"
        digest = WeightDigest(
            filename="model.safetensors",
            sha256=hashlib.sha256(contents).hexdigest(),
            size_bytes=len(contents),
        )
        loaded = LoadedFoundationModel(
            model=object(),
            resolved_revision=REVISION,
            weight_digests=(digest,),
            snapshot_path=Path("cache") / REVISION,
        )
        loader_calls: list[FoundationSmokeConfig] = []

        def fake_loader(config: FoundationSmokeConfig) -> LoadedFoundationModel:
            loader_calls.append(config)
            return loaded

        def fake_executor(
            config: FoundationSmokeConfig,
            fake_loaded: LoadedFoundationModel,
        ) -> dict[str, object]:
            self.assertIs(fake_loaded, loaded)
            return {
                "status": "passed",
                "resolved_device": "mps",
                "scientific_validation_passed": True,
                "lora": {"requested": False, "performed": False},
            }

        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "foundation-smoke.json"
            config = FoundationSmokeConfig(
                revision=REVISION,
                cache_dir=Path(temp) / "cache",
                allow_download=False,
            )
            report = run_foundation_smoke(
                config,
                output_json=output,
                model_loader=fake_loader,
                executor=fake_executor,
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(loader_calls, [config])
        self.assertEqual(report["result_type"], "engineering_foundation_smoke_only")
        self.assertIs(report["scientific_validation_passed"], False)
        self.assertIs(report["policy"]["offline"], True)
        self.assertEqual(report["model"]["requested_revision"], REVISION)
        self.assertEqual(report["model"]["resolved_revision"], REVISION)
        self.assertEqual(report["model"]["weight_files"][0]["sha256"], digest.sha256)
        self.assertEqual(persisted, report)
        self.assertIn("real WSI", " ".join(report["limitations"]))


if __name__ == "__main__":
    unittest.main()
