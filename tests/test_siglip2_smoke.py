from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from foldcrack_qc.foundation import FoundationFeatures
from foldcrack_qc.siglip2_smoke import (
    SIGLIP2_BASE_ASSET_SHA256,
    SIGLIP2_BASE_MODEL_ID,
    SIGLIP2_BASE_REVISION,
    SigLIP2SmokeConfig,
    _agreement_gate,
    _execute_siglip2_smoke,
    _feature_agreement,
    _last_four_query_value_modules,
    _parameter_delta_records,
    _validate_locked_provenance,
    deterministic_semantic_rgb_patches,
    main,
    run_siglip2_smoke,
)


def _locked_provenance(snapshot: Path) -> dict[str, object]:
    return {
        "id": SIGLIP2_BASE_MODEL_ID,
        "source": {
            "repository": f"https://huggingface.co/{SIGLIP2_BASE_MODEL_ID}",
            "revision": SIGLIP2_BASE_REVISION,
        },
        "assets": {
            name: {
                "path": str(snapshot / name),
                "sha256": digest,
                "size_bytes": 1,
            }
            for name, digest in SIGLIP2_BASE_ASSET_SHA256.items()
        },
        "license": {"spdx": "Apache-2.0"},
        "input": {
            "processor": "transformers.SiglipImageProcessor",
            "resample": 2,
            "resample_semantics": "PIL.Image.Resampling.BILINEAR",
            "rescale_factor": 1.0 / 255.0,
            "source_dtype_boundary": "round_clipped_unit_RGB_to_uint8",
            "normalization_mean": [0.5, 0.5, 0.5],
            "normalization_std": [0.5, 0.5, 0.5],
        },
        "trust_remote_code": False,
        "token_used": False,
        "network_access_allowed": False,
    }


class ConfigAndInputTests(unittest.TestCase):
    def test_configuration_is_offline_and_restricts_device_and_rank(self) -> None:
        config = SigLIP2SmokeConfig("snapshot", device="mps", lora_rank=4)
        self.assertTrue(config.as_dict()["offline"])
        self.assertEqual(config.lora_rank, 4)
        with self.assertRaisesRegex(ValueError, "cpu.*mps"):
            SigLIP2SmokeConfig("snapshot", device="auto")
        with self.assertRaisesRegex(ValueError, "None, 4, or 8"):
            SigLIP2SmokeConfig("snapshot", lora_rank=16)

    def test_semantic_rgb_patches_are_exactly_reproducible(self) -> None:
        first = deterministic_semantic_rgb_patches(56)
        second = deterministic_semantic_rgb_patches(56)
        self.assertEqual(first.shape, (2, 56, 56, 3))
        self.assertEqual(first.dtype, np.float32)
        self.assertTrue(np.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertFalse(np.array_equal(first[0], first[1]))
        np.testing.assert_array_equal(first, second)


class ProvenanceAndSelectionTests(unittest.TestCase):
    def test_exact_provenance_lock_accepts_only_expected_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            provenance = _locked_provenance(snapshot)
            accepted = _validate_locked_provenance(provenance, snapshot)
            self.assertEqual(accepted["id"], SIGLIP2_BASE_MODEL_ID)

            tampered = json.loads(json.dumps(provenance))
            tampered["assets"]["model.safetensors"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "model.safetensors"):
                _validate_locked_provenance(tampered, snapshot)

            online = {**provenance, "network_access_allowed": True}
            with self.assertRaisesRegex(RuntimeError, "offline"):
                _validate_locked_provenance(online, snapshot)

            wrong_processor = json.loads(json.dumps(provenance))
            wrong_processor["input"]["resample"] = 3
            with self.assertRaisesRegex(RuntimeError, "processor contract"):
                _validate_locked_provenance(wrong_processor, snapshot)

    def test_lora_targets_q_and_v_in_exactly_last_four_blocks(self) -> None:
        modules = []
        for block in range(12):
            for projection in ("q_proj", "k_proj", "v_proj"):
                name = f"encoder.layers.{block}.self_attn.{projection}"
                modules.append((name, SimpleNamespace(in_features=8, out_features=8)))
        model = SimpleNamespace(named_modules=lambda: iter(modules))
        targets, blocks = _last_four_query_value_modules(model)
        self.assertEqual(blocks, [8, 9, 10, 11])
        self.assertEqual(len(targets), 8)
        self.assertTrue(all("k_proj" not in target for target in targets))

        incomplete = SimpleNamespace(
            named_modules=lambda: iter(
                (name, module)
                for name, module in modules
                if not name.endswith("v_proj")
            )
        )
        with self.assertRaisesRegex(RuntimeError, "query and one value"):
            _last_four_query_value_modules(incomplete)


class AgreementAndDeltaTests(unittest.TestCase):
    @staticmethod
    def _features(offset: float = 0.0) -> FoundationFeatures:
        return FoundationFeatures(
            cls_embedding=np.full((2, 3), 1.0 + offset, dtype=np.float32),
            patch_grid=np.full((2, 2, 2, 3), 2.0 + offset, dtype=np.float32),
            input_size=(32, 32),
            patch_size=(16, 16),
            semantic_channels=("red", "green", "blue"),
        )

    def test_device_agreement_is_a_fail_closed_gate(self) -> None:
        config = SigLIP2SmokeConfig("snapshot")
        agreement = _feature_agreement(self._features(), self._features(1e-4))
        self.assertTrue(_agreement_gate(agreement, config)["passed"])
        failed = {**agreement, "max_abs_error": 0.1}
        self.assertFalse(_agreement_gate(failed, config)["passed"])

    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch") is not None,
        "PyTorch is optional",
    )
    def test_parameter_delta_summary_rejects_no_update(self) -> None:
        import torch

        before = {"adapter": torch.zeros(4)}
        with self.assertRaisesRegex(RuntimeError, "did not update"):
            _parameter_delta_records(before, before, torch)
        records, summary = _parameter_delta_records(
            before, {"adapter": torch.ones(4)}, torch
        )
        self.assertEqual(records[0]["changed_element_count"], 4)
        self.assertTrue(summary["nonzero_update"])


class ReportAndFailureTests(unittest.TestCase):
    @unittest.skipUnless(
        __import__("importlib").util.find_spec("torch") is not None,
        "PyTorch is optional",
    )
    def test_light_model_exercises_frozen_dense_and_global_processor_path(
        self,
    ) -> None:
        import torch

        class TinySigLIP(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.patch_projection = torch.nn.Conv2d(
                    3, 8, kernel_size=16, stride=16, bias=False
                )

            def forward(self, *, pixel_values: object) -> object:
                dense = self.patch_projection(pixel_values)
                tokens = dense.flatten(2).transpose(1, 2)
                return SimpleNamespace(
                    last_hidden_state=tokens,
                    pooler_output=tokens.mean(dim=1),
                )

        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary)
            calls: list[tuple[str, str, str]] = []

            def locked_processor(
                images: object, *, semantic_channels: tuple[str, str, str]
            ) -> np.ndarray:
                calls.append(semantic_channels)
                array = np.asarray(images, dtype=np.float32)
                return np.ascontiguousarray(
                    np.moveaxis((array - 0.5) / 0.5, -1, 1),
                    dtype=np.float32,
                )

            loaded = SimpleNamespace(
                model=TinySigLIP(),
                provenance=_locked_provenance(snapshot),
                preprocessor=locked_processor,
            )
            execution = _execute_siglip2_smoke(
                SigLIP2SmokeConfig(snapshot, device="cpu", steady_runs=1),
                loaded,
            )

        frozen = execution["frozen_inference"]
        self.assertEqual(
            frozen["cpu_reference"]["outputs"]["global_pooler_output"]["shape"],
            [2, 8],
        )
        self.assertEqual(
            frozen["cpu_reference"]["outputs"]["dense_last_hidden_state"]["shape"],
            [2, 14, 14, 8],
        )
        self.assertTrue(frozen["cpu_device_agreement_gate"]["passed"])
        self.assertEqual(
            execution["model_parameter_stats_frozen"]["trainable_parameter_count"], 0
        )
        self.assertGreaterEqual(len(calls), 6)
        self.assertTrue(all(call == ("red", "green", "blue") for call in calls))

    def test_fake_execution_persists_engineering_only_report_atomically(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            output = root / "artifact" / "siglip2.json"
            loaded = SimpleNamespace(
                model=object(),
                provenance=_locked_provenance(snapshot),
                preprocessor=lambda images, **_kwargs: images,
            )

            def fake_executor(
                config: SigLIP2SmokeConfig, fake_loaded: object
            ) -> dict[str, object]:
                self.assertEqual(config.snapshot_path, snapshot)
                self.assertIs(fake_loaded, loaded)
                return {
                    "status": "passed",
                    "engineering_smoke_test_passed": True,
                    "scientific_validation_passed": True,
                    "resolved_device": "mps",
                    "lora": {"requested": True, "performed": True},
                }

            report = run_siglip2_smoke(
                SigLIP2SmokeConfig(snapshot, lora_rank=4),
                output_json=output,
                model_loader=lambda _path: loaded,
                executor=fake_executor,
            )
            persisted = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(persisted, report)
            self.assertEqual(report["result_type"], "engineering_siglip2_smoke_only")
            self.assertIs(report["scientific_validation_passed"], False)
            self.assertTrue(report["policy"]["offline"])
            self.assertEqual(
                report["model"]["source"]["revision"], SIGLIP2_BASE_REVISION
            )
            self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))

    def test_missing_snapshot_fails_without_writing_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "should-not-exist.json"
            errors = StringIO()
            with redirect_stderr(errors):
                status = main(
                    [
                        "--snapshot-path",
                        str(Path(temporary) / "missing"),
                        "--device",
                        "mps",
                        "--output-json",
                        str(output),
                    ]
                )
            self.assertEqual(status, 2)
            self.assertIn("existing local directory", errors.getvalue())
            self.assertFalse(output.exists())

    def test_failed_execution_is_not_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            output = root / "failed.json"
            loaded = SimpleNamespace(
                model=object(),
                provenance=_locked_provenance(snapshot),
                preprocessor=lambda images, **_kwargs: images,
            )
            with self.assertRaisesRegex(RuntimeError, "device agreement"):
                run_siglip2_smoke(
                    SigLIP2SmokeConfig(snapshot),
                    output_json=output,
                    model_loader=lambda _path: loaded,
                    executor=lambda _config, _loaded: (_ for _ in ()).throw(
                        RuntimeError("device agreement failed")
                    ),
                )
            self.assertFalse(output.exists())

    def test_dependency_error_is_reported_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            with (
                patch(
                    "foldcrack_qc.siglip2_smoke.load_local_siglip2_base_vision",
                    side_effect=RuntimeError("locked loader failed"),
                ),
                redirect_stderr(StringIO()) as errors,
            ):
                status = main(["--snapshot-path", str(snapshot), "--device", "cpu"])
            self.assertEqual(status, 2)
            self.assertIn("locked loader failed", errors.getvalue())


if __name__ == "__main__":
    unittest.main()
