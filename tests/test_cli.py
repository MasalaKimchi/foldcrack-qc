from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from foldcrack_qc.cli import _clean_generated_output, _run_tests, entrypoint, main


class TestCommandTests(unittest.TestCase):
    @patch.dict(
        "foldcrack_qc.cli.os.environ",
        {
            "PYTEST_ADDOPTS": "--collect-only",
            "PYTEST_PLUGINS": "unapproved_plugin",
        },
    )
    @patch("foldcrack_qc.cli.subprocess.run")
    @patch("foldcrack_qc.cli.importlib.util.find_spec", return_value=object())
    def test_complete_suite_delegates_to_pytest_with_pattern(
        self,
        _find_spec: object,
        run: Mock,
    ) -> None:
        run.return_value = SimpleNamespace(returncode=7)

        status = _run_tests("check_*.py")

        self.assertEqual(status, 7)
        command = run.call_args.args[0]
        self.assertEqual(command[1:4], ["-m", "pytest", "-q"])
        self.assertIn("python_files=check_*.py", command)
        self.assertEqual(run.call_args.kwargs["check"], False)
        environment = run.call_args.kwargs["env"]
        self.assertNotIn("PYTEST_ADDOPTS", environment)
        self.assertNotIn("PYTEST_PLUGINS", environment)
        self.assertEqual(environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")

    @patch("foldcrack_qc.cli.subprocess.run")
    @patch("foldcrack_qc.cli.importlib.util.find_spec", return_value=None)
    def test_missing_pytest_has_actionable_exit(
        self,
        _find_spec: object,
        run: Mock,
    ) -> None:
        stderr = StringIO()
        with redirect_stderr(stderr):
            status = _run_tests("test*.py")

        self.assertEqual(status, 2)
        self.assertIn("'.[dev]'", stderr.getvalue())
        run.assert_not_called()


class CleanCommandTests(unittest.TestCase):
    @staticmethod
    def _write_marker(directory: Path, **updates: object) -> None:
        payload: dict[str, object] = {
            "kind": "foldcrack_qc_generated_output",
            "schema_version": 1,
            "status": "complete",
        }
        payload.update(updates)
        (directory / "RUN_MANIFEST.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    def test_clean_removes_only_versioned_output_below_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            output = root / "run-001"
            output.mkdir(parents=True)
            self._write_marker(output)
            (output / "result.json").write_text("{}", encoding="utf-8")

            with redirect_stdout(StringIO()):
                result = _clean_generated_output(output, approved_roots=(root,))
            self.assertEqual(result, 0)
            self.assertFalse(output.exists())
            self.assertTrue(root.exists())

    def test_clean_refuses_root_unknown_marker_and_running_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "artifacts"
            output = root / "run-001"
            output.mkdir(parents=True)

            with self.assertRaises(ValueError):
                _clean_generated_output(root, approved_roots=(root,))

            self._write_marker(output, kind="unrecognized")
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))
            self.assertTrue(output.exists())

            self._write_marker(output, schema_version=2)
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))

            self._write_marker(output, status="running")
            with self.assertRaises(ValueError):
                _clean_generated_output(output, approved_roots=(root,))

    def test_clean_refuses_unapproved_and_symlinked_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            approved = directory / "approved"
            outside = directory / "outside" / "run"
            outside.mkdir(parents=True)
            approved.mkdir()
            self._write_marker(outside)
            with self.assertRaises(ValueError):
                _clean_generated_output(outside, approved_roots=(approved,))
            self.assertTrue(outside.exists())

            real = approved / "real-run"
            real.mkdir()
            self._write_marker(real)
            linked = approved / "linked-run"
            linked.symlink_to(real, target_is_directory=True)
            with self.assertRaises(ValueError):
                _clean_generated_output(linked, approved_roots=(approved,))
            self.assertTrue(real.exists())


class EvaluationCommandTests(unittest.TestCase):
    def test_multiplex_proxy_cli_defaults_to_locked_logo_cross_validation(self) -> None:
        fields = [
            SimpleNamespace(modality="comet"),
            SimpleNamespace(modality="cosmx"),
        ]
        report = {
            "schema_version": "multiplex-real-background-proxy-logo-cv-v3",
            "report_eligible": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "proxy.json"
            with (
                patch(
                    "foldcrack_qc.multiplex_proxy_benchmark.load_public_multiplex_fields",
                    return_value=fields,
                ) as loader,
                patch(
                    "foldcrack_qc.multiplex_proxy_benchmark.run_multiplex_proxy_cross_validation",
                    return_value=report,
                ) as cross_validator,
                patch(
                    "foldcrack_qc.multiplex_proxy_benchmark.run_multiplex_proxy_benchmark"
                ) as locked_split,
                patch(
                    "foldcrack_qc.multiplex_proxy_benchmark.write_multiplex_proxy_report",
                    return_value=output,
                ) as writer,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "multiplex-proxy-benchmark",
                        "--comet-dir",
                        "data/comet",
                        "--cosmx-dir",
                        "data/cosmx-a",
                        "data/cosmx-b",
                        "--max-dimension",
                        "256",
                        "--group-bootstrap-resamples",
                        "100",
                        "--output-json",
                        str(output),
                    ]
                )

        self.assertEqual(status, 0)
        loader.assert_called_once_with(
            comet_dir=Path("data/comet"),
            cosmx_dir=(Path("data/cosmx-a"), Path("data/cosmx-b")),
            max_dimension=256,
        )
        config = cross_validator.call_args.args[1]
        self.assertEqual(config.group_bootstrap_resamples, 100)
        locked_split.assert_not_called()
        writer.assert_called_once_with(report, output)

    def test_feasibility_cli_uses_corrected_thin_crack_patch_default(self) -> None:
        outcome = {
            "engineering_smoke_test_passed": True,
            "summary": "passed",
            "report_path": "report.md",
        }
        with (
            patch(
                "foldcrack_qc.benchmark.run_feasibility",
                return_value=outcome,
            ) as runner,
            redirect_stdout(StringIO()),
        ):
            status = main(["feasibility", "--output", "artifacts/test-default"])

        self.assertEqual(status, 0)
        self.assertEqual(runner.call_args.args[0].patch_size, 32)

    def test_foundation_smoke_cli_wires_auditable_configuration(self) -> None:
        revision = "a" * 40
        captured = StringIO()
        with (
            patch(
                "foldcrack_qc.foundation_smoke.run_foundation_smoke",
                return_value={
                    "status": "passed",
                    "result_type": "engineering_foundation_smoke_only",
                },
            ) as runner,
            redirect_stdout(captured),
        ):
            status = main(
                [
                    "foundation-smoke",
                    "--revision",
                    revision,
                    "--cache-dir",
                    "model-cache",
                    "--device",
                    "mps",
                    "--steady-runs",
                    "2",
                    "--lora-rank",
                    "4",
                    "--output-json",
                    "smoke.json",
                ]
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(captured.getvalue())["status"], "passed")
        config = runner.call_args.args[0]
        self.assertEqual(config.revision, revision)
        self.assertEqual(config.device, "mps")
        self.assertEqual(config.steady_runs, 2)
        self.assertEqual(config.lora_rank, 4)
        self.assertFalse(config.allow_download)
        self.assertEqual(runner.call_args.kwargs["output_json"], Path("smoke.json"))

    def test_frozen_feature_cli_locks_model_provenance_and_physical_scale(
        self,
    ) -> None:
        revision = "b" * 40
        report = {
            "method": {},
            "evidence_boundary": {},
            "outcome_summary": {
                "test_sample_count": 2,
                "evaluated_count": 2,
                "abstained_count": 0,
            },
        }
        loaded = SimpleNamespace(
            model=object(),
            resolved_revision=revision,
            weight_digests=(
                SimpleNamespace(
                    as_dict=lambda: {
                        "filename": "model.safetensors",
                        "sha256": "c" * 64,
                        "size_bytes": 123,
                    }
                ),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            output_path = Path(temporary) / "benchmark.json"
            with (
                patch(
                    "foldcrack_qc.foundation_smoke.load_huggingface_model",
                    return_value=loaded,
                ) as loader,
                patch(
                    "foldcrack_qc.foundation_smoke.dinov2_model_geometry",
                    return_value=((14, 14), 1),
                ),
                patch(
                    "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                    return_value=object(),
                ),
                patch(
                    "foldcrack_qc.frozen_benchmark.run_frozen_anomaly_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "frozen-feature-benchmark",
                        "--fit-manifest",
                        "fit.json",
                        "--calibration-manifest",
                        "calibration.json",
                        "--locked-test-manifest",
                        "test.json",
                        "--revision",
                        revision,
                        "--cache-dir",
                        "model-cache",
                        "--device",
                        "mps",
                        "--patch-size-um",
                        "112",
                        "--stride-um",
                        "56",
                        "--n-resamples",
                        "10",
                        "--output-json",
                        str(output_path),
                    ]
                )

            self.assertEqual(status, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["evidence_boundary"]["model_identity_locked"])
            model = payload["method"]["model_identity"]
            self.assertEqual(model["requested_revision"], revision)
            self.assertEqual(model["resolved_revision"], revision)
            self.assertEqual(model["weight_files"][0]["sha256"], "c" * 64)
            self.assertFalse(loader.call_args.args[0].allow_download)
            self.assertEqual(runner.call_args.kwargs["patch_size_um"], 112.0)
            self.assertEqual(runner.call_args.kwargs["stride_um"], 56.0)

    def test_public_fold_cli_classical_only_does_not_build_provider(self) -> None:
        report = {
            "status": "complete_nonreportable_feasibility_run",
            "report_eligible": False,
            "methods": {
                "classical_fold": {
                    "locked_test": {"pixel_all_fields_micro": {"dice": 0.5}}
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "classical-fold.json"
            with (
                patch(
                    "foldcrack_qc.public_fold_providers.build_public_fold_encoder"
                ) as builder,
                patch(
                    "foldcrack_qc.cli._public_fold_run_provenance",
                    return_value={"capture": "test"},
                ) as provenance,
                patch(
                    "foldcrack_qc.public_fold_benchmark.run_public_fold_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "classical_fold",
                        "--foundation-encoder",
                        "hibou-b-local",
                        "--allow-download",
                        "--bootstrap-resamples",
                        "0",
                        "--output-json",
                        str(output),
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(status, 0)
        builder.assert_not_called()
        self.assertIsNone(runner.call_args.kwargs["encoder"])
        provenance.assert_called_once()
        self.assertIsNone(provenance.call_args.args[1])
        self.assertNotIn("model_identity", payload)

    def test_public_fold_cli_validates_config_before_loading_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "invalid.json"
            errors = StringIO()
            with (
                patch(
                    "foldcrack_qc.public_fold_providers.build_public_fold_encoder"
                ) as builder,
                patch("foldcrack_qc.cli._public_fold_run_provenance") as provenance,
                patch(
                    "foldcrack_qc.public_fold_benchmark.run_public_fold_benchmark"
                ) as runner,
                redirect_stderr(errors),
            ):
                status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_patchknn",
                        "--tile-size",
                        "224",
                        "--tile-stride",
                        "225",
                        "--bootstrap-resamples",
                        "0",
                        "--output-json",
                        str(output),
                    ]
                )
            output_created = output.exists()

        self.assertEqual(status, 2)
        self.assertIn("tile_stride cannot exceed tile_size", errors.getvalue())
        builder.assert_not_called()
        provenance.assert_not_called()
        runner.assert_not_called()
        self.assertFalse(output_created)

    def test_public_fold_cli_wires_real_dataset_and_audited_mask_exclusion(
        self,
    ) -> None:
        revision = "d" * 40
        digest = SimpleNamespace(
            as_dict=lambda: {
                "filename": "model.safetensors",
                "sha256": "e" * 64,
                "size_bytes": 456,
            }
        )
        loaded = SimpleNamespace(
            model=object(),
            resolved_revision=revision,
            weight_digests=(digest,),
            configuration_digests=(),
        )
        report = {
            "status": "complete_reportable_real_public_fold_benchmark",
            "report_eligible": True,
            "methods": {
                "dinov2_linear_probe": {
                    "locked_test": {"pixel_all_fields_micro": {"dice": 0.7}}
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "public-fold.json"
            with (
                patch(
                    "foldcrack_qc.foundation_smoke.load_huggingface_model",
                    return_value=loaded,
                ),
                patch(
                    "foldcrack_qc.foundation_smoke.dinov2_model_geometry",
                    return_value=((14, 14), 1),
                ),
                patch(
                    "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                    return_value=SimpleNamespace(device="mps", encode=Mock()),
                ),
                patch(
                    "foldcrack_qc.public_fold_benchmark.run_public_fold_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "dinov2_linear_probe",
                        "--revision",
                        revision,
                        "--device",
                        "mps",
                        "--exclude-empty-positive-masks",
                        "--bootstrap-resamples",
                        "10",
                        "--probe-max-iterations",
                        "123",
                        "--output-json",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            config = runner.call_args.kwargs["config"]
            self.assertEqual(config.methods, ("dinov2_linear_probe",))
            self.assertEqual(config.empty_positive_mask_policy, "exclude_localization")
            self.assertEqual(config.bootstrap_resamples, 10)
            self.assertEqual(config.probe_max_iterations, 123)
            provenance = runner.call_args.kwargs["run_provenance"]
            self.assertEqual(provenance["method_model"]["weights_sha256"], "e" * 64)
            self.assertTrue(provenance["capture"]["captured_before_scoring"])
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_identity"]["requested_revision"], revision)
            self.assertEqual(
                payload["model_identity"]["weight_files"][0]["sha256"], "e" * 64
            )

    def test_public_fold_cli_loads_local_hibou_with_official_contract(self) -> None:
        local = SimpleNamespace(
            model=object(),
            provenance={
                "id": "HistAI/Hibou-B",
                "weights": {"sha256": "a" * 64},
                "source": {"commit": "b" * 40},
                "license": {"spdx": "Apache-2.0"},
                "trust_remote_code": False,
                "network_access_allowed": False,
            },
        )
        report = {
            "status": "complete_nonreportable_feasibility_run",
            "report_eligible": False,
            "methods": {
                "foundation_linear_probe": {
                    "locked_test": {"pixel_all_fields_micro": {"dice": 0.8}}
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "hibou-fold.json"
            with (
                patch(
                    "foldcrack_qc.foundation.load_local_hibou_b",
                    return_value=local,
                ) as loader,
                patch(
                    "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                    return_value=SimpleNamespace(device="mps", encode=Mock()),
                ) as extractor,
                patch(
                    "foldcrack_qc.public_fold_benchmark.run_public_fold_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_linear_probe",
                        "--foundation-encoder",
                        "hibou-b-local",
                        "--hibou-weights",
                        "models/hibou-b/hibou-b.pth",
                        "--hibou-source",
                        "models/hibou-b/source",
                        "--hibou-weights-sha256",
                        "a" * 64,
                        "--hibou-source-commit",
                        "b" * 40,
                        "--device",
                        "mps",
                        "--bootstrap-resamples",
                        "0",
                        "--output-json",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            loader.assert_called_once_with(
                Path("models/hibou-b/hibou-b.pth"),
                Path("models/hibou-b/source"),
                expected_weights_sha256="a" * 64,
                expected_source_commit="b" * 40,
            )
            kwargs = extractor.call_args.kwargs
            self.assertEqual(kwargs["model_input_name"], None)
            self.assertEqual(kwargs["patch_size"], 14)
            self.assertEqual(kwargs["prefix_tokens"], 5)
            self.assertEqual(kwargs["normalization_mean"], (0.7068, 0.5755, 0.722))
            self.assertEqual(kwargs["normalization_std"], (0.195, 0.2316, 0.1816))
            self.assertEqual(
                runner.call_args.kwargs["config"].methods, ("foundation_linear_probe",)
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["model_identity"]["id"], "HistAI/Hibou-B")
            self.assertEqual(payload["model_identity"]["resolved_device"], "mps")
            self.assertFalse(payload["model_identity"]["trust_remote_code"])

    def test_public_fold_cli_loads_local_siglip2_with_dense_contract(self) -> None:
        local = SimpleNamespace(
            model=object(),
            preprocessor=object(),
            provenance={
                "id": "google/siglip2-base-patch16-224",
                "source": {"revision": "a" * 40},
                "assets": {"model.safetensors": {"sha256": "b" * 64}},
                "license": {"spdx": "Apache-2.0"},
                "trust_remote_code": False,
                "token_used": False,
                "network_access_allowed": False,
            },
        )
        report = {
            "status": "complete_nonreportable_feasibility_run",
            "report_eligible": False,
            "methods": {
                "foundation_linear_probe": {
                    "locked_test": {"pixel_all_fields_micro": {"dice": 0.8}}
                }
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "siglip2-fold.json"
            with (
                patch(
                    "foldcrack_qc.foundation.load_local_siglip2_base_vision",
                    return_value=local,
                ) as loader,
                patch(
                    "foldcrack_qc.foundation.DINOv2FeatureExtractor",
                    return_value=SimpleNamespace(device="mps", encode=Mock()),
                ) as extractor,
                patch(
                    "foldcrack_qc.public_fold_benchmark.run_public_fold_benchmark",
                    return_value=report,
                ) as runner,
                redirect_stdout(StringIO()),
            ):
                status = main(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_linear_probe",
                        "--foundation-encoder",
                        "siglip2-base-local",
                        "--siglip2-snapshot",
                        "models/siglip2-base",
                        "--device",
                        "mps",
                        "--bootstrap-resamples",
                        "0",
                        "--output-json",
                        str(output),
                    ]
                )

            self.assertEqual(status, 0)
            loader.assert_called_once_with(Path("models/siglip2-base"))
            kwargs = extractor.call_args.kwargs
            self.assertEqual(kwargs["model_input_name"], "pixel_values")
            self.assertEqual(kwargs["global_embedding_name"], "pooler_output")
            self.assertEqual(kwargs["patch_size"], 16)
            self.assertEqual(kwargs["prefix_tokens"], 0)
            self.assertEqual(kwargs["normalization_mean"], (0.5, 0.5, 0.5))
            self.assertEqual(kwargs["normalization_std"], (0.5, 0.5, 0.5))
            self.assertIs(kwargs["preprocessor"], local.preprocessor)
            self.assertEqual(
                runner.call_args.kwargs["config"].methods,
                ("foundation_linear_probe",),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            identity = payload["model_identity"]
            self.assertEqual(identity["id"], "google/siglip2-base-patch16-224")
            self.assertEqual(identity["resolved_device"], "mps")
            self.assertEqual(identity["output_contract"]["global_key"], "pooler_output")
            self.assertEqual(
                identity["output_contract"]["patch_key"], "last_hidden_state"
            )
            self.assertEqual(identity["output_contract"]["prefix_tokens"], 0)
            self.assertFalse(identity["network_access_allowed"])

    def test_public_fold_cli_rejects_unsafe_siglip2_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "out.json")

            errors = StringIO()
            with redirect_stderr(errors):
                missing_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_patchknn",
                        "--foundation-encoder",
                        "siglip2-base-local",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(missing_status, 2)
            self.assertIn("requires --siglip2-snapshot", errors.getvalue())

            errors = StringIO()
            with redirect_stderr(errors):
                alias_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "dinov2_patchknn",
                        "--foundation-encoder",
                        "siglip2-base-local",
                        "--siglip2-snapshot",
                        "models/siglip2-base",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(alias_status, 2)
            self.assertIn("not DINOv2 aliases", errors.getvalue())

            errors = StringIO()
            with redirect_stderr(errors):
                download_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_patchknn",
                        "--foundation-encoder",
                        "siglip2-base-local",
                        "--siglip2-snapshot",
                        "models/siglip2-base",
                        "--allow-download",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(download_status, 2)
            self.assertIn("local-only SigLIP2 Base loader", errors.getvalue())

    def test_public_fold_cli_rejects_incomplete_or_ambiguous_hibou_selection(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = str(Path(temporary) / "out.json")
            errors = StringIO()
            with redirect_stderr(errors):
                missing_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_patchknn",
                        "--foundation-encoder",
                        "hibou-b-local",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(missing_status, 2)
            self.assertIn("requires both --hibou-weights", errors.getvalue())

            errors = StringIO()
            with redirect_stderr(errors):
                unpinned_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "foundation_patchknn",
                        "--foundation-encoder",
                        "hibou-b-local",
                        "--hibou-weights",
                        "weights.pth",
                        "--hibou-source",
                        "source",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(unpinned_status, 2)
            self.assertIn("requires both --hibou-weights-sha256", errors.getvalue())

            errors = StringIO()
            with redirect_stderr(errors):
                alias_status = entrypoint(
                    [
                        "public-fold-benchmark",
                        "--dataset-root",
                        "public-data",
                        "--methods",
                        "dinov2_patchknn",
                        "--foundation-encoder",
                        "hibou-b-local",
                        "--hibou-weights",
                        "weights.pth",
                        "--hibou-source",
                        "source",
                        "--output-json",
                        output,
                    ]
                )
            self.assertEqual(alias_status, 2)
            self.assertIn("not DINOv2 aliases", errors.getvalue())

    def test_validate_benchmark_distinguishes_valid_plan_from_report_ready(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        contract = root / "configs" / "benchmark.real.example.json"
        captured = StringIO()
        with redirect_stdout(captured):
            status = main(["validate-benchmark", str(contract), "--json"])
        payload = json.loads(captured.getvalue())
        self.assertEqual(status, 0)
        self.assertTrue(payload["configuration_valid"])
        self.assertFalse(payload["report_eligible"])

        with redirect_stdout(StringIO()):
            strict_status = main(
                [
                    "validate-benchmark",
                    str(contract),
                    "--require-report-eligible",
                ]
            )
        self.assertEqual(strict_status, 3)

    def test_shared_entrypoint_converts_validation_error_to_exit_code(self) -> None:
        errors = StringIO()
        with redirect_stderr(errors):
            status = entrypoint(["validate-benchmark", "missing.json"])
        self.assertEqual(status, 2)
        self.assertIn("error:", errors.getvalue())

    def test_validate_manifest_cli_json_and_strict_exit_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            np.save(directory / "image.npy", np.zeros((16, 18, 3), dtype=np.uint8))
            record = {
                "sample_id": "sample-opaque",
                "patient_id": "patient-opaque",
                "modality": "he",
                "image_path": "image.npy",
                "split": "development",
                "pixel_size_um": 0.5,
            }
            path = directory / "manifest.json"
            path.write_text(json.dumps([record]), encoding="utf-8")

            output = StringIO()
            with redirect_stdout(output):
                exploratory_status = main(["validate-manifest", str(path), "--json"])
            payload = json.loads(output.getvalue())
            self.assertEqual(exploratory_status, 0)
            self.assertTrue(payload["valid"])
            self.assertGreater(payload["warning_count"], 0)

            with redirect_stdout(StringIO()):
                strict_status = main(["validate-manifest", str(path), "--strict"])
            self.assertEqual(strict_status, 2)

    def test_operational_eval_preserves_synthetic_not_evaluated_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [
                {
                    "modality": "he",
                    "decision": "REVIEW",
                    "reference_actionable_artifact": True,
                    "reference_severe": True,
                    "technical_abstention": False,
                    "high_confidence_mask_precision": 1.0,
                    "valid_tissue_overmask_fraction": 0.0,
                }
            ]
            acceptance = {
                "severe_artifact_sensitivity_lcb": 0.0,
                "auto_pass_npv": 0.0,
                "high_confidence_mask_precision_lcb": 0.0,
                "valid_tissue_overmask_rate_max": 1.0,
                "review_referral_rate_max": 1.0,
                "minimum_severe_positive_samples": 1,
                "minimum_auto_pass_samples": 1,
                "minimum_mask_evaluated_samples": 1,
                "minimum_prevalence_samples": 1,
            }
            records_path = directory / "records.json"
            acceptance_path = directory / "acceptance.json"
            output_path = directory / "report.json"
            records_path.write_text(json.dumps({"records": records}), encoding="utf-8")
            acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")

            captured = StringIO()
            with redirect_stdout(captured):
                status = main(
                    [
                        "operational-eval",
                        "--records",
                        str(records_path),
                        "--acceptance",
                        str(acceptance_path),
                        "--synthetic",
                        "--output",
                        str(output_path),
                    ]
                )
            stdout_payload = json.loads(captured.getvalue())
            file_payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(status, 0)
            self.assertEqual(stdout_payload, file_payload)
            self.assertEqual(
                stdout_payload["overall_status"], "NOT_EVALUATED_SYNTHETIC"
            )
            self.assertFalse(stdout_payload["acceptance_eligible"])

            with redirect_stdout(StringIO()):
                locked_status = main(
                    [
                        "operational-eval",
                        "--records",
                        str(records_path),
                        "--acceptance",
                        str(acceptance_path),
                    ]
                )
            self.assertEqual(locked_status, 3)


if __name__ == "__main__":
    unittest.main()
