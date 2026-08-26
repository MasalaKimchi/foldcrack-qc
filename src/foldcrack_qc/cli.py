"""Command-line entry points for feasibility, data discovery, and testing."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .registry import format_registry, load_registry


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foldcrack-qc",
        description="Fold/crack QC feasibility toolkit for H&E, COMET, and CosMx",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    feasibility = subparsers.add_parser(
        "feasibility", help="run the end-to-end synthetic engineering benchmark"
    )
    feasibility.add_argument(
        "--output", type=Path, default=Path("artifacts/feasibility")
    )
    feasibility.add_argument("--samples-per-modality", type=_positive_int, default=12)
    feasibility.add_argument(
        "--clean-samples-per-modality", type=_positive_int, default=6
    )
    feasibility.add_argument("--size", type=_positive_int, default=384)
    feasibility.add_argument("--seed", type=int, default=17)
    feasibility.add_argument("--patch-size", type=_positive_int, default=64)
    feasibility.add_argument("--overlays-per-modality", type=int, default=2)

    datasets = subparsers.add_parser(
        "datasets", help="list public resources and license caveats"
    )
    datasets.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    datasets.add_argument("--registry", type=Path, default=None)

    tests = subparsers.add_parser(
        "test", help="run the dependency-light unittest suite"
    )
    tests.add_argument("--pattern", default="test*.py")

    manifest = subparsers.add_parser(
        "validate-manifest", help="validate an internal JSON/JSONL data manifest"
    )
    manifest.add_argument("manifest", type=Path)
    manifest.add_argument(
        "--strict",
        action="store_true",
        help="enforce locked-evaluation metadata, checksums, and isolation",
    )
    manifest.add_argument("--json", action="store_true", help="emit a JSON report")

    benchmark_contract = subparsers.add_parser(
        "validate-benchmark",
        help="validate a real-data benchmark contract and scientific eligibility",
    )
    benchmark_contract.add_argument("contract", type=Path)
    benchmark_contract.add_argument(
        "--cohort-records",
        type=Path,
        default=None,
        help="optional realized fit/calibration/test record JSON",
    )
    benchmark_contract.add_argument(
        "--require-report-eligible",
        action="store_true",
        help="fail unless real data, labels, methods, and governance gates are ready",
    )
    benchmark_contract.add_argument(
        "--json", action="store_true", help="emit a JSON report"
    )

    foundation_smoke = subparsers.add_parser(
        "foundation-smoke",
        help="run an auditable DINOv2 CPU/MPS and optional LoRA engineering smoke",
    )
    foundation_smoke.add_argument(
        "--revision",
        required=True,
        help="exact immutable 40-character Hugging Face commit hash",
    )
    foundation_smoke.add_argument("--model-id", default="facebook/dinov2-small")
    foundation_smoke.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/foldcrack_qc/huggingface"),
    )
    foundation_smoke.add_argument(
        "--device", choices=("auto", "cpu", "mps"), default="auto"
    )
    foundation_smoke.add_argument(
        "--allow-download",
        action="store_true",
        help="explicitly allow an unauthenticated public model download",
    )
    foundation_smoke.add_argument("--image-size", type=_positive_int, default=224)
    foundation_smoke.add_argument("--steady-runs", type=_positive_int, default=3)
    foundation_smoke.add_argument("--lora-rank", type=int, choices=(4, 8))
    foundation_smoke.add_argument("--max-device-abs-error", type=float, default=1e-3)
    foundation_smoke.add_argument(
        "--min-device-cosine-similarity", type=float, default=0.9999
    )
    foundation_smoke.add_argument("--output-json", type=Path, default=None)

    frozen_benchmark = subparsers.add_parser(
        "frozen-feature-benchmark",
        help="run the strict real H&E DINOv2 artifact-union anomaly benchmark",
    )
    frozen_benchmark.add_argument("--fit-manifest", type=Path, required=True)
    frozen_benchmark.add_argument(
        "--calibration-manifest", type=Path, required=True
    )
    frozen_benchmark.add_argument(
        "--locked-test-manifest", type=Path, required=True
    )
    frozen_benchmark.add_argument("--revision", required=True)
    frozen_benchmark.add_argument("--model-id", default="facebook/dinov2-small")
    frozen_benchmark.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/foldcrack_qc/huggingface"),
    )
    frozen_benchmark.add_argument(
        "--device", choices=("auto", "cpu", "mps"), default="auto"
    )
    frozen_benchmark.add_argument("--allow-download", action="store_true")
    frozen_benchmark.add_argument(
        "--encoder-image-size", type=_positive_int, default=224
    )
    frozen_benchmark.add_argument(
        "--patch-size-um",
        type=float,
        required=True,
        help="physical H&E patch height/width in micrometres",
    )
    frozen_benchmark.add_argument(
        "--stride-um",
        type=float,
        default=None,
        help="physical stride in micrometres; defaults to half the patch size",
    )
    frozen_benchmark.add_argument("--batch-size", type=_positive_int, default=8)
    frozen_benchmark.add_argument(
        "--min-valid-token-fraction", type=float, default=0.5
    )
    frozen_benchmark.add_argument("--neighbors", type=_positive_int, default=1)
    frozen_benchmark.add_argument(
        "--calibration-quantile", type=float, default=0.995
    )
    frozen_benchmark.add_argument(
        "--max-reference-tokens", type=_positive_int, default=100_000
    )
    frozen_benchmark.add_argument(
        "--max-calibration-pixels", type=_positive_int, default=1_000_000
    )
    frozen_benchmark.add_argument(
        "--max-raster-pixels", type=_positive_int, default=25_000_000
    )
    frozen_benchmark.add_argument("--n-resamples", type=_positive_int, default=2_000)
    frozen_benchmark.add_argument("--bootstrap-seed", type=int, default=0)
    frozen_benchmark.add_argument(
        "--minimum-positive-test-samples", type=_positive_int, default=1
    )
    frozen_benchmark.add_argument(
        "--minimum-negative-test-samples", type=_positive_int, default=1
    )
    frozen_benchmark.add_argument(
        "--minimum-test-patient-clusters", type=_positive_int, default=2
    )
    frozen_benchmark.add_argument("--output-json", type=Path, required=True)
    frozen_benchmark.add_argument(
        "--json", action="store_true", help="also emit the complete report to stdout"
    )

    operational = subparsers.add_parser(
        "operational-eval", help="evaluate PASS/REVIEW/FAIL acceptance records"
    )
    operational.add_argument("--records", type=Path, required=True)
    operational.add_argument("--acceptance", type=Path, required=True)
    operational.add_argument("--synthetic", action="store_true")
    operational.add_argument("--output", type=Path, default=None)

    clean = subparsers.add_parser(
        "clean", help="remove a generated benchmark directory"
    )
    clean.add_argument("--output", type=Path, required=True)
    return parser


def _run_tests(pattern: str) -> int:
    root = Path(__file__).resolve().parents[2]
    suite = unittest.defaultTestLoader.discover(str(root / "tests"), pattern=pattern)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def _path_has_symlink_below(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    if root.is_symlink():
        return True
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            return True
    return False


def _clean_generated_output(
    path: Path, *, approved_roots: Sequence[Path] | None = None
) -> int:
    lexical = path.expanduser()
    lexical = lexical if lexical.is_absolute() else Path.cwd() / lexical
    resolved = lexical.resolve()
    repository_root = Path(__file__).resolve().parents[2]
    lexical_roots = tuple(
        expanded
        if (expanded := root.expanduser()).is_absolute()
        else Path.cwd() / expanded
        for root in (approved_roots or (repository_root / "artifacts",))
    )
    roots = tuple(root.resolve() for root in lexical_roots)
    denied = {
        Path("/").resolve(),
        Path.home().resolve(),
        Path.cwd().resolve(),
        repository_root,
        *roots,
    }
    if resolved in denied or len(resolved.parts) < 4:
        raise ValueError("Refusing to remove a broad or unsafe path")
    if not any(resolved.is_relative_to(root) and resolved != root for root in roots):
        raise ValueError(
            "Refusing to remove output outside the approved artifacts root"
        )
    matching_lexical_roots = [
        root for root in lexical_roots if lexical.is_relative_to(root)
    ]
    if not matching_lexical_roots or any(
        _path_has_symlink_below(lexical, root) for root in matching_lexical_roots
    ):
        raise ValueError("Refusing to remove a path containing a symbolic link")
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(
            "Refusing to remove a missing, non-directory, or linked output"
        )
    marker = resolved / "RUN_MANIFEST.json"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError(
            "Refusing to remove output without a regular RUN_MANIFEST.json marker"
        )
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Refusing to remove output with an unreadable marker") from exc
    if not isinstance(payload, Mapping):
        raise TypeError("Refusing to remove output with an invalid marker")
    if (
        payload.get("kind") != "foldcrack_qc_generated_output"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(
            "Refusing to remove output with an unknown marker kind or version"
        )
    if payload.get("status") not in {"complete", "incomplete"}:
        raise ValueError(
            "Refusing to remove output whose generated run is not finished"
        )
    shutil.rmtree(resolved)
    print(f"Removed generated benchmark output: {resolved}")
    return 0


def _read_json(path: Path, *, collection_key: str | None = None) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if collection_key is not None and isinstance(payload, Mapping):
        payload = payload.get(collection_key)
    return payload


def _run_manifest_validation(path: Path, *, strict: bool, emit_json: bool) -> int:
    from .manifest import validate_manifest

    report = validate_manifest(path, strict=strict)
    if emit_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        mode = "strict" if strict else "exploratory"
        print(
            f"Manifest {mode} validation: {'PASS' if report.valid else 'FAIL'}; "
            f"{report.valid_sample_count}/{report.record_count} records loadable; "
            f"{report.error_count} errors, {report.warning_count} warnings"
        )
        for issue in report.issues:
            location = (
                f"record {issue.record_index}"
                if issue.record_index is not None
                else "manifest"
            )
            sample = (
                f", sample_id={issue.sample_id!r}"
                if issue.sample_id is not None
                else ""
            )
            print(
                f"{issue.severity.upper()} [{issue.code}] {location}{sample}: {issue.message}"
            )
    return 0 if report.valid else 2


def _run_benchmark_validation(
    contract_path: Path,
    *,
    cohort_records_path: Path | None,
    require_report_eligible: bool,
    emit_json: bool,
) -> int:
    from .benchmark_contract import validate_benchmark_contract

    cohort_records = (
        None if cohort_records_path is None else _read_json(cohort_records_path)
    )
    report = validate_benchmark_contract(
        contract_path,
        cohort_records=cohort_records,
    )
    if emit_json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            "Benchmark contract: "
            f"{report.status}; configuration_valid={report.configuration_valid}; "
            f"scientific_report_eligible={report.report_eligible}"
        )
        print(
            f"Eligible methods: {', '.join(report.eligible_method_ids) or 'none'}"
        )
        for issue in report.issues:
            print(
                f"{issue.severity.upper()} [{issue.code}] "
                f"{issue.path}: {issue.message}"
            )
    if not report.configuration_valid:
        return 2
    if require_report_eligible and not report.report_eligible:
        return 3
    return 0


def _run_foundation_smoke(args: argparse.Namespace) -> int:
    from .foundation_smoke import FoundationSmokeConfig, run_foundation_smoke

    config = FoundationSmokeConfig(
        revision=args.revision,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        allow_download=args.allow_download,
        image_size=args.image_size,
        steady_runs=args.steady_runs,
        lora_rank=args.lora_rank,
        max_device_abs_error=args.max_device_abs_error,
        min_device_cosine_similarity=args.min_device_cosine_similarity,
    )
    report = run_foundation_smoke(config, output_json=args.output_json)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report.get("status") == "passed" else 2


def _run_frozen_feature_benchmark(args: argparse.Namespace) -> int:
    from .foundation import DINOv2FeatureExtractor
    from .foundation_smoke import (
        FoundationSmokeConfig,
        dinov2_model_geometry,
        load_huggingface_model,
    )
    from .frozen_benchmark import run_frozen_anomaly_benchmark

    model_config = FoundationSmokeConfig(
        revision=args.revision,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        allow_download=args.allow_download,
        image_size=args.encoder_image_size,
        steady_runs=1,
    )
    loaded = load_huggingface_model(model_config)
    patch_size, prefix_tokens = dinov2_model_geometry(
        loaded.model, args.encoder_image_size
    )
    encoder = DINOv2FeatureExtractor(
        loaded.model,
        device=args.device,
        image_size=args.encoder_image_size,
        patch_size=patch_size,
        prefix_tokens=prefix_tokens,
        model_input_name="pixel_values",
    )
    report = run_frozen_anomaly_benchmark(
        args.fit_manifest,
        args.calibration_manifest,
        args.locked_test_manifest,
        encoder=encoder,
        patch_size_um=args.patch_size_um,
        stride_um=args.stride_um,
        batch_size=args.batch_size,
        min_valid_token_fraction=args.min_valid_token_fraction,
        neighbors=args.neighbors,
        calibration_quantile=args.calibration_quantile,
        max_reference_tokens=args.max_reference_tokens,
        max_calibration_pixels=args.max_calibration_pixels,
        max_raster_pixels=args.max_raster_pixels,
        n_resamples=args.n_resamples,
        bootstrap_seed=args.bootstrap_seed,
        minimum_positive_test_samples=args.minimum_positive_test_samples,
        minimum_negative_test_samples=args.minimum_negative_test_samples,
        minimum_test_patient_clusters=args.minimum_test_patient_clusters,
    )
    report["method"]["model_identity"] = {
        "id": model_config.model_id,
        "requested_revision": model_config.revision,
        "resolved_revision": loaded.resolved_revision,
        "weight_files": [digest.as_dict() for digest in loaded.weight_digests],
        "trust_remote_code": False,
        "token_used": False,
    }
    report["evidence_boundary"]["model_identity_locked"] = True
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered, encoding="utf-8")
    summary = report["outcome_summary"]
    if args.json:
        print(rendered, end="")
    else:
        print(
            "Frozen-feature benchmark complete: "
            f"{summary['evaluated_count']}/{summary['test_sample_count']} evaluated; "
            f"{summary['abstained_count']} abstained; report={args.output_json}"
        )
    return 0 if int(summary["abstained_count"]) == 0 else 3


def _run_operational_evaluation(
    records_path: Path,
    acceptance_path: Path,
    *,
    synthetic: bool,
    output: Path | None,
) -> int:
    from .operational import evaluate_operational_decisions

    records = _read_json(records_path, collection_key="records")
    acceptance = _read_json(acceptance_path)
    if not isinstance(records, list) or not all(
        isinstance(item, Mapping) for item in records
    ):
        raise ValueError(
            "Operational records must be a JSON list or an object containing 'records'"
        )
    if not isinstance(acceptance, Mapping):
        raise TypeError("Acceptance configuration must be a JSON object")
    report = evaluate_operational_decisions(records, acceptance, synthetic=synthetic)
    rendered = json.dumps(report, indent=2) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if synthetic:
        return 0
    return 3 if report.get("overall_status") in {"FAIL", "INSUFFICIENT_EVIDENCE"} else 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "datasets":
        records = load_registry(args.registry)
        if args.json:
            print(json.dumps(records, indent=2))
        else:
            print(format_registry(records))
        return 0
    if args.command == "test":
        return _run_tests(args.pattern)
    if args.command == "validate-manifest":
        return _run_manifest_validation(
            args.manifest, strict=args.strict, emit_json=args.json
        )
    if args.command == "validate-benchmark":
        return _run_benchmark_validation(
            args.contract,
            cohort_records_path=args.cohort_records,
            require_report_eligible=args.require_report_eligible,
            emit_json=args.json,
        )
    if args.command == "foundation-smoke":
        return _run_foundation_smoke(args)
    if args.command == "frozen-feature-benchmark":
        return _run_frozen_feature_benchmark(args)
    if args.command == "operational-eval":
        return _run_operational_evaluation(
            args.records,
            args.acceptance,
            synthetic=args.synthetic,
            output=args.output,
        )
    if args.command == "clean":
        return _clean_generated_output(args.output)
    if args.command == "feasibility":
        from .benchmark import BenchmarkConfig, run_feasibility

        config = BenchmarkConfig(
            output_dir=args.output,
            samples_per_modality=args.samples_per_modality,
            clean_samples_per_modality=args.clean_samples_per_modality,
            image_size=(args.size, args.size),
            seed=args.seed,
            patch_size=args.patch_size,
            overlays_per_modality=max(0, args.overlays_per_modality),
        )
        outcome = run_feasibility(config)
        print(outcome["summary"])
        print(f"Report: {outcome['report_path']}")
        return 0 if outcome["engineering_smoke_test_passed"] else 2
    raise AssertionError(f"Unhandled command {args.command}")


def entrypoint(argv: list[str] | None = None) -> int:
    """Console/package entry point with concise, consistent failures."""

    try:
        return main(argv)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(entrypoint())
