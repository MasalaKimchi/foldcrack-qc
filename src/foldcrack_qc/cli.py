"""Command-line entry points for feasibility, data discovery, and testing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .public_fold_providers import PUBLIC_FOLD_ENCODER_NAMES
from .registry import format_registry, load_registry


def _canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _public_fold_code_identity() -> dict[str, Any]:
    """Capture the committed revision plus every uncommitted runtime source byte."""

    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "diff", "--binary", "HEAD", "--", "src", "pyproject.toml"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
        untracked_output = subprocess.run(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "pyproject.toml",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Public fold reporting requires a readable Git source identity"
        ) from error
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise RuntimeError("Public fold reporting requires an exact Git commit")
    digest = hashlib.sha256(diff)
    untracked: list[str] = []
    for raw_relative in sorted(untracked_output.splitlines()):
        relative = Path(raw_relative)
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError("Unsafe untracked source path in Git output") from error
        if not candidate.is_file():
            continue
        untracked.append(relative.as_posix())
        digest.update(b"\0untracked\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(candidate.read_bytes())
    return {
        "identity_type": "git",
        "commit": commit,
        "dirty_diff_sha256": digest.hexdigest(),
        "dirty_diff_capture": "git_diff_HEAD_plus_untracked_runtime_sources",
        "untracked_runtime_sources": untracked,
    }


def _foundation_weight_identity(model_identity: Mapping[str, Any]) -> str:
    candidates: list[str] = []
    weights = model_identity.get("weights")
    if isinstance(weights, Mapping) and isinstance(weights.get("sha256"), str):
        candidates.append(str(weights["sha256"]))
    weight_files = model_identity.get("weight_files")
    if isinstance(weight_files, list):
        candidates.extend(
            str(item["sha256"])
            for item in weight_files
            if isinstance(item, Mapping) and isinstance(item.get("sha256"), str)
        )
    assets = model_identity.get("assets")
    if isinstance(assets, Mapping):
        model_asset = assets.get("model.safetensors")
        if isinstance(model_asset, Mapping) and isinstance(
            model_asset.get("sha256"), str
        ):
            candidates.append(str(model_asset["sha256"]))
    unique = sorted(set(candidates))
    if not unique or any(
        len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in unique
    ):
        raise RuntimeError(
            "Foundation benchmark weights lack an exact SHA-256 identity"
        )
    return unique[0] if len(unique) == 1 else _canonical_json_sha256(unique)


def _public_fold_run_provenance(
    config: Any,
    model_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the runner's strict pre-scoring provenance object."""

    import cv2
    import numpy as np
    import scipy

    foundation_requested = any(method != "classical_fold" for method in config.methods)
    if foundation_requested and model_identity is None:
        raise RuntimeError("Foundation methods require captured model identity")
    identity: Mapping[str, Any] = model_identity or {
        "id": "classical-fold-candidates-v1",
        "loader": "in_process_foldcrack_qc.detectors.classical_fold_candidates",
    }
    weights_sha256 = (
        _foundation_weight_identity(identity) if foundation_requested else None
    )
    dependencies = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "opencv": cv2.__version__,
    }
    if foundation_requested:
        for distribution, key in (
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("huggingface-hub", "huggingface_hub"),
        ):
            try:
                dependencies[key] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as error:
                raise RuntimeError(
                    f"Foundation provenance cannot resolve {distribution!r} version"
                ) from error
    return {
        "schema_version": "public-fold-run-provenance-1.1",
        "capture": {
            "captured_before_scoring": True,
            "validation_status": "structurally_validated",
            "validator_id": "foldcrack-qc-cli-preflight-v1.1",
            "approval_scope": "reproducibility_structure_not_corporate_model_governance",
        },
        "code": _public_fold_code_identity(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": dependencies,
        },
        "method_model": {
            "selected_methods": list(config.methods),
            "benchmark_configuration_sha256": _canonical_json_sha256(config.as_dict()),
            "implementation_id": "foldcrack_qc.public_fold_benchmark:v1.2",
            "model_id": str(identity.get("id", "unknown")),
            "model_config_sha256": _canonical_json_sha256(dict(identity)),
            "weights_sha256": weights_sha256,
            "weights_not_applicable": not foundation_requested,
            "loader_identity": str(
                identity.get(
                    "loader",
                    "transformers_pretrained_trust_remote_code_false",
                )
            ),
            "frozen_evaluation": True,
            "transductive_updates": False,
        },
        "execution": {
            "device": str(identity.get("resolved_device", "cpu")),
            "precision": "float32",
        },
    }


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _fixed_hex(value: str, *, length: int, label: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise argparse.ArgumentTypeError(
            f"{label} must contain exactly {length} hexadecimal characters"
        )
    return normalized


def _sha256(value: str) -> str:
    return _fixed_hex(value, length=64, label="SHA-256")


def _git_commit(value: str) -> str:
    return _fixed_hex(value, length=40, label="Git commit")


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
    # Keep this in lock-step with BenchmarkConfig.patch_size.  A 64-pixel
    # context diluted thin crack signals and is intentionally not the default.
    feasibility.add_argument("--patch-size", type=_positive_int, default=32)
    feasibility.add_argument("--overlays-per-modality", type=int, default=2)

    datasets = subparsers.add_parser(
        "datasets", help="list public resources and license caveats"
    )
    datasets.add_argument(
        "--json", action="store_true", help="emit JSON instead of a table"
    )
    datasets.add_argument("--registry", type=Path, default=None)

    tests = subparsers.add_parser(
        "test", help="run the complete pytest suite from a source checkout"
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
    frozen_benchmark.add_argument("--calibration-manifest", type=Path, required=True)
    frozen_benchmark.add_argument("--locked-test-manifest", type=Path, required=True)
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
    frozen_benchmark.add_argument("--min-valid-token-fraction", type=float, default=0.5)
    frozen_benchmark.add_argument("--neighbors", type=_positive_int, default=1)
    frozen_benchmark.add_argument("--calibration-quantile", type=float, default=0.995)
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

    public_fold = subparsers.add_parser(
        "public-fold-benchmark",
        help="run the slide-grouped real H&E fold benchmark from Zenodo 21493260",
    )
    public_fold.add_argument("--dataset-root", type=Path, required=True)
    public_fold.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "classical_fold",
            "foundation_patchknn",
            "foundation_linear_probe",
            "dinov2_patchknn",
            "dinov2_linear_probe",
        ),
        default=("classical_fold", "dinov2_patchknn", "dinov2_linear_probe"),
    )
    public_fold.add_argument(
        "--foundation-encoder",
        choices=PUBLIC_FOLD_ENCODER_NAMES,
        default="dinov2-hf",
        help=(
            "frozen encoder for foundation_* methods; the default preserves the "
            "original Hugging Face DINOv2 benchmark"
        ),
    )
    public_fold.add_argument(
        "--revision",
        default="ed25f3a31f01632728cabb09d1542f84ab7b0056",
        help="exact immutable DINOv2 revision used by foundation methods",
    )
    public_fold.add_argument("--model-id", default="facebook/dinov2-small")
    public_fold.add_argument(
        "--cache-dir", type=Path, default=Path("models/hf_home/hub")
    )
    public_fold.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    public_fold.add_argument("--allow-download", action="store_true")
    public_fold.add_argument(
        "--hibou-weights",
        type=Path,
        default=None,
        help="explicit official local Hibou-B .pth (required for hibou-b-local)",
    )
    public_fold.add_argument(
        "--hibou-source",
        type=Path,
        default=None,
        help="explicit clean official HistAI/hibou checkout (required for hibou-b-local)",
    )
    public_fold.add_argument(
        "--hibou-weights-sha256",
        type=_sha256,
        default=None,
        help="optional expected SHA-256 lock for the local Hibou-B .pth",
    )
    public_fold.add_argument(
        "--hibou-source-commit",
        type=_git_commit,
        default=None,
        help="optional expected 40-character commit lock for the Hibou checkout",
    )
    public_fold.add_argument(
        "--siglip2-snapshot",
        type=Path,
        default=None,
        help=(
            "explicit hash-locked local google/siglip2-base-patch16-224 snapshot "
            "(required for siglip2-base-local)"
        ),
    )
    public_fold.add_argument("--max-dimension", type=_positive_int, default=896)
    public_fold.add_argument("--tile-size", type=_positive_int, default=224)
    public_fold.add_argument("--tile-stride", type=_positive_int, default=224)
    public_fold.add_argument("--batch-size", type=_positive_int, default=8)
    public_fold.add_argument(
        "--max-reference-tokens", type=_positive_int, default=4_096
    )
    public_fold.add_argument(
        "--max-probe-tokens-per-class", type=_positive_int, default=8_192
    )
    public_fold.add_argument(
        "--probe-max-iterations",
        type=_positive_int,
        default=100,
        help="explicit L-BFGS iteration ceiling; non-convergence fails the run",
    )
    public_fold.add_argument("--neighbors", type=_positive_int, default=3)
    public_fold.add_argument("--bootstrap-resamples", type=int, default=1_000)
    public_fold.add_argument(
        "--limit-slides-per-stratum-per-split", type=_positive_int, default=None
    )
    public_fold.add_argument(
        "--exclude-empty-positive-masks",
        action="store_true",
        help=(
            "retain the two released fold-presence labels but explicitly exclude "
            "their empty masks from localization evidence"
        ),
    )
    public_fold.add_argument(
        "--no-asset-hashes",
        action="store_true",
        help="skip per-asset SHA-256 hashing for a development smoke only",
    )
    public_fold.add_argument(
        "--skip-dimension-validation",
        action="store_true",
        help="skip complete image/mask decode validation for a development smoke only",
    )
    public_fold.add_argument("--output-json", type=Path, required=True)
    public_fold.add_argument(
        "--json", action="store_true", help="also emit the complete report to stdout"
    )

    multiplex_proxy = subparsers.add_parser(
        "multiplex-proxy-benchmark",
        help=(
            "run the checksum-locked real COMET/CosMx synthetic-spike proxy; "
            "this is not real-artifact efficacy"
        ),
    )
    multiplex_proxy.add_argument("--comet-dir", type=Path, required=True)
    multiplex_proxy.add_argument(
        "--cosmx-dir",
        type=Path,
        nargs="+",
        required=True,
        help="one or more locked public CosMx cohort directories",
    )
    multiplex_proxy.add_argument(
        "--mode",
        choices=("logo-cv", "locked-split"),
        default="logo-cv",
        help="leave-one-source-group-out is the stronger default proxy protocol",
    )
    multiplex_proxy.add_argument("--max-dimension", type=_positive_int, default=896)
    multiplex_proxy.add_argument("--seed", type=int, default=29)
    multiplex_proxy.add_argument(
        "--group-bootstrap-resamples", type=_positive_int, default=2_000
    )
    multiplex_proxy.add_argument("--group-bootstrap-seed", type=int, default=20_260_826)
    multiplex_proxy.add_argument("--output-json", type=Path, required=True)
    multiplex_proxy.add_argument(
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
    tests_root = root / "tests"
    if not tests_root.is_dir():
        print(
            "The test command requires a source checkout containing tests/. "
            "Installed wheels intentionally do not bundle the test suite.",
            file=sys.stderr,
        )
        return 2
    if importlib.util.find_spec("pytest") is None:
        print(
            "The complete test suite requires pytest. Install the project's "
            "development extra with: python -m pip install -e '.[dev]'",
            file=sys.stderr,
        )
        return 2
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.pop("PYTEST_PLUGINS", None)
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            f"python_files={pattern}",
            str(tests_root),
        ],
        cwd=root,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


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
        print(f"Eligible methods: {', '.join(report.eligible_method_ids) or 'none'}")
        for issue in report.issues:
            print(
                f"{issue.severity.upper()} [{issue.code}] {issue.path}: {issue.message}"
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


def _run_public_fold_benchmark(args: argparse.Namespace) -> int:
    from .public_fold_benchmark import (
        PublicFoldBenchmarkConfig,
        run_public_fold_benchmark,
    )

    methods = tuple(args.methods)
    config = PublicFoldBenchmarkConfig(
        methods=methods,
        max_dimension=args.max_dimension,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        encoder_batch_size=args.batch_size,
        max_reference_tokens=args.max_reference_tokens,
        max_probe_tokens_per_class=args.max_probe_tokens_per_class,
        probe_max_iterations=args.probe_max_iterations,
        patchknn_neighbors=args.neighbors,
        bootstrap_resamples=args.bootstrap_resamples,
        limit_slides_per_stratum_per_split=(args.limit_slides_per_stratum_per_split),
        empty_positive_mask_policy=(
            "exclude_localization" if args.exclude_empty_positive_masks else "error"
        ),
        hash_assets=not args.no_asset_hashes,
        validate_asset_dimensions=not args.skip_dimension_validation,
        strict_public_v1=not (args.no_asset_hashes or args.skip_dimension_validation),
    )
    foundation_methods = {
        "foundation_patchknn",
        "foundation_linear_probe",
        "dinov2_patchknn",
        "dinov2_linear_probe",
    }
    encoder = None
    model_identity: dict[str, Any] | None = None
    if set(methods) & foundation_methods:
        from .public_fold_providers import build_public_fold_encoder

        built_provider = build_public_fold_encoder(
            args.foundation_encoder,
            args,
            methods,
        )
        encoder = built_provider.encoder
        model_identity = dict(built_provider.model_identity)
    run_provenance = _public_fold_run_provenance(config, model_identity)
    report = run_public_fold_benchmark(
        args.dataset_root,
        encoder=encoder,
        config=config,
        run_provenance=run_provenance,
    )
    if model_identity is not None:
        report["model_identity"] = model_identity
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(rendered, encoding="utf-8")
    if args.json:
        print(rendered, end="")
    else:
        scores = ", ".join(
            (
                f"{method} all-field-micro-Dice="
                f"{value['locked_test']['pixel_all_fields_micro']['dice']:.3f}"
            )
            for method, value in report["methods"].items()
        )
        evidence_status = (
            "report-eligible" if report["report_eligible"] else "nonreportable"
        )
        print(
            f"Public real H&E fold benchmark complete ({evidence_status}): {scores}; "
            f"report={args.output_json}"
        )
    return 0


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


def _run_multiplex_proxy_benchmark(args: argparse.Namespace) -> int:
    from .multiplex_proxy_benchmark import (
        MultiplexProxyConfig,
        load_public_multiplex_fields,
        run_multiplex_proxy_benchmark,
        run_multiplex_proxy_cross_validation,
        write_multiplex_proxy_report,
    )

    fields = load_public_multiplex_fields(
        comet_dir=args.comet_dir,
        cosmx_dir=tuple(args.cosmx_dir),
        max_dimension=args.max_dimension,
    )
    config = MultiplexProxyConfig(
        seed=args.seed,
        group_bootstrap_resamples=args.group_bootstrap_resamples,
        group_bootstrap_seed=args.group_bootstrap_seed,
    )
    report = (
        run_multiplex_proxy_cross_validation(fields, config)
        if args.mode == "logo-cv"
        else run_multiplex_proxy_benchmark(fields, config)
    )
    destination = write_multiplex_proxy_report(report, args.output_json)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    else:
        modalities = sorted({field.modality for field in fields})
        print(
            "Real-background multiplex proxy complete: "
            f"mode={args.mode}; fields={len(fields)}; modalities={','.join(modalities)}; "
            f"report={destination}; efficacy_claim=false"
        )
    return 0


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
    if args.command == "public-fold-benchmark":
        return _run_public_fold_benchmark(args)
    if args.command == "multiplex-proxy-benchmark":
        return _run_multiplex_proxy_benchmark(args)
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
