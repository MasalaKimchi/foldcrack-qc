"""Regenerate hash-selected H&E qualitative overlays for the frozen report.

This is intentionally a separate, explicit inference step.  The hardened JSON
artifacts retain scalar outcomes and confusion counts, but not spatial score
maps or fitted linear-probe parameters.  The script therefore refits only the
deterministic shallow readout on the exact frozen fit split, reuses the locked
thresholds, and fails closed unless every selected outcome matches the stored
artifact before writing any report-facing overlay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from foldcrack_qc.public_fold_benchmark import (
    PublicFoldBenchmarkConfig,
    _fit_foundation_models,
    _load_image,
    _pixel_counts,
    _score_classical,
    _score_foundation,
    build_public_fold_splits,
    load_public_fold_dataset,
)
from foldcrack_qc.public_fold_providers import (
    build_public_fold_encoder,
)

REPORT_DATE = "2026-08-27"
HIBOU_ARTIFACT = REPO / "artifacts/public_fold/hibou_hardened_v1_2.json"
CLASSICAL_ARTIFACT = REPO / "artifacts/public_fold/classical_hardened_v1_2.json"
DATASET_ROOT = REPO / "data/public/histology_tissue_fold_v1"
OVERLAY_DIR = HERE / "qualitative_cache"
CHECKS_PATH = HERE / "qualitative_checks.json"

ORGAN_ORDER = ("Brain", "Kidney", "Liver", "Small_Intestine", "Testis")
EXPECTED_DISPLAY_ORDER = (
    "Brain_Fold_-20260410140426972.jpg",
    "Kidney__Fold_-20260409155008912.jpg",
    "Liver__Fold_-20260406110823972.jpg",
    "Small_Intestine_Fold_-20260413150334899.jpg",
    "Testis__Fold_-20260406101118662.jpg",
    "Brain_Clean-20260423152633332.jpg",
    "Kidney_Clean-20260420151005412.jpg",
)

# Color-blind-conscious RGB contours.  Line style and direct labels repeat the
# semantics so interpretation does not depend on color alone.
REFERENCE_COLOR = (0, 121, 107)
CLASSICAL_COLOR = (213, 94, 0)
HIBOU_COLOR = (148, 0, 211)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _outcomes(report: dict[str, Any], method: str) -> dict[str, dict[str, Any]]:
    rows = report["methods"][method]["locked_test_outcomes"]
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        field_key = str(row["field_key"])
        if field_key in indexed:
            raise AssertionError(f"duplicate frozen outcome field key: {field_key}")
        indexed[field_key] = row
    return indexed


def _presence_category(row: dict[str, Any]) -> str:
    label = int(row["label"])
    prediction = int(row["image_prediction"])
    if label == 1 and prediction == 1:
        return "TP"
    if label == 1:
        return "FN"
    if prediction == 1:
        return "FP"
    return "TN"


def _pixel_dice(row: dict[str, Any]) -> float | None:
    if int(row["label"]) == 0:
        return None
    denominator = 2 * int(row["tp"]) + int(row["fp"]) + int(row["fn"])
    if denominator == 0:
        return None
    return 2.0 * int(row["tp"]) / denominator


def _select_fields(
    records_by_name: dict[str, Any],
    classical_rows: dict[str, dict[str, Any]],
    hibou_rows: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    selected: set[str] = set()
    for organ in ORGAN_ORDER:
        candidates = [
            record
            for record in records_by_name.values()
            if record.organ == organ and record.is_fold
        ]
        if not candidates:
            raise AssertionError(f"no positive locked-test record for {organ}")
        selected.add(
            min(candidates, key=lambda record: record.image_sha256).image_filename
        )

    for rows in (classical_rows, hibou_rows):
        for category in ("FP", "FN"):
            candidates = [
                records_by_name[name]
                for name, row in rows.items()
                if _presence_category(row) == category
            ]
            if not candidates:
                raise AssertionError(f"no {category} case available for frozen method")
            selected.add(
                min(candidates, key=lambda record: record.image_sha256).image_filename
            )

    expected = set(EXPECTED_DISPLAY_ORDER)
    if selected != expected:
        raise AssertionError(
            "hash-selected qualitative cohort changed: "
            f"expected={sorted(expected)}, observed={sorted(selected)}"
        )
    return EXPECTED_DISPLAY_ORDER


def _assert_split_identity(
    report: dict[str, Any], splits: dict[str, tuple[Any, ...]]
) -> None:
    for role, records in splits.items():
        observed = [record.manifest_entry() for record in records]
        expected = report["splits"][role]["manifest"]
        if observed != expected:
            raise AssertionError(f"{role} split manifest differs from frozen artifact")
        if _canonical_sha256(observed) != report["splits"][role]["manifest_sha256"]:
            raise AssertionError(f"{role} split manifest hash mismatch")


def _provider_args(
    report: dict[str, Any],
    device: str,
    hibou_source: Path | None,
    hibou_weights: Path | None,
) -> SimpleNamespace:
    identity = report["model_identity"]
    return SimpleNamespace(
        allow_download=False,
        cache_dir=REPO / ".cache",
        device=device,
        hibou_source=(
            hibou_source.resolve()
            if hibou_source is not None
            else Path(identity["source"]["path"])
        ),
        hibou_source_commit=str(identity["source"]["commit"]),
        hibou_weights=(
            hibou_weights.resolve()
            if hibou_weights is not None
            else Path(identity["weights"]["path"])
        ),
        hibou_weights_sha256=str(identity["weights"]["sha256"]),
        model_id="",
        revision="",
        siglip2_snapshot=None,
    )


def _assert_configuration_compatibility(
    hibou_report: dict[str, Any], classical_report: dict[str, Any]
) -> None:
    """Require identical outcome-relevant data, split, and scoring settings."""

    ignored = {"methods", "probe_max_iterations"}
    hibou = hibou_report["configuration"]
    classical = classical_report["configuration"]
    for key in sorted(set(hibou) | set(classical)):
        if key in ignored:
            continue
        if hibou.get(key) != classical.get(key):
            raise AssertionError(
                f"classical and Hibou outcome-relevant configuration differs: {key}"
            )


def _git_output(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return result.stdout


def _generation_provenance(hibou_report: dict[str, Any], device: str) -> dict[str, Any]:
    runtime_sources = [
        SRC / "foldcrack_qc/public_fold_benchmark.py",
        SRC / "foldcrack_qc/public_fold_providers.py",
    ]
    runtime_diff = _git_output(
        "diff",
        "--binary",
        "HEAD",
        "--",
        *(str(path.relative_to(REPO)) for path in runtime_sources),
    )
    return {
        "script_sha256": _sha256_file(Path(__file__).resolve()),
        "repository_commit": _git_output("rev-parse", "HEAD").decode().strip(),
        "tracked_runtime_diff_sha256": hashlib.sha256(runtime_diff).hexdigest(),
        "tracked_runtime_dirty": bool(runtime_diff),
        "runtime_source_sha256": {
            str(path.relative_to(REPO)): _sha256_file(path) for path in runtime_sources
        },
        "environment": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "dependencies": {
                "numpy": np.__version__,
                "opencv": cv2.__version__,
                "scipy": __import__("scipy").__version__,
                "torch": __import__("torch").__version__,
            },
            "device": device,
        },
        "frozen_run_provenance": hibou_report["run_provenance"],
    }


def _assert_training_identity(
    observed: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    exact_fields = (
        "probe_negative_tokens_seen",
        "probe_positive_tokens_seen",
        "probe_tokens_stored_per_class",
    )
    for field in exact_fields:
        if observed[field] != expected[field]:
            raise AssertionError(f"foundation training field differs: {field}")
    observed_opt = observed["probe_optimization"]
    expected_opt = expected["probe_optimization"]
    for field in ("success", "status", "iterations", "function_evaluations"):
        if observed_opt[field] != expected_opt[field]:
            raise AssertionError(f"probe optimizer field differs: {field}")
    if not math.isclose(
        float(observed_opt["final_loss"]),
        float(expected_opt["final_loss"]),
        rel_tol=1e-8,
        abs_tol=1e-10,
    ):
        raise AssertionError("probe optimizer final loss differs from frozen artifact")
    return {
        "token_counts_match": True,
        "optimizer_status_iterations_match": True,
        "final_loss_match": True,
        "final_loss": float(observed_opt["final_loss"]),
        "iterations": int(observed_opt["iterations"]),
        "observed_training_statistics": {
            "probe_negative_tokens_seen": int(observed["probe_negative_tokens_seen"]),
            "probe_positive_tokens_seen": int(observed["probe_positive_tokens_seen"]),
            "probe_tokens_stored_per_class": observed[
                "probe_tokens_stored_per_class"
            ],
            "probe_optimization": {
                "success": bool(observed_opt["success"]),
                "status": int(observed_opt["status"]),
                "iterations": int(observed_opt["iterations"]),
                "function_evaluations": int(observed_opt["function_evaluations"]),
                "final_loss": float(observed_opt["final_loss"]),
            },
        },
        "patchknn_fit_skipped_as_unused": True,
    }


def _assert_outcome(
    scored: Any,
    stored: dict[str, Any],
    pixel_threshold: float,
    image_threshold: float,
) -> dict[str, Any]:
    observed_counts = _pixel_counts(scored, pixel_threshold)
    expected_counts = tuple(
        int(stored[key]) for key in ("tp", "fp", "fn", "tn", "n_valid")
    )
    if observed_counts != expected_counts:
        raise AssertionError(
            f"{stored['field_key']}: pixel outcome mismatch; "
            f"expected={expected_counts}, observed={observed_counts}"
        )
    observed_score = float(scored.image_score)
    stored_score = float(stored["image_score"])
    score_tolerance = max(
        2e-7,
        2e-6 * max(abs(observed_score), abs(stored_score)),
    )
    score_difference = abs(observed_score - stored_score)
    if score_difference > score_tolerance:
        raise AssertionError(
            f"{stored['field_key']}: image score differs from frozen artifact"
        )
    observed_prediction = int(scored.image_score >= image_threshold)
    if observed_prediction != int(stored["image_prediction"]):
        raise AssertionError(
            f"{stored['field_key']}: image prediction differs from frozen artifact"
        )
    return {
        "counts_match": True,
        "image_score_within_tolerance": True,
        "image_prediction_match": True,
        "observed_counts": {
            key: value
            for key, value in zip(
                ("tp", "fp", "fn", "tn", "n_valid"),
                observed_counts,
                strict=True,
            )
        },
        "observed_image_score": observed_score,
        "stored_image_score": stored_score,
        "image_score_absolute_difference": score_difference,
        "image_score_absolute_tolerance": score_tolerance,
    }


def _contour_line_mask(
    mask: np.ndarray,
    thickness: int,
    dash: tuple[int, int] | None,
) -> np.ndarray:
    """Rasterize solid or regularly dashed contours into an antialiased mask."""

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    line_mask = np.zeros(mask.shape, dtype=np.uint8)
    if not contours:
        return line_mask
    if dash is None:
        cv2.drawContours(
            line_mask,
            contours,
            -1,
            255,
            thickness,
            lineType=cv2.LINE_AA,
        )
        return line_mask

    on_pixels, off_pixels = dash
    period = on_pixels + off_pixels
    for contour in contours:
        points = contour[:, 0, :]
        if len(points) == 1:
            cv2.circle(
                line_mask,
                tuple(int(value) for value in points[0]),
                max(1, thickness // 2),
                255,
                -1,
                lineType=cv2.LINE_AA,
            )
            continue
        for point_index, point in enumerate(points):
            if point_index % period >= on_pixels:
                continue
            next_point = points[(point_index + 1) % len(points)]
            cv2.line(
                line_mask,
                tuple(int(value) for value in point),
                tuple(int(value) for value in next_point),
                255,
                thickness,
                lineType=cv2.LINE_AA,
            )
    return line_mask


def _blend_contours(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
    *,
    dash: tuple[int, int] | None = None,
    opacity: float = 1.0,
) -> None:
    line_mask = _contour_line_mask(mask, thickness, dash)
    alpha = (opacity * line_mask.astype(np.float32) / 255.0)[..., None]
    color_array = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    blended = image.astype(np.float32) * (1.0 - alpha) + color_array * alpha
    np.copyto(image, np.clip(blended, 0, 255).astype(np.uint8))


def _render_overlay(
    image: np.ndarray,
    target: np.ndarray,
    classical_prediction: np.ndarray,
    hibou_prediction: np.ndarray,
) -> np.ndarray:
    overlay = np.ascontiguousarray(image.copy())
    # Predictions are deliberately lighter and non-solid so tissue morphology
    # remains inspectable even when a detector produces many small components.
    _blend_contours(
        overlay,
        classical_prediction,
        CLASSICAL_COLOR,
        2,
        dash=(6, 4),
        opacity=0.72,
    )
    _blend_contours(
        overlay,
        hibou_prediction,
        HIBOU_COLOR,
        2,
        dash=(1, 3),
        opacity=0.88,
    )
    # Draw the reference last with a white halo so coincident predictions do
    # not hide the ground-truth boundary.
    _blend_contours(overlay, target, (255, 255, 255), 7, opacity=0.92)
    _blend_contours(overlay, target, REFERENCE_COLOR, 3, opacity=1.0)
    return overlay


def _encode_png(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(
        ".png",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_PNG_COMPRESSION, 6],
    )
    if not ok:
        raise OSError("could not encode qualitative overlay")
    return encoded.tobytes()


def _publish_bundle(
    output: dict[str, Any], encoded_overlays: list[tuple[Path, bytes]]
) -> None:
    """Publish content-addressed images, then atomically swap the JSON pointer."""

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    checks_temp: Path | None = None
    try:
        for final_path, payload in encoded_overlays:
            with tempfile.NamedTemporaryFile(
                dir=OVERLAY_DIR,
                prefix=".staging-",
                suffix=".png",
                delete=False,
            ) as handle:
                handle.write(payload)
                staged.append((Path(handle.name), final_path))
        for temporary_path, final_path in staged:
            os.replace(temporary_path, final_path)

        serialized = (
            json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        with tempfile.NamedTemporaryFile(
            dir=HERE,
            prefix=".qualitative-checks-",
            suffix=".json",
            delete=False,
        ) as handle:
            handle.write(serialized)
            checks_temp = Path(handle.name)
        os.replace(checks_temp, CHECKS_PATH)
        checks_temp = None
        referenced = {path.resolve() for path, _ in encoded_overlays}
        for prior_path in OVERLAY_DIR.glob("case*.png"):
            if (
                prior_path.is_file()
                and not prior_path.is_symlink()
                and prior_path.resolve() not in referenced
            ):
                prior_path.unlink()
    finally:
        for temporary_path, _ in staged:
            temporary_path.unlink(missing_ok=True)
        if checks_temp is not None:
            checks_temp.unlink(missing_ok=True)


def _method_thresholds(report: dict[str, Any], method: str) -> tuple[float, float]:
    thresholds = report["methods"][method]["thresholds"]
    return (
        float(thresholds["pixel_localization"]["value"]),
        float(thresholds["image_presence"]["value"]),
    )


def run(
    device: str,
    hibou_source: Path | None = None,
    hibou_weights: Path | None = None,
) -> dict[str, Any]:
    hibou_report = _load_json(HIBOU_ARTIFACT)
    classical_report = _load_json(CLASSICAL_ARTIFACT)
    _assert_configuration_compatibility(hibou_report, classical_report)
    config = PublicFoldBenchmarkConfig(**hibou_report["configuration"])
    # Spatial regeneration needs only the supervised readout.  Skipping the
    # unused PatchKNN memory bank materially reduces time and peak memory while
    # leaving every scored linear-probe outcome unchanged.
    linear_config = replace(config, methods=("foundation_linear_probe",))
    dataset = load_public_fold_dataset(
        DATASET_ROOT,
        strict_public_v1=config.strict_public_v1,
        validate_asset_dimensions=config.validate_asset_dimensions,
        hash_assets=config.hash_assets,
        empty_positive_mask_policy=config.empty_positive_mask_policy,
    )
    splits = build_public_fold_splits(dataset.records, config)
    _assert_split_identity(hibou_report, splits)
    _assert_split_identity(classical_report, splits)

    test_records = {record.image_filename: record for record in splits["locked_test"]}
    classical_rows = _outcomes(classical_report, "classical_fold")
    hibou_rows = _outcomes(hibou_report, "foundation_linear_probe")
    selected = _select_fields(test_records, classical_rows, hibou_rows)

    built = build_public_fold_encoder(
        "hibou-b-local",
        _provider_args(hibou_report, device, hibou_source, hibou_weights),
        linear_config.methods,
    )
    if built.model_identity["resolved_device"] != device:
        raise AssertionError(
            f"requested {device}, resolved {built.model_identity['resolved_device']}"
        )
    knn, probe, training = _fit_foundation_models(
        splits["fit"], built.encoder, linear_config
    )
    if knn is not None:
        raise AssertionError("unused PatchKNN bank was unexpectedly fitted")
    if probe is None:
        raise AssertionError("linear probe was not fitted")
    training_check = _assert_training_identity(
        training, hibou_report["foundation_training"]
    )

    classical_pixel, classical_image = _method_thresholds(
        classical_report, "classical_fold"
    )
    hibou_pixel, hibou_image = _method_thresholds(
        hibou_report, "foundation_linear_probe"
    )
    cases: list[dict[str, Any]] = []
    pending_overlays: list[tuple[int, str, np.ndarray]] = []
    for index, field_key in enumerate(selected, start=1):
        record = test_records[field_key]
        loaded = _load_image(record, config.max_dimension)
        classical_scored = _score_classical(record, config)
        hibou_scored = _score_foundation(
            record, built.encoder, linear_config, None, probe
        )["foundation_linear_probe"]
        classical_check = _assert_outcome(
            classical_scored,
            classical_rows[field_key],
            classical_pixel,
            classical_image,
        )
        hibou_check = _assert_outcome(
            hibou_scored,
            hibou_rows[field_key],
            hibou_pixel,
            hibou_image,
        )
        pending_overlays.append(
            (
                index,
                field_key,
                _render_overlay(
                    loaded.image,
                    loaded.target,
                    classical_scored.score >= classical_pixel,
                    hibou_scored.score >= hibou_pixel,
                ),
            )
        )
        cases.append(
            {
                "display_order": index,
                "field_key": field_key,
                "organ": record.organ,
                "label": int(record.is_fold),
                "source_slide_id": record.slide_id,
                "image_sha256": record.image_sha256,
                "mask_sha256": record.mask_sha256,
                "classical": {
                    "presence_category": _presence_category(classical_rows[field_key]),
                    "pixel_dice": _pixel_dice(classical_rows[field_key]),
                    **classical_check,
                },
                "hibou_linear": {
                    "presence_category": _presence_category(hibou_rows[field_key]),
                    "pixel_dice": _pixel_dice(hibou_rows[field_key]),
                    **hibou_check,
                },
            }
        )

    # All scientific and identity checks above must pass before any new
    # report-facing image is encoded or written. Content-addressed filenames
    # plus an atomic JSON-pointer replacement prevent partial bundles from
    # becoming visible to the report generator.
    encoded_overlays: list[tuple[Path, bytes]] = []
    for case, (index, field_key, overlay) in zip(cases, pending_overlays, strict=True):
        payload = _encode_png(overlay)
        overlay_sha256 = hashlib.sha256(payload).hexdigest()
        filename = f"case{index}_{Path(field_key).stem}_{overlay_sha256[:12]}.png"
        final_path = OVERLAY_DIR / filename
        case["overlay_path"] = str(final_path.relative_to(HERE))
        case["overlay_sha256"] = overlay_sha256
        encoded_overlays.append((final_path, payload))

    output = {
        "schema_version": "he-qualitative-audit-1.1",
        "report_date": REPORT_DATE,
        "status": "passed",
        "selection_rule": (
            "Algorithmically selected without manual image review during this "
            "audit: SHA-256-minimum fold-positive locked-test field within each "
            "organ, plus the SHA-256-minimum presence FP and FN separately for "
            "each compared method; union deduplicated. Whole "
            "896×504 px analysis fields isotropically resized from 3840×2160; "
            "no manual crops."
        ),
        "n_cases": len(cases),
        "all_outcomes_match_frozen_artifacts": True,
        "dataset": {
            "root": str(DATASET_ROOT.relative_to(REPO)),
            "assignment_manifest_sha256": hibou_report["split_protocol"][
                "assignment_manifest_sha256"
            ],
            "release_identity_sha256": hibou_report["dataset"]["release_identity"][
                "canonical_identity_sha256"
            ],
        },
        "configuration": {
            "hibou_sha256": hibou_report["configuration_sha256"],
            "classical_sha256": classical_report["configuration_sha256"],
            "outcome_relevant_compatibility_checked": True,
        },
        "execution_override": {
            "foundation_methods": list(linear_config.methods),
            "patchknn_fit_skipped_as_unused": True,
            "outcome_relevant_settings_unchanged": True,
        },
        "model": {
            "id": hibou_report["model_identity"]["id"],
            "device": device,
            "weights_sha256": hibou_report["model_identity"]["weights"]["sha256"],
            "source_commit": hibou_report["model_identity"]["source"]["commit"],
            "encoder_frozen": True,
            "shallow_readout_refit_required": True,
            "calibration_rerun": False,
        },
        "thresholds": {
            "classical": {
                "pixel": classical_pixel,
                "presence": classical_image,
            },
            "hibou_linear": {"pixel": hibou_pixel, "presence": hibou_image},
        },
        "overlay_encoding": {
            "reference": {
                "color_rgb": list(REFERENCE_COLOR),
                "line_style": "solid",
                "line_width_px": 3,
                "white_halo_width_px": 7,
                "draw_order": "last",
            },
            "classical": {
                "color_rgb": list(CLASSICAL_COLOR),
                "line_style": "dashed",
                "dash_on_off_px": [6, 4],
                "line_width_px": 2,
                "opacity": 0.72,
            },
            "hibou_linear": {
                "color_rgb": list(HIBOU_COLOR),
                "line_style": "dotted",
                "dash_on_off_px": [1, 3],
                "line_width_px": 2,
                "opacity": 0.88,
            },
        },
        "training_identity_check": training_check,
        "overlay_generation_provenance": _generation_provenance(hibou_report, device),
        "input_hashes": {
            str(HIBOU_ARTIFACT.relative_to(REPO)): _sha256_file(HIBOU_ARTIFACT),
            str(CLASSICAL_ARTIFACT.relative_to(REPO)): _sha256_file(CLASSICAL_ARTIFACT),
        },
        "cases": cases,
    }
    _publish_bundle(output, encoded_overlays)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("mps", "cpu"),
        default="mps",
        help="Device used for exact frozen-encoder regeneration (default: mps).",
    )
    parser.add_argument(
        "--hibou-source",
        type=Path,
        default=None,
        help="Optional relocated audited Hibou source checkout.",
    )
    parser.add_argument(
        "--hibou-weights",
        type=Path,
        default=None,
        help="Optional relocated hash-locked Hibou weights file.",
    )
    args = parser.parse_args()
    result = run(args.device, args.hibou_source, args.hibou_weights)
    print(
        json.dumps(
            {
                "status": result["status"],
                "n_cases": result["n_cases"],
                "checks": str(CHECKS_PATH),
                "overlay_dir": str(OVERLAY_DIR),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
