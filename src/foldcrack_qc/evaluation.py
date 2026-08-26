"""Dependency-light evaluation utilities for fold and crack artifact masks.

The module deliberately operates on in-memory two-dimensional arrays.  Reading
whole-slide images, choosing pyramid levels, and registering modality channels
belong to the caller so that evaluation remains deterministic and auditable.

All public results are JSON-safe dictionaries.  Undefined rates are represented
by ``None``; the only intentional empty-set exception is a pair of empty masks,
which is treated as a correct negative and receives overlap scores of 1.0.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


__all__ = [
    "aggregate_by_slide",
    "aggregate_results",
    "bootstrap_ci_by_cluster",
    "bootstrap_ci_by_sample",
    "boundary_metrics",
    "build_report",
    "burden_metrics",
    "centerline_metrics",
    "confusion_counts",
    "evaluate_sample",
    "froc_counts",
    "instance_metrics",
    "pixel_metrics",
    "report_to_markdown",
    "runtime_summary",
    "skeletonize_binary",
    "write_csv_report",
    "write_json_report",
    "write_markdown_report",
]


_COUNT_KEYS = ("tp", "fp", "fn", "tn")
_DEFAULT_BOOTSTRAP_METRICS = (
    "pixel.dice",
    "pixel.iou",
    "boundary.surface_dice",
    "centerline.centerline_f1",
    "instance.f1",
    "burden.absolute_fraction_error",
)


def _as_2d_bool(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array; got {array.shape}")
    if not (array.dtype == np.bool_ or np.issubdtype(array.dtype, np.number)):
        raise TypeError(f"{name} must be a numeric or boolean binary mask")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} cannot be complex-valued")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if array.dtype != np.bool_:
        unique = np.unique(array)
        allowed = {0.0, 1.0}
        if np.issubdtype(array.dtype, np.integer):
            allowed.add(255.0)
        if any(float(item) not in allowed for item in unique):
            raise ValueError(
                f"{name} must be binary (boolean, 0/1, or integer 0/255); "
                f"found {len(unique)} distinct encoded values"
            )
    return array.astype(bool, copy=False)


def _prepare_masks(
    target: Any,
    prediction: Any,
    valid_mask: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_mask = _as_2d_bool(target, "target")
    prediction_mask = _as_2d_bool(prediction, "prediction")
    if target_mask.shape != prediction_mask.shape:
        raise ValueError(
            "target and prediction must have the same shape; "
            f"got {target_mask.shape} and {prediction_mask.shape}"
        )
    if valid_mask is None:
        valid = np.ones(target_mask.shape, dtype=bool)
    else:
        valid = _as_2d_bool(valid_mask, "valid_mask")
        if valid.shape != target_mask.shape:
            raise ValueError(
                "valid_mask must have the same shape as target; "
                f"got {valid.shape} and {target_mask.shape}"
            )
    if not np.any(valid):
        raise ValueError("valid_mask excludes every pixel")
    return target_mask & valid, prediction_mask & valid, valid


def _normalize_spacing(spacing: Sequence[float] | None) -> tuple[float, float]:
    if spacing is None:
        return (1.0, 1.0)
    if len(spacing) != 2:
        raise ValueError("spacing must contain (row_spacing, column_spacing)")
    normalized = (float(spacing[0]), float(spacing[1]))
    if not all(math.isfinite(item) and item > 0 for item in normalized):
        raise ValueError("spacing values must be finite and greater than zero")
    return normalized


def _nonnegative_finite(value: float, name: str) -> float:
    normalized = float(value)
    if not math.isfinite(normalized) or normalized < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    empty_value: float | None = None,
) -> float | None:
    if denominator == 0:
        return empty_value
    return float(numerator / denominator)


def _harmonic_mean(first: float, second: float) -> float:
    if first + second == 0:
        return 0.0
    return float(2.0 * first * second / (first + second))


def confusion_counts(
    target: Any,
    prediction: Any,
    valid_mask: Any | None = None,
) -> dict[str, int]:
    """Return binary pixel confusion counts within ``valid_mask``."""

    target_mask, prediction_mask, valid = _prepare_masks(target, prediction, valid_mask)
    tp = int(np.count_nonzero(target_mask & prediction_mask))
    fp = int(np.count_nonzero(~target_mask & prediction_mask & valid))
    fn = int(np.count_nonzero(target_mask & ~prediction_mask & valid))
    tn = int(np.count_nonzero(~target_mask & ~prediction_mask & valid))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_valid": int(valid.sum())}


def _pixel_metrics_from_counts(counts: Mapping[str, int]) -> dict[str, int | float]:
    tp = int(counts["tp"])
    fp = int(counts["fp"])
    fn = int(counts["fn"])
    tn = int(counts["tn"])
    n_valid = int(counts.get("n_valid", tp + fp + fn + tn))
    if n_valid <= 0:
        raise ValueError("pixel counts must contain at least one valid pixel")

    true_positive_pixels = tp + fn
    predicted_positive_pixels = tp + fp
    precision = _safe_divide(
        tp,
        predicted_positive_pixels,
        empty_value=1.0 if true_positive_pixels == 0 else 0.0,
    )
    recall = _safe_divide(tp, true_positive_pixels, empty_value=1.0)
    specificity = _safe_divide(tn, tn + fp, empty_value=1.0)
    dice = _safe_divide(2 * tp, 2 * tp + fp + fn, empty_value=1.0)
    iou = _safe_divide(tp, tp + fp + fn, empty_value=1.0)
    accuracy = _safe_divide(tp + tn, n_valid, empty_value=1.0)
    fpr = _safe_divide(fp, fp + tn, empty_value=0.0)
    fnr = _safe_divide(fn, fn + tp, empty_value=0.0)

    mcc_denominator = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    if mcc_denominator:
        mcc = float((tp * tn - fp * fn) / mcc_denominator)
    else:
        mcc = 1.0 if fp == 0 and fn == 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_valid": n_valid,
        "true_positive_pixels": true_positive_pixels,
        "predicted_positive_pixels": predicted_positive_pixels,
        "prevalence": float(true_positive_pixels / n_valid),
        "precision": float(precision),
        "recall": float(recall),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "dice": float(dice),
        "iou": float(iou),
        "false_positive_rate": float(fpr),
        "false_negative_rate": float(fnr),
        "mcc": mcc,
    }


def pixel_metrics(
    target: Any,
    prediction: Any,
    valid_mask: Any | None = None,
) -> dict[str, int | float]:
    """Compute binary segmentation metrics and their source confusion counts."""

    return _pixel_metrics_from_counts(confusion_counts(target, prediction, valid_mask))


def _binary_boundary(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    boundary = mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)

    # Do not create scored boundaries where an ignore region truncates an object.
    # ``border_value=1`` preserves legitimate boundaries at the image perimeter.
    valid_interior = ndimage.binary_erosion(valid, structure=structure, border_value=1)
    ignore_interface = valid & ~valid_interior
    return boundary & ~ignore_interface


def _distance_match_metrics(
    target_points: np.ndarray,
    predicted_points: np.ndarray,
    *,
    tolerance: float,
    spacing: tuple[float, float],
    name_prefix: str,
) -> dict[str, int | float | None]:
    n_target = int(target_points.sum())
    n_predicted = int(predicted_points.sum())
    if n_target == 0 and n_predicted == 0:
        return {
            f"n_target_{name_prefix}_pixels": 0,
            f"n_predicted_{name_prefix}_pixels": 0,
            f"{name_prefix}_precision": 1.0,
            f"{name_prefix}_recall": 1.0,
            f"{name_prefix}_f1": 1.0,
            "assd": 0.0,
            "hd95": 0.0,
        }

    predicted_distances: np.ndarray | None = None
    target_distances: np.ndarray | None = None
    matched_predicted = 0
    matched_target = 0
    if n_target:
        distance_to_target = ndimage.distance_transform_edt(
            ~target_points, sampling=spacing
        )
        predicted_distances = distance_to_target[predicted_points]
        matched_predicted = int(np.count_nonzero(predicted_distances <= tolerance))
    if n_predicted:
        distance_to_prediction = ndimage.distance_transform_edt(
            ~predicted_points, sampling=spacing
        )
        target_distances = distance_to_prediction[target_points]
        matched_target = int(np.count_nonzero(target_distances <= tolerance))

    precision = float(matched_predicted / n_predicted) if n_predicted else 1.0
    recall = float(matched_target / n_target) if n_target else 1.0
    f1 = _harmonic_mean(precision, recall)

    if predicted_distances is None or target_distances is None:
        assd: float | None = None
        hd95: float | None = None
    else:
        all_distances = np.concatenate((predicted_distances, target_distances))
        assd = float(all_distances.mean())
        hd95 = float(np.percentile(all_distances, 95))

    return {
        f"n_target_{name_prefix}_pixels": n_target,
        f"n_predicted_{name_prefix}_pixels": n_predicted,
        f"{name_prefix}_precision": precision,
        f"{name_prefix}_recall": recall,
        f"{name_prefix}_f1": f1,
        "assd": assd,
        "hd95": hd95,
    }


def boundary_metrics(
    target: Any,
    prediction: Any,
    *,
    tolerance: float = 2.0,
    spacing: Sequence[float] | None = None,
    valid_mask: Any | None = None,
) -> dict[str, int | float | None]:
    """Compute tolerance-aware boundary precision/recall/F1 and distances.

    ``tolerance`` is expressed in the same units as ``spacing``.  With no
    spacing, both are interpreted as pixels.
    """

    normalized_tolerance = _nonnegative_finite(tolerance, "tolerance")
    normalized_spacing = _normalize_spacing(spacing)
    target_mask, prediction_mask, valid = _prepare_masks(target, prediction, valid_mask)
    target_boundary = _binary_boundary(target_mask, valid)
    predicted_boundary = _binary_boundary(prediction_mask, valid)
    matched = _distance_match_metrics(
        target_boundary,
        predicted_boundary,
        tolerance=normalized_tolerance,
        spacing=normalized_spacing,
        name_prefix="boundary",
    )
    n_target = int(matched["n_target_boundary_pixels"])
    n_prediction = int(matched["n_predicted_boundary_pixels"])
    if n_target == 0 and n_prediction == 0:
        surface_dice = 1.0
    elif n_target == 0 or n_prediction == 0:
        surface_dice = 0.0
    else:
        distance_to_target = ndimage.distance_transform_edt(
            ~target_boundary, sampling=normalized_spacing
        )
        distance_to_prediction = ndimage.distance_transform_edt(
            ~predicted_boundary, sampling=normalized_spacing
        )
        matched_prediction = int(
            np.count_nonzero(
                distance_to_target[predicted_boundary] <= normalized_tolerance
            )
        )
        matched_target = int(
            np.count_nonzero(
                distance_to_prediction[target_boundary] <= normalized_tolerance
            )
        )
        surface_dice = float(
            (matched_prediction + matched_target) / (n_prediction + n_target)
        )
    return {
        "tolerance": normalized_tolerance,
        "spacing": list(normalized_spacing),
        "surface_dice": surface_dice,
        "precision": matched["boundary_precision"],
        "recall": matched["boundary_recall"],
        "f1": matched["boundary_f1"],
        **matched,
    }


def skeletonize_binary(mask: Any) -> np.ndarray:
    """Return a one-pixel-wide skeleton using deterministic Zhang-Suen thinning.

    This implementation avoids the endpoint branches produced by a basic
    morphological skeleton on narrow rectangles, a common geometry for cracks.
    """

    image = _as_2d_bool(mask, "mask").copy()
    if not np.any(image):
        return image

    while True:
        changed = False
        for step in (0, 1):
            padded = np.pad(image, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
            neighbor_count = sum(neighbor.astype(np.uint8) for neighbor in neighbors)
            transitions = np.zeros(image.shape, dtype=np.uint8)
            for current, following in zip(neighbors, neighbors[1:] + neighbors[:1]):
                transitions += (~current & following).astype(np.uint8)

            removable = (
                image
                & (neighbor_count >= 2)
                & (neighbor_count <= 6)
                & (transitions == 1)
            )
            if step == 0:
                removable &= ~(p2 & p4 & p6)
                removable &= ~(p4 & p6 & p8)
            else:
                removable &= ~(p2 & p4 & p8)
                removable &= ~(p2 & p6 & p8)
            if np.any(removable):
                image[removable] = False
                changed = True
        if not changed:
            return image


def centerline_metrics(
    target: Any,
    prediction: Any,
    *,
    tolerance: float = 2.0,
    spacing: Sequence[float] | None = None,
    valid_mask: Any | None = None,
) -> dict[str, int | float | None]:
    """Compute topology-aware metrics suited to thin tears and cracks.

    ``centerline_f1`` matches skeleton pixels within the requested tolerance.
    ``cldice`` is the topology precision/sensitivity harmonic mean computed by
    testing each mask's centerline against the opposite full mask.
    """

    normalized_tolerance = _nonnegative_finite(tolerance, "tolerance")
    normalized_spacing = _normalize_spacing(spacing)
    target_mask, prediction_mask, _ = _prepare_masks(target, prediction, valid_mask)
    target_skeleton = skeletonize_binary(target_mask)
    predicted_skeleton = skeletonize_binary(prediction_mask)
    matched = _distance_match_metrics(
        target_skeleton,
        predicted_skeleton,
        tolerance=normalized_tolerance,
        spacing=normalized_spacing,
        name_prefix="centerline",
    )

    n_target = int(target_skeleton.sum())
    n_prediction = int(predicted_skeleton.sum())
    if n_target == 0 and n_prediction == 0:
        topology_precision = topology_sensitivity = cldice = 1.0
    elif n_target == 0 or n_prediction == 0:
        topology_precision = 0.0 if n_prediction else 1.0
        topology_sensitivity = 0.0 if n_target else 1.0
        cldice = 0.0
    else:
        topology_precision = float(
            np.count_nonzero(predicted_skeleton & target_mask) / n_prediction
        )
        topology_sensitivity = float(
            np.count_nonzero(target_skeleton & prediction_mask) / n_target
        )
        cldice = _harmonic_mean(topology_precision, topology_sensitivity)

    return {
        "tolerance": normalized_tolerance,
        "spacing": list(normalized_spacing),
        "cldice": cldice,
        "topology_precision": topology_precision,
        "topology_sensitivity": topology_sensitivity,
        "precision": matched["centerline_precision"],
        "recall": matched["centerline_recall"],
        "f1": matched["centerline_f1"],
        **matched,
    }


def _component_labels(
    mask: np.ndarray,
    *,
    min_area: int,
    connectivity: int,
) -> tuple[np.ndarray, np.ndarray]:
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if int(min_area) != min_area or min_area < 1:
        raise ValueError("min_area must be a positive integer")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == 4 else 2)
    labels, _ = ndimage.label(mask, structure=structure)
    areas = np.bincount(labels.ravel())
    keep = np.flatnonzero(areas >= int(min_area))
    keep = keep[keep != 0]
    mapping = np.zeros(len(areas), dtype=np.int32)
    mapping[keep] = np.arange(1, len(keep) + 1, dtype=np.int32)
    relabeled = mapping[labels]
    kept_areas = areas[keep].astype(np.int64, copy=False)
    return relabeled, kept_areas


def instance_metrics(
    target: Any,
    prediction: Any,
    *,
    iou_threshold: float = 0.1,
    min_area: int = 1,
    connectivity: int = 8,
    valid_mask: Any | None = None,
) -> dict[str, Any]:
    """Match connected components one-to-one and return detection counts.

    Matching first maximizes the number of pairs meeting ``iou_threshold`` and
    then maximizes total IoU, avoiding a high-IoU pair from reducing cardinality.
    """

    threshold = float(iou_threshold)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("iou_threshold must be between zero and one")
    target_mask, prediction_mask, _ = _prepare_masks(target, prediction, valid_mask)
    target_labels, target_areas = _component_labels(
        target_mask, min_area=min_area, connectivity=connectivity
    )
    prediction_labels, prediction_areas = _component_labels(
        prediction_mask, min_area=min_area, connectivity=connectivity
    )
    n_target = int(len(target_areas))
    n_prediction = int(len(prediction_areas))

    matches: list[dict[str, int | float]] = []
    if n_target and n_prediction:
        combined = target_labels.astype(np.int64) * (n_prediction + 1)
        combined += prediction_labels.astype(np.int64)
        intersections = np.bincount(
            combined.ravel(), minlength=(n_target + 1) * (n_prediction + 1)
        ).reshape(n_target + 1, n_prediction + 1)[1:, 1:]
        unions = target_areas[:, None] + prediction_areas[None, :] - intersections
        ious = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions > 0,
        )
        # Even at a requested threshold of zero, disconnected objects are not a
        # match merely because their mathematical IoU equals the threshold.
        valid_pairs = (ious >= threshold) & (intersections > 0)
        # A cardinality bonus dominates IoU while retaining deterministic IoU
        # tie-breaking within maximum-cardinality assignments.
        benefit = np.where(valid_pairs, 1_000_000.0 + ious, 0.0)
        target_indices, prediction_indices = linear_sum_assignment(-benefit)
        for target_index, prediction_index in zip(
            target_indices.tolist(), prediction_indices.tolist()
        ):
            if not valid_pairs[target_index, prediction_index]:
                continue
            matches.append(
                {
                    "target_label": target_index + 1,
                    "prediction_label": prediction_index + 1,
                    "iou": float(ious[target_index, prediction_index]),
                    "target_area": int(target_areas[target_index]),
                    "prediction_area": int(prediction_areas[prediction_index]),
                }
            )

    tp = len(matches)
    fp = n_prediction - tp
    fn = n_target - tp
    precision = float(tp / (tp + fp)) if tp + fp else 1.0
    recall = float(tp / (tp + fn)) if tp + fn else 1.0
    f1 = _harmonic_mean(precision, recall)
    if matches:
        mean_iou: float | None = float(np.mean([item["iou"] for item in matches]))
    elif n_target == 0 and n_prediction == 0:
        mean_iou = 1.0
    else:
        mean_iou = None

    return {
        "iou_threshold": threshold,
        "min_area": int(min_area),
        "connectivity": int(connectivity),
        "n_true_instances": n_target,
        "n_predicted_instances": n_prediction,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positive_instances": tp,
        "false_positive_instances": fp,
        "false_negative_instances": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
        "matches": matches,
    }


def froc_counts(
    target: Any,
    score_map: Any,
    thresholds: Iterable[float],
    *,
    iou_threshold: float = 0.1,
    min_area: int = 1,
    connectivity: int = 8,
    valid_mask: Any | None = None,
) -> list[dict[str, int | float]]:
    """Return per-threshold connected-component counts for a FROC curve.

    Each returned row represents one sample, so ``fp_per_sample`` equals its
    false-positive instance count.  Dataset callers should sum counts across
    samples and divide false positives by the number of samples or slides.
    """

    target_mask = _as_2d_bool(target, "target")
    scores = np.asarray(score_map, dtype=float)
    if scores.ndim != 2 or scores.shape != target_mask.shape:
        raise ValueError("score_map must be two-dimensional and match target")
    valid = (
        np.ones(target_mask.shape, dtype=bool)
        if valid_mask is None
        else _as_2d_bool(valid_mask, "valid_mask")
    )
    if valid.shape != target_mask.shape:
        raise ValueError("valid_mask must match target")
    if not np.any(valid):
        raise ValueError("valid_mask excludes every pixel")
    if not np.all(np.isfinite(scores[valid])):
        raise ValueError("score_map contains non-finite values in the valid region")

    rows: list[dict[str, int | float]] = []
    for raw_threshold in thresholds:
        threshold = float(raw_threshold)
        if not math.isfinite(threshold):
            raise ValueError("thresholds must be finite")
        metrics = instance_metrics(
            target_mask,
            scores >= threshold,
            iou_threshold=iou_threshold,
            min_area=min_area,
            connectivity=connectivity,
            valid_mask=valid,
        )
        rows.append(
            {
                "threshold": threshold,
                "tp": int(metrics["tp"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
                "n_true_instances": int(metrics["n_true_instances"]),
                "n_predicted_instances": int(metrics["n_predicted_instances"]),
                "sensitivity": float(metrics["recall"]),
                "fp_per_sample": float(metrics["fp"]),
            }
        )
    return rows


def burden_metrics(
    target: Any,
    prediction: Any,
    *,
    valid_mask: Any | None = None,
    pixel_area: float = 1.0,
) -> dict[str, int | float | None]:
    """Measure error in artifact-positive area and fraction of evaluable tissue."""

    normalized_pixel_area = float(pixel_area)
    if not math.isfinite(normalized_pixel_area) or normalized_pixel_area <= 0:
        raise ValueError("pixel_area must be finite and greater than zero")
    target_mask, prediction_mask, valid = _prepare_masks(target, prediction, valid_mask)
    true_pixels = int(target_mask.sum())
    predicted_pixels = int(prediction_mask.sum())
    n_valid = int(valid.sum())
    true_fraction = float(true_pixels / n_valid)
    predicted_fraction = float(predicted_pixels / n_valid)
    signed_fraction_error = predicted_fraction - true_fraction
    absolute_fraction_error = abs(signed_fraction_error)
    if true_fraction == 0:
        relative_absolute_error = 0.0 if predicted_fraction == 0 else None
    else:
        relative_absolute_error = float(absolute_fraction_error / true_fraction)
    true_area = true_pixels * normalized_pixel_area
    predicted_area = predicted_pixels * normalized_pixel_area
    signed_area_error = predicted_area - true_area
    return {
        "n_valid": n_valid,
        "true_pixels": true_pixels,
        "predicted_pixels": predicted_pixels,
        "pixel_area": normalized_pixel_area,
        "true_fraction": true_fraction,
        "predicted_fraction": predicted_fraction,
        "signed_fraction_error": signed_fraction_error,
        "absolute_fraction_error": absolute_fraction_error,
        "relative_absolute_error": relative_absolute_error,
        "true_area": true_area,
        "predicted_area": predicted_area,
        "signed_area_error": signed_area_error,
        "absolute_area_error": abs(signed_area_error),
    }


def evaluate_sample(
    target: Any,
    prediction: Any | None = None,
    *,
    score_map: Any | None = None,
    threshold: float = 0.5,
    sample_id: str | None = None,
    slide_id: str | None = None,
    modality: str | None = None,
    valid_mask: Any | None = None,
    spacing: Sequence[float] | None = None,
    boundary_tolerance: float = 2.0,
    centerline_tolerance: float = 2.0,
    instance_iou_threshold: float = 0.1,
    min_instance_area: int = 1,
    min_instance_area_physical: float | None = None,
    connectivity: int = 8,
    pixel_area: float | None = None,
    runtime_seconds: float | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate one sample or registered slide region.

    If ``prediction`` is omitted, it is derived as ``score_map >= threshold``.
    The threshold is recorded even when an explicit prediction is supplied so
    experiment manifests remain comparable.
    """

    normalized_threshold = float(threshold)
    if not math.isfinite(normalized_threshold):
        raise ValueError("threshold must be finite")
    target_array = _as_2d_bool(target, "target")
    if prediction is None:
        if score_map is None:
            raise ValueError("provide prediction or score_map")
        scores = np.asarray(score_map, dtype=float)
        if scores.shape != target_array.shape or scores.ndim != 2:
            raise ValueError("score_map must be two-dimensional and match target")
        if not np.all(np.isfinite(scores)):
            raise ValueError("score_map contains non-finite values")
        prediction_array = scores >= normalized_threshold
    else:
        prediction_array = _as_2d_bool(prediction, "prediction")

    normalized_spacing = _normalize_spacing(spacing)
    derived_pixel_area = float(normalized_spacing[0] * normalized_spacing[1])
    normalized_pixel_area = (
        derived_pixel_area if pixel_area is None else float(pixel_area)
    )
    if not math.isfinite(normalized_pixel_area) or normalized_pixel_area <= 0:
        raise ValueError("pixel_area must be finite and greater than zero")
    effective_min_instance_area = int(min_instance_area)
    if min_instance_area_physical is not None:
        physical_area = float(min_instance_area_physical)
        if not math.isfinite(physical_area) or physical_area <= 0:
            raise ValueError("min_instance_area_physical must be finite and positive")
        effective_min_instance_area = max(
            1, int(math.ceil(physical_area / normalized_pixel_area))
        )
    normalized_runtime = None
    if runtime_seconds is not None:
        normalized_runtime = _nonnegative_finite(runtime_seconds, "runtime_seconds")

    result = {
        "sample_id": None if sample_id is None else str(sample_id),
        "slide_id": None if slide_id is None else str(slide_id),
        "modality": None if modality is None else str(modality),
        "threshold": normalized_threshold,
        "runtime_seconds": normalized_runtime,
        "config": {
            "spacing": list(normalized_spacing),
            "boundary_tolerance": float(boundary_tolerance),
            "centerline_tolerance": float(centerline_tolerance),
            "instance_iou_threshold": float(instance_iou_threshold),
            "min_instance_area": effective_min_instance_area,
            "min_instance_area_physical": (
                None
                if min_instance_area_physical is None
                else float(min_instance_area_physical)
            ),
            "connectivity": int(connectivity),
            "pixel_area": normalized_pixel_area,
        },
        "pixel": pixel_metrics(target_array, prediction_array, valid_mask),
        "boundary": boundary_metrics(
            target_array,
            prediction_array,
            tolerance=boundary_tolerance,
            spacing=normalized_spacing,
            valid_mask=valid_mask,
        ),
        "centerline": centerline_metrics(
            target_array,
            prediction_array,
            tolerance=centerline_tolerance,
            spacing=normalized_spacing,
            valid_mask=valid_mask,
        ),
        "instance": instance_metrics(
            target_array,
            prediction_array,
            iou_threshold=instance_iou_threshold,
            min_area=effective_min_instance_area,
            connectivity=connectivity,
            valid_mask=valid_mask,
        ),
        "burden": burden_metrics(
            target_array,
            prediction_array,
            valid_mask=valid_mask,
            pixel_area=normalized_pixel_area,
        ),
        "metadata": dict(metadata or {}),
    }
    return _to_builtin(result)


def _mean_finite(values: Iterable[Any]) -> float | None:
    numeric: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            converted = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(converted):
            numeric.append(converted)
    return float(np.mean(numeric)) if numeric else None


def _median_finite(values: Iterable[Any]) -> float | None:
    numeric = [
        float(value)
        for value in values
        if value is not None
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ]
    return float(np.median(numeric)) if numeric else None


def _aggregate_instance_counts(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item["instance"]["tp"]) for item in results)
    fp = sum(int(item["instance"]["fp"]) for item in results)
    fn = sum(int(item["instance"]["fn"]) for item in results)
    n_true = sum(int(item["instance"]["n_true_instances"]) for item in results)
    n_predicted = sum(
        int(item["instance"]["n_predicted_instances"]) for item in results
    )
    precision = float(tp / (tp + fp)) if tp + fp else 1.0
    recall = float(tp / (tp + fn)) if tp + fn else 1.0
    matched_ious = [
        match["iou"]
        for item in results
        for match in item["instance"].get("matches", [])
    ]
    return {
        "n_true_instances": n_true,
        "n_predicted_instances": n_predicted,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "f1": _harmonic_mean(precision, recall),
        "mean_matched_iou": _mean_finite(matched_ious),
        "false_positives_per_sample": float(fp / len(results)),
    }


def _aggregate_burden(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n_valid = sum(int(item["burden"]["n_valid"]) for item in results)
    true_pixels = sum(int(item["burden"]["true_pixels"]) for item in results)
    predicted_pixels = sum(int(item["burden"]["predicted_pixels"]) for item in results)
    true_fraction = float(true_pixels / n_valid)
    predicted_fraction = float(predicted_pixels / n_valid)
    signed_fraction_error = predicted_fraction - true_fraction
    return {
        "n_valid": n_valid,
        "true_pixels": true_pixels,
        "predicted_pixels": predicted_pixels,
        "true_fraction": true_fraction,
        "predicted_fraction": predicted_fraction,
        "signed_fraction_error": signed_fraction_error,
        "absolute_fraction_error": abs(signed_fraction_error),
        "mean_sample_absolute_fraction_error": _mean_finite(
            item["burden"].get("absolute_fraction_error") for item in results
        ),
        "median_sample_absolute_fraction_error": _median_finite(
            item["burden"].get("absolute_fraction_error") for item in results
        ),
        "total_true_area": float(
            sum(float(item["burden"]["true_area"]) for item in results)
        ),
        "total_predicted_area": float(
            sum(float(item["burden"]["predicted_area"]) for item in results)
        ),
    }


def _aggregate_one(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("results must not be empty")
    counts = {
        key: sum(int(item["pixel"][key]) for item in results) for key in _COUNT_KEYS
    }
    counts["n_valid"] = sum(int(item["pixel"]["n_valid"]) for item in results)
    boundary_keys = (
        "surface_dice",
        "boundary_precision",
        "boundary_recall",
        "boundary_f1",
        "assd",
        "hd95",
    )
    centerline_keys = (
        "centerline_precision",
        "centerline_recall",
        "centerline_f1",
        "cldice",
        "topology_precision",
        "topology_sensitivity",
        "assd",
        "hd95",
    )
    slide_ids = {item.get("slide_id") for item in results if item.get("slide_id")}
    modalities = sorted(
        {str(item["modality"]) for item in results if item.get("modality") is not None}
    )
    boundary = {
        key: _mean_finite(item["boundary"].get(key) for item in results)
        for key in boundary_keys
    }
    boundary.update(
        {
            "precision": boundary["boundary_precision"],
            "recall": boundary["boundary_recall"],
            "f1": boundary["boundary_f1"],
        }
    )
    centerline = {
        key: _mean_finite(item["centerline"].get(key) for item in results)
        for key in centerline_keys
    }
    centerline.update(
        {
            "precision": centerline["centerline_precision"],
            "recall": centerline["centerline_recall"],
            "f1": centerline["centerline_f1"],
        }
    )
    return {
        "n_samples": len(results),
        "n_slides": len(slide_ids),
        "modalities": modalities,
        "pixel": _pixel_metrics_from_counts(counts),
        "boundary": boundary,
        "centerline": centerline,
        "instance": _aggregate_instance_counts(results),
        "burden": _aggregate_burden(results),
        "runtime": runtime_summary(results),
    }


def aggregate_results(
    results: Sequence[Mapping[str, Any]],
    *,
    group_by: str | Sequence[str] | None = None,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Pool counts and summarize sample evaluations.

    Pixel and instance rates are recomputed from pooled counts.  Boundary and
    centerline values are sample-macro means.  With ``group_by=None`` a single
    summary dictionary is returned; otherwise a list of group summaries is
    returned in deterministic key order.
    """

    normalized_results = list(results)
    if not normalized_results:
        raise ValueError("results must not be empty")
    if group_by is None:
        return _to_builtin(_aggregate_one(normalized_results))
    keys = (group_by,) if isinstance(group_by, str) else tuple(group_by)
    if not keys:
        raise ValueError("group_by must contain at least one key")
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for result in normalized_results:
        grouped[tuple(result.get(key) for key in keys)].append(result)

    summaries: list[dict[str, Any]] = []
    for values, group_results in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        summary = _aggregate_one(group_results)
        summary["group"] = dict(zip(keys, values))
        summaries.append(_to_builtin(summary))
    return summaries


def aggregate_by_slide(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return pooled metrics for each ``slide_id``."""

    return aggregate_results(results, group_by="slide_id")  # type: ignore[return-value]


def _nested_value(item: Mapping[str, Any], path: str) -> Any:
    value: Any = item
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"metric path {path!r} is missing at {part!r}")
        value = value[part]
    return value


def bootstrap_ci_by_sample(
    results: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, int | float | str]:
    """Percentile bootstrap CI for a sample-level scalar metric."""

    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    normalized_confidence = float(confidence)
    if not 0 < normalized_confidence < 1:
        raise ValueError("confidence must be between zero and one")
    values: list[float] = []
    for item in results:
        value = _nested_value(item, metric)
        if value is None or isinstance(value, bool):
            continue
        converted = float(value)
        if math.isfinite(converted):
            values.append(converted)
    if not values:
        raise ValueError(f"metric {metric!r} has no finite sample values")

    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(int(n_resamples), len(array)), replace=True)
    statistics = samples.mean(axis=1)
    alpha = (1.0 - normalized_confidence) / 2.0
    return {
        "metric": metric,
        "estimate": float(array.mean()),
        "lower": float(np.quantile(statistics, alpha)),
        "upper": float(np.quantile(statistics, 1.0 - alpha)),
        "confidence": normalized_confidence,
        "n_samples": len(array),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def bootstrap_ci_by_cluster(
    results: Sequence[Mapping[str, Any]],
    metric: str,
    *,
    cluster_key: str = "slide_id",
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict[str, int | float | str]:
    """Bootstrap the same pooled aggregate statistic shown in the report.

    Independent clusters—not tiles or repeated predictions—are sampled with
    replacement. After each draw, :func:`aggregate_results` is recomputed and
    ``metric`` is extracted from that aggregate. A dotted key such as
    ``metadata.patient_id`` can be used when slides share a higher-level unit.
    Missing cluster identifiers fall back to ``sample_id`` and finally row ID.
    """

    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    normalized_confidence = float(confidence)
    if not 0 < normalized_confidence < 1:
        raise ValueError("confidence must be between zero and one")
    normalized_results = list(results)
    if not normalized_results:
        raise ValueError("results must not be empty")

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, item in enumerate(normalized_results):
        try:
            raw_cluster = _nested_value(item, cluster_key)
        except KeyError:
            raw_cluster = None
        if raw_cluster is None or str(raw_cluster).strip() == "":
            raw_cluster = item.get("sample_id")
        cluster = (
            f"__row_{index}"
            if raw_cluster is None or str(raw_cluster).strip() == ""
            else str(raw_cluster)
        )
        grouped[cluster].append(item)

    clusters = tuple(sorted(grouped))
    estimate_summary = aggregate_results(normalized_results)
    assert isinstance(estimate_summary, Mapping)
    estimate_value = _nested_value(estimate_summary, metric)
    if estimate_value is None or not math.isfinite(float(estimate_value)):
        raise ValueError(f"aggregate metric {metric!r} is not finite")

    rng = np.random.default_rng(seed)
    statistics = np.empty(int(n_resamples), dtype=float)
    for resample_index in range(int(n_resamples)):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        resampled = [item for cluster in chosen for item in grouped[str(cluster)]]
        summary = aggregate_results(resampled)
        assert isinstance(summary, Mapping)
        value = _nested_value(summary, metric)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"bootstrap aggregate metric {metric!r} is not finite")
        statistics[resample_index] = float(value)

    alpha = (1.0 - normalized_confidence) / 2.0
    return {
        "metric": metric,
        "estimate": float(estimate_value),
        "lower": float(np.quantile(statistics, alpha)),
        "upper": float(np.quantile(statistics, 1.0 - alpha)),
        "confidence": normalized_confidence,
        "cluster_key": cluster_key,
        "n_clusters": len(clusters),
        "n_samples": len(normalized_results),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def runtime_summary(
    results_or_seconds: Iterable[Mapping[str, Any] | float | int | None],
) -> dict[str, int | float | None]:
    """Summarize non-negative sample runtimes."""

    seconds: list[float] = []
    for item in results_or_seconds:
        value = item.get("runtime_seconds") if isinstance(item, Mapping) else item
        if value is None:
            continue
        converted = _nonnegative_finite(float(value), "runtime_seconds")
        seconds.append(converted)
    if not seconds:
        return {
            "n": 0,
            "total_seconds": 0.0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
            "samples_per_second": None,
        }
    array = np.asarray(seconds, dtype=float)
    total = float(array.sum())
    return {
        "n": len(seconds),
        "total_seconds": total,
        "mean_seconds": float(array.mean()),
        "median_seconds": float(np.median(array)),
        "p95_seconds": float(np.percentile(array, 95)),
        "min_seconds": float(array.min()),
        "max_seconds": float(array.max()),
        "samples_per_second": float(len(seconds) / total) if total > 0 else None,
    }


def build_report(
    results: Sequence[Mapping[str, Any]],
    *,
    bootstrap_metrics: Sequence[str] = _DEFAULT_BOOTSTRAP_METRICS,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
    bootstrap_cluster_key: str = "slide_id",
) -> dict[str, Any]:
    """Build a versioned report with CIs for its displayed aggregates."""

    normalized_results = [_to_builtin(dict(item)) for item in results]
    if not normalized_results:
        raise ValueError("results must not be empty")
    bootstrap: dict[str, Any] = {}
    for index, metric in enumerate(bootstrap_metrics):
        bootstrap[metric] = bootstrap_ci_by_cluster(
            normalized_results,
            metric,
            cluster_key=bootstrap_cluster_key,
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed + index,
        )
    return {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": aggregate_results(normalized_results),
        "by_slide": aggregate_by_slide(normalized_results),
        "by_modality": aggregate_results(normalized_results, group_by="modality"),
        "bootstrap": bootstrap,
        "samples": normalized_results,
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _flatten_mapping(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_mapping(item, name))
        elif isinstance(item, (list, tuple)):
            flattened[name] = json.dumps(_to_builtin(item), sort_keys=True)
        else:
            flattened[name] = _to_builtin(item)
    return flattened


def write_json_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write a report as strict JSON and return the output path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(_to_builtin(report), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


def write_csv_report(results: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Write one flattened CSV row per sample evaluation."""

    rows = [_flatten_mapping(item) for item in results]
    if not rows:
        raise ValueError("results must not be empty")
    fieldnames = sorted({key for row in rows for key in row})
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return destination


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Render a compact human-readable summary from ``build_report`` output."""

    summary = report["summary"]
    lines = [
        "# Fold/Crack QC Evaluation",
        "",
        f"Generated: {report.get('generated_at', 'unknown')}",
        "",
        f"Samples: {summary['n_samples']}  ",
        f"Slides: {summary['n_slides']}  ",
        f"Modalities: {', '.join(summary.get('modalities', [])) or 'unspecified'}",
        "",
        "## Overall metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metric_rows = (
        ("Pixel Dice", summary["pixel"].get("dice")),
        ("Pixel IoU", summary["pixel"].get("iou")),
        ("Pixel sensitivity", summary["pixel"].get("recall")),
        ("Pixel specificity", summary["pixel"].get("specificity")),
        ("Surface Dice", summary["boundary"].get("surface_dice")),
        ("Centerline F1", summary["centerline"].get("centerline_f1")),
        ("clDice", summary["centerline"].get("cldice")),
        ("Instance F1", summary["instance"].get("f1")),
        ("FP instances/sample", summary["instance"].get("false_positives_per_sample")),
        (
            "Artifact burden absolute error",
            summary["burden"].get("absolute_fraction_error"),
        ),
        ("Mean runtime (s)", summary["runtime"].get("mean_seconds")),
    )
    lines.extend(f"| {name} | {_format_metric(value)} |" for name, value in metric_rows)

    bootstrap = report.get("bootstrap", {})
    if bootstrap:
        lines.extend(
            [
                "",
                "## Sample-bootstrap confidence intervals",
                "",
                "| Metric | Estimate | Lower | Upper |",
                "|---|---:|---:|---:|",
            ]
        )
        for metric, interval in bootstrap.items():
            lines.append(
                "| "
                + " | ".join(
                    (
                        metric,
                        _format_metric(interval.get("estimate")),
                        _format_metric(interval.get("lower")),
                        _format_metric(interval.get("upper")),
                    )
                )
                + " |"
            )

    by_modality = report.get("by_modality", [])
    if by_modality:
        lines.extend(
            [
                "",
                "## By modality",
                "",
                "| Modality | Samples | Dice | Surface Dice | Centerline F1 | Instance F1 |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for item in by_modality:
            modality = item.get("group", {}).get("modality") or "unspecified"
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(modality),
                        str(item["n_samples"]),
                        _format_metric(item["pixel"].get("dice")),
                        _format_metric(item["boundary"].get("surface_dice")),
                        _format_metric(item["centerline"].get("centerline_f1")),
                        _format_metric(item["instance"].get("f1")),
                    )
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def write_markdown_report(report: Mapping[str, Any], path: str | Path) -> Path:
    """Write ``report_to_markdown(report)`` and return the output path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report_to_markdown(report), encoding="utf-8")
    return destination
