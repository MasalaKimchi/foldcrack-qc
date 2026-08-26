"""End-to-end, dependency-light engineering feasibility benchmark.

This module deliberately separates *software feasibility* from *scientific
validation*.  Synthetic images provide exact masks for testing coordinate,
detector, metric, reporting, and modality-adapter behavior.  They cannot prove
performance on Merck data; the generated report says so prominently.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .detectors import HybridQCDetector, classical_candidate_masks
from .evaluation import (
    aggregate_results,
    bootstrap_ci_by_cluster,
    build_report,
    evaluate_sample,
    write_csv_report,
    write_json_report,
)
from .features import extract_patch_feature_table
from .schema import ChannelRole, Modality, QCSample
from .synthetic import generate_synthetic_sample
from .visualization import write_overlay


@dataclass(frozen=True)
class BenchmarkConfig:
    output_dir: Path = Path("artifacts/feasibility")
    samples_per_modality: int = 12
    clean_samples_per_modality: int = 6
    image_size: tuple[int, int] = (384, 384)
    seed: int = 17
    patch_size: int = 64
    overlays_per_modality: int = 2
    bootstrap_resamples: int = 400
    min_component_area_um2: float = 4.0
    crack_neighborhood_radius_um: float = 2.0
    fold_morphology_radius_um: float = 0.5
    evaluation_min_instance_area_um2: float = 2.0

    def __post_init__(self) -> None:
        if self.samples_per_modality < len(SYNTHETIC_SCENARIOS):
            raise ValueError(
                "samples_per_modality must be at least "
                f"{len(SYNTHETIC_SCENARIOS)} so every synthetic scenario is covered"
            )
        if self.clean_samples_per_modality <= 0:
            raise ValueError("sample counts must be positive")
        if min(self.image_size) < 128:
            raise ValueError("image_size must be at least 128x128")
        if self.patch_size <= 8 or self.patch_size > min(self.image_size):
            raise ValueError("patch_size must be >8 and fit within the image")
        physical_values = (
            self.min_component_area_um2,
            self.crack_neighborhood_radius_um,
            self.fold_morphology_radius_um,
            self.evaluation_min_instance_area_um2,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in physical_values):
            raise ValueError(
                "physical detector/evaluation geometry must be finite and positive"
            )


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    include_fold: bool
    include_crack: bool
    include_hard_negatives: bool


SYNTHETIC_SCENARIOS: tuple[SyntheticScenario, ...] = (
    SyntheticScenario("clean", False, False, False),
    SyntheticScenario("hard_negative_only", False, False, True),
    SyntheticScenario("fold_only", True, False, False),
    SyntheticScenario("crack_only", False, True, False),
    SyntheticScenario("both", True, True, False),
)

EXPECTED_OUTPUT_GROUPS: tuple[tuple[str, str, str], ...] = (
    ("classical", "fold", "full_structural"),
    ("classical", "crack", "full_structural"),
    ("classical", "artifact", "full_structural"),
    ("clean_reference_anomaly", "artifact", "full_structural"),
    ("hybrid", "artifact", "full_structural"),
    ("clean_reference_anomaly", "artifact", "minimal_structural"),
    ("hybrid", "artifact", "minimal_structural"),
)

ANOMALY_DECISION_THRESHOLD = 0.75


class StructuralViewError(ValueError):
    """Raised when semantic roles cannot support the requested detector view."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(value), handle, indent=2, sort_keys=True, allow_nan=False)
    temporary.replace(path)


def _environment() -> dict[str, Any]:
    torch_available = importlib.util.find_spec("torch") is not None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "torch_available": torch_available,
        "timm_available": importlib.util.find_spec("timm") is not None,
        "tifffile_available": importlib.util.find_spec("tifffile") is not None,
        "optional_dinov2": {
            "backend_available": torch_available,
            "backend": "PyTorch Hub or an injected approved model",
            "scope": "H&E comparator only",
            "benchmarked_in_this_run": False,
        },
    }


_ROLE_ORDER = {
    role: index
    for index, role in enumerate(
        (
            ChannelRole.BRIGHTFIELD_RED,
            ChannelRole.BRIGHTFIELD_GREEN,
            ChannelRole.BRIGHTFIELD_BLUE,
            ChannelRole.NUCLEAR,
            ChannelRole.AUTOFLUORESCENCE,
            ChannelRole.MEMBRANE,
            ChannelRole.CYTOPLASM,
            ChannelRole.IMMUNE,
            ChannelRole.MORPHOLOGY,
        )
    )
}


def _ordered_indices_for_roles(
    sample: QCSample, eligible_roles: set[ChannelRole]
) -> tuple[int, ...]:
    records = [
        (role, sample.image.channel_names[index].casefold(), index)
        for index, role in enumerate(sample.image.channel_roles)
        if role in eligible_roles
    ]
    records.sort(key=lambda item: (_ROLE_ORDER[item[0]], item[1]))
    return tuple(item[2] for item in records)


def _full_structural_indices(sample: QCSample) -> tuple[int, ...]:
    """Resolve a panel-order-invariant structural view from adapter roles.

    Marker-only and unknown channels are never silently treated as structural.
    A missing required role is an abstention condition for real data and a hard
    failure in this engineering benchmark.
    """

    roles = set(sample.image.channel_roles)
    if sample.modality is Modality.HE:
        rgb_roles = {
            ChannelRole.BRIGHTFIELD_RED,
            ChannelRole.BRIGHTFIELD_GREEN,
            ChannelRole.BRIGHTFIELD_BLUE,
        }
        if rgb_roles.issubset(roles):
            return _ordered_indices_for_roles(sample, rgb_roles)
        morphology = sample.image.indices_for_role(ChannelRole.MORPHOLOGY)
        if len(morphology) == 1:
            return morphology
        missing = sorted(role.value for role in rgb_roles - roles)
        raise StructuralViewError(
            "abstain: H&E structural view requires RGB roles or one morphology "
            f"channel; missing roles={missing}"
        )

    if ChannelRole.NUCLEAR not in roles:
        raise StructuralViewError(
            f"abstain: {sample.modality.value} structural view requires a nuclear role"
        )
    eligible = {
        ChannelRole.NUCLEAR,
        ChannelRole.AUTOFLUORESCENCE,
        ChannelRole.MEMBRANE,
        ChannelRole.CYTOPLASM,
        ChannelRole.IMMUNE,
        ChannelRole.MORPHOLOGY,
    }
    indices = _ordered_indices_for_roles(sample, eligible)
    if sample.modality is Modality.COSMX and len(indices) < 2:
        raise StructuralViewError(
            "abstain: CosMx full structural view requires nuclear plus at least "
            "one broad morphology channel"
        )
    if not indices:  # Defensive: the nuclear check above should make this unreachable.
        raise StructuralViewError(
            f"abstain: no eligible structural roles for {sample.modality.value}"
        )
    return indices


def _minimal_structural_view(sample: QCSample) -> np.ndarray:
    data = sample.image.as_float32(scale_integer=True)
    if sample.modality is Modality.HE:
        indices = _full_structural_indices(sample)
        selected = data[..., indices]
        if selected.shape[-1] == 1:
            gray = selected[..., 0]
        else:
            gray = cv2.cvtColor(selected[..., :3], cv2.COLOR_RGB2GRAY)
        # H&E feature extraction explicitly expects RGB, so preserve that input
        # contract while making all three channels the same structural signal.
        return np.repeat(gray[..., None], 3, axis=-1)
    nuclear = sample.image.indices_for_role(ChannelRole.NUCLEAR)
    if not nuclear:
        raise StructuralViewError(
            f"abstain: {sample.modality.value} minimal view requires a nuclear role"
        )
    return data[..., nuclear[:1]]


def _view(sample: QCSample, view_name: str) -> np.ndarray:
    if view_name == "full_structural":
        indices = _full_structural_indices(sample)
        return sample.image.as_float32(scale_integer=True)[..., indices]
    if view_name == "minimal_structural":
        return _minimal_structural_view(sample)
    raise ValueError(f"Unknown view {view_name}")


def _analysis_region(sample: QCSample) -> np.ndarray:
    # Evaluate only in and immediately around the specimen. This avoids the
    # background-dominance trap while retaining crack pixels that may replace
    # tissue in the rendered image.
    core = sample.tissue_mask | sample.reference_artifact_mask
    kernel = np.ones((9, 9), dtype=np.uint8)
    return cv2.dilate(core.astype(np.uint8), kernel, iterations=1).astype(bool)


def _scenario_sample(
    modality: str,
    *,
    sample_index: int,
    modality_index: int,
    config: BenchmarkConfig,
) -> tuple[QCSample, SyntheticScenario, str]:
    scenario = SYNTHETIC_SCENARIOS[sample_index % len(SYNTHETIC_SCENARIOS)]
    pair_index = sample_index // len(SYNTHETIC_SCENARIOS)
    # A scenario block shares a seed, making clean/both comparisons paired with
    # respect to tissue geometry and base texture. The scenario suffix keeps
    # image identifiers unique even though the underlying seed is shared.
    paired_seed = config.seed + 100_000 * (modality_index + 1) + pair_index
    sample = generate_synthetic_sample(
        modality,
        seed=paired_seed,
        size=config.image_size,
        include_fold=scenario.include_fold,
        include_crack=scenario.include_crack,
        include_hard_negatives=scenario.include_hard_negatives,
    )
    pair_id = f"{modality}-pair-{pair_index:04d}"
    sample.sample_id = f"{sample.sample_id}-{scenario.name}"
    sample.metadata.update(
        {
            "scenario": scenario.name,
            "paired_scenario_id": pair_id,
        }
    )
    return sample, scenario, pair_id


def _prepare_output_directory(output: Path) -> None:
    """Create an empty owned destination without reusing stale artifacts."""

    if output.exists() and not output.is_dir():
        raise ValueError(f"Benchmark output exists and is not a directory: {output}")
    if output.exists():
        entries = list(output.iterdir())
        if entries:
            marker = output / "RUN_MANIFEST.json"
            owned = False
            if marker.is_file():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    owned = (
                        payload.get("kind") == "foldcrack_qc_generated_output"
                        and payload.get("schema_version") == 1
                    )
                except (OSError, json.JSONDecodeError):
                    owned = False
            if owned:
                raise FileExistsError(
                    f"Refusing to mix a new run with existing benchmark output at {output}; "
                    "use the guarded clean command or choose a new directory"
                )
            raise ValueError(
                f"Refusing nonempty unowned benchmark output directory: {output}"
            )
    output.mkdir(parents=True, exist_ok=True)


def _reference_matrix(
    modality: str,
    view_name: str,
    *,
    count: int,
    seed: int,
    size: tuple[int, int],
    patch_size: int,
) -> np.ndarray:
    tables = []
    for index in range(count):
        sample = generate_synthetic_sample(
            modality,
            seed=seed + index,
            size=size,
            include_fold=False,
            include_crack=False,
            include_hard_negatives=True,
        )
        table = extract_patch_feature_table(
            _view(sample, view_name),
            modality=modality,
            patch_size=patch_size,
            stride=max(16, patch_size // 2),
        )
        if table.values.size:
            tables.append(table.values)
    if not tables:
        raise RuntimeError(
            f"No clean reference features were extracted for {modality}/{view_name}"
        )
    return np.vstack(tables)


def _evaluate(
    sample: QCSample,
    *,
    prediction: np.ndarray,
    score_map: np.ndarray,
    method: str,
    target_name: str,
    view_name: str,
    runtime_seconds: float,
    scenario: str,
    pair_id: str,
    decision_rule: str,
    decision_threshold: float | None,
    runtime_scope: str,
    min_instance_area_um2: float,
) -> dict[str, Any]:
    if target_name == "artifact":
        target = sample.reference_artifact_mask
    else:
        target = sample.mask(target_name, required=True)
    prediction_array = np.asarray(prediction, dtype=bool)
    scores = np.asarray(score_map, dtype=float)
    if (
        prediction_array.shape != sample.spatial_shape
        or scores.shape != sample.spatial_shape
    ):
        raise ValueError("Prediction and score_map must match the sample spatial shape")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError(f"{method} produced a non-finite or out-of-range score map")
    valid = _analysis_region(sample)
    valid_scores = scores[valid]
    spacing = sample.image.pixel_size_um
    pixel_area_um2 = float(spacing[0] * spacing[1])
    min_instance_area_pixels = max(
        1, int(math.ceil(min_instance_area_um2 / pixel_area_um2))
    )
    recorded_threshold = 0.5 if decision_threshold is None else decision_threshold
    result = evaluate_sample(
        target,
        prediction_array,
        score_map=scores,
        threshold=recorded_threshold,
        sample_id=sample.sample_id,
        slide_id=sample.sample_id,
        modality=sample.modality.value,
        valid_mask=valid,
        spacing=spacing,
        boundary_tolerance=5.0,
        centerline_tolerance=5.0,
        instance_iou_threshold=0.1,
        min_instance_area=min_instance_area_pixels,
        pixel_area=pixel_area_um2,
        runtime_seconds=runtime_seconds,
        metadata={
            "method": method,
            "target": target_name,
            "view": view_name,
            "scenario": scenario,
            "paired_scenario_id": pair_id,
            "decision_rule": decision_rule,
            "decision_threshold": decision_threshold,
            "threshold_field_semantics": (
                "decision threshold"
                if decision_threshold is not None
                else "not applicable to composite explicit prediction"
            ),
            "runtime_scope": runtime_scope,
            "evaluation_min_instance_area_um2": min_instance_area_um2,
            "evaluation_min_instance_area_pixels": min_instance_area_pixels,
            "score_summary": {
                "mean": float(np.mean(valid_scores)),
                "p95": float(np.quantile(valid_scores, 0.95)),
                "p99": float(np.quantile(valid_scores, 0.99)),
                "maximum": float(np.max(valid_scores)),
                "predicted_fraction": float(np.mean(prediction_array[valid])),
            },
            "data_kind": "synthetic_engineering_smoke_test",
        },
    )
    return result


def _comparison_rows(
    results: Iterable[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        metadata = result.get("metadata", {})
        key = (
            str(result.get("modality", "unknown")),
            str(metadata.get("method", "unknown")),
            str(metadata.get("target", "unknown")),
            str(metadata.get("view", "unknown")),
        )
        groups.setdefault(key, []).append(result)

    rows: list[dict[str, Any]] = []
    for group_index, ((modality, method, target, view_name), items) in enumerate(
        sorted(groups.items())
    ):
        runtimes = np.asarray(
            [float(item.get("runtime_seconds") or 0.0) for item in items], dtype=float
        )
        aggregate = aggregate_results(items)
        assert isinstance(aggregate, Mapping)
        intervals = {
            metric: bootstrap_ci_by_cluster(
                items,
                metric,
                cluster_key="slide_id",
                n_resamples=n_resamples,
                confidence=0.95,
                seed=seed + 10 * group_index + metric_index,
            )
            for metric_index, metric in enumerate(
                ("pixel.precision", "pixel.recall", "pixel.dice")
            )
        }
        metadata_items = [item.get("metadata", {}) for item in items]
        decisions = sorted(
            {str(metadata.get("decision_rule")) for metadata in metadata_items}
        )
        thresholds = {metadata.get("decision_threshold") for metadata in metadata_items}
        runtime_scopes = sorted(
            {str(metadata.get("runtime_scope")) for metadata in metadata_items}
        )
        scenario_counts = {
            scenario.name: sum(
                metadata.get("scenario") == scenario.name for metadata in metadata_items
            )
            for scenario in SYNTHETIC_SCENARIOS
        }
        rows.append(
            {
                "modality": modality,
                "method": method,
                "target": target,
                "view": view_name,
                "n_samples": len(items),
                "n_unique_images": len({str(item.get("sample_id")) for item in items}),
                "scenario_counts": json.dumps(scenario_counts, sort_keys=True),
                "decision_rule": " | ".join(decisions),
                "decision_threshold": (
                    next(iter(thresholds)) if len(thresholds) == 1 else "mixed"
                ),
                "runtime_semantics": " | ".join(runtime_scopes),
                "precision": aggregate["pixel"]["precision"],
                "precision_ci_lower": intervals["pixel.precision"]["lower"],
                "precision_ci_upper": intervals["pixel.precision"]["upper"],
                "recall": aggregate["pixel"]["recall"],
                "recall_ci_lower": intervals["pixel.recall"]["lower"],
                "recall_ci_upper": intervals["pixel.recall"]["upper"],
                "dice": aggregate["pixel"]["dice"],
                "dice_ci_lower": intervals["pixel.dice"]["lower"],
                "dice_ci_upper": intervals["pixel.dice"]["upper"],
                "iou": aggregate["pixel"]["iou"],
                "boundary_f1": aggregate["boundary"]["f1"],
                "centerline_f1": aggregate["centerline"]["f1"],
                "instance_recall": aggregate["instance"]["recall"],
                "false_positives_per_sample": aggregate["instance"][
                    "false_positives_per_sample"
                ],
                "mean_runtime_seconds": float(runtimes.mean()),
            }
        )
    return rows


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty comparison")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _grouped_evaluation_report(
    results: Sequence[dict[str, Any]],
    *,
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Build one evaluation report per semantically comparable output group."""

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for result in results:
        metadata = result["metadata"]
        key = (
            str(result["modality"]),
            str(metadata["method"]),
            str(metadata["target"]),
            str(metadata["view"]),
        )
        grouped.setdefault(key, []).append(result)

    reports: list[dict[str, Any]] = []
    for group_index, (key, items) in enumerate(sorted(grouped.items())):
        modality, method, target, view_name = key
        report = build_report(
            items,
            bootstrap_metrics=("pixel.dice", "pixel.precision", "pixel.recall"),
            n_resamples=n_resamples,
            confidence=0.95,
            seed=seed + 100 * group_index,
        )
        reports.append(
            {
                "group": {
                    "modality": modality,
                    "method": method,
                    "target": target,
                    "view": view_name,
                },
                "unique_image_count": len({item["sample_id"] for item in items}),
                "prediction_count": len(items),
                "scenario_counts": {
                    scenario.name: sum(
                        item["metadata"]["scenario"] == scenario.name for item in items
                    )
                    for scenario in SYNTHETIC_SCENARIOS
                },
                "ci_semantics": (
                    "slide-cluster percentile bootstrap of the same pooled aggregate "
                    "shown in this fixed output group's summary"
                ),
                "report": report,
            }
        )
    return {
        "schema_version": "1.1",
        "kind": "grouped_foldcrack_qc_evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_kind": "synthetic_engineering_smoke_test",
        "scientific_validation": False,
        "scientific_validation_reason": (
            "Synthetic images test implementation behavior but cannot estimate real-data efficacy."
        ),
        "grouping_fields": ["modality", "method", "target", "view"],
        "unique_image_count": len({item["sample_id"] for item in results}),
        "prediction_count": len(results),
        "output_group_count": len(reports),
        "groups": reports,
    }


def _grouped_evaluation_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Grouped Fold/Crack QC Evaluation",
        "",
        "> **Synthetic engineering output, not scientific validation.** Metrics are",
        "> never pooled across methods, targets, or channel views.",
        "",
        f"- Unique images: {report['unique_image_count']}",
        f"- Predictions evaluated: {report['prediction_count']}",
        f"- Comparable output groups: {report['output_group_count']}",
        "",
        "Each confidence interval resamples image/slide clusters within one",
        "`(modality, method, target, view)` group and recomputes the same pooled",
        "aggregate shown as the estimate.",
        "",
        "| Modality | Method | Target | View | Images | Pooled Dice [95% CI] | Pooled precision [95% CI] | Pooled recall [95% CI] |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for group in report["groups"]:
        key = group["group"]
        bootstrap = group["report"]["bootstrap"]

        def interval(metric: str) -> str:
            values = bootstrap[metric]
            return (
                f"{_format_value(values['estimate'])} "
                f"[{_format_value(values['lower'])}, {_format_value(values['upper'])}]"
            )

        lines.append(
            "| {modality} | {method} | {target} | {view} | {images} | {dice} | "
            "{precision} | {recall} |".format(
                modality=key["modality"].upper(),
                method=key["method"],
                target=key["target"],
                view=key["view"],
                images=group["unique_image_count"],
                dice=interval("pixel.dice"),
                precision=interval("pixel.precision"),
                recall=interval("pixel.recall"),
            )
        )
    return "\n".join(lines) + "\n"


def _format_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "—" if not np.isfinite(number) else f"{number:.3f}"


def _format_interval(row: dict[str, Any], metric: str) -> str:
    estimate = _format_value(row[metric])
    lower = _format_value(row[f"{metric}_ci_lower"])
    upper = _format_value(row[f"{metric}_ci_upper"])
    return f"{estimate} [{lower}, {upper}]"


def _feasibility_markdown(
    rows: list[dict[str, Any]],
    *,
    environment: dict[str, Any],
    elapsed: float,
    result_count: int,
    unique_image_count: int,
    engineering_checks: Mapping[str, bool],
) -> str:
    lines = [
        "# Fold/Crack QC Feasibility Report",
        "",
        "> **Engineering smoke test only.** Every number below is measured on",
        "> deterministic synthetic images. It tests software wiring and comparative",
        "> behavior; it is not evidence of performance on Merck H&E, COMET, or CosMx data.",
        "> The generator and classical rules intentionally share simple visual cues, so",
        "> these values must not be used to select or rank a production method.",
        "",
        "## Run status",
        "",
        f"- Evaluated {result_count} predictions from {unique_image_count} unique images in {elapsed:.2f} seconds.",
        f"- Python {environment['python']} on {environment['machine']}; OpenCV {environment['opencv']}.",
        f"- Optional DINOv2 PyTorch backend available: `{environment['torch_available']}`; H&E-only comparator, not run.",
        f"- Optional OME-TIFF metadata backend available: `{environment['tifffile_available']}`.",
        f"- Scenarios: {', '.join(scenario.name for scenario in SYNTHETIC_SCENARIOS)}.",
        "",
        "The core classical, anomaly, hybrid, channel-ablation, evaluation, and",
        "reporting paths completed without external downloads.",
        "",
        "## Engineering checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in engineering_checks.items()
    )
    lines.extend(
        [
            "",
            "## Per-target comparison (fold, crack, and artifact union)",
            "",
            "| Modality | Method | Target | View | Precision [95% CI] | Recall [95% CI] | Dice [95% CI] | Instance recall | Shared inference call (s) |",
            "|---|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {modality} | {method} | {target} | {view} | {precision} | {recall} | {dice} | "
            "{instance_recall} | {runtime} |".format(
                modality=row["modality"].upper(),
                method=row["method"],
                target=row["target"],
                view=row["view"],
                precision=_format_interval(row, "precision"),
                recall=_format_interval(row, "recall"),
                dice=_format_interval(row, "dice"),
                instance_recall=_format_value(row["instance_recall"]),
                runtime=_format_value(row["mean_runtime_seconds"]),
            )
        )
    lines.extend(
        [
            "",
            "## Feasibility interpretation",
            "",
            "- **Classical branch:** supplies interpretable fold/crack proposals and a",
            "  corporate-safe baseline, but must be tuned on internal development data.",
            "- **Clean-reference anomaly branch:** demonstrates label-light training, but",
            "  anomaly is not a semantic fold/crack label and will flag rare normal anatomy.",
            "- **Hybrid branch:** is the recommended phase-1 architecture because it combines",
            "  physical cues with novelty while retaining an abstention/review pathway.",
            "- **Minimal-channel comparison:** is an engineering channel ablation, not channel",
            "  selection. Final channel roles and thresholds must be locked on internal data.",
            "- **Foundation model:** remains an optional comparator. The runnable core does not",
            "  require PyTorch or download restricted pathology weights. DINOv2 was not",
            "  benchmarked here and is scoped as an H&E-only comparator until separately validated.",
            "- **Runtime:** rows sharing a detector call repeat that same end-to-end call time;",
            "  these are not additive component timings. Exact semantics are recorded per row.",
            "",
            "## What is still required for a performance claim",
            "",
            "1. Confirm the intended action and distinguish tissue tear, glass/coverslip",
            "   crack, knife line, and acquisition seam.",
            "2. Build independently reviewed, adjudicated, modality-stratified internal",
            "   reference sets; keep the final test cohorts locked.",
            "3. Select thresholds on development data and report confidence intervals by",
            "   patient/block/slide/run—not by tile.",
            "4. Validate H&E, COMET, and CosMx separately across device, site, panel, tissue,",
            "   batch, time, missing channels, and downstream scientific impact.",
            "",
            "See `docs/EVALUATION.md`, `docs/ANNOTATION_GUIDE.md`, and `docs/AI-SPEC.md`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _metrics_are_finite_and_bounded(results: Sequence[Mapping[str, Any]]) -> bool:
    unit_metrics = (
        ("pixel", "precision"),
        ("pixel", "recall"),
        ("pixel", "specificity"),
        ("pixel", "accuracy"),
        ("pixel", "balanced_accuracy"),
        ("pixel", "dice"),
        ("pixel", "iou"),
        ("boundary", "surface_dice"),
        ("boundary", "f1"),
        ("centerline", "f1"),
        ("centerline", "cldice"),
        ("instance", "precision"),
        ("instance", "recall"),
        ("instance", "f1"),
        ("burden", "true_fraction"),
        ("burden", "predicted_fraction"),
        ("burden", "absolute_fraction_error"),
    )
    for result in results:
        for section, metric in unit_metrics:
            value = result.get(section, {}).get(metric)
            if value is None:
                return False
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                return False
        runtime = float(result.get("runtime_seconds", float("nan")))
        if not math.isfinite(runtime) or runtime < 0.0:
            return False
        score_summary = result.get("metadata", {}).get("score_summary", {})
        for value in score_summary.values():
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                return False
    return True


def _metamorphic_diagnostics(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare paired both-artifact and clean score means for artifact outputs."""

    paired: dict[tuple[str, str, str, str, str], dict[str, float]] = {}
    for result in results:
        metadata = result["metadata"]
        if metadata["target"] != "artifact":
            continue
        key = (
            str(result["modality"]),
            str(metadata["method"]),
            str(metadata["target"]),
            str(metadata["view"]),
            str(metadata["paired_scenario_id"]),
        )
        paired.setdefault(key, {})[str(metadata["scenario"])] = float(
            metadata["score_summary"]["mean"]
        )

    by_group: dict[tuple[str, str, str], list[float]] = {}
    comparisons: list[dict[str, Any]] = []
    for key, scenarios in sorted(paired.items()):
        if "clean" not in scenarios or "both" not in scenarios:
            continue
        modality, method, _, view_name, pair_id = key
        difference = scenarios["both"] - scenarios["clean"]
        by_group.setdefault((modality, method, view_name), []).append(difference)
        comparisons.append(
            {
                "modality": modality,
                "method": method,
                "view": view_name,
                "paired_scenario_id": pair_id,
                "clean_mean_score": scenarios["clean"],
                "both_mean_score": scenarios["both"],
                "difference": difference,
                "positive_separation": difference > 0.0,
            }
        )

    expected = {
        (modality.value, method, view_name)
        for modality in (Modality.HE, Modality.COMET, Modality.COSMX)
        for method, target, view_name in EXPECTED_OUTPUT_GROUPS
        if target == "artifact"
    }
    group_status: dict[str, bool] = {}
    for key in sorted(expected):
        differences = by_group.get(key, [])
        # This is intentionally a coarse engineering invariant, not a claimed
        # efficacy threshold: paired artifact phantoms should raise mean score.
        passed = bool(differences) and float(np.mean(differences)) > 0.0
        group_status["/".join(key)] = passed
    return {
        "definition": (
            "Within paired synthetic geometry, mean artifact score for scenario=both "
            "must exceed scenario=clean in every artifact-output group."
        ),
        "groups": group_status,
        "comparisons": comparisons,
        "passed": bool(group_status) and all(group_status.values()),
        "scientific_validation": False,
    }


def _operational_acceptance_disabled() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "acceptance_eligible": False,
        "overall_status": "NOT_EVALUATED_SYNTHETIC",
        "decision_records_created": False,
        "modalities": [
            {
                "modality": modality.value,
                "status": "NOT_EVALUATED_SYNTHETIC",
            }
            for modality in (Modality.HE, Modality.COMET, Modality.COSMX)
        ],
        "reason": (
            "Segmentation outputs on synthetic images do not define supported "
            "PASS/REVIEW/FAIL decisions and cannot satisfy operational acceptance gates."
        ),
    }


def run_feasibility(config: BenchmarkConfig) -> dict[str, Any]:
    output = Path(config.output_dir).resolve()
    _prepare_output_directory(output)
    marker = output / "RUN_MANIFEST.json"
    environment = _environment()
    # Synthetic fixtures are generated at 0.5 um/px. Convert the legacy CLI's
    # pixel patch knob into a locked physical context so the detector exercises
    # spacing-aware patch conversion at inference.
    detector_patch_size_um = float(config.patch_size * 0.5)
    detector_stride_um = float(max(16, config.patch_size // 2) * 0.5)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "foldcrack_qc_generated_output",
        "status": "running",
        "config": {
            "samples_per_modality": config.samples_per_modality,
            "clean_samples_per_modality": config.clean_samples_per_modality,
            "image_size": list(config.image_size),
            "seed": config.seed,
            "patch_size": config.patch_size,
            "bootstrap_resamples": config.bootstrap_resamples,
            "scenarios": [scenario.name for scenario in SYNTHETIC_SCENARIOS],
            "physical_geometry": {
                "min_component_area_um2": config.min_component_area_um2,
                "crack_neighborhood_radius_um": config.crack_neighborhood_radius_um,
                "fold_morphology_radius_um": config.fold_morphology_radius_um,
                "evaluation_min_instance_area_um2": config.evaluation_min_instance_area_um2,
                "patch_size_um": detector_patch_size_um,
                "stride_um": detector_stride_um,
            },
        },
        "environment": environment,
        "validation_boundary": (
            "Synthetic engineering smoke test only; scientific and operational "
            "validation are explicitly ineligible."
        ),
    }
    _json_dump(marker, manifest)

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    target_branch_checks: list[bool] = []
    overlay_paths: list[Path] = []
    modalities = (Modality.HE.value, Modality.COMET.value, Modality.COSMX.value)

    for modality_index, modality in enumerate(modalities):
        detectors: dict[str, HybridQCDetector] = {}
        for view_name in ("full_structural", "minimal_structural"):
            reference = _reference_matrix(
                modality,
                view_name,
                count=config.clean_samples_per_modality,
                seed=config.seed + 10_000 * (modality_index + 1),
                size=config.image_size,
                patch_size=config.patch_size,
            )
            detectors[view_name] = HybridQCDetector().fit(reference)

        for sample_index in range(config.samples_per_modality):
            sample, scenario, pair_id = _scenario_sample(
                modality,
                sample_index=sample_index,
                modality_index=modality_index,
                config=config,
            )
            full_image = _view(sample, "full_structural")

            fold_positive = bool(np.any(sample.mask("fold", required=True)))
            crack_positive = bool(np.any(sample.mask("crack", required=True)))
            hard_negative_positive = bool(
                np.any(sample.mask("hard_negative", required=True))
            )
            target_branch_checks.extend(
                (
                    fold_positive == scenario.include_fold,
                    crack_positive == scenario.include_crack,
                    hard_negative_positive == scenario.include_hard_negatives,
                    bool(np.any(sample.reference_artifact_mask))
                    == (scenario.include_fold or scenario.include_crack),
                )
            )

            classical_started = time.perf_counter()
            classical = classical_candidate_masks(
                full_image,
                modality=modality,
                pixel_size_um=sample.image.pixel_size_um,
                min_component_area_um2=config.min_component_area_um2,
                crack_neighborhood_radius_um=config.crack_neighborhood_radius_um,
                fold_morphology_radius_um=config.fold_morphology_radius_um,
            )
            classical_runtime = time.perf_counter() - classical_started
            combined_prediction = classical.fold | classical.crack
            combined_score = np.maximum(classical.fold_score, classical.crack_score)
            if scenario.include_fold:
                target_branch_checks.append(
                    bool(np.any(classical.fold & sample.mask("fold", required=True)))
                )
            if scenario.include_crack:
                target_branch_checks.append(
                    bool(np.any(classical.crack & sample.mask("crack", required=True)))
                )
            if scenario.name in {"clean", "hard_negative_only"}:
                target_branch_checks.append(
                    not bool(np.all(combined_prediction[_analysis_region(sample)]))
                )
            results.extend(
                [
                    _evaluate(
                        sample,
                        prediction=classical.fold,
                        score_map=classical.fold_score,
                        method="classical",
                        target_name="fold",
                        view_name="full_structural",
                        runtime_seconds=classical_runtime,
                        scenario=scenario.name,
                        pair_id=pair_id,
                        decision_rule=(
                            f"fold_score >= 0.53; min_component_area="
                            f"{config.min_component_area_um2:g} um^2; closing_radius="
                            f"{config.fold_morphology_radius_um:g} um"
                        ),
                        decision_threshold=0.53,
                        runtime_scope=(
                            "one shared classical_candidate_masks call repeated on the "
                            "fold, crack, and artifact evaluation rows"
                        ),
                        min_instance_area_um2=config.evaluation_min_instance_area_um2,
                    ),
                    _evaluate(
                        sample,
                        prediction=classical.crack,
                        score_map=classical.crack_score,
                        method="classical",
                        target_name="crack",
                        view_name="full_structural",
                        runtime_seconds=classical_runtime,
                        scenario=scenario.name,
                        pair_id=pair_id,
                        decision_rule=(
                            f"crack_score >= 0.54; neighborhood_radius="
                            f"{config.crack_neighborhood_radius_um:g} um; physical-spacing-aware "
                            f"elongation; min_component_area={config.min_component_area_um2:g} um^2"
                        ),
                        decision_threshold=0.54,
                        runtime_scope=(
                            "one shared classical_candidate_masks call repeated on the "
                            "fold, crack, and artifact evaluation rows"
                        ),
                        min_instance_area_um2=config.evaluation_min_instance_area_um2,
                    ),
                    _evaluate(
                        sample,
                        prediction=combined_prediction,
                        score_map=combined_score,
                        method="classical",
                        target_name="artifact",
                        view_name="full_structural",
                        runtime_seconds=classical_runtime,
                        scenario=scenario.name,
                        pair_id=pair_id,
                        decision_rule=(
                            "(physically cleaned fold_score >= 0.53) OR "
                            "(physically cleaned crack_score >= 0.54); geometry in run manifest"
                        ),
                        decision_threshold=None,
                        runtime_scope=(
                            "one shared classical_candidate_masks call repeated on the "
                            "fold, crack, and artifact evaluation rows"
                        ),
                        min_instance_area_um2=config.evaluation_min_instance_area_um2,
                    ),
                ]
            )

            representative_prediction = combined_prediction
            for view_name, detector in detectors.items():
                view_data = _view(sample, view_name)
                hybrid_started = time.perf_counter()
                hybrid = detector.score(
                    view_data,
                    modality=modality,
                    patch_size=config.patch_size,
                    stride=max(16, config.patch_size // 2),
                    patch_size_um=detector_patch_size_um,
                    stride_um=detector_stride_um,
                    pixel_size_um=sample.image.pixel_size_um,
                    min_component_area_um2=config.min_component_area_um2,
                    crack_neighborhood_radius_um=config.crack_neighborhood_radius_um,
                    fold_morphology_radius_um=config.fold_morphology_radius_um,
                )
                hybrid_runtime = time.perf_counter() - hybrid_started
                anomaly_threshold = float(
                    getattr(
                        detector,
                        "anomaly_decision_threshold",
                        ANOMALY_DECISION_THRESHOLD,
                    )
                )
                anomaly_prediction = hybrid.anomaly_score >= anomaly_threshold
                results.append(
                    _evaluate(
                        sample,
                        prediction=anomaly_prediction,
                        score_map=hybrid.anomaly_score,
                        method="clean_reference_anomaly",
                        target_name="artifact",
                        view_name=view_name,
                        runtime_seconds=hybrid_runtime,
                        scenario=scenario.name,
                        pair_id=pair_id,
                        decision_rule=(
                            f"calibrated clean-reference anomaly score >= {anomaly_threshold:g}; "
                            "0.5 represents the fitted upper clean-reference quantile"
                        ),
                        decision_threshold=anomaly_threshold,
                        runtime_scope=(
                            "one end-to-end HybridQCDetector.score call repeated on the "
                            "anomaly-only and hybrid evaluation rows"
                        ),
                        min_instance_area_um2=config.evaluation_min_instance_area_um2,
                    )
                )
                hybrid_threshold = float(detector.decision_threshold)
                hybrid_rule = (
                    f"cleanup((fused_score >= {hybrid_threshold:g}) OR classical_fold "
                    f"OR classical_crack OR (anomaly_score >= {anomaly_threshold:g})) "
                    "within review_support; min_area=max(4, patch_area//128), "
                    f"physical min_area={config.min_component_area_um2:g} um^2, "
                    "closing_radius=1 px"
                )
                recorded_hybrid_threshold: float | None = None
                results.append(
                    _evaluate(
                        sample,
                        prediction=hybrid.predicted_mask,
                        score_map=hybrid.fused_score,
                        method="hybrid",
                        target_name="artifact",
                        view_name=view_name,
                        runtime_seconds=hybrid_runtime,
                        scenario=scenario.name,
                        pair_id=pair_id,
                        decision_rule=hybrid_rule,
                        decision_threshold=recorded_hybrid_threshold,
                        runtime_scope=(
                            "one end-to-end HybridQCDetector.score call repeated on the "
                            "anomaly-only and hybrid evaluation rows"
                        ),
                        min_instance_area_um2=config.evaluation_min_instance_area_um2,
                    )
                )
                if view_name == "full_structural":
                    representative_prediction = hybrid.predicted_mask

            overlay_priority = (4, 2, 3, 1, 0)
            selected_overlay_indices = set(
                overlay_priority[
                    : min(config.overlays_per_modality, len(overlay_priority))
                ]
            )
            if sample_index in selected_overlay_indices:
                overlay_path = write_overlay(
                    output / "overlays" / modality / f"{sample.sample_id}.png",
                    full_image,
                    target_fold=sample.mask("fold"),
                    target_crack=sample.mask("crack"),
                    prediction=representative_prediction,
                )
                overlay_paths.append(overlay_path)

    elapsed = time.perf_counter() - started
    comparison_rows = _comparison_rows(
        results,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    evaluation_report = _grouped_evaluation_report(
        results,
        n_resamples=config.bootstrap_resamples,
        seed=config.seed,
    )
    write_csv_report(results, output / "per_sample_results.csv")
    write_json_report(evaluation_report, output / "evaluation_report.json")
    (output / "evaluation_report.md").write_text(
        _grouped_evaluation_markdown(evaluation_report), encoding="utf-8"
    )
    _write_comparison_csv(comparison_rows, output / "comparison.csv")
    _json_dump(output / "comparison.json", comparison_rows)

    operational_acceptance = _operational_acceptance_disabled()
    _json_dump(output / "operational_acceptance.json", operational_acceptance)

    expected_group_keys = {
        (modality, method, target, view_name)
        for modality in modalities
        for method, target, view_name in EXPECTED_OUTPUT_GROUPS
    }
    actual_group_keys: dict[tuple[str, str, str, str], set[str]] = {}
    for result in results:
        metadata = result["metadata"]
        key = (
            str(result["modality"]),
            str(metadata["method"]),
            str(metadata["target"]),
            str(metadata["view"]),
        )
        actual_group_keys.setdefault(key, set()).add(str(metadata["scenario"]))
    expected_scenarios = {scenario.name for scenario in SYNTHETIC_SCENARIOS}
    expected_images = len(modalities) * config.samples_per_modality
    expected_results = expected_images * len(EXPECTED_OUTPUT_GROUPS)
    unique_images = {str(result["sample_id"]) for result in results}
    metamorphic = _metamorphic_diagnostics(results)

    expected_overlay_count = len(modalities) * min(
        config.overlays_per_modality, len(SYNTHETIC_SCENARIOS)
    )
    overlays_decodable = len(overlay_paths) == expected_overlay_count and all(
        cv2.imread(str(path), cv2.IMREAD_UNCHANGED) is not None
        for path in overlay_paths
    )
    output_files = (
        "per_sample_results.csv",
        "evaluation_report.json",
        "evaluation_report.md",
        "comparison.csv",
        "comparison.json",
        "operational_acceptance.json",
    )
    engineering_checks: dict[str, bool] = {
        "prediction_count": len(results) == expected_results,
        "unique_image_count": len(unique_images) == expected_images,
        "method_target_view_group_coverage": set(actual_group_keys)
        == expected_group_keys,
        "factorial_scenario_coverage": (
            set(actual_group_keys) == expected_group_keys
            and all(
                scenarios == expected_scenarios
                for scenarios in actual_group_keys.values()
            )
        ),
        "finite_bounded_metrics_and_scores": _metrics_are_finite_and_bounded(results),
        "nondegenerate_generated_target_branches": bool(target_branch_checks)
        and all(target_branch_checks),
        "paired_artifact_vs_clean_score_invariant": bool(metamorphic["passed"]),
        "decodable_fresh_overlays": overlays_decodable,
        "grouped_report_cardinality": (
            evaluation_report["unique_image_count"] == expected_images
            and evaluation_report["prediction_count"] == expected_results
            and evaluation_report["output_group_count"] == len(expected_group_keys)
        ),
        "operational_acceptance_disabled": (
            operational_acceptance["acceptance_eligible"] is False
            and operational_acceptance["overall_status"] == "NOT_EVALUATED_SYNTHETIC"
        ),
        "required_output_files": all(
            (output / name).is_file() for name in output_files
        ),
        "decision_and_runtime_semantics_recorded": all(
            bool(result["metadata"].get("decision_rule"))
            and bool(result["metadata"].get("runtime_scope"))
            for result in results
        ),
    }

    report_path = output / "FEASIBILITY_REPORT.md"
    report_path.write_text(
        _feasibility_markdown(
            comparison_rows,
            environment=environment,
            elapsed=elapsed,
            result_count=len(results),
            unique_image_count=len(unique_images),
            engineering_checks=engineering_checks,
        ),
        encoding="utf-8",
    )

    engineering_smoke_test_passed = all(engineering_checks.values())
    manifest.update(
        {
            "status": "complete" if engineering_smoke_test_passed else "incomplete",
            "elapsed_seconds": elapsed,
            "result_count": len(results),
            "unique_image_count": len(unique_images),
            "prediction_count": len(results),
            "output_group_count": len(actual_group_keys),
            "engineering_checks": engineering_checks,
            "metamorphic_diagnostics": metamorphic,
            "runtime_measurement": (
                "Wall-clock inference-call times. Alternative target/output rows that share a "
                "call repeat its timing and must not be summed."
            ),
            "engineering_smoke_test_passed": engineering_smoke_test_passed,
            "scientific_validation_passed": False,
            "scientific_validation_reason": "Synthetic data cannot establish real-data efficacy.",
        }
    )
    _json_dump(marker, manifest)
    summary = (
        f"Engineering smoke test {'PASSED' if engineering_smoke_test_passed else 'FAILED'}: "
        f"{len(results)} evaluations across H&E, COMET, and CosMx in {elapsed:.2f}s. "
        "Scientific validation remains intentionally unclaimed."
    )
    return {
        "engineering_smoke_test_passed": engineering_smoke_test_passed,
        "summary": summary,
        "report_path": str(report_path),
        "output_dir": str(output),
        "result_count": len(results),
        "unique_image_count": len(unique_images),
        "engineering_checks": engineering_checks,
    }


__all__ = ["BenchmarkConfig", "run_feasibility"]
