"""Rebuild the audited fold/crack benchmark report and manuscript figures.

The script reads immutable benchmark artifacts from ``artifacts/`` and does not
run model inference.  It independently recomputes the highest-impact aggregate
metrics from per-field outcomes, asserts cross-method cohort identity, writes a
bounded Data Analytics report artifact, and exports publication-oriented SVG,
PDF, and 300-dpi PNG figures.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from reportlab import Version as REPORTLAB_VERSION
from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import (
    Circle,
    Drawing,
    Line,
    Polygon,
    Rect,
    String,
)
from reportlab.graphics.shapes import Image as DrawingImage
from reportlab.lib.colors import Color, HexColor
from reportlab.pdfgen.canvas import Canvas

REPORT_DATE = "2026-08-27"
REPORT_TIMESTAMP = "2026-08-27T09:30:00-04:00"
TITLE = "Fold and Crack Artifact QC Across H&E, COMET, and CosMx"

REFERENCES_MD = """1. Koparir OF, Tarakçı Gençer B, Sengur A. *Histology Tissue Fold Dataset with Pixel-Level Annotations for Artificial Intelligence-Based Detection and Segmentation of Tissue Fold Artifacts in H&E-Stained Teaching Slides* [dataset]. Zenodo. 2026; version 1.0. [doi:10.5281/zenodo.21493260](https://doi.org/10.5281/zenodo.21493260).
2. Koparir OF, Tarakci Gencer B, Sengur A. Deep learning-assisted quality control of histology teaching slides: detection and localization of tissue fold artifacts in H&E-stained images. *Bioengineering*. 2026;13(8):937. [doi:10.3390/bioengineering13080937](https://doi.org/10.3390/bioengineering13080937).
3. Janowczyk A, Zuo R, Gilmore H, Feldman M, Madabhushi A. HistoQC: an open-source quality control tool for digital pathology slides. *JCO Clinical Cancer Informatics*. 2019;3:1–7. [doi:10.1200/CCI.18.00157](https://doi.org/10.1200/CCI.18.00157).
4. Weng Z, Seper A, Pryalukhin A, et al. GrandQC: a comprehensive solution to quality control problem in digital pathology. *Nature Communications*. 2024;15:10685. [doi:10.1038/s41467-024-54769-y](https://doi.org/10.1038/s41467-024-54769-y).
5. Tolkach Y. *Test Dataset from Weng Z. et al. Nat Communications 2024* [GrandQC manually annotated test dataset]. Zenodo. 2024. [doi:10.5281/zenodo.14039591](https://doi.org/10.5281/zenodo.14039591).
6. Wang Z, Zhou Z, Wen Z, Kook JH, Wojcik JB, Kang J. DiffusionQC: artifact detection and quality control in histopathology images via diffusion model. In: *2026 IEEE 23rd International Symposium on Biomedical Imaging (ISBI)*. IEEE; 2026:1–5. [doi:10.1109/ISBI61048.2026.11515418](https://doi.org/10.1109/ISBI61048.2026.11515418).
7. Oquab M, Darcet T, Moutakanni T, et al. DINOv2: learning robust visual features without supervision. *Transactions on Machine Learning Research*. 2024. [Official OpenReview record](https://openreview.net/forum?id=a68SUt6zFt).
8. Siméoni O, Vo HV, Seitzer M, et al. DINOv3. arXiv [Preprint]. 2025; arXiv:2508.10104. [doi:10.48550/arXiv.2508.10104](https://doi.org/10.48550/arXiv.2508.10104).
9. Meta AI. *Model Card for DINOv3* and *DINOv3 License* [model/software documentation]. 2025. [Model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md); [custom license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Accessed 2026-08-27.
10. Nechaev D, Pchelnikov A, Ivanova E. Hibou: a family of foundational vision transformers for pathology. arXiv [Preprint]. 2024; arXiv:2406.05074. [doi:10.48550/arXiv.2406.05074](https://doi.org/10.48550/arXiv.2406.05074).
11. HistAI. *Hibou-B (histai/hibou-b)* [pretrained model and model card]. Hugging Face; 2024. [Official model card](https://huggingface.co/histai/hibou-b). Accessed 2026-08-27.
12. Tschannen M, Gritsenko A, Wang X, et al. SigLIP 2: multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv [Preprint]. 2025; arXiv:2502.14786. [doi:10.48550/arXiv.2502.14786](https://doi.org/10.48550/arXiv.2502.14786).
13. Google. *SigLIP2 Base Patch16-224 (google/siglip2-base-patch16-224)* [pretrained model and model card]. Hugging Face; 2025. [Official model card](https://huggingface.co/google/siglip2-base-patch16-224). Accessed 2026-08-27.
14. Mahmood Lab. *UNI2-h (MahmoodLab/UNI2-h)* [pretrained model and model card]. Hugging Face; 2025. [Official model card](https://huggingface.co/MahmoodLab/UNI2-h). Accessed 2026-08-27.
15. Chen RJ, Ding T, Lu MY, Williamson DFK, et al. Towards a general-purpose foundation model for computational pathology. *Nature Medicine*. 2024;30:850–862. [doi:10.1038/s41591-024-02857-3](https://doi.org/10.1038/s41591-024-02857-3).
16. Mahmood Lab. *CONCHv1.5 (MahmoodLab/conchv1_5)* [pretrained model and model card]. Hugging Face; 2024. [Official model card](https://huggingface.co/MahmoodLab/conchv1_5). Accessed 2026-08-27.
17. Lu MY, Chen B, Williamson DFK, et al. A visual-language foundation model for computational pathology. *Nature Medicine*. 2024;30:863–874. [doi:10.1038/s41591-024-02856-4](https://doi.org/10.1038/s41591-024-02856-4).
18. Mahmood Lab. *KRONOS2 (MahmoodLab/KRONOS2)* [pretrained model and model card]. Hugging Face; 2026. [Official model card](https://huggingface.co/MahmoodLab/KRONOS2). Accessed 2026-08-27.
19. Shaban M, Chang Y, Qiu H, et al. A foundation model for spatial proteomics. arXiv [Preprint]. 2025; arXiv:2506.03373. [doi:10.48550/arXiv.2506.03373](https://doi.org/10.48550/arXiv.2506.03373).
20. Andhari MD, Rinaldi G, Nazari P, et al. Quality control of immunofluorescence images using artificial intelligence. *Cell Reports Physical Science*. 2024;5(10):102220. [doi:10.1016/j.xcrp.2024.102220](https://doi.org/10.1016/j.xcrp.2024.102220).
21. Andhari MD. *QualIFAI* [COMET/Lunaphore-related dataset and trained-model archive]. Zenodo. 2024; version v2. [doi:10.5281/zenodo.12699470](https://doi.org/10.5281/zenodo.12699470).
22. Tsubosaka A, Ishikawa S. *CosMx Spatial transcriptome dataset of human gastric mucosa* [dataset]. Zenodo. 2023. [doi:10.5281/zenodo.8333281](https://doi.org/10.5281/zenodo.8333281).
23. Van den Broek TJM; Princess Máxima Center. *Single-cell spatial analysis of pediatric high-grade glioma reveals a novel population of SPP1+/GPNMB+ myeloid cells with immunosuppressive and tumor-promoting capabilities* [CosMx dataset]. Zenodo. 2025. [Official record 16877090](https://zenodo.org/records/16877090). Accessed 2026-08-27.
24. Gautam K, Raipuria G, Singhal N. AIRAQc: pre-analytical tool for accurate identification and quantification of artefacts in histopathology. In: *Medical Imaging with Deep Learning—Short Papers (MIDL 2025)*. 2025. [Official OpenReview record](https://openreview.net/forum?id=XNNsQqs1UP).
25. Kanwal N. *HistoArtifacts* [dataset]. Zenodo. 2024; version v1. [doi:10.5281/zenodo.10809442](https://doi.org/10.5281/zenodo.10809442).
26. Kanwal N, Khoraminia F, Kiraz U, et al. Equipping computational pathology systems with artifact processing pipelines: a showcase for computation and performance trade-offs. *BMC Medical Informatics and Decision Making*. 2024;24:288. [doi:10.1186/s12911-024-02676-z](https://doi.org/10.1186/s12911-024-02676-z).
27. Foucart A. *Artefact segmentation in digital pathology whole-slide images* [dataset]. Zenodo. 2020; version v1. [doi:10.5281/zenodo.3773097](https://doi.org/10.5281/zenodo.3773097).
28. Foucart A, Debeir O, Decaestecker C. Snow supervision in digital pathology: managing imperfect annotations for segmentation in deep learning. Research Square [Preprint]. 2020; version 1. [doi:10.21203/rs.3.rs-116512/v1](https://doi.org/10.21203/rs.3.rs-116512/v1).
29. Meta AI. *DINOv2* [official repository, pretrained weights, and license]. [GitHub repository](https://github.com/facebookresearch/dinov2). Accessed 2026-08-27. Code and the standard DINOv2 weights are released under Apache License 2.0; separately listed domain-specific weights may use different terms.
30. Weng Z, Seper A, Pryalukhin A, et al. *GrandQC* [official repository and license]. [GitHub repository](https://github.com/cpath-ukk/grandqc). Accessed 2026-08-27. CC BY-NC-SA 4.0."""

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURES = HERE / "figures"

HE_ARTIFACTS = {
    "classical": REPO / "artifacts/public_fold/classical_hardened_v1_2.json",
    "dinov2": REPO / "artifacts/public_fold/dinov2_hardened_v1_2.json",
    "siglip2": REPO / "artifacts/public_fold/siglip2_hardened_v1_2.json",
    "hibou": REPO / "artifacts/public_fold/hibou_hardened_v1_2.json",
}
PAIRED_ARTIFACT = (
    REPO / "artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json"
)
MULTIPLEX_ARTIFACT = REPO / "artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json"
MULTIPLEX_SENSITIVITY_ARTIFACT = (
    REPO / "artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json"
)
DINO_SMOKE = REPO / "artifacts/foundation_smoke/foundation_smoke.json"
SIGLIP_SMOKE = REPO / "artifacts/foundation_smoke/siglip2_base_mps_lora.json"
FEASIBILITY_MANIFEST = REPO / "artifacts/feasibility/RUN_MANIFEST.json"
QUALITATIVE_CHECKS = HERE / "qualitative_checks.json"
EVALUATION_PROTOCOL = REPO / "docs/EVALUATION.md"
PUBLIC_BENCHMARK_AUDIT = REPO / "docs/PUBLIC_BENCHMARK_AUDIT.md"
COMPLETION_MANIFEST = HERE / "BUILD_COMPLETE.json"


# Restrained, color-blind-conscious palette.  Distinctions are also encoded by
# marker shape, fill state, direct labels, and line style.
INK = HexColor("#1F2937")
MUTED = HexColor("#5F6B7A")
GRID = HexColor("#D9DEE7")
LIGHT_GRID = HexColor("#EEF1F5")
PAPER = HexColor("#FFFFFF")
BLUE = HexColor("#2F6BBA")
BLUE_DARK = HexColor("#174A8B")
BLUE_LIGHT = HexColor("#DCE9F7")
GOLD = HexColor("#C38B00")
GOLD_LIGHT = HexColor("#F4E7BF")
ORANGE = HexColor("#D76A3A")
ORANGE_LIGHT = HexColor("#F7DED2")
OLIVE = HexColor("#788A39")
PINK = HexColor("#B85C83")
NEUTRAL = HexColor("#7B8491")
NEUTRAL_LIGHT = HexColor("#E6E9EE")
REFERENCE_TEAL = HexColor("#00796B")
CLASSICAL_ORANGE = HexColor("#D55E00")
HIBOU_MAGENTA = HexColor("#9400D3")

FIGURE_METADATA = {
    "figure1_he_locked_test": (
        "H&E locked-test performance and clean-field burden",
        "Three-panel comparison of positive-field macro Dice, presence AUROC, and clean-field predicted area for seven H&E fold-detection methods.",
    ),
    "figure2_he_organ_heatmap": (
        "Organ-stratified H&E fold-localization performance",
        "Heatmap of positive-field macro Dice for seven methods across brain, kidney, liver, small intestine, and testis.",
    ),
    "figure3_he_paired_differences": (
        "Selected paired H&E macro-Dice differences",
        "Forest plot of six selected descriptive method contrasts with source-slide-cluster bootstrap intervals.",
    ),
    "figure4_multiplex_proxy": (
        "COMET and CosMx controlled-perturbation proxy evidence",
        "Fold and crack perturbation Dice plus untouched real-field alert burden for classical, nominal-reference anomaly, and hybrid branches.",
    ),
    "figure5_proxy_resolution_sensitivity": (
        "Multiplex proxy resolution sensitivity",
        "Signed change in COMET and CosMx perturbation-response Dice between 256-pixel and 896-pixel analyses.",
    ),
    "figure6_evidence_scope": (
        "Evidence boundary and next validation gate",
        "Matrix distinguishing demonstrated execution, exploratory H&E evidence, proxy-only evidence, and required internal and prospective validation.",
    ),
    "figure7_he_qualitative": (
        "Audit of regenerated locked-threshold H&E predictions",
        "Seven whole-field H&E overlays algorithmically selected without manual image review during this audit, with solid reference, dashed classical, and dotted Hibou-B contours plus a reproducibility legend.",
    ),
}


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    artifact_key: str
    method_id: str
    head: str
    supervision: str


METHODS = [
    MethodSpec(
        "classical",
        "Classical fold candidate",
        "classical",
        "classical_fold",
        "Classical",
        "Calibration labels; hand-engineered color/texture/morphology",
    ),
    MethodSpec(
        "dinov2_patchknn",
        "DINOv2-small PatchKNN",
        "dinov2",
        "foundation_patchknn",
        "PatchKNN",
        "Clean-token bank plus calibration labels",
    ),
    MethodSpec(
        "dinov2_linear",
        "DINOv2-small linear probe",
        "dinov2",
        "foundation_linear_probe",
        "Linear probe",
        "Fit masks plus calibration labels",
    ),
    MethodSpec(
        "siglip2_patchknn",
        "SigLIP2 Base PatchKNN",
        "siglip2",
        "foundation_patchknn",
        "PatchKNN",
        "Clean-token bank plus calibration labels",
    ),
    MethodSpec(
        "siglip2_linear",
        "SigLIP2 Base linear probe",
        "siglip2",
        "foundation_linear_probe",
        "Linear probe",
        "Fit masks plus calibration labels",
    ),
    MethodSpec(
        "hibou_patchknn",
        "Hibou-B PatchKNN",
        "hibou",
        "foundation_patchknn",
        "PatchKNN",
        "Clean-token bank plus calibration labels",
    ),
    MethodSpec(
        "hibou_linear",
        "Hibou-B linear probe",
        "hibou",
        "foundation_linear_probe",
        "Linear probe",
        "Fit masks plus calibration labels",
    ),
]

ORGANS = ["Brain", "Kidney", "Liver", "Small_Intestine", "Testis"]
QUALITATIVE_IMAGE_SIZE = (896, 504)
ORGAN_LABELS = {
    "Brain": "Brain",
    "Kidney": "Kidney",
    "Liver": "Liver",
    "Small_Intestine": "Small intestine",
    "Testis": "Testis",
}

METHOD_COLORS = {
    "classical": NEUTRAL,
    "dinov2_patchknn": GOLD,
    "dinov2_linear": BLUE,
    "siglip2_patchknn": GOLD,
    "siglip2_linear": BLUE,
    "hibou_patchknn": GOLD,
    "hibou_linear": BLUE,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark artifact is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object in {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Atomically publish canonical, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    _atomic_write(path, payload.encode("utf-8"))


def write_text(path: Path, value: str) -> None:
    """Atomically publish UTF-8 text with one trailing newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(path, (value.rstrip() + "\n").encode("utf-8"))


def _atomic_write(path: Path, payload: bytes) -> None:
    """Replace one package file only after its complete payload is durable."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_text_nonempty(path: Path) -> str:
    value = path.read_text(encoding="utf-8")
    if not value.strip():
        raise AssertionError(f"required source document is empty: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assert_finite_tree(value: Any, label: str) -> None:
    """Reject non-finite numeric evidence anywhere in a nested JSON value."""

    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise AssertionError(f"{label}: non-finite numeric value")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tree(child, f"{label}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            assert_finite_tree(child, f"{label}[{index}]")


def index_unique(
    rows: Iterable[dict[str, Any]], key: str, label: str
) -> dict[str, dict[str, Any]]:
    """Index evidence rows without allowing duplicate identities to collapse."""

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity = str(row[key])
        if identity in indexed:
            raise AssertionError(f"{label}: duplicate {key} {identity}")
        indexed[identity] = row
    return indexed


def dice_from_counts(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 1.0 if denominator == 0 else (2.0 * tp) / denominator


def assert_close(
    actual: float, expected: float, label: str, tolerance: float = 1e-12
) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: recomputed {actual!r} != reported {expected!r}")


def extract_he_results(
    reports: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    organ_rows: list[dict[str, Any]] = []
    reference_field_keys: list[str] | None = None
    reference_domain: list[tuple[str, str, int, bool]] | None = None
    reference_assignment_hash: str | None = None
    reference_full_manifest: tuple[tuple[str, str, str, str], ...] | None = None
    reference_leakage_audit: dict[str, Any] | None = None
    checks: list[dict[str, Any]] = []

    for spec in METHODS:
        report = reports[spec.artifact_key]
        if report.get("schema_version") != "public-fold-benchmark-1.2":
            raise AssertionError(f"{spec.key}: unexpected benchmark schema")
        if report.get("execution_status") != "complete" or not report.get(
            "report_eligible"
        ):
            raise AssertionError(
                f"{spec.key}: report is not complete and report-eligible"
            )
        split_protocol = report.get("split_protocol", {})
        if (
            split_protocol.get("group_unit") != "provided_source_slide_id"
            or split_protocol.get("full_record_coverage") is not True
            or split_protocol.get("smoke_limit_applied") is not False
        ):
            raise AssertionError(f"{spec.key}: H&E split contract differs")
        leakage_audit = report.get("leakage_audit", {})
        if (
            leakage_audit.get("passed") is not True
            or leakage_audit.get("group_unit") != "source_slide_id"
            or any(
                leakage_audit.get(field) != 0
                for field in (
                    "fit_calibration_overlap",
                    "fit_test_overlap",
                    "calibration_test_overlap",
                )
            )
        ):
            raise AssertionError(f"{spec.key}: H&E leakage audit failed")
        assignment_hash = report["split_protocol"]["assignment_manifest_sha256"]
        manifest_rows = [
            (
                role,
                item["image_filename"],
                item["image_sha256"],
                item["slide_id"],
            )
            for role, split in report["splits"].items()
            for item in split["manifest"]
        ]
        filenames = [item[1] for item in manifest_rows]
        image_hashes = [item[2] for item in manifest_rows]
        if len(filenames) != len(set(filenames)):
            raise AssertionError(f"{spec.key}: duplicate H&E image filename")
        if len(image_hashes) != len(set(image_hashes)):
            raise AssertionError(f"{spec.key}: duplicate H&E image SHA-256")
        full_manifest = tuple(sorted(manifest_rows))
        if reference_assignment_hash is None:
            reference_assignment_hash = assignment_hash
            reference_full_manifest = full_manifest
            reference_leakage_audit = leakage_audit
        elif (
            assignment_hash != reference_assignment_hash
            or full_manifest != reference_full_manifest
            or leakage_audit != reference_leakage_audit
        ):
            raise AssertionError(
                f"{spec.key}: full split manifest differs across methods"
            )
        method = report["methods"][spec.method_id]
        locked = method["locked_test"]
        intervals = method["bootstrap_ci"]["intervals"]
        interval = intervals["positive_field_macro.dice.mean"]
        micro_interval = intervals["pixel_all_fields_micro.dice"]
        auroc_interval = intervals["image.auroc"]
        auprc_interval = intervals["image.auprc"]
        clean_fp_interval = intervals["clean_burden.false_positive_pixel_fraction"]
        outcomes = method["locked_test_outcomes"]
        field_keys = [row["field_key"] for row in outcomes]
        if len(field_keys) != len(set(field_keys)):
            raise AssertionError(f"{spec.key}: duplicate locked-test field key")
        locked_manifest_names = [
            item["image_filename"] for item in report["splits"]["locked_test"]["manifest"]
        ]
        if field_keys != locked_manifest_names:
            raise AssertionError(
                f"{spec.key}: outcome rows differ from locked-test manifest"
            )
        domain = [
            (
                row["field_key"],
                row["source_slide_id"],
                int(row["label"]),
                bool(row["localization_reference_valid"]),
            )
            for row in outcomes
        ]
        if reference_field_keys is None:
            reference_field_keys = field_keys
            reference_domain = domain
        elif field_keys != reference_field_keys or domain != reference_domain:
            raise AssertionError(
                f"{spec.key}: locked test cohort/order differs across methods"
            )

        positive = [
            row
            for row in outcomes
            if int(row["label"]) == 1 and row["localization_reference_valid"]
        ]
        clean = [row for row in outcomes if int(row["label"]) == 0]
        recomputed_macro = statistics.fmean(
            dice_from_counts(int(row["tp"]), int(row["fp"]), int(row["fn"]))
            for row in positive
        )
        reported_macro = float(locked["positive_field_macro"]["dice"]["mean"])
        assert_close(recomputed_macro, reported_macro, f"{spec.key} macro Dice")

        clean_fp_fraction = sum(int(row["fp"]) for row in clean) / sum(
            int(row["n_valid"]) for row in clean
        )
        reported_clean_fp = float(
            locked["clean_burden"]["false_positive_pixel_fraction"]
        )
        assert_close(
            clean_fp_fraction, reported_clean_fp, f"{spec.key} clean FP fraction"
        )

        pooled_tp = sum(int(row["tp"]) for row in outcomes)
        pooled_fp = sum(int(row["fp"]) for row in outcomes)
        pooled_fn = sum(int(row["fn"]) for row in outcomes)
        recomputed_micro = dice_from_counts(pooled_tp, pooled_fp, pooled_fn)
        reported_micro = float(locked["pixel_all_fields_micro"]["dice"])
        assert_close(recomputed_micro, reported_micro, f"{spec.key} micro Dice")

        runtime = method["runtime"]
        row = {
            "method_key": spec.key,
            "method": spec.label,
            "head": spec.head,
            "supervision": spec.supervision,
            "macro_dice": reported_macro,
            "ci_low": float(interval["low"]),
            "ci_high": float(interval["high"]),
            "ci_resamples": int(interval["n_valid_resamples"]),
            "micro_dice": reported_micro,
            "micro_ci_low": float(micro_interval["low"]),
            "micro_ci_high": float(micro_interval["high"]),
            "presence_auroc": float(locked["image"]["auroc"]),
            "presence_auroc_ci_low": float(auroc_interval["low"]),
            "presence_auroc_ci_high": float(auroc_interval["high"]),
            "presence_auprc": float(locked["image"]["auprc"]),
            "presence_auprc_ci_low": float(auprc_interval["low"]),
            "presence_auprc_ci_high": float(auprc_interval["high"]),
            "presence_sensitivity": float(locked["image"]["sensitivity"]),
            "presence_specificity": float(locked["image"]["specificity"]),
            "clean_fp_fraction": reported_clean_fp,
            "clean_fp_percent": 100.0 * reported_clean_fp,
            "clean_fp_ci_low_percent": 100.0 * float(clean_fp_interval["low"]),
            "clean_fp_ci_high_percent": 100.0 * float(clean_fp_interval["high"]),
            "median_seconds_per_field": float(runtime["median_seconds_per_image"]),
            "n_test_fields": int(locked["image"]["n_images"]),
            "n_positive_fields": int(locked["image"]["n_positive"]),
            "n_clean_fields": int(locked["image"]["n_negative"]),
        }
        rows.append(row)
        checks.append(
            {
                "method": spec.key,
                "macro_dice_recomputed": recomputed_macro,
                "macro_dice_reported": reported_macro,
                "micro_dice_recomputed": recomputed_micro,
                "micro_dice_reported": reported_micro,
                "clean_fp_fraction_recomputed": clean_fp_fraction,
                "clean_fp_fraction_reported": reported_clean_fp,
                "n_positive_fields": len(positive),
                "n_clean_fields": len(clean),
            }
        )
        for organ in ORGANS:
            organ_metric = method["per_organ"][organ]
            organ_rows.append(
                {
                    "method_key": spec.key,
                    "method": spec.label,
                    "head": spec.head,
                    "organ_key": organ,
                    "organ": ORGAN_LABELS[organ],
                    "macro_dice": float(
                        organ_metric["positive_field_macro"]["dice"]["mean"]
                    ),
                    "n_positive_fields": int(
                        organ_metric["positive_field_macro"]["dice"]["n"]
                    ),
                    "presence_auroc": float(organ_metric["image"]["auroc"]),
                    "clean_fp_percent": 100.0
                    * float(
                        organ_metric["clean_burden"]["false_positive_pixel_fraction"]
                    ),
                }
            )

    if (
        reference_domain is None
        or reference_full_manifest is None
        or reference_leakage_audit is None
    ):
        raise AssertionError("no H&E outcome rows were loaded")
    all_image_hashes = [row[2] for row in reference_full_manifest]
    duplicate_hash_count = len(all_image_hashes) - len(set(all_image_hashes))
    if duplicate_hash_count != 0:
        raise AssertionError("duplicate H&E image hash in full split manifest")
    positive_slides = {
        slide for _, slide, label, valid in reference_domain if label == 1 and valid
    }
    clean_slides = {slide for _, slide, label, _ in reference_domain if label == 0}
    audit = {
        "cohort_identity_equal_across_all_methods": True,
        "n_test_fields": len(reference_domain),
        "n_positive_fields": sum(
            1 for _, _, label, valid in reference_domain if label == 1 and valid
        ),
        "n_clean_fields": sum(1 for _, _, label, _ in reference_domain if label == 0),
        "n_positive_source_slides": len(positive_slides),
        "n_clean_source_slides": len(clean_slides),
        "n_source_slides": len(positive_slides | clean_slides),
        "assignment_manifest_sha256": reference_assignment_hash,
        "full_split_manifest_equal_across_all_methods": True,
        "n_dataset_images": len(all_image_hashes),
        "n_unique_image_sha256": len(set(all_image_hashes)),
        "duplicate_image_sha256_count": duplicate_hash_count,
        "split_overlap_audit": reference_leakage_audit,
        "metric_spot_checks": checks,
    }
    return rows, organ_rows, audit


def validate_paired_source_identity(
    paired: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Bind paired contrasts to the exact current H&E source reports."""

    if paired.get("schema_version") != "public-fold-paired-comparison-1.0":
        raise AssertionError("unexpected paired-comparison schema")
    if paired.get("status") != "complete_exploratory_descriptive_paired_comparison":
        raise AssertionError("paired comparison is not complete")
    if paired.get("claim_scope", {}).get("descriptive_exploratory_only") is not True:
        raise AssertionError("paired comparison lost its exploratory boundary")

    report_specs = {spec.key: spec for spec in METHODS}
    embedded_reports = paired.get("reports", {})
    if set(embedded_reports) != set(report_specs):
        raise AssertionError("paired comparison source-method set differs")
    for method_key, spec in report_specs.items():
        report = reports[spec.artifact_key]
        embedded = embedded_reports[method_key]
        artifact_path = HE_ARTIFACTS[spec.artifact_key]
        if embedded.get("artifact_sha256") != sha256_file(artifact_path):
            raise AssertionError(f"paired source artifact is stale: {method_key}")
        if Path(str(embedded.get("path"))).name != artifact_path.name:
            raise AssertionError(f"paired source filename differs: {method_key}")
        if embedded.get("selected_method") != spec.method_id:
            raise AssertionError(f"paired selected method differs: {method_key}")
        method = report["methods"][spec.method_id]
        if embedded.get("method_identity") != method["method_identity"]:
            raise AssertionError(f"paired method identity differs: {method_key}")
        if embedded.get("method_identity_sha256") != canonical_sha256(
            method["method_identity"]
        ):
            raise AssertionError(f"paired method identity hash differs: {method_key}")
        expected_model_hash = (
            None
            if report.get("model_identity") is None
            else canonical_sha256(report["model_identity"])
        )
        if embedded.get("model_identity_sha256") != expected_model_hash:
            raise AssertionError(f"paired model identity differs: {method_key}")
        provenance = report["run_provenance"]
        provenance_value = provenance["value"]
        expected_provenance = {
            "schema_version": provenance["schema_version"],
            "identity_sha256": provenance["identity_sha256"],
            "execution_identity": {
                "code_sha256": canonical_sha256(provenance_value["code"]),
                "environment_sha256": canonical_sha256(
                    provenance_value["environment"]
                ),
                "method_model_sha256": canonical_sha256(
                    provenance_value["method_model"]
                ),
                "execution": provenance_value["execution"],
            },
        }
        if embedded.get("run_provenance") != expected_provenance:
            raise AssertionError(f"paired run provenance differs: {method_key}")
        macro = method["locked_test"]["positive_field_macro"]["dice"]
        embedded_macro = embedded.get("positive_field_macro_dice", {})
        assert_close(
            float(embedded_macro["point_mean"]),
            float(macro["mean"]),
            f"paired source point mean {method_key}",
        )
        if embedded_macro.get("n_fields") != macro["n"]:
            raise AssertionError(f"paired source field count differs: {method_key}")

    reference = reports["classical"]
    dataset = reference["dataset"]
    release = dataset["release_identity"]
    dataset_fields = (
        "dataset_name",
        "dataset_version",
        "license",
        "claimable_artifacts",
        "crack_reference_available",
        "data_origin",
        "empty_positive_mask_policy",
        "n_records",
        "n_slides",
    )
    dataset_identity = {
        field: dataset[field] for field in dataset_fields
    } | {
        "release_identity": {
            "identity_version": release["identity_version"],
            "canonical_identity_sha256": release["canonical_identity_sha256"],
            "identity": release["identity"],
        }
    }
    protocol = reference["split_protocol"]
    manifest_hashes = {
        role: reference["splits"][role]["manifest_sha256"]
        for role in ("fit", "calibration", "locked_test")
    }
    split_contract = {
        "protocol": protocol["protocol"],
        "group_unit": protocol["group_unit"],
        "requested_role_fractions": protocol["requested_role_fractions"],
        "full_record_coverage": True,
        "smoke_limit_applied": False,
        "assignment_manifest_sha256": protocol["assignment_manifest_sha256"],
        "manifest_sha256_by_role": manifest_hashes,
    }
    evaluation_fields = (
        "max_dimension",
        "tile_size",
        "tile_stride",
        "seed",
        "fit_fraction",
        "calibration_fraction",
        "test_fraction",
        "empty_positive_mask_policy",
        "limit_slides_per_stratum_per_split",
        "strict_public_v1",
        "validate_asset_dimensions",
        "hash_assets",
        "calibration_score_sample",
        "image_score_quantile",
        "threshold_candidates",
    )
    evaluation_contract = {
        "configuration": {
            field: reference["configuration"][field] for field in evaluation_fields
        },
        "split": split_contract,
        "reference": {
            "dataset_release_identity_sha256": release[
                "canonical_identity_sha256"
            ],
            "localization_exclusion_manifest_sha256": release["identity"][
                "localization_exclusion_manifest_sha256"
            ],
            "empty_positive_mask_policy": dataset["empty_positive_mask_policy"],
        },
    }
    shared = paired.get("shared_evidence", {})
    if shared.get("dataset_identity") != dataset_identity or shared.get(
        "dataset_identity_sha256"
    ) != canonical_sha256(dataset_identity):
        raise AssertionError("paired dataset identity differs from current reports")
    if shared.get("evaluation_contract") != evaluation_contract or shared.get(
        "evaluation_contract_sha256"
    ) != canonical_sha256(evaluation_contract):
        raise AssertionError("paired evaluation contract differs from current reports")
    if shared.get("locked_test_manifest_sha256") != manifest_hashes["locked_test"]:
        raise AssertionError("paired locked-test manifest differs")

    outcomes = reference["methods"]["classical_fold"]["locked_test_outcomes"]
    index_unique(outcomes, "field_key", "paired H&E evaluation domain")
    evaluation_domain = {
        row["field_key"]: (
            int(row["n_valid"]),
            int(row["tp"]) + int(row["fn"]),
            int(row["fp"]) + int(row["tn"]),
        )
        for row in outcomes
    }
    if shared.get("per_field_evaluation_domain_sha256") != canonical_sha256(
        evaluation_domain
    ):
        raise AssertionError("paired per-field evaluation domain differs")
    positive_rows = [
        row
        for row in outcomes
        if int(row["label"]) == 1 and row["localization_reference_valid"] is True
    ]
    positive_slides: dict[str, set[str]] = {organ: set() for organ in ORGANS}
    for row in positive_rows:
        positive_slides[str(row["organ"])].add(str(row["source_slide_id"]))
    expected_positive_slides = {
        organ: len(positive_slides[organ]) for organ in ORGANS
    }
    if shared.get("n_locked_test_fields") != len(outcomes):
        raise AssertionError("paired locked-test field count differs")
    if shared.get("n_positive_localization_fields") != len(positive_rows):
        raise AssertionError("paired positive-field count differs")
    if shared.get("positive_source_slides_by_organ") != expected_positive_slides:
        raise AssertionError("paired positive source-slide counts differ")
    return {
        "schema_and_status_checked": True,
        "source_artifact_hashes_checked": len(embedded_reports),
        "method_model_run_identities_checked": len(embedded_reports),
        "shared_dataset_evaluation_and_cohort_identity_checked": True,
    }


def extract_paired_results(
    paired: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_identity_audit = validate_paired_source_identity(paired, reports)
    desired = [
        ("hibou_linear", "dinov2_linear"),
        ("dinov2_linear", "siglip2_linear"),
        ("siglip2_linear", "classical"),
        ("classical", "dinov2_patchknn"),
        ("classical", "hibou_patchknn"),
        ("dinov2_patchknn", "hibou_patchknn"),
    ]
    labels = {spec.key: spec.label for spec in METHODS}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    unordered_pairs: set[frozenset[str]] = set()
    for item in paired["paired_differences"]:
        left = str(item["left"])
        right = str(item["right"])
        if left not in labels or right not in labels or left == right:
            raise AssertionError("paired comparison contains an invalid method pair")
        pair = frozenset((left, right))
        if pair in unordered_pairs:
            raise AssertionError("paired comparison contains a duplicate method pair")
        unordered_pairs.add(pair)
        index[(left, right)] = item
        assert_finite_tree(item, f"paired comparison {left}/{right}")
    method_keys = list(labels)
    expected_pairs = {
        frozenset((method_keys[left], method_keys[right]))
        for left in range(len(method_keys))
        for right in range(left + 1, len(method_keys))
    }
    if unordered_pairs != expected_pairs:
        raise AssertionError("paired comparison is not the complete 7-method matrix")

    rows: list[dict[str, Any]] = []
    for winner, comparator in desired:
        if (winner, comparator) in index:
            item = index[(winner, comparator)]
            point = float(item["point_difference"])
            low = float(item["descriptive_bootstrap_ci"]["lower"])
            high = float(item["descriptive_bootstrap_ci"]["upper"])
        elif (comparator, winner) in index:
            item = index[(comparator, winner)]
            point = -float(item["point_difference"])
            low = -float(item["descriptive_bootstrap_ci"]["upper"])
            high = -float(item["descriptive_bootstrap_ci"]["lower"])
        else:
            raise AssertionError(
                f"missing paired contrast {winner} versus {comparator}"
            )
        rows.append(
            {
                "contrast_key": f"{winner}_minus_{comparator}",
                "contrast": f"{labels[winner]} − {labels[comparator]}",
                "short_contrast": f"{short_label(winner)} − {short_label(comparator)}",
                "point_difference": point,
                "ci_low": low,
                "ci_high": high,
                "ci_excludes_zero": low > 0.0 or high < 0.0,
            }
        )
    audit = {
        **source_identity_audit,
        "status": paired["status"],
        "estimand": paired["configuration"]["estimand"],
        "resamples": paired["configuration"]["bootstrap_resamples"],
        "seed": paired["configuration"]["seed"],
        "p_values_computed": paired["claim_scope"]["p_values_computed"],
        "superiority_claim_made": paired["claim_scope"]["superiority_claim_made"],
        "n_locked_test_fields": paired["shared_evidence"]["n_locked_test_fields"],
        "n_positive_localization_fields": paired["shared_evidence"][
            "n_positive_localization_fields"
        ],
        "n_positive_source_slide_groups": sum(
            paired["shared_evidence"]["positive_source_slides_by_organ"].values()
        ),
    }
    return rows, audit


def short_label(method_key: str) -> str:
    mapping = {
        "classical": "Classical",
        "dinov2_patchknn": "DINOv2 PatchKNN",
        "dinov2_linear": "DINOv2 linear",
        "siglip2_patchknn": "SigLIP2 PatchKNN",
        "siglip2_linear": "SigLIP2 linear",
        "hibou_patchknn": "Hibou PatchKNN",
        "hibou_linear": "Hibou linear",
    }
    return mapping[method_key]


def validate_multiplex_proxy(proxy: dict[str, Any], label: str) -> dict[str, Any]:
    """Validate the proxy evidence contract before extracting report values."""

    if proxy.get("schema_version") != "multiplex-real-background-proxy-logo-cv-v3":
        raise AssertionError(f"{label}: unexpected multiplex proxy schema")
    if (
        proxy.get("benchmark_kind")
        != "label_free_proxy_cross_validation_not_real_artifact_efficacy"
        or proxy.get("report_eligible") is not False
        or proxy.get("scientific_validation_passed") is not False
    ):
        raise AssertionError(f"{label}: multiplex scientific boundary differs")

    identity_audit = proxy.get("input_identity_audit", {})
    sources = proxy.get("sources", [])
    if (
        identity_audit.get("checked_before_split") is not True
        or identity_audit.get("source_ids_unique") is not True
        or identity_audit.get("sha256_content_digests_unique") is not True
        or identity_audit.get("canonical_source_paths_unique") is not True
        or identity_audit.get("field_count") != len(sources)
    ):
        raise AssertionError(f"{label}: source-identity audit failed")
    source_index = index_unique(sources, "source_id", f"{label} sources")
    source_hashes = [str(source["sha256"]) for source in sources]
    if len(source_hashes) != len(set(source_hashes)):
        raise AssertionError(f"{label}: duplicate source content SHA-256")
    for source in sources:
        if (
            source.get("lock_verified") is not True
            or source.get("modality") not in {"comet", "cosmx"}
            or not source.get("channel_names")
            or not source.get("native_shape")
            or not source.get("loaded_shape")
        ):
            raise AssertionError(f"{label}: incomplete locked source identity")

    coverage = proxy.get("test_group_coverage_audit", {})
    appearances = coverage.get("appearances", {})
    expected_groups = {
        modality: {str(source["group_id"]) for source in sources if source["modality"] == modality}
        for modality in ("comet", "cosmx")
    }
    if coverage.get("every_group_tested_exactly_once") is not True:
        raise AssertionError(f"{label}: test-group coverage failed")
    for modality, expected in expected_groups.items():
        observed = appearances.get(modality, {})
        if set(observed) != expected or any(count != 1 for count in observed.values()):
            raise AssertionError(f"{label}: incomplete {modality} test-group coverage")

    fold_manifests = proxy.get("fold_manifests", [])
    fold_index = index_unique(fold_manifests, "fold_id", f"{label} folds")
    expected_fold_ids = {
        f"{modality}:test={group}"
        for modality, groups in expected_groups.items()
        for group in groups
    }
    if set(fold_index) != expected_fold_ids:
        raise AssertionError(f"{label}: fold identities differ from source groups")
    for fold in fold_manifests:
        if fold.get("all_role_overlaps_empty") is not True or any(
            fold.get("role_overlap_audit", {}).get(pair) != []
            for pair in ("fit_calibration", "fit_test", "calibration_test")
        ):
            raise AssertionError(f"{label}: nonempty group-role overlap")
        for section in ("groups_by_role", "source_ids_by_role"):
            roles = fold.get(section, {})
            role_sets = [set(roles.get(role, [])) for role in ("fit", "calibration", "test")]
            if any(role_sets[left] & role_sets[right] for left, right in ((0, 1), (0, 2), (1, 2))):
                raise AssertionError(f"{label}: nonempty {section} overlap")

    methods = {"classical", "clean_reference_anomaly", "hybrid"}
    artifacts = {"fold", "crack"}
    response_rows = proxy.get("out_of_fold_test_response_rows", [])
    response_identities: set[tuple[Any, ...]] = set()
    for row in response_rows:
        identity = (
            row.get("fold_id"),
            row.get("source_id"),
            row.get("method"),
            row.get("artifact"),
            row.get("severity"),
        )
        if identity in response_identities:
            raise AssertionError(f"{label}: duplicate response row")
        response_identities.add(identity)
        source = source_index.get(str(row.get("source_id")))
        if (
            source is None
            or row.get("out_of_fold_test") is not True
            or row.get("method") not in methods
            or row.get("artifact") not in artifacts
            or row.get("modality") != source["modality"]
            or row.get("group_id") != source["group_id"]
            or row.get("cohort_id") != source["cohort_id"]
            or row.get("fold_id")
            != f"{source['modality']}:test={source['group_id']}"
            or row.get("score_definition") != "max(score(injected)-score(base),0)"
        ):
            raise AssertionError(f"{label}: invalid out-of-fold response row")
        assert_finite_tree(row, f"{label}.response")

    unmodified_rows = proxy.get("out_of_fold_test_unmodified_field_rows", [])
    unmodified_identities: set[tuple[Any, ...]] = set()
    for row in unmodified_rows:
        identity = (row.get("fold_id"), row.get("source_id"), row.get("method"))
        if identity in unmodified_identities:
            raise AssertionError(f"{label}: duplicate unmodified-field row")
        unmodified_identities.add(identity)
        source = source_index.get(str(row.get("source_id")))
        flip = row.get("horizontal_flip_consistency", {})
        if (
            source is None
            or row.get("out_of_fold_test") is not True
            or row.get("method") not in methods
            or row.get("modality") != source["modality"]
            or row.get("group_id") != source["group_id"]
            or row.get("cohort_id") != source["cohort_id"]
            or row.get("fold_id")
            != f"{source['modality']}:test={source['group_id']}"
            or flip.get("transform") != "horizontal_flip_then_inverse"
        ):
            raise AssertionError(f"{label}: invalid unmodified-field row")
        assert_finite_tree(row, f"{label}.unmodified")

    severity_rows = proxy.get("out_of_fold_test_severity_monotonicity_rows", [])
    severity_identities: set[tuple[Any, ...]] = set()
    for row in severity_rows:
        identity = (
            row.get("fold_id"),
            row.get("source_id"),
            row.get("method"),
            row.get("artifact"),
        )
        if identity in severity_identities:
            raise AssertionError(f"{label}: duplicate severity row")
        severity_identities.add(identity)
        source = source_index.get(str(row.get("source_id")))
        if (
            source is None
            or row.get("out_of_fold_test") is not True
            or row.get("method") not in methods
            or row.get("artifact") not in artifacts
            or row.get("n_severities") != len(proxy.get("config", {}).get("severities", []))
            or row.get("modality") != source["modality"]
            or row.get("group_id") != source["group_id"]
            or row.get("cohort_id") != source["cohort_id"]
        ):
            raise AssertionError(f"{label}: invalid severity row")
        assert_finite_tree(row, f"{label}.severity")

    response_summary = proxy.get("out_of_fold_group_macro", {}).get(
        "response_by_modality_method_artifact", {}
    )
    alert_summary = proxy.get("out_of_fold_group_macro", {}).get(
        "unmodified_by_modality_method", {}
    )
    expected_response_keys = {
        f"{modality}:{method}:{artifact}"
        for modality in ("comet", "cosmx")
        for method in methods
        for artifact in artifacts
    }
    expected_alert_keys = {
        f"{modality}:{method}"
        for modality in ("comet", "cosmx")
        for method in methods
    }
    if set(response_summary) != expected_response_keys or set(alert_summary) != expected_alert_keys:
        raise AssertionError(f"{label}: incomplete group-macro summary")
    assert_finite_tree(response_summary, f"{label}.response_summary")
    assert_finite_tree(alert_summary, f"{label}.alert_summary")
    return {
        "schema_and_claim_boundary_checked": True,
        "n_locked_sources": len(sources),
        "unique_source_ids_and_hashes_checked": True,
        "n_disjoint_folds": len(fold_manifests),
        "all_groups_tested_exactly_once": True,
        "n_finite_response_rows": len(response_rows),
        "n_finite_unmodified_rows": len(unmodified_rows),
        "n_finite_flip_inverse_rows": len(unmodified_rows),
        "n_finite_severity_rows": len(severity_rows),
    }


def extract_multiplex_results(
    proxy: dict[str, Any],
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    contract_audit = validate_multiplex_proxy(proxy, "multiplex proxy")
    response = proxy["out_of_fold_group_macro"]["response_by_modality_method_artifact"]
    alert = proxy["out_of_fold_group_macro"]["unmodified_by_modality_method"]
    method_labels = {
        "classical": "Classical",
        "clean_reference_anomaly": "Nominal-reference anomaly",
        "hybrid": "Hybrid",
    }
    modality_labels = {"comet": "COMET", "cosmx": "CosMx"}
    artifact_labels = {"fold": "Fold perturbation", "crack": "Crack perturbation"}

    response_rows: list[dict[str, Any]] = []
    for key, value in sorted(response.items()):
        modality, method, artifact = key.split(":")
        metric = value["group_macro_metrics"]["calibration_thresholded_dice"]
        response_rows.append(
            {
                "modality_key": modality,
                "modality": modality_labels[modality],
                "method_key": method,
                "method": method_labels[method],
                "artifact_key": artifact,
                "artifact": artifact_labels[artifact],
                "dice": float(metric["mean"]),
                "ci_low": float(metric["ci95"]["lower"]),
                "ci_high": float(metric["ci95"]["upper"]),
                "n_groups": int(metric["n_groups"]),
                "n_test_rows": int(value["n_test_rows"]),
            }
        )

    alert_rows: list[dict[str, Any]] = []
    for key, value in sorted(alert.items()):
        modality, method = key.split(":")
        metric = value["group_macro_metrics"]["unmodified_alert_burden_fraction"]
        alert_rows.append(
            {
                "modality_key": modality,
                "modality": modality_labels[modality],
                "method_key": method,
                "method": method_labels[method],
                "alert_burden": float(metric["mean"]),
                "alert_burden_percent": 100.0 * float(metric["mean"]),
                "ci_low": float(metric["ci95"]["lower"]),
                "ci_high": float(metric["ci95"]["upper"]),
                "ci_low_percent": 100.0 * float(metric["ci95"]["lower"]),
                "ci_high_percent": 100.0 * float(metric["ci95"]["upper"]),
                "n_groups": int(metric["n_groups"]),
            }
        )

    derived_rows: list[dict[str, Any]] = []
    for modality in ("comet", "cosmx"):
        for method in ("classical", "clean_reference_anomaly", "hybrid"):
            subset = [
                row
                for row in response_rows
                if row["modality_key"] == modality and row["method_key"] == method
            ]
            if len(subset) != 2:
                raise AssertionError(
                    f"expected fold and crack proxy rows for {modality}/{method}"
                )
            fold = next(row for row in subset if row["artifact_key"] == "fold")
            crack = next(row for row in subset if row["artifact_key"] == "crack")
            burden = next(
                row
                for row in alert_rows
                if row["modality_key"] == modality and row["method_key"] == method
            )
            derived_rows.append(
                {
                    "modality": modality_labels[modality],
                    "method": method_labels[method],
                    "fold_dice": fold["dice"],
                    "fold_ci_low": fold["ci_low"],
                    "fold_ci_high": fold["ci_high"],
                    "crack_dice": crack["dice"],
                    "crack_ci_low": crack["ci_low"],
                    "crack_ci_high": crack["ci_high"],
                    "untouched_alert_burden_percent": burden["alert_burden_percent"],
                    "alert_ci_low_percent": burden["ci_low_percent"],
                    "alert_ci_high_percent": burden["ci_high_percent"],
                    "interpretation": "generator-conditional proxy; not natural-artifact accuracy",
                }
            )

    source_counts = {
        "comet_fields": sum(
            1 for source in proxy["sources"] if source["modality"] == "comet"
        ),
        "cosmx_fields": sum(
            1 for source in proxy["sources"] if source["modality"] == "cosmx"
        ),
        "comet_provisional_groups": len(
            proxy["test_group_coverage_audit"]["appearances"]["comet"]
        ),
        "cosmx_provisional_groups": len(
            proxy["test_group_coverage_audit"]["appearances"]["cosmx"]
        ),
    }
    audit = {
        **contract_audit,
        **source_counts,
        "all_groups_tested_exactly_once": proxy["test_group_coverage_audit"][
            "every_group_tested_exactly_once"
        ],
        "higher_level_independence_declared": proxy["group_independence_audit"][
            "all_modalities_have_declared_group_independence"
        ],
        "report_eligible": proxy["report_eligible"],
        "scientific_validation_passed": proxy["scientific_validation_passed"],
        "mandatory_language": proxy["claim_boundary"]["mandatory_language"],
    }
    return response_rows, alert_rows, derived_rows, audit


def validate_proxy_sensitivity_identity(
    primary_proxy: dict[str, Any], sensitivity_proxy: dict[str, Any]
) -> dict[str, Any]:
    """Prove that the proxy sensitivity pair differs only in derived resolution."""

    exact_top_level = (
        "schema_version",
        "benchmark_kind",
        "config",
        "fold_construction",
        "cross_validation",
        "aggregation_contract",
        "computational_plan",
        "claim_boundary",
        "fold_dependence_warning",
        "input_identity_audit",
        "group_independence_audit",
        "test_group_coverage_audit",
        "report_eligible",
        "scientific_validation_passed",
    )
    for field in exact_top_level:
        if primary_proxy.get(field) != sensitivity_proxy.get(field):
            raise AssertionError(f"proxy sensitivity contract differs: {field}")

    def source_identity(source: dict[str, Any]) -> dict[str, Any]:
        value = copy.deepcopy(source)
        value.pop("loaded_shape", None)
        value.pop("effective_pixel_size_um_yx", None)
        return value

    primary_sources = primary_proxy.get("sources", [])
    sensitivity_sources = sensitivity_proxy.get("sources", [])
    primary_index = index_unique(primary_sources, "source_id", "896 proxy sources")
    sensitivity_index = index_unique(
        sensitivity_sources, "source_id", "256 proxy sources"
    )
    if list(primary_index) != list(sensitivity_index):
        raise AssertionError("proxy sensitivity source order or identity differs")
    for source_id in primary_index:
        primary = primary_index[source_id]
        sensitivity = sensitivity_index[source_id]
        if source_identity(primary) != source_identity(sensitivity):
            raise AssertionError(
                f"proxy sensitivity native source identity differs: {source_id}"
            )
        if max(int(value) for value in primary["loaded_shape"][-2:]) != 896:
            raise AssertionError(f"proxy primary resolution differs: {source_id}")
        if max(int(value) for value in sensitivity["loaded_shape"][-2:]) != 256:
            raise AssertionError(f"proxy sensitivity resolution differs: {source_id}")

    def fold_identity(fold: dict[str, Any]) -> dict[str, Any]:
        return {
            field: fold[field]
            for field in (
                "fold_id",
                "modality",
                "groups_by_role",
                "source_ids_by_role",
                "role_overlap_audit",
                "all_role_overlaps_empty",
            )
        }

    primary_folds = [fold_identity(fold) for fold in primary_proxy["fold_manifests"]]
    sensitivity_folds = [
        fold_identity(fold) for fold in sensitivity_proxy["fold_manifests"]
    ]
    if primary_folds != sensitivity_folds:
        raise AssertionError("proxy sensitivity fold assignments differ")
    return {
        "native_source_identity_equal": True,
        "source_order_and_hashes_equal": True,
        "n_matched_sources": len(primary_sources),
        "fold_assignments_equal": True,
        "non_resolution_contract_equal": True,
        "primary_max_dimension": 896,
        "sensitivity_max_dimension": 256,
    }


def extract_proxy_resolution_sensitivity(
    primary_proxy: dict[str, Any],
    sensitivity_proxy: dict[str, Any],
    primary_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare matched 896- and 256-pixel perturbation-response estimates."""

    identity_audit = validate_proxy_sensitivity_identity(
        primary_proxy, sensitivity_proxy
    )
    sensitivity_index = {
        (row["modality_key"], row["method_key"], row["artifact_key"]): row
        for row in sensitivity_rows
    }
    if len(sensitivity_index) != len(sensitivity_rows):
        raise AssertionError("duplicate 256-pixel proxy summary comparison")
    primary_keys = {
        (row["modality_key"], row["method_key"], row["artifact_key"])
        for row in primary_rows
    }
    if len(primary_keys) != len(primary_rows) or primary_keys != set(sensitivity_index):
        raise AssertionError("proxy summary comparison identities differ")
    rows: list[dict[str, Any]] = []
    for primary in primary_rows:
        key = (
            primary["modality_key"],
            primary["method_key"],
            primary["artifact_key"],
        )
        sensitivity = sensitivity_index.get(key)
        if sensitivity is None:
            raise AssertionError(f"missing 256-pixel sensitivity row for {key}")
        delta = primary["dice"] - sensitivity["dice"]
        rows.append(
            {
                "comparison_key": ":".join(key),
                "modality": primary["modality"],
                "method": primary["method"],
                "artifact": primary["artifact"].replace(" perturbation", ""),
                "label": (
                    f"{primary['method']} · "
                    f"{primary['artifact'].replace(' perturbation', '').lower()}"
                ),
                "dice_256": sensitivity["dice"],
                "dice_896": primary["dice"],
                "delta_896_minus_256": delta,
                "absolute_delta": abs(delta),
            }
        )
    largest = max(rows, key=lambda row: row["absolute_delta"])
    audit = {
        **identity_audit,
        "n_matched_comparisons": len(rows),
        "largest_absolute_delta": largest["absolute_delta"],
        "largest_shift_label": (
            f"{largest['modality']} {largest['method']} {largest['artifact'].lower()}"
        ),
        "largest_shift_delta_896_minus_256": largest["delta_896_minus_256"],
    }
    return rows, audit


def extract_smoke_results(
    dino: dict[str, Any], siglip: dict[str, Any]
) -> list[dict[str, Any]]:
    def validate_common(
        report: dict[str, Any], *, model_id: str, label: str
    ) -> tuple[dict[str, Any], float, float, float, dict[str, Any]]:
        if (
            report.get("schema_version") != "1.0"
            or report.get("status") != "passed"
            or report.get("scientific_validation_passed") is not False
            or report.get("model", {}).get("id") != model_id
        ):
            raise AssertionError(f"{label}: smoke identity or claim boundary differs")
        execution = report.get("execution", {})
        if execution.get("status") != "passed" or execution.get(
            "resolved_device"
        ) != "mps":
            raise AssertionError(f"{label}: smoke did not pass on MPS")
        input_record = execution.get("input", {})
        if (
            input_record.get("finite") is not True
            or input_record.get("shape") != [2, 224, 224, 3]
            or input_record.get("semantic_channels") != ["red", "green", "blue"]
        ):
            raise AssertionError(f"{label}: smoke input contract differs")
        frozen = execution.get("frozen_inference", {})
        cpu = frozen.get("cpu_reference", {})
        requested = frozen.get("requested_device", {})
        if requested.get("device") != "mps":
            raise AssertionError(f"{label}: requested-device record is not MPS")
        for device_label, result in (("cpu", cpu), ("mps", requested)):
            outputs = result.get("outputs", {})
            finite_flags = [
                value.get("finite")
                for value in outputs.values()
                if isinstance(value, dict) and "finite" in value
            ]
            if not finite_flags or any(flag is not True for flag in finite_flags):
                raise AssertionError(f"{label}: non-finite {device_label} output")
            if outputs.get("input_size") != [224, 224]:
                raise AssertionError(f"{label}: {device_label} input size differs")
        gate = frozen.get("cpu_device_agreement_gate", {})
        if (
            gate.get("passed") is not True
            or float(gate["observed_max_abs_error"])
            > float(gate["allowed_max_abs_error"])
            or float(gate["observed_min_cosine_similarity"])
            < float(gate["required_min_cosine_similarity"])
        ):
            raise AssertionError(f"{label}: CPU/MPS parity gate failed")
        cpu_seconds = float(cpu["timing"]["steady_median_seconds"])
        mps_seconds = float(requested["timing"]["steady_median_seconds"])
        max_error = float(frozen["cpu_device_agreement"]["max_abs_error"])
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (cpu_seconds, mps_seconds, max_error)
        ) or min(cpu_seconds, mps_seconds) <= 0.0:
            raise AssertionError(f"{label}: invalid timing or parity value")
        lora = execution.get("lora", {})
        if (
            lora.get("requested") is not True
            or lora.get("performed") is not True
            or lora.get("rank") != 4
            or lora.get("target_blocks") != [8, 9, 10, 11]
        ):
            raise AssertionError(f"{label}: rank-4 LoRA smoke differs")
        trainable_fraction = float(lora["trainable_fraction"])
        if not math.isfinite(trainable_fraction) or not 0.0 < trainable_fraction < 0.01:
            raise AssertionError(f"{label}: invalid LoRA trainable fraction")
        assert_finite_tree(execution, f"{label}.execution")
        return execution, cpu_seconds, mps_seconds, max_error, lora

    dino_execution, dino_cpu, dino_mps, dino_error, dino_lora = validate_common(
        dino, model_id="facebook/dinov2-small", label="DINOv2-small smoke"
    )
    if (
        dino.get("model", {}).get("requested_revision")
        != dino.get("model", {}).get("resolved_revision")
        or dino_lora.get("loss_finite") is not True
        or dino_lora.get("nonzero_trainable_weight_delta") is not True
    ):
        raise AssertionError("DINOv2-small smoke revision or update differs")

    sig_execution, sig_cpu, sig_mps, sig_error, sig_lora = validate_common(
        siglip,
        model_id="google/siglip2-base-patch16-224",
        label="SigLIP2 Base smoke",
    )
    if (
        siglip.get("engineering_smoke_test_passed") is not True
        or sig_execution.get("engineering_smoke_test_passed") is not True
        or sig_lora.get("losses_finite") is not True
        or sig_lora.get("parameter_update", {}).get("nonzero_update") is not True
    ):
        raise AssertionError("SigLIP2 Base smoke update differs")

    return [
        {
            "model": "DINOv2-small",
            "cpu_seconds_two_patches": float(dino_cpu),
            "mps_seconds_two_patches": float(dino_mps),
            "mps_speedup": float(dino_cpu / dino_mps),
            "max_abs_cpu_mps_error": dino_error,
            "lora_rank": int(dino_lora["rank"]),
            "trainable_fraction": float(dino_lora["trainable_fraction"]),
            "engineering_passed": True,
            "scientific_validation_passed": False,
        },
        {
            "model": "SigLIP2 Base",
            "cpu_seconds_two_patches": float(sig_cpu),
            "mps_seconds_two_patches": float(sig_mps),
            "mps_speedup": float(sig_cpu / sig_mps),
            "max_abs_cpu_mps_error": sig_error,
            "lora_rank": int(sig_lora["rank"]),
            "trainable_fraction": float(sig_lora["trainable_fraction"]),
            "engineering_passed": True,
            "scientific_validation_passed": False,
        },
    ]


def validate_feasibility_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the all-synthetic engineering evidence used in the scope table."""

    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "foldcrack_qc_generated_output"
        or manifest.get("status") != "complete"
        or manifest.get("engineering_smoke_test_passed") is not True
        or manifest.get("scientific_validation_passed") is not False
        or manifest.get("unique_image_count") != 36
    ):
        raise AssertionError("synthetic feasibility manifest identity differs")
    checks = manifest.get("engineering_checks", {})
    if not checks or any(value is not True for value in checks.values()):
        raise AssertionError("synthetic feasibility engineering check failed")
    diagnostics = manifest.get("metamorphic_diagnostics", {})
    if (
        diagnostics.get("passed") is not True
        or diagnostics.get("scientific_validation") is not False
    ):
        raise AssertionError("synthetic feasibility metamorphic check failed")
    modalities = {
        str(row["modality"]) for row in diagnostics.get("comparisons", [])
    }
    if modalities != {"he", "comet", "cosmx"}:
        raise AssertionError("synthetic feasibility modality coverage differs")
    config = manifest.get("config", {})
    if (
        config.get("samples_per_modality") != 12
        or int(config["samples_per_modality"]) * len(modalities)
        != int(manifest["unique_image_count"])
    ):
        raise AssertionError("synthetic feasibility image cardinality differs")
    assert_finite_tree(manifest, "synthetic feasibility manifest")
    return {
        "schema_status_and_claim_boundary_checked": True,
        "engineering_checks_passed": len(checks),
        "unique_image_count": 36,
        "modalities": sorted(modalities),
    }


def png_dimensions(path: Path) -> tuple[int, int]:
    """Validate the 8-bit RGB PNG contract and return its IHDR dimensions."""

    with path.open("rb") as handle:
        header = handle.read(29)
    if (
        len(header) != 29
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or header[24] != 8
        or header[25] != 2
        or header[26:29] != b"\x00\x00\x00"
    ):
        raise AssertionError(f"invalid 8-bit RGB qualitative PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def _qualitative_presence_category(row: dict[str, Any]) -> str:
    label = int(row["label"])
    prediction = int(row["image_prediction"])
    if label == 1 and prediction == 1:
        return "TP"
    if label == 1:
        return "FN"
    if prediction == 1:
        return "FP"
    return "TN"


def _qualitative_expected_selection(
    manifest: dict[str, dict[str, Any]],
    method_rows: tuple[dict[str, dict[str, Any]], ...],
) -> list[str]:
    """Re-derive the algorithmic case cohort from current frozen artifacts."""

    selected: list[str] = []
    for organ in ORGANS:
        candidates = [
            entry
            for entry in manifest.values()
            if entry["organ"] == organ and entry["class"] == "tissue_fold"
        ]
        if not candidates:
            raise AssertionError(f"no fold-positive qualitative candidate for {organ}")
        selected.append(
            min(candidates, key=lambda row: row["image_sha256"])["image_filename"]
        )
    for rows in method_rows:
        for category in ("FP", "FN"):
            candidates = [
                manifest[field_key]
                for field_key, row in rows.items()
                if _qualitative_presence_category(row) == category
            ]
            if not candidates:
                raise AssertionError(f"no qualitative candidate for {category}")
            field_key = min(candidates, key=lambda row: row["image_sha256"])[
                "image_filename"
            ]
            if field_key not in selected:
                selected.append(field_key)
    return selected


def validate_qualitative_checks(
    checks: dict[str, Any], reports: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Validate qualitative evidence against current sources and fail closed."""

    if checks.get("schema_version") != "he-qualitative-audit-1.1":
        raise AssertionError("unexpected qualitative-audit schema")
    if checks.get("status") != "passed":
        raise AssertionError("qualitative regeneration did not pass")
    if checks.get("n_cases") != 7 or len(checks.get("cases", [])) != 7:
        raise AssertionError("qualitative audit must contain exactly seven cases")
    if checks.get("all_outcomes_match_frozen_artifacts") is not True:
        raise AssertionError("qualitative outcomes differ from frozen artifacts")

    classical_report = reports["classical"]
    hibou_report = reports["hibou"]
    expected_inputs = {
        str(HE_ARTIFACTS["classical"].relative_to(REPO)): sha256_file(
            HE_ARTIFACTS["classical"]
        ),
        str(HE_ARTIFACTS["hibou"].relative_to(REPO)): sha256_file(
            HE_ARTIFACTS["hibou"]
        ),
    }
    if checks.get("input_hashes") != expected_inputs:
        raise AssertionError("qualitative source-artifact hashes are stale")

    dataset = checks.get("dataset", {})
    expected_dataset = {
        "root": "data/public/histology_tissue_fold_v1",
        "assignment_manifest_sha256": hibou_report["split_protocol"][
            "assignment_manifest_sha256"
        ],
        "release_identity_sha256": hibou_report["dataset"]["release_identity"][
            "canonical_identity_sha256"
        ],
    }
    if dataset != expected_dataset:
        raise AssertionError("qualitative dataset or split identity is stale")
    configuration = checks.get("configuration", {})
    if configuration != {
        "hibou_sha256": hibou_report["configuration_sha256"],
        "classical_sha256": classical_report["configuration_sha256"],
        "outcome_relevant_compatibility_checked": True,
    }:
        raise AssertionError("qualitative configuration identity is stale")
    for key in set(hibou_report["configuration"]) | set(
        classical_report["configuration"]
    ):
        if key not in {"methods", "probe_max_iterations"} and hibou_report[
            "configuration"
        ].get(key) != classical_report["configuration"].get(key):
            raise AssertionError(f"qualitative source configurations differ: {key}")

    model = checks.get("model", {})
    expected_model_fields = {
        "id": hibou_report["model_identity"]["id"],
        "weights_sha256": hibou_report["model_identity"]["weights"]["sha256"],
        "source_commit": hibou_report["model_identity"]["source"]["commit"],
        "encoder_frozen": True,
        "shallow_readout_refit_required": True,
        "calibration_rerun": False,
    }
    for field, expected in expected_model_fields.items():
        if model.get(field) != expected:
            raise AssertionError(f"qualitative model identity differs: {field}")
    if model.get("device") not in {"mps", "cpu"}:
        raise AssertionError("qualitative model device is unsupported")
    if checks.get("execution_override") != {
        "foundation_methods": ["foundation_linear_probe"],
        "patchknn_fit_skipped_as_unused": True,
        "outcome_relevant_settings_unchanged": True,
    }:
        raise AssertionError("qualitative linear-only execution override differs")

    def threshold(report: dict[str, Any], method: str, kind: str) -> float:
        section = "pixel_localization" if kind == "pixel" else "image_presence"
        return float(report["methods"][method]["thresholds"][section]["value"])

    expected_thresholds = {
        "classical": {
            "pixel": threshold(classical_report, "classical_fold", "pixel"),
            "presence": threshold(classical_report, "classical_fold", "presence"),
        },
        "hibou_linear": {
            "pixel": threshold(hibou_report, "foundation_linear_probe", "pixel"),
            "presence": threshold(hibou_report, "foundation_linear_probe", "presence"),
        },
    }
    if checks.get("thresholds") != expected_thresholds:
        raise AssertionError("qualitative thresholds differ from frozen artifacts")

    expected_encoding = {
        "reference": {
            "color_rgb": [0, 121, 107],
            "line_style": "solid",
            "line_width_px": 3,
            "white_halo_width_px": 7,
            "draw_order": "last",
        },
        "classical": {
            "color_rgb": [213, 94, 0],
            "line_style": "dashed",
            "dash_on_off_px": [6, 4],
            "line_width_px": 2,
            "opacity": 0.72,
        },
        "hibou_linear": {
            "color_rgb": [148, 0, 211],
            "line_style": "dotted",
            "dash_on_off_px": [1, 3],
            "line_width_px": 2,
            "opacity": 0.88,
        },
    }
    if checks.get("overlay_encoding") != expected_encoding:
        raise AssertionError("qualitative overlay encoding differs")

    provenance = checks.get("overlay_generation_provenance", {})
    overlay_script = HERE / "generate_qualitative_overlays.py"
    if provenance.get("script_sha256") != sha256_file(overlay_script):
        raise AssertionError("qualitative overlay script changed after regeneration")
    runtime_sources = (
        REPO / "src/foldcrack_qc/public_fold_benchmark.py",
        REPO / "src/foldcrack_qc/public_fold_providers.py",
    )
    expected_runtime_hashes = {
        str(path.relative_to(REPO)): sha256_file(path) for path in runtime_sources
    }
    if provenance.get("runtime_source_sha256") != expected_runtime_hashes:
        raise AssertionError("qualitative runtime source changed after regeneration")
    runtime_diff = subprocess.run(
        [
            "git",
            "diff",
            "--binary",
            "HEAD",
            "--",
            *(str(path.relative_to(REPO)) for path in runtime_sources),
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    if provenance.get("tracked_runtime_diff_sha256") != hashlib.sha256(
        runtime_diff
    ).hexdigest() or provenance.get("tracked_runtime_dirty") is not bool(runtime_diff):
        raise AssertionError("qualitative tracked-runtime provenance is stale")
    if provenance.get("frozen_run_provenance") != hibou_report["run_provenance"]:
        raise AssertionError("qualitative frozen-run provenance differs")
    if provenance.get("environment", {}).get("device") != model["device"]:
        raise AssertionError("qualitative device provenance differs")

    training = checks.get("training_identity_check", {})
    for field in (
        "token_counts_match",
        "optimizer_status_iterations_match",
        "final_loss_match",
        "patchknn_fit_skipped_as_unused",
    ):
        if training.get(field) is not True:
            raise AssertionError(f"qualitative training identity failed: {field}")
    observed_training = training.get("observed_training_statistics", {})
    expected_training = hibou_report["foundation_training"]
    for field in (
        "probe_negative_tokens_seen",
        "probe_positive_tokens_seen",
        "probe_tokens_stored_per_class",
    ):
        if observed_training.get(field) != expected_training[field]:
            raise AssertionError(f"qualitative observed training differs: {field}")
    observed_optimization = observed_training.get("probe_optimization", {})
    expected_optimization = expected_training["probe_optimization"]
    for field in (
        "success",
        "status",
        "iterations",
        "function_evaluations",
    ):
        if observed_optimization.get(field) != expected_optimization[field]:
            raise AssertionError(f"qualitative optimizer identity differs: {field}")
    assert_close(
        float(observed_optimization["final_loss"]),
        float(expected_optimization["final_loss"]),
        "qualitative optimizer final loss",
        1e-10,
    )
    if training.get("iterations") != expected_optimization["iterations"]:
        raise AssertionError("qualitative top-level optimizer iterations differ")
    assert_close(
        float(training["final_loss"]),
        float(expected_optimization["final_loss"]),
        "qualitative top-level optimizer final loss",
        1e-10,
    )

    hibou_manifest_rows = hibou_report["splits"]["locked_test"]["manifest"]
    classical_manifest_rows = classical_report["splits"]["locked_test"]["manifest"]
    manifest = index_unique(
        hibou_manifest_rows, "image_filename", "qualitative Hibou manifest"
    )
    index_unique(
        classical_manifest_rows, "image_filename", "qualitative classical manifest"
    )
    if hibou_manifest_rows != classical_manifest_rows:
        raise AssertionError("qualitative source cohort differs across methods")
    classical_rows = index_unique(
        classical_report["methods"]["classical_fold"]["locked_test_outcomes"],
        "field_key",
        "qualitative classical outcomes",
    )
    hibou_rows = index_unique(
        hibou_report["methods"]["foundation_linear_probe"][
            "locked_test_outcomes"
        ],
        "field_key",
        "qualitative Hibou outcomes",
    )
    if set(manifest) != set(classical_rows) or set(manifest) != set(hibou_rows):
        raise AssertionError("qualitative outcome identities differ from manifest")
    expected_selection = _qualitative_expected_selection(
        manifest, (classical_rows, hibou_rows)
    )
    observed_selection = [str(case.get("field_key")) for case in checks["cases"]]
    if observed_selection != expected_selection:
        raise AssertionError("qualitative selection no longer matches frozen outcomes")

    rows: list[dict[str, Any]] = []
    method_sources = {
        "classical": (classical_rows, expected_thresholds["classical"]),
        "hibou_linear": (hibou_rows, expected_thresholds["hibou_linear"]),
    }
    seen_overlay_paths: set[Path] = set()
    seen_overlay_hashes: set[str] = set()
    for expected_order, case in enumerate(checks["cases"], start=1):
        if case.get("display_order") != expected_order:
            raise AssertionError("qualitative cases are not in derived display order")
        field_key = str(case["field_key"])
        manifest_row = manifest[field_key]
        expected_label = int(classical_rows[field_key]["label"])
        expected_case_identity = {
            "organ": manifest_row["organ"],
            "label": expected_label,
            "source_slide_id": manifest_row["slide_id"],
            "image_sha256": manifest_row["image_sha256"],
            "mask_sha256": manifest_row["mask_sha256"],
        }
        for field, expected in expected_case_identity.items():
            if case.get(field) != expected:
                raise AssertionError(f"{field_key}: qualitative {field} differs")

        for method, (source_rows, method_thresholds) in method_sources.items():
            method_check = case.get(method, {})
            source = source_rows[field_key]
            if method_check.get("counts_match") is not True:
                raise AssertionError(f"{field_key}: {method} count check failed")
            if method_check.get("image_score_within_tolerance") is not True:
                raise AssertionError(f"{field_key}: {method} score check failed")
            if method_check.get("image_prediction_match") is not True:
                raise AssertionError(f"{field_key}: {method} presence check failed")
            expected_counts = {
                key: int(source[key]) for key in ("tp", "fp", "fn", "tn", "n_valid")
            }
            if method_check.get("observed_counts") != expected_counts:
                raise AssertionError(f"{field_key}: {method} observed counts differ")
            if method_check.get("stored_image_score") != float(source["image_score"]):
                raise AssertionError(f"{field_key}: {method} stored score differs")
            observed_score = float(method_check["observed_image_score"])
            if not math.isfinite(observed_score):
                raise AssertionError(f"{field_key}: {method} score is non-finite")
            difference = abs(observed_score - float(source["image_score"]))
            tolerance = max(
                2e-7,
                2e-6 * max(abs(observed_score), abs(float(source["image_score"]))),
            )
            assert_close(
                float(method_check["image_score_absolute_difference"]),
                difference,
                f"{field_key}: {method} score difference",
                1e-15,
            )
            assert_close(
                float(method_check["image_score_absolute_tolerance"]),
                tolerance,
                f"{field_key}: {method} score tolerance",
                1e-15,
            )
            if difference > tolerance:
                raise AssertionError(f"{field_key}: {method} score exceeds tolerance")
            prediction = int(observed_score >= method_thresholds["presence"])
            if prediction != int(source["image_prediction"]):
                raise AssertionError(f"{field_key}: {method} presence differs")
            category = _qualitative_presence_category(source)
            if method_check.get("presence_category") != category:
                raise AssertionError(f"{field_key}: {method} category differs")
            denominator = 2 * int(source["tp"]) + int(source["fp"]) + int(source["fn"])
            expected_dice = (
                None
                if expected_label == 0 or denominator == 0
                else 2.0 * int(source["tp"]) / denominator
            )
            if expected_dice is None:
                if method_check.get("pixel_dice") is not None:
                    raise AssertionError(f"{field_key}: clean-field Dice must be null")
            else:
                assert_close(
                    float(method_check["pixel_dice"]),
                    expected_dice,
                    f"{field_key}: {method} pixel Dice",
                )

        relative_overlay = Path(str(case["overlay_path"]))
        if (
            relative_overlay.is_absolute()
            or len(relative_overlay.parts) != 2
            or relative_overlay.parts[0] != "qualitative_cache"
            or relative_overlay.suffix.lower() != ".png"
        ):
            raise AssertionError(f"unsafe qualitative overlay path: {relative_overlay}")
        overlay_path = HERE / relative_overlay
        if overlay_path.is_symlink() or not overlay_path.is_file():
            raise FileNotFoundError(
                f"qualitative overlay is missing or linked: {overlay_path}"
            )
        cache_root = (HERE / "qualitative_cache").resolve(strict=True)
        if overlay_path.resolve(strict=True).parent != cache_root:
            raise AssertionError(f"qualitative overlay escaped cache: {overlay_path}")
        overlay_sha256 = str(case.get("overlay_sha256"))
        if sha256_file(overlay_path) != overlay_sha256:
            raise AssertionError(f"qualitative overlay hash differs: {field_key}")
        expected_filename = (
            f"case{expected_order}_{Path(field_key).stem}_{overlay_sha256[:12]}.png"
        )
        if overlay_path.name != expected_filename:
            raise AssertionError(
                f"qualitative overlay is not bound to its case: {field_key}"
            )
        if relative_overlay in seen_overlay_paths or overlay_sha256 in seen_overlay_hashes:
            raise AssertionError(f"duplicate qualitative overlay: {field_key}")
        seen_overlay_paths.add(relative_overlay)
        seen_overlay_hashes.add(overlay_sha256)
        if png_dimensions(overlay_path) != QUALITATIVE_IMAGE_SIZE:
            raise AssertionError(f"qualitative overlay dimensions differ: {field_key}")

        rows.append(
            {
                "case": chr(64 + expected_order),
                "organ": ORGAN_LABELS[str(case["organ"])],
                "reference": "Fold" if expected_label == 1 else "Clean",
                "classical_presence": case["classical"]["presence_category"],
                "classical_pixel_dice": case["classical"]["pixel_dice"],
                "hibou_presence": case["hibou_linear"]["presence_category"],
                "hibou_pixel_dice": case["hibou_linear"]["pixel_dice"],
                "source_slide_id": case["source_slide_id"],
                "image_sha256": case["image_sha256"],
                "overlay_path": str(relative_overlay),
            }
        )
    return rows


def interpolate_color(low: Color, high: Color, fraction: float) -> Color:
    fraction = min(1.0, max(0.0, fraction))
    return Color(
        low.red + (high.red - low.red) * fraction,
        low.green + (high.green - low.green) * fraction,
        low.blue + (high.blue - low.blue) * fraction,
    )


def add_text(
    drawing: Drawing,
    x: float,
    y: float,
    text: str,
    *,
    size: float = 8,
    color: Color = INK,
    anchor: str = "start",
    font: str = "Helvetica",
) -> None:
    drawing.add(
        String(
            x,
            y,
            text,
            fontName=font,
            fontSize=size,
            fillColor=color,
            textAnchor=anchor,
        )
    )


def add_title(drawing: Drawing, title: str, subtitle: str | None = None) -> None:
    add_text(drawing, 16, drawing.height - 21, title, size=12, font="Helvetica-Bold")
    if subtitle:
        add_text(drawing, 16, drawing.height - 36, subtitle, size=7.5, color=MUTED)


def add_panel_label(drawing: Drawing, x: float, y: float, label: str) -> None:
    add_text(drawing, x, y, label, size=10, font="Helvetica-Bold")


def draw_marker(
    drawing: Drawing, x: float, y: float, head: str, color: Color, size: float = 4.2
) -> None:
    if head == "Linear probe":
        drawing.add(
            Rect(
                x - size,
                y - size,
                2 * size,
                2 * size,
                fillColor=color,
                strokeColor=INK,
                strokeWidth=0.55,
            )
        )
    elif head == "PatchKNN":
        drawing.add(
            Circle(
                x,
                y,
                size,
                fillColor=PAPER,
                strokeColor=color,
                strokeWidth=1.8,
            )
        )
    else:
        drawing.add(
            Polygon(
                [
                    x,
                    y + size + 0.5,
                    x + size + 0.5,
                    y,
                    x,
                    y - size - 0.5,
                    x - size - 0.5,
                    y,
                ],
                fillColor=color,
                strokeColor=INK,
                strokeWidth=0.55,
            )
        )


def save_drawing(drawing: Drawing, stem: str) -> dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    try:
        title, description = FIGURE_METADATA[stem]
    except KeyError as error:
        raise AssertionError(f"missing manuscript metadata for {stem}") from error
    paths = {
        "svg": FIGURES / f"{stem}.svg",
        "pdf": FIGURES / f"{stem}.pdf",
        "png": FIGURES / f"{stem}.png",
    }
    renderSVG.drawToFile(drawing, str(paths["svg"]))
    svg_text = paths["svg"].read_text(encoding="utf-8")
    svg_text = svg_text.replace(
        "<title>...</title>", f"<title>{xml_escape(title)}</title>", 1
    ).replace("<desc>...</desc>", f"<desc>{xml_escape(description)}</desc>", 1)
    paths["svg"].write_text(svg_text, encoding="utf-8")
    # ``invariant=1`` suppresses wall-clock PDF metadata/IDs so manuscript
    # exports are byte-for-byte reproducible from the same audited inputs.
    pdf = Canvas(
        str(paths["pdf"]),
        pagesize=(drawing.width, drawing.height),
        invariant=1,
    )
    pdf.setTitle(title)
    pdf.setSubject(description)
    pdf.setAuthor("FoldCrackArtifact benchmark report generator")
    renderPDF.draw(drawing, pdf, 0, 0)
    pdf.showPage()
    pdf.save()
    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "300",
            "-singlefile",
            str(paths["pdf"]),
            str(paths["png"].with_suffix("")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return {kind: str(path.relative_to(HERE)) for kind, path in paths.items()}


def figure_he_performance(rows: list[dict[str, Any]]) -> dict[str, str]:
    drawing = Drawing(700, 350)
    drawing.add(
        Rect(0, 0, drawing.width, drawing.height, fillColor=PAPER, strokeColor=None)
    )
    ordered = sorted(rows, key=lambda row: row["macro_dice"], reverse=True)
    add_title(
        drawing,
        "H&E locked-test performance and clean-field burden",
        "424 real microscope fields (245 fold-positive, 179 clean); identical split and calibration protocol",
    )

    plot_top = 267
    plot_bottom = 72
    step = (plot_top - plot_bottom) / (len(ordered) - 1)
    ys = [plot_top - index * step for index in range(len(ordered))]

    # Panel A: macro Dice with cluster-bootstrap intervals.
    ax_x0, ax_x1 = 143, 365
    add_panel_label(drawing, 16, 292, "A")
    add_text(
        drawing, 35, 292, "Positive-field macro Dice", size=9, font="Helvetica-Bold"
    )
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = ax_x0 + (tick / 0.85) * (ax_x1 - ax_x0)
        drawing.add(Line(x, 60, x, 275, strokeColor=LIGHT_GRID, strokeWidth=0.6))
        add_text(drawing, x, 48, f"{tick:.1f}", size=7, color=MUTED, anchor="middle")
    drawing.add(Line(ax_x0, 60, ax_x1, 60, strokeColor=INK, strokeWidth=0.75))
    for row, y in zip(ordered, ys, strict=True):
        add_text(drawing, 135, y - 2.5, row["method"], size=7.2, anchor="end")
        low_x = ax_x0 + (row["ci_low"] / 0.85) * (ax_x1 - ax_x0)
        high_x = ax_x0 + (row["ci_high"] / 0.85) * (ax_x1 - ax_x0)
        point_x = ax_x0 + (row["macro_dice"] / 0.85) * (ax_x1 - ax_x0)
        drawing.add(Line(low_x, y, high_x, y, strokeColor=INK, strokeWidth=1.15))
        drawing.add(Line(low_x, y - 3, low_x, y + 3, strokeColor=INK, strokeWidth=0.8))
        drawing.add(
            Line(high_x, y - 3, high_x, y + 3, strokeColor=INK, strokeWidth=0.8)
        )
        draw_marker(drawing, point_x, y, row["head"], METHOD_COLORS[row["method_key"]])
        add_text(
            drawing,
            high_x + 4,
            y - 2.5,
            f"{row['macro_dice']:.3f}",
            size=6.6,
            color=MUTED,
        )

    # Panel B: presence AUROC; focused scale with explicit cue.
    bx0, bx1 = 421, 530
    add_panel_label(drawing, 385, 292, "B")
    add_text(drawing, 404, 292, "Presence AUROC", size=9, font="Helvetica-Bold")
    add_text(drawing, 404, 280, "focused 0.65–1.00 scale", size=6.5, color=MUTED)
    for tick in (0.65, 0.75, 0.85, 0.95, 1.00):
        x = bx0 + ((tick - 0.65) / 0.35) * (bx1 - bx0)
        drawing.add(Line(x, 60, x, 275, strokeColor=LIGHT_GRID, strokeWidth=0.6))
        add_text(drawing, x, 48, f"{tick:.2f}", size=6.5, color=MUTED, anchor="middle")
    drawing.add(Line(bx0, 60, bx1, 60, strokeColor=INK, strokeWidth=0.75))
    for row, y in zip(ordered, ys, strict=True):
        low_x = bx0 + ((row["presence_auroc_ci_low"] - 0.65) / 0.35) * (bx1 - bx0)
        high_x = bx0 + ((row["presence_auroc_ci_high"] - 0.65) / 0.35) * (bx1 - bx0)
        x = bx0 + ((row["presence_auroc"] - 0.65) / 0.35) * (bx1 - bx0)
        drawing.add(Line(low_x, y, high_x, y, strokeColor=INK, strokeWidth=0.9))
        drawing.add(
            Line(low_x, y - 2.4, low_x, y + 2.4, strokeColor=INK, strokeWidth=0.7)
        )
        drawing.add(
            Line(high_x, y - 2.4, high_x, y + 2.4, strokeColor=INK, strokeWidth=0.7)
        )
        draw_marker(
            drawing, x, y, row["head"], METHOD_COLORS[row["method_key"]], size=3.6
        )

    # Panel C: clean-field false-positive area on a log scale.
    cx0, cx1 = 582, 684
    add_panel_label(drawing, 547, 292, "C")
    add_text(drawing, 566, 292, "Clean FP area", size=9, font="Helvetica-Bold")
    add_text(drawing, 566, 280, "percent; log scale", size=6.5, color=MUTED)
    log_min, log_max = math.log10(0.02), math.log10(20.0)
    for tick in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
        x = cx0 + ((math.log10(tick) - log_min) / (log_max - log_min)) * (cx1 - cx0)
        drawing.add(Line(x, 60, x, 275, strokeColor=LIGHT_GRID, strokeWidth=0.6))
        label = f"{tick:g}"
        add_text(drawing, x, 48, label, size=6.2, color=MUTED, anchor="middle")
    drawing.add(Line(cx0, 60, cx1, 60, strokeColor=INK, strokeWidth=0.75))
    for row, y in zip(ordered, ys, strict=True):
        low_x = cx0 + (
            (math.log10(max(row["clean_fp_ci_low_percent"], 0.0201)) - log_min)
            / (log_max - log_min)
        ) * (cx1 - cx0)
        high_x = cx0 + (
            (math.log10(max(row["clean_fp_ci_high_percent"], 0.0201)) - log_min)
            / (log_max - log_min)
        ) * (cx1 - cx0)
        x = cx0 + (
            (math.log10(max(row["clean_fp_percent"], 0.0201)) - log_min)
            / (log_max - log_min)
        ) * (cx1 - cx0)
        drawing.add(Line(low_x, y, high_x, y, strokeColor=INK, strokeWidth=0.9))
        drawing.add(
            Line(low_x, y - 2.4, low_x, y + 2.4, strokeColor=INK, strokeWidth=0.7)
        )
        drawing.add(
            Line(high_x, y - 2.4, high_x, y + 2.4, strokeColor=INK, strokeWidth=0.7)
        )
        draw_marker(
            drawing, x, y, row["head"], METHOD_COLORS[row["method_key"]], size=3.6
        )

    # Non-color legend and evidence note.
    draw_marker(drawing, 156, 25, "Linear probe", BLUE, size=3.4)
    add_text(drawing, 164, 22.8, "linear probe", size=6.5, color=MUTED)
    draw_marker(drawing, 235, 25, "PatchKNN", GOLD, size=3.4)
    add_text(drawing, 243, 22.8, "PatchKNN", size=6.5, color=MUTED)
    draw_marker(drawing, 310, 25, "Classical", NEUTRAL, size=3.4)
    add_text(drawing, 318, 22.8, "classical", size=6.5, color=MUTED)
    add_text(
        drawing,
        684,
        22.8,
        "Whiskers: 95% source-slide-cluster bootstrap CI (1,000 resamples)",
        size=6.4,
        color=MUTED,
        anchor="end",
    )
    return save_drawing(drawing, "figure1_he_locked_test")


def figure_organ_heatmap(
    rows: list[dict[str, Any]], organ_rows: list[dict[str, Any]]
) -> dict[str, str]:
    drawing = Drawing(650, 350)
    add_title(
        drawing,
        "Organ-stratified H&E fold-localization point estimates",
        "Focused 0–0.85 color scale; 42–57 positive fields and 3–9 positive source-slide groups per organ",
    )
    by_key = {(row["method_key"], row["organ_key"]): row for row in organ_rows}
    ordered = sorted(rows, key=lambda row: row["macro_dice"], reverse=True)
    left, bottom = 185, 76
    cell_w, cell_h = 80, 31
    for col, organ in enumerate(ORGANS):
        add_text(
            drawing,
            left + col * cell_w + cell_w / 2,
            bottom + len(ordered) * cell_h + 11,
            ORGAN_LABELS[organ],
            size=7.4,
            anchor="middle",
            font="Helvetica-Bold",
        )
    for row_index, method in enumerate(ordered):
        y = bottom + (len(ordered) - 1 - row_index) * cell_h
        add_text(
            drawing,
            left - 9,
            y + cell_h / 2 - 2.5,
            method["method"],
            size=7.3,
            anchor="end",
        )
        for col, organ in enumerate(ORGANS):
            value = by_key[(method["method_key"], organ)]["macro_dice"]
            fraction = value / 0.85
            fill = interpolate_color(HexColor("#F2F6FB"), BLUE_DARK, fraction)
            x = left + col * cell_w
            drawing.add(
                Rect(
                    x,
                    y,
                    cell_w - 1,
                    cell_h - 1,
                    fillColor=fill,
                    strokeColor=PAPER,
                    strokeWidth=0.7,
                )
            )
            text_color = PAPER if fraction > 0.60 else INK
            add_text(
                drawing,
                x + cell_w / 2,
                y + cell_h / 2 - 3,
                f"{value:.3f}",
                size=7.6,
                color=text_color,
                anchor="middle",
                font="Helvetica-Bold" if fraction > 0.72 else "Helvetica",
            )

    # Compact sequential legend.
    legend_x, legend_y, legend_w = 185, 39, 400
    segments = 80
    for index in range(segments):
        frac = index / (segments - 1)
        drawing.add(
            Rect(
                legend_x + index * legend_w / segments,
                legend_y,
                legend_w / segments + 0.2,
                7,
                fillColor=interpolate_color(HexColor("#F2F6FB"), BLUE_DARK, frac),
                strokeColor=None,
            )
        )
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        x = legend_x + (tick / 0.85) * legend_w
        add_text(
            drawing,
            x,
            legend_y - 10,
            f"{tick:.1f}",
            size=6.5,
            color=MUTED,
            anchor="middle",
        )
    add_text(drawing, 596, legend_y - 10, "Dice", size=6.5, color=MUTED)
    add_text(
        drawing,
        16,
        18,
        "Point estimates only; no per-organ interval or heterogeneity test. Differences may reflect tissue, source-slide, annotation, or acquisition.",
        size=6.7,
        color=MUTED,
    )
    return save_drawing(drawing, "figure2_he_organ_heatmap")


def figure_paired_differences(rows: list[dict[str, Any]]) -> dict[str, str]:
    drawing = Drawing(650, 335)
    add_title(
        drawing,
        "Selected paired differences on the identical H&E locked test",
        "Difference in positive-field macro Dice; positive values favor the first named method",
    )
    x0, x1 = 255, 625
    y_top, y_bottom = 256, 76
    step = (y_top - y_bottom) / (len(rows) - 1)
    x_min, x_max = -0.04, 0.27
    for tick in (-0.04, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25):
        x = x0 + ((tick - x_min) / (x_max - x_min)) * (x1 - x0)
        color = INK if tick == 0 else LIGHT_GRID
        width = 1.0 if tick == 0 else 0.6
        drawing.add(Line(x, 61, x, 270, strokeColor=color, strokeWidth=width))
        add_text(drawing, x, 48, f"{tick:+.2f}", size=6.7, color=MUTED, anchor="middle")
    drawing.add(Line(x0, 61, x1, 61, strokeColor=INK, strokeWidth=0.75))

    for index, row in enumerate(rows):
        y = y_top - index * step
        add_text(
            drawing, x0 - 10, y - 2.7, row["short_contrast"], size=7.2, anchor="end"
        )
        low_x = x0 + ((row["ci_low"] - x_min) / (x_max - x_min)) * (x1 - x0)
        high_x = x0 + ((row["ci_high"] - x_min) / (x_max - x_min)) * (x1 - x0)
        point_x = x0 + ((row["point_difference"] - x_min) / (x_max - x_min)) * (x1 - x0)
        color = BLUE if row["ci_excludes_zero"] else GOLD
        drawing.add(Line(low_x, y, high_x, y, strokeColor=INK, strokeWidth=1.15))
        drawing.add(Line(low_x, y - 3, low_x, y + 3, strokeColor=INK, strokeWidth=0.8))
        drawing.add(
            Line(high_x, y - 3, high_x, y + 3, strokeColor=INK, strokeWidth=0.8)
        )
        drawing.add(
            Circle(point_x, y, 4.1, fillColor=color, strokeColor=INK, strokeWidth=0.55)
        )
        add_text(
            drawing,
            min(high_x + 6, 623),
            y - 2.6,
            f"{row['point_difference']:+.3f}",
            size=6.8,
            color=MUTED,
            anchor="end" if high_x + 44 > 625 else "start",
        )
    add_text(
        drawing,
        16,
        29,
        "Display set: adjacent linear/classical ranks, classical versus the top two PatchKNN estimates, and the top-two PatchKNN contrast.",
        size=6.5,
        color=MUTED,
    )
    add_text(
        drawing,
        16,
        16,
        "n=245 positive fields / 28 source-slide groups; 95% paired cluster-bootstrap intervals (10,000 resamples); descriptive only.",
        size=6.7,
        color=MUTED,
    )
    return save_drawing(drawing, "figure3_he_paired_differences")


def _draw_proxy_panel(
    drawing: Drawing,
    rows: list[dict[str, Any]],
    modality: str,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    panel: str,
) -> None:
    subset = [row for row in rows if row["modality"] == modality]
    methods = ["Classical", "Nominal-reference anomaly", "Hybrid"]
    artifacts = ["Fold perturbation", "Crack perturbation"]
    colors = {"Classical": NEUTRAL, "Nominal-reference anomaly": GOLD, "Hybrid": BLUE}
    add_panel_label(drawing, x0, y0 + height + 15, panel)
    add_text(
        drawing,
        x0 + 20,
        y0 + height + 15,
        f"{modality}: controlled perturbation Dice",
        size=8.6,
        font="Helvetica-Bold",
    )
    for tick in (0.0, 0.2, 0.4, 0.6, 0.8):
        y = y0 + (tick / 0.8) * height
        drawing.add(
            Line(x0 + 34, y, x0 + width, y, strokeColor=LIGHT_GRID, strokeWidth=0.6)
        )
        add_text(
            drawing,
            x0 + 29,
            y - 2.5,
            f"{tick:.1f}",
            size=6.3,
            color=MUTED,
            anchor="end",
        )
    plot_left = x0 + 45
    plot_w = width - 55
    group_w = plot_w / len(methods)
    bar_w = 19
    for method_index, method in enumerate(methods):
        center = plot_left + (method_index + 0.5) * group_w
        add_text(
            drawing,
            center,
            y0 - 13,
            "Nominal-ref. anomaly" if method == "Nominal-reference anomaly" else method,
            size=6.6,
            anchor="middle",
        )
        for artifact_index, artifact in enumerate(artifacts):
            row = next(
                item
                for item in subset
                if item["method"] == method and item["artifact"] == artifact
            )
            x = center + (-0.62 if artifact_index == 0 else 0.62) * bar_w - bar_w / 2
            bar_h = min(0.8, row["dice"]) / 0.8 * height
            fill = colors[method] if artifact_index == 0 else PAPER
            drawing.add(
                Rect(
                    x,
                    y0,
                    bar_w,
                    bar_h,
                    fillColor=fill,
                    strokeColor=colors[method],
                    strokeWidth=1.2,
                )
            )
            cx = x + bar_w / 2
            low_y = y0 + min(0.8, row["ci_low"]) / 0.8 * height
            high_y = y0 + min(0.8, row["ci_high"]) / 0.8 * height
            drawing.add(Line(cx, low_y, cx, high_y, strokeColor=INK, strokeWidth=0.75))
            drawing.add(
                Line(
                    cx - 2.5, low_y, cx + 2.5, low_y, strokeColor=INK, strokeWidth=0.75
                )
            )
            drawing.add(
                Line(
                    cx - 2.5,
                    high_y,
                    cx + 2.5,
                    high_y,
                    strokeColor=INK,
                    strokeWidth=0.75,
                )
            )
            add_text(
                drawing,
                cx,
                bar_h + y0 + 4,
                f"{row['dice']:.2f}",
                size=5.8,
                anchor="middle",
            )
    drawing.add(Line(x0 + 34, y0, x0 + width, y0, strokeColor=INK, strokeWidth=0.75))


def figure_multiplex_proxy(
    response_rows: list[dict[str, Any]], alert_rows: list[dict[str, Any]]
) -> dict[str, str]:
    drawing = Drawing(700, 445)
    add_title(
        drawing,
        "Real-background COMET and CosMx experiments remain proxy evidence",
        "Synthetic fold/crack perturbations were inserted into held-out real fields; natural artifacts were not labeled",
    )
    _draw_proxy_panel(
        drawing,
        response_rows,
        "COMET",
        x0=16,
        y0=242,
        width=325,
        height=120,
        panel="A",
    )
    _draw_proxy_panel(
        drawing,
        response_rows,
        "CosMx",
        x0=360,
        y0=242,
        width=325,
        height=120,
        panel="B",
    )

    # Panel C: untouched alert burden, including descriptive group-bootstrap CIs.
    x0, y0, width, height = 16, 62, 669, 115
    add_panel_label(drawing, x0, y0 + height + 18, "C")
    add_text(
        drawing,
        x0 + 20,
        y0 + height + 18,
        "Untouched real-field alert burden (not a false-positive rate)",
        size=8.6,
        font="Helvetica-Bold",
    )
    y_max = max(row["ci_high_percent"] for row in alert_rows) * 1.06
    y_max = max(25.0, math.ceil(y_max / 10.0) * 10.0)
    for tick in range(0, int(y_max) + 1, 10):
        y = y0 + (tick / y_max) * height
        drawing.add(
            Line(x0 + 35, y, x0 + width, y, strokeColor=LIGHT_GRID, strokeWidth=0.6)
        )
        add_text(
            drawing, x0 + 30, y - 2.5, f"{tick}%", size=6.4, color=MUTED, anchor="end"
        )
    methods = ["Classical", "Nominal-reference anomaly", "Hybrid"]
    modalities = ["COMET", "CosMx"]
    colors = {"Classical": NEUTRAL, "Nominal-reference anomaly": GOLD, "Hybrid": BLUE}
    plot_left, plot_w = x0 + 48, width - 60
    group_w = plot_w / len(methods)
    for method_index, method in enumerate(methods):
        center = plot_left + (method_index + 0.5) * group_w
        add_text(
            drawing,
            center,
            y0 - 13,
            "Nominal-ref. anomaly" if method == "Nominal-reference anomaly" else method,
            size=6.8,
            anchor="middle",
        )
        for modality_index, modality in enumerate(modalities):
            row = next(
                item
                for item in alert_rows
                if item["method"] == method and item["modality"] == modality
            )
            cx = center + (-14 if modality_index == 0 else 14)
            point_y = y0 + row["alert_burden_percent"] / y_max * height
            low_y = y0 + row["ci_low_percent"] / y_max * height
            high_y = y0 + min(y_max, row["ci_high_percent"]) / y_max * height
            drawing.add(Line(cx, low_y, cx, high_y, strokeColor=INK, strokeWidth=0.8))
            drawing.add(
                Line(cx - 3, low_y, cx + 3, low_y, strokeColor=INK, strokeWidth=0.8)
            )
            drawing.add(
                Line(cx - 3, high_y, cx + 3, high_y, strokeColor=INK, strokeWidth=0.8)
            )
            if modality == "COMET":
                drawing.add(
                    Circle(
                        cx,
                        point_y,
                        4.0,
                        fillColor=colors[method],
                        strokeColor=INK,
                        strokeWidth=0.6,
                    )
                )
            else:
                drawing.add(
                    Polygon(
                        [
                            cx,
                            point_y + 4.5,
                            cx + 4.5,
                            point_y,
                            cx,
                            point_y - 4.5,
                            cx - 4.5,
                            point_y,
                        ],
                        fillColor=PAPER,
                        strokeColor=colors[method],
                        strokeWidth=1.5,
                    )
                )
            add_text(
                drawing,
                cx,
                point_y + 6,
                f"{row['alert_burden_percent']:.1f}%",
                size=6.0,
                anchor="middle",
            )
    drawing.add(Line(x0 + 35, y0, x0 + width, y0, strokeColor=INK, strokeWidth=0.75))

    # Panel-specific legends keep visual channels semantically stable.
    drawing.add(Rect(18, 210, 12, 8, fillColor=NEUTRAL, strokeColor=NEUTRAL))
    add_text(
        drawing,
        34,
        210,
        "A–B: filled = fold",
        size=6.1,
        color=MUTED,
    )
    drawing.add(
        Rect(132, 210, 12, 8, fillColor=PAPER, strokeColor=NEUTRAL, strokeWidth=1.1)
    )
    add_text(
        drawing,
        148,
        210,
        "outline = crack",
        size=6.1,
        color=MUTED,
    )
    drawing.add(
        Circle(471, 214, 4, fillColor=NEUTRAL, strokeColor=INK, strokeWidth=0.6)
    )
    add_text(drawing, 480, 210, "C: COMET", size=6.1, color=MUTED)
    drawing.add(
        Polygon(
            [536, 218.5, 540.5, 214, 536, 209.5, 531.5, 214],
            fillColor=PAPER,
            strokeColor=NEUTRAL,
            strokeWidth=1.5,
        )
    )
    add_text(drawing, 546, 210, "C: CosMx", size=6.1, color=MUTED)
    add_text(
        drawing,
        16,
        18,
        "Whiskers: descriptive 95% group-bootstrap intervals. COMET n=5 provisional field groups; CosMx n=4 provisional slide/run groups; higher-level independence unverified.",
        size=6.5,
        color=MUTED,
    )
    return save_drawing(drawing, "figure4_multiplex_proxy")


def _draw_resolution_panel(
    drawing: Drawing,
    rows: list[dict[str, Any]],
    modality: str,
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    panel: str,
) -> None:
    method_order = ["Classical", "Nominal-reference anomaly", "Hybrid"]
    artifact_order = ["Fold", "Crack"]
    colors = {"Classical": NEUTRAL, "Nominal-reference anomaly": GOLD, "Hybrid": BLUE}
    subset = [row for row in rows if row["modality"] == modality]
    ordered = [
        next(
            row
            for row in subset
            if row["method"] == method and row["artifact"] == artifact
        )
        for method in method_order
        for artifact in artifact_order
    ]
    add_panel_label(drawing, x0, y0 + height + 18, panel)
    add_text(
        drawing,
        x0 + 20,
        y0 + height + 18,
        modality,
        size=9,
        font="Helvetica-Bold",
    )
    plot_left = x0 + 137
    plot_right = x0 + width - 8
    x_min, x_max = -0.40, 0.40
    for tick in (-0.4, -0.2, 0.0, 0.2, 0.4):
        x = plot_left + ((tick - x_min) / (x_max - x_min)) * (plot_right - plot_left)
        drawing.add(
            Line(
                x,
                y0 - 2,
                x,
                y0 + height,
                strokeColor=INK if tick == 0 else LIGHT_GRID,
                strokeWidth=1.0 if tick == 0 else 0.6,
            )
        )
        add_text(
            drawing,
            x,
            y0 - 15,
            f"{tick:+.1f}",
            size=6.5,
            color=MUTED,
            anchor="middle",
        )
    step = height / len(ordered)
    zero_x = plot_left + ((0.0 - x_min) / (x_max - x_min)) * (plot_right - plot_left)
    for index, row in enumerate(ordered):
        y = y0 + height - (index + 0.5) * step
        label = (
            "Nominal-ref. anomaly"
            if row["method"] == "Nominal-reference anomaly"
            else row["method"]
        )
        add_text(
            drawing,
            plot_left - 8,
            y - 2.4,
            f"{label} · {row['artifact'].lower()}",
            size=6.7,
            anchor="end",
        )
        delta = row["delta_896_minus_256"]
        point_x = plot_left + ((delta - x_min) / (x_max - x_min)) * (
            plot_right - plot_left
        )
        color = colors[row["method"]]
        drawing.add(Line(zero_x, y, point_x, y, strokeColor=color, strokeWidth=1.5))
        if row["artifact"] == "Fold":
            drawing.add(
                Circle(
                    point_x,
                    y,
                    4,
                    fillColor=color,
                    strokeColor=INK,
                    strokeWidth=0.55,
                )
            )
        else:
            drawing.add(
                Rect(
                    point_x - 4,
                    y - 4,
                    8,
                    8,
                    fillColor=PAPER,
                    strokeColor=color,
                    strokeWidth=1.4,
                )
            )
        add_text(
            drawing,
            point_x + 5,
            y - 2.4 if delta >= 0 else y + 5,
            f"{delta:+.3f}",
            size=6.2,
            color=MUTED,
            anchor="start",
        )


def figure_proxy_resolution_sensitivity(
    rows: list[dict[str, Any]], audit: dict[str, Any]
) -> dict[str, str]:
    drawing = Drawing(700, 395)
    add_title(
        drawing,
        "Multiplex proxy response changes with analysis resolution",
        "Change in group-macro perturbation Dice: 896-pixel analysis minus matched 256-pixel analysis",
    )
    _draw_resolution_panel(
        drawing,
        rows,
        "COMET",
        x0=16,
        y0=88,
        width=325,
        height=220,
        panel="A",
    )
    _draw_resolution_panel(
        drawing,
        rows,
        "CosMx",
        x0=360,
        y0=88,
        width=325,
        height=220,
        panel="B",
    )
    drawing.add(
        Circle(190, 47, 4, fillColor=NEUTRAL, strokeColor=INK, strokeWidth=0.55)
    )
    add_text(drawing, 199, 44.5, "fold", size=6.5, color=MUTED)
    drawing.add(
        Rect(246, 43, 8, 8, fillColor=PAPER, strokeColor=NEUTRAL, strokeWidth=1.4)
    )
    add_text(drawing, 259, 44.5, "crack", size=6.5, color=MUTED)
    add_text(
        drawing,
        684,
        44.5,
        (
            f"Largest |shift|: {audit['largest_absolute_delta']:.3f} "
            f"({audit['largest_shift_label']})"
        ),
        size=6.5,
        color=MUTED,
        anchor="end",
    )
    add_text(
        drawing,
        16,
        18,
        "Sensitivity analysis only; no interval is estimated for the paired resolution difference, and neither resolution is natural-artifact ground truth.",
        size=6.5,
        color=MUTED,
    )
    return save_drawing(drawing, "figure5_proxy_resolution_sensitivity")


def figure_evidence_scope() -> dict[str, str]:
    drawing = Drawing(700, 365)
    add_title(
        drawing,
        "Evidence boundary and the shortest path to a defensible internal benchmark",
        "A shared software platform is feasible; efficacy evidence must remain modality- and artifact-specific",
    )
    columns = [
        "Software\nexecution",
        "H&E fold\nlocalization",
        "Crack/tear\nlocalization",
        "Natural COMET/\nCosMx artifacts",
        "Operational\nsafety",
    ]
    rows = [
        (
            "Synthetic engineering smoke",
            [
                "Demonstrated",
                "Not efficacy",
                "Not efficacy",
                "Not efficacy",
                "Not measured",
            ],
        ),
        (
            "Public H&E locked test",
            [
                "Demonstrated",
                "Exploratory",
                "No labels",
                "Out of scope",
                "Not measured",
            ],
        ),
        (
            "Public multiplex proxy",
            [
                "Demonstrated",
                "Out of scope",
                "Proxy only",
                "Proxy only",
                "Not measured",
            ],
        ),
        (
            "Internal adjudicated locked test",
            [
                "Required next",
                "Required next",
                "Required next",
                "Required next",
                "Required next",
            ],
        ),
        (
            "Prospective silent validation",
            ["Required", "Monitor", "Monitor", "Monitor", "Required"],
        ),
    ]
    status_fill = {
        "Demonstrated": BLUE_LIGHT,
        "Exploratory": GOLD_LIGHT,
        "Proxy only": GOLD_LIGHT,
        "Required next": ORANGE_LIGHT,
        "Required": ORANGE_LIGHT,
        "Monitor": ORANGE_LIGHT,
        "Not efficacy": NEUTRAL_LIGHT,
        "No labels": NEUTRAL_LIGHT,
        "Out of scope": NEUTRAL_LIGHT,
        "Not measured": NEUTRAL_LIGHT,
    }
    left, bottom = 177, 73
    cell_w, cell_h = 100, 42
    for col, header in enumerate(columns):
        x = left + col * cell_w + cell_w / 2
        first, second = header.split("\n")
        add_text(
            drawing,
            x,
            bottom + len(rows) * cell_h + 27,
            first,
            size=7.0,
            anchor="middle",
            font="Helvetica-Bold",
        )
        add_text(
            drawing,
            x,
            bottom + len(rows) * cell_h + 17,
            second,
            size=7.0,
            anchor="middle",
            font="Helvetica-Bold",
        )
    for row_index, (label, statuses) in enumerate(rows):
        y = bottom + (len(rows) - 1 - row_index) * cell_h
        add_text(drawing, left - 9, y + cell_h / 2 - 2.5, label, size=7.2, anchor="end")
        for col, status in enumerate(statuses):
            x = left + col * cell_w
            drawing.add(
                Rect(
                    x,
                    y,
                    cell_w - 1,
                    cell_h - 1,
                    fillColor=status_fill[status],
                    strokeColor=PAPER,
                    strokeWidth=0.8,
                )
            )
            add_text(
                drawing,
                x + cell_w / 2,
                y + cell_h / 2 - 2.5,
                status,
                size=6.8,
                anchor="middle",
                font="Helvetica-Bold"
                if status
                in {
                    "Demonstrated",
                    "Exploratory",
                    "Proxy only",
                    "Required next",
                    "Required",
                }
                else "Helvetica",
            )
    add_text(
        drawing,
        16,
        29,
        "Next decision gate: dual-reviewer ontology pilot → group-disjoint development/calibration → untouched internal test → prospective workflow validation.",
        size=7.1,
        color=INK,
        font="Helvetica-Bold",
    )
    add_text(
        drawing,
        16,
        17,
        "A pooled three-modality score must not compensate for failure in any required modality, artifact subtype, or acquisition stratum.",
        size=6.6,
        color=MUTED,
    )
    return save_drawing(drawing, "figure6_evidence_scope")


def figure_he_qualitative(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Render hash-selected whole-field overlays with vector labels and legend."""

    if len(rows) != 7:
        raise AssertionError("Figure 7 requires exactly seven qualitative cases")
    # 510 pt is 7.08 inches: the intended final two-column manuscript width.
    # All reader-facing labels remain at least 7.4 pt at that placement size.
    drawing = Drawing(510, 800)
    drawing.add(
        Rect(0, 0, drawing.width, drawing.height, fillColor=PAPER, strokeColor=None)
    )
    add_text(
        drawing,
        14,
        779,
        "Audit of regenerated locked-threshold H&E predictions",
        size=11.5,
        font="Helvetica-Bold",
    )
    add_text(
        drawing,
        14,
        762,
        "Solid reference · dashed classical · dotted Hibou-B; presence and pixel thresholds are separate",
        size=8.0,
        color=MUTED,
    )

    margin, gap = 14.0, 12.0
    panel_width = (drawing.width - 2 * margin - gap) / 2
    image_height = panel_width / (896.0 / 504.0)
    grid_top, row_height = 730.0, 176.0
    panel_letters = "ABCDEFG"
    for index, row in enumerate(rows):
        grid_row, grid_col = divmod(index, 2)
        x = margin + grid_col * (panel_width + gap)
        top = grid_top - grid_row * row_height
        image_y = top - 156.5
        add_panel_label(drawing, x, top - 12.0, panel_letters[index])
        add_text(
            drawing,
            x + 18.0,
            top - 12.0,
            f"{row['organ']} · {row['reference'].lower()}",
            size=8.5,
            font="Helvetica-Bold",
        )
        overlay_path = HERE / row["overlay_path"]
        drawing.add(
            DrawingImage(
                x,
                image_y,
                panel_width,
                image_height,
                str(overlay_path),
            )
        )
        drawing.add(
            Rect(
                x,
                image_y,
                panel_width,
                image_height,
                fillColor=None,
                strokeColor=GRID,
                strokeWidth=0.55,
            )
        )
        classical = row["classical_presence"]
        hibou = row["hibou_presence"]
        if row["reference"] == "Fold":
            result = (
                f"C {classical}, Dice {row['classical_pixel_dice']:.2f}  ·  "
                f"H {hibou}, Dice {row['hibou_pixel_dice']:.2f}"
            )
        else:
            result = f"C {classical}  ·  H {hibou}  ·  clean reference"
        add_text(drawing, x, image_y - 12.0, result, size=7.4, color=MUTED)

    # Panel H: the legend repeats color with line style and direct labels.
    info_top = grid_top - 3 * row_height
    info_x = margin + panel_width + gap
    add_panel_label(drawing, info_x, info_top - 12.0, "H")
    add_text(
        drawing,
        info_x + 18.0,
        info_top - 12.0,
        "Contour and metric key",
        size=8.5,
        font="Helvetica-Bold",
    )
    legend_items = [
        (REFERENCE_TEAL, 3.0, None, "Reference fold mask · solid"),
        (CLASSICAL_ORANGE, 2.0, [6, 4], "Classical prediction (C) · dashed"),
        (HIBOU_MAGENTA, 2.0, [1, 3], "Hibou-B prediction (H) · dotted"),
    ]
    for legend_index, (color, width, dash, label) in enumerate(legend_items):
        y = info_top - 42.0 - legend_index * 34.0
        if legend_index == 0:
            drawing.add(
                Line(
                    info_x + 4.0,
                    y,
                    info_x + 48.0,
                    y,
                    strokeColor=PAPER,
                    strokeWidth=6.0,
                )
            )
        drawing.add(
            Line(
                info_x + 4.0,
                y,
                info_x + 48.0,
                y,
                strokeColor=color,
                strokeWidth=width,
                strokeDashArray=dash,
            )
        )
        add_text(drawing, info_x + 58.0, y - 2.7, label, size=7.6)
    add_text(
        drawing,
        info_x + 4.0,
        info_top - 148.0,
        "TP/FP/FN/TN: image-presence call",
        size=7.4,
        color=MUTED,
    )
    add_text(
        drawing,
        info_x + 4.0,
        info_top - 164.0,
        "Dice: pixel localization on fold fields",
        size=7.4,
        color=MUTED,
    )
    return save_drawing(drawing, "figure7_he_qualitative")


def model_status_rows() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "model_or_method": "HistoQC",
            "domain": "H&E WSI QC",
            "current_evidence": "Not run in this repository",
            "access": "BSD-3-Clause-Clear",
            "decision": (
                "Add as an open operational QC baseline; define an output-to-fold "
                "mapping before localization scoring"
            ),
        },
        {
            "priority": 2,
            "model_or_method": "GrandQC",
            "domain": "H&E multiclass artifact segmentation",
            "current_evidence": "Not run; public manual test masks exist",
            "access": "Non-commercial / CC BY-NC-SA; legal review required",
            "decision": "Strong public comparator after written clearance",
        },
        {
            "priority": 3,
            "model_or_method": "DiffusionQC",
            "domain": "H&E artifact-union anomaly mapping",
            "current_evidence": "Paper only; no official public code/checkpoint found",
            "access": "Obtain Merck internal assets and exact split",
            "decision": "Internal reproduction is preferable to reimplementation",
        },
        {
            "priority": 4,
            "model_or_method": "Hibou-B",
            "domain": "Pathology tile encoder",
            "current_evidence": "Frozen PatchKNN and mask-supervised linear probe completed",
            "access": "Apache-2.0",
            "decision": "Retain as current pathology-specific H&E comparator",
        },
        {
            "priority": 5,
            "model_or_method": "DINOv2-small",
            "domain": "Generic RGB encoder",
            "current_evidence": "Frozen heads plus MPS/LoRA engineering smoke completed",
            "access": "Apache-2.0",
            "decision": "Retain as stable, permissive generic control",
        },
        {
            "priority": 6,
            "model_or_method": "SigLIP2 Base",
            "domain": "Generic vision-language RGB encoder",
            "current_evidence": "Frozen heads plus MPS/LoRA engineering smoke completed",
            "access": "Apache-2.0",
            "decision": "Retain as a permissive matched-readout comparator",
        },
        {
            "priority": 7,
            "model_or_method": "DINOv3",
            "domain": "Newer generic dense vision encoder",
            "current_evidence": "Not run",
            "access": "Meta custom license acceptance; institutional review",
            "decision": "Run same frozen heads after approval; newer is not evidence of superiority",
        },
        {
            "priority": 8,
            "model_or_method": "UNI2-h",
            "domain": "Large H&E/IHC pathology encoder",
            "current_evidence": "Not run",
            "access": "Gated CC BY-NC-ND; commercial permission required",
            "decision": "Defer until license and 681M-model resource case are justified",
        },
        {
            "priority": 9,
            "model_or_method": "CONCHv1.5",
            "domain": "Pathology vision-language encoder",
            "current_evidence": "Not run; no official CONCH v2 located",
            "access": "Gated CC BY-NC-ND; commercial permission required",
            "decision": "Use exact model ID; defer until legal and semantic-use case are clear",
        },
        {
            "priority": 10,
            "model_or_method": "KRONOS2",
            "domain": "Marker-aware multiplex spatial-proteomics encoder",
            "current_evidence": "Not run; no fold/crack localization head",
            "access": "Gated CC BY-NC-ND; commercial/derivative permission required",
            "decision": "Most aligned reviewed frozen COMET candidate after written approval and labels",
        },
    ]


def evidence_scope_rows() -> list[dict[str, Any]]:
    return [
        {
            "evidence": "Synthetic engineering smoke",
            "input": "36 generated images / 3 modality contracts",
            "reference": "Generator masks",
            "valid_claim": "Software execution, geometry, metric and failure-path checks",
            "invalid_claim": "Real-data efficacy or method ranking",
            "status": "Engineering only",
        },
        {
            "evidence": "Public H&E locked test",
            "input": "424 microscope fields from 55 supplied source-slide groups",
            "reference": "245 manual fold masks; 179 clean fields",
            "valid_claim": "Cohort-conditional H&E fold localization/presence",
            "invalid_claim": "Crack, WSI, Merck, COMET, CosMx or clinical performance",
            "status": "Exploratory real-label evidence",
        },
        {
            "evidence": "Public COMET/CosMx proxy",
            "input": "5 COMET DAPI fields; 6 CosMx morphology FOVs / 4 groups",
            "reference": "Controlled synthetic perturbations on real backgrounds",
            "valid_claim": "Perturbation recovery and untouched alert burden",
            "invalid_claim": "Natural-artifact Dice, ROC, sensitivity, specificity or FPR",
            "status": "Non-reportable proxy",
        },
        {
            "evidence": "DINOv2/SigLIP2 MPS + LoRA smoke",
            "input": "Two deterministic 224×224 engineering patches",
            "reference": "CPU/MPS numerical agreement and one optimizer step",
            "valid_claim": "Local execution and narrow PEFT feasibility",
            "invalid_claim": "LoRA efficacy or generalization",
            "status": "Engineering only",
        },
    ]


def acceptance_rows() -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "criterion": "Severe actionable artifact sensitivity",
            "proposed_gate": "95% CI lower bound ≥ 0.95",
            "cohort": "Enriched adjudicated challenge cohort",
        },
        {
            "priority": 2,
            "criterion": "Safety of automatic PASS",
            "proposed_gate": "95% CI lower bound for NPV ≥ 0.99",
            "cohort": "Production-prevalence cohort",
        },
        {
            "priority": 3,
            "criterion": "High-confidence localization precision",
            "proposed_gate": "95% CI lower bound ≥ 0.95",
            "cohort": "Adjudicated mask subset",
        },
        {
            "priority": 4,
            "criterion": "Valid-tissue overmask",
            "proposed_gate": "Mean ≤ 0.5%",
            "cohort": "Prevalence and hard-negative cohorts",
        },
        {
            "priority": 5,
            "criterion": "Review referral rate",
            "proposed_gate": "≤ 25%",
            "cohort": "Production-prevalence cohort",
        },
        {
            "priority": 6,
            "criterion": "Missing, invalid or OOD input",
            "proposed_gate": "100% routes to REVIEW",
            "cohort": "Degraded-input cohort",
        },
        {
            "priority": 7,
            "criterion": "Downstream assay effect",
            "proposed_gate": "Meets approved noninferiority margin",
            "cohort": "Paired downstream-impact cohort",
        },
    ]


def source_objects() -> list[dict[str, Any]]:
    he_files = [
        "artifacts/public_fold/classical_hardened_v1_2.json",
        "artifacts/public_fold/dinov2_hardened_v1_2.json",
        "artifacts/public_fold/siglip2_hardened_v1_2.json",
        "artifacts/public_fold/hibou_hardened_v1_2.json",
        "artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json",
    ]
    evidence_inventory_files = [
        "artifacts/feasibility/RUN_MANIFEST.json",
        *he_files,
        "artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json",
        "artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json",
        "artifacts/foundation_smoke/foundation_smoke.json",
        "artifacts/foundation_smoke/siglip2_base_mps_lora.json",
    ]
    return [
        {
            "id": "he_benchmark",
            "label": "Hardened public H&E fold benchmark artifacts",
            "path": "artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json",
            "query": {
                "engine": "local_json_audit",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    f"paths = {he_files!r}\n"
                    "reports = {path: json.loads(Path(path).read_text(encoding='utf-8')) "
                    "for path in paths}"
                ),
                "description": "Extracts and independently verifies locked-test outcomes from all report-eligible schema-v1.2 H&E artifacts.",
                "tables_used": he_files,
                "filters": [
                    "locked_test role only",
                    "localization_reference_valid=true for positive-field overlap",
                    "empty positive masks excluded under the frozen benchmark policy",
                ],
                "metric_definitions": [
                    "Positive-field macro Dice is the arithmetic mean of field-level 2TP/(2TP+FP+FN) across 245 valid fold-positive fields.",
                    "All-field micro Dice pools TP, FP and FN over 424 locked-test fields.",
                    "Clean FP area is predicted positive pixels divided by valid pixels across 179 clean fields.",
                    "Intervals resample supplied source-slide groups within organ/class strata.",
                ],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "multiplex_proxy",
            "label": "Real-background COMET/CosMx perturbation benchmark",
            "path": "artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json",
            "query": {
                "engine": "local_json_audit",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    "paths = ['artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json', "
                    "'artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json']\n"
                    "proxies = {path: json.loads(Path(path).read_text(encoding='utf-8')) "
                    "for path in paths}"
                ),
                "description": "Extracts out-of-fold group-macro controlled-perturbation response and untouched real-field alert burden.",
                "tables_used": [
                    "artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json",
                    "artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json",
                ],
                "filters": [
                    "leave-one-declared-source-group-out results",
                    "896-pixel primary sensitivity configuration",
                    "fold and crack perturbations reported separately",
                ],
                "metric_definitions": [
                    "Calibration-thresholded Dice compares paired incremental response with the inserted perturbation mask.",
                    "Untouched alert burden is the predicted-area fraction on unmodified real fields and is not a false-positive rate.",
                ],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "foundation_smoke",
            "label": "Frozen-encoder Apple MPS and rank-4 LoRA engineering smokes",
            "path": "artifacts/foundation_smoke/foundation_smoke.json",
            "query": {
                "engine": "local_json_audit",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    "paths = ['artifacts/foundation_smoke/foundation_smoke.json', "
                    "'artifacts/foundation_smoke/siglip2_base_mps_lora.json']\n"
                    "smokes = {path: json.loads(Path(path).read_text(encoding='utf-8')) "
                    "for path in paths}"
                ),
                "description": "Reads CPU/MPS parity, timing, memory and one-step adapter outcomes from deterministic engineering smokes.",
                "tables_used": [
                    "artifacts/foundation_smoke/foundation_smoke.json",
                    "artifacts/foundation_smoke/siglip2_base_mps_lora.json",
                ],
                "filters": ["engineering smoke only; no efficacy interpretation"],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "he_qualitative_audit",
            "label": "Regenerated H&E qualitative audit bundle",
            "path": "reports/benchmark_findings_2026-08-27/qualitative_checks.json",
            "query": {
                "engine": "local_json_and_binary_audit",
                "language": "python",
                "sql": (
                    "import hashlib, json\nfrom pathlib import Path\n"
                    "checks = json.loads(Path('reports/benchmark_findings_2026-08-27/qualitative_checks.json').read_text(encoding='utf-8'))\n"
                    "overlay_hashes = {case['overlay_path']: hashlib.sha256((Path('reports/benchmark_findings_2026-08-27') / case['overlay_path']).read_bytes()).hexdigest() for case in checks['cases']}"
                ),
                "description": "Validates the exact selected cases, shallow-head refit identity, locked outcomes, and content-addressed overlays used in Figure 7.",
                "tables_used": [
                    "reports/benchmark_findings_2026-08-27/qualitative_checks.json",
                    "reports/benchmark_findings_2026-08-27/qualitative_cache/*.png",
                ],
                "filters": [
                    "seven algorithmically selected whole fields",
                    "locked thresholds; no calibration rerun",
                    "counts and calls exact; image scores tolerance checked",
                ],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "evidence_inventory",
            "label": "Validated cross-tier evidence inventory",
            "path": "artifacts/feasibility/RUN_MANIFEST.json",
            "query": {
                "engine": "local_json_audit",
                "language": "python",
                "sql": (
                    "import json\nfrom pathlib import Path\n"
                    f"paths = {evidence_inventory_files!r}\n"
                    "inventory = {path: json.loads(Path(path).read_text(encoding='utf-8')) for path in paths}"
                ),
                "description": "Loads the validated synthetic, H&E, multiplex, and foundation-smoke artifacts that support the evidence-tier inventory and its counts.",
                "tables_used": evidence_inventory_files,
                "filters": ["claim status and validation boundary retained per tier"],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "evaluation_protocol",
            "label": "Locked evaluation protocol",
            "path": "docs/EVALUATION.md",
            "query": {
                "engine": "local_document",
                "language": "python",
                "sql": (
                    "from pathlib import Path\n"
                    "protocol = Path('docs/EVALUATION.md').read_text(encoding='utf-8')"
                ),
                "description": "Reads the repository evaluation protocol used to formulate the proposed internal gates.",
                "tables_used": ["docs/EVALUATION.md"],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "method_audit",
            "label": "Public data and model audit",
            "path": "docs/PUBLIC_BENCHMARK_AUDIT.md",
            "query": {
                "engine": "local_document",
                "language": "python",
                "sql": (
                    "from pathlib import Path\n"
                    "audit = Path('docs/PUBLIC_BENCHMARK_AUDIT.md').read_text(encoding='utf-8')"
                ),
                "description": "Reads the repository audit used to classify method execution and evidence status.",
                "tables_used": ["docs/PUBLIC_BENCHMARK_AUDIT.md"],
                "executed_at": REPORT_TIMESTAMP,
            },
        },
        {
            "id": "histology_fold_dataset",
            "label": "Histology Tissue Fold Dataset v1",
            "href": "https://zenodo.org/records/21493260",
        },
        {
            "id": "grandqc",
            "label": "GrandQC manual test masks and primary paper",
            "href": "https://doi.org/10.1038/s41467-024-54769-y",
        },
        {
            "id": "histoqc",
            "label": "HistoQC primary paper",
            "href": "https://doi.org/10.1200/CCI.18.00157",
        },
        {
            "id": "diffusionqc",
            "label": "DiffusionQC ISBI 2026 paper",
            "href": "https://doi.org/10.1109/ISBI61048.2026.11515418",
        },
        {
            "id": "qualifai",
            "label": "QUALIFAI public COMET release",
            "href": "https://zenodo.org/records/12699470",
        },
        {
            "id": "cosmx_gastric",
            "label": "CosMx gastric public dataset",
            "href": "https://zenodo.org/records/8333281",
        },
        {
            "id": "cosmx_phgg",
            "label": "CosMx pediatric high-grade glioma public dataset",
            "href": "https://zenodo.org/records/16877090",
        },
        {
            "id": "dinov3",
            "label": "Official DINOv3 repository and license",
            "href": "https://github.com/facebookresearch/dinov3",
        },
        {
            "id": "dinov2",
            "label": "Official DINOv2 repository and license",
            "href": "https://github.com/facebookresearch/dinov2",
        },
        {
            "id": "grandqc_repo",
            "label": "Official GrandQC repository and noncommercial license",
            "href": "https://github.com/cpath-ukk/grandqc",
        },
        {
            "id": "uni2h",
            "label": "Official UNI2-h model card",
            "href": "https://huggingface.co/MahmoodLab/UNI2-h",
        },
        {
            "id": "conch15",
            "label": "Official CONCHv1.5 model card",
            "href": "https://huggingface.co/MahmoodLab/conchv1_5",
        },
        {
            "id": "kronos2",
            "label": "Official KRONOS2 model card",
            "href": "https://huggingface.co/MahmoodLab/KRONOS2",
        },
    ]


def build_artifact(
    he_rows: list[dict[str, Any]],
    organ_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    qualitative_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    alert_rows: list[dict[str, Any]],
    proxy_summary: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best = max(he_rows, key=lambda row: row["macro_dice"])
    hibou_delta = next(
        row
        for row in paired_rows
        if row["contrast_key"] == "hibou_linear_minus_dinov2_linear"
    )
    sources = source_objects()
    comet_rows = [row for row in proxy_rows if row["modality"] == "COMET"]
    cosmx_rows = [row for row in proxy_rows if row["modality"] == "CosMx"]

    summary = [
        {
            "best_macro_dice": best["macro_dice"],
            "best_method": best["method"],
            "hibou_vs_dinov2_delta": hibou_delta["point_difference"],
            "locked_test_fields": 424,
            "labeled_natural_multiplex_fields": 0,
        }
    ]
    charts = [
        {
            "id": "he_macro_dice",
            "title": "Positive-field macro Dice by method",
            "subtitle": "Locked public H&E fold test; point estimates shown here, exact 95% intervals in the adjacent table",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "Which evaluated method has the strongest H&E fold-localization point estimate?",
            "rationale": "A horizontal ranked bar is the clearest native report view for seven long method labels; the vector manuscript figure supplies interval marks.",
            "dataset": "he_methods",
            "sourceId": "he_benchmark",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {
                    "field": "macro_dice",
                    "type": "quantitative",
                    "label": "Positive-field macro Dice",
                    "format": "number",
                },
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI low"},
                    {
                        "field": "ci_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                    },
                    {
                        "field": "clean_fp_percent",
                        "type": "quantitative",
                        "label": "Clean FP area (%)",
                    },
                ],
            },
            "comparisonContext": {
                "unit": "Dice (0–1)",
                "grain": "positive microscope field",
                "denominator": "245 valid fold-positive fields",
            },
            "layout": "full",
        },
        {
            "id": "he_organ_heatmap",
            "title": "Positive-field macro Dice by method and organ",
            "subtitle": "Point estimates reveal tissue-dependent variation that aggregate Dice obscures",
            "type": "bar",
            "intent": "comparison",
            "question": "How stable is each method across the five represented organs?",
            "rationale": "Grouped bars keep the five organ estimates visible for each method in the native report; the vector manuscript figure provides the denser matrix view.",
            "dataset": "he_organs",
            "sourceId": "he_benchmark",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {
                    "field": "macro_dice",
                    "type": "quantitative",
                    "label": "Positive-field macro Dice",
                },
                "color": {
                    "field": "organ",
                    "type": "nominal",
                    "label": "Organ",
                },
            },
            "comparisonContext": {
                "unit": "Dice (0–1)",
                "grain": "organ × method",
                "denominator": "42–57 positive fields per organ",
            },
            "layout": "full",
        },
        {
            "id": "he_paired_differences",
            "title": "Selected paired macro-Dice differences",
            "subtitle": "Positive values favor the first named method; exact descriptive intervals are in the table",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "What is the magnitude of selected within-cohort method differences?",
            "rationale": "A zero-referenced horizontal difference chart preserves sign and accommodates long contrast labels.",
            "dataset": "paired_differences",
            "sourceId": "he_benchmark",
            "encodings": {
                "x": {
                    "field": "short_contrast",
                    "type": "nominal",
                    "label": "Contrast",
                },
                "y": {
                    "field": "point_difference",
                    "type": "quantitative",
                    "label": "Macro-Dice difference",
                    "format": "number",
                },
                "tooltip": [
                    {"field": "ci_low", "type": "quantitative", "label": "95% CI low"},
                    {
                        "field": "ci_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                    },
                ],
            },
            "comparisonContext": {
                "baseline": "zero difference",
                "unit": "Dice difference",
                "grain": "paired positive field with source-slide cluster resampling",
            },
            "layout": "full",
        },
        {
            "id": "proxy_comet",
            "title": "COMET controlled-perturbation Dice",
            "subtitle": "Five public DAPI fields; generator-conditional proxy only",
            "type": "bar",
            "intent": "comparison",
            "question": "How do the three branches respond to inserted fold and crack perturbations on COMET backgrounds?",
            "rationale": "Grouped bars compare the two perturbation types across the same three methods without implying a natural-artifact test.",
            "dataset": "proxy_comet",
            "sourceId": "multiplex_proxy",
            "encodings": {
                "x": {
                    "field": "artifact",
                    "type": "nominal",
                    "label": "Inserted perturbation",
                },
                "y": {
                    "field": "dice",
                    "type": "quantitative",
                    "label": "Thresholded Dice",
                },
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
            "comparisonContext": {
                "unit": "Dice (0–1)",
                "grain": "provisional source-group macro mean",
                "denominator": "5 COMET public-field groups",
            },
            "layout": "full",
        },
        {
            "id": "proxy_cosmx",
            "title": "CosMx controlled-perturbation Dice",
            "subtitle": "Six morphology FOVs from four provisional slide/run groups; generator-conditional proxy only",
            "type": "bar",
            "intent": "comparison",
            "question": "How do the three branches respond to inserted fold and crack perturbations on CosMx backgrounds?",
            "rationale": "The same grouped-bar form as COMET enables a bounded modality comparison while keeping each modality separate.",
            "dataset": "proxy_cosmx",
            "sourceId": "multiplex_proxy",
            "encodings": {
                "x": {
                    "field": "artifact",
                    "type": "nominal",
                    "label": "Inserted perturbation",
                },
                "y": {
                    "field": "dice",
                    "type": "quantitative",
                    "label": "Thresholded Dice",
                },
                "color": {"field": "method", "type": "nominal", "label": "Method"},
            },
            "comparisonContext": {
                "unit": "Dice (0–1)",
                "grain": "provisional source-group macro mean",
                "denominator": "4 CosMx provisional slide/run groups",
            },
            "layout": "full",
        },
        {
            "id": "proxy_alert_burden",
            "title": "Untouched real-field alert burden",
            "subtitle": "Predicted area on unmodified fields; this is not a false-positive rate",
            "type": "bar",
            "intent": "comparison",
            "question": "What review burden does each branch create on untouched public multiplex fields?",
            "rationale": "Grouped bars expose modality differences and prevent the lower COMET burden from hiding the much larger CosMx anomaly/hybrid burden.",
            "dataset": "proxy_alert",
            "sourceId": "multiplex_proxy",
            "encodings": {
                "x": {"field": "method", "type": "nominal", "label": "Method"},
                "y": {
                    "field": "alert_burden",
                    "type": "quantitative",
                    "label": "Alert burden",
                    "format": "percent",
                },
                "color": {"field": "modality", "type": "nominal", "label": "Modality"},
            },
            "comparisonContext": {
                "unit": "fraction of field area",
                "grain": "provisional source-group macro mean",
                "normalization": "predicted pixels / unmodified field pixels",
            },
            "layout": "full",
        },
        {
            "id": "proxy_resolution_sensitivity",
            "title": "Multiplex proxy Dice shift from 256- to 896-pixel analysis",
            "subtitle": "Large signed changes show that current proxy conclusions depend on analysis resolution",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "How much does matched perturbation-response Dice change with analysis resolution?",
            "rationale": "A zero-referenced difference chart makes direction and magnitude visible without ranking the two modalities together.",
            "dataset": "proxy_resolution",
            "sourceId": "multiplex_proxy",
            "encodings": {
                "x": {
                    "field": "label",
                    "type": "nominal",
                    "label": "Method and perturbation",
                },
                "y": {
                    "field": "delta_896_minus_256",
                    "type": "quantitative",
                    "label": "Dice difference (896 − 256)",
                    "format": "number",
                },
                "color": {
                    "field": "modality",
                    "type": "nominal",
                    "label": "Modality",
                },
                "tooltip": [
                    {
                        "field": "dice_256",
                        "type": "quantitative",
                        "label": "Dice at 256",
                    },
                    {
                        "field": "dice_896",
                        "type": "quantitative",
                        "label": "Dice at 896",
                    },
                ],
            },
            "comparisonContext": {
                "baseline": "zero resolution shift",
                "unit": "Dice difference",
                "grain": "matched modality × method × perturbation",
            },
            "layout": "full",
        },
    ]

    cards = [
        {
            "id": "best_he_dice",
            "description": "Best current point estimate on the public H&E fold-only locked test; not a Merck or crack result.",
            "dataset": "summary",
            "sourceId": "he_benchmark",
            "metrics": [
                {
                    "label": "Best H&E positive-field macro Dice",
                    "field": "best_macro_dice",
                    "format": "number",
                }
            ],
        },
        {
            "id": "paired_delta",
            "description": "Hibou-B linear minus DINOv2-small linear on identical positive fields; descriptive paired interval only.",
            "dataset": "summary",
            "sourceId": "he_benchmark",
            "metrics": [
                {
                    "label": "Hibou linear − DINOv2 linear",
                    "field": "hibou_vs_dinov2_delta",
                    "format": "number",
                    "signed": True,
                }
            ],
        },
        {
            "id": "he_test_fields",
            "description": "Locked real H&E test fields from 55 supplied source-slide groups.",
            "dataset": "summary",
            "sourceId": "he_benchmark",
            "metrics": [
                {
                    "label": "Locked H&E test fields",
                    "field": "locked_test_fields",
                    "format": "number",
                }
            ],
        },
        {
            "id": "multiplex_labels",
            "description": "Public COMET/CosMx fields with independently usable natural fold/crack masks in this benchmark.",
            "dataset": "summary",
            "sourceId": "multiplex_proxy",
            "metrics": [
                {
                    "label": "Natural multiplex labels",
                    "field": "labeled_natural_multiplex_fields",
                    "format": "number",
                }
            ],
        },
    ]

    tables = [
        {
            "id": "he_method_table",
            "title": "H&E method results with uncertainty and burden",
            "subtitle": "Same 424-field locked test; macro-Dice CIs use 1,000 source-slide-cluster bootstrap resamples",
            "dataset": "he_methods",
            "sourceId": "he_benchmark",
            "defaultSort": {"field": "macro_dice", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "head", "label": "Readout", "type": "text"},
                {"field": "macro_dice", "label": "Macro Dice", "format": "number"},
                {"field": "ci_low", "label": "95% CI low", "format": "number"},
                {"field": "ci_high", "label": "95% CI high", "format": "number"},
                {"field": "micro_dice", "label": "Micro Dice", "format": "number"},
                {
                    "field": "micro_ci_low",
                    "label": "Micro CI low",
                    "format": "number",
                },
                {
                    "field": "micro_ci_high",
                    "label": "Micro CI high",
                    "format": "number",
                },
                {"field": "presence_auroc", "label": "AUROC", "format": "number"},
                {
                    "field": "presence_auroc_ci_low",
                    "label": "AUROC CI low",
                    "format": "number",
                },
                {
                    "field": "presence_auroc_ci_high",
                    "label": "AUROC CI high",
                    "format": "number",
                },
                {
                    "field": "clean_fp_percent",
                    "label": "Clean FP area",
                    "format": "number",
                    "unit": "%",
                },
                {
                    "field": "clean_fp_ci_low_percent",
                    "label": "Clean FP CI low",
                    "format": "number",
                    "unit": "%",
                },
                {
                    "field": "clean_fp_ci_high_percent",
                    "label": "Clean FP CI high",
                    "format": "number",
                    "unit": "%",
                },
            ],
        },
        {
            "id": "he_qualitative_table",
            "title": "Hash-selected qualitative H&E audit cases",
            "subtitle": "Algorithmically selected without manual review during this audit; counts and calls matched",
            "dataset": "he_qualitative",
            "sourceId": "he_qualitative_audit",
            "defaultSort": {"field": "case", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "case", "label": "Panel", "type": "text"},
                {"field": "organ", "label": "Organ", "type": "text"},
                {"field": "reference", "label": "Reference", "type": "text"},
                {
                    "field": "classical_presence",
                    "label": "Classical presence",
                    "type": "text",
                },
                {
                    "field": "classical_pixel_dice",
                    "label": "Classical pixel Dice",
                    "format": "number",
                },
                {
                    "field": "hibou_presence",
                    "label": "Hibou presence",
                    "type": "text",
                },
                {
                    "field": "hibou_pixel_dice",
                    "label": "Hibou pixel Dice",
                    "format": "number",
                },
            ],
        },
        {
            "id": "paired_table",
            "title": "Selected paired descriptive differences",
            "subtitle": "Positive values favor the first method; 10,000 paired source-slide-cluster resamples",
            "dataset": "paired_differences",
            "sourceId": "he_benchmark",
            "defaultSort": {"field": "point_difference", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "contrast", "label": "Contrast", "type": "text"},
                {
                    "field": "point_difference",
                    "label": "Difference",
                    "format": "number",
                    "movement": True,
                },
                {"field": "ci_low", "label": "95% CI low", "format": "number"},
                {"field": "ci_high", "label": "95% CI high", "format": "number"},
            ],
        },
        {
            "id": "proxy_summary_table",
            "title": "Multiplex proxy response by artifact and method",
            "subtitle": "Fold and crack remain separate estimands; intervals use provisional source-group bootstrap resampling",
            "dataset": "proxy_summary",
            "sourceId": "multiplex_proxy",
            "defaultSort": {"field": "modality", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "modality", "label": "Modality", "type": "text"},
                {"field": "method", "label": "Method", "type": "text"},
                {"field": "fold_dice", "label": "Fold Dice", "format": "number"},
                {"field": "fold_ci_low", "label": "Fold CI low", "format": "number"},
                {"field": "fold_ci_high", "label": "Fold CI high", "format": "number"},
                {
                    "field": "crack_dice",
                    "label": "Crack Dice",
                    "format": "number",
                },
                {"field": "crack_ci_low", "label": "Crack CI low", "format": "number"},
                {
                    "field": "crack_ci_high",
                    "label": "Crack CI high",
                    "format": "number",
                },
                {
                    "field": "untouched_alert_burden_percent",
                    "label": "Alert burden",
                    "format": "number",
                    "unit": "%",
                },
                {
                    "field": "alert_ci_low_percent",
                    "label": "Alert CI low",
                    "format": "number",
                    "unit": "%",
                },
                {
                    "field": "alert_ci_high_percent",
                    "label": "Alert CI high",
                    "format": "number",
                    "unit": "%",
                },
                {"field": "interpretation", "label": "Claim boundary", "type": "text"},
            ],
        },
        {
            "id": "evidence_table",
            "title": "Evidence tiers and permitted claims",
            "subtitle": "Results are only comparable within the stated input, reference and evidence tier",
            "dataset": "evidence_scope",
            "sourceId": "evidence_inventory",
            "defaultSort": {"field": "evidence", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "evidence", "label": "Evidence", "type": "text"},
                {"field": "input", "label": "Input", "type": "text"},
                {"field": "reference", "label": "Reference", "type": "text"},
                {"field": "valid_claim", "label": "Permitted claim", "type": "text"},
                {"field": "invalid_claim", "label": "Prohibited claim", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
            ],
        },
        {
            "id": "model_status_table",
            "title": "Foundation and QC method readiness",
            "subtitle": "Exact names, present execution status and corporate-access constraints",
            "dataset": "model_status",
            "sourceId": "method_audit",
            "defaultSort": {"field": "priority", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "priority", "label": "Priority", "type": "number"},
                {"field": "model_or_method", "label": "Model / method", "type": "text"},
                {"field": "domain", "label": "Domain", "type": "text"},
                {
                    "field": "current_evidence",
                    "label": "Current evidence",
                    "type": "text",
                },
                {"field": "access", "label": "Access / license", "type": "text"},
                {"field": "decision", "label": "Recommended role", "type": "text"},
            ],
        },
        {
            "id": "smoke_table",
            "title": "Apple MPS engineering feasibility",
            "subtitle": "Two deterministic patches per model; runtime and PEFT execution only",
            "dataset": "smoke_results",
            "sourceId": "foundation_smoke",
            "defaultSort": {"field": "mps_speedup", "direction": "desc"},
            "density": "dense",
            "columns": [
                {"field": "model", "label": "Model", "type": "text"},
                {
                    "field": "cpu_seconds_two_patches",
                    "label": "CPU median",
                    "format": "number",
                    "unit": "s",
                },
                {
                    "field": "mps_seconds_two_patches",
                    "label": "MPS median",
                    "format": "number",
                    "unit": "s",
                },
                {
                    "field": "mps_speedup",
                    "label": "MPS speedup",
                    "format": "number",
                    "unit": "×",
                },
                {
                    "field": "max_abs_cpu_mps_error",
                    "label": "Max |CPU−MPS|",
                    "format": "number",
                },
                {
                    "field": "trainable_fraction",
                    "label": "LoRA trainable fraction",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "acceptance_table",
            "title": "Proposed internal acceptance criteria",
            "subtitle": "Starting hypotheses for stakeholder approval and power analysis; not validated Merck requirements",
            "dataset": "acceptance",
            "sourceId": "evaluation_protocol",
            "defaultSort": {"field": "priority", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "priority", "label": "Priority", "type": "number"},
                {"field": "criterion", "label": "Criterion", "type": "text"},
                {"field": "proposed_gate", "label": "Proposed gate", "type": "text"},
                {"field": "cohort", "label": "Evidence cohort", "type": "text"},
            ],
        },
    ]

    # The report schema accepts sourceId references, but the hosted renderer's
    # provenance validator currently also expects the canonical source object
    # inline on each native card/chart/table.  Keep both so the artifact remains
    # self-contained while preserving stable source identifiers.
    sources_by_id = {source["id"]: source for source in sources}
    for native_item in [*cards, *charts, *tables]:
        source_id = native_item.get("sourceId")
        if source_id is not None:
            dataset_id = native_item["dataset"]
            source = copy.deepcopy(sources_by_id[source_id])
            snapshot_path = (
                f"reports/benchmark_findings_{REPORT_DATE}/data/{dataset_id}.json"
            )
            source["path"] = snapshot_path
            query = source.setdefault("query", {})
            query.update(
                {
                    "engine": "duckdb",
                    "language": "sql",
                    "sql": f"SELECT * FROM read_json_auto('{snapshot_path}')",
                    "description": (
                        "Reads the bounded, report-versioned dataset generated from the "
                        "audited source artifacts. Raw upstream identities and metric "
                        "definitions remain listed in tables_used and the source notes."
                    ),
                    "executed_at": REPORT_TIMESTAMP,
                }
            )
            native_item["source"] = source

    blocks = [
        {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
        {
            "id": "technical_summary",
            "type": "markdown",
            "body": (
                "## Technical summary\n\n"
                "**Overall assessment: share internally with explicit caveats; ready for an adjudicated pilot, not for deployment or a three-modality efficacy claim.** "
                "The strongest current real-label result is Hibou-B with a mask-supervised linear token probe on a bounded public H&E fold-only cohort. "
                "On this fixed cohort and configuration, the three PatchKNN heads had lower observed localization Dice than the three mask-supervised linear probes; this does not establish a universal requirement for labels or invalidate anomaly detection. "
                "COMET and CosMx have only real-background synthetic-perturbation evidence: those results diagnose representation and scale sensitivity but cannot estimate natural-artifact Dice, ROC, sensitivity, specificity, or false-positive rate. No public COMET/CosMx natural fold/crack masks were located in the source audit completed 2026-08-27 [20–23].\n\n"
                "The practical recommendation is a **shared QC software/evaluation platform with modality-specific input construction, reference banks, calibration and acceptance**. "
                "Freeze labels, group splits and operating rules before an internal test; use HistoQC as an operational QC baseline after defining an output-to-ontology mapping, evaluate GrandQC only after legal clearance, and seek Merck's DiffusionQC implementation and split metadata; use KRONOS2 only after written commercial/derivative permission and an explicit marker/MPP contract."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [
                "best_he_dice",
                "paired_delta",
                "he_test_fields",
                "multiplex_labels",
            ],
        },
        {
            "id": "he_finding",
            "type": "markdown",
            "sourceId": "he_benchmark",
            "body": (
                "## Mask-supervised frozen pathology features have the highest observed H&E point estimates\n\n"
                "Hibou-B linear achieved **0.667 positive-field macro Dice (95% CI 0.603–0.730)**, **0.827 all-field micro Dice**, **0.985 presence AUROC**, and **0.047% clean-field predicted area**. "
                "DINOv2-small linear followed at 0.599 macro Dice; SigLIP2 Base linear reached 0.526; the classical detector reached 0.446. "
                "The three one-class PatchKNN heads ranged from 0.208 to 0.341.\n\n"
                "This supports a representation-plus-light-supervision path for H&E fold localization on this cohort. It does **not** show that Hibou-B is universally superior: linear probes use fit-set pixel masks, while PatchKNN uses clean fit tokens and calibration masks. Same-head encoder contrasts are more scientifically interpretable than cross-head rankings."
            ),
        },
        {
            "id": "he_chart",
            "type": "chart",
            "chartId": "he_macro_dice",
            "layout": "full",
        },
        {
            "id": "he_chart_interpretation",
            "type": "markdown",
            "body": (
                "The ranked chart shows the point-estimate separation; consult the table for interval width and clean alert burden. "
                "The standalone vector Figure 1 additionally displays macro-Dice intervals, AUROC and clean-field burden without conflating their scales."
            ),
        },
        {
            "id": "he_table_block",
            "type": "table",
            "tableId": "he_method_table",
            "layout": "full",
        },
        {
            "id": "organ_finding",
            "type": "markdown",
            "sourceId": "he_benchmark",
            "body": (
                "## Organ-stratified H&E estimates identify a robustness question\n\n"
                "Hibou-B linear ranged from **0.459 in brain** to **0.797 in testis**. "
                "Every method has a wide descriptive organ range, but several strata contain only three to nine positive supplied source-slide groups and no per-organ interval or heterogeneity test was computed. "
                "A single overall Dice therefore cannot establish robustness; the internal locked test must predeclare tissue, scanner, site and time strata and must not average away a failed required stratum."
            ),
        },
        {
            "id": "organ_chart",
            "type": "chart",
            "chartId": "he_organ_heatmap",
            "layout": "full",
        },
        {
            "id": "organ_chart_interpretation",
            "type": "markdown",
            "body": (
                "Read each row for stability and each column for tissue-specific difficulty. These are descriptive point estimates without per-organ confidence intervals; apparent differences may reflect tissue, source-slide composition, annotation or acquisition."
            ),
        },
        {
            "id": "paired_finding",
            "type": "markdown",
            "sourceId": "he_benchmark",
            "body": (
                "## Paired contrasts strengthen the cohort-specific ranking but do not prove superiority\n\n"
                "Using identical source-slide bootstrap draws, Hibou-B linear minus DINOv2 linear was **+0.068 (descriptive 95% interval +0.053 to +0.084)**. "
                "DINOv2 linear minus SigLIP2 linear was +0.073, while DINOv2 PatchKNN minus Hibou-B PatchKNN was +0.022 with an interval spanning zero. "
                "The paired artifact intentionally computes no p-values and applies no multiplicity adjustment; these intervals are exploratory uncertainty summaries conditional on the observed public cohort."
            ),
        },
        {
            "id": "paired_chart",
            "type": "chart",
            "chartId": "he_paired_differences",
            "layout": "full",
        },
        {
            "id": "paired_chart_interpretation",
            "type": "markdown",
            "body": (
                "Positive bars favor the first method. The corresponding vector forest plot and exact table show whether each descriptive interval crosses zero; neither should be relabeled as a hypothesis test."
            ),
        },
        {
            "id": "paired_table_block",
            "type": "table",
            "tableId": "paired_table",
            "layout": "full",
        },
        {
            "id": "qualitative_finding",
            "type": "markdown",
            "sourceId": "he_qualitative_audit",
            "body": (
                "## Hash-selected qualitative audit confirms outcome reproducibility\n\n"
                "Seven whole 896×504 px H&E analysis fields were algorithmically selected without manual image review during this audit: the SHA-256-minimum fold-positive field within each organ plus the SHA-256-minimum presence false-positive and false-negative separately for each compared method; the union was deduplicated. "
                "Because the hardened result artifacts did not retain spatial maps or shallow-probe parameters, the exact frozen Hibou-B encoder and fit split were used to refit only the deterministic shallow head; calibration was not rerun and locked thresholds were reused. "
                "All 14 regenerated method outcomes matched the stored pixel counts and presence calls; regenerated image scores were within the recorded numerical tolerances. Figure 7 is therefore an audited localization sanity check, not a performance sample or evidence of generalization."
            ),
        },
        {
            "id": "qualitative_table_block",
            "type": "table",
            "tableId": "he_qualitative_table",
            "layout": "full",
        },
        {
            "id": "multiplex_finding",
            "type": "markdown",
            "sourceId": "multiplex_proxy",
            "body": (
                "## The current multiplex experiment identifies a weak generic anomaly configuration\n\n"
                "Fold and crack remain separate endpoints. Classical fold/crack response Dice was **0.500/0.567 on COMET** and **0.393/0.235 on CosMx**; the nominal-reference anomaly branch was 0.135/0.065 and 0.193/0.075; the hybrid was 0.382/0.567 and 0.376/0.234. "
                "The nominal reference fields were unannotated, not expert-confirmed clean. Recorded integrity checks identified no runtime or transformation failure, while 256-versus-896 sensitivity materially changed results. "
                "That pattern is consistent with a scientific mismatch in representation, channel projection, perturbation scale or threshold, although semantic implementation defects cannot be fully excluded."
            ),
        },
        {
            "id": "proxy_comet_chart",
            "type": "chart",
            "chartId": "proxy_comet",
            "layout": "full",
        },
        {
            "id": "proxy_comet_interpretation",
            "type": "markdown",
            "body": (
                "COMET uses DAPI-only public fields in this benchmark. Classical response is substantially stronger than the generic nominal-reference representation, but all values remain conditional on the inserted perturbation generator and provisional field grouping."
            ),
        },
        {
            "id": "proxy_cosmx_chart",
            "type": "chart",
            "chartId": "proxy_cosmx",
            "layout": "full",
        },
        {
            "id": "proxy_cosmx_interpretation",
            "type": "markdown",
            "body": (
                "CosMx combines five morphology channels, yet the generic anomaly and hybrid branches create large background responses. This is evidence against shipping the current projection/reference-bank configuration, not evidence against all anomaly detection or marker-aware models."
            ),
        },
        {
            "id": "proxy_alert_chart",
            "type": "chart",
            "chartId": "proxy_alert_burden",
            "layout": "full",
        },
        {
            "id": "proxy_alert_interpretation",
            "type": "markdown",
            "body": (
                "Untouched alert burden is low and similar across COMET branches but much larger and unstable for the CosMx anomaly/hybrid branches. Because the public fields were not adjudicated as artifact-free, alert burden must never be renamed false-positive rate."
            ),
        },
        {
            "id": "proxy_summary_block",
            "type": "table",
            "tableId": "proxy_summary_table",
            "layout": "full",
        },
        {
            "id": "proxy_resolution_finding",
            "type": "markdown",
            "sourceId": "multiplex_proxy",
            "body": (
                "## Multiplex conclusions are sensitive to analysis resolution\n\n"
                "The matched 256-versus-896 experiment shifts individual proxy Dice values by as much as **0.332**. "
                "The largest change is CosMx classical crack response, which falls from 0.568 at 256 pixels to 0.235 at 896 pixels. "
                "This instability strengthens the case for physical MPP normalization and predeclared multiscale testing before any model or channel configuration is selected."
            ),
        },
        {
            "id": "proxy_resolution_chart",
            "type": "chart",
            "chartId": "proxy_resolution_sensitivity",
            "layout": "full",
        },
        {
            "id": "proxy_resolution_interpretation",
            "type": "markdown",
            "body": (
                "Positive values favor the 896-pixel analysis and negative values favor 256 pixels. "
                "These are paired configuration differences without uncertainty intervals; they diagnose engineering sensitivity, not natural-artifact accuracy."
            ),
        },
        {
            "id": "scope_definitions",
            "type": "markdown",
            "body": (
                "## Scope, data and metric definitions\n\n"
                "**H&E cohort.** The Histology Tissue Fold Dataset contains 2,127 real 3,840 W × 2,160 H px teaching-slide microscope fields across brain, kidney, liver, small intestine and testis [1,2]. "
                "The frozen 60/20/20 supplied-source-slide split yields 424 locked-test fields from 55 groups. It supplies fold masks only; it is not WSI and has no crack class.\n\n"
                "**Multiplex cohort.** The proxy uses five public COMET DAPI fields [20,21] and six five-channel CosMx morphology FOVs from four provisional slide/run groups [22,23]. Neither audited multiplex release supplies independently usable natural fold/crack masks.\n\n"
                "**Primary H&E metric.** Positive-field macro Dice weights each valid positive field equally; all-field micro Dice pools pixels. Presence AUROC/AUPRC and clean-field predicted area answer different questions and are reported separately. For a thin crack/tear target, tolerant centerline F1, clDice and instance FROC should become primary geometric measures because width-sensitive Dice alone is inadequate."
            ),
        },
        {
            "id": "evidence_table_block",
            "type": "table",
            "tableId": "evidence_table",
            "layout": "full",
        },
        {
            "id": "methodology",
            "type": "markdown",
            "sourceId": "evaluation_protocol",
            "body": (
                "## Methodology and validation design\n\n"
                "All H&E methods share the exact organ-by-class supplied-source-slide split, 896-pixel analysis bound, 224-pixel non-overlapping tiling and calibration-only threshold selection. "
                "The classical method uses color/texture/morphology cues. Frozen foundation encoders produce spatial tokens for either (1) clean-bank PatchKNN anomaly scoring or (2) a shallow pixel-mask-supervised linear probe. "
                "Bootstrap intervals resample source-slide groups within locked strata. The paired comparator reuses each cluster draw across methods.\n\n"
                "The multiplex proxy performs leave-one-declared-source-group-out fitting/calibration/testing, inserts controlled perturbations only after role assignment, and evaluates paired incremental response. Groups recur in fit/calibration across folds, so folds are dependent; higher-level biological independence is unverified."
            ),
        },
        {
            "id": "foundation_status",
            "type": "markdown",
            "body": (
                "## Foundation-model feasibility is selective, not comprehensive\n\n"
                "DINOv2 remains a useful Apache-2.0 generic control even though DINOv3 is newer [7–9,29]. Hibou-B is the strongest currently runnable pathology-specific comparator [10,11]. "
                "The checkpoint names are **UNI2-h** and **CONCHv1.5** [14,16]; their cited papers describe the original UNI and CONCH lineages, not validation of these checkpoint versions [15,17]. No official CONCH v2 was located in the official source search completed 2026-08-27. "
                "KRONOS2 is the most aligned public marker-aware encoder among the reviewed COMET-like multiplex IF candidates [18], while [19] describes the earlier KRONOS lineage rather than KRONOS2 validation. It is not a ready artifact detector and is only partially applicable to CosMx morphology/protein channels—not decoded transcript coordinates or expression matrices.\n\n"
                "At Merck, gated CC BY-NC-ND models must not be assumed usable for frozen inference or LoRA. Adaptation can create a derivative; obtain written institutional permission before downloading, feature extraction or PEFT."
            ),
        },
        {
            "id": "model_status_block",
            "type": "table",
            "tableId": "model_status_table",
            "layout": "full",
        },
        {
            "id": "mps_finding",
            "type": "markdown",
            "sourceId": "foundation_smoke",
            "body": (
                "## Apple MPS and narrow LoRA execution are technically feasible\n\n"
                "DINOv2-small and SigLIP2 Base produced finite dense features on MPS with near-identical CPU outputs, and each completed a rank-4 LoRA update. "
                "These are two-patch engineering smokes, not WSI throughput or efficacy studies. "
                "The shared runtime currently resolves `auto` to **MPS then CPU only**; it does not detect CUDA, so CUDA portability remains an implementation gap despite the MPS success."
            ),
        },
        {
            "id": "smoke_table_block",
            "type": "table",
            "tableId": "smoke_table",
            "layout": "full",
        },
        {
            "id": "public_resources",
            "type": "markdown",
            "body": (
                "## Public resources for the next benchmark\n\n"
                "| Resource | Best use | Critical limitation |\n"
                "| --- | --- | --- |\n"
                "| GrandQC manual test set [4,5] | Expert patch/crop-level artifact masks sampled from H&E WSIs | No distinct crack class; method assets are noncommercial [30] |\n"
                "| AIRAQc TCGA test [24] | Manually annotated real TCGA H&E WSIs | Verify release files and frozen split before use |\n"
                "| Histology Tissue Fold Dataset [1,2] | Current 1,228-mask fold benchmark | Teaching-slide fields, not WSI; fold only |\n"
                "| HistoArtifacts [25,26] | Fold/damaged-tissue patch labels | No dedicated pixel crack masks; reuse license needs confirmation |\n"
                "| Foucart WSI artifacts [27,28] | Partial Tear&Fold/knife-damage annotations | Incomplete labels make ordinary negative-pixel Dice invalid |\n"
                "| QUALIFAI COMET and public CosMx releases [20–23] | Compatibility, drift, and blinded annotation pilots | No natural fold/crack ground truth found |\n\n"
                "GrandQC model-generated TCGA masks are pseudo-labels and are not independent ground truth. No single labeled H&E–COMET–CosMx benchmark or mature, complete pixel-level histologic crack dataset was located in the source audit completed 2026-08-27."
            ),
        },
        {
            "id": "limitations",
            "type": "markdown",
            "body": (
                "## Limitations, uncertainty and robustness checks\n\n"
                "- **Domain limitation:** the only natural pixel ground truth is fold-only H&E microscope imagery; there is no current crack, human clinical WSI, internal Merck, natural COMET or natural CosMx accuracy estimate.\n"
                "- **Supervision mismatch:** classical, PatchKNN and linear-probe rows use different annotation regimes; compare encoders within the same head before attributing gains to the backbone.\n"
                "- **Small correlated strata:** the H&E test has 28 positive supplied-source-slide groups, with only 3–9 per organ. Multiplex has 5 and 4 provisional groups with unverified higher-level independence.\n"
                "- **Exploratory inference:** bootstrap intervals are descriptive and conditional on the fixed cohort/configuration. There are no p-values, multiplicity adjustment, noninferiority test or external validation.\n"
                "- **Adaptive reuse:** this public test has informed multiple result reviews; an untouched external confirmation set is required before further model selection or confirmatory claims.\n"
                "- **Method coverage:** HistoQC, GrandQC, DiffusionQC, DINOv3, UNI2-h, CONCHv1.5 and KRONOS2 were not executed; the current benchmark is not a comprehensive SOTA leaderboard.\n"
                "- **Qualitative reproduction boundary:** the hardened artifacts did not retain score maps or probe parameters, so Figure 7 required a deterministic shallow-head refit. Selected-field counts and calls matched and scores were within recorded tolerances, but future frozen releases should persist a hashed inference bundle and maps.\n"
                "- **Operational gap:** accuracy, review burden, auto-pass safety, technical abstention and downstream assay impact have not been prospectively measured.\n"
                "- **Reproduction boundary:** the upstream benchmark artifacts are local and gitignored; external reproduction requires a frozen evidence bundle or governed object-store/DVC release matching the recorded hashes."
            ),
        },
        {
            "id": "acceptance_intro",
            "type": "markdown",
            "sourceId": "evaluation_protocol",
            "body": (
                "## Evaluation criteria for the internal locked benchmark\n\n"
                "Use separate production-prevalence, enriched challenge, external-generalization, degraded-input and downstream-impact cohorts. "
                "Split at the highest leakage unit (patient/block/slide/run), freeze ontology and thresholds before opening test predictions, bootstrap that independent unit, and report every required modality/stratum separately. "
                "For unlabeled deployment data, monitor score distributions, transform consistency, alert burden, missing channels and drift—but do not compute natural-artifact Dice or ROC until blinded references exist."
            ),
        },
        {
            "id": "acceptance_block",
            "type": "table",
            "tableId": "acceptance_table",
            "layout": "full",
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "body": (
                "## Recommended next steps\n\n"
                "1. **Lock the ontology and action.** Define fold, tissue tear, glass/coverslip crack, knife line and acquisition seam separately; map each to PASS/REVIEW/FAIL and remediation.\n"
                "2. **Create an adjudicated pilot.** Target at least 30 independent patient/slide/run groups per modality, enriched to at least 50 fold-positive, 50 crack-positive and 100 clean/hard-negative regions, with two reviewers and adjudication.\n"
                "3. **Run accessible controls.** Add HistoQC as an operational WSI-QC baseline only after defining its output-to-fold ontology/metric mapping; request legal clearance for GrandQC and verify the manual GrandQC/AIRAQc resources; obtain Merck's DiffusionQC implementation and reconcile its split.\n"
                "4. **Use matched heads for model selection.** Compare classical, DINOv2, Hibou-B and approved DINOv3 with the same frozen readouts; add a small decoder/LoRA only after a frozen model fails a predeclared real-label criterion.\n"
                "5. **Open a governed multiplex track.** Define marker vocabulary, MPP, channel-missingness and projection baselines; request KRONOS2 permission; retain the current anomaly branch as a documented negative result until a marker-aware representation and real labels are available.\n"
                "6. **Repair the runtime portability gap.** Add explicit CUDA selection/synchronization and parity tests before claiming automatic CUDA/MPS support."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions that can change the decision\n\n"
                "- What exactly counts as a crack in each acquisition workflow, and is a pixel mask actionable?\n"
                "- Which COMET/CosMx structural channels and MPP metadata are guaranteed at inference time?\n"
                "- What are expected production prevalence, acceptable review capacity and costs of false PASS versus false FAIL?\n"
                "- Which scanner, site, panel, tissue, species and time strata are required claims rather than exploratory cuts?\n"
                "- Can Legal approve GrandQC, DINOv3 and KRONOS2, and can Merck provide the internal DiffusionQC assets?"
            ),
        },
        {
            "id": "references",
            "type": "markdown",
            "body": "## References\n\n" + REFERENCES_MD,
        },
    ]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": TITLE,
            "description": "Technical feasibility benchmark, evidence audit and internal evaluation plan for fold/crack QC across H&E, COMET and CosMx.",
            "generatedAt": REPORT_TIMESTAMP,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": REPORT_TIMESTAMP,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "he_methods": he_rows,
                "he_organs": organ_rows,
                "paired_differences": paired_rows,
                "he_qualitative": qualitative_rows,
                "proxy_comet": comet_rows,
                "proxy_cosmx": cosmx_rows,
                "proxy_alert": alert_rows,
                "proxy_summary": proxy_summary,
                "proxy_resolution": resolution_rows,
                "evidence_scope": evidence_scope_rows(),
                "model_status": model_status_rows(),
                "smoke_results": smoke_rows,
                "acceptance": acceptance_rows(),
            },
        },
        "sources": sources,
        "package_info": {
            "reportDate": REPORT_DATE,
            "generator": "reports/benchmark_findings_2026-08-27/generate_report.py",
        },
    }


def markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    """Render a compact GitHub-flavored Markdown table."""

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    rendered = ["| " + " | ".join(cell(value) for value in headers) + " |"]
    rendered.append("| " + " | ".join("---" for _ in headers) + " |")
    rendered.extend(
        "| " + " | ".join(cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(rendered)


def build_markdown_report(
    he_rows: list[dict[str, Any]],
    organ_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    qualitative_rows: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
    alert_rows: list[dict[str, Any]],
    proxy_summary: list[dict[str, Any]],
    resolution_rows: list[dict[str, Any]],
    smoke_rows: list[dict[str, Any]],
    he_audit: dict[str, Any],
    paired_audit: dict[str, Any],
    proxy_audit: dict[str, Any],
    resolution_audit: dict[str, Any],
) -> str:
    """Create the durable, manuscript-oriented source report."""

    ranked = sorted(he_rows, key=lambda row: row["macro_dice"], reverse=True)
    hibou_organs = {
        row["organ"]: row["macro_dice"]
        for row in organ_rows
        if row["method_key"] == "hibou_linear"
    }
    he_table = markdown_table(
        [
            "Method",
            "Readout",
            "Positive-field macro Dice (95% CI)",
            "All-field micro Dice (95% CI)",
            "Presence AUROC (95% CI)",
            "Clean predicted area (95% CI)",
        ],
        (
            (
                row["method"],
                row["head"],
                f"{row['macro_dice']:.3f} ({row['ci_low']:.3f}–{row['ci_high']:.3f})",
                f"{row['micro_dice']:.3f} ({row['micro_ci_low']:.3f}–{row['micro_ci_high']:.3f})",
                f"{row['presence_auroc']:.3f} ({row['presence_auroc_ci_low']:.3f}–{row['presence_auroc_ci_high']:.3f})",
                f"{row['clean_fp_percent']:.3f}% ({row['clean_fp_ci_low_percent']:.3f}–{row['clean_fp_ci_high_percent']:.3f})",
            )
            for row in ranked
        ),
    )
    paired_table = markdown_table(
        ["Paired contrast", "Dice difference", "Descriptive 95% interval"],
        (
            (
                row["contrast"],
                f"{row['point_difference']:+.3f}",
                f"{row['ci_low']:+.3f} to {row['ci_high']:+.3f}",
            )
            for row in paired_rows
        ),
    )
    proxy_table = markdown_table(
        [
            "Modality",
            "Method",
            "Fold perturbation Dice (95% CI)",
            "Crack perturbation Dice (95% CI)",
            "Untouched alert burden (95% CI)",
        ],
        (
            (
                row["modality"],
                row["method"],
                f"{row['fold_dice']:.3f} ({row['fold_ci_low']:.3f}–{row['fold_ci_high']:.3f})",
                f"{row['crack_dice']:.3f} ({row['crack_ci_low']:.3f}–{row['crack_ci_high']:.3f})",
                f"{row['untouched_alert_burden_percent']:.2f}% ({row['alert_ci_low_percent']:.2f}–{row['alert_ci_high_percent']:.2f})",
            )
            for row in proxy_summary
        ),
    )
    smoke_table = markdown_table(
        [
            "Model",
            "CPU median (two patches)",
            "MPS median (two patches)",
            "Observed speed-up",
            "Max |CPU−MPS|",
            "LoRA trainable fraction",
        ],
        (
            (
                row["model"],
                f"{row['cpu_seconds_two_patches']:.5f} s",
                f"{row['mps_seconds_two_patches']:.5f} s",
                f"{row['mps_speedup']:.2f}×",
                f"{row['max_abs_cpu_mps_error']:.6f}",
                f"{100 * row['trainable_fraction']:.3f}%",
            )
            for row in smoke_rows
        ),
    )
    model_table = markdown_table(
        ["Model or method", "Status in this repository", "Recommended role"],
        (
            (row["model_or_method"], row["current_evidence"], row["decision"])
            for row in sorted(model_status_rows(), key=lambda row: row["priority"])
        ),
    )
    acceptance_table = markdown_table(
        ["Criterion", "Proposed gate", "Cohort"],
        (
            (row["criterion"], row["proposed_gate"], row["cohort"])
            for row in sorted(acceptance_rows(), key=lambda row: row["priority"])
        ),
    )
    best = ranked[0]
    proxy_anomaly_crack = {
        row["modality"]: row["dice"]
        for row in proxy_rows
        if row["method_key"] == "clean_reference_anomaly"
        and row["artifact_key"] == "crack"
    }

    return f"""# {TITLE}

**Technical findings report · {REPORT_DATE} · Evidence status: internal feasibility, share with caveats**

## Technical summary

The repository now supports one supplied-source-slide-disjoint, cohort-conditional exploratory real-label comparison for **tissue-fold localization on public H&E microscope fields**. The strongest point estimate was **{best["method"]}**, with positive-field macro Dice **{best["macro_dice"]:.3f} (95% source-slide-cluster bootstrap CI {best["ci_low"]:.3f}–{best["ci_high"]:.3f})**, all-field micro Dice {best["micro_dice"]:.3f}, field-presence AUROC {best["presence_auroc"]:.3f}, and {best["clean_fp_percent"]:.3f}% predicted area on clean fields.

That result is promising, but it is not yet a WSI, crack, COMET, CosMx, comprehensive-SOTA, or deployment claim. The best H&E readout uses manual fit-set masks to train a shallow linear probe on frozen features. The real-background COMET and CosMx experiments use inserted synthetic perturbations and therefore establish decode/runtime compatibility and generator-conditional detector response—not natural-artifact accuracy. No public COMET/CosMx natural fold-or-crack annotation set was located in the source audit completed 2026-08-27 [20–23].

The correct architecture is consequently a **shared QC platform with modality-specific channel construction, reference banks, calibration, and acceptance gates**. The immediate high-value step is a blinded, adjudicated internal pilot. Foundation-model expansion is warranted only under matched readouts and cleared licenses: retain DINOv2 as a permissive generic control; add DINOv3 after legal review; treat UNI2-h/CONCHv1.5 as H&E encoders; and evaluate KRONOS2 for marker-aware COMET only after written commercial and derivative-use permission.

## Decision and claim boundary

| Evidence tier | Data | What may be claimed | What must not be claimed |
| --- | --- | --- | --- |
| Real H&E held-out test | 424 fields from 55 supplied source-slide groups; manual fold masks | Cohort-specific H&E fold localization/presence and clean-field alert burden | WSI, human-clinical, crack, COMET/CosMx, or deployment performance |
| Real-background multiplex proxy | 5 COMET DAPI fields and 6 CosMx morphology FOVs/4 slide-run IDs; synthetic insertions | Pipeline compatibility, perturbation response, transform checks, alert burden | Natural-artifact Dice, ROC, sensitivity, specificity, or false-positive rate |
| Foundation/LoRA smoke | Two deterministic RGB patches per model | MPS execution, CPU/MPS parity, one rank-4 update | Accuracy, throughput at WSI scale, generalization, or PEFT benefit |

**Recommendation:** advance to a governed internal annotation and locked-test pilot. Do not represent the present package as production validated or as a comprehensive SOTA benchmark.

## 1. Real H&E fold benchmark

### Dataset and split

The [Histology Tissue Fold Dataset v1](https://zenodo.org/records/21493260) comprises 2,127 real 3,840 W × 2,160 H px RGB images acquired at 10× from veterinary teaching slides: 1,228 fold-positive fields with manual masks and 899 clean fields across brain, kidney, liver, small intestine, and testis [1,2]. Supplied source-slide IDs were stratified by organ and class and assigned to fit, calibration, and test in a 60:20:20 ratio. The hardened split contains 1,276 fields/170 slides for fit, 427/58 for calibration, and **424/55 for locked test**. No field or supplied source slide crosses partitions. Two supplied positive masks were empty; their presence labels were retained, while those fields were excluded from localization fitting/calibration under the frozen policy.

The locked test contains {he_audit["n_positive_fields"]} valid fold-positive fields from {he_audit["n_positive_source_slides"]} positive source-slide groups and {he_audit["n_clean_fields"]} clean fields from {he_audit["n_clean_source_slides"]} clean groups. This is real annotated microscopy, but it is **not WSI**, contains no crack class, and has no patient/block identifiers beyond supplied slide ID.

### Methods

All methods use the identical split, 896-pixel maximum analysis dimension, 224-pixel tiling, and calibration-only operating-point selection.

- The classical candidate detector combines optical-density, saturation, texture, and morphology signals. It uses no fit masks, but its operating threshold uses calibration labels.
- Frozen DINOv2-small, Hibou-B, and SigLIP2 Base encoders provide spatial tokens [7,10–13]. The “PatchKNN” readout uses up to 4,096 clean-labeled fit tokens and three nearest neighbours; it is PatchCore-like, not a full PatchCore implementation, and uses calibration masks for thresholding.
- The class-balanced linear probe uses up to 8,192 tokens per class derived from fit-set pixel masks. It is lightly supervised, not zero-shot or unsupervised; only the encoder remains frozen.

Localization thresholds maximize pooled calibration pixel Dice over a deterministic score-quantile grid. Presence uses the 99.5th percentile of each score map and a calibration threshold maximizing balanced accuracy. Thresholds are fixed before locked-test inference. Reported intervals resample supplied source-slide groups within organ/class strata and condition on one fixed split, fit, random reservoir, and threshold.

![Figure 1. H&E locked-test performance](figures/figure1_he_locked_test.svg)

**Figure 1. H&E locked-test fold performance.** (A) Positive-field macro Dice. (B) Field-presence AUROC. (C) Predicted positive pixel area on clean fields, shown on a log scale. Whiskers in all panels are 95% source-slide-cluster bootstrap intervals from 1,000 resamples; marker shape distinguishes readout type. All methods use the same 424-field test cohort, but readout supervision differs; cross-head rankings should not be interpreted as controlled backbone comparisons.

{he_table}

### Main finding

The three mask-supervised linear probes have the highest point estimates in this cohort. The observed Hibou-B linear minus DINOv2 linear difference is **+0.068 Dice (descriptive paired 95% interval +0.053 to +0.084)** on the identical locked fields; DINOv2 linear minus SigLIP2 linear is +0.073 (+0.062 to +0.083). The strongest one-class localization point estimate is DINOv2 PatchKNN at 0.341 macro Dice; Hibou PatchKNN nevertheless achieves 0.947 presence AUROC, demonstrating that field ranking and semantic pixel localization are distinct tasks.

![Figure 2. H&E organ heterogeneity](figures/figure2_he_organ_heatmap.svg)

**Figure 2. Organ-stratified H&E fold localization.** Positive-field macro Dice point estimates on a focused 0–0.85 color scale. Hibou-B linear spans **{hibou_organs["Brain"]:.3f} in brain to {hibou_organs["Testis"]:.3f} in testis**. The five organs contain 42–57 positive fields but only 3–9 positive supplied source-slide groups each. No per-organ interval or heterogeneity test was computed; differences may reflect tissue, source composition, acquisition, or annotation.

![Figure 3. Paired H&E differences](figures/figure3_he_paired_differences.svg)

**Figure 3. Selected paired H&E contrasts.** Differences in positive-field macro Dice across 245 positive fields from 28 supplied source-slide groups, with 95% intervals from {paired_audit["resamples"]:,} common cluster-bootstrap draws. The display includes adjacent linear/classical ranks, classical versus the two highest PatchKNN point estimates, and the top-two PatchKNN contrast; remaining pairwise contrasts are not shown. Positive values favor the first method. The analysis is descriptive: no p-values, multiplicity control, noninferiority margin, or superiority claim was applied.

{paired_table}

### H&E uncertainty and validity

The highest-impact aggregates were independently recomputed from per-field TP/FP/FN, and exact cohort/order identity was asserted across all seven rows. No direct slide leakage or duplicate image hash was found. However, this same public test has been inspected across multiple development cycles; any further selection based on these results risks adaptive test-set overfitting. The bootstrap does not capture split uncertainty, inter-reader variability, training-seed variability, model-selection bias, or domain shift. All pixels outside fold masks are treated as negatives without a tissue-specific ignore mask.

## 2. COMET and CosMx: real-background proxy, not natural-artifact validation

Five public COMET DAPI fields [20,21] and six five-channel CosMx morphology FOVs from four distinct slide/run identifiers [22,23] were decoded successfully. They lack natural fold/crack masks. The benchmark therefore inserts controlled fold-like shifted-signal perturbations and crack-like all-channel attenuation into real backgrounds, then evaluates the nonnegative incremental score relative to each untouched field. The nominal anomaly reference bank is fitted from **unannotated fields, not expert-confirmed clean fields**, so natural artifacts could be absorbed into the reference distribution. Thresholds are selected on calibration perturbations. Leave-one-declared-group-out test coverage is complete, but training/calibration groups recur across folds and higher-level biological independence is unverified.

![Figure 4. COMET and CosMx proxy evidence](figures/figure4_multiplex_proxy.svg)

**Figure 4. Multiplex proxy response and alert burden.** (A–B) Group-macro Dice against inserted fold/crack perturbation supports; filled/outlined bars denote fold/crack. These values quantify generator-conditional incremental response, not natural-artifact accuracy. (C) Predicted area on untouched real fields; circles/diamonds denote COMET/CosMx. Those fields were not adjudicated as artifact-free, so the quantity is alert burden—not false-positive rate.

{proxy_table}

The nominal-reference anomaly branch produces crack-response Dice of only **{proxy_anomaly_crack["COMET"]:.3f} on COMET** and **{proxy_anomaly_crack["CosMx"]:.3f} on CosMx**. Source-identity, complete out-of-fold coverage, within-fold role-overlap, finite-output, incremental-score-definition, and horizontal-flip/inverse-transform checks passed. The weak response is consistent with a representation/channel/physical-scale/threshold mismatch, although those recorded checks cannot exclude a semantic implementation defect. The correct operational conclusion is that the present configuration should not ship; it is not that anomaly detection as a field is invalid.

![Figure 5. Multiplex proxy resolution sensitivity](figures/figure5_proxy_resolution_sensitivity.svg)

**Figure 5. Multiplex proxy resolution sensitivity.** Signed change in group-macro perturbation Dice from 256- to 896-pixel analysis for each matched modality, method, and artifact. The largest absolute change is **{resolution_audit["largest_absolute_delta"]:.3f}** for {resolution_audit["largest_shift_label"]}; no interval is estimated for these paired configuration differences. This is an engineering sensitivity analysis, not natural-artifact validation.

The proxy is explicitly marked `report_eligible=false` and `scientific_validation_passed=false`. Its {proxy_audit["comet_fields"]} COMET fields are DAPI-only and lack usable MPP metadata in the current manifest; CosMx uses all five morphology channels as structural input. Pooling them into a unified accuracy score would be scientifically invalid.

## 3. Foundation models, SOTA coverage, and compute feasibility

{model_table}

The current executable H&E registry contains only DINOv2, Hibou-B, and SigLIP2 [7,10–13]. HistoQC [3], GrandQC [4,5,30], DiffusionQC [6], DINOv3 [8,9], UNI2-h [14], CONCHv1.5 [16], KRONOS2 [18], PaDiM, full PatchCore, U-Net, and SegFormer were **not executed**. References 15, 17, and 19 describe the original UNI, CONCH, and KRONOS lineages, respectively—not validation studies of UNI2-h, CONCHv1.5, or KRONOS2. Accordingly, this is not yet a comprehensive SOTA leaderboard.

DINOv2 remains worth retaining because it is a stable Apache-2.0 generic dense-feature control [7,29]; newer does not imply better for histology artifacts. DINOv3 is a worthwhile general-vision addition after Meta-license review [8,9]. UNI2-h and CONCHv1.5 are pathology encoders, not artifact segmenters, and require a matched dense readout [14,16]. No official “CONCH v2” checkpoint was located in the official-source search completed 2026-08-27. KRONOS2 is the most technically aligned public marker-aware candidate among those reviewed for COMET-like multiplex IF [18], but it is not a ready detector and is only partially relevant to CosMx morphology/protein—not decoded transcript coordinates; the KRONOS paper [19] is lineage context, not KRONOS2 validation.

The UNI2-h, CONCHv1.5, and KRONOS2 cards use gated CC BY-NC-ND terms [14,16,18]. At Merck, frozen feature extraction and especially LoRA must not be presumed permitted; obtain written institutional approval first. HistoQC is an open operational WSI-QC baseline [3], not a directly scoreable fold-localization comparator until its outputs are mapped to the project ontology and metric contract. GrandQC is scientifically important but noncommercial [30]. No public DiffusionQC code/checkpoint was located in the author, publisher, and repository search completed 2026-08-27; because the paper includes Merck authors, internal asset and exact-split access is preferable to reimplementation.

### Apple MPS and LoRA engineering smoke

{smoke_table}

Both DINOv2-small and SigLIP2 Base completed finite frozen inference on Apple MPS with close CPU agreement and a one-step rank-4 LoRA update. These two-patch smokes establish only engineering feasibility. The central foundation runtime and CLI selector currently resolve `auto` to **MPS then CPU** and do not accept or detect CUDA; unified automatic CUDA/MPS support should not be claimed until explicit CUDA selection, synchronization, and parity tests are added.

## 4. Public datasets suitable for the next benchmark

| Resource | Ground truth and best use | Critical limitation |
| --- | --- | --- |
| [GrandQC manual test set](https://zenodo.org/records/14039591) [4,5] | Expert patch/crop-level artifact masks sampled from H&E WSIs; useful external patch-level benchmark | No distinct crack class; noncommercial method assets [30]; deliberate artifact enrichment |
| [AIRAQc TCGA test](https://openreview.net/attachment?id=XNNsQqs1UP&name=pdf) [24] | 50 manually annotated real TCGA H&E WSIs | Verify release URL, exact files, and split before use |
| [Histology Tissue Fold Dataset v1](https://zenodo.org/records/21493260) [1,2] | 1,228 fold masks plus 899 clean fields; current real benchmark | Teaching-slide microscope fields, not WSI; fold only |
| [HistoArtifacts](https://zenodo.org/records/10809442) [25,26] | Patch labels including fold and damaged tissue | No dedicated pixel crack masks; license needs confirmation |
| [Foucart artifact WSI set](https://zenodo.org/records/3773097) [27,28] | Partial Tear&Fold and knife-damage annotations on 22 WSIs | Most artifacts unannotated; ordinary negative-pixel Dice is invalid; license blank |
| Public COMET/CosMx releases [20–23] | Real fields for compatibility, drift, and blinded annotation pilots | No fold/crack ground truth found |

GrandQC TCGA masks are model-generated pseudo-labels and must not be used as independent ground truth. No mature public dataset with a separate, complete, pixel-level histologic crack class was found; tear, knife damage, or damaged tissue are only proxies. There is no single labeled cross-modality H&E–COMET–CosMx benchmark.

## 5. Evaluation criteria for the internal locked study

The evaluation must define modality, artifact ontology, unit of inference, and operational action before annotation. Fold and crack should not share a primary geometric endpoint: folds are regions, while cracks/tears are often thin branching structures.

![Figure 6. Evidence boundary and next validation gate](figures/figure6_evidence_scope.svg)

**Figure 6. Evidence boundary and validation sequence.** The current software platform executes across all three input types, but natural-label efficacy is limited to H&E folds. A dual-reviewer ontology pilot, group-disjoint development/calibration, untouched internal test, and prospective silent validation are sequential—not interchangeable—evidence gates.

{acceptance_table}

Additional design requirements:

1. Split at the highest available patient/block/slide/run unit; keep all regions and repeated scans from that unit together.
2. Predeclare production-prevalence, enriched-challenge, external-generalization, degraded-input, and downstream-impact cohorts. Do not pool a primary score across H&E, COMET, and CosMx.
3. Use two blinded reviewers plus adjudication; include tissue/background validity and ignore regions; report inter-reader agreement.
4. For folds, prioritize positive-slide macro Dice, lesion sensitivity, surface Dice at a physical tolerance, and clean-tissue overmask burden.
5. For cracks/tears, prioritize clDice or tolerant centerline F1, lesion sensitivity, false components per tissue area, and fragmentation; keep pixel Dice diagnostic.
6. For triage, report average precision/AUROC and sensitivity at a prespecified review burden. Raw score quantiles are not calibrated probabilities; do not report ECE/Brier until probability calibration is defined.
7. Repeat head fitting/reference-bank sampling and calibration across seeds, or use nested resampling that refits; bootstrap the highest independent unit.
8. Predeclare DAPI-only, morphology-RGB, and all-structural channel ablations with semantic marker roles and microns-per-pixel. Test missing/swapped channels and batch shifts.
9. Measure WSI latency, peak memory, failed/abstained inputs, review area, and downstream assay impact. Synthetic perturbations remain regression/stress tests only.

The numerical thresholds above are **proposed starting hypotheses**, not established Merck requirements. They require stakeholder approval and power analysis based on production prevalence and asymmetric costs of false PASS versus false FAIL.

## 6. Prioritized implementation plan

1. **Lock the ontology and action.** Separate tissue fold, tissue tear, glass/coverslip crack, knife line, and acquisition seam; map each to PASS/REVIEW/FAIL and remediation.
2. **Build an adjudicated pilot.** Aim initially for at least 30 independent patient/slide/run groups per modality, enriched to at least 50 fold-positive, 50 crack-positive, and 100 clean/hard-negative regions, then revise through power analysis.
3. **Add accessible controls.** Execute HistoQC as an operational WSI-QC baseline after defining its output-to-fold ontology/metric mapping, verify AIRAQc access, seek GrandQC permission, and obtain Merck's DiffusionQC code/checkpoints and exact split metadata.
4. **Use matched readouts.** Compare classical, DINOv2, Hibou-B, and approved DINOv3 under the same frozen heads. Add a small decoder or LoRA only after a predeclared frozen-model criterion fails.
5. **Open a governed multiplex track.** Define marker vocabulary, MPP, channel presence/order, and projection baselines. Request KRONOS2 permission, then pair it with a localization head and real labels.
6. **Close deployment gaps.** Add CUDA auto-detection and parity tests; preserve score maps/masks for overlays; add WSI pyramid I/O, seam handling, peak-memory tests, and technical abstention.

## 7. Hash-selected qualitative audit

![Figure 7. Hash-selected H&E qualitative audit](figures/figure7_he_qualitative.svg)

**Figure 7. Audited whole-field H&E localization examples.** The {len(qualitative_rows)} displayed fields were algorithmically selected without manual image review during this audit as the SHA-256-minimum fold-positive field within each organ plus the SHA-256-minimum presence FP and FN separately for each compared method; the union was deduplicated. The whole 896 W × 504 H px analysis fields were isotropically resized from 3,840 W × 2,160 H px and were not cropped. Solid teal with a white halo marks the supplied reference fold mask, dashed orange the classical prediction, and dotted magenta the Hibou-B linear-probe prediction. TP/FP/FN/TN denote the image-presence operating point; Dice is the pixel-localization value on fold fields. Presence calls and pixel masks use separate locked thresholds; a presence TN or FN may still contain thresholded localization pixels. All regenerated pixel counts and presence calls matched the frozen artifacts, and image scores were within recorded numerical tolerances. The fields are a reproducibility and failure-mode audit—not a representative sample or an additional performance estimate. Any embedded scale bar is source-supplied; no new physical-scale conversion was applied.

The panels make two metric distinctions tangible. A field can have substantial pixel overlap yet be an image-level false negative because localization and presence use different calibrated summaries, as in the small-intestine and testis classical cases. Conversely, both methods have a hash-selected clean-field false-positive example. Visual inspection therefore supports retaining localization, presence, and clean alert burden as separate endpoints.

## Limitations

- Natural pixel ground truth is currently fold-only H&E microscopy. There is no natural crack, human clinical WSI, internal Merck, COMET, or CosMx efficacy result.
- Readout supervision differs. Linear-probe, PatchKNN, and classical results answer different label-efficiency questions.
- The H&E test has only 28 positive supplied-source-slide groups, with small organ strata, and no patient/block identity.
- The same held-out public cohort has informed multiple result reviews; an untouched external confirmation set is now required.
- The multiplex experiment has only five and four provisional groups, unverified biological independence, generator-derived targets, shared thresholds, and resolution sensitivity.
- The hardened artifacts retain per-field counts and field scores, not reusable score maps or fitted-probe parameters. Figure 7 therefore required a deterministic shallow-head refit on the exact frozen fit split; calibration was not rerun, counts and calls matched, and scores were within recorded tolerances. Future releases should preserve a hashed inference bundle and spatial maps.
- The benchmark does not yet cover the main named QC pipelines or the newer/gated foundation models.

## Conclusion

The project has moved beyond a toy smoke: it contains a locally reproducible real-label H&E fold result in which a frozen pathology encoder plus shallow mask supervision had higher observed cohort-specific Dice than the evaluated one-class heads. The evaluated nominal-reference anomaly configuration was weak on the controlled multiplex proxy, but that result does not invalidate anomaly detection as a field. The decisive remaining bottleneck is independently adjudicated real data for the intended artifact ontology and modalities, followed by matched-head, group-locked, operationally calibrated evaluation.

## Reproducibility and source traceability

All values in this report are generated from hardened **local** JSON artifacts by `generate_report.py`. The script recomputes macro Dice, micro Dice, and clean predicted-area fractions from per-field counts, asserts identical H&E split manifests and cohort order across methods, and records SHA-256 hashes in `analysis_checks.json`. The upstream `artifacts/` directory is gitignored and is not duplicated in this report package; external or manuscript reproduction therefore requires a frozen evidence bundle or governed DVC/object-store release containing the hashed inputs. Standalone Figures 1–7 are exported as SVG/PDF and 300-dpi PNG; Figures 1–6 are wholly vector, while Figure 7 embeds audited raster H&E fields under vector labels and legends. Figures 4–5 are proxy/sensitivity evidence, Figure 6 summarizes the evidence hierarchy, and Figure 7 is a qualitative reproducibility audit rather than an efficacy endpoint.

## References

{REFERENCES_MD}
"""


def build_source_notes(
    artifact_hashes: dict[str, str],
    figure_files: dict[str, dict[str, str]],
) -> str:
    figure_rows = []
    for name, formats in figure_files.items():
        figure_rows.append((name, formats["svg"], formats["pdf"], formats["png"]))
    return f"""# Source notes — {TITLE}

## Delivery contract

- Primary audience: technical deep-learning, computational pathology, and QC stakeholders.
- Primary mode: native interactive technical report (`artifact.json`).
- Durable manuscript source: `REPORT.md`.
- Supporting exports: vector SVG/PDF and 300-dpi PNG figures.
- Decision frame: internal feasibility, **share with caveats**; not production validation.

## Required-section map

| Requirement | Location |
| --- | --- |
| Technical summary | `REPORT.md` opening and `technical_summary` report block |
| Findings with evidence and comparisons | Sections 1–3 and 7; native charts/tables and Figures 1–7 |
| Scope, data, metric definitions | Sections 1, 2, and 5 |
| Methodology | H&E and multiplex Methods text; report `methodology` block |
| Limitations and uncertainty | H&E validity, multiplex caveats, Limitations section |
| Recommended next steps | Prioritized implementation plan |
| Questions that can change the decision | Native report `further_questions` block |

## Figure map

{markdown_table(["Figure", "SVG", "PDF", "300-dpi PNG"], figure_rows)}

## Evidence and omission policy

- Reportable efficacy is restricted to the four hardened schema-v1.2 H&E JSON reports plus the paired-comparison artifact.
- The 896-pixel COMET/CosMx LOGO (leave-one-group-out) artifact is retained only as nonreportable proxy evidence. Its matched 256-pixel counterpart is independently loaded, hashed, and used only for resolution sensitivity.
- Foundation smokes support MPS/CPU parity and one-step LoRA execution only.
- Figure 7 uses algorithmically selected whole fields and a separately validated deterministic shallow-head refit because the hardened artifacts did not retain spatial maps or fitted-probe parameters; calibration was not rerun, counts and calls matched, and scores were within recorded tolerances.
- Earlier all-synthetic feasibility results were omitted from rankings because they establish wiring, not external validity.
- DINOv3, KRONOS2, UNI2-h, CONCHv1.5, HistoQC, GrandQC, DiffusionQC, PaDiM, full PatchCore, U-Net, and SegFormer are explicitly labeled not run.
- No p-values, superiority claims, pooled multimodality score, natural multiplex FPR, or WSI/deployment claim is presented.

## Validation and QA policy

- Independent recomputation: macro Dice, micro Dice, and clean-field predicted fraction from per-field counts.
- Cohort identity: exact ordered H&E field/domain rows asserted equal for all seven methods.
- Leakage interpretation: supplied source-slide-disjoint split verified; patient/block identity unavailable.
- Uncertainty: source-slide-cluster bootstrap intervals are conditional on the fixed dataset/configuration.
- Visualization: no 3D, no dual axes, direct labels where feasible, redundant marker/fill coding, units and focused/log scales disclosed in subtitles/captions; Figure 7 combines audited raster fields with vector labels and legends.
- Interactive artifact: validated against the Data Analytics report schema and rendered for visual inspection after generation.

## Input artifact SHA-256 hashes

{markdown_table(["Path", "SHA-256"], sorted(artifact_hashes.items()))}
"""


def main() -> None:
    # Consumers must treat the package as incomplete unless this marker is
    # recreated after every payload has been validated and written.
    COMPLETION_MANIFEST.unlink(missing_ok=True)
    reports = {key: load_json(path) for key, path in HE_ARTIFACTS.items()}
    paired = load_json(PAIRED_ARTIFACT)
    proxy = load_json(MULTIPLEX_ARTIFACT)
    sensitivity_proxy = load_json(MULTIPLEX_SENSITIVITY_ARTIFACT)
    dino = load_json(DINO_SMOKE)
    siglip = load_json(SIGLIP_SMOKE)
    feasibility = load_json(FEASIBILITY_MANIFEST)
    qualitative_checks = load_json(QUALITATIVE_CHECKS)
    source_documents = {
        EVALUATION_PROTOCOL: load_text_nonempty(EVALUATION_PROTOCOL),
        PUBLIC_BENCHMARK_AUDIT: load_text_nonempty(PUBLIC_BENCHMARK_AUDIT),
    }
    if "ground truth" not in source_documents[EVALUATION_PROTOCOL].lower():
        raise AssertionError("evaluation protocol content is not recognized")
    if "public" not in source_documents[PUBLIC_BENCHMARK_AUDIT].lower():
        raise AssertionError("public benchmark audit content is not recognized")

    he_rows, organ_rows, he_audit = extract_he_results(reports)
    paired_rows, paired_audit = extract_paired_results(paired, reports)
    proxy_rows, alert_rows, proxy_summary, proxy_audit = extract_multiplex_results(
        proxy
    )
    sensitivity_rows, _, _, _ = extract_multiplex_results(sensitivity_proxy)
    resolution_rows, resolution_audit = extract_proxy_resolution_sensitivity(
        proxy, sensitivity_proxy, proxy_rows, sensitivity_rows
    )
    smoke_rows = extract_smoke_results(dino, siglip)
    feasibility_audit = validate_feasibility_manifest(feasibility)
    qualitative_rows = validate_qualitative_checks(qualitative_checks, reports)

    figure_files = {
        "Figure 1 — H&E locked test": figure_he_performance(he_rows),
        "Figure 2 — organ heterogeneity": figure_organ_heatmap(he_rows, organ_rows),
        "Figure 3 — paired differences": figure_paired_differences(paired_rows),
        "Figure 4 — multiplex proxy": figure_multiplex_proxy(proxy_rows, alert_rows),
        "Figure 5 — proxy resolution sensitivity": figure_proxy_resolution_sensitivity(
            resolution_rows, resolution_audit
        ),
        "Figure 6 — evidence scope": figure_evidence_scope(),
        "Figure 7 — qualitative H&E audit": figure_he_qualitative(qualitative_rows),
    }
    report_artifact = build_artifact(
        he_rows,
        organ_rows,
        paired_rows,
        qualitative_rows,
        proxy_rows,
        alert_rows,
        proxy_summary,
        resolution_rows,
        smoke_rows,
    )
    for dataset_id, rows in report_artifact["snapshot"]["datasets"].items():
        write_json(HERE / "data" / f"{dataset_id}.json", rows)

    inputs = [
        *HE_ARTIFACTS.values(),
        PAIRED_ARTIFACT,
        MULTIPLEX_ARTIFACT,
        MULTIPLEX_SENSITIVITY_ARTIFACT,
        DINO_SMOKE,
        SIGLIP_SMOKE,
        FEASIBILITY_MANIFEST,
        QUALITATIVE_CHECKS,
        *source_documents,
        *(HERE / row["overlay_path"] for row in qualitative_rows),
    ]
    artifact_hashes = {
        str(path.relative_to(REPO)): sha256_file(path) for path in inputs
    }
    poppler_version = subprocess.run(
        ["pdftoppm", "-v"],
        check=True,
        capture_output=True,
        text=True,
    )
    poppler_banner = (poppler_version.stderr or poppler_version.stdout).splitlines()[0]
    checks = {
        "report_date": REPORT_DATE,
        "numeric_generation_checks_passed": True,
        "claim_status": "internal feasibility; share with caveats",
        "input_availability": (
            "hashed inputs are local and gitignored; publish a frozen evidence bundle "
            "or governed object-store/DVC release for external reproduction"
        ),
        "independently_recomputed": [
            "positive-field macro Dice from per-field TP/FP/FN",
            "all-field micro Dice from pooled per-field TP/FP/FN",
            "clean-field predicted-area fraction from per-field FP and valid pixels",
            "exact H&E split-manifest/cohort identity across all seven methods",
            "duplicate image SHA-256 count across all H&E split manifests",
            "matched 896-minus-256 multiplex proxy point differences",
            "exact regenerated counts and presence calls, plus image scores within recorded tolerances, for seven algorithmically selected qualitative fields",
        ],
        "extracted_not_independently_recomputed": [
            "all bootstrap confidence intervals",
            "AUROC, average precision, sensitivity and specificity",
            "per-organ metrics",
            "paired-comparison point estimates and intervals",
            "multiplex proxy Dice and alert-burden source metrics",
            "foundation-smoke timing and parity metrics",
            "external literature and license assertions",
        ],
        "input_hashes": artifact_hashes,
        "report_build_environment": {
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "python_version": platform.python_version(),
            "reportlab_version": REPORTLAB_VERSION,
            "poppler_pdftoppm_version": poppler_banner,
            "determinism_scope": "byte-identical when audited inputs, generator, and recorded build toolchain are unchanged",
        },
        "source_documents_read": {
            str(path.relative_to(REPO)): {
                "sha256": sha256_file(path),
                "utf8_character_count": len(content),
            }
            for path, content in source_documents.items()
        },
        "he_recomputation_and_cohort_audit": he_audit,
        "paired_audit": paired_audit,
        "multiplex_proxy_audit": proxy_audit,
        "resolution_sensitivity_audit": resolution_audit,
        "synthetic_feasibility_audit": feasibility_audit,
        "qualitative_audit": {
            "status": qualitative_checks["status"],
            "n_cases": qualitative_checks["n_cases"],
            "selection_rule": qualitative_checks["selection_rule"],
            "all_outcomes_match_frozen_artifacts": qualitative_checks[
                "all_outcomes_match_frozen_artifacts"
            ],
            "encoder_frozen": qualitative_checks["model"]["encoder_frozen"],
            "shallow_readout_refit_required": qualitative_checks["model"][
                "shallow_readout_refit_required"
            ],
            "calibration_rerun": qualitative_checks["model"]["calibration_rerun"],
            "training_identity_check": qualitative_checks["training_identity_check"],
        },
        "figure_files": figure_files,
        "scientific_boundaries": {
            "real_label_efficacy": "H&E fold only",
            "wsi_efficacy": False,
            "natural_crack_efficacy": False,
            "natural_comet_efficacy": False,
            "natural_cosmx_efficacy": False,
            "comprehensive_sota_comparison": False,
        },
    }

    write_json(HERE / "artifact.json", report_artifact)
    write_json(HERE / "analysis_checks.json", checks)
    write_text(
        HERE / "REPORT.md",
        build_markdown_report(
            he_rows,
            organ_rows,
            paired_rows,
            qualitative_rows,
            proxy_rows,
            alert_rows,
            proxy_summary,
            resolution_rows,
            smoke_rows,
            he_audit,
            paired_audit,
            proxy_audit,
            resolution_audit,
        ),
    )
    write_text(
        HERE / "SOURCE_NOTES.md", build_source_notes(artifact_hashes, figure_files)
    )
    payload_paths = [
        HERE / "artifact.json",
        HERE / "analysis_checks.json",
        HERE / "REPORT.md",
        HERE / "SOURCE_NOTES.md",
        *(HERE / "data" / f"{dataset_id}.json" for dataset_id in report_artifact["snapshot"]["datasets"]),
        *(
            HERE / relative_path
            for formats in figure_files.values()
            for relative_path in formats.values()
        ),
    ]
    if len(payload_paths) != 39 or len(set(payload_paths)) != len(payload_paths):
        raise AssertionError("unexpected generated report payload inventory")
    if any(not path.is_file() or path.is_symlink() for path in payload_paths):
        raise AssertionError("generated report payload is missing or linked")
    write_json(
        COMPLETION_MANIFEST,
        {
            "schema_version": "foldcrack-report-build-complete-1.0",
            "status": "complete",
            "report_date": REPORT_DATE,
            "generator_sha256": sha256_file(Path(__file__).resolve()),
            "n_payloads": len(payload_paths),
            "payload_sha256": {
                str(path.relative_to(HERE)): sha256_file(path)
                for path in sorted(payload_paths)
            },
        },
    )


if __name__ == "__main__":
    main()
