# Validation record — Fold and Crack Artifact QC report

**Validated:** 2026-08-27

**Status:** PASS for internal technical review, with the scientific claim boundaries stated in `REPORT.md`.

## Numerical and provenance checks

- `analysis_checks.json` reports `numeric_generation_checks_passed=true`.
- Positive-field macro Dice, all-field micro Dice, and clean-field predicted-area fraction were recomputed from per-field TP/FP/FN for all seven H&E methods and matched the source artifacts.
- Exact ordered test-cohort identity and the full fit/calibration/test assignment manifest were equal across all seven H&E methods.
- The 2,127 H&E image hashes were unique; no field or supplied source-slide group crossed split boundaries.
- The 896-minus-256 multiplex sensitivity values were recomputed from independently loaded and hashed source artifacts.
- Seven whole H&E fields were algorithmically selected by an image-SHA-256 rule without manual image review during this audit. The exact frozen Hibou-B encoder and fit split were used to refit only the shallow readout; calibration was not rerun. All 14 regenerated method outcomes matched stored pixel counts and presence calls, and image scores were within recorded numerical tolerances.
- Confidence intervals, AUROC/AP, per-organ metrics, paired-bootstrap outputs, source multiplex metrics, smoke timings, and literature/license assertions were extracted or source-audited but were not independently recomputed; this is disclosed in `analysis_checks.json`.

## Software checks

- Ruff formatting: PASS.
- Ruff lint: PASS.
- Generator byte-code compilation: PASS with the bundled report Python runtime.
- Targeted regression suite: **81 tests and 2 subtests passed** in 2.77 seconds.
- Test command:

  ```bash
  PYTHONPATH=src:. ./.venv/bin/python -m pytest -q \
    tests/test_compare_hardened_public_fold.py \
    tests/test_public_fold_benchmark.py \
    tests/test_multiplex_proxy_benchmark.py \
    tests/test_foundation_smoke.py \
    tests/test_evaluation.py
  ```

- The repository `.venv` is required for this suite; the global Python environment does not contain the project package and OpenCV.
- `git diff --check`: PASS.
- Qualitative regeneration: PASS on native Apple MPS (`status=passed`, `n_cases=7`, encoder frozen, shallow-head identity checks passed, no calibration rerun, no model download).

## Report and figure checks

- Native Data Analytics report schema validation: PASS (`ok=true`, report surface, 14 bounded datasets, 18 provenance sources, snapshot status `ready`).
- Validated native artifact render: PASS.
- Seven standalone figures exported in SVG, single-page PDF, and PNG. Figures 1–6 are wholly vector; Figure 7 embeds audited raster fields under vector labels and legends.
- Every PNG has 300 × 300 dpi metadata; dimensions range from 2,125–2,917 pixels wide. Figure 7 is 2,125 × 3,334 px (7.08 × 11.11 in at 300 dpi); its live-area height should be checked against the target journal before submission.
- All seven SVG files pass XML parsing.
- Visual QA was performed at full resolution for hierarchy, clipping, label collisions, legends, scales, units, and caveat visibility.
- A second complete generator run produced identical SHA-256 hashes for all **39** generated JSON, Markdown, SVG, PDF, and PNG outputs under the same recorded build environment. ReportLab PDF export uses invariant metadata; byte-identical determinism is scoped to unchanged audited inputs, generator, Python 3.12.13, ReportLab 4.4.9, and Poppler `pdftoppm` 26.05.0.

## Scientific release boundary

This validation confirms report generation, source traceability, numerical spot checks, software regressions, and figure integrity. It does **not** upgrade the scientific evidence. Natural-label efficacy remains limited to tissue-fold localization on the public H&E microscope-field cohort. COMET/CosMx values are controlled synthetic-perturbation responses on real backgrounds, not natural-artifact accuracy, and no natural crack or WSI deployment claim is supported.

The upstream benchmark artifacts are local and gitignored. External reproduction requires a frozen evidence bundle or governed object-store/DVC release matching the SHA-256 values in `analysis_checks.json`.
