# Benchmark findings report package

This directory contains the reproducible technical report for the current fold
and crack QC evidence across H&E, COMET, and CosMx.

## Start here

- `REPORT.md` — manuscript-oriented findings, methods, limitations, citations,
  evaluation criteria, and prioritized next steps.
- `artifact.json` — validated native Data Analytics report.
- `figures/` — seven manuscript figures in SVG, PDF, and 300-dpi PNG.
- `analysis_checks.json` — independent metric recomputation, cohort-identity
  checks, scientific claim boundaries, and input SHA-256 hashes.
- `qualitative_checks.json` and `qualitative_cache/` — fail-closed regeneration
  record and seven hash-selected whole-field overlays used by Figure 7.
- `VALIDATION.md` — schema/render, regression-test, deterministic-build, and
  figure-integrity record.
- `SOURCE_NOTES.md` — provenance, section/figure map, omissions, and QA policy.
- `data/` — bounded JSON tables that back the native charts and tables.

## Rebuild

The report generator reads existing immutable local benchmark artifacts and the
validated qualitative bundle; it does not rerun model inference. The upstream
`artifacts/` directory is gitignored and is not copied into this package, so
external reproduction requires the separately released evidence bundle
identified by `analysis_checks.json` hashes.

```bash
python reports/benchmark_findings_2026-08-27/generate_report.py
```

The audited build used Python 3.12.13, ReportLab 4.4.9, and Poppler
`pdftoppm` 26.05.0; `requirements-report.txt` pins ReportLab. SVG and PDF are
manuscript masters; PNG is rasterized from PDF at 300 dpi. Byte-identical
exports are claimed only when the hashed inputs, generator, and recorded
toolchain in `analysis_checks.json` are unchanged.
Figures 1–6 are wholly vector. Figure 7 embeds audited raster H&E fields under
vector labels and legends.

Figure 7 can be regenerated separately when the frozen public data, source
tree, and hash-locked local Hibou-B weights are present:

```bash
PYTHONPATH=src:. ./.venv/bin/python \
  reports/benchmark_findings_2026-08-27/generate_qualitative_overlays.py \
  --device mps
```

That no-download step refits only the deterministic shallow readout, does not
rerun calibration, reuses the locked thresholds, and fails before writing
report-facing overlays unless regenerated counts and presence calls match the
stored artifacts and image scores fall within recorded numerical tolerances.
It takes several minutes on the tested Apple Silicon host. A `--device cpu`
code path is implemented but was not audited by this report package. If the
audited local Hibou checkout or weights were relocated, pass `--hibou-source
/path/to/source` and `--hibou-weights /path/to/hibou-b.pth`; the frozen commit
and weights SHA-256 are still enforced.

## Scientific status

The real-label efficacy claim is restricted to fold localization on the public
H&E microscope-field benchmark. COMET and CosMx results are real-background
synthetic-perturbation proxies and are explicitly not natural-artifact
accuracy. No natural crack, WSI deployment, or three-modality efficacy claim is
supported by the present evidence.
