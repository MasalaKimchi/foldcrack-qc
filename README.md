# FoldCrack QC Feasibility Framework

A runnable, modality-aware starting point for fold and crack-like artifact QC
on H&E whole-slide images, Lunaphore COMET multiplex fluorescence, and Bruker
CosMx morphology imaging.

The design deliberately separates two questions:

1. **Does the software pipeline work end to end?** The included deterministic
   synthetic benchmark answers this immediately on a CPU-only Mac.
2. **Does a method work on Merck data?** That requires a locked, independently
   reviewed internal reference set. Synthetic and public data cannot establish
   that claim.

## What runs now

- Canonical H&E, COMET, and CosMx channel-role adapters
- Synthetic tissue, fold, crack/tear, and hard-negative generation
- Factorial clean, hard-negative-only, fold-only, crack-only, and combined cohorts
- Interpretable classical fold/crack candidates
- Clean-reference unsupervised anomaly scoring
- Hybrid classical + anomaly fusion
- Semantic structural-channel versus nuclear-only ablation
- Pixel, boundary, centerline, instance, burden, runtime, and bootstrap metrics
- Method-separated, cluster-bootstrap comparison reports
- Strict real-data manifest validation and leakage/checksum checks
- Operational `PASS / REVIEW / FAIL` safety-gate evaluator
- Physical-scale tile/halo/stitching contract for WSI reader integration
- JSON, CSV, Markdown, and visual-overlay reports
- Strict, machine-readable real-benchmark eligibility validation
- A strict, source-slide-grouped public H&E fold benchmark with real expert masks
- A real-background, label-free proxy benchmark on public COMET and CosMx inputs
- Optional OME-TIFF support and frozen DINOv2/Hibou-B feature paths
- An auditable offline DINOv2 CPU/MPS smoke test with optional BF16 LoRA
- A hash-locked Apache-2.0 SigLIP2 Base dense-feature path with MPS/LoRA smoke

## Run the complete feasibility test

The current environment already contains NumPy, SciPy, and OpenCV, so no model
or dataset download is required:

```bash
make test
make feasibility
```

Equivalent direct commands:

```bash
PYTHONPATH=src python3 -m foldcrack_qc test
PYTHONPATH=src python3 -m foldcrack_qc feasibility \
  --output artifacts/feasibility \
  --samples-per-modality 12 \
  --size 384 \
  --patch-size 32
```

The benchmark writes:

```text
artifacts/feasibility/
├── FEASIBILITY_REPORT.md
├── RUN_MANIFEST.json
├── comparison.csv
├── comparison.json
├── evaluation_report.json
├── evaluation_report.md
├── operational_acceptance.json
├── per_sample_results.csv
└── overlays/{he,comet,cosmx}/
```

`engineering_smoke_test_passed=true` means the software paths completed. It
never means that scientific validation passed; the manifest records that
distinction explicitly.

The synthetic runner uses five factorial scenarios per modality. Therefore
`--samples-per-modality` must be at least 5.

## Methods compared

| Method | Training requirement | Purpose |
|---|---|---|
| Classical | None | Interpretable fold/crack baseline and candidate generator |
| Clean-reference anomaly | Reviewed clean tiles only | Label-light detection of unusual regions |
| Hybrid | Clean reference plus fixed fusion | Recommended phase-1 high-recall review aid |
| Frozen DINOv2 | Pinned pretrained weights plus clean fit/calibration cohorts | Small, generic RGB control for one-class and supervised frozen-token heads |
| Frozen Hibou-B | Pinned official pathology weights plus the same cohorts and heads | Pathology-specific H&E comparator runnable on Apple MPS |
| Frozen SigLIP2 Base | Hash-locked ungated Apache-2.0 snapshot plus the same cohorts and heads | Newer generic RGB comparator; MPS/LoRA smoke and hardened real H&E frozen-head evaluation completed |
| DINOv2 + LoRA | Adjudicated fit and disjoint calibration cohorts | Resource-feasibility path; efficacy still requires adjudicated target-domain labels and a locked test |

An anomaly score is not a fold or crack label. It must be calibrated against
expert-reviewed positives and difficult normal anatomy.

## Real-data ingestion

### Public real H&E fold cohort

The repository now has an executable benchmark for the permissively licensed
[Histology Tissue Fold Dataset v1.0](https://doi.org/10.5281/zenodo.21493260).
It contains 2,127 real 10x H&E microscope fields, 1,228 publisher-supplied fold
masks, and a slide-level mapping. The associated peer-reviewed
[Bioengineering article](https://doi.org/10.3390/bioengineering13080937)
documents acquisition and expert review. This is real fold evidence, but it is
veterinary teaching-slide microscopy—not human clinical WSI, crack, COMET, or
CosMx evidence.

Download about 6.7 GB of archives (about 12 GB extracted) from the publisher:

```bash
mkdir -p data/public/histology_tissue_fold_v1
curl -L 'https://zenodo.org/records/21493260/files/images.zip?download=1' \
  -o data/public/histology_tissue_fold_v1/images.zip
curl -L 'https://zenodo.org/records/21493260/files/masks.zip?download=1' \
  -o data/public/histology_tissue_fold_v1/masks.zip
curl -L 'https://zenodo.org/records/21493260/files/metadata.csv?download=1' \
  -o data/public/histology_tissue_fold_v1/metadata.csv
curl -L 'https://zenodo.org/records/21493260/files/slide_image_mapping.xlsx?download=1' \
  -o data/public/histology_tissue_fold_v1/slide_image_mapping.xlsx
curl -L 'https://zenodo.org/records/21493260/files/LICENSE.txt?download=1' \
  -o data/public/histology_tissue_fold_v1/LICENSE.txt
curl -L 'https://zenodo.org/records/21493260/files/README.md?download=1' \
  -o data/public/histology_tissue_fold_v1/README.source.md
```

Verify the exact publisher assets before extraction:

```bash
md5 data/public/histology_tissue_fold_v1/images.zip \
  data/public/histology_tissue_fold_v1/masks.zip \
  data/public/histology_tissue_fold_v1/metadata.csv \
  data/public/histology_tissue_fold_v1/slide_image_mapping.xlsx
```

Expected MD5 values are `2d4116bf652d1dba761d3164909d8df3`,
`7418aa15933b013e759db4e02b565cb7`,
`bb4bd671afa378675c32f5b288bebd30`, and
`fd3dcab8da600d35955fde179b2db993`, respectively. Extract both archives in the
dataset root. The strict loader decodes every asset, checks binary masks and
pairing, and detects two empty publisher masks; use the explicit audited
localization-exclusion option rather than silently converting them to clean.

All current methods use the same organ-by-class, source-slide-grouped
`60% / 20% / 20%` assignment: 1,276 fit fields from 170 groups, 427 calibration
fields from 58 groups, and a locked test of 424 fields from 55 groups (179 clean
and 245 fold-positive). Each current artifact uses schema v1.2, validates
release/run/model provenance before scoring, records hashed per-field outcomes,
and is `report_eligible=true` for this bounded public fold cohort.

| Method | Supervision | Positive-field macro Dice (95% CI) | All-field micro Dice (95% CI) | Presence AUROC | Clean FP area |
|---|---|---:|---:|---:|---:|
| Classical fold candidate | Calibration masks/presence | 0.446 (0.375–0.521) | 0.383 (0.318–0.446) | 0.792 | 10.731% |
| DINOv2-small PatchKNN | Clean fit bank + calibration masks/presence | 0.341 (0.285–0.389) | 0.434 (0.352–0.497) | 0.884 | 1.789% |
| DINOv2-small linear probe | Fit masks + calibration masks/presence | 0.599 (0.532–0.668) | 0.770 (0.724–0.802) | 0.980 | 0.098% |
| SigLIP2 Base PatchKNN | Clean fit bank + calibration masks/presence | 0.208 (0.178–0.236) | 0.224 (0.177–0.257) | 0.803 | 7.282% |
| SigLIP2 Base linear probe | Fit masks + calibration masks/presence | 0.526 (0.462–0.590) | 0.679 (0.625–0.714) | 0.965 | 0.299% |
| Hibou-B PatchKNN | Clean fit bank + calibration masks/presence | 0.319 (0.264–0.368) | 0.418 (0.362–0.457) | 0.947 | 1.221% |
| Hibou-B linear probe | Fit masks + calibration masks/presence | 0.667 (0.603–0.730) | 0.827 (0.791–0.852) | 0.985 | 0.047% |

PatchKNN is one-class at fit time but uses labeled calibration masks and
presence labels; it is label-light, not fully unsupervised. The linear probe is
supervised by fit masks, while every encoder remains frozen. Hibou-B linear has
the strongest point estimates. In the completed exploratory paired analysis,
its positive-field macro-Dice difference versus DINOv2 linear is +0.067974
[+0.053130, +0.084232]; DINOv2 linear minus SigLIP2 linear is +0.072716
[+0.061862, +0.083173]. These source-slide bootstrap intervals are descriptive:
no p-values or multiplicity adjustment were computed, and no superiority claim
is made. The byte-reproducible
[`public-fold-paired-comparison-1.0` artifact](artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json)
has SHA-256
`9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2`.
This is seven matched method/head rows, not a comprehensive SOTA leaderboard.

Representative hardened rerun commands are:

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods classical_fold --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/classical_hardened_v1_2.json

PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods dinov2_patchknn dinov2_linear_probe \
  --foundation-encoder dinov2-hf --model-id facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --cache-dir models/hf_home/hub --device mps --max-dimension 896 \
  --probe-max-iterations 500 --bootstrap-resamples 1000 \
  --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/dinov2_hardened_v1_2.json

PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods foundation_patchknn foundation_linear_probe \
  --foundation-encoder siglip2-base-local \
  --siglip2-snapshot /path/to/hash-locked/siglip2-base-patch16-224 \
  --device mps --probe-max-iterations 500 --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/siglip2_hardened_v1_2.json

PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods foundation_patchknn foundation_linear_probe \
  --foundation-encoder hibou-b-local \
  --hibou-weights models/hibou-b/hibou-b.pth \
  --hibou-source models/hibou-b/source \
  --hibou-weights-sha256 9d3e5ebc4e1ffaf6d7a0b672273e4fbef109cdd03df73c52920d6e886f2327e1 \
  --hibou-source-commit c453bbe4dab0fec6f7df343b09ea87048629c58d \
  --device mps --probe-max-iterations 500 --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/hibou_hardened_v1_2.json
```

The current report, exact artifact hashes, and full commands are documented in
[`docs/REAL_PUBLIC_BENCHMARK.md`](docs/REAL_PUBLIC_BENCHMARK.md). Public source,
license, integrity, COMET/CosMx availability, and model-access findings are in
[`docs/PUBLIC_BENCHMARK_AUDIT.md`](docs/PUBLIC_BENCHMARK_AUDIT.md).

### Public real COMET/CosMx proxy cohort

Real public multiplex images are now executed, not only catalogued. The locked
[proxy report](docs/MULTIPLEX_REAL_PROXY_BENCHMARK.md) uses five complete DAPI
fields recovered from
[QUALIFAI COMET v2](https://zenodo.org/records/12699470), four five-channel
CosMx FOVs from the
[gastric mucosa cohort](https://zenodo.org/records/8333281), and two more from
the [pediatric high-grade glioma cohort](https://zenodo.org/records/16877090).
The six CosMx FOVs have four distinct locked slide/run identifiers; the public
releases do not establish higher-level biological independence.

None of those released inputs supplies independently usable natural fold/crack
masks. Within each LOGO fold, fit/calibration/test roles are source-group
disjoint; across folds, groups are reused, so folds are statistically dependent.
The benchmark inserts controlled folds/cracks into held-out real backgrounds and
evaluates the paired incremental response. Mean calibrated Dice was:

| Modality | Classical | Clean-reference anomaly | Hybrid |
|---|---:|---:|---:|
| COMET | 0.534 | 0.100 | 0.475 |
| CosMx | 0.314 | 0.134 | 0.305 |

These v3 values are derived arithmetic means of the separate crack and fold
group-macro point estimates after each provisional source group was held out
once; that derived average has no direct CI. Group-bootstrap intervals in the
raw report are descriptive because higher-level biological independence is not
declared. The clean-reference anomaly branch was particularly weak on the
controlled thin crack cue: Dice 0.065 on COMET and 0.075 on CosMx. Coverage,
finite-score,
paired-subtraction, determinism, and flip/inverse checks passed, so zero anomaly
Dice is not being hidden as a coding success. These metrics are
**generator-conditional proxy evidence**, not natural-artifact accuracy. The
JSON correctly records `report_eligible=false`; activity on untouched fields is
reported as alert burden, never false-positive rate. Natural-artifact Dice, ROC,
sensitivity, specificity, and FPR are not estimable from these public inputs.
The [primary v3 artifact](artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json)
has SHA-256
`a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`.

The 256-pixel sensitivity run changed Dice by as much as 0.332, and COMET crack
severity response was nonmonotonic at 896 px. The proxy therefore falsifies any
claim that the current ranking is resolution-robust or decision-ready.

The adapters accept NumPy arrays directly and load `.npy`, `.npz`, and common
image formats. `tifffile` is an optional dependency for richer TIFF/OME-TIFF
support:

```bash
python3 -m pip install -e '.[wsi]'
```

For internal COMET and CosMx files, extract channel identities from OME or
experiment metadata and map them to semantic roles. Never assume a fixed
channel index. See `configs/channel_roles/`.

Validate a de-identified internal export before inference:

```bash
PYTHONPATH=src python3 -m foldcrack_qc validate-manifest \
  /secure/foldcrack/manifest.json --strict --json
```

The strict contract verifies MPP, channel metadata, lossless binary masks,
patient/block/slide/run split isolation, valid/ignore regions, and SHA-256
provenance. See `configs/internal_manifest.example.json` and `data/README.md`.

Before any result can be called a real-data benchmark, validate the explicit
method/cohort/task contract:

```bash
PYTHONPATH=src python3 -m foldcrack_qc validate-benchmark \
  configs/benchmark.real.example.json \
  --require-report-eligible --json
```

The checked-in example is intentionally a plan, not a fabricated result. It
returns exit code `3` until realized, disjoint fit/calibration/locked-test
records, complete adjudicated masks, governance approvals, and enabled method
assets are supplied. Configuration validity and scientific report eligibility
are separate states.

The present executable is an ROI/downsampled-slide feasibility runner, not a
production native-pyramid WSI engine. For deployment, wrap the same canonical
contract with OpenSlide/cuCIM/vendor readers and use a two-scale path: screen
tissue at a locked physical resolution, refine candidates at a higher physical
resolution, carry a halo around tiles, and stitch in level-0 coordinates.
Record scanner MPP and pyramid level for every result; a pixel-sized threshold
must never be shared across scanners or modalities.

`foldcrack_qc.wsi` provides tested physical-scale level selection, tile halos,
level-0 coordinate mapping, and seam-free core stitching. Its in-memory pyramid
is a test double; production still needs an approved OpenSlide, cuCIM,
OME-Zarr, or vendor reader.

## Foundation-model path

The executable core does not require PyTorch. To install the optional frozen
feature and parameter-efficient adaptation stack after model-governance review:

```bash
python3 -m pip install -e '.[foundation,adaptation]'
```

An actual `facebook/dinov2-small` run has been verified on Apple MPS with an
exact revision and SHA-256 weight identity. Reproduce it offline from an
explicit local cache:

```bash
PYTHONPATH=src python3 -m foldcrack_qc foundation-smoke \
  --model-id facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --cache-dir models/hf_home/hub \
  --device mps --steady-runs 3 --lora-rank 4 \
  --output-json artifacts/foundation_smoke/foundation_smoke.json
```

That run performs frozen CPU and MPS inference, checks global and spatial-token
agreement, and takes one BF16 LoRA optimization step on the last four attention
blocks. It is deliberately labeled `engineering_foundation_smoke_only`: the two
deterministic patches establish runtime feasibility, not artifact-detection
efficacy. See `docs/FOUNDATION_FEASIBILITY.md` for the measured result.

The hash-locked
[SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224) vision
tower has also passed local Apple-MPS inference, CPU/MPS agreement, and a rank-4
LoRA engineering step. Its official model card is ungated Apache-2.0. The
[reproducible smoke report](docs/SIGLIP2_MPS_LORA_SMOKE.md) records a 0.032-second
MPS steady pass, 0.00115 maximum dense CPU/MPS difference, and 49,921 trainable
adapter-plus-head parameters (0.0537%). It is a generic RGB model, and these
engineering measurements are not artifact-efficacy claims.

Once three strict, mutually isolated H&E manifests exist, run the initial real
frozen-feature comparator with physical patch geometry:

```bash
PYTHONPATH=src python3 -m foldcrack_qc frozen-feature-benchmark \
  --fit-manifest /secure/foldcrack/he_fit.json \
  --calibration-manifest /secure/foldcrack/he_calibration.json \
  --locked-test-manifest /secure/foldcrack/he_locked_test.json \
  --model-id facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --cache-dir models/hf_home/hub --device mps \
  --patch-size-um 112 --stride-um 56 \
  --output-json artifacts/real_he_dinov2/report.json
```

The runner requires positive `acquired_real` plus approved-provenance declarations
and explicit adjudicated all-zero fold and crack masks for every reviewed-clean
fit/calibration case—absence of a mask never means negative. It rejects synthetic
provenance and cross-split patient/block/slide/run/source/file/content overlap,
chooses the threshold only on calibration, evaluates the locked test at native
resolution, preserves failures as abstentions, and serializes only anonymous
keys. Its generic anomaly output is compared only with the
`artifact_union` reference, never presented as a fold/crack subtype classifier.
This command is deliberately limited to pre-extracted ROI/downsampled rasters;
native streaming WSI integration still requires an approved pyramid reader and
disk-backed tiled score output.

Use a separate clean reference bank and calibration per modality. Generic
DINOv2 and pathology RGB encoders are treated as H&E/RGB-only unless a governed
semantic projection for multiplex channels is defined; the encoder rejects
silently taking the first three COMET/CosMx channels. Hibou-B's official
Apache-2.0 assets are pinned and verified locally. The official current names
for the larger Mahmood Lab candidates are
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and
[CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5); no official
“CONCH-v2” artifact was found in the date-stamped audit. UNI2-h, CONCHv1.5,
Virchow2, and KRONOS2 have gated/noncommercial terms that require written
Merck approval before evaluation or adaptation. KRONOS2 is the more
modality-aligned foundation candidate for COMET/CosMx, but it still needs
artifact labels and a validated localization head. See
[`docs/PATHOLOGY_FOUNDATION_MODEL_DECISION.md`](docs/PATHOLOGY_FOUNDATION_MODEL_DECISION.md).

[DINOv3](https://github.com/facebookresearch/dinov3) is newer than DINOv2 and
technically relevant for dense localization, but it was not downloaded or run.
Official checkpoint access requires license acceptance, and the
[custom DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
defines the accepting party to include an employer/entity when accepted on its
behalf. Merck license/governance approval is required first; this repository
does not assert that the DINOv3 license is noncommercial.

## Evaluation contract

- [`docs/TEAM_BRIEF.md`](docs/TEAM_BRIEF.md): decision-ready recommendation,
  evidence plan, illustrative gates, and immediate team asks
- [`docs/AI-SPEC.md`](docs/AI-SPEC.md): architecture, risks, model ladder, and
  guardrails
- [`docs/FOUNDATION_FEASIBILITY.md`](docs/FOUNDATION_FEASIBILITY.md): pinned
  DINOv2/MPS/LoRA execution result and its evidence boundary
- [`docs/PUBLIC_BENCHMARK_AUDIT.md`](docs/PUBLIC_BENCHMARK_AUDIT.md): verified
  public methods, real datasets, label coverage, and license/access gaps
- [`docs/MULTIPLEX_REAL_PROXY_BENCHMARK.md`](docs/MULTIPLEX_REAL_PROXY_BENCHMARK.md):
  real COMET/CosMx execution, generator-conditional proxy results, label-free
  metrics, and exact evidence limits
- [`docs/REAL_PUBLIC_BENCHMARK.md`](docs/REAL_PUBLIC_BENCHMARK.md): current
  natural-mask H&E split, supervision budgets, seven matched results,
  method-wise and exploratory paired intervals, artifact hashes, and hardened
  rerun commands
- [`docs/PATHOLOGY_FOUNDATION_MODEL_DECISION.md`](docs/PATHOLOGY_FOUNDATION_MODEL_DECISION.md):
  DINOv2 control rationale, Hibou-B evidence, governed alternatives, and
  multiplex model decision
- [`docs/ANNOTATION_GUIDE.md`](docs/ANNOTATION_GUIDE.md): operational fold,
  tear/crack, confounder, severity, and adjudication definitions
- [`docs/EVALUATION.md`](docs/EVALUATION.md): locked cohorts, metrics,
  confidence intervals, generalization matrix, acceptance gates, and
  downstream-impact tests
- [`configs/acceptance.example.json`](configs/acceptance.example.json):
  illustrative thresholds that stakeholders must approve before use

Operational decisions can be evaluated separately from localization:

```bash
PYTHONPATH=src python3 -m foldcrack_qc operational-eval \
  --records configs/operational_records.example.json \
  --acceptance configs/acceptance.example.json \
  --synthetic
```

Synthetic execution intentionally returns `NOT_EVALUATED_SYNTHETIC`; no mask
benchmark can satisfy a real operational acceptance gate.

## Public-resource registry

```bash
PYTHONPATH=src python3 -m foldcrack_qc datasets
```

The registry records modality, label coverage, and license uncertainty. It is
a discovery aid, not legal clearance. No public resource found provides a
credible fold-and-crack reference across all three modalities.

## Recommended internal workflow

1. Confirm whether “crack” means tissue tear, coverslip/glass crack, knife line,
   or acquisition seam, and define the downstream action.
2. Label a 30–50 case-per-modality taxonomy pilot with two independent
   reviewers and adjudication.
3. Run this framework on that development set, tune only there, and freeze the
   ontology, channel roles, aggregation, and thresholds.
4. Evaluate on separate prevalence, enriched-challenge, device/site, temporal,
   and new-panel cohorts.
5. Start as `PASS / REVIEW / FAIL` decision support with abstention. Do not
   automatically discard tissue until prospective human-in-the-loop validation
   and downstream noninferiority have passed.

## Repository layout

```text
configs/                 Channel roles, resource registry, example gates
data/                    Data-handling rules; private/raw files stay ignored
docs/                    AI, annotation, and evaluation contracts
src/foldcrack_qc/        Adapters, WSI contract, detectors, benchmark, evaluation
tests/                   Dependency-light unit and end-to-end tests
artifacts/               Generated reports; ignored by Git
```
