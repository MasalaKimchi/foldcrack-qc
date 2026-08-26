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
- Optional OME-TIFF support and a frozen DINOv2 feature/anomaly path
- An auditable offline DINOv2 CPU/MPS smoke test with optional BF16 LoRA

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
  --size 384
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
| Frozen DINOv2 | Pinned pretrained weights plus clean fit/calibration cohorts | H&E/RGB feature and generic artifact-union anomaly comparator |
| DINOv2 + LoRA | Adjudicated fit and disjoint calibration cohorts | Resource-feasibility path only until real labels exist |

An anomaly score is not a fold or crack label. It must be calibrated against
expert-reviewed positives and difficult normal anatomy.

## Real-data ingestion

The adapters accept NumPy arrays directly and load `.npy`, `.npz`, and common
image formats. `tifffile` is an optional dependency for richer TIFF/OME-TIFF
support:

```bash
python3 -m pip install -e '.[wsi]'
```

For proprietary COMET and CosMx files, extract channel identities from OME or
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
DINOv2 is treated as H&E/RGB-only unless a governed semantic projection for
multiplex channels is defined; the encoder rejects silently taking the first
three COMET/CosMx channels. Pathology
foundation weights such as UNI, Virchow2, Cell-DINO, and GrandQC may carry
noncommercial or research-only restrictions; permissive wrapper code does not
override model-weight terms.

## Evaluation contract

- [`docs/TEAM_BRIEF.md`](docs/TEAM_BRIEF.md): decision-ready recommendation,
  evidence plan, illustrative gates, and immediate team asks
- [`docs/AI-SPEC.md`](docs/AI-SPEC.md): architecture, risks, model ladder, and
  guardrails
- [`docs/FOUNDATION_FEASIBILITY.md`](docs/FOUNDATION_FEASIBILITY.md): pinned
  DINOv2/MPS/LoRA execution result and its evidence boundary
- [`docs/PUBLIC_BENCHMARK_AUDIT.md`](docs/PUBLIC_BENCHMARK_AUDIT.md): verified
  public methods, real datasets, label coverage, and license/access gaps
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
