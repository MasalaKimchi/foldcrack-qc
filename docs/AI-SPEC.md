# AI design contract: fold and crack artifact QC

Status: feasibility contract, not a validated product claim  
System type: medical-image quality-control localization and triage  
Modalities: H&E whole-slide imaging, Lunaphore COMET, Bruker CosMx morphology imaging  
Primary implementation: dependency-light Python with NumPy, SciPy, and OpenCV where available

## 1. Intended use and decision

The system highlights image regions that may contain tissue folds or a precisely
defined crack/tear artifact, estimates their burden, and assigns a conservative
`PASS`, `REVIEW`, or `FAIL` recommendation. It is decision support for image-QC
operators and downstream assay teams. It is not a diagnostic device and must not
make claims about disease, treatment, biomarker status, or specimen adequacy.

The first operational question is the action attached to an alert:

- rescan when the source glass slide is intact but acquisition failed;
- restain, recut, or remount when the physical section or coverslip is affected;
- mask only the affected region when downstream analysis remains valid;
- send to a trained reviewer when the artifact type or consequence is uncertain.

Those actions are not interchangeable. Before efficacy work begins, stakeholders
must decide whether “crack” means a tissue tear, coverslip/glass crack, knife line,
acquisition seam, or a union of separately annotated subtypes. The label must not
be inferred from visual appearance alone when the remediation differs.

### Out of scope for the first release

- autonomous deletion of tissue, fields of view, cells, or transcripts;
- one threshold claimed to generalize across H&E, COMET, and CosMx;
- diagnosis or assessment of tissue biology;
- artifact restoration or generation of a diagnostically “corrected” image;
- using transcript counts, cell type, or downstream labels as required inputs;
- a validation claim based only on synthetic or public data.

## 2. Domain ontology and output contract

The reference ontology is defined operationally in
[`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md). Its top-level labels are:

| Label | Meaning | Phase-1 treatment |
|---|---|---|
| `fold` | Tissue overlaps, doubles, wrinkles, or rolls over itself | Primary target |
| `tissue_tear` | Physical discontinuity or split in the tissue section | Primary target when “crack” means tear |
| `glass_crack` | Crack/crazing in coverslip or slide glass affecting visible tissue | Separate target and remediation |
| `knife_line` | Microtomy chatter, score, or repeated cutting line | Confounder or separate target |
| `acquisition_seam` | Stitching, registration, scan-line, cycle, or tile-boundary defect | Confounder or separate target |
| `other_artifact` | Bubble, foreign material, saturation, OOF, aggregate, edge defect | Secondary class or confounder |
| `hard_negative` | Normal structures likely to resemble a target | Explicit negative stratum |
| `uncertain` / `ignore` | Not reliably judgeable or outside reference scope | Excluded with `valid_mask` |

The API may expose a phase-1 union named `crack`, but the stored annotation must
retain its physical subtype. A union is derived only after the target definition
and downstream action are locked.

Every inference result should retain:

- sample, slide/FOV, case, block, run, site, scanner/instrument, panel, and time identifiers;
- modality, native pixel spacing, pyramid level, transform, and semantic channel roles;
- continuous score maps and locked binary masks per target;
- artifact burden inside evaluable tissue;
- connected components with size, confidence, and bounding geometry;
- `PASS`, `REVIEW`, or `FAIL`, plus the rule that produced it;
- explicit abstention reason for missing channels, corrupt metadata, OOD input, or runtime failure;
- model, code, configuration, threshold, and reference-manifest versions.

## 3. Modality and channel contract

One software contract may serve all three modalities, but normalization,
reference banks, calibration, thresholds, and acceptance results remain separate.
Channel positions are never hard-coded; instrument metadata is mapped to semantic
roles and saved in the run manifest.

| Modality | Required structural view | Useful derived/optional views | Channels not required for core QC |
|---|---|---|---|
| H&E | Native RGB at a known physical resolution | Hematoxylin/eosin optical density, total OD, saturation, local texture | Aggressively normalized RGB that removes QC evidence |
| COMET | DAPI/nuclear plus available autofluorescence or broad structural channels | Cycle-to-cycle DAPI stability, registration residual, saturation, per-channel aggregate evidence | Biological marker identity; marker channels may be optional evidence only |
| CosMx | Nuclear and broad membrane/tissue morphology; union or z-spread where available | Per-cycle/z consistency, focus, registration, morphology-channel agreement | Transcript density, vendor cell masks, cell type, outcome labels |

For COMET and CosMx, a minimal nuclear-only view is a required ablation, not an
assumed final choice. Some folds are visible in DAPI, while tissue tears,
autofluorescent debris, saturation, or z-dependent defects may require additional
structural channels. Missing required roles force `REVIEW`; missing optional roles
must be recorded and evaluated as a generalization stratum.

QC should operate on minimally transformed data. H&E stain normalization or
fluorescence per-channel scaling may erase saturation, intensity doubling, or
background clues. Any normalization is fit on development data, versioned, and
applied identically to the locked test set.

## 4. Model ladder

The project uses a staged comparison rather than assuming one advanced model is
best before reference data exist.

1. **Interpretable classical baseline.** Multiscale intensity, optical density,
   gradient, texture, line/ridge, saturation, and morphological features produce
   candidate masks. This is CPU-capable, debuggable, and establishes a lower bound.
2. **Clean-reference anomaly baseline.** Robust feature distance or one-class
   scoring is fit only on reviewed clean development tiles, separately by modality.
   A disjoint clean calibration subset sets score scaling and thresholds; the
   model-fitting bank must not also serve as its own performance test. Anomaly is
   evidence, not a semantic fold/crack label.
3. **Hybrid review model.** Fuse physical candidates and calibrated anomaly score.
   This is the preferred phase-1 high-recall review aid because each branch covers
   different failure modes and remains inspectable.
4. **Frozen feature comparator.** After security and license review, extract frozen
   embeddings and train only a small linear, nearest-neighbor, or one-class head.
   Compare against the lightweight baselines at equal locked data and thresholds.
5. **Supervised or low-rank adaptation.** Consider only after the ontology and an
   adequate adjudicated development set exist. Pixel segmentation is justified
   only if localization changes a downstream action.

Separate modality heads are the default. A shared encoder is accepted only if it
matches or exceeds the separate baselines in every pre-specified modality stratum
and does not conceal modality-specific failure. No pooled average may compensate
for a failed modality.

### Foundation-model governance

Frozen weights can reduce local training cost, but do not eliminate domain shift,
calibration, or licensing obligations. HistoART's permissive wrapper does not
override UNI's non-commercial/no-derivatives model terms. GrandQC code, weights,
and test data carry non-commercial restrictions. QUALIFAI and every individual
dataset/model version require provenance review. A public download is not legal
clearance for Merck use.

The run manifest must record repository, exact revision, weight checksum, model
card, license text/version, access date, intended corporate use, approval owner,
and any upstream-data restrictions. If approval is absent, the asset cannot enter
the corporate benchmark; use a permissive baseline or internally trained weights.

### Implemented foundation feasibility boundary

The repository now contains an offline-first frozen DINOv2 adapter, spatial-token
clean-reference kNN scorer, and an auditable CPU/MPS smoke command. The executed
reference was `facebook/dinov2-small` at immutable revision
`ed25f3a31f01632728cabb09d1542f84ab7b0056`; the cached
`model.safetensors` SHA-256 was
`ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`.
On Apple MPS, frozen global and 16-by-16 spatial embeddings were finite and
closely matched CPU, and one BF16 rank-4 LoRA step updated the query/value
projections in transformer blocks 8–11.

This establishes only that the pinned model and a small PEFT configuration fit
and execute on the available Mac. It does not establish artifact sensitivity,
localization, calibration, superiority, or modality generalization. Generic
clean-reference anomaly maps have only `artifact_union` semantics. Fold versus
crack claims require subtype labels and a semantic supervised head. COMET and
CosMx require governed channel projections or modality-native encoders; silently
taking the first three channels is prohibited. The complete measured execution
record and limitations are in [`FOUNDATION_FEASIBILITY.md`](FOUNDATION_FEASIBILITY.md).

## 5. Evaluation strategy

[`EVALUATION.md`](EVALUATION.md) is the normative protocol. The locked reference
set is independent of all threshold selection and contains prevalence, enriched
challenge, hard-negative, device/site, temporal, and new-panel cohorts. Splitting
occurs at the highest correlated unit—patient/block/slide/run—not by tile.

| Dimension | Primary measure | Why it matters |
|---|---|---|
| Artifact presence | Severe-artifact sensitivity and auto-pass NPV | Avoid silently passing consequential damage |
| Region overlap | Dice/IoU plus surface Dice at a physical tolerance | Quantify mask utility without over-penalizing uncertain edges |
| Thin structures | Tolerant centerline F1 and clDice | Avoid declaring a narrow tear missed because its width differs by a few pixels |
| Instances | Component sensitivity and FP per sample/slide; FROC | Reflect reviewer workload and multiple lesions |
| Burden | Absolute error in affected-tissue fraction/area | Supports pass/review/fail rules and downstream masking |
| Tissue preservation | Valid-tissue overmask rate | Prevent loss of useful tissue and biased biology |
| Generalization | Every metric by modality and locked stratum | Pooled performance cannot hide a failed panel or scanner |
| Downstream effect | Pre-specified noninferiority for assay outputs | A good-looking mask is insufficient if scientific results change |
| Operations | Runtime, failure, abstention, and referral rates | Determine whether 100% QC is practical |

Confidence intervals use sample-level bootstrap resampling; tiles are never
treated as independent. If several samples share a patient, block, slide, or run,
that highest correlated unit becomes the resampling cluster.

## 6. Acceptance gates and guardrails

Illustrative gates are stored in `configs/acceptance.example.json`; stakeholders
must approve values and decision consequences before the test set is opened.
At minimum, acceptance requires all of the following in every modality:

- lower confidence bound for severe-artifact sensitivity meets the approved gate;
- auto-pass NPV meets the approved gate at observed prevalence;
- high-confidence mask precision meets its gate;
- valid-tissue overmask and false-positive review burden stay below their gates;
- no pre-specified device/site/panel/tissue stratum fails its safety floor;
- missing required channels or metadata, OOD inputs, and pipeline errors abstain to `REVIEW`;
- downstream assays meet pre-specified noninferiority margins;
- prospective human-in-the-loop operation confirms acceptable referral and miss rates.

Thresholds are selected once on development data and locked by modality and
target. Post-test tuning invalidates the test and requires a new untouched cohort.

Online guardrails:

- validate dimensions, transforms, spacing, channel identities, and finite values;
- reject unsupported scale or silently resampled masks;
- never auto-pass after parser/model failure;
- retain overlays and score maps for reviewer audit;
- cap or flag artifact masks that would remove implausibly large tissue fractions;
- log model/config checksums and deterministic seeds;
- route discordant detector branches and OOD scores to `REVIEW`.

Offline monitoring:

- referral, override, repeat-scan/recut, and abstention rates by modality/site/device;
- score and burden drift by panel, batch, tissue, time, and software revision;
- sampled expert review of auto-pass and auto-fail decisions;
- emerging hard negatives and false-negative root-cause categories;
- downstream metric drift before and after masking.

## 7. Implementation contract

The evaluator in `foldcrack_qc.evaluation` exposes stable, array-based entry points:

```python
result = evaluate_sample(
    target_mask,
    score_map=artifact_score,
    threshold=locked_threshold,
    sample_id=sample_id,
    slide_id=slide_id,
    modality="comet",
    valid_mask=tissue_or_reviewable_region,
    spacing=(row_um, column_um),
    boundary_tolerance=approved_um,
    centerline_tolerance=approved_um,
    runtime_seconds=elapsed,
)

report = build_report(
    results,
    seed=locked_seed,
    bootstrap_cluster_key="metadata.patient_id",
)
write_json_report(report, output_dir / "evaluation.json")
write_csv_report(results, output_dir / "per_sample.csv")
write_markdown_report(report, output_dir / "evaluation.md")
```

Evaluation consumes final masks at reference resolution; it does not resize or
register them. Metric outputs are JSON-safe. Pixel and instance summary rates are
recomputed from pooled counts, while surface and centerline summaries are
sample-macro means. Empty target and prediction masks count as a correct negative;
one-sided emptiness receives zero overlap.

Every release bundles:

- immutable environment and input/reference manifests;
- per-sample JSON/CSV values, aggregate and modality/slide summaries;
- cluster-bootstrap intervals for the displayed aggregate and FROC operating points;
- reviewer overlays for all errors and a seeded audit sample of true negatives;
- a limitations statement distinguishing engineering, development, locked-test,
  and prospective evidence.

## 8. Critical failure modes

| Failure | Consequence | Required control |
|---|---|---|
| Fold confused with dense normal tissue or tumor | Tissue overmask and biased quantification | Hard-negative strata, precision gate, overlay review |
| Tear confused with vessel/lumen/section edge | Excess referrals or tissue loss | Subtype ontology, centerline metric, anatomical hard negatives |
| Glass crack confused with tissue tear | Wrong remediation | Separate labels and workflow action |
| Nuclear-only multiplex input misses structural damage | False pass | Full-structural comparison and missing-channel abstention |
| Rare normal anatomy appears anomalous | False alarms | Clean-reference diversity and explicit anomaly disclaimer |
| Test tiles leak from a training slide | Inflated performance | Split and bootstrap at highest correlated unit |
| A pooled result hides weak CosMx performance | Unsupported generalization | Per-modality gates; no pooled compensation |
| Non-commercial weights enter a Merck pipeline | Legal/compliance exposure | Asset inventory and approval before download/use |

## 9. Completion checklist

- [x] Intended and prohibited uses are explicit.
- [x] Fold, tissue tear, glass crack, and common lookalikes are separated.
- [x] Modality channel roles and missing-channel behavior are specified.
- [x] Classical, anomaly, hybrid, and frozen-feature comparators are defined.
- [x] A locked, leakage-resistant reference protocol is defined.
- [x] Pixel, boundary, centerline, instance, FROC, burden, runtime, and downstream metrics are defined.
- [x] Acceptance is per modality and based on confidence bounds.
- [x] Synthetic smoke tests are explicitly excluded from validation claims.
- [x] Synthetic method rankings are rejected because generators and rules may share cues.
- [x] Foundation-weight and dataset licensing require corporate review.
- [ ] Merck stakeholders have approved the exact “crack” target and remediation.
- [ ] Pathology, assay, operations, statistics, privacy, security, and legal owners are named.
- [ ] Acceptance thresholds and noninferiority margins are approved before test access.
