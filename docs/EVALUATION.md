# Locked evaluation protocol

This document defines how fold/crack QC performance is measured and what may be
claimed from each evidence tier. It is normative for internal evaluation. Any
change to an ontology, cohort, mask policy, channel role, resolution, threshold,
metric, or gate after test access creates a new protocol version and requires a
new untouched test cohort.

## 1. Evidence tiers

| Tier | Permitted purpose | Prohibited claim |
|---|---|---|
| Unit tests | Verify metric arithmetic, geometry, serialization, and failure handling | Image-QC performance |
| Synthetic smoke test | Verify end-to-end modality adapters, detectors, masks, reports, and deterministic behavior | Performance or generalization on real H&E, COMET, CosMx, or Merck data |
| Public exploratory data | Compare implementation behavior and discover failure modes subject to license and domain mismatch | Merck performance, commercial clearance, or three-modality generalization |
| Internal development set | Fit clean references, train models, choose features, thresholds, channels, and gates | Unbiased final performance |
| Locked internal test | Estimate performance for pre-specified intended-use strata | Prospective operational benefit outside tested conditions |
| Prospective silent/live validation | Establish workflow impact, referral burden, drift, and human/model interaction | Autonomous tissue deletion unless separately authorized and validated |

Synthetic masks are exact because the generator created them; this makes them
excellent software tests and unrealistically clean reference standards. Synthetic
results must always be labeled `synthetic_engineering_smoke_test` in data and in
human-readable reports. They never count toward an acceptance gate.

Synthetic method rankings are also invalid as scientific comparisons: the image
generator and hand-designed detector can share the same intensity, ridge, texture,
or geometry cues. A method that wins that closed loop may simply match generator
assumptions. Synthetic comparisons may identify broken or insensitive code paths,
but cannot select the best model for real data.

## 2. Evaluation questions

The protocol answers five separate questions:

1. Does the system detect consequential artifacts without silently auto-passing them?
2. Are localized masks accurate enough for review or downstream exclusion?
3. Does it preserve valid tissue and avoid biased removal of difficult biology?
4. Does performance hold within every intended modality, panel, scanner, site, and time stratum?
5. Does using the output improve or preserve the downstream scientific result and workflow?

No single Dice score answers all five.

## 3. Units, identifiers, and independence

The **evaluation sample** is the smallest independently scored unit on which an
operator could act: normally a WSI, COMET image/region, or CosMx FOV. Tiles from
one sample are not independent observations. Store the following hierarchy:

`patient -> specimen/block -> slide -> run -> image/FOV -> tile`

For split assignment, use the highest unit that could leak appearance, preparation,
or batch information. A patient, block, slide, run, or adjacent section cannot
cross development and test. Repeated scans belong to one split. For uncertainty,
bootstrap the independent sample; if several samples share a higher correlated
unit, resample that higher unit and include all its samples.

Every row in the manifest includes:

- opaque case, block, slide/FOV, run, site, and time identifiers;
- modality, tissue/organ, preparation, stain/panel, species if applicable;
- scanner/instrument, objective/resolution, pixel spacing, software version;
- channel names, semantic roles, missing/extra channels, cycle/z information;
- artifact subtype, severity, annotators, adjudication status, and valid/ignore area;
- cohort role and reason for inclusion;
- source and license/provenance approval status.

Direct identifiers must not enter the repository or reports.

## 4. Reference-set design

### 4.1 Taxonomy pilot

Begin with approximately 30–50 cases per modality selected to span positives,
clean images, and difficult lookalikes. Two qualified reviewers annotate them
independently using [`ANNOTATION_GUIDE.md`](ANNOTATION_GUIDE.md), then an adjudicator
resolves disagreements. The pilot estimates prevalence, subtype ambiguity,
inter-reviewer agreement, annotation time, and whether a pixel mask is actionable.
It is development data, never the locked test.

### 4.2 Development cohort

Use development data for every adaptive choice:

- channel-role mapping and preprocessing;
- patch scale, features, model parameters, and clean-reference bank;
- artifact unions, size filters, and post-processing;
- boundary/centerline tolerance and component IoU threshold;
- `PASS`, `REVIEW`, `FAIL` rules and score thresholds;
- severity definition, acceptance margins, and reporting layout.

Cross-validation folds remain grouped by the independence hierarchy. Record all
experiments; the selected configuration is frozen before test access.

### 4.3 Locked test cohorts

Maintain distinct cohorts because one mixture cannot estimate both operational
burden and rare severe-artifact sensitivity:

1. **Prevalence cohort:** consecutive or randomly sampled production-like images.
   Estimates NPV, specificity, review rate, burden distribution, and workload.
2. **Enriched challenge cohort:** independently sourced fold/tear/glass-crack
   positives across severity and difficult hard negatives. Estimates sensitivity
   and localization with adequate precision.
3. **External/generalization cohort:** held-out site, device, tissue, panel, batch,
   time period, or software revision not used for development.
4. **Missing/degraded-input cohort:** missing optional channels, failed cycles,
   altered z coverage, compression, registration error, and metadata defects.
5. **Downstream-impact cohort:** images with the assay outputs needed for the
   noninferiority analysis.

Case selection and enrichment weights must be retained. Predictive values are
reported on the prevalence cohort or prevalence-adjusted transparently; they are
not taken directly from an enriched case-control mixture.

### 4.4 Lock procedure

Before opening predictions on the test set, an independent owner records and
checksums:

- cohort manifest and inclusion/exclusion rules;
- reference masks, adjudication log, and valid/ignore masks;
- preprocessing, channel-role files, resolution, and coordinate transforms;
- model/weights, source revision, environment, and seed;
- per-modality target definitions and thresholds;
- metric version, tolerance in micrometres, empty-mask policy, and aggregation;
- confidence interval method, number of resamples, multiplicity plan, and gates;
- planned strata, minimum stratum sizes, and downstream margins.

The test runner receives read-only references. Failed parsing or inference is an
abstention/failure, not a dropped case. Exclusions after lock require a documented,
blinded reason that does not use model output.

## 5. Prediction/reference alignment

Evaluation occurs at the reference coordinate system and physical resolution.
The evaluator does not register or resize masks. The caller must:

1. resolve the source pyramid level and pixel spacing;
2. apply the frozen transform with nearest-neighbor interpolation for labels;
3. verify dimensions, orientation, origin, and a set of landmark checks;
4. intersect predictions and references with the locked `valid_mask`;
5. retain a transform audit record and overlay.

Artifact burden is calculated within reviewable tissue/assay area, not the full
rectangular canvas. Ignore regions are excluded from all counts. Boundaries created
only by an ignore-region cut are not scored.

## 6. Metrics

Sections 6.1–6.5 are **localization metrics**: they assess the geometry of a
candidate mask. Section 6.6 is a separate **operational slide/FOV decision
evaluation**. A high Dice, surface Dice, or centerline F1 does not establish a safe
auto-pass workflow. Operational acceptance is governed by severe-artifact
sensitivity lower confidence bound, auto-pass NPV, referral rate, valid-tissue
overmasking, technical abstention/failure, and downstream noninferiority.

### 6.1 Pixel metrics

From valid-region counts `TP`, `FP`, `FN`, and `TN`, report:

- sensitivity/recall `TP / (TP + FN)`;
- precision/PPV `TP / (TP + FP)`;
- specificity `TN / (TN + FP)`;
- Dice `2TP / (2TP + FP + FN)`;
- IoU `TP / (TP + FP + FN)`;
- false-positive and false-negative rates, balanced accuracy, and MCC.

Report per sample and by pooled slide/modality counts. Because background can
dominate, accuracy and specificity never replace positive-class measures.

Empty-mask policy is locked: when reference and prediction are both empty, the
sample is a correct negative and overlap is 1.0. With one-sided emptiness, overlap
is 0.0. Always report clean-sample false-positive burden separately so empty-case
conventions cannot hide alarms.

### 6.2 Surface and boundary metrics

Pixel overlap over-penalizes a narrow mask shifted by one or two pixels. Report:

- surface Dice at tolerance `tau`;
- tolerant boundary precision, recall, and F1;
- average symmetric surface distance (ASSD);
- 95th-percentile Hausdorff distance (HD95).

`tau` is selected on the annotation pilot from inter-reviewer edge uncertainty,
expressed in micrometres, and frozen separately if acquisition resolution differs.
Never use an undocumented pixel tolerance across modalities.

### 6.3 Thin crack/tear metrics

For tissue tears, glass cracks, and other line-like targets, report:

- tolerant centerline precision, recall, and F1 after deterministic thinning;
- clDice topology precision/sensitivity;
- instance sensitivity and false-positive components per sample or slide.

Centerline F1 is primary for thin targets when width is annotation-dependent.
clDice supplements it by testing whether each predicted/reference centerline lies
inside the opposite full mask. Neither replaces expert review of connectivity and
remediation subtype.

### 6.4 Connected components and FROC

Connected components are formed with the locked connectivity and minimum physical
area/length. One-to-one matching maximizes the number of target/prediction pairs
above a locked IoU threshold, then total IoU. Report:

- true-, false-positive, and false-negative instance counts;
- instance precision, sensitivity, F1, and mean matched IoU;
- false-positive instances per independent sample and per slide;
- FROC sensitivity versus false positives per sample/slide across score thresholds.

If a long reference artifact is fragmented into many predictions, one component
can match and the rest remain false positives. If multiple reference artifacts are
merged, only one can match. Add prespecified merge/split summaries when these
errors change reviewer burden.

### 6.5 Artifact burden and tissue preservation

Within the valid region, report true and predicted positive fraction, signed and
absolute fraction error, physical area error, and sample-wise MAE. Also report:

- valid-tissue overmask rate: artifact-predicted pixels adjudicated as valid tissue;
- severe-artifact undermask rate;
- fraction of tissue/cells/FOVs removed at each operating point;
- error in `PASS`, `REVIEW`, `FAIL` burden category.

Relative error is undefined when true burden is zero and prediction is nonzero;
use absolute error and false-positive burden instead of inventing a finite ratio.

### 6.6 Sample/slide decision metrics

For the locked action thresholds, report a confusion matrix for artifact presence
and for `PASS`, `REVIEW`, `FAIL`. Primary operational measures are:

- severe-artifact sensitivity with a confidence interval;
- auto-pass NPV on the prevalence cohort;
- specificity and PPV;
- review referral rate and alerts per slide/FOV;
- auto-fail rate, reviewer override rate, and rescan/recut yield;
- abstention and technical failure rates.

Treat `REVIEW` separately rather than silently calling it positive or negative.
Provide both conservative analyses: count review as failure for auto-pass safety,
and show human-resolved performance for the intended human-in-the-loop workflow.

### 6.7 Runtime and reliability

Report total, mean, median, p95, min, and max seconds per sample; tissue area or
megapixels processed per second where available; peak memory; parser/model failure;
and abstention rate. Include cold start and representative I/O. Runtime on cropped
synthetic arrays is not a WSI throughput claim.

## 7. Aggregation and uncertainty

- Retain all per-sample results.
- Recompute pixel and instance rates from pooled counts for a slide or modality.
- Use sample-macro means for boundary and centerline metrics; disclose excluded
  undefined distances.
- Report median and distribution plots in addition to means for skewed burden.
- Generate percentile bootstrap intervals using a fixed seed and at least the
  pre-specified number of resamples.
- Resample by independent sample; cluster by patient/block/slide/run when needed.
- Stratify before pooling. Do not let numerous H&E images dominate COMET or CosMx.

If many gates/strata are tested, pre-specify a hierarchical gate or multiplicity
control. “No significant difference” is not evidence of noninferiority; use an
approved margin and confidence-bound test.

## 8. Acceptance gates

The numerical values in `configs/acceptance.example.json` are illustrative and
not approved requirements. The decision owners must sign the final values before
test access. A reasonable gate structure is:

| Gate | Cohort | Pass rule |
|---|---|---|
| Severe-artifact safety | Enriched challenge | Lower CI bound for sensitivity meets approved minimum |
| Auto-pass safety | Prevalence | NPV meets approved minimum and no severe subgroup miss pattern |
| High-confidence localization | Enriched challenge | Lower CI bound for mask precision meets approved minimum |
| Tissue preservation | Prevalence/hard negative | Valid-tissue overmask rate below approved maximum |
| Workload | Prevalence | Review rate and FP instances/slide below approved maxima |
| Generalization | External matrix | Every mandatory modality/stratum meets its floor |
| Reliability | All | Failure/abstention behavior is safe and rates are acceptable |
| Downstream utility | Impact cohort | All primary endpoints meet noninferiority margins |

The example values—0.95 severe-artifact sensitivity lower bound, 0.99 auto-pass
NPV, 0.95 high-confidence precision lower bound, 0.005 valid-tissue overmask,
and 0.25 review rate—are hypotheses for stakeholder discussion, not evidence-based
acceptance criteria.

A gate is evaluated separately for H&E, COMET, and CosMx. A pooled pass cannot
offset a modality failure. Small strata that cannot produce a useful interval are
reported as insufficient evidence, not passed.

## 9. Generalization matrix

Populate each planned cell with sample counts, positive counts, severity, metric
intervals, and pass/fail/insufficient-evidence status.

| Axis | H&E | COMET | CosMx |
|---|---|---|---|
| Tissue/anatomy | Organs, biopsy/resection, frozen/permanent, dense/necrotic/adipose | Organs and tissue states represented in intended panels | Organs, morphology patterns, low/high cell density |
| Preparation | Fixation, section thickness, stain batch, coverslip/mount | Fixation, retrieval, staining run/cycle | Fixation, slide prep, morphology stain run |
| Instrument | Intended scanner vendors/models and focus modes | Intended COMET instruments, objectives, run settings | Intended CosMx instruments, objectives, z strategy |
| Site/operator | Internal/external labs and histotechnologists | Sites, operators, reagent lots | Sites, operators, reagent lots |
| Panel/channel | Native RGB and special-stain exclusion | Panel versions, DAPI/AF availability, marker count | Panel versions, morphology-channel identities/order |
| Batch/time | Historical and prospective temporal holdout | Run/batch/lot and temporal holdout | Run/batch/lot/software and temporal holdout |
| Missing/degraded input | Compression, focus, partial scan, metadata errors | Missing channel/cycle, saturation, registration error | Missing morphology channel/z, low signal, registration error |
| Target subtype | Fold, tissue tear, glass crack, knife-line confounder | Same plus cycle/registration and antibody-aggregate confounders | Same plus tile/FOV seam and morphology dropout confounders |
| Hard negatives | Lumen, vessel, edge, cautery, necrosis, dense tumor, mucus | Bright biology, AF, marker aggregates, tissue edges | Membranes, lumina, sparse tissue, bright cells, FOV edges |

Cross-modality zero-shot transfer is an exploratory cell, not an intended-use
claim. A unified model must be compared with three separately calibrated baselines
and pass every modality gate.

## 10. Downstream-impact evaluation

Localization is useful only if its action improves or preserves the scientific
pipeline. On a locked cohort, compare at least:

1. no mask;
2. expert reference mask;
3. predicted high-confidence mask;
4. predicted mask plus human review/override;
5. where feasible, repeat scan/recut/remount reference.

Pre-specify endpoints relevant to each modality:

| Modality | Example downstream endpoints |
|---|---|
| H&E | Tissue retained, nuclei/cell segmentation, morphology features, tumor/region model outputs, pathologist review time |
| COMET | Cell count, segmentation QC, phenotype proportions, marker intensity/distribution, neighborhood/spatial statistics, run acceptance |
| CosMx | Cell segmentation, transcripts/cell, negative-probe background, detection efficiency, cell-type proportions, spatial neighborhoods, FOV retention |

Use paired analyses at the independent unit and pre-approved noninferiority margins.
Report differential tissue/cell loss by anatomy and phenotype; an artifact mask can
produce deceptively stable global averages while selectively deleting a biologic
subpopulation. Diagnostic or biomarker endpoints require their own governed study.

## 11. Reproducible reports

The public evaluation API is:

- `pixel_metrics`, `boundary_metrics`, `centerline_metrics`;
- `instance_metrics`, `froc_counts`, `burden_metrics`;
- `evaluate_sample`, `aggregate_results`, `aggregate_by_slide`;
- `bootstrap_ci_by_sample`, `bootstrap_ci_by_cluster`, `runtime_summary`;
- `build_report`, `write_json_report`, `write_csv_report`,
  `report_to_markdown`, and `write_markdown_report`.

`build_report` defaults to slide-cluster resampling. For repeated slides or FOVs,
pass the locked highest correlated identifier (for example,
`metadata.patient_id`) through `bootstrap_cluster_key`. The operational gate API
is `foldcrack_qc.operational.evaluate_operational_decisions`; it is intentionally
separate from localization reports and rejects synthetic evidence as acceptance
eligible.

JSON is the canonical machine-readable record; CSV contains flattened per-sample
rows; Markdown is a review convenience. Reports include code/config/model/input
checksums through the surrounding run manifest. Reports never omit failures and
never mix synthetic development output with locked validation output.

## 12. Public data and licensing

Public data may be used only after privacy, security, legal, and model-governance
review of the exact version:

- GrandQC provides strong H&E fold masks but carries non-commercial restrictions.
- QUALIFAI is directly relevant to COMET folds but not CosMx and has no dedicated
  crack class; confirm code, data, and model terms individually.
- HistoArtifacts and the Foucart `Tear&Fold` data have unclear or incomplete
  reuse terms/annotations and cannot be assumed corporate-ready.
- No verified public set establishes coverslip/glass crack performance or a
  common fold/crack target across all three modalities.

Public benchmarks supplement, but never replace, the locked Merck intended-use
reference set.
