# Fold/crack QC: feasibility decision brief

## Recommendation

Proceed with a **shared QC platform and modality-specific evidence paths**, not a
single blindly shared model. Reuse the ingestion contract, physical-coordinate
handling, outputs, audit trail, and evaluator across H&E, COMET, and CosMx. Keep
channel construction, clean-reference banks, calibration, thresholds, and final
acceptance separate by modality until a unified encoder proves noninferior in
every locked stratum.

Start with an interpretable classical detector plus a clean-reference anomaly
branch, fused into a conservative `PASS / REVIEW / FAIL` workflow. Treat a frozen
foundation encoder as a controlled comparator after license/security review—not
as a prerequisite and not as a substitute for reference labels.

## What is demonstrated now

- A CPU-only implementation runs end to end on all three modality contracts.
- Classical, clean-reference anomaly, hybrid, and structural-channel ablation
  paths are compared under identical synthetic scenarios.
- Physical-scale geometry, semantic channel roles, missing-channel abstention,
  WSI tile halos/stitching, localization metrics, cluster bootstrap reports,
  strict manifests, split-leakage checks, and operational gates are executable.
- A pinned DINOv2-small model has now run offline on Apple MPS: frozen global
  and spatial features closely matched CPU, and a BF16 rank-4 LoRA update
  completed within local memory. This is hardware feasibility, not efficacy.
- A machine-readable real-benchmark contract now blocks scientific reporting
  when realized annotated cohorts, method assets, license approval, or valid
  comparisons are absent.
- The latest verified full smoke run used 36 unique synthetic images and produced
  252 prediction/evaluation rows across 21 comparable groups in 27.34 seconds.
- All 12 engineering gates and all 158 automated tests plus 7 subtests passed.

This establishes **software feasibility only**. It does not establish sensitivity,
specificity, generalization, or workflow benefit on Merck data. Synthetic method
rankings are invalid because the generator and rules can share visual cues.

## Decisions needed from the team

1. What does “crack” mean: tissue tear, glass/coverslip crack, knife line,
   acquisition seam, or separately stored subtypes?
2. What action follows each finding: rescan, recut/restain/remount, regional mask,
   review, or fail?
3. What is the independently actionable unit for each modality: WSI, region, or
   CosMx FOV?
4. Which structural channels are reliably available for each COMET panel and
   CosMx assay, and what missing input must force review?
5. Who owns pathology/assay adjudication, operational thresholds, statistics,
   downstream noninferiority, privacy/security, and model/data licensing?

## Proposed evidence plan

### Stage 1 — taxonomy pilot

Select roughly 30–50 cases per modality spanning clean images, target artifacts,
and hard negatives. Two trained reviewers label independently; adjudicate all
target disagreements. Measure reviewer agreement, ambiguity, annotation time,
prevalence, severity distribution, and whether pixel masks change an action.

### Stage 2 — development and calibration

Split at patient/block/slide/run, never by tile. Use development data for channel
roles, features, post-processing, and model fitting. Use a disjoint clean
calibration subset for anomaly scaling and decision thresholds. Compare classical,
anomaly, hybrid, and—if approved—frozen feature models. Freeze the ontology,
preprocessing, model, thresholds, metric tolerances, and decision rules before
opening the test set.

### Stage 3 — locked test

Keep separate cohorts for:

- production-prevalence workload and predictive values;
- enriched severe positives and hard negatives;
- unseen site/device/panel/tissue/time generalization;
- missing/degraded channels and metadata;
- downstream assay noninferiority.

Report H&E, COMET, and CosMx separately. A pooled average cannot compensate for a
failed modality or required stratum.

## Illustrative acceptance criteria for stakeholder approval

| Criterion | Proposed gate | Evidence source |
|---|---:|---|
| Severe actionable artifact sensitivity | 95% CI lower bound >= 0.95 | Enriched challenge cohort |
| Safety of automatic pass | NPV >= 0.99 | Production-prevalence cohort |
| High-confidence localization precision | 95% CI lower bound >= 0.95 | Adjudicated mask subset |
| Valid-tissue overmask | Mean <= 0.5% | Prevalence and hard negatives |
| Review referral rate | <= 25% | Production-prevalence cohort |
| Missing/invalid/OOD input | 100% routes to `REVIEW` | Degraded-input cohort |
| Downstream effect | Meets approved noninferiority margin | Paired impact cohort |

These are starting hypotheses, not approved Merck requirements. Minimum evidence
counts and confidence-interval methods must be power-checked against the expected
prevalence and risk tolerance. A severe miss or technical failure must never be
silently converted into `PASS`.

## Next two weeks

1. Hold a 60-minute ontology/action workshop with pathology, COMET/CosMx assay,
   operations, and downstream analytics owners.
2. Export a de-identified, checksum-locked 30–50 case pilot per modality using
   `configs/internal_manifest.example.json`.
3. Complete the reviewer calibration set and measure inter-reviewer agreement.
4. Run the existing baselines unchanged, build a failure-mode gallery, and size
   the locked cohorts from observed prevalence and uncertainty.
5. Run the implemented frozen DINOv2 anomaly comparator on the adjudicated H&E
   pilot, then decide from the locked development evidence whether LoRA or a
   supervised localization head justifies added governance and annotation cost.

## Go/no-go interpretation

The project is technically feasible and ready for internal-data integration.
It is not ready for autonomous QC, tissue deletion, or a three-modality efficacy
claim. The highest-value next investment is a precise ontology plus a small,
high-quality adjudicated pilot—not extensive fine-tuning.
