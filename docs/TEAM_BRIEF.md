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

For H&E, retain DINOv2-small as the generic RGB control and Hibou-B as the
current pathology-specific, commercially permissive comparator. Classical,
DINOv2-small, SigLIP2 Base, and Hibou-B have all completed the same hardened
v1.2 public H&E benchmark. Hibou-B's frozen linear head has the strongest point
estimate—positive-field macro Dice 0.667 (95% CI 0.603–0.730). The completed
exploratory paired analysis estimates Hibou-B linear minus DINOv2 linear as
+0.067974 [+0.053130, +0.084232]. It is a descriptive source-slide bootstrap
interval without p-values or multiplicity control, so it does not support a
statistical-superiority claim.

[DINOv3](https://github.com/facebookresearch/dinov3) is newer and technically
relevant, but checkpoint access requires accepting a custom
[license whose licensee definition can include an employer/entity](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md);
it was not run without institutional approval. COMET/CosMx require a separate
marker-aware evidence path. [KRONOS2](https://huggingface.co/MahmoodLab/KRONOS2)
is the most aligned public candidate found, but its gated CC BY-NC-ND terms
block ordinary corporate testing/adaptation without written approval.

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
- DINOv2-small, SigLIP2 Base, and the official Apache-2.0 Hibou-B pathology
  encoder have completed full, offline Apple-MPS runs on the same real public
  H&E split. All four current run artifacts, including classical, use the
  hardened v1.2 provenance contract and are report-eligible for this cohort.
- The current [real public H&E benchmark](REAL_PUBLIC_BENCHMARK.md) uses all
  2,127 released fields, disjoint supplied source-slide groups, full decode and
  mask QA, a locked 424-field test, and 1,000 cluster-bootstrap resamples.
- A hash-locked, ungated Apache-2.0
  [SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224)
  engineering smoke passed on Apple MPS, including CPU/MPS agreement and a
  rank-4 LoRA update. Separately, its
  [hardened real H&E artifact](../artifacts/public_fold/siglip2_hardened_v1_2.json)
  is report-eligible for the bounded public fold cohort; it does not cover crack,
  human clinical WSI, COMET, or CosMx efficacy.
- A [real COMET/CosMx proxy benchmark](MULTIPLEX_REAL_PROXY_BENCHMARK.md) now
  executes five public COMET DAPI fields and six five-channel CosMx morphology
  FOVs from four slide/run groups. It measures controlled perturbation recovery
  and untouched alert burden, not natural-artifact accuracy.
- A machine-readable real-benchmark contract now blocks scientific reporting
  when realized annotated cohorts, method assets, license approval, or valid
  comparisons are absent.
- The latest verified full smoke run used 36 unique synthetic images and produced
  252 prediction/evaluation rows across 21 comparable groups in 27.34 seconds.
- The current automated suite and engineering gates pass; exact counts are
  intentionally left to the test runner rather than frozen in this brief.

This establishes software compatibility across all three modality contracts,
bounded current real-data fold-localization evidence for one external H&E
teaching-slide cohort, and label-free proxy evidence on real COMET/CosMx
backgrounds. It does **not** establish performance on Merck human WSI,
naturally occurring crack/tear, COMET or CosMx artifacts, scanners/sites, or
workflow benefit. Synthetic rankings remain invalid because the generator and
rules can share visual cues.

## Current real public H&E fold result

The locked test contains 179 clean and 245 fold-positive fields from 55 supplied
source-slide groups. Metrics below are point estimates; Dice intervals are 95%
source-slide-cluster bootstrap intervals.

All seven rows below use current schema-v1.2 artifacts with validated release,
code/environment, configuration, model/weight, and per-field outcome provenance.

| Method | Supervision | Positive-field macro Dice | All-field micro Dice | Presence AUROC | Clean FP area |
|---|---|---:|---:|---:|---:|
| Classical fold candidate | No learned representation; calibration labels | 0.446 (0.375–0.521) | 0.383 (0.318–0.446) | 0.792 | 10.73% |
| DINOv2-small PatchKNN | Clean-token bank; calibration labels | 0.341 (0.285–0.389) | 0.434 (0.352–0.497) | 0.884 | 1.79% |
| DINOv2-small linear probe | Fit masks plus calibration labels | 0.599 (0.532–0.668) | 0.770 (0.724–0.802) | 0.980 | 0.098% |
| SigLIP2 Base PatchKNN | Clean-token bank; calibration labels | 0.208 (0.178–0.236) | 0.224 (0.177–0.257) | 0.803 | 7.28% |
| SigLIP2 Base linear probe | Fit masks plus calibration labels | 0.526 (0.462–0.590) | 0.679 (0.625–0.714) | 0.965 | 0.299% |
| Hibou-B PatchKNN | Clean-token bank; calibration labels | 0.319 (0.264–0.368) | 0.418 (0.362–0.457) | 0.947 | 1.22% |
| Hibou-B linear probe | Fit masks plus calibration labels | 0.667 (0.603–0.730) | 0.827 (0.791–0.852) | 0.985 | 0.047% |

The deterministic paired artifact reports the following exploratory
positive-field macro-Dice differences: Hibou-B linear minus DINOv2 linear
+0.067974 [+0.053130, +0.084232]; DINOv2 linear minus SigLIP2 linear +0.072716
[+0.061862, +0.083173]; SigLIP2 linear minus classical +0.080439 [+0.045191,
+0.115669]; and DINOv2 PatchKNN minus Hibou-B PatchKNN +0.021906 [-0.010370,
+0.054272]. These are descriptive, cohort-conditional intervals—not hypothesis
tests. The [paired artifact](../artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json)
has SHA-256
`9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2`;
an independent rerun was byte-identical.

The seven rows span three method families and are a feasibility benchmark, not a
comprehensive SOTA comparison. The publisher reports DeepLabV3-ResNet50 Dice
`0.7630 +/- 0.2425`, but that is a different split, preprocessing, and
positive-image aggregation, so it must not be ranked directly against this
table. See the [public resource audit](PUBLIC_BENCHMARK_AUDIT.md) and
[foundation-model decision](PATHOLOGY_FOUNDATION_MODEL_DECISION.md).

## Real COMET/CosMx proxy result

No public COMET or CosMx source audited here provides independently usable
natural fold/crack masks. To quantify something valid now, each LOGO fold uses
source-group-disjoint fit/calibration/test roles, inserts controlled fold/crack
perturbations into held-out real backgrounds, and compares paired incremental
responses. Groups are reused across folds, so the folds are dependent.

| Modality | Classical Dice | Anomaly Dice | Hybrid Dice | Untouched alert burden: classical / anomaly / hybrid |
|---|---:|---:|---:|---:|
| COMET | 0.534 | 0.100 | 0.475 | 1.34% / 1.15% / 1.36% |
| CosMx | 0.314 | 0.134 | 0.305 | 3.04% / 14.33% / 19.46% |

These final v3 values are derived averages of separate fold/crack group-macro
point estimates after every provisional source group was held out once; the
derived averages have no direct CI. Higher-level biological independence is not
declared, so raw group-bootstrap intervals are descriptive. The generic anomaly
branch's controlled thin-crack Dice was 0.065 on COMET and 0.075 on CosMx.
Coverage, finite-score, paired-subtraction, determinism, and flip/inverse checks
passed, so this is a scientific failure of the chosen representation/scale on
this proxy—not evidence of a zero-map coding bug. Dice changed by up to 0.332 in
the 256-pixel sensitivity run and COMET crack response was nonmonotonic with
severity. The proxy artifact is intentionally `report_eligible=false`; alert
burden is not false-positive rate. Natural-artifact Dice, ROC, sensitivity,
specificity, and FPR remain unavailable. The
[primary v3 artifact](../artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json)
has SHA-256
`a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`.

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

Select at least 30 independent patient/slide/run groups per modality spanning
clean images, target artifacts, and hard negatives. Enrich the pilot to contain
at least 50 fold-positive, 50 crack-positive, and 100 clean/hard-negative
regions. Two trained reviewers label independently; adjudicate all target
disagreements. Measure reviewer agreement, ambiguity, annotation time,
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
2. Export a de-identified, checksum-locked pilot with at least 30 independent
   groups per modality using
   `configs/internal_manifest.example.json`.
3. Complete the reviewer calibration set and measure inter-reviewer agreement.
4. Run the existing baselines unchanged, build a failure-mode gallery, and size
   the locked cohorts from observed prevalence and uncertainty.
5. Run the frozen DINOv2-small, Hibou-B, and SigLIP2 heads unchanged on the
   adjudicated H&E pilot. Use the locked failures to decide whether a small
   decoder or LoRA justifies added governance and annotation cost; do not
   fine-tune against the locked test.

## Go/no-go interpretation

The project is technically feasible, has a reproducible real-background
multiplex proxy and a current hardened public H&E fold benchmark, and is ready
for internal-data integration.
It is not ready for autonomous QC, tissue deletion, a crack claim, or a
three-modality efficacy claim. The highest-value next investment is a precise
ontology plus a small, high-quality adjudicated pilot—not extensive fine-tuning.
