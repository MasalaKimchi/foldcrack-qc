# Real COMET/CosMx proxy benchmark

**Run date:** 2026-08-26
**Primary artifact:** [`real_public_logo_cv_896_v3.json`](../artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json), SHA-256 `a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`
**Resolution sensitivity:** [`real_public_logo_cv_256_v3.json`](../artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json), SHA-256 `a96edf1a02878d98d7ffd6f6d5a2acd3bc482a5a8a10956ebc7a18679579da5c`
**Evidence status:** label-free, generator-conditional proxy; **not natural-artifact efficacy**

## Answer first

The code now executes real, checksum-locked public COMET and CosMx images. The
final v3 run used five COMET DAPI fields and six five-channel CosMx morphology
FOVs. Every one of the five COMET field IDs and four CosMx slide/run IDs was a
held-out test group exactly once in modality-specific leave-one-group-out
cross-validation. Source IDs, canonical paths, and SHA-256 content digests were
all unique before splitting.

No audited public input supplies independently usable masks for naturally
occurring folds or cracks. Natural-artifact Dice, ROC, sensitivity, specificity,
and false-positive rate therefore remain **not estimable**. Activity on an
untouched field is called alert burden, not a false positive.

The controlled-perturbation result is mixed:

- Classical morphology was the strongest or tied response for most injected
  cues, but it is favored by a generator built from intensity, edge, and signal
  superposition/loss cues.
- Clean-reference PCA/Mahalanobis anomaly detection was a poor localizer. At
  896 px its Dice was 0.065/0.135 for COMET crack/fold and 0.075/0.193 for CosMx.
  Most of its incremental activation lay outside the realized perturbation.
- Hybrid fusion did not consistently improve classical morphology. It tied
  COMET crack Dice, reduced COMET fold Dice, slightly reduced CosMx crack Dice,
  and slightly reduced CosMx fold Dice.
- Results changed materially between 256 and 896 px. This prevents a robust
  method ranking and shows that resolution/physical-scale locking is essential.
- COMET classical and hybrid crack dose-response Spearman was `-1.0` at 896 px,
  while fold response was `+1.0`. The current synthetic crack family is not a
  validated severity scale and cannot support an operational claim.

The artifact intentionally records `report_eligible=false` and
`scientific_validation_passed=false`.

## Public inputs and provenance

| Cohort | Inputs | Provisional grouping unit | Native content | Natural artifact masks | Lock manifest |
|---|---:|---|---|---|---|
| [QUALIFAI COMET v2](https://zenodo.org/records/12699470) | 5 | 5 field IDs; higher-level slide/patient mapping unavailable | DAPI-only `uint16`; MPP absent | None recoverable | [`qualifai_comet_v2.json`](../configs/public_data/qualifai_comet_v2.json) |
| [CosMx gastric mucosa](https://zenodo.org/records/8333281) | 4 FOVs | 2 distinct slide/run IDs | 5-channel `uint16`, `BGYRU`, 0.18 µm/px | None | [`cosmx_gastric_v1.json`](../configs/public_data/cosmx_gastric_v1.json) |
| [CosMx pHGG](https://zenodo.org/records/16877090) | 2 FOVs | 2 distinct slide/run directories | 5-channel `uint16`, `BGYRU`, 0.120280945 µm/px | None | [`cosmx_phgg_v1.json`](../configs/public_data/cosmx_phgg_v1.json) |

The four CosMx identifiers are distinct locked source slide/run IDs, but the
public releases do not establish higher-level biological independence. COMET
higher-level independence is also unknown. Group-bootstrap intervals below are
therefore descriptive under provisional grouping—not population-level CIs.

Every local TIFF is hash locked. Extracted ZIP/ZIP64 members were checked against
their upstream CRCs. The QUALIFAI deposit has the exact published size and MD5,
but its gzip stream ends after ten image members and exposes no masks or split
manifest. The paper's reported fold result cannot be independently recomputed
from the public object.

## Evaluation design and audit corrections

| Component | Final v3 contract |
|---|---|
| Split | Modality-specific leave-one-source-group-out; one distinct calibration group; remaining groups fit; held-out rows only |
| Classical | Modality-aware intensity, edge, texture, and morphology scores |
| Anomaly | Robust scaling, PCA, shrinkage covariance, Mahalanobis patch distance; clean fit fields only |
| Hybrid | Fixed 0.55 classical + 0.45 anomaly; no test-selected weight |
| Perturbations | Curved local signal superposition (fold) and multichannel attenuation (crack), three severities |
| Paired score | `max(score(injected) - score(unmodified), 0)` |
| Metric support | Only pixels that actually changed after clipping and dtype casting; intended geometry retained for audit |
| Severity support | Fixed intended geometry at all severities; contextual score effects allowed and realized-support fraction reported |
| Downsampling | Bounded-memory exact area integration over every native pixel; no point decimation |
| Threshold | Calibration-only maximum Dice among a declared deterministic quantile candidate grid; not a global optimum claim |
| Unmodified fields | Alert burden and transform consistency; never false-positive rate |

The audit found and fixed two material defects before v3: intended geometry had
previously been treated as realized changed support, and memmapped COMET images
had been point-decimated before area resizing. The final resampler agrees with
full OpenCV area resizing within one integer level on random arrays and preserves
phase-offset one-pixel lines. Duplicate content, source IDs, and canonical paths
are rejected before any split. All pre-v3 proxy artifacts are superseded.

## Final 896-pixel out-of-fold proxy results

Values are equal-weight means of source-group means. Brackets are descriptive
95% group-bootstrap intervals. `Outside` is the fraction of incremental score
mass outside realized changed support.

| Modality | Method | Artifact | AUPRC | Calibrated Dice [95% interval] | Lesion hit | Outside |
|---|---|---|---:|---:|---:|---:|
| COMET | Classical | Crack | 0.618 | 0.567 [0.373, 0.708] | 0.900 | 0.328 |
| COMET | Classical | Fold | 0.545 | 0.500 [0.412, 0.573] | 0.900 | 0.269 |
| COMET | Clean-reference anomaly | Crack | 0.049 | 0.065 [0.029, 0.098] | 0.667 | 0.942 |
| COMET | Clean-reference anomaly | Fold | 0.149 | 0.135 [0.036, 0.234] | 0.733 | 0.875 |
| COMET | Hybrid | Crack | 0.621 | 0.567 [0.335, 0.739] | 0.900 | 0.405 |
| COMET | Hybrid | Fold | 0.484 | 0.382 [0.279, 0.486] | 0.900 | 0.521 |
| CosMx | Classical | Crack | 0.276 | 0.235 [0.079, 0.438] | 1.000 | 0.640 |
| CosMx | Classical | Fold | 0.363 | 0.393 [0.252, 0.603] | 1.000 | 0.440 |
| CosMx | Clean-reference anomaly | Crack | 0.038 | 0.075 [0.035, 0.129] | 0.792 | 0.956 |
| CosMx | Clean-reference anomaly | Fold | 0.203 | 0.193 [0.050, 0.369] | 0.792 | 0.746 |
| CosMx | Hybrid | Crack | 0.272 | 0.234 [0.067, 0.401] | 1.000 | 0.749 |
| CosMx | Hybrid | Fold | 0.418 | 0.376 [0.242, 0.569] | 1.000 | 0.497 |

### Untouched-field alert burden

| Modality | Classical | Clean-reference anomaly | Hybrid |
|---|---:|---:|---:|
| COMET | 1.34% [0.68%, 2.11%] | 1.15% [0.49%, 1.82%] | 1.36% [0.68%, 2.16%] |
| CosMx | 3.04% [0.12%, 8.68%] | 14.33% [0.00%, 42.76%] | 19.46% [0.08%, 57.61%] |

These are review-workload/drift proxies. Unlabeled fields can already contain
artifacts, so the values are not specificity estimates.

## Resolution sensitivity

The same v3 protocol at 256 px completed in 20.9 seconds; the 896-pixel run took
454.6 seconds. Dice changed substantially:

| Modality/method/artifact | Dice at 256 | Dice at 896 | Change |
|---|---:|---:|---:|
| COMET classical crack | 0.656 | 0.567 | -0.089 |
| COMET classical fold | 0.537 | 0.500 | -0.037 |
| COMET anomaly crack | 0.006 | 0.065 | +0.058 |
| COMET anomaly fold | 0.082 | 0.135 | +0.053 |
| COMET hybrid crack | 0.391 | 0.567 | +0.177 |
| COMET hybrid fold | 0.225 | 0.382 | +0.157 |
| CosMx classical crack | 0.568 | 0.235 | -0.332 |
| CosMx classical fold | 0.465 | 0.393 | -0.072 |
| CosMx anomaly crack | 0.030 | 0.075 | +0.045 |
| CosMx anomaly fold | 0.069 | 0.193 | +0.124 |
| CosMx hybrid crack | 0.434 | 0.234 | -0.200 |
| CosMx hybrid fold | 0.234 | 0.376 | +0.142 |

This instability is itself a feasibility result. Final preprocessing must use
physical units and a frozen native/analysis MPP; COMET MPP is absent from this
deposit, which is a blocking metadata gap for physical-scale claims.

## Why generic anomaly detection underperformed

The low anomaly result is not a zero-map, orientation, leakage, or unsupported-
pixel coding accident. Coverage-aware reconstruction, finite-score checks,
paired subtraction, input-identity uniqueness, split isolation, deterministic
injection, and flip/inverse checks passed. Scientifically:

1. One-class distance means unusual relative to a small fit set, not fold or
   crack. Panel, exposure, tissue, and cohort variation can dominate.
2. Patch summaries and overlap averaging dilute a thin line.
3. PCA can discard the low-variance direction containing a subtle artifact.
4. The classical method shares intensity/edge cues with the perturbation
   generator, so the proxy is structurally favorable to it.

## What can be measured without labels

| Evidence | Valid metric/claim | Invalid claim |
|---|---|---|
| Compatibility | Decode, channel/MPP audit, coverage, runtime, memory, abstention | Artifact accuracy |
| Controlled perturbation | Incremental AUPRC/Dice, hit, outside activation, fixed-domain severity | Natural fold/crack performance |
| Metamorphic | Flip/inverse consistency, tile/full agreement, gain/noise/blur response | Correctness |
| Repeat/cycle | Registered DAPI/cycle disagreement, SSIM, structural change | Static artifact truth |
| Untouched inputs | Alert burden per tissue area/FOV | False-positive rate |
| Downstream association | Cell yield/fragmentation, unassigned transcripts, transcripts/cell, control counts | Causal ground truth |
| Model agreement | IoU/discordance for review sampling | Consensus as truth |

## Decision-grade evaluation criteria

The output contract should contain native-coordinate fold and crack masks, their
union, presence scores, tissue-normalized burden, uncertainty, valid coverage,
and an explicit abstention reason. Retain raw channel names/order, channel
mapping, native and analysis MPP, panel/run/scanner/site, and every transform.

For a pilot, collect at least 30 source groups per modality, enriched to at least
50 fold-positive, 50 crack-positive, and 100 clean/hard-negative regions. Use two
blinded readers plus adjudication. For a decision-grade comparison, target at
least 100 fold-positive, 100 crack-positive, and 200 clean regions across at
least 30 groups, with specimen/panel/run diversity and an external cohort.

Predeclare positive-field macro Dice by artifact as primary; sensitivity at a
clean alert-burden ceiling as the operating endpoint; lesion recall, boundary
tolerance, clean FP area, AUROC/AUPRC, Brier/ECE, coverage/abstention, runtime,
and memory as secondary endpoints. Report modality/subgroup floors and paired
source-group bootstrap differences. Synthetic data may test and augment; it
must never be the operational acceptance set.

## Reproduce

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc \
  multiplex-proxy-benchmark \
  --comet-dir data/public/qualifai_comet_v2/images \
  --cosmx-dir data/public/cosmx_gastric_v1/raw_morphology \
              data/public/cosmx_phgg_v1/raw_morphology \
  --mode logo-cv --max-dimension 896 --group-bootstrap-resamples 2000 \
  --output-json artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json
```
