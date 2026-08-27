# Current real H&E tissue-fold benchmark

**Run date:** 2026-08-26
**Schema:** `public-fold-benchmark-1.2`
**Evidence status:** all seven rows below are `report_eligible=true` for this
bounded public cohort.

## Answer first

Classical morphology, DINOv2-small, SigLIP2 Base, and Hibou-B have now completed
the same hardened benchmark on real H&E images with natural expert fold masks.
The encoders were frozen; each foundation encoder used both a clean-reference
PatchKNN head and a supervised linear token probe.

Hibou-B plus the linear probe has the strongest point estimates in this table:
positive-field macro Dice 0.667, all-field micro Dice 0.827, presence AUROC
0.985, and clean false-positive area 0.047%. An exploratory paired
source-slide bootstrap estimates its macro-Dice difference versus the DINOv2
linear probe as +0.068 [95% interval +0.053, +0.084]. This is descriptive
uncertainty on the locked cohort—not a hypothesis test or superiority claim.

This is real natural-mask fold evidence, but it remains narrow: veterinary 10x
teaching-slide fields from one public release. It does not establish crack
detection, human/Merck generalization, native WSI execution, COMET/CosMx
efficacy, or production readiness.

## Matched natural-mask H&E results

All methods use the same locked test: 424 fields from 55 supplied source-slide
groups, including 245 fold-positive and 179 clean fields. Brackets are 95%
source-slide-cluster bootstrap intervals from 1,000 resamples stratified by
organ and class.

| Method | Supervision | Positive-field macro Dice [95% CI] | All-field micro Dice [95% CI] | Presence AUROC [95% CI] | Presence AUPRC [95% CI] | Clean FP area [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Classical fold candidate | Calibration masks/presence only | 0.446 [0.375, 0.521] | 0.383 [0.318, 0.446] | 0.792 [0.692, 0.888] | 0.866 [0.777, 0.937] | 10.731% [10.094%, 11.425%] |
| DINOv2-small PatchKNN | Clean fit bank + calibration masks/presence | 0.341 [0.285, 0.389] | 0.434 [0.352, 0.497] | 0.884 [0.836, 0.921] | 0.898 [0.835, 0.941] | 1.789% [1.247%, 2.474%] |
| DINOv2-small linear probe | Fit masks + calibration masks/presence | 0.599 [0.532, 0.668] | 0.770 [0.724, 0.802] | 0.980 [0.963, 0.991] | 0.986 [0.974, 0.995] | 0.098% [0.029%, 0.209%] |
| SigLIP2 Base PatchKNN | Clean fit bank + calibration masks/presence | 0.208 [0.178, 0.236] | 0.224 [0.177, 0.257] | 0.803 [0.740, 0.856] | 0.817 [0.722, 0.885] | 7.282% [5.876%, 8.605%] |
| SigLIP2 Base linear probe | Fit masks + calibration masks/presence | 0.526 [0.462, 0.590] | 0.679 [0.625, 0.714] | 0.965 [0.933, 0.988] | 0.976 [0.951, 0.993] | 0.299% [0.127%, 0.577%] |
| Hibou-B PatchKNN | Clean fit bank + calibration masks/presence | 0.319 [0.264, 0.368] | 0.418 [0.362, 0.457] | 0.947 [0.910, 0.981] | 0.954 [0.898, 0.989] | 1.221% [0.717%, 1.768%] |
| Hibou-B linear probe | Fit masks + calibration masks/presence | 0.667 [0.603, 0.730] | 0.827 [0.791, 0.852] | 0.985 [0.973, 0.994] | 0.990 [0.981, 0.996] | 0.047% [0.026%, 0.076%] |

The supervision distinction is essential. PatchKNN is one-class at fit time but
uses labeled calibration masks and presence labels to select thresholds; it is
label-light, not fully unsupervised. The linear probe uses fit masks to learn a
spatial readout over frozen tokens. No encoder was fine-tuned, and LoRA was not
used in this efficacy benchmark.

The linear probe improves localization and clean burden for every tested
foundation representation. Conversely, pathology pretraining alone does not
turn a generic one-class distance into a semantic fold segmenter: Hibou-B
PatchKNN ranks field presence well but has lower overlap than DINOv2 PatchKNN.
SigLIP2 PatchKNN is weaker still. Representation and task head must therefore be
selected together.

### Exploratory paired differences

The deterministic paired-comparison artifact reuses the same stratified
source-slide bootstrap draw for both methods in each contrast. These intervals
describe macro-Dice differences on this one locked cohort. No p-values or
multiplicity adjustment were computed, and no superiority or noninferiority
claim is made.

| Positive-field macro-Dice contrast | Point difference | Exploratory 95% paired interval |
|---|---:|---:|
| Hibou-B linear minus DINOv2-small linear | +0.067974 | [+0.053130, +0.084232] |
| DINOv2-small linear minus SigLIP2 Base linear | +0.072716 | [+0.061862, +0.083173] |
| SigLIP2 Base linear minus classical | +0.080439 | [+0.045191, +0.115669] |
| DINOv2-small PatchKNN minus Hibou-B PatchKNN | +0.021906 | [-0.010370, +0.054272] |

The last interval crosses zero, illustrating why a point-estimate ordering is
not itself a statistical conclusion. The
[`public-fold-paired-comparison-1.0` artifact](../artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json)
has SHA-256
`9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2`;
an independent comparator rerun produced the same bytes and hash.

## Real dataset, integrity, and split

The benchmark uses
[Histology Tissue Fold Dataset v1.0](https://doi.org/10.5281/zenodo.21493260),
released under CC BY 4.0. The associated
[Bioengineering article](https://doi.org/10.3390/bioengineering13080937)
documents 10x acquisition, QuPath annotation, and second-expert review. The
release contains 2,127 `3840 x 2160` RGB fields from brain, kidney, liver, small
intestine, and testis: 899 clean and 1,228 fold-positive, with 283 supplied
source-slide IDs.

The hardened loader verifies the six-part canonical release identity before
scoring, decodes every image/mask, checks dimensions and binary values, and
requires exact metadata pairing. It detected two released positive masks that
are empty:

- `Kidney__Fold_-20260409133646374_mask.png`
- `Liver__Fold_-20260406112311790_mask.png`

Their presence labels remain positive, while their masks are excluded from
localization under the explicit exclusion manifest. One is in fit and one in
calibration; neither is in locked test. The exclusion-manifest SHA-256 is
`2002f53e1beb42f8743169d0d023f385b4d7a3cb943d972c5e7a13bb1bf57926`.

| Role | Fields | Supplied source-slide IDs | Clean | Fold-positive | Valid localization references |
|---|---:|---:|---:|---:|---:|
| Fit | 1,276 | 170 | 539 | 737 | 1,275 |
| Calibration | 427 | 58 | 181 | 246 | 426 |
| Locked test | 424 | 55 | 179 | 245 | 424 |

The organ-by-class, source-slide-grouped assignment-manifest SHA-256 is
`f4bd0267ca4c46e51748ea54c790ea98550af21d24955d9b26bef6f3d16b46a1`.
All fit/calibration/test source-slide overlaps are empty. These IDs are the
strongest public independence unit available; patient/block identifiers were
not released.

Pixel and presence thresholds are selected on calibration only. The bootstrap
intervals are conditional on this dataset, fixed assignment, fitted readout,
and calibrated thresholds. They do not include split instability, annotation
uncertainty, retraining variation, or domain shift.

## Metric interpretation

- Positive-field macro Dice gives each of the 245 positive fields equal weight.
- All-field micro Dice pools valid pixels across positive and clean fields, so
  false-positive area on clean fields affects the value.
- Presence AUROC/AUPRC rank all 424 fields.
- Clean FP area is the pooled predicted fold fraction over 179 released clean
  fields. Here, unlike the unlabeled multiplex proxy, it is a false-positive
  quantity because these fields carry released clean references.

The publisher reports DeepLabV3-ResNet50 Dice `0.7630 +/- 0.2425` on a different
185-positive-image subset and preprocessing path. It is useful context, not a
head-to-head rank against this table.

## Model, code, and artifact provenance

Every current artifact uses schema v1.2, validates run provenance before
scoring, records the Git commit plus dirty/untracked runtime-source digest,
captures Python/dependency/device/precision identities, binds the model and
configuration, and emits a hashed per-field locked-test outcome table. This is
reproducibility evidence, not corporate model-governance approval.

| Run | Locked model identity | Artifact | Artifact SHA-256 |
|---|---|---|---|
| Classical | Configuration SHA-256 `ad90e797ef5d101116ee7636dce752dae9a682cb7fe6aee4b3c83d77af1afcbd` | [`classical_hardened_v1_2.json`](../artifacts/public_fold/classical_hardened_v1_2.json) | `a23d1836cbda7e4a1835068d485c03463d178a084dcfe55266fc2183f96bcd19` |
| DINOv2-small | Revision `ed25f3a31f01632728cabb09d1542f84ab7b0056`; weight SHA-256 `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` | [`dinov2_hardened_v1_2.json`](../artifacts/public_fold/dinov2_hardened_v1_2.json) | `5846a4edf7b7f8a882c5d211d37934bc0e3f15c18ab906cde5951e98f6b47fbd` |
| SigLIP2 Base | Revision `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`; weight SHA-256 `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` | [`siglip2_hardened_v1_2.json`](../artifacts/public_fold/siglip2_hardened_v1_2.json) | `4a8cbd2f45a3023a6c1313daf16be05913256b1381d5f83023bd9120e93b2596` |
| Hibou-B | Official source commit `c453bbe4dab0fec6f7df343b09ea87048629c58d`; weight SHA-256 `9d3e5ebc4e1ffaf6d7a0b672273e4fbef109cdd03df73c52920d6e886f2327e1` | [`hibou_hardened_v1_2.json`](../artifacts/public_fold/hibou_hardened_v1_2.json) | `43d46447c7bcfd971691c97a3a99d10d8a78dddf4486e269f2f5ed8f1173301d` |
| Paired comparator | Schema `public-fold-paired-comparison-1.0`; exploratory descriptive contrasts only | [`hardened_all_methods_paired_comparison_v1.json`](../artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json) | `9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2` |

All three encoders resolved to Apple MPS with float32 inference. Model fitting
and shared calibration/test scoring took 207.5/317.1 seconds for DINOv2,
354.8/491.2 seconds for SigLIP2, and 472.1/553.5 seconds for Hibou-B. The two
heads share encoder scoring time, so their runtimes must not be added. Classical
calibration and locked-test wall times were 141.0 and 66.7 seconds. These scopes
exclude dataset download, full integrity validation, bootstrap/report assembly,
and native WSI I/O.

## Image-scale and claim limits

Every source field was downsampled aspect-preservingly to at most 896 pixels,
then scored in `224 x 224` image-space tiles with stride 224. The release does
not supply trustworthy per-image MPP, so this is an image-space comparison, not
a physical-scale upper bound. A locked multi-resolution study remains required.

This benchmark does not establish:

- crack/tear, glass/coverslip crack, knife line, or acquisition-seam detection;
- human clinical or Merck-distribution validity;
- native-pyramid WSI throughput, scanner/site robustness, or prospective use;
- COMET/CosMx efficacy or H&E-to-multiplex transfer;
- comprehensive SOTA coverage; same-split U-Net/DeepLab, HistoQC, GrandQC,
  DiffusionQC, and governed larger encoders were not all executed;
- statistical superiority among the seven rows: the paired intervals are
  exploratory descriptive summaries without p-values or multiplicity control.

No synthetic image was used in fit, calibration, or locked test for these H&E
results. Synthetic perturbations are used elsewhere for software regression and
the explicitly separate multiplex proxy; they never replace a natural-mask test.

## Exact rerun commands

All commands require the downloaded public cohort and the checked-in hardened
implementation. Model paths must point to the exact locked local assets.

### Classical

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods classical_fold --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/classical_hardened_v1_2.json
```

### DINOv2-small

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods dinov2_patchknn dinov2_linear_probe \
  --foundation-encoder dinov2-hf --model-id facebook/dinov2-small \
  --revision ed25f3a31f01632728cabb09d1542f84ab7b0056 \
  --cache-dir models/hf_home/hub --device mps \
  --probe-max-iterations 500 --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/dinov2_hardened_v1_2.json
```

### SigLIP2 Base

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc public-fold-benchmark \
  --dataset-root data/public/histology_tissue_fold_v1 \
  --methods foundation_patchknn foundation_linear_probe \
  --foundation-encoder siglip2-base-local \
  --siglip2-snapshot /path/to/hash-locked/siglip2-base-patch16-224 \
  --device mps --probe-max-iterations 500 --max-dimension 896 \
  --bootstrap-resamples 1000 --exclude-empty-positive-masks \
  --output-json artifacts/public_fold/siglip2_hardened_v1_2.json
```

### Hibou-B

```bash
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

## Legacy appendix

The pre-v1.2 balanced artifacts remain preserved for audit only:

- [`classical_balanced_formal.json`](../artifacts/public_fold/classical_balanced_formal.json)
- [`dinov2_balanced_formal.json`](../artifacts/public_fold/dinov2_balanced_formal.json)
- [`hibou_balanced_formal.json`](../artifacts/public_fold/hibou_balanced_formal.json)

They explicitly record `report_eligible=false` and
`complete_legacy_nonreportable_pre_hardening`. Their numeric values happen to
match the corresponding reruns closely, but they do not carry the complete
v1.2 provenance and per-field contract and must not be substituted for the four
current hardened artifacts.

The old 302-field group-count split and Hibou smoke artifact are also
nonreportable and excluded from every current result table.

The real COMET/CosMx analysis is documented separately in
[MULTIPLEX_REAL_PROXY_BENCHMARK.md](MULTIPLEX_REAL_PROXY_BENCHMARK.md). It uses
real backgrounds with synthetic perturbation masks and is not comparable to the
natural-mask H&E table above. Its primary v3 artifact SHA-256 is
`a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`.
