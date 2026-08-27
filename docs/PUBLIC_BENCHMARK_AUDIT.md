# Public real-data and method audit

Verified: 2026-08-26  
Decision: no public benchmark supplies fold **and** crack references across H&E,
COMET, and CosMx. Public evidence must be separated into pixel localization,
patch classification, and unlabeled domain testing.

## Executable method candidates

| Method | Actual scope | Public assets and restrictions | Valid benchmark role |
|---|---|---|---|
| [HistoQC](https://github.com/choosehappy/HistoQC) | Classical H&E WSI QC and usable-tissue masks; not distinct semantic fold/crack outputs | BSD-3-Clause-Clear code; no fold/crack reference dataset bundled | External baseline after an explicit output-to-ontology mapping; H&E only |
| [GrandQC](https://github.com/cpath-ukk/grandqc) | H&E multiclass segmentation including fold, OOF, pen, dark spot/foreign object, edge/air bubble; no crack class | Code, checkpoints, and [expert test masks](https://zenodo.org/records/14039591) are CC BY-NC-SA/noncommercial | Strongest ready public real H&E fold-localization comparator, but Merck use needs legal/licensor clearance |
| [HistoArtifacts MoE](https://github.com/NeelKanwal/Equipping-Computational-Pathology-Systems-with-Artifact-Processing-Pipeline) | H&E patch classification including folded and damaged tissue | GPL-3.0 code and weights; dataset rights are not clearly declared | Patch classification only; “damaged tissue” is not a crack mask |
| [QUALIFAI](https://github.com/augpath/QualIFAI) | Multiplex-IF tile classification plus segmentation for fold, OOF, bubble, external artifact, and aggregate; includes COMET development evidence, not CosMx | [v2 data/models](https://zenodo.org/records/12699470) state CC BY 4.0; repository code is academic-only | Five complete DAPI fields support real-background proxy testing, but the checksum-valid v2 archive is truncated and contains no masks or split manifest; no independently reproducible natural-artifact Dice or crack claim |
| [DiffusionQC](https://arxiv.org/abs/2601.12233) | Merck-led H&E diffusion anomaly heatmaps/masks for an artifact union including folds, pen, OOF, and bubbles | No official public code, weights, manifest, or reusable dataset located; manuscript license is not an implementation license | Not independently executable today; obtain internal assets and reconcile reported split counts before comparison |
| [HistoART](https://github.com/DIDSR/HistoART) | H&E patch classification for fold and other artifacts | Wrapper repository is CC0, but its upstream UNI dependency is gated/noncommercial | Patch classification only; upstream model terms still govern the weights |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Generic RGB frozen CLS/spatial tokens with no pathology-QC ontology | Apache-2.0 base code/weights | Current hardened MPS H&E control with matched PatchKNN and linear-probe heads |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Newer 2025 generic vision family with dense representations | Checkpoint access requires license acceptance; the custom [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md) defines the accepting party to include an employer/entity when acting on its behalf | Technically relevant newer generic control, but not downloaded or run because institutional acceptance is outside this benchmark's authority; this is not a claim that the license prohibits commercial use |
| [SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224) | Generic RGB ViT-B/16 vision-language tower with 14×14 dense vision tokens at 224 px | Ungated Apache-2.0 official card; exact local snapshot is hash locked | Apple-MPS/LoRA smoke passed; hardened public H&E linear-head macro Dice 0.526 (0.462–0.590), PatchKNN 0.208 (0.178–0.236) |
| [Hibou-B](https://huggingface.co/histai/hibou-b) | 85.7M-parameter pathology ViT-B/14 trained with DINOv2 on mixed-magnification, mixed-stain WSI tiles | Apache-2.0 code and weights; official [repository](https://github.com/HistAI/hibou) also supplies a direct weight mirror | Current hardened MPS pathology comparator; linear-head point estimate leads and exploratory paired difference versus DINOv2 linear is positive, without a superiority claim |
| [GigaPath-Flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash) | 22M-parameter 2026 pathology tile encoder plus a 21M-parameter slide encoder | Apache-2.0 but gated; its model card limits intended use to research/reproducibility and says deployment is out of scope | Attractive newer lightweight comparator after Merck governance and access approval; unavailable to the current authenticated account |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | 681M-parameter pathology ViT-H/14 trained on more than 200M H&E/IHC tiles | Gated CC BY-NC-ND 4.0; commercial use requires separate approval | Scientifically relevant comparator, but not runnable for Merck work without institutional access and licensing |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) | 632M-parameter mixed-magnification pathology ViT-H/14 with dense spatial tokens | Gated CC BY-NC-ND 4.0 and academic-research-only conditions | Relevant but substantially heavier and license-blocked for the present corporate benchmark |
| [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) | 307M-class ViT-L/16 pathology vision-language-derived encoder | Gated CC BY-NC-ND 4.0; commercial use requires approval | Relevant for semantic/zero-shot studies, but the official newer name is **CONCHv1.5**, not “CONCH-v2”; license-blocked here |
| [KRONOS2](https://huggingface.co/MahmoodLab/KRONOS2) | Marker-aware DINOv2 ViT-B/16 for arbitrary multiplex spatial-proteomics panels, conditioned on marker names and MPP | Gated CC BY-NC-ND 4.0; ordinary commercial testing/adaptation and derivatives require separate approval | Architecturally the best-aligned public candidate for COMET/CosMx, but deliberately not downloaded or run without written institutional approval; it would still need a localization head and artifact labels |

OME-TIFF compatibility does not establish COMET or CosMx validity. Likewise, an
anomaly heatmap is not a fold/crack semantic classifier. The benchmark contract
therefore compares generic anomaly models only against `artifact_union`.

## Real public datasets

| Dataset | Ground truth | Defensible use |
|---|---|---|
| [Histology Tissue Fold Dataset v1.0](https://doi.org/10.5281/zenodo.21493260) | 2,127 real 10x H&E microscope fields from five veterinary tissues: 899 clean and 1,228 fold-positive, with pixel masks and 283 provided source-slide groups; CC BY 4.0 | Best permissively licensed public real H&E fold benchmark found. Use slide-grouped splits and report it as single-site teaching-slide fold evidence, not human clinical WSI or crack evidence. Independent decode QA found two released fold-positive masks are empty; retain their presence labels but exclude them from localization metrics with an explicit audit trail. The [associated study](https://doi.org/10.3390/bioengineering13080937) reports a DeepLabV3-ResNet50 test Dice of 0.7630 ± 0.2425, but comparisons require matching its split and preprocessing. |
| [GrandQC manually annotated test set](https://zenodo.org/records/14039591) | Expert pixel masks from real H&E source WSIs, including fold; no crack class | Real H&E fold-localization evaluation if noncommercial terms are cleared |
| [GrandQC TCGA predicted masks](https://zenodo.org/records/14041578) | Model predictions, not independent references | Mining or triage only; never unbiased efficacy ground truth |
| [HistoArtifacts](https://zenodo.org/records/10809442) | Real H&E patch/folder labels including fold and damaged tissue; no pixel masks | Patch classification and representation probing, subject to data-rights review |
| [QUALIFAI v2](https://zenodo.org/records/12699470) | Real COMET images and model checkpoints; the checksum-valid 12.5 GB image archive is truncated after ten image members and contains no masks, labels, or split manifest | Real-input inference and qualitative domain testing only. The paper's COMET Dice/IoU cannot be independently reproduced from the public deposit. |
| [Lunaphore COMET lung TMA](https://lunaphore.com/download-center-tma-downstream-analysis/) | Real 20-plex OME-TIFF plus DAPI; no artifact masks | Ingestion/runtime smoke only if restrictive vendor terms are approved |
| [CosMx gastric mucosa](https://zenodo.org/records/8333281) | Public five-channel `uint16` raw morphology FOVs at 0.18 µm/px; no fold/crack masks | Four FOVs from two slide/run groups were executed in the real-background proxy benchmark; CC BY 4.0, not natural-artifact efficacy |
| [CosMx pediatric high-grade glioma](https://zenodo.org/records/16877090) | Public five-channel `uint16` raw morphology FOVs at 0.120280945 µm/px; no fold/crack masks | Two FOVs from two additional slide/run groups were range-extracted with ZIP CRC verification and executed in the proxy benchmark; CC BY 4.0, not natural-artifact efficacy |
| [Dryad CosMx melanoma RNA-SMI](https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7b1) | Public metadata describes four CosMx slides with DAPI/protein morphology FOVs; no fold/crack masks | Potential CC0 annotation source, but authenticated file access was unavailable in this audit and no files from it were benchmarked |

No verified public pixelwise crack benchmark was found for H&E, COMET, or CosMx.
Fold, tissue tear, tissue damage, knife line, section separation, acquisition seam,
and coverslip/glass crack must not be silently treated as equivalent targets.

## Download and integrity audit

### Histology Tissue Fold Dataset v1.0

The benchmark uses the exact Zenodo files and the publisher's supplied
`slide_image_mapping.xlsx`. The archive identities are:

| Asset | Bytes | Publisher MD5 |
|---|---:|---|
| `images.zip` | 6,595,418,947 | `2d4116bf652d1dba761d3164909d8df3` |
| `masks.zip` | 23,088,245 | `7418aa15933b013e759db4e02b565cb7` |
| `metadata.csv` | 407,081 | `bb4bd671afa378675c32f5b288bebd30` |
| `slide_image_mapping.xlsx` | 51,532 | `fd3dcab8da600d35955fde179b2db993` |

Strict decode QA verified all 2,127 images as `3840 x 2160 x 3`, all 1,228
released masks as dimension-matched and binary, exact asset pairing, no orphan
files, ten expected organ/class strata, and 283 distinct provided `slide_id`
groups. Two fold-positive masks are entirely empty:

- `Kidney__Fold_-20260409133646374_mask.png`
- `Liver__Fold_-20260406112311790_mask.png`

Their fold-presence labels are retained. Their masks are excluded from
localization metrics under a named, hashed exclusion policy. This is preferable
to silently treating them as background. The paper describes 302 teaching
specimens, whereas the released mapping contains 283 unique group IDs; the
executable benchmark uses the released mapping and records this evidence limit.

### Current hardened real H&E fold benchmark

Four current schema-v1.2 reports use one hashed, organ-by-class,
source-slide-grouped assignment: 1,276 fields/170 groups for fit, 427/58 for
calibration, and a locked test of 424/55 (179 clean; 245 fold-positive). All
split-overlap checks pass, every current artifact validates release and run
provenance before scoring, and intervals use 1,000 source-slide-cluster
bootstrap resamples. DINOv2-small, SigLIP2 Base, and Hibou-B resolved to Apple
MPS.

| Method | Supervision budget | Positive-field macro Dice (95% CI) | All-field micro Dice (95% CI) | Presence AUROC | Clean FP area |
|---|---|---:|---:|---:|---:|
| Classical fold candidate | Calibration masks/presence only | 0.446 (0.375–0.521) | 0.383 (0.318–0.446) | 0.792 | 10.73% |
| DINOv2-small PatchKNN | Clean fit tokens plus calibration masks/presence | 0.341 (0.285–0.389) | 0.434 (0.352–0.497) | 0.884 | 1.79% |
| DINOv2-small linear probe | Fit masks plus calibration masks/presence | 0.599 (0.532–0.668) | 0.770 (0.724–0.802) | 0.980 | 0.098% |
| SigLIP2 Base PatchKNN | Clean fit tokens plus calibration masks/presence | 0.208 (0.178–0.236) | 0.224 (0.177–0.257) | 0.803 | 7.28% |
| SigLIP2 Base linear probe | Fit masks plus calibration masks/presence | 0.526 (0.462–0.590) | 0.679 (0.625–0.714) | 0.965 | 0.299% |
| Hibou-B PatchKNN | Clean fit tokens plus calibration masks/presence | 0.319 (0.264–0.368) | 0.418 (0.362–0.457) | 0.947 | 1.22% |
| Hibou-B linear probe | Fit masks plus calibration masks/presence | 0.667 (0.603–0.730) | 0.827 (0.791–0.852) | 0.985 | 0.047% |

Interpretation: the pathology-specific encoder has the strongest supervised-
head point estimate and lowest clean burden, but does not improve PatchKNN Dice
over DINOv2. The method-wise table does not establish statistical superiority.
The completed exploratory paired source-slide bootstrap reports:

| Positive-field macro-Dice contrast | Difference | Exploratory 95% paired interval |
|---|---:|---:|
| Hibou-B linear minus DINOv2-small linear | +0.067974 | [+0.053130, +0.084232] |
| DINOv2-small linear minus SigLIP2 Base linear | +0.072716 | [+0.061862, +0.083173] |
| SigLIP2 Base linear minus classical | +0.080439 | [+0.045191, +0.115669] |
| DINOv2-small PatchKNN minus Hibou-B PatchKNN | +0.021906 | [-0.010370, +0.054272] |

These are descriptive, cohort-conditional intervals using the same cluster draw
for each paired contrast; no p-values, multiplicity adjustment, superiority, or
noninferiority claim was produced. The byte-reproducible
[paired artifact](../artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json)
has SHA-256
`9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2`.
This remains seven rows across three method families, not a comprehensive SOTA
benchmark.

The associated paper's DeepLabV3-ResNet50 Dice `0.7630 +/- 0.2425` uses a
different split and preprocessing pipeline and cannot be ranked directly.
Commands, artifact hashes, provenance, and limitations are in the
[current real public benchmark report](REAL_PUBLIC_BENCHMARK.md). Current
artifacts are
[`classical_hardened_v1_2.json`](../artifacts/public_fold/classical_hardened_v1_2.json),
[`dinov2_hardened_v1_2.json`](../artifacts/public_fold/dinov2_hardened_v1_2.json),
[`siglip2_hardened_v1_2.json`](../artifacts/public_fold/siglip2_hardened_v1_2.json),
and [`hibou_hardened_v1_2.json`](../artifacts/public_fold/hibou_hardened_v1_2.json).

The pre-v1.2 `*_balanced_formal.json` files remain in the legacy appendix of
that report with `report_eligible=false`; they are not substituted for these
current artifacts.

This evidence is real but narrow: veterinary 10x JPEG fields, fold only,
downsampled evaluation, and supplied source-slide IDs as the highest available
independence unit. It is not evidence for human clinical WSI, cracks/tears,
Merck distributions, COMET, or CosMx. Public COMET and CosMx resources reviewed
below still lack usable fold/crack ground truth.

### QUALIFAI / COMET

The official v2 assets were downloaded and verified exactly:

| Asset | Bytes | Publisher MD5 | Audit outcome |
|---|---:|---|---|
| `Lunaphore_dataset.tar.gz` | 12,485,349,376 | `733c12420230f68c81e02925c8b809a1` | `gzip -t` fails with unexpected EOF; ten image members only; no masks/manifest |
| `models.tar.gz` | 2,434,804,913 | `08940d90409d4b878edf34966d5e7d2f` | Model deposit; separate code terms still require review |

The matching byte count and MD5 after two downloads establish an upstream
deposit defect rather than a downloader or evaluator error. Five complete real
16-bit DAPI images (`COMET02` through `COMET06`) can be recovered for domain
inference. The peer-reviewed [QUALIFAI article](https://doi.org/10.1016/j.xcrp.2024.102220)
reports two COMET slides, 0.22 micrometres/pixel, 71/37/39 fold tiles for
train/test/validation, and fold IoU `0.72 +/- 0.12`; masks and an independently
grouped test are not reproducible from the deposit.

The released implementation also differs from the paper in material ways:

- paper q99 normalization versus code 2nd-97th percentile rescaling;
- stated 20-pixel overlap versus 512-pixel tiles at stride 312 (200 pixels);
- a non-assigned `replace("doubt", "yes")` operation can leave doubtful tiles
  unsegmented;
- the wrapper emits an artifact-union mask, not a fold-specific mask.

For future use, call the verified fold segmenter directly with frozen,
documented preprocessing instead of treating the wrapper as a SOTA oracle.

### CosMx

The executed CosMx proxy uses two separate public CC-BY-4.0 cohorts:

- [gastric mucosa](https://zenodo.org/records/8333281): four real five-channel
  morphology FOVs from two slide/run groups at 0.18 micrometres/pixel;
- [pediatric high-grade glioma](https://zenodo.org/records/16877090): two real
  five-channel morphology FOVs from two additional slide/run groups at
  0.120280945 micrometres/pixel.

The raw images are useful for ingestion, channel ablation, real-background
perturbation testing, alert-burden measurement, and creating an expert reference
set. Their cell/compartment products are not fold or crack QC references. The
Dryad melanoma metadata was also audited, but file download required
authentication and no Dryad image was used in the reported run.

### Executed real COMET/CosMx proxy benchmark

The [locked proxy report](MULTIPLEX_REAL_PROXY_BENCHMARK.md) executes five
complete QUALIFAI COMET DAPI fields and six CosMx morphology FOVs. Within each
LOGO fold, fit/calibration/test source groups are disjoint; groups are reused
across folds, so folds are statistically dependent. Controlled folds and cracks
are inserted into held-out real backgrounds, and evaluation uses the paired
incremental response `max(score(injected) - score(base), 0)`.

| Modality | Method | Mean incremental AUPRC | Mean calibrated Dice | Untouched alert burden |
|---|---|---:|---:|---:|
| COMET | Classical | 0.582 | 0.534 | 1.34% |
| COMET | Clean-reference anomaly | 0.099 | 0.100 | 1.15% |
| COMET | Hybrid | 0.552 | 0.475 | 1.36% |
| CosMx | Classical | 0.319 | 0.314 | 3.04% |
| CosMx | Clean-reference anomaly | 0.121 | 0.134 | 14.33% |
| CosMx | Hybrid | 0.345 | 0.305 | 19.46% |

The anomaly result was verified against finite-score, coverage, split,
paired-subtraction, determinism, and horizontal-flip checks; it is not explained
by a zero-output or orientation bug. It remains weak for the controlled thin
crack cue (Dice 0.065 on COMET and 0.075 on CosMx). Values in the table average
the separate fold/crack group-macro estimates from v3 LOGO evaluation. These are
**generator-conditional proxy metrics**, not natural-artifact Dice. Untouched
activity is alert burden, not a false-positive rate, and the JSON correctly sets
`report_eligible=false` and `scientific_validation_passed=false`. Higher-level
biological independence is undeclared, so group-bootstrap intervals are
descriptive. Natural-artifact Dice, ROC, sensitivity, specificity, and FPR are
not estimable. The
[primary v3 artifact](../artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json)
has SHA-256
`a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`.

The 256-pixel sensitivity run changed individual Dice values by up to 0.332;
COMET crack severity response was nonmonotonic at 896 px. This is additional
evidence against a production ranking.

## Feasible comparison matrix

| Target/evidence | H&E | COMET | CosMx |
|---|---|---|---|
| Real public fold localization | Histology Tissue Fold v1.0 (CC BY 4.0); GrandQC if separately license-cleared | QUALIFAI candidate after archive/split audit | None |
| Real public crack localization | None verified | None verified | None verified |
| Patch classification | HistoArtifacts, HistoART | QUALIFAI tile stage | None identified |
| Unlabeled real-domain testing | Public pathology WSIs | Executed on five QUALIFAI DAPI fields with controlled perturbations and alert burden | Executed on six morphology FOVs/four slide-run groups from two Zenodo cohorts with controlled perturbations and alert burden |
| Corporate-controlled comparison | Hardened classical, DINOv2-small, SigLIP2 Base, and Hibou-B rows plus exploratory paired contrasts completed; add same-split segmentation anchors, governed models, and internal labels before SOTA claims | Classical + QUALIFAI/internal head; KRONOS2 only after license/access and annotation | Classical + channel-aware internal head; KRONOS2 only after license/access and annotation |

Synthetic perturbations remain useful for regression, stress testing, and perhaps
pretraining, but never replace a real held-out specimen-level test. The most
credible cross-modal benchmark is therefore a shared evaluation protocol with
separate modality cohorts—not a claim that one public dataset or model already
solves all three.
