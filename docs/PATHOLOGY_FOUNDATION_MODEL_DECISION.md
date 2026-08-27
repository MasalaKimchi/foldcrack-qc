# Pathology foundation-model decision for fold/crack QC

**Review date:** 2026-08-26

**Scope:** H&E, COMET and CosMx image-quality control; artifact presence, localization and burden—not diagnosis.

## Decision

Retain DINOv2-small as the generic control and Hibou-B as the current locally
runnable pathology-specific comparator. Classical, DINOv2-small, SigLIP2 Base,
and Hibou-B have now completed one matched schema-v1.2 real H&E benchmark with
validated pre-scoring provenance. Hibou-B plus the frozen linear probe has the
strongest point estimates. The completed exploratory paired source-slide
bootstrap estimates Hibou-B linear minus DINOv2 linear macro Dice as +0.067974
[+0.053130, +0.084232]. No p-values or multiplicity adjustment were computed;
do not convert the descriptive interval into a statistical-superiority claim.

DINOv3 is newer than DINOv2 and is technically relevant for dense localization,
but it was not downloaded or run. Official checkpoint access requires license
acceptance, and the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
defines the accepting party to include an employer/entity when accepted on its
behalf. That is an institutional-governance decision, not a coding omission;
this document does not characterize the custom license as noncommercial.

For COMET/CosMx, [KRONOS2](https://huggingface.co/MahmoodLab/KRONOS2) is the
most modality-aligned public candidate found, but it returns a marker-conditioned
patch embedding rather than an artifact mask and is gated under CC BY-NC-ND
4.0. Ordinary Merck testing or adaptation therefore requires written approval.
[UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) and
[CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) have the same gated,
noncommercial/NoDerivatives issue. The permissive near-term control is
[SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224): its
official card is Apache-2.0, and a hash-locked Apple-MPS inference plus rank-4
LoRA engineering smoke passed locally. Its hardened real H&E run also completed:
the frozen linear head reached positive-field macro Dice 0.526 (0.462–0.590),
while PatchKNN reached 0.208 (0.178–0.236). It remains a generic RGB encoder and
this is fold-only evidence on one veterinary public cohort.

A unified product does not require one encoder. The defensible design is a **shared QC ontology, output contract and evaluation protocol**, with an RGB pathology encoder for H&E and a marker-aware encoder or channel-aware model for multiplex images.

No official model card cited below reports fold/crack-QC performance. Model scale or downstream pathology accuracy must not be presented as evidence of artifact detection.

## Is DINOv2 too old?

Verified: DINOv2 dates to 2023; Meta's official
[DINOv3 repository](https://github.com/facebookresearch/dinov3) identifies its
2025 continuation. DINOv2 nevertheless remains a useful controlled baseline:
its official model card exposes class and dense patch tokens, gives a
21M-parameter ViT-S/14 option, and uses Apache-2.0 terms. Several newer pathology
models—including Hibou-B, UNI2-h, Virchow2 and KRONOS2—use DINOv2 or a modified
DINOv2 recipe. See the official
[DINOv2 model card](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md).

Decision: keep DINOv2 because it measures the gain from pathology- or
modality-specific pretraining. Do not treat its rank as the final model-selection
result. The two-patch
[foundation smoke artifact](../artifacts/foundation_smoke/foundation_smoke.json)
remains engineering-only. The numerical H&E comparison below is bounded fold
evidence on one veterinary teaching-slide cohort, and its four current run
artifacts satisfy the hardened v1.2 provenance contract.

## Naming clarification: UNI2-h and CONCH

Verified public artifacts are:

- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), a gated 681M-parameter ViT-H/14 pathology vision backbone.
- [CONCH](https://huggingface.co/MahmoodLab/CONCH), the original ViT-B/16 pathology vision-language model.
- [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5), a gated ViT-L/16 successor whose current official card still lists its repository and paper as `TBD`.

As of the review date, no official Mahmood Lab model card or repository named **CONCH-v2** was located. Mahmood Lab's own [TRIDENT encoder table](https://github.com/mahmoodlab/TRIDENT) lists `CONCH` and `CONCHv1.5`, not CONCH-v2. This is a date-stamped source audit, not a claim that such a name can never appear.

## Current matched natural-mask H&E comparison

The benchmark uses all 2,127 Histology Tissue Fold v1.0 fields with one hashed,
organ-by-class, supplied-source-slide-grouped assignment. The locked test has
424 fields from 55 groups (179 clean; 245 fold-positive). All current artifacts
use 1,000 source-slide-cluster bootstrap resamples, validate run provenance
before scoring, and are `report_eligible=true` for this bounded cohort.

| Encoder / head | Supervision | Positive-field macro Dice (95% CI) | All-field micro Dice (95% CI) | Presence AUROC | Clean FP area |
|---|---|---:|---:|---:|---:|
| Classical fold candidate | Calibration masks/presence only | 0.446 (0.375–0.521) | 0.383 (0.318–0.446) | 0.792 | 10.73% |
| DINOv2-small PatchKNN | Clean fit bank plus calibration labels | 0.341 (0.285–0.389) | 0.434 (0.352–0.497) | 0.884 | 1.79% |
| DINOv2-small linear probe | Fit masks plus calibration labels | 0.599 (0.532–0.668) | 0.770 (0.724–0.802) | 0.980 | 0.098% |
| SigLIP2 Base PatchKNN | Clean fit bank plus calibration labels | 0.208 (0.178–0.236) | 0.224 (0.177–0.257) | 0.803 | 7.28% |
| SigLIP2 Base linear probe | Fit masks plus calibration labels | 0.526 (0.462–0.590) | 0.679 (0.625–0.714) | 0.965 | 0.299% |
| Hibou-B PatchKNN | Clean fit bank plus calibration labels | 0.319 (0.264–0.368) | 0.418 (0.362–0.457) | 0.947 | 1.22% |
| Hibou-B linear probe | Fit masks plus calibration labels | 0.667 (0.603–0.730) | 0.827 (0.791–0.852) | 0.985 | 0.047% |

Hibou-B's supervised-head point estimate is higher. The separate method-wise
intervals in the table are complemented by these exploratory paired
positive-field macro-Dice contrasts:

| Contrast | Difference | Exploratory 95% paired interval |
|---|---:|---:|
| Hibou-B linear minus DINOv2-small linear | +0.067974 | [+0.053130, +0.084232] |
| DINOv2-small linear minus SigLIP2 Base linear | +0.072716 | [+0.061862, +0.083173] |
| SigLIP2 Base linear minus classical | +0.080439 | [+0.045191, +0.115669] |
| DINOv2-small PatchKNN minus Hibou-B PatchKNN | +0.021906 | [-0.010370, +0.054272] |

The comparator used the same source-slide bootstrap draw for both members of
each contrast. Its intervals are cohort-conditional descriptive summaries,
not hypothesis tests; no p-values, multiplicity adjustment, superiority, or
noninferiority claim is provided. The byte-reproducible
[paired artifact](../artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json)
has SHA-256
`9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2`.

The runs establish that the three frozen stacks are practical on this Mac:
feature fitting plus shared calibration/test scoring took about
207/317 seconds for DINOv2, 355/491 seconds for SigLIP2, and 472/554 seconds for
Hibou-B. These scopes exclude some validation, metric, bootstrap, and I/O work.

This is a seven-row matched method/head comparison, not comprehensive SOTA. It covers real H&E fold
localization after downsampling, not WSI-scale streaming, crack, human/Merck
generalization, COMET, or CosMx.

## Candidate matrix

“Token geometry” describes the encoder grid, not native-slide resolution. The physical footprint is `patch stride × effective MPP after resampling`; it must be recorded per run.

| Model | Verified architecture/interface | Localization consequence | Mac/MPS assessment | Access and Merck-use assessment | Benchmark decision |
|---|---|---|---|---|---|
| [DINOv2-small](https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md) | Generic ViT-S/14, 21M parameters; 224 px gives 16×16 patch tokens of dimension 384 | Dense tokens are directly suitable for a frozen token probe or PatchKNN heatmap | **Verified locally:** CPU/MPS agreement, one BF16 LoRA engineering step, and a current hardened H&E run on MPS | Apache-2.0; no pathology pretraining | Retain as the current generic control |
| [DINOv3](https://github.com/facebookresearch/dinov3) | Newer 2025 generic vision family with dense representations and small ViT options | Technically suitable for the same frozen dense-token heads after geometry verification | **Not run:** checkpoint access and the custom license must first be accepted by an authorized Merck party | Custom DINOv3 license; acceptance/use binds the accepting employer/entity when acting on its behalf | High-priority generic control after institutional license/governance approval; do not imply the license is noncommercial |
| [SigLIP2 Base](https://huggingface.co/google/siglip2-base-patch16-224) | Generic ViT-B/16 vision-language tower; 224 px gives a 14×14 dense grid of dimension 768 | Dense vision tokens fit the existing PatchKNN/linear-head interface | **Verified:** hash-locked Apple-MPS/LoRA smoke plus hardened public H&E frozen-head run; linear macro Dice 0.526, PatchKNN 0.208 | Ungated Apache-2.0 official model card | Retain as the current permissive generic comparator; bounded fold-only evidence, not multiplex/crack validation |
| [Hibou-B](https://huggingface.co/histai/hibou-b) / [official source](https://github.com/HistAI/hibou) | Pathology ViT-B/14, 85.7M parameters, DINOv2-based; official source defaults to 224 px and four registers | 16×16 dense patch-token grid at 224 px after excluding CLS/register tokens | **Verified locally:** official pinned source/weights completed the current hardened H&E run on MPS | Apache-2.0; HF access form is gated, while the official repository provides a direct weight link | Current locally runnable pathology comparator; linear-head point estimate leads and exploratory paired interval versus DINOv2 linear is positive, without a superiority claim |
| [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) | Pathology ViT-H/14, 681M parameters, 1,536-d embeddings, eight registers; trained with a DINOv2 recipe on over 200M H&E/IHC tiles | At 224 px the nominal grid is 16×16, but the card's supported example returns a global embedding; dense-token extraction/registration must be verified | **High-risk on a laptop:** frozen batched inference may fit with careful precision, but throughput and PEFT back-propagation are much heavier than Hibou-B | Gated CC BY-NC-ND 4.0; card prohibits commercial use and derivatives without prior approval | Scientifically valuable H&E comparator only after written access/license approval |
| [Virchow2](https://huggingface.co/paige-ai/Virchow2) | Pathology ViT-H/14, 632M parameters, four registers; card explicitly exposes 256×1,280 patch tokens for 224 px input | Verified 16×16 dense token grid; well specified for a frozen dense probe | **High-risk on a laptop:** similar scale concern to UNI2-h; official guidance demonstrates CUDA mixed precision, not MPS | Gated CC BY-NC-ND 4.0; card restricts use to academic research and includes additional medical-use restrictions | Add only after legal/governance approval; not the fastest next run |
| [CONCH](https://huggingface.co/MahmoodLab/CONCH) / [CONCHv1.5](https://huggingface.co/MahmoodLab/conchv1_5) | Pathology vision-language encoders; CONCHv1.5 is ViT-L/16 and its current card documents global feature extraction | A global embedding or text similarity is insufficient for pixel Dice. Dense-token access or overlapping-tile scoring needs separate validation. “Fold” prompts are hypotheses, not labels | **Moderate-to-high risk:** ViT-L is materially heavier than Hibou-B; current v1.5 packaging/documentation is less mature | Both are gated/noncommercial; CONCHv1.5 expressly requires prior approval for commercial use and derivatives | Secondary H&E semantic baseline after licensing; do not substitute zero-shot prompts for annotated evaluation |
| [GigaPath-Flash](https://huggingface.co/prov-gigapath/prov-gigapath-flash) | 2026 two-stage model: ~22M DINOv2-small ViT-S/16 tile encoder plus ~21M LongNet slide encoder; official tile API returns one 384-d vector per 224 px tile | The slide encoder compresses local evidence and is not the preferred artifact localizer. Use overlapping tile scores unless dense internal tokens are verified | **Promising size:** likely more practical than UNI2-h/Virchow2, but custom stack and MPS compatibility require a smoke test | Apache-2.0 but gated; card says checkpoints are for research/reproducibility and deployed use is out of scope | Strong lightweight H&E candidate after access/governance review; use the tile encoder first |
| [KRONOS2](https://huggingface.co/MahmoodLab/KRONOS2) | Marker-aware DINOv2 ViT-B/16 for multiplex IF/spatial proteomics; 256 px reference patches, 768-d output, 268-marker vocabulary; input includes marker names and MPP | Official interface returns one embedding per multiplex patch, not a pixel map. Start with overlapping-patch localization or add a labeled decoder; verify internal dense tokens before claiming segmentation | **Not run:** accepting gated terms for a corporate test is outside this benchmark's authority | Gated CC BY-NC-ND 4.0; ordinary commercial use and derivatives require prior approval | **Best-aligned public candidate for COMET/CosMx after written approval and channel-name audit** |

Mac assessments marked “likely,” “promising,” “plausible” or “high-risk” are engineering inferences from architecture/size and the documented execution path. They are not official MPS support claims.

## Real multiplex evidence available now

The [real COMET/CosMx proxy benchmark](MULTIPLEX_REAL_PROXY_BENCHMARK.md) ran
classical, clean-reference anomaly, and fixed hybrid methods on five public
COMET DAPI fields plus six five-channel CosMx FOVs from four slide/run groups.
Because the releases have no usable natural fold/crack masks, the known masks
come from paired controlled perturbations inserted into held-out real
backgrounds. In the final v3 LOGO evaluation, classical mean calibrated Dice
across fold/crack strata was 0.534 on COMET and 0.314 on CosMx;
clean-reference anomaly was 0.100 and 0.134, respectively. The anomaly branch
was especially weak for the controlled thin-crack cue (Dice 0.065 on COMET and
0.075 on CosMx). Dice changed by up to 0.332 at 256 px, so the result is not
resolution-robust.

These are generator-conditional proxy results, not natural-artifact Dice or a
production model ranking. Untouched activity is reported as **alert burden**,
never false-positive rate. The artifact itself is explicitly
`report_eligible=false`. Roles are group-disjoint within each LOGO fold, but
groups are reused across folds; folds are dependent, and the releases do not
declare higher-level biological independence. Natural-artifact Dice, ROC, and
FPR cannot be estimated. The
[primary v3 artifact](../artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json)
has SHA-256
`a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004`.

## Why frozen probes come before LoRA/PEFT

1. A frozen comparison isolates representation quality. The completed DINOv2,
   SigLIP2, and Hibou-B runs preserve the same split, tiling, calibration,
   token-to-pixel reconstruction, and two heads—one-class PatchKNN and a
   supervised linear token probe.
2. LoRA reduces trainable parameters, not the cost of loading the backbone or retaining activations for back-propagation. A 600M-plus encoder can still be inconvenient on MPS.
3. Without adjudicated labels, LoRA has no trustworthy task signal. Training on synthetic folds alone risks learning the generator rather than real artifacts.
4. Access terms apply before adaptation. For UNI2-h, Virchow2, CONCHv1.5 and KRONOS2, Merck should obtain written approval before feature extraction, fine-tuning, or training a derivative on model outputs.
5. PEFT becomes justified only when a frozen model underperforms on a locked, group-disjoint real test set and there are enough train/calibration labels to tune without touching that test set.

## Recommended benchmark ladder

| Stage | H&E | COMET/CosMx | Evidence gate |
|---|---|---|---|
| 0. Non-foundation controls | Classical color/texture/morphology detector | Per-channel intensity/texture/morphology detector with DAPI reported separately | Real images; locked preprocessing and native-to-analysis MPP |
| 1. Permissive generic controls | Hardened DINOv2-small and SigLIP2 Base comparisons complete | Only a declared RGB projection, explicitly labeled a projection control | Current provenance contract, same split, heads and calibration |
| 2. Runnable domain model | Hardened Hibou-B PatchKNN and linear token probe complete | Not a preferred multiplex model; optionally test the RGB projection as a negative-control transfer experiment | Repeat on an internal group-disjoint real test |
| 3. Governed external models | DINOv3 after custom-license approval; then GigaPath-Flash, UNI2-h/Virchow2, and CONCHv1.5 as governed comparators | KRONOS2 with exact marker vocabulary mapping, unmatched-marker statistics and MPP | Written access/license approval plus reproducible weight/config hashes |
| 4. Adaptation | Small decoder or LoRA on the best frozen encoder | Modality-specific decoder or LoRA on KRONOS2/internal channel-aware encoder | Adjudicated labels; fit/calibration/test isolation; synthetic data only as a documented augmentation ablation |
| 5. Product comparison | Common artifact mask/heatmap, presence, burden, confidence and abstention outputs | Same output contract, modality-specific front end | Positive-image macro Dice/IoU, pooled Dice, clean false-positive area, presence AUROC/AUPRC, per-tissue/scanner sensitivity, calibration, runtime/memory; slide/patient-level bootstrap CIs |

The selection rule should be evidence-first: choose the smallest commercially usable model whose confidence interval meets the predeclared QC operating point. A larger or newer foundation model wins only if it improves real, locked-test performance—not because it is newer, pathology-specific or called a WSI foundation model.
