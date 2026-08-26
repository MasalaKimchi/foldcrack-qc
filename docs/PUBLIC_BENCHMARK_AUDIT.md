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
| [QUALIFAI](https://github.com/augpath/QualIFAI) | Multiplex-IF tile classification plus segmentation for fold, OOF, bubble, external artifact, and aggregate; includes COMET development evidence, not CosMx | [v2 data/models](https://zenodo.org/records/12699470) state CC BY 4.0; repository code has separate custom terms | Best public COMET-related fold candidate after archive, subject, split, and license audit; no crack class |
| [DiffusionQC](https://arxiv.org/abs/2601.12233) | Merck-led H&E diffusion anomaly heatmaps/masks for an artifact union including folds, pen, OOF, and bubbles | No official public code, weights, manifest, or reusable dataset located; manuscript license is not an implementation license | Not independently executable today; obtain internal assets and reconcile reported split counts before comparison |
| [HistoART](https://github.com/DIDSR/HistoART) | H&E patch classification for fold and other artifacts | Wrapper repository is CC0, but its upstream UNI dependency is gated/noncommercial | Patch classification only; upstream model terms still govern the weights |
| [DINOv2](https://github.com/facebookresearch/dinov2) | Generic RGB frozen CLS/spatial tokens with no pathology-QC ontology | Apache-2.0 base code/weights | Corporate-friendly custom H&E frozen kNN/PatchCore/linear baseline after internal approval; COMET/CosMx need governed semantic projections |

OME-TIFF compatibility does not establish COMET or CosMx validity. Likewise, an
anomaly heatmap is not a fold/crack semantic classifier. The benchmark contract
therefore compares generic anomaly models only against `artifact_union`.

## Real public datasets

| Dataset | Ground truth | Defensible use |
|---|---|---|
| [GrandQC manually annotated test set](https://zenodo.org/records/14039591) | Expert pixel masks from real H&E source WSIs, including fold; no crack class | Real H&E fold-localization evaluation if noncommercial terms are cleared |
| [GrandQC TCGA predicted masks](https://zenodo.org/records/14041578) | Model predictions, not independent references | Mining or triage only; never unbiased efficacy ground truth |
| [HistoArtifacts](https://zenodo.org/records/10809442) | Real H&E patch/folder labels including fold and damaged tissue; no pixel masks | Patch classification and representation probing, subject to data-rights review |
| [QUALIFAI v2](https://zenodo.org/records/12699470) | Multiplex artifact dataset/model deposit associated with COMET/CODEX/MILAN segmentation | Candidate COMET fold evaluation after inspecting exact masks, patients, and frozen splits |
| [Lunaphore COMET lung TMA](https://lunaphore.com/download-center-tma-downstream-analysis/) | Real 20-plex OME-TIFF plus DAPI; no artifact masks | Ingestion/runtime smoke only if restrictive vendor terms are approved |
| [Dryad CosMx melanoma RNA-SMI](https://datadryad.org/dataset/doi:10.5061/dryad.ksn02v7b1) | Four real CosMx slides with DAPI/protein morphology FOVs; no fold/crack masks | CC0 domain/ingestion testing and a source for new expert annotations; not efficacy as released |

No verified public pixelwise crack benchmark was found for H&E, COMET, or CosMx.
Fold, tissue tear, tissue damage, knife line, section separation, acquisition seam,
and coverslip/glass crack must not be silently treated as equivalent targets.

## Feasible comparison matrix

| Target/evidence | H&E | COMET | CosMx |
|---|---|---|---|
| Real public fold localization | GrandQC, if license-cleared | QUALIFAI candidate after archive/split audit | None |
| Real public crack localization | None verified | None verified | None verified |
| Patch classification | HistoArtifacts, HistoART | QUALIFAI tile stage | None identified |
| Unlabeled domain testing | Public pathology WSIs | Vendor example, if terms allow | Dryad CC0 morphology FOVs |
| Corporate-controlled comparison | Classical + frozen DINOv2 + supervised/LoRA after annotation | Classical + QUALIFAI/internal head after annotation | Classical + explicit morphology projection/internal head after annotation |

Synthetic perturbations remain useful for regression, stress testing, and perhaps
pretraining, but never replace a real held-out specimen-level test. The most
credible cross-modal benchmark is therefore a shared evaluation protocol with
separate modality cohorts—not a claim that one public dataset or model already
solves all three.
