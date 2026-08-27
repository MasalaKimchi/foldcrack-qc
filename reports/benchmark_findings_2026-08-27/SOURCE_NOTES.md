# Source notes — Fold and Crack Artifact QC Across H&E, COMET, and CosMx

## Delivery contract

- Primary audience: technical deep-learning, computational pathology, and QC stakeholders.
- Primary mode: native interactive technical report (`artifact.json`).
- Durable manuscript source: `REPORT.md`.
- Supporting exports: vector SVG/PDF and 300-dpi PNG figures.
- Decision frame: internal feasibility, **share with caveats**; not production validation.

## Required-section map

| Requirement | Location |
| --- | --- |
| Technical summary | `REPORT.md` opening and `technical_summary` report block |
| Findings with evidence and comparisons | Sections 1–3 and 7; native charts/tables and Figures 1–7 |
| Scope, data, metric definitions | Sections 1, 2, and 5 |
| Methodology | H&E and multiplex Methods text; report `methodology` block |
| Limitations and uncertainty | H&E validity, multiplex caveats, Limitations section |
| Recommended next steps | Prioritized implementation plan |
| Questions that can change the decision | Native report `further_questions` block |

## Figure map

| Figure | SVG | PDF | 300-dpi PNG |
| --- | --- | --- | --- |
| Figure 1 — H&E locked test | figures/figure1_he_locked_test.svg | figures/figure1_he_locked_test.pdf | figures/figure1_he_locked_test.png |
| Figure 2 — organ heterogeneity | figures/figure2_he_organ_heatmap.svg | figures/figure2_he_organ_heatmap.pdf | figures/figure2_he_organ_heatmap.png |
| Figure 3 — paired differences | figures/figure3_he_paired_differences.svg | figures/figure3_he_paired_differences.pdf | figures/figure3_he_paired_differences.png |
| Figure 4 — multiplex proxy | figures/figure4_multiplex_proxy.svg | figures/figure4_multiplex_proxy.pdf | figures/figure4_multiplex_proxy.png |
| Figure 5 — proxy resolution sensitivity | figures/figure5_proxy_resolution_sensitivity.svg | figures/figure5_proxy_resolution_sensitivity.pdf | figures/figure5_proxy_resolution_sensitivity.png |
| Figure 6 — evidence scope | figures/figure6_evidence_scope.svg | figures/figure6_evidence_scope.pdf | figures/figure6_evidence_scope.png |
| Figure 7 — qualitative H&E audit | figures/figure7_he_qualitative.svg | figures/figure7_he_qualitative.pdf | figures/figure7_he_qualitative.png |

## Evidence and omission policy

- Reportable efficacy is restricted to the four hardened schema-v1.2 H&E JSON reports plus the paired-comparison artifact.
- The 896-pixel COMET/CosMx LOGO (leave-one-group-out) artifact is retained only as nonreportable proxy evidence. Its matched 256-pixel counterpart is independently loaded, hashed, and used only for resolution sensitivity.
- Foundation smokes support MPS/CPU parity and one-step LoRA execution only.
- Figure 7 uses algorithmically selected whole fields and a separately validated deterministic shallow-head refit because the hardened artifacts did not retain spatial maps or fitted-probe parameters; calibration was not rerun, counts and calls matched, and scores were within recorded tolerances.
- Earlier all-synthetic feasibility results were omitted from rankings because they establish wiring, not external validity.
- DINOv3, KRONOS2, UNI2-h, CONCHv1.5, HistoQC, GrandQC, DiffusionQC, PaDiM, full PatchCore, U-Net, and SegFormer are explicitly labeled not run.
- No p-values, superiority claims, pooled multimodality score, natural multiplex FPR, or WSI/deployment claim is presented.

## Validation and QA policy

- Independent recomputation: macro Dice, micro Dice, and clean-field predicted fraction from per-field counts.
- Cohort identity: exact ordered H&E field/domain rows asserted equal for all seven methods.
- Leakage interpretation: supplied source-slide-disjoint split verified; patient/block identity unavailable.
- Uncertainty: source-slide-cluster bootstrap intervals are conditional on the fixed dataset/configuration.
- Visualization: no 3D, no dual axes, direct labels where feasible, redundant marker/fill coding, units and focused/log scales disclosed in subtitles/captions; Figure 7 combines audited raster fields with vector labels and legends.
- Interactive artifact: validated against the Data Analytics report schema and rendered for visual inspection after generation.

## Input artifact SHA-256 hashes

| Path | SHA-256 |
| --- | --- |
| artifacts/foundation_smoke/foundation_smoke.json | 1e00a80513770222fc7323fc5c4ec211662160c7cc15bdaa9897d348a296189f |
| artifacts/foundation_smoke/siglip2_base_mps_lora.json | 046f27977e35aeee794fb500df3516d38bf0d213313233d12953c06fe555f8a8 |
| artifacts/multiplex_proxy/real_public_logo_cv_256_v3.json | a96edf1a02878d98d7ffd6f6d5a2acd3bc482a5a8a10956ebc7a18679579da5c |
| artifacts/multiplex_proxy/real_public_logo_cv_896_v3.json | a506e3e2bc53732b9470c9c6b528bb35d095bf44af85070394fecd7c8a0e4004 |
| artifacts/public_fold/classical_hardened_v1_2.json | a23d1836cbda7e4a1835068d485c03463d178a084dcfe55266fc2183f96bcd19 |
| artifacts/public_fold/dinov2_hardened_v1_2.json | 5846a4edf7b7f8a882c5d211d37934bc0e3f15c18ab906cde5951e98f6b47fbd |
| artifacts/public_fold/hardened_all_methods_paired_comparison_v1.json | 9772cc30afa63c62befbc447fc8228fbc3937765a018de76ea58a28f5d021ac2 |
| artifacts/public_fold/hibou_hardened_v1_2.json | 43d46447c7bcfd971691c97a3a99d10d8a78dddf4486e269f2f5ed8f1173301d |
| artifacts/public_fold/siglip2_hardened_v1_2.json | 4a8cbd2f45a3023a6c1313daf16be05913256b1381d5f83023bd9120e93b2596 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case1_Brain_Fold_-20260410140426972_f43adca74814.png | f43adca748149a777264db5a061f747cbbbdbae55fb6f67abdd7129b2a0805b7 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case2_Kidney__Fold_-20260409155008912_19c6e416eb54.png | 19c6e416eb54f484b3370e2ce2170e60e6202b84fe9a23bee1f8f52c45565a73 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case3_Liver__Fold_-20260406110823972_c707cacc6327.png | c707cacc632747cbe7917a6cf414f1873638bc27c0752ea0adc89ab09cfa0f74 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case4_Small_Intestine_Fold_-20260413150334899_be5b63afe2f5.png | be5b63afe2f55a902ef31b896486652c15dc595b54e2939db7a432358ceda408 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case5_Testis__Fold_-20260406101118662_bc52f9b19b17.png | bc52f9b19b178b8ce7889950775b44995a23f5cc510e3c62cbcc38d9991fc6d2 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case6_Brain_Clean-20260423152633332_1d20edb52790.png | 1d20edb52790688bd52a9fd7074d76a0c3dde78e6031600e9559d7186fd2f579 |
| reports/benchmark_findings_2026-08-27/qualitative_cache/case7_Kidney_Clean-20260420151005412_9e5cacdfce6b.png | 9e5cacdfce6b734dc5aebad0b56a4270283e20c77b7e7fa5c18f9e7ae79e16ce |
| reports/benchmark_findings_2026-08-27/qualitative_checks.json | 9737b77b3bf981aa9f2fa81b2ae4e52d7c580663c3d6a1d050f8ef4111b928e5 |
