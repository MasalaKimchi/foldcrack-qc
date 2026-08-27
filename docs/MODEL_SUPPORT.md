# Model support matrix

“Runnable” means the repository has an explicit loader/adapter and benchmark
path. It does not mean the method is validated for Merck data or approved for
corporate use. Numerical results live only in the canonical benchmark reports.

## Current implementation

| Method or encoder | Input/output contract | Runtime path | Evidence status |
|---|---|---|---|
| Classical, clean-reference anomaly, hybrid | Modality-aware H&E or multichannel COMET/CosMx adapters; pixel score/mask | Core install; CPU | Deterministic synthetic engineering benchmark. Classical/anomaly/hybrid also run in the public multiplex controlled-perturbation proxy. |
| DINOv2-small | RGB; frozen CLS plus dense patch tokens | `dinov2-hf`; pinned Hugging Face revision; PatchKNN and frozen linear heads | Completed hardened real public H&E tissue-fold benchmark. This is a generic RGB control, not COMET/CosMx or crack evidence. |
| Hibou-B | RGB H&E; frozen CLS plus dense patch tokens | `hibou-b-local`; approved local source and weights with required source-commit and weight-hash locks | Completed the same hardened public H&E benchmark; current strongest point estimate is cohort-bounded and not a universal superiority claim. |
| SigLIP2 Base | RGB; locked processor, pooled output plus dense patch tokens | `siglip2-base-local`; hash-locked local snapshot; PatchKNN and frozen linear heads | Completed the same hardened public H&E benchmark plus an MPS/LoRA engineering smoke. Not multiplex or crack evidence. |
| DINOv3 | Expected RGB dense ViT path, exact contract to be verified from approved checkpoint/code | **Not implemented.** Existing `FrozenEncoder` and dense-token adapter may be reusable after provider, preprocessing, prefix/register geometry, and provenance work. | Not run. Access/terms and model assets require the applicable institutional approval before integration. |
| KRONOS2 | Marker-aware multiplex patches with marker names and MPP; documented interface yields a patch embedding | **Not implemented.** Requires a new multichannel encoder contract and localization path; it is not compatible with the current RGB adapter. | Not run. Public COMET/CosMx data currently provide proxy backgrounds, not independent natural fold/crack masks. |
| Other pathology encoders (for example UNI2-h or CONCH-family models) | Model-specific; dense-token availability and preprocessing must be verified | Not integrated or automatically discovered | Candidate analysis only. See the [foundation-model decision](PATHOLOGY_FOUNDATION_MODEL_DECISION.md). |

The public H&E CLI currently recognizes exactly:

```text
dinov2-hf
hibou-b-local
siglip2-base-local
```

These names come from the explicit lazy registry in
[`public_fold_providers.py`](../src/foldcrack_qc/public_fold_providers.py); model
providers are not discovered from arbitrary local files.

Use encoder-neutral method IDs `foundation_patchknn` and
`foundation_linear_probe` for Hibou-B, SigLIP2, or any future provider. The
`dinov2_patchknn` and `dinov2_linear_probe` IDs are retained DINOv2 aliases and
must not be used to mislabel another encoder.

## Device support

| Path | Accepted selection | Automatic behavior |
|---|---|---|
| Main frozen foundation runtime and public H&E benchmark | `auto`, `mps`, `cpu` | `auto` uses MPS when PyTorch reports it available; otherwise CPU. Explicit unavailable MPS fails. |
| DINOv2 foundation smoke | `auto`, `mps`, `cpu` | Same MPS-to-CPU automatic policy. |
| SigLIP2 LoRA smoke | explicit `mps` or `cpu` | No `auto`; default is MPS. |
| Core classical/anomaly/hybrid paths | CPU | No accelerator selection. |

CUDA is not accepted by the current central foundation selector or CLI. On a
CUDA-only host, `auto` resolves to CPU. Do not describe the repository as
automatically CUDA-enabled until one shared CUDA/MPS/CPU resolver,
synchronization policy, diagnostics, precision policy, and device-parity tests
are implemented.

## Evidence boundaries

- The [real public H&E report](REAL_PUBLIC_BENCHMARK.md) is the canonical source
  for matched method results and exact rerun commands. It covers tissue fold on
  external 10x veterinary teaching-slide fields—not cracks, human clinical WSI,
  target scanners/sites, COMET, or CosMx.
- The [real COMET/CosMx proxy report](MULTIPLEX_REAL_PROXY_BENCHMARK.md) measures
  recovery of controlled perturbations on real public backgrounds. It is
  intentionally non-reportable as natural-artifact efficacy because independent
  natural fold/crack masks are unavailable.
- The [DINOv2](FOUNDATION_FEASIBILITY.md) and
  [SigLIP2](SIGLIP2_MPS_LORA_SMOKE.md) smoke reports establish bounded runtime,
  determinism/device-agreement, or adaptation feasibility—not operational
  accuracy.
- License status in the resource registry is informational. The exact model and
  dataset version still require the applicable Merck legal/security approval.

## Adding or promoting a model

Follow [Adding a foundation model](ADDING_FOUNDATION_MODEL.md). A model moves
from candidate to runnable only after its approved immutable assets, exact
preprocessing/output geometry, loader, CLI route, provenance, negative tests,
and clean quality-gate run exist. It moves from runnable to evidence-bearing
only after a leakage-free calibrated evaluation on a locked reference set.
