# Adding a foundation model

This guide describes the current implementation, not a future plug-in system.
The scientific H&E runner already accepts an injected frozen encoder, but the
command-line providers are registered explicitly in code; there is no automatic
model discovery. Dropping a checkpoint into a directory is therefore
insufficient.

## Current seam

The H&E benchmark consumes the minimal
[`FrozenEncoder`](../src/foldcrack_qc/public_fold_benchmark.py) protocol:

```python
def encode(images, *, semantic_channels, batch_size) -> FoundationFeatures: ...
```

[`FoundationFeatures`](../src/foldcrack_qc/foundation.py) contains a batch of
global embeddings, a spatial patch-token grid, input/patch geometry, and the
semantic channel contract. The benchmark owns tiling, PatchKNN and frozen
linear heads, calibration, the locked test, metrics, and reporting.

[`DINOv2FeatureExtractor`](../src/foldcrack_qc/foundation.py) is a reusable RGB
ViT adapter. It supports:

- Hugging Face-style `last_hidden_state` outputs;
- mappings containing `x_norm_clstoken` and `x_norm_patchtokens`;
- an explicitly named global embedding for models with no prefix token; and
- an injected locked preprocessor for model-specific resize/normalization.

It validates patch geometry instead of guessing token layout. Its current input
contract is three distinct semantic channels in RGB display order; the public
H&E path passes (`red`, `green`, `blue`). The lazy
[`public-fold provider registry`](../src/foldcrack_qc/public_fold_providers.py)
currently exposes only `dinov2-hf`, `hibou-b-local`, and
`siglip2-base-local`.

## Safe process for a DINOv3-like RGB dense encoder

Use this process for an approved frozen RGB ViT that exposes spatial tokens.

### 1. Freeze the scientific question

State the modality, artifact class, intended action, output resolution, physical
patch size, group-independence unit, and comparator before loading a model. Use
`foundation_patchknn` and/or `foundation_linear_probe` for a new provider;
`dinov2_*` method names are legacy DINOv2 aliases and must not label another
encoder.

### 2. Approve and lock every executable asset

Complete model-license, security, privacy, and corporate-use review. Pin an
immutable upstream revision. Prefer a local, offline, safetensors-based loader
with `trust_remote_code=False`. Record SHA-256 for every weight and
configuration file. If approved custom source is unavoidable, pin its exact Git
commit, audit it, and prevent an unreviewed checkout from entering `sys.path`.
Never treat a model name or mutable tag as reproducible identity.

### 3. Reproduce the official input contract

Lock and test the exact color space, channel order, resize/crop/interpolation,
range, normalization mean/std, image size, patch size, and physical scale.
Do not silently reuse ImageNet/DINOv2 preprocessing for a model with different
published transforms.

### 4. Verify the output contract

Determine from the approved implementation—not by trial-and-error truncation:

- embedding dimension;
- number and ordering of CLS/register/prefix tokens;
- spatial patch-grid dimensions;
- whether a separate pooled/global output is required; and
- whether returned spatial tokens correspond one-to-one with image patches.

For a compatible model, construct the existing adapter with explicit geometry:

```python
from foldcrack_qc.foundation import DINOv2FeatureExtractor

encoder = DINOv2FeatureExtractor(
    approved_model,
    device="auto",
    image_size=APPROVED_IMAGE_SIZE,
    patch_size=APPROVED_PATCH_SIZE,
    prefix_tokens=VERIFIED_PREFIX_TOKEN_COUNT,
    model_input_name="pixel_values",  # or None for positional forward_features
    normalization_mean=APPROVED_MEAN,
    normalization_std=APPROVED_STD,
    preprocessor=locked_preprocessor,  # omit only when the built-in path is exact
)
```

If the model does not expose a supported output shape, write a small adapter
implementing `FrozenEncoder`; do not weaken `FoundationFeatures` geometry
validation. A global-only embedding is not a localization feature map.

### 5. Integrate the provider explicitly

Add a builder and distinct name to
[`PUBLIC_FOLD_ENCODER_PROVIDERS`](../src/foldcrack_qc/public_fold_providers.py).
The CLI derives its allowed provider names from that registry. If the provider
needs new arguments, add only those arguments to
[`cli.py`](../src/foldcrack_qc/cli.py) and the provider-argument protocol. Keep
model loading separate from the benchmark runner. The builder must return the
frozen encoder plus a complete identity object; it must not download or execute
remote code unless an explicit, approved policy and user action permits it.

Capture at least:

- requested model ID and immutable revision;
- resolved revision;
- weight/config/source hashes;
- loader identity and `trust_remote_code` state;
- complete preprocessing and token/output contract;
- requested/resolved device and precision; and
- frozen-evaluation and no-transductive-update assertions.

The public-fold run provenance additionally binds pre-scoring capture status,
Git commit plus dirty-diff hash (or wheel hash), Python/platform and dependency
versions, selected methods, benchmark-configuration hash, and execution
identity. Do not retrofit this information after scoring.

### 6. Add tests before a full benchmark

Add focused tests alongside
[`test_foundation.py`](../tests/test_foundation.py),
[`test_public_fold_benchmark.py`](../tests/test_public_fold_benchmark.py), and
[`test_cli.py`](../tests/test_cli.py). Required coverage includes:

- exact preprocessing against a locked golden fixture;
- CLS/register/prefix and patch-grid geometry, including rejection tests;
- frozen parameters and evaluation mode;
- finite outputs, deterministic repeated output, and microbatch equivalence;
- CPU reference behavior and bounded CPU/MPS agreement where supported;
- explicit failure for an unavailable requested device;
- no implicit network or remote-code execution;
- asset/revision/hash mismatch failures;
- complete model and run provenance, including mutation failures;
- CLI routing and model-specific argument validation; and
- fit/calibration/locked-test group isolation with no test-time fitting.

Then run:

```bash
make check PYTHON=./.venv/bin/python
```

### 7. Escalate evidence in order

1. Run a tiny deterministic encoder smoke on CPU.
2. Compare the target device against the CPU reference.
3. Run a limited-slide end-to-end development smoke and expect it to remain
   non-reportable.
4. Run the complete locked public benchmark only after provenance and integrity
   checks pass.
5. Compare on identical records, masks, heads, thresholds, and bootstrap units.
6. Re-evaluate on an adjudicated, group-disjoint target-domain cohort before any
   Merck efficacy or generalization claim.

Frozen evaluation should precede LoRA/PEFT. Fine-tuning without an adjudicated
training/calibration cohort and locked test adds parameters without creating a
valid efficacy endpoint.

## KRONOS2-like multiplex models require a separate path

Do not force a marker-aware multiplex encoder through the RGB
`DINOv2FeatureExtractor`. The current COMET/CosMx proxy preserves multichannel
arrays and explicit channel names, but its benchmark methods do not yet accept a
foundation encoder.

A production-quality multiplex integration needs a new contract similar to:

```python
def encode(
    images, *, channel_names, microns_per_pixel, batch_size
) -> MultiplexFoundationFeatures: ...
```

Before a KRONOS2-like provider is runnable, implement and verify:

1. `C x H x W` input with exact marker names, channel order, and DAPI identity;
2. a reviewed alias map to the model vocabulary, with unresolved/duplicate/
   unsupported markers causing an explicit abstention rather than remapping;
3. validated native and effective microns-per-pixel plus physical patch/stride;
4. the model's official per-marker normalization and missing-marker behavior;
5. a localization strategy—overlapping patch embeddings with audited stitching,
   or dense tokens only if the official output contract verifies them;
6. COMET- and CosMx-specific fit banks, calibration, thresholds, and reports;
7. model/source/marker-map/preprocessing/device provenance; and
8. unit, device-agreement, integration, leakage, abstention, and perturbation
   metamorphic tests.

The official interface described in the current
[foundation-model decision](PATHOLOGY_FOUNDATION_MODEL_DECISION.md) returns one
embedding per multiplex patch, not a pixel segmentation map. A localization
head is therefore required. The public multiplex proxy can verify execution,
determinism, controlled-response direction, and alert burden, but it cannot
yield natural-artifact Dice/ROC/FPR without independent natural fold/crack
labels. See the canonical
[multiplex proxy report](MULTIPLEX_REAL_PROXY_BENCHMARK.md).

## Completion checklist

A provider is supported only when all boxes are true:

- [ ] Institutional use and redistribution terms are approved.
- [ ] Model, source, configuration, preprocessing, and vocabulary are immutable
      and hash-locked.
- [ ] Input semantics and physical scale are explicit.
- [ ] Dense output geometry or patch-localization behavior is verified.
- [ ] Device behavior is explicit; no unavailable accelerator silently falls
      back.
- [ ] Required provenance is captured before scoring.
- [ ] Unit, integration, determinism, negative, and leakage tests pass.
- [ ] `make check` passes from a clean environment.
- [ ] A development smoke is clearly labeled non-reportable.
- [ ] Full results use a locked test and retain the claim limits documented in
      [Evaluation](EVALUATION.md).
