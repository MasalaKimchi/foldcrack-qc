# Data handling and internal manifests

This repository contains no patient images or real-data manifests. Everything
under `data/`, plus common WSI, microscopy, array, and medical-image formats, is
ignored by Git. Only this policy file and the opaque schema example at
`configs/internal_manifest.example.json` belong in source control.

Never put direct identifiers, PHI, confidential study names, private paths, or
real checksums in committed examples. Public availability also does not imply
corporate reuse permission; data, model, and code licenses require separate
review.

## Validation modes

Exploratory validation checks required fields, readability, finite image values,
lossless binary masks, dimensions, duplicate sample IDs, and group/source split
leakage. Missing lock-readiness fields are warnings:

```bash
PYTHONPATH=src python -m foldcrack_qc validate-manifest /secure/manifest.json --json
```

Strict mode is the precondition for a locked, claim-bearing evaluation. It also
requires and verifies explicit positive `pixel_size_um`, a controlled `split`
and `cohort`, valid/ignore coverage, SHA-256 values, explicit fluorescence
channel names/axis, and a resolvable nuclear channel:

```bash
PYTHONPATH=src python -m foldcrack_qc validate-manifest /secure/manifest.json --strict --json
```

Strict validation hashes files and decoded image content. This is deliberately
I/O-intensive on WSI and should run once while freezing a read-only cohort, not
inside every inference loop.

## Record contract

JSON may be a record list or `{ "samples": [...] }`; JSONL/NDJSON uses one
record per line. Relative paths resolve from the manifest directory.

Required in both modes:

- `sample_id`, `modality`, `image_path`, `split`;
- at least one opaque `patient_id`, `block_id`, `slide_id`, or `run_id`.

Required for strict evaluation:

- `pixel_size_um` as a positive scalar or `[y_um, x_um]`;
- `cohort` from the vocabulary below;
- `valid_mask_path` or `ignore_mask_path`;
- `image_sha256` and the corresponding `*_sha256` for every referenced mask;
- COMET/CosMx `channel_names` and `channel_axis`, including a DAPI/Hoechst-like
  channel that resolves to the nuclear semantic role.

Optional source and labels:

- `source_id` for repeated-export detection;
- `fold_mask_path`, `crack_mask_path`, `tissue_mask_path`, `valid_mask_path`,
  `ignore_mask_path` containing only finite `0`/`1` values;
- `fold_instance_mask_path`, `crack_instance_mask_path` containing non-negative
  integer instance IDs. These label maps are returned in sample metadata because
  the current canonical `QCSample.masks` contract is binary.

Controlled `split` values are `development`, `train`, `validation`, `test`, and
`locked_test`. Controlled `cohort` values are `development`, `prevalence`,
`enriched_challenge`, `external_generalization`, `missing_degraded_input`, and
`downstream_impact`.

The validator prevents patient, block, slide, run, source path, source ID, file
digest, and decoded image content from crossing splits. Diagnostics deliberately
omit every identifier and path except `sample_id`.

Split at the highest correlated patient/block/slide/run level before tiling. Keep
scanner/instrument, software, panel, channel mapping, site, batch/time, specimen
preparation, annotation/adjudication provenance, and license approval in the
secure manifest or its governed companion metadata.

Run `python -m foldcrack_qc datasets` to print the curated public-resource
registry and its limitations.
