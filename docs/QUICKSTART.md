# Quickstart

This path verifies a clean checkout without downloading a dataset or model.
Python 3.10 or newer is required.

## 1. Install

From the repository root:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install -e '.[dev]'
```

The core install contains NumPy/SciPy/OpenCV methods and does not require
PyTorch. Install optional capabilities only when needed:

```bash
# OME-TIFF and WSI-oriented readers
./.venv/bin/python -m pip install -e '.[dev,wsi]'

# Frozen foundation encoders
./.venv/bin/python -m pip install -e '.[dev,foundation]'

# Foundation encoders plus PEFT/LoRA engineering experiments
./.venv/bin/python -m pip install -e '.[dev,foundation,adaptation]'
```

Inspect the installed command surface with:

```bash
./.venv/bin/foldcrack-qc --help
./.venv/bin/foldcrack-qc datasets
```

Registry entries are discovery aids, not legal approval. Confirm the exact
asset version and terms through the applicable Merck process before download or
corporate use.

## 2. Run every engineering quality gate

```bash
make check PYTHON=./.venv/bin/python
```

This runs Ruff lint and format checks, the complete pytest suite, byte-code
compilation, and an isolated installed-wheel smoke test. Treat any failure as a
release blocker; do not report a benchmark from a checkout that fails this
command.

## 3. Run the deterministic end-to-end feasibility benchmark

```bash
make feasibility PYTHON=./.venv/bin/python
```

The default command exercises H&E, COMET, and CosMx adapters plus classical,
clean-reference anomaly, and hybrid paths. It writes reports, manifests, CSVs,
JSON, and overlays under `artifacts/feasibility/`.

Review at minimum:

```text
artifacts/feasibility/FEASIBILITY_REPORT.md
artifacts/feasibility/RUN_MANIFEST.json
artifacts/feasibility/evaluation_report.md
artifacts/feasibility/operational_acceptance.json
```

`engineering_smoke_test_passed=true` means the pipeline completed its software
checks. It does **not** mean scientific validation, target-domain
generalization, or operational acceptance passed.

To remove only this generated output:

```bash
make clean PYTHON=./.venv/bin/python
```

## 4. Validate internal inputs before scoring

The real-data path expects explicit identities, physical scale, channel
metadata, checksums, and group-disjoint roles. Start from
[`configs/internal_manifest.example.json`](../configs/internal_manifest.example.json),
then run:

```bash
./.venv/bin/foldcrack-qc validate-manifest \
  configs/internal_manifest.example.json --strict --json
```

Strict validation must pass before a locked evaluation. COMET/CosMx records must
declare channel names and channel axis explicitly; unresolved required channel
roles should cause abstention or review, not a guessed mapping.

## 5. Run public benchmarks only within their evidence boundary

Use the canonical reports rather than duplicating long commands here:

- [Real public H&E benchmark and exact commands](REAL_PUBLIC_BENCHMARK.md#exact-rerun-commands)
- [Real COMET/CosMx proxy benchmark and exact command](MULTIPLEX_REAL_PROXY_BENCHMARK.md#reproduce)
- [Dataset integrity and licensing audit](PUBLIC_BENCHMARK_AUDIT.md)

A development run that limits slides, skips hashes, skips complete dimension
validation, omits required provenance, or changes the audited empty-mask policy
is expected to be non-reportable. Never remove those safeguards merely to make
`report_eligible` true.

## Device behavior

The current central foundation runtime accepts only `auto`, `mps`, or `cpu`.
`auto` selects Apple MPS when the active PyTorch runtime reports it available;
otherwise it selects CPU. CUDA is not currently accepted by the main foundation
CLI, so a CUDA-only machine does **not** receive automatic CUDA acceleration.
An explicitly requested unavailable MPS device fails rather than silently
falling back.

Record both requested and resolved device in benchmark provenance. Treat a
successful MPS or LoRA smoke as hardware feasibility, not model efficacy. See
the [model support matrix](MODEL_SUPPORT.md) before choosing a command.

## Next reading

- To interpret metrics or design a locked internal study, read
  [Evaluation](EVALUATION.md) and the [Annotation guide](ANNOTATION_GUIDE.md).
- To integrate another encoder, read
  [Adding a foundation model](ADDING_FOUNDATION_MODEL.md).
- For current recommendations and decision boundaries, read the
  [Team brief](TEAM_BRIEF.md).
