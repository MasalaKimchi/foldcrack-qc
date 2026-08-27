# Documentation map

This directory separates operating instructions, design decisions, and evidence.
Start with the shortest document that answers the question; do not copy numerical
results from secondary summaries when a canonical benchmark report exists.

## Start here

- [Quickstart](QUICKSTART.md) — clean installation, quality gates, and the
  deterministic end-to-end smoke run.
- [Team brief](TEAM_BRIEF.md) — current recommendation, demonstrated scope, and
  decisions needed from the Merck team.
- [AI system specification](AI-SPEC.md) — intended architecture, guardrails,
  evaluation strategy, and production-monitoring contract.
- [Annotation guide](ANNOTATION_GUIDE.md) — fold/crack taxonomy and reviewer
  workflow for building an internal reference set.

## Canonical evidence

- [Real public H&E benchmark](REAL_PUBLIC_BENCHMARK.md) — canonical numerical
  results, artifact hashes, exact rerun commands, and bounded claim scope for
  the public tissue-fold cohort.
- [Real COMET/CosMx proxy benchmark](MULTIPLEX_REAL_PROXY_BENCHMARK.md) —
  canonical numerical results for controlled perturbations on real public
  multiplex backgrounds.
- [Public benchmark audit](PUBLIC_BENCHMARK_AUDIT.md) — dataset provenance,
  integrity, label availability, licensing caveats, and model-access findings.
- [Evaluation contract](EVALUATION.md) — metrics, split rules, uncertainty,
  abstention, and proposed decision gates.

The H&E benchmark supports only a bounded claim about tissue-fold localization
on one external 10x veterinary teaching-slide field cohort. It does not validate
cracks, human clinical WSI, Merck data, scanners/sites, COMET, or CosMx. The
COMET/CosMx benchmark has no independently usable natural-artifact masks; it is
a controlled-perturbation proxy and is intentionally not natural-artifact
efficacy evidence. Synthetic feasibility output verifies software behavior, not
clinical or assay performance.

## Foundation models

- [Model support matrix](MODEL_SUPPORT.md) — what is runnable now, devices,
  input contracts, and evidence status.
- [Adding a foundation model](ADDING_FOUNDATION_MODEL.md) — the current encoder
  seam, safe RGB dense-token integration, required provenance/tests, and the
  separate work needed for a KRONOS2-like multiplex encoder.
- [Pathology foundation-model decision](PATHOLOGY_FOUNDATION_MODEL_DECISION.md)
  — technical, evidence, and licensing comparison of candidate encoders.
- [DINOv2 feasibility](FOUNDATION_FEASIBILITY.md) — frozen-feature and optional
  LoRA engineering smoke evidence.
- [SigLIP2 MPS/LoRA smoke](SIGLIP2_MPS_LORA_SMOKE.md) — hash-locked hardware and
  adaptation smoke evidence.

Model smoke tests answer whether an implementation can execute and agree across
tested devices. They do not establish artifact-detection efficacy. A new model
is not supported merely because its weights can be loaded.
