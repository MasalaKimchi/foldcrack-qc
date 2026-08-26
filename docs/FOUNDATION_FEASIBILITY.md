# Frozen DINOv2 / Apple MPS feasibility result

Status: **engineering execution passed; scientific efficacy not evaluated**  
Execution date: 2026-08-26  
Purpose: prove that a pinned frozen vision encoder and small LoRA adaptation can
execute locally before any Merck data are exposed to the pipeline.

## Executed configuration

| Item | Value |
|---|---|
| Model | `facebook/dinov2-small` |
| Exact revision | `ed25f3a31f01632728cabb09d1542f84ab7b0056` |
| Weight file | `model.safetensors`, 88,249,960 bytes |
| Weight SHA-256 | `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1` |
| Runtime | PyTorch 2.13.0, Apple MPS |
| Input | two deterministic 224-by-224 RGB engineering patches |
| Frozen output | CLS `(2, 384)` plus spatial tokens `(2, 16, 16, 384)` |
| Adaptation smoke | BF16 rank-4 LoRA, query/value in blocks 8–11, tiny binary head |

The reusable command ran in offline mode from an explicit cache with
`trust_remote_code=False` and no authentication token. The report hashes the
actual cached weights and both model configuration files before execution, and
records the Python, platform, and package versions.

## Measured result

| Check | Result |
|---|---:|
| CPU steady median, two patches | 0.0378 s |
| MPS steady median, two patches | 0.0167 s |
| CPU/MPS maximum absolute embedding difference | 0.0001787 |
| CLS cosine similarity | 0.999999999995 |
| Spatial-token cosine similarity | 0.999999999993 |
| CPU/MPS parity gate | Passed (`max error <= 0.001`, both cosines `>= 0.9999`) |
| LoRA one-step loss | 0.5700, finite |
| LoRA step duration | 0.0987 s |
| Trainable parameters including tiny head | 24,961 / 22,081,537 (0.1130%) |
| LoRA model-weight delta L2 | 0.1111, nonzero |
| MPS current allocation after LoRA step | 125,669,376 bytes |
| MPS recommended maximum working-set size | 14,302,248,960 bytes |

These timings are a tiny deterministic smoke test, not a WSI throughput study.
They show that frozen inference and a narrowly scoped PEFT update are technically
practical on this Mac without full fine-tuning.

## What this result does and does not answer

It answers:

- Can the exact model be loaded reproducibly from a local cache? **Yes.**
- Does MPS return finite global and spatial features close to CPU? **Yes.**
- Can a rank-4 LoRA update run in BF16 within local memory? **Yes.**

It does not answer:

- whether DINOv2 detects folds or cracks on real H&E;
- whether a generic anomaly map distinguishes fold from crack;
- whether RGB DINOv2 generalizes to COMET or CosMx projections;
- whether LoRA improves any locked metric;
- whether the method is safe or useful in the Merck workflow.

Those questions require disjoint reviewed-clean fit data, calibration data, and
an adjudicated locked test set. The real benchmark contract is validated with:

```bash
PYTHONPATH=src python3 -m foldcrack_qc validate-benchmark \
  configs/benchmark.real.example.json --require-report-eligible --json
```

The example correctly remains report-blocked until real cohort records and gated
method assets are supplied. A zero Dice from an uncalibrated anomaly baseline is
possible when no predicted pixel crosses the threshold, when its generic anomaly
score does not align with a semantic target, or when the representation/projection
is mismatched. It is evidence that the configuration failed that synthetic case,
not proof of a code crash and not a valid estimate of real-world performance.
