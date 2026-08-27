# SigLIP2 Base Apple-MPS and LoRA engineering smoke

**Status:** passed engineering feasibility; **not** fold/crack efficacy
**Artifact:** [`siglip2_base_mps_lora.json`](../artifacts/foundation_smoke/siglip2_base_mps_lora.json)
**Artifact SHA-256:** `046f27977e35aeee794fb500df3516d38bf0d213313233d12953c06fe555f8a8`

This run used the official, ungated Apache-2.0
[`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224)
vision tower at immutable revision
`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`. The snapshot, configuration,
preprocessor, README, and safetensors file are individually SHA-256 locked. The
runner disables network access, tokens, and remote code and uses the official
`SiglipImageProcessor` bilinear, rescale, and normalization contract.

## Measured result

| Check | Observed result |
|---|---:|
| Frozen base parameters | 92,884,224 |
| CPU steady inference | 0.1220–0.1308 s |
| MPS steady inference | 0.0321–0.0324 s |
| Dense CPU/MPS maximum absolute difference | 0.0011463 |
| Global CPU/MPS maximum absolute difference | 0.000008404 |
| Minimum CPU/MPS cosine similarity | 0.9999999998 |
| Rank-4 LoRA parameters, last four q/v blocks | 49,152 |
| LoRA plus one-pixel head trainable parameters | 49,921 / 92,934,145 (0.05372%) |
| One-step MPS loss | 0.608373 → 0.561885 |
| Adapter/head update L2 | 0.159166 across 25,345 changed elements |
| MPS allocated memory, frozen inference | 371,538,688 bytes |
| MPS allocated memory after LoRA step | 379,688,960 bytes |

The agreement gate passed and the LoRA gradients, loss, and parameter update
were finite. This establishes that a frozen SigLIP2 dense-token comparator and a
small adapter are practical on this Mac. It does not show that SigLIP2 detects a
fold, detects a crack, generalizes to pathology, or transfers to COMET/CosMx.

## Reproduce

```bash
PYTHONPATH=src ./.venv/bin/python -m foldcrack_qc.siglip2_smoke \
  --snapshot-path /tmp/foldcrack_siglip2_model \
  --device mps --steady-runs 3 --lora-rank 4 \
  --output-json artifacts/foundation_smoke/siglip2_base_mps_lora.json
```

MPS must be visible to the process. The command never downloads a checkpoint;
the exact approved snapshot must already exist locally.
