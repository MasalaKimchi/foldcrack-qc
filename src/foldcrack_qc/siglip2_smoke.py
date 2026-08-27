"""Offline SigLIP2 CPU/MPS and optional PEFT-LoRA engineering smoke.

This runner deliberately does not evaluate fold/crack efficacy.  It verifies
that one exact, public, Apache-2.0 SigLIP2 checkpoint can be reproduced from a
local hash-locked snapshot, that its global and dense features agree between
CPU and a requested device, and that a tightly scoped LoRA adapter can complete
one finite optimization step.  Network access, tokens, and remote code are not
supported by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._torch_runtime import synchronize_torch_device
from .foundation import (
    SIGLIP2_BASE_MEAN,
    SIGLIP2_BASE_STD,
    DINOv2FeatureExtractor,
    FoundationFeatures,
    LoadedLocalFoundationModel,
    foundation_runtime_diagnostics,
    load_local_siglip2_base_vision,
    select_torch_device,
)
from .foundation_smoke import deterministic_smoke_patches

SIGLIP2_BASE_MODEL_ID = "google/siglip2-base-patch16-224"
SIGLIP2_BASE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
SIGLIP2_BASE_ASSET_SHA256 = {
    "README.md": "39ac3705d62af9ffa1a14675b8ccb220a75f2d81acd530e564a3b1e3dfe418d8",
    "config.json": "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
    "model.safetensors": (
        "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b"
    ),
    "preprocessor_config.json": (
        "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc"
    ),
}
_DEPENDENCY_GUIDANCE = (
    "Install the approved foundation and adaptation dependencies: torch, "
    "transformers, safetensors, peft, and accelerate."
)
_SEED = 17_291


class SigLIP2SmokeDependencyError(RuntimeError):
    """Raised when a requested optional runtime dependency is unavailable."""


@dataclass(frozen=True)
class SigLIP2SmokeConfig:
    """Immutable execution contract for the locked local SigLIP2 smoke."""

    snapshot_path: str | Path
    device: str = "mps"
    steady_runs: int = 3
    lora_rank: int | None = None
    max_device_abs_error: float = 5e-3
    min_device_cosine_similarity: float = 0.99999

    def __post_init__(self) -> None:
        snapshot = Path(self.snapshot_path).expanduser()
        device = str(self.device).strip().lower()
        if not str(snapshot):
            raise ValueError("snapshot_path must be explicit and non-empty")
        if device not in {"cpu", "mps"}:
            raise ValueError("device must be exactly 'cpu' or 'mps'")
        if int(self.steady_runs) <= 0:
            raise ValueError("steady_runs must be positive")
        if self.lora_rank not in {None, 4, 8}:
            raise ValueError("lora_rank must be None, 4, or 8")
        maximum_error = float(self.max_device_abs_error)
        minimum_cosine = float(self.min_device_cosine_similarity)
        if not math.isfinite(maximum_error) or not 0.0 < maximum_error < 1.0:
            raise ValueError("max_device_abs_error must lie in (0, 1)")
        if not math.isfinite(minimum_cosine) or not 0.0 < minimum_cosine <= 1.0:
            raise ValueError("min_device_cosine_similarity must lie in (0, 1]")
        object.__setattr__(self, "snapshot_path", snapshot)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "steady_runs", int(self.steady_runs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_path": str(self.snapshot_path),
            "device": self.device,
            "steady_runs": self.steady_runs,
            "lora_rank": self.lora_rank,
            "max_device_abs_error": float(self.max_device_abs_error),
            "min_device_cosine_similarity": float(self.min_device_cosine_similarity),
            "seed": _SEED,
            "offline": True,
        }


def deterministic_semantic_rgb_patches(image_size: int = 224) -> np.ndarray:
    """Return deterministic clean/proxy-artifact patches with explicit RGB semantics."""

    return deterministic_smoke_patches(image_size)


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_locked_provenance(
    provenance: Mapping[str, Any], snapshot_path: Path
) -> dict[str, Any]:
    """Fail closed unless loader evidence matches the exact public checkpoint lock."""

    if provenance.get("id") != SIGLIP2_BASE_MODEL_ID:
        raise RuntimeError("SigLIP2 loader returned an unexpected model identity")
    source = provenance.get("source")
    if not isinstance(source, Mapping) or (
        source.get("revision") != SIGLIP2_BASE_REVISION
    ):
        raise RuntimeError("SigLIP2 loader returned an unexpected immutable revision")
    license_record = provenance.get("license")
    if not isinstance(license_record, Mapping) or (
        license_record.get("spdx") != "Apache-2.0"
    ):
        raise RuntimeError("SigLIP2 loader did not preserve Apache-2.0 evidence")
    if provenance.get("trust_remote_code") is not False:
        raise RuntimeError("SigLIP2 loader must disable remote code")
    if provenance.get("token_used") is not False:
        raise RuntimeError("SigLIP2 loader must not use a token")
    if provenance.get("network_access_allowed") is not False:
        raise RuntimeError("SigLIP2 loader must remain offline")

    input_record = provenance.get("input")
    if not isinstance(input_record, Mapping) or (
        input_record.get("processor") != "transformers.SiglipImageProcessor"
        or input_record.get("resample") != 2
        or input_record.get("resample_semantics") != "PIL.Image.Resampling.BILINEAR"
        or float(input_record.get("rescale_factor", -1.0)) != 1.0 / 255.0
        or input_record.get("normalization_mean") != list(SIGLIP2_BASE_MEAN)
        or input_record.get("normalization_std") != list(SIGLIP2_BASE_STD)
    ):
        raise RuntimeError("SigLIP2 loader returned an unexpected processor contract")

    assets = provenance.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != set(SIGLIP2_BASE_ASSET_SHA256):
        raise RuntimeError("SigLIP2 loader returned an incomplete snapshot asset lock")
    resolved_snapshot = snapshot_path.resolve()
    for filename, expected_sha256 in SIGLIP2_BASE_ASSET_SHA256.items():
        record = assets.get(filename)
        if not isinstance(record, Mapping) or (record.get("sha256") != expected_sha256):
            raise RuntimeError(f"SigLIP2 asset lock mismatch: {filename}")
        recorded_path = Path(str(record.get("path", "")))
        if recorded_path.resolve() != (resolved_snapshot / filename).resolve():
            raise RuntimeError(f"SigLIP2 asset path mismatch: {filename}")

    # A JSON round trip both copies the mapping and rejects non-serializable data.
    return json.loads(json.dumps(dict(provenance), allow_nan=False, sort_keys=True))


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SigLIP2SmokeDependencyError(
            f"PyTorch is unavailable. {_DEPENDENCY_GUIDANCE}"
        ) from error
    return torch


def _runtime_versions(torch: Any) -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "opencv-python",
        "torch",
        "transformers",
        "safetensors",
        "peft",
        "accelerate",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": packages,
        "torch_default_dtype": str(torch.get_default_dtype()),
    }


def _parameter_stats(model: Any) -> dict[str, Any]:
    by_dtype: dict[str, dict[str, int]] = {}
    total_count = 0
    trainable_count = 0
    parameter_bytes = 0
    for parameter in model.parameters():
        count = int(parameter.numel())
        size_bytes = count * int(parameter.element_size())
        dtype = str(parameter.dtype)
        record = by_dtype.setdefault(dtype, {"count": 0, "bytes": 0})
        record["count"] += count
        record["bytes"] += size_bytes
        total_count += count
        parameter_bytes += size_bytes
        if bool(parameter.requires_grad):
            trainable_count += count
    buffer_count = 0
    buffer_bytes = 0
    for buffer in model.buffers():
        count = int(buffer.numel())
        buffer_count += count
        buffer_bytes += count * int(buffer.element_size())
    return {
        "parameter_count": total_count,
        "trainable_parameter_count": trainable_count,
        "parameter_bytes": parameter_bytes,
        "buffer_count": buffer_count,
        "buffer_bytes": buffer_bytes,
        "tensor_bytes_total": parameter_bytes + buffer_bytes,
        "parameters_by_dtype": by_dtype,
    }


def _make_extractor(
    model: Any,
    device: str,
    torch: Any,
    preprocessor: Callable[..., np.ndarray],
) -> DINOv2FeatureExtractor:
    return DINOv2FeatureExtractor(
        model,
        device=device,
        image_size=224,
        patch_size=16,
        prefix_tokens=0,
        model_input_name="pixel_values",
        global_embedding_name="pooler_output",
        normalization_mean=SIGLIP2_BASE_MEAN,
        normalization_std=SIGLIP2_BASE_STD,
        preprocessor=preprocessor,
        torch_module=torch,
    )


def _locked_preprocess(
    preprocessor: Callable[..., np.ndarray], patches: np.ndarray
) -> np.ndarray:
    values = np.asarray(
        preprocessor(
            patches,
            semantic_channels=("red", "green", "blue"),
        ),
        dtype=np.float32,
    )
    if values.shape != (2, 3, 224, 224) or not np.isfinite(values).all():
        raise RuntimeError("Locked SigLIP2 processor returned invalid pixel values")
    return np.ascontiguousarray(values)


def _processor_record(
    preprocessor: Callable[..., np.ndarray],
    provenance: Mapping[str, Any],
    processed: np.ndarray,
) -> dict[str, Any]:
    input_record = provenance["input"]
    assert isinstance(input_record, Mapping)
    processor_asset = provenance["assets"]["preprocessor_config.json"]
    return {
        "official_class": input_record["processor"],
        "locked_wrapper_module": str(getattr(preprocessor, "__module__", "unknown")),
        "locked_wrapper_name": str(
            getattr(preprocessor, "__qualname__", type(preprocessor).__qualname__)
        ),
        "configuration": {
            "do_resize": True,
            "size": {"height": 224, "width": 224},
            "resample": input_record["resample"],
            "resample_semantics": input_record["resample_semantics"],
            "source_dtype_boundary": input_record["source_dtype_boundary"],
            "do_rescale": True,
            "rescale_factor": input_record["rescale_factor"],
            "do_normalize": True,
            "image_mean": input_record["normalization_mean"],
            "image_std": input_record["normalization_std"],
        },
        "configuration_asset": dict(processor_asset),
        "output_shape": [int(value) for value in processed.shape],
        "output_dtype": str(processed.dtype),
        "output_sha256": _array_sha256(processed),
        "output_finite": True,
    }


def _time_encode(
    extractor: DINOv2FeatureExtractor,
    patches: np.ndarray,
    *,
    steady_runs: int,
) -> tuple[FoundationFeatures, dict[str, Any]]:
    torch = extractor._torch
    synchronize_torch_device(torch, extractor.device, require_available=True)
    started = time.perf_counter()
    features = extractor.encode(patches, batch_size=2)
    synchronize_torch_device(torch, extractor.device, require_available=True)
    warm_seconds = time.perf_counter() - started

    durations: list[float] = []
    output_hashes: list[str] = []
    for _ in range(steady_runs):
        synchronize_torch_device(torch, extractor.device, require_available=True)
        started = time.perf_counter()
        features = extractor.encode(patches, batch_size=2)
        synchronize_torch_device(torch, extractor.device, require_available=True)
        durations.append(time.perf_counter() - started)
        output_hashes.append(
            _array_sha256(features.cls_embedding)
            + ":"
            + _array_sha256(features.patch_grid)
        )
    return features, {
        "scope": "two_patch_exact_preprocess_and_frozen_vision_forward",
        "warm_seconds": float(warm_seconds),
        "steady_runs": steady_runs,
        "steady_seconds": [float(value) for value in durations],
        "steady_min_seconds": float(min(durations)),
        "steady_median_seconds": float(statistics.median(durations)),
        "steady_mean_seconds": float(statistics.fmean(durations)),
        "steady_output_hashes": output_hashes,
        "steady_outputs_bitwise_repeatable": len(set(output_hashes)) == 1,
    }


def _tensor_summary(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array)
    if value.dtype != np.float32:
        raise RuntimeError("SigLIP2 frozen outputs must be materialized as float32")
    if not np.isfinite(value).all():
        raise RuntimeError("SigLIP2 frozen inference produced non-finite outputs")
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "finite": True,
        "sha256": _array_sha256(value),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
        "l2_norm": float(np.linalg.norm(value.astype(np.float64))),
    }


def _feature_summary(features: FoundationFeatures) -> dict[str, Any]:
    return {
        "global_pooler_output": _tensor_summary(features.cls_embedding),
        "dense_last_hidden_state": _tensor_summary(features.patch_grid),
        "input_size": [int(value) for value in features.input_size],
        "patch_size": [int(value) for value in features.patch_size],
        "grid_shape": [int(value) for value in features.grid_shape],
    }


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    value = float(np.dot(left, right) / denominator)
    if not math.isfinite(value):
        raise RuntimeError("CPU/device cosine agreement is non-finite")
    return float(np.clip(value, -1.0, 1.0))


def _feature_agreement(
    cpu: FoundationFeatures, requested: FoundationFeatures
) -> dict[str, float]:
    if cpu.cls_embedding.shape != requested.cls_embedding.shape:
        raise RuntimeError("CPU/device global output shapes differ")
    if cpu.patch_grid.shape != requested.patch_grid.shape:
        raise RuntimeError("CPU/device dense output shapes differ")
    global_delta = np.abs(cpu.cls_embedding - requested.cls_embedding)
    dense_delta = np.abs(cpu.patch_grid - requested.patch_grid)
    if not np.isfinite(global_delta).all() or not np.isfinite(dense_delta).all():
        raise RuntimeError("CPU/device feature difference is non-finite")
    global_max = float(np.max(global_delta))
    dense_max = float(np.max(dense_delta))
    return {
        "max_abs_error": max(global_max, dense_max),
        "global_max_abs_error": global_max,
        "dense_max_abs_error": dense_max,
        "global_mean_abs_error": float(np.mean(global_delta)),
        "dense_mean_abs_error": float(np.mean(dense_delta)),
        "global_cosine_similarity": _cosine(cpu.cls_embedding, requested.cls_embedding),
        "dense_cosine_similarity": _cosine(cpu.patch_grid, requested.patch_grid),
    }


def _agreement_gate(
    agreement: Mapping[str, float], config: SigLIP2SmokeConfig
) -> dict[str, Any]:
    maximum_error = float(agreement["max_abs_error"])
    minimum_cosine = min(
        float(agreement["global_cosine_similarity"]),
        float(agreement["dense_cosine_similarity"]),
    )
    if not math.isfinite(maximum_error) or not math.isfinite(minimum_cosine):
        raise RuntimeError("CPU/device agreement metrics are non-finite")
    passed = (
        maximum_error <= config.max_device_abs_error
        and minimum_cosine >= config.min_device_cosine_similarity
    )
    return {
        "passed": passed,
        "observed_max_abs_error": maximum_error,
        "allowed_max_abs_error": float(config.max_device_abs_error),
        "observed_min_cosine_similarity": minimum_cosine,
        "required_min_cosine_similarity": float(config.min_device_cosine_similarity),
    }


_BLOCK_PATTERN = re.compile(r"(?:^|\.)(?:layer|layers|block|blocks)\.(\d+)(?:\.|$)")


def _last_four_query_value_modules(model: Any) -> tuple[list[str], list[int]]:
    candidates: dict[int, dict[str, list[str]]] = {}
    for name, _module in model.named_modules():
        terminal = name.rsplit(".", 1)[-1]
        kind = (
            "query"
            if terminal in {"query", "q_proj"}
            else "value"
            if terminal in {"value", "v_proj"}
            else None
        )
        match = _BLOCK_PATTERN.search(name)
        if kind is not None and match is not None:
            candidates.setdefault(int(match.group(1)), {}).setdefault(kind, []).append(
                name
            )
    blocks = sorted(candidates)
    if len(blocks) < 4:
        raise RuntimeError(
            "LoRA requires query/value projections in at least four blocks"
        )
    selected_blocks = blocks[-4:]
    targets: list[str] = []
    for block in selected_blocks:
        per_kind = candidates[block]
        query = sorted(set(per_kind.get("query", ())))
        value = sorted(set(per_kind.get("value", ())))
        if len(query) != 1 or len(value) != 1:
            raise RuntimeError(
                f"LoRA block {block} must expose exactly one query and one value module"
            )
        targets.extend((query[0], value[0]))
    return targets, selected_blocks


def _pooler_tensor(output: Any, torch: Any) -> Any:
    pooler = getattr(output, "pooler_output", None)
    if pooler is None and isinstance(output, Mapping):
        pooler = output.get("pooler_output")
    if not torch.is_tensor(pooler) or pooler.ndim != 2:
        raise TypeError("SigLIP2 LoRA smoke requires a 2-D pooler_output tensor")
    return pooler


def _import_peft_api() -> tuple[Any, Callable[..., Any]]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise SigLIP2SmokeDependencyError(
            f"PEFT is unavailable for requested LoRA. {_DEPENDENCY_GUIDANCE}"
        ) from error
    return LoraConfig, get_peft_model


def _parameter_delta_records(
    before: Mapping[str, Any], after: Mapping[str, Any], torch: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(before) != set(after):
        raise RuntimeError("Trainable parameter identities changed during LoRA step")
    records: list[dict[str, Any]] = []
    aggregate_squared = 0.0
    aggregate_max = 0.0
    changed_elements = 0
    changed_tensors = 0
    for name in sorted(before):
        left = before[name].detach().float().cpu().contiguous()
        right = after[name].detach().float().cpu().contiguous()
        delta = right - left
        if not bool(torch.isfinite(delta).all().item()):
            raise RuntimeError(f"LoRA parameter delta is non-finite: {name}")
        nonzero = int(torch.count_nonzero(delta).item())
        squared = float(torch.sum(delta * delta).item())
        maximum = float(torch.max(torch.abs(delta)).item())
        aggregate_squared += squared
        aggregate_max = max(aggregate_max, maximum)
        changed_elements += nonzero
        changed_tensors += int(nonzero > 0)
        records.append(
            {
                "name": name,
                "parameter_count": int(left.numel()),
                "changed_element_count": nonzero,
                "delta_l2": float(squared**0.5),
                "delta_max_abs": maximum,
                "before_sha256": hashlib.sha256(left.numpy().tobytes()).hexdigest(),
                "after_sha256": hashlib.sha256(right.numpy().tobytes()).hexdigest(),
            }
        )
    aggregate_l2 = float(aggregate_squared**0.5)
    if not math.isfinite(aggregate_l2) or changed_elements <= 0:
        raise RuntimeError("LoRA trainable parameters did not update")
    return records, {
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
        "delta_l2": aggregate_l2,
        "delta_max_abs": aggregate_max,
        "nonzero_update": True,
    }


def _run_lora_step(
    model: Any,
    preprocessed_patches: np.ndarray,
    *,
    device: str,
    rank: int,
    embedding_dim: int,
    torch: Any,
) -> dict[str, Any]:
    LoraConfig, get_peft_model = _import_peft_api()
    targets, blocks = _last_four_query_value_modules(model)
    module_lookup = dict(model.named_modules())
    expected_lora_count = 0
    for target in targets:
        module = module_lookup[target]
        input_features = int(getattr(module, "in_features", 0))
        output_features = int(getattr(module, "out_features", 0))
        if input_features <= 0 or output_features <= 0:
            raise RuntimeError(
                f"LoRA target is not a finite linear projection: {target}"
            )
        expected_lora_count += rank * (input_features + output_features)

    torch.manual_seed(_SEED)
    mps_seed = getattr(getattr(torch, "mps", None), "manual_seed", None)
    if device == "mps" and callable(mps_seed):
        mps_seed(_SEED)
    base_parameter_count = sum(int(item.numel()) for item in model.parameters())
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        inference_mode=False,
    )
    adapted = get_peft_model(model, lora_config).to(device=device, dtype=torch.float32)
    adapted.train()
    head = torch.nn.Linear(embedding_dim, 1, bias=True).to(
        device=device, dtype=torch.float32
    )

    model_trainable = {
        f"model::{name}": parameter
        for name, parameter in adapted.named_parameters()
        if bool(parameter.requires_grad)
    }
    if not model_trainable or any(
        ".lora_A." not in name and ".lora_B." not in name for name in model_trainable
    ):
        raise RuntimeError(
            "PEFT exposed trainable parameters outside the locked LoRA set"
        )
    observed_lora_count = sum(int(item.numel()) for item in model_trainable.values())
    if observed_lora_count != expected_lora_count:
        raise RuntimeError(
            "PEFT trainable count differs from the exact q/v rank contract: "
            f"expected {expected_lora_count}, observed {observed_lora_count}"
        )
    head_trainable = {
        f"head::{name}": parameter
        for name, parameter in head.named_parameters()
        if bool(parameter.requires_grad)
    }
    expected_head_count = embedding_dim + 1
    observed_head_count = sum(int(item.numel()) for item in head_trainable.values())
    if observed_head_count != expected_head_count:
        raise RuntimeError("Tiny pooler head parameter count is inconsistent")
    all_trainable = {**model_trainable, **head_trainable}
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in all_trainable.items()
    }

    if (
        preprocessed_patches.shape != (2, 3, 224, 224)
        or not np.isfinite(preprocessed_patches).all()
    ):
        raise RuntimeError("LoRA received invalid locked SigLIP2 pixel values")
    tensor = torch.as_tensor(preprocessed_patches, dtype=torch.float32, device=device)
    labels = torch.as_tensor((0.0, 1.0), dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        tuple(all_trainable.values()), lr=1e-3, weight_decay=0.0
    )

    synchronize_torch_device(torch, device, require_available=True)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = adapted(pixel_values=tensor)
    pooler = _pooler_tensor(output, torch)
    logits = head(pooler).reshape(-1).float()
    loss_before = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    if not bool(torch.isfinite(loss_before).item()):
        raise RuntimeError("LoRA smoke produced a non-finite pre-update loss")
    loss_before.backward()
    squared_gradient = 0.0
    for name, parameter in all_trainable.items():
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(f"LoRA smoke produced a non-finite gradient: {name}")
        squared_gradient += float(torch.sum(gradient.float() ** 2).item())
    gradient_l2 = float(squared_gradient**0.5)
    if not math.isfinite(gradient_l2) or gradient_l2 <= 0.0:
        raise RuntimeError("LoRA smoke produced no finite non-zero gradient")
    optimizer.step()
    synchronize_torch_device(torch, device, require_available=True)
    step_seconds = time.perf_counter() - started

    after = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in all_trainable.items()
    }
    delta_records, delta_summary = _parameter_delta_records(before, after, torch)
    model_changed = sum(
        record["changed_element_count"]
        for record in delta_records
        if str(record["name"]).startswith("model::")
    )
    if model_changed <= 0:
        raise RuntimeError("LoRA adapter weights did not update")

    adapted.eval()
    head.eval()
    with torch.no_grad():
        updated_pooler = _pooler_tensor(adapted(pixel_values=tensor), torch)
        updated_logits = head(updated_pooler).reshape(-1).float()
        loss_after = torch.nn.functional.binary_cross_entropy_with_logits(
            updated_logits, labels
        )
    if not bool(torch.isfinite(loss_after).item()):
        raise RuntimeError("LoRA smoke produced a non-finite post-update loss")

    adapted_total = sum(int(item.numel()) for item in adapted.parameters())
    head_total = sum(int(item.numel()) for item in head.parameters())
    expected_total = base_parameter_count + expected_lora_count + expected_head_count
    observed_total = adapted_total + head_total
    if observed_total != expected_total:
        raise RuntimeError("PEFT total parameter count differs from the exact contract")
    trainable_count = observed_lora_count + observed_head_count
    return {
        "requested": True,
        "performed": True,
        "rank": rank,
        "alpha": 2 * rank,
        "dropout": 0.0,
        "precision": "float32",
        "seed": _SEED,
        "optimizer": {"name": "AdamW", "learning_rate": 1e-3, "weight_decay": 0.0},
        "target_policy": "q_and_v_last_four_vision_transformer_blocks",
        "target_blocks": blocks,
        "target_modules": targets,
        "tiny_head": "linear_binary_on_siglip_pooler_output",
        "base_parameter_count": base_parameter_count,
        "lora_trainable_parameter_count": observed_lora_count,
        "tiny_head_trainable_parameter_count": observed_head_count,
        "trainable_parameter_count": trainable_count,
        "total_parameter_count_including_adapter_and_head": observed_total,
        "trainable_fraction": float(trainable_count / observed_total),
        "loss_before_step": float(loss_before.detach().cpu().item()),
        "loss_after_step": float(loss_after.detach().cpu().item()),
        "loss_delta": float(
            loss_after.detach().cpu().item() - loss_before.detach().cpu().item()
        ),
        "losses_finite": True,
        "gradient_l2": gradient_l2,
        "parameter_update": delta_summary,
        "parameter_deltas": delta_records,
        "step_seconds": float(step_seconds),
        "memory_after_step": foundation_runtime_diagnostics(
            torch_module=torch
        ).as_dict(),
    }


def _execute_siglip2_smoke(
    config: SigLIP2SmokeConfig, loaded: LoadedLocalFoundationModel
) -> dict[str, Any]:
    torch = _import_torch()
    resolved_device = select_torch_device(config.device, torch_module=torch)
    if resolved_device != config.device:
        raise RuntimeError("Requested device changed during resolution")
    patches = deterministic_semantic_rgb_patches(224)
    if patches.shape != (2, 224, 224, 3) or patches.dtype != np.float32:
        raise RuntimeError("Deterministic semantic RGB input contract changed")
    preprocessor = getattr(loaded, "preprocessor", None)
    if not callable(preprocessor):
        raise TypeError("Locked SigLIP2 loader did not provide its image processor")
    processed_once = _locked_preprocess(preprocessor, patches)
    processed_twice = _locked_preprocess(preprocessor, patches)
    if not np.array_equal(processed_once, processed_twice):
        raise RuntimeError("Locked SigLIP2 image processor is not bitwise repeatable")
    processor_record = _processor_record(
        preprocessor,
        loaded.provenance,
        processed_once,
    )
    runtime_at_start = foundation_runtime_diagnostics(torch_module=torch).as_dict()
    loaded_parameter_stats = _parameter_stats(loaded.model)

    cpu_extractor = _make_extractor(loaded.model, "cpu", torch, preprocessor)
    cpu_features, cpu_timing = _time_encode(
        cpu_extractor, patches, steady_runs=config.steady_runs
    )
    cpu_summary = _feature_summary(cpu_features)
    runtime_after_cpu = foundation_runtime_diagnostics(torch_module=torch).as_dict()

    requested_extractor = _make_extractor(
        loaded.model, resolved_device, torch, preprocessor
    )
    requested_features, requested_timing = _time_encode(
        requested_extractor, patches, steady_runs=config.steady_runs
    )
    requested_summary = _feature_summary(requested_features)
    frozen_parameter_stats = _parameter_stats(loaded.model)
    if frozen_parameter_stats["trainable_parameter_count"] != 0:
        raise RuntimeError("Frozen SigLIP2 extractor retained trainable parameters")
    runtime_after_requested = foundation_runtime_diagnostics(
        torch_module=torch
    ).as_dict()
    agreement = _feature_agreement(cpu_features, requested_features)
    gate = _agreement_gate(agreement, config)
    if not gate["passed"]:
        raise RuntimeError(
            "SigLIP2 CPU/device agreement gate failed: "
            f"max_abs_error={gate['observed_max_abs_error']}, "
            f"min_cosine={gate['observed_min_cosine_similarity']}"
        )

    if config.lora_rank is None:
        lora: dict[str, Any] = {
            "requested": False,
            "performed": False,
            "reason": "lora_rank_not_requested",
        }
    else:
        lora = _run_lora_step(
            loaded.model,
            processed_once,
            device=resolved_device,
            rank=config.lora_rank,
            embedding_dim=cpu_features.embedding_dim,
            torch=torch,
        )

    return {
        "status": "passed",
        "engineering_smoke_test_passed": True,
        "scientific_validation_passed": False,
        "resolved_device": resolved_device,
        "runtime_versions": _runtime_versions(torch),
        "input": {
            "generator": "deterministic_semantic_rgb_clean_and_proxy_artifact_v1",
            "labels": ["clean_proxy", "fold_and_crack_proxy"],
            "shape": [int(value) for value in patches.shape],
            "dtype": str(patches.dtype),
            "range": [float(np.min(patches)), float(np.max(patches))],
            "finite": bool(np.isfinite(patches).all()),
            "semantic_channels": ["red", "green", "blue"],
            "sha256": _array_sha256(patches),
            "normalization_mean": list(SIGLIP2_BASE_MEAN),
            "normalization_std": list(SIGLIP2_BASE_STD),
            "processor": processor_record,
            "processor_bitwise_repeatable": True,
        },
        "model_parameter_stats_at_load": loaded_parameter_stats,
        "model_parameter_stats_frozen": frozen_parameter_stats,
        "memory_at_execution_start": runtime_at_start,
        "frozen_inference": {
            "cpu_reference": {
                "device": "cpu",
                "outputs": cpu_summary,
                "timing": cpu_timing,
            },
            "requested_device": {
                "device": resolved_device,
                "outputs": requested_summary,
                "timing": requested_timing,
            },
            "cpu_device_agreement": agreement,
            "cpu_device_agreement_gate": gate,
        },
        "memory_after_cpu_inference": runtime_after_cpu,
        "memory_after_requested_device_inference": runtime_after_requested,
        "lora": lora,
    }


ModelLoader = Callable[[str | Path], LoadedLocalFoundationModel]
SmokeExecutor = Callable[
    [SigLIP2SmokeConfig, LoadedLocalFoundationModel], dict[str, Any]
]


def _build_report(
    config: SigLIP2SmokeConfig,
    provenance: Mapping[str, Any],
    execution: Mapping[str, Any],
    *,
    model_load_and_hash_seconds: float,
) -> dict[str, Any]:
    if execution.get("engineering_smoke_test_passed") is not True:
        raise RuntimeError("SigLIP2 execution did not pass its engineering gates")
    report = {
        "schema_version": "1.0",
        "result_type": "engineering_siglip2_smoke_only",
        "status": "passed",
        "engineering_smoke_test_passed": True,
        "scientific_validation_passed": False,
        "model": dict(provenance),
        "configuration": config.as_dict(),
        "policy": {
            "offline": True,
            "network_access_allowed": False,
            "token_used": False,
            "trust_remote_code": False,
        },
        "model_load_and_hash_seconds": float(model_load_and_hash_seconds),
        "execution": dict(execution),
        "limitations": [
            "The two inputs are deterministic synthetic engineering patches, not real WSI.",
            "No fold/crack ground truth, efficacy metric, clinical endpoint, COMET field, or CosMx field is evaluated.",
            "LoRA completes one optimization step only; it does not establish adaptation quality or generalizability.",
            "A passing smoke result must not be interpreted as scientific or clinical validation.",
        ],
    }
    return json.loads(json.dumps(report, allow_nan=False, sort_keys=True))


def _atomic_write_json(path: str | Path, report: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


def run_siglip2_smoke(
    config: SigLIP2SmokeConfig,
    *,
    output_json: str | Path | None = None,
    model_loader: ModelLoader | None = None,
    executor: SmokeExecutor | None = None,
) -> dict[str, Any]:
    """Run the locked offline smoke and atomically persist only a passing report."""

    snapshot = Path(config.snapshot_path)
    if snapshot.is_symlink() or not snapshot.is_dir():
        raise RuntimeError("SigLIP2 snapshot_path must be an existing local directory")
    loader = load_local_siglip2_base_vision if model_loader is None else model_loader
    smoke_executor = _execute_siglip2_smoke if executor is None else executor
    started = time.perf_counter()
    loaded = loader(snapshot)
    load_seconds = time.perf_counter() - started
    provenance = _validate_locked_provenance(loaded.provenance, snapshot)
    execution = smoke_executor(config, loaded)
    report = _build_report(
        config,
        provenance,
        execution,
        model_load_and_hash_seconds=load_seconds,
    )
    if output_json is not None:
        _atomic_write_json(output_json, report)
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the locked offline SigLIP2 CPU/MPS/LoRA engineering smoke"
    )
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--steady-runs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, choices=(4, 8), default=None)
    parser.add_argument("--max-device-abs-error", type=float, default=5e-3)
    parser.add_argument("--min-device-cosine-similarity", type=float, default=0.99999)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        config = SigLIP2SmokeConfig(
            snapshot_path=args.snapshot_path,
            device=args.device,
            steady_runs=args.steady_runs,
            lora_rank=args.lora_rank,
            max_device_abs_error=args.max_device_abs_error,
            min_device_cosine_similarity=args.min_device_cosine_similarity,
        )
        report = run_siglip2_smoke(config, output_json=args.output_json)
    except (
        ImportError,
        OSError,
        RuntimeError,
        SigLIP2SmokeDependencyError,
        TypeError,
        ValueError,
    ) as error:
        print(f"SigLIP2 smoke failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())


__all__ = [
    "SIGLIP2_BASE_ASSET_SHA256",
    "SIGLIP2_BASE_MODEL_ID",
    "SIGLIP2_BASE_REVISION",
    "SigLIP2SmokeConfig",
    "SigLIP2SmokeDependencyError",
    "deterministic_semantic_rgb_patches",
    "main",
    "run_siglip2_smoke",
]
