"""Auditable DINOv2 CPU/MPS and optional LoRA engineering smoke test.

This module is deliberately *not* a scientific efficacy benchmark.  It checks
that an exact public Hugging Face revision can be loaded without implicit
credentials, produces finite global and spatial embeddings on CPU and the
requested device, and (optionally) survives one BF16 LoRA optimization step.

Network access is disabled by default.  A caller must explicitly opt in with
``allow_download=True`` and must always supply an immutable 40-character Hugging
Face commit revision.  Model weights are SHA256-hashed before execution so the
JSON report is independently auditable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import statistics
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .foundation import (
    DINOv2FeatureExtractor,
    FoundationFeatures,
    foundation_runtime_diagnostics,
    preprocess_dinov2_rgb,
    select_torch_device,
)

_EXACT_HF_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")
_APPROVED_MODEL_IDS = frozenset(("facebook/dinov2-small",))
_WEIGHT_PATTERNS = ("*.safetensors",)
_DEPENDENCY_GUIDANCE = (
    "Install an approved Apple-silicon PyTorch runtime plus transformers, "
    "huggingface_hub, and (for LoRA) peft; for example: "
    "python -m pip install torch torchvision transformers huggingface_hub peft"
)


class FoundationSmokeDependencyError(RuntimeError):
    """Raised when an optional foundation-model dependency is unavailable."""


@dataclass(frozen=True)
class FoundationSmokeConfig:
    """Immutable execution and provenance contract for a smoke run."""

    revision: str
    model_id: str = "facebook/dinov2-small"
    cache_dir: str | Path = Path(".cache/foldcrack_qc/huggingface")
    device: str = "auto"
    allow_download: bool = False
    image_size: int = 224
    steady_runs: int = 3
    lora_rank: int | None = None
    max_device_abs_error: float = 1e-3
    min_device_cosine_similarity: float = 0.9999

    def __post_init__(self) -> None:
        revision = str(self.revision).strip().lower()
        model_id = str(self.model_id).strip()
        device = str(self.device).strip().lower()
        cache_dir = Path(self.cache_dir)
        if not _EXACT_HF_REVISION.fullmatch(revision):
            raise ValueError(
                "revision must be an exact 40-character Hugging Face commit hash; "
                "mutable names such as 'main' are not auditable"
            )
        if model_id not in _APPROVED_MODEL_IDS:
            raise ValueError(
                f"model_id must be in the approved smoke allowlist: "
                f"{sorted(_APPROVED_MODEL_IDS)}"
            )
        if device not in {"auto", "cpu", "mps"}:
            raise ValueError("device must be one of 'auto', 'cpu', or 'mps'")
        if not str(cache_dir):
            raise ValueError("cache_dir must be explicit and non-empty")
        if int(self.image_size) <= 0:
            raise ValueError("image_size must be positive")
        if int(self.steady_runs) <= 0:
            raise ValueError("steady_runs must be positive")
        if self.lora_rank not in {None, 4, 8}:
            raise ValueError("lora_rank must be None, 4, or 8")
        if not math.isfinite(float(self.max_device_abs_error)) or not (
            0.0 < float(self.max_device_abs_error) < 1.0
        ):
            raise ValueError("max_device_abs_error must lie in (0, 1)")
        if not math.isfinite(float(self.min_device_cosine_similarity)) or not (
            0.0 < float(self.min_device_cosine_similarity) <= 1.0
        ):
            raise ValueError("min_device_cosine_similarity must lie in (0, 1]")
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "cache_dir", cache_dir)
        object.__setattr__(self, "image_size", int(self.image_size))
        object.__setattr__(self, "steady_runs", int(self.steady_runs))

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "cache_dir": str(self.cache_dir),
            "device": self.device,
            "allow_download": bool(self.allow_download),
            "offline": not bool(self.allow_download),
            "image_size": self.image_size,
            "steady_runs": self.steady_runs,
            "lora_rank": self.lora_rank,
            "max_device_abs_error": float(self.max_device_abs_error),
            "min_device_cosine_similarity": float(
                self.min_device_cosine_similarity
            ),
        }


@dataclass(frozen=True)
class WeightDigest:
    """Publicly serializable identity for one cached weight file."""

    filename: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).is_absolute():
            raise ValueError("weight filename must be a non-empty relative path")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("weight sha256 must contain exactly 64 lowercase hex digits")
        if self.size_bytes <= 0:
            raise ValueError("weight file must be non-empty")

    def as_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "size_bytes": int(self.size_bytes),
        }


@dataclass(frozen=True)
class LoadedFoundationModel:
    """Loaded model plus immutable cached-weight provenance."""

    model: Any
    resolved_revision: str
    weight_digests: tuple[WeightDigest, ...]
    snapshot_path: Path
    configuration_digests: tuple[WeightDigest, ...] = ()

    def __post_init__(self) -> None:
        resolved = str(self.resolved_revision).strip().lower()
        if not _EXACT_HF_REVISION.fullmatch(resolved):
            raise ValueError("resolved_revision must be an exact commit hash")
        if not self.weight_digests:
            raise ValueError("at least one hashed model weight file is required")
        object.__setattr__(self, "resolved_revision", resolved)
        object.__setattr__(self, "snapshot_path", Path(self.snapshot_path))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_snapshot_weights(snapshot_path: Path) -> tuple[WeightDigest, ...]:
    files: set[Path] = set()
    for pattern in _WEIGHT_PATTERNS:
        files.update(path for path in snapshot_path.rglob(pattern) if path.is_file())
    digests = tuple(
        WeightDigest(
            filename=path.relative_to(snapshot_path).as_posix(),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    if not digests:
        raise RuntimeError(
            f"No model weight files were found in the exact snapshot {snapshot_path}. "
            "Only safetensors weights are accepted."
        )
    return digests


def _hash_snapshot_configuration(snapshot_path: Path) -> tuple[WeightDigest, ...]:
    return tuple(
        WeightDigest(
            filename=path.relative_to(snapshot_path).as_posix(),
            sha256=_sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(snapshot_path.rglob("*.json"), key=lambda item: item.as_posix())
        if path.is_file() and path.stat().st_size > 0
    )


def load_huggingface_model(
    config: FoundationSmokeConfig,
    *,
    transformers_module: Any | None = None,
    hub_module: Any | None = None,
) -> LoadedFoundationModel:
    """Load and hash one exact public Hugging Face model revision.

    ``token=False`` and ``trust_remote_code=False`` are invariant.  The snapshot
    operation is ``local_files_only`` unless the caller explicitly enables
    downloads.  The subsequent model construction is always local-only so it
    cannot make an additional, unaccounted network request.
    """

    if transformers_module is None:
        try:
            import transformers as transformers_module
        except ImportError as error:  # pragma: no cover - runtime dependent
            raise FoundationSmokeDependencyError(
                f"Transformers is unavailable. {_DEPENDENCY_GUIDANCE}"
            ) from error
    if hub_module is None:
        try:
            import huggingface_hub as hub_module
        except ImportError as error:  # pragma: no cover - runtime dependent
            raise FoundationSmokeDependencyError(
                f"huggingface_hub is unavailable. {_DEPENDENCY_GUIDANCE}"
            ) from error

    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download = getattr(hub_module, "snapshot_download", None)
    auto_model = getattr(transformers_module, "AutoModel", None)
    if not callable(snapshot_download) or auto_model is None:
        raise TypeError(
            "Injected Hugging Face modules must expose snapshot_download and AutoModel"
        )
    try:
        snapshot = snapshot_download(
            repo_id=config.model_id,
            revision=config.revision,
            cache_dir=str(cache_dir),
            local_files_only=not config.allow_download,
            token=False,
            allow_patterns=("*.json",) + _WEIGHT_PATTERNS,
        )
    except (OSError, RuntimeError, ValueError) as error:
        mode = "offline cache lookup" if not config.allow_download else "public download"
        raise RuntimeError(
            f"Hugging Face {mode} failed for {config.model_id}@{config.revision}. "
            "Populate the explicit cache first or rerun with allow_download=True."
        ) from error

    snapshot_path = Path(snapshot).resolve()
    resolved_revision = snapshot_path.name.lower()
    if resolved_revision != config.revision:
        raise RuntimeError(
            "Resolved Hugging Face snapshot does not match the requested immutable "
            f"revision: requested {config.revision}, resolved {resolved_revision}"
        )
    weight_digests = _hash_snapshot_weights(snapshot_path)
    configuration_digests = _hash_snapshot_configuration(snapshot_path)
    try:
        model = auto_model.from_pretrained(
            config.model_id,
            revision=config.revision,
            cache_dir=str(cache_dir),
            local_files_only=True,
            trust_remote_code=False,
            token=False,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise RuntimeError(
            f"Unable to construct {config.model_id}@{config.revision} from the "
            "verified local snapshot."
        ) from error
    return LoadedFoundationModel(
        model=model,
        resolved_revision=resolved_revision,
        weight_digests=weight_digests,
        snapshot_path=snapshot_path,
        configuration_digests=configuration_digests,
    )


def deterministic_smoke_patches(image_size: int = 224) -> np.ndarray:
    """Return two deterministic float32 RGB patches in the unit interval."""

    if int(image_size) <= 0:
        raise ValueError("image_size must be positive")
    side = int(image_size)
    y, x = np.mgrid[:side, :side].astype(np.float32)
    denominator = np.float32(max(side - 1, 1))
    x /= denominator
    y /= denominator
    texture = np.sin(np.float32(18.0) * x) * np.cos(np.float32(14.0) * y)
    first = np.stack(
        (
            np.float32(0.62) + np.float32(0.22) * texture,
            np.float32(0.38) + np.float32(0.18) * np.sin(np.float32(11.0) * y),
            np.float32(0.55) + np.float32(0.20) * np.cos(np.float32(9.0) * x),
        ),
        axis=-1,
    )
    second = first.copy()
    fold_band = np.abs(y - (np.float32(0.25) + np.float32(0.45) * x)) < np.float32(0.06)
    crack_line = np.abs(y - (np.float32(0.82) - np.float32(0.50) * x)) < np.float32(0.012)
    second[fold_band] = np.clip(second[fold_band] * np.float32(0.55), 0.0, 1.0)
    second[crack_line] = np.float32(0.97)
    return np.ascontiguousarray(np.clip(np.stack((first, second)), 0.0, 1.0), dtype=np.float32)


def _synchronize(torch: Any, device: str) -> None:
    if device != "mps":
        return
    synchronize = getattr(getattr(torch, "mps", None), "synchronize", None)
    if callable(synchronize):
        synchronize()


def _time_encode(
    extractor: DINOv2FeatureExtractor,
    patches: np.ndarray,
    *,
    steady_runs: int,
) -> tuple[FoundationFeatures, dict[str, Any]]:
    torch = extractor._torch
    _synchronize(torch, extractor.device)
    start = time.perf_counter()
    features = extractor.encode(patches, batch_size=2)
    _synchronize(torch, extractor.device)
    warm_seconds = time.perf_counter() - start

    durations: list[float] = []
    for _ in range(steady_runs):
        _synchronize(torch, extractor.device)
        start = time.perf_counter()
        features = extractor.encode(patches, batch_size=2)
        _synchronize(torch, extractor.device)
        durations.append(time.perf_counter() - start)
    return features, {
        "scope": "two_patch_preprocess_and_frozen_forward",
        "warm_seconds": float(warm_seconds),
        "steady_runs": int(steady_runs),
        "steady_seconds": [float(value) for value in durations],
        "steady_min_seconds": float(min(durations)),
        "steady_median_seconds": float(statistics.median(durations)),
        "steady_mean_seconds": float(statistics.fmean(durations)),
    }


def _feature_summary(features: FoundationFeatures) -> dict[str, Any]:
    cls = np.asarray(features.cls_embedding)
    spatial = np.asarray(features.patch_grid)
    if cls.dtype != np.float32 or spatial.dtype != np.float32:
        raise RuntimeError("Foundation outputs must be materialized as float32")
    if not np.isfinite(cls).all() or not np.isfinite(spatial).all():
        raise RuntimeError("Foundation inference produced non-finite embeddings")
    return {
        "cls": {
            "shape": [int(value) for value in cls.shape],
            "dtype": str(cls.dtype),
            "finite": True,
        },
        "spatial": {
            "shape": [int(value) for value in spatial.shape],
            "dtype": str(spatial.dtype),
            "finite": True,
        },
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
    return float(np.dot(left, right) / denominator)


def _feature_agreement(
    cpu: FoundationFeatures,
    requested: FoundationFeatures,
) -> dict[str, float]:
    if cpu.cls_embedding.shape != requested.cls_embedding.shape:
        raise RuntimeError("CPU and requested-device CLS shapes differ")
    if cpu.patch_grid.shape != requested.patch_grid.shape:
        raise RuntimeError("CPU and requested-device spatial shapes differ")
    cls_error = float(np.max(np.abs(cpu.cls_embedding - requested.cls_embedding)))
    spatial_error = float(np.max(np.abs(cpu.patch_grid - requested.patch_grid)))
    return {
        "max_abs_error": max(cls_error, spatial_error),
        "cls_max_abs_error": cls_error,
        "spatial_max_abs_error": spatial_error,
        "cls_cosine_similarity": _cosine(cpu.cls_embedding, requested.cls_embedding),
        "spatial_cosine_similarity": _cosine(cpu.patch_grid, requested.patch_grid),
    }


def _device_agreement_gate(
    agreement: Mapping[str, float], config: FoundationSmokeConfig
) -> dict[str, Any]:
    max_error = float(agreement["max_abs_error"])
    minimum_cosine = min(
        float(agreement["cls_cosine_similarity"]),
        float(agreement["spatial_cosine_similarity"]),
    )
    passed = (
        max_error <= config.max_device_abs_error
        and minimum_cosine >= config.min_device_cosine_similarity
    )
    return {
        "passed": passed,
        "observed_max_abs_error": max_error,
        "allowed_max_abs_error": float(config.max_device_abs_error),
        "observed_min_cosine_similarity": minimum_cosine,
        "required_min_cosine_similarity": float(
            config.min_device_cosine_similarity
        ),
    }


def _runtime_versions() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for name in (
        "numpy",
        "opencv-python",
        "torch",
        "transformers",
        "huggingface-hub",
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
    }


def _pair(value: Any, *, fallback: int) -> tuple[int, int]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = tuple(int(item) for item in value)
        if len(values) == 2 and min(values) > 0:
            return values
    try:
        side = int(value)
    except (TypeError, ValueError):
        side = fallback
    if side <= 0:
        side = fallback
    return side, side


def dinov2_model_geometry(
    model: Any, image_size: int
) -> tuple[tuple[int, int], int]:
    """Resolve patch and prefix-token geometry from a loaded DINOv2 config."""

    model_config = getattr(model, "config", None)
    patch_size = _pair(getattr(model_config, "patch_size", 14), fallback=14)
    if any(image_size % patch != 0 for patch in patch_size):
        raise RuntimeError(
            f"image_size={image_size} is not divisible by model patch size {patch_size}"
        )
    register_tokens = int(getattr(model_config, "num_register_tokens", 0) or 0)
    return patch_size, 1 + register_tokens


def _block_number(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:layer|layers|block|blocks)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _last_four_query_value_modules(model: Any) -> tuple[list[str], list[int]]:
    candidates: dict[int, list[str]] = {}
    for name, _module in model.named_modules():
        terminal = name.rsplit(".", 1)[-1]
        if terminal not in {"query", "value", "q_proj", "v_proj"}:
            continue
        block = _block_number(name)
        if block is not None:
            candidates.setdefault(block, []).append(name)
    blocks = sorted(candidates)
    if len(blocks) < 4:
        raise RuntimeError(
            "Could not identify query/value projections in at least four transformer "
            "blocks; this model is not compatible with the locked LoRA smoke contract"
        )
    selected_blocks = blocks[-4:]
    targets: list[str] = []
    for block in selected_blocks:
        block_targets = sorted(set(candidates[block]))
        terminals = {target.rsplit(".", 1)[-1] for target in block_targets}
        has_query = bool(terminals & {"query", "q_proj"})
        has_value = bool(terminals & {"value", "v_proj"})
        if not has_query or not has_value:
            raise RuntimeError(
                f"Transformer block {block} does not expose both query and value projections"
            )
        targets.extend(block_targets)
    return targets, selected_blocks


def _cls_tensor(output: Any, torch: Any) -> Any:
    if hasattr(output, "last_hidden_state"):
        hidden = output.last_hidden_state
        return hidden[:, 0, :]
    if isinstance(output, Mapping):
        if "x_norm_clstoken" in output:
            return output["x_norm_clstoken"]
        if "last_hidden_state" in output:
            return output["last_hidden_state"][:, 0, :]
    if isinstance(output, (tuple, list)) and output:
        return output[0][:, 0, :]
    if torch.is_tensor(output) and output.ndim == 3:
        return output[:, 0, :]
    raise TypeError("LoRA smoke step requires a model output with a CLS token")


def _run_lora_step(
    model: Any,
    patches: np.ndarray,
    *,
    device: str,
    rank: int,
    image_size: int,
    embedding_dim: int,
    torch: Any,
) -> dict[str, Any]:
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise FoundationSmokeDependencyError(
            f"PEFT is unavailable for the requested LoRA step. {_DEPENDENCY_GUIDANCE}"
        ) from error

    targets, blocks = _last_four_query_value_modules(model)
    manual_seed = getattr(torch, "manual_seed", None)
    if callable(manual_seed):
        manual_seed(17_291)
    mps_seed = getattr(getattr(torch, "mps", None), "manual_seed", None)
    if device == "mps" and callable(mps_seed):
        mps_seed(17_291)

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=2 * rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=targets,
        inference_mode=False,
    )
    adapted = get_peft_model(model, lora_config)
    adapted = adapted.to(device=device, dtype=torch.bfloat16)
    adapted.train()
    head = torch.nn.Linear(embedding_dim, 1, bias=True).to(
        device=device,
        dtype=torch.bfloat16,
    )

    trainable_model = [
        (name, parameter)
        for name, parameter in adapted.named_parameters()
        if bool(parameter.requires_grad)
    ]
    if not trainable_model:
        raise RuntimeError("PEFT created no trainable model parameters")
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in trainable_model
    }
    trainable = [parameter for _, parameter in trainable_model]
    trainable.extend(
        parameter for parameter in head.parameters() if bool(parameter.requires_grad)
    )
    optimizer = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0)
    normalized = preprocess_dinov2_rgb(patches, image_size=image_size)
    tensor = torch.as_tensor(normalized, dtype=torch.bfloat16, device=device)
    labels = torch.as_tensor((0.0, 1.0), dtype=torch.float32, device=device)

    _synchronize(torch, device)
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    output = adapted(pixel_values=tensor)
    cls = _cls_tensor(output, torch)
    logits = head(cls).reshape(-1).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    finite_loss = bool(torch.isfinite(loss).item())
    if not finite_loss:
        raise RuntimeError("LoRA smoke step produced a non-finite loss")
    loss.backward()
    optimizer.step()
    _synchronize(torch, device)
    elapsed = time.perf_counter() - start

    squared_delta = 0.0
    max_delta = 0.0
    for name, parameter in trainable_model:
        difference = parameter.detach().float().cpu() - before[name]
        squared_delta += float(torch.sum(difference * difference).item())
        max_delta = max(max_delta, float(torch.max(torch.abs(difference)).item()))
    delta_l2 = float(squared_delta**0.5)
    if not np.isfinite(delta_l2) or delta_l2 <= 0.0:
        raise RuntimeError("LoRA trainable model weights did not change after one step")

    model_total = sum(int(parameter.numel()) for parameter in adapted.parameters())
    head_total = sum(int(parameter.numel()) for parameter in head.parameters())
    model_trainable = sum(int(parameter.numel()) for _, parameter in trainable_model)
    head_trainable = sum(
        int(parameter.numel())
        for parameter in head.parameters()
        if bool(parameter.requires_grad)
    )
    total = model_total + head_total
    trainable_count = model_trainable + head_trainable
    return {
        "requested": True,
        "performed": True,
        "precision": "bfloat16",
        "rank": int(rank),
        "target_policy": "query_and_value_last_four_transformer_blocks",
        "target_blocks": [int(block) for block in blocks],
        "target_modules": targets,
        "tiny_head": "linear_cls_binary",
        "loss": float(loss.detach().cpu().item()),
        "loss_finite": finite_loss,
        "trainable_parameter_count": int(trainable_count),
        "total_parameter_count": int(total),
        "trainable_fraction": float(trainable_count / total),
        "model_trainable_weight_delta_l2": delta_l2,
        "model_trainable_weight_delta_max_abs": max_delta,
        "nonzero_trainable_weight_delta": True,
        "step_seconds": float(elapsed),
        "mps_memory_after_step": foundation_runtime_diagnostics(
            torch_module=torch
        ).as_dict(),
    }


def _execute_foundation_smoke(
    config: FoundationSmokeConfig,
    loaded: LoadedFoundationModel,
) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - runtime dependent
        raise FoundationSmokeDependencyError(
            f"PyTorch is unavailable. {_DEPENDENCY_GUIDANCE}"
        ) from error

    patches = deterministic_smoke_patches(config.image_size)
    patch_size, prefix_tokens = dinov2_model_geometry(
        loaded.model, config.image_size
    )
    runtime_before = foundation_runtime_diagnostics(torch_module=torch).as_dict()

    cpu_extractor = DINOv2FeatureExtractor(
        loaded.model,
        device="cpu",
        image_size=config.image_size,
        patch_size=patch_size,
        prefix_tokens=prefix_tokens,
        model_input_name="pixel_values",
        torch_module=torch,
    )
    cpu_features, cpu_timing = _time_encode(
        cpu_extractor,
        patches,
        steady_runs=config.steady_runs,
    )

    resolved_device = select_torch_device(config.device, torch_module=torch)
    requested_extractor = DINOv2FeatureExtractor(
        loaded.model,
        device=resolved_device,
        image_size=config.image_size,
        patch_size=patch_size,
        prefix_tokens=prefix_tokens,
        model_input_name="pixel_values",
        torch_module=torch,
    )
    requested_features, requested_timing = _time_encode(
        requested_extractor,
        patches,
        steady_runs=config.steady_runs,
    )
    cpu_summary = _feature_summary(cpu_features)
    requested_summary = _feature_summary(requested_features)
    agreement = _feature_agreement(cpu_features, requested_features)
    agreement_gate = _device_agreement_gate(agreement, config)
    runtime_after_inference = foundation_runtime_diagnostics(
        torch_module=torch
    ).as_dict()

    if not agreement_gate["passed"]:
        lora = {
            "requested": config.lora_rank is not None,
            "performed": False,
            "reason": "device_agreement_gate_failed",
        }
    elif config.lora_rank is None:
        lora: dict[str, Any] = {
            "requested": False,
            "performed": False,
            "reason": "lora_rank_not_requested",
        }
    else:
        lora = _run_lora_step(
            loaded.model,
            patches,
            device=resolved_device,
            rank=config.lora_rank,
            image_size=config.image_size,
            embedding_dim=cpu_features.embedding_dim,
            torch=torch,
        )

    return {
        "status": (
            "passed" if agreement_gate["passed"] else "failed_device_agreement"
        ),
        "resolved_device": resolved_device,
        "runtime_versions": _runtime_versions(),
        "input": {
            "generator": "deterministic_two_patch_v1",
            "shape": [int(value) for value in patches.shape],
            "dtype": str(patches.dtype),
            "finite": bool(np.isfinite(patches).all()),
            "range": [float(np.min(patches)), float(np.max(patches))],
            "semantic_channels": ["red", "green", "blue"],
        },
        "runtime_before": runtime_before,
        "frozen_inference": {
            "cpu_reference": {"outputs": cpu_summary, "timing": cpu_timing},
            "requested_device": {
                "device": resolved_device,
                "outputs": requested_summary,
                "timing": requested_timing,
            },
            "cpu_device_agreement": agreement,
            "cpu_device_agreement_gate": agreement_gate,
        },
        "runtime_after_inference": runtime_after_inference,
        "lora": lora,
    }


ModelLoader = Callable[[FoundationSmokeConfig], LoadedFoundationModel]
SmokeExecutor = Callable[[FoundationSmokeConfig, LoadedFoundationModel], dict[str, Any]]


def _build_report(
    config: FoundationSmokeConfig,
    loaded: LoadedFoundationModel,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    if loaded.resolved_revision != config.revision:
        raise RuntimeError("loaded model revision changed before report assembly")
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "result_type": "engineering_foundation_smoke_only",
        "scientific_validation_passed": False,
        "status": str(execution.get("status", "passed")),
        "model": {
            "id": config.model_id,
            "requested_revision": config.revision,
            "resolved_revision": loaded.resolved_revision,
            "network_access_allowed": bool(config.allow_download),
            "trust_remote_code": False,
            "token_used": False,
            "weight_files": [
                digest.as_dict() for digest in loaded.weight_digests
            ],
            "configuration_files": [
                digest.as_dict() for digest in loaded.configuration_digests
            ],
        },
        "policy": {
            "allow_download": bool(config.allow_download),
            "offline": not bool(config.allow_download),
            "cache_dir": str(config.cache_dir),
        },
        "configuration": config.as_dict(),
        "execution": dict(execution),
        "limitations": [
            "Two deterministic synthetic patches test engineering execution only.",
            "No real WSI, artifact annotation, efficacy metric, or clinical endpoint is evaluated.",
            "A passing result does not establish fold/crack detection performance or generalizability.",
        ],
    }
    # Enforce JSON safety and reject NaN/Infinity instead of silently emitting it.
    return json.loads(json.dumps(report, allow_nan=False, sort_keys=True))


def run_foundation_smoke(
    config: FoundationSmokeConfig,
    *,
    output_json: str | Path | None = None,
    model_loader: ModelLoader | None = None,
    executor: SmokeExecutor | None = None,
) -> dict[str, Any]:
    """Execute the auditable engineering smoke and optionally persist JSON."""

    loader = load_huggingface_model if model_loader is None else model_loader
    smoke_executor = _execute_foundation_smoke if executor is None else executor
    loaded = loader(config)
    execution = smoke_executor(config, loaded)
    report = _build_report(config, loaded, execution)
    if output_json is not None:
        destination = Path(output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an offline-first DINOv2 CPU/MPS engineering smoke test"
    )
    parser.add_argument("--revision", required=True, help="exact 40-character commit hash")
    parser.add_argument("--model-id", default="facebook/dinov2-small")
    parser.add_argument(
        "--cache-dir",
        default=".cache/foldcrack_qc/huggingface",
        help="explicit Hugging Face cache directory",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="explicitly permit a public unauthenticated snapshot download",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--steady-runs", type=int, default=3)
    parser.add_argument("--lora-rank", type=int, choices=(4, 8), default=None)
    parser.add_argument("--max-device-abs-error", type=float, default=1e-3)
    parser.add_argument(
        "--min-device-cosine-similarity", type=float, default=0.9999
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _argument_parser().parse_args(argv)
    try:
        config = FoundationSmokeConfig(
            revision=args.revision,
            model_id=args.model_id,
            cache_dir=args.cache_dir,
            device=args.device,
            allow_download=args.allow_download,
            image_size=args.image_size,
            steady_runs=args.steady_runs,
            lora_rank=args.lora_rank,
            max_device_abs_error=args.max_device_abs_error,
            min_device_cosine_similarity=args.min_device_cosine_similarity,
        )
        report = run_foundation_smoke(config, output_json=args.output_json)
    except (FoundationSmokeDependencyError, RuntimeError, ValueError) as error:
        print(f"foundation smoke failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())


__all__ = [
    "FoundationSmokeConfig",
    "FoundationSmokeDependencyError",
    "LoadedFoundationModel",
    "WeightDigest",
    "deterministic_smoke_patches",
    "dinov2_model_geometry",
    "load_huggingface_model",
    "main",
    "run_foundation_smoke",
]
