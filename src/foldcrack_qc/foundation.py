"""Optional frozen-foundation feature extraction and patch anomaly scoring.

This module deliberately keeps PyTorch optional.  NumPy-only validation,
calibration, and nearest-neighbour scoring remain importable in the core
installation, while :class:`DINOv2FeatureExtractor` imports PyTorch only when
it is instantiated.

The API is intentionally strict about image semantics: callers must supply an
actual three-channel semantic RGB projection.  Multiplex arrays are never
silently truncated to their first three channels.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
HIBOU_B_MEAN = (0.7068, 0.5755, 0.7220)
HIBOU_B_STD = (0.1950, 0.2316, 0.1816)
SIGLIP2_BASE_MEAN = (0.5, 0.5, 0.5)
SIGLIP2_BASE_STD = (0.5, 0.5, 0.5)
_HIBOU_OFFICIAL_REMOTE = "https://github.com/HistAI/hibou.git"
_HIBOU_B_APPROVED_RELEASES: Mapping[str, Mapping[str, Any]] = {
    "c453bbe4dab0fec6f7df343b09ea87048629c58d": {
        "weights_sha256": (
            "9d3e5ebc4e1ffaf6d7a0b672273e4fbef109cdd03df73c52920d6e886f2327e1"
        ),
        "source_sha256": {
            "hibou/models/__init__.py": (
                "a0ee97aaa0e802fec397bb6eb3d09dfe1d63759ecf1c702cd1fb04b8a2d51ae8"
            ),
            "hibou/models/vision_transformer.py": (
                "4b729de30c673b3805e90fb65fc80d44b2aaacadb471959f50611d7948914b1d"
            ),
            "hibou/models/layers/__init__.py": (
                "5b5f637b2371089e34e0bec9a49518e2250f915aa9e10cae83a32f8e671ad24f"
            ),
            "hibou/models/layers/attention.py": (
                "069926c3335aaa6287058284f5f99685450fb06dbbb19e8acf7f4f5e8a78add3"
            ),
            "hibou/models/layers/block.py": (
                "c89918d40c09d846c7b38979079429ed98c90bf087dced234e8821de3cc3dead"
            ),
            "hibou/models/layers/dino_head.py": (
                "909bcae0f694da055809bb23815873010e809c7a91c63e90f693f3477e887eb4"
            ),
            "hibou/models/layers/drop_path.py": (
                "81471280a70c0282f24f482b4b27656aac652851472a2bd06cf5b2bb44cb1783"
            ),
            "hibou/models/layers/layer_scale.py": (
                "abb7bbfae152a9de2e6d0961ab5e75c79428e849281bfe50dd07b66b54b485d1"
            ),
            "hibou/models/layers/mlp.py": (
                "255825c73b60a916dd00eb1e38aacbcdbf316e40d6a005efb46e245b7edb43aa"
            ),
            "hibou/models/layers/patch_embed.py": (
                "cc295a8b139a642c77eaa2b3cc675b108fc341ff6456d4e00a81587bd04ad1e0"
            ),
            "hibou/models/layers/swiglu_ffn.py": (
                "816539ba3958644009cb45fc00f033881574843beedaa41028ca191fcedea271"
            ),
            "README.md": (
                "efa8b0fcbf36a4c1652afedbb6f14d497ca59bf040faa9fe9ee9ee638cf9b4f4"
            ),
            "LICENSE": (
                "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
            ),
        },
    }
}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
_HEX_GIT_COMMIT = re.compile(r"[0-9a-f]{40}")
_SIGLIP2_BASE_MODEL_ID = "google/siglip2-base-patch16-224"
_SIGLIP2_BASE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
_SIGLIP2_BASE_ASSET_SHA256 = {
    "model.safetensors": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
    "config.json": "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
    "preprocessor_config.json": "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
    "README.md": "39ac3705d62af9ffa1a14675b8ccb220a75f2d81acd530e564a3b1e3dfe418d8",
}


def _import_torch() -> Any:
    """Import torch lazily and provide an actionable optional-dependency error."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "Foundation feature extraction requires PyTorch. Install the "
            "project's 'foundation' extra or inject an approved torch runtime."
        ) from error
    return torch


def _validate_semantic_channels(
    semantic_channels: Sequence[str],
) -> tuple[str, str, str]:
    channels = tuple(str(channel).strip() for channel in semantic_channels)
    if len(channels) != 3 or any(not channel for channel in channels):
        raise ValueError(
            "semantic_channels must name exactly three non-empty channels in RGB "
            "display order"
        )
    if len({channel.casefold() for channel in channels}) != 3:
        raise ValueError("semantic_channels must contain three distinct names")
    return channels  # type: ignore[return-value]


def validate_semantic_rgb(
    images: ArrayLike,
    *,
    semantic_channels: Sequence[str] = ("red", "green", "blue"),
) -> Float32Array:
    """Validate and scale an HWC/NHWC semantic RGB image batch to ``[0, 1]``.

    Three-channel input is required.  A multiplex caller must first construct
    an explicit, documented RGB projection and pass its channel names through
    ``semantic_channels``.  Floating-point data must already use the unit
    interval; unsigned integer inputs are scaled by their dtype maximum.

    Returns
    -------
    numpy.ndarray
        A contiguous ``(N, H, W, 3)`` float32 array.
    """

    _validate_semantic_channels(semantic_channels)
    array = np.asarray(images)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4:
        raise ValueError("images must have shape (H,W,3) or (N,H,W,3)")
    if array.shape[-1] != 3:
        raise ValueError(
            "Foundation encoders require an explicit three-channel semantic RGB "
            "projection; multiplex channels are never selected or truncated "
            "implicitly"
        )
    if min(array.shape[1:3]) <= 0 or array.shape[0] <= 0:
        raise ValueError("images must contain at least one non-empty image")
    if np.issubdtype(array.dtype, np.unsignedinteger):
        maximum = float(np.iinfo(array.dtype).max)
        normalized = array.astype(np.float32) / maximum
    elif np.issubdtype(array.dtype, np.floating):
        normalized = array.astype(np.float32, copy=False)
        if not np.isfinite(normalized).all():
            raise ValueError("images must contain only finite values")
        low = float(np.min(normalized))
        high = float(np.max(normalized))
        if low < 0.0 or high > 1.0:
            raise ValueError(
                "floating-point semantic RGB images must already lie in [0, 1]"
            )
    else:
        raise TypeError(
            "semantic RGB images must use an unsigned-integer or floating dtype"
        )
    return np.ascontiguousarray(normalized, dtype=np.float32)


def preprocess_dinov2_rgb(
    images: ArrayLike,
    *,
    image_size: int | Sequence[int] = 224,
    semantic_channels: Sequence[str] = ("red", "green", "blue"),
    normalization_mean: Sequence[float] = tuple(_IMAGENET_MEAN),
    normalization_std: Sequence[float] = tuple(_IMAGENET_STD),
) -> Float32Array:
    """Deterministically resize and normalize semantic RGB patches.

    The returned array is NCHW and ready to convert to a torch tensor.  Direct
    resizing is appropriate here because inputs are already extracted patches;
    no content-dependent crop or per-image intensity normalization is applied.
    The defaults preserve the original ImageNet normalization used by DINOv2;
    pathology encoders can supply their published RGB mean and standard
    deviation explicitly.
    """

    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        size_values = tuple(int(value) for value in image_size)
        if len(size_values) != 2:
            raise ValueError("image_size must be an integer or (height, width)")
        output_size = size_values
    else:
        side = int(image_size)
        output_size = (side, side)
    if min(output_size) <= 0:
        raise ValueError("image_size values must be positive")
    mean, std = _validate_normalization(normalization_mean, normalization_std)

    batch = validate_semantic_rgb(
        images,
        semantic_channels=semantic_channels,
    )
    # OpenCV is a core dependency and gives a deterministic bicubic resize on
    # CPU without importing the optional torch stack.
    import cv2

    resized = np.empty(
        (batch.shape[0], output_size[0], output_size[1], 3),
        dtype=np.float32,
    )
    for index, image in enumerate(batch):
        resized[index] = cv2.resize(
            image,
            (output_size[1], output_size[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    # Cubic interpolation can overshoot the unit interval by a tiny amount.
    resized = np.clip(resized, 0.0, 1.0)
    normalized = (resized - mean) / std
    return np.ascontiguousarray(np.moveaxis(normalized, -1, 1), dtype=np.float32)


def _validate_normalization(
    mean: Sequence[float], std: Sequence[float]
) -> tuple[Float32Array, Float32Array]:
    mean_array = np.asarray(tuple(mean), dtype=np.float32)
    std_array = np.asarray(tuple(std), dtype=np.float32)
    if mean_array.shape != (3,) or std_array.shape != (3,):
        raise ValueError("normalization mean and std must each contain three values")
    if not np.isfinite(mean_array).all() or not np.isfinite(std_array).all():
        raise ValueError("normalization mean and std must contain only finite values")
    if np.any(std_array <= 0):
        raise ValueError("normalization std values must be positive")
    return mean_array.reshape(1, 1, 3), std_array.reshape(1, 1, 3)


def select_torch_device(
    requested: str = "auto",
    *,
    torch_module: Any | None = None,
) -> str:
    """Resolve ``auto``, ``mps``, or ``cpu`` without silently changing requests."""

    torch = _import_torch() if torch_module is None else torch_module
    requested = str(requested).lower()
    if requested not in {"auto", "mps", "cpu"}:
        raise ValueError("device must be one of 'auto', 'mps', or 'cpu'")
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(
        mps_backend is not None
        and callable(getattr(mps_backend, "is_available", None))
        and mps_backend.is_available()
    )
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    if requested == "mps" and not mps_available:
        raise RuntimeError(
            "MPS was requested but is unavailable in this PyTorch/macOS runtime"
        )
    return requested


@dataclass(frozen=True)
class LoadedLocalFoundationModel:
    """A locally built foundation model plus reproducibility evidence."""

    model: Any
    provenance: Mapping[str, Any]
    preprocessor: Callable[..., Float32Array] | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_output(source: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(source), *arguments),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(
            f"Hibou source must be a readable local Git checkout: {source}"
        ) from error
    return completed.stdout.strip()


def _import_hibou_build_model(source: Path) -> Any:
    """Load the official ``hibou.models.build_model`` without CellViT extras."""

    package_root = source / "hibou"
    models_root = package_root / "models"
    models_init = models_root / "__init__.py"
    if not (package_root / "__init__.py").is_file() or not models_init.is_file():
        raise ValueError("Hibou source does not expose hibou.models.build_model")
    namespace_digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
    package_name = f"_foldcrack_hibou_{namespace_digest}"
    models_name = f"{package_name}.models"
    existing = sys.modules.get(models_name)
    if existing is not None:
        build_model = getattr(existing, "build_model", None)
        if callable(build_model):
            return build_model

    package = ModuleType(package_name)
    package.__path__ = [str(package_root)]  # type: ignore[attr-defined]
    package.__package__ = package_name
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        models_name,
        models_init,
        submodule_search_locations=[str(models_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError("Unable to construct an import spec for Hibou build_model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[models_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(f"{package_name}."):
                sys.modules.pop(name, None)
        raise
    build_model = getattr(module, "build_model", None)
    if not callable(build_model):
        raise TypeError("Official Hibou source does not define build_model")
    return build_model


def _load_strict_torch_state_dict(model: Any, weights: Path) -> None:
    """Load an audited tensor-only checkpoint with an exact state contract."""

    torch = _import_torch()
    try:
        state = torch.load(
            str(weights),
            map_location="cpu",
            weights_only=True,
        )
    except TypeError as error:  # pragma: no cover - depends on torch version
        raise RuntimeError(
            "The Hibou loader requires torch.load(weights_only=True); upgrade "
            "the approved PyTorch runtime"
        ) from error
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Hibou checkpoint must contain a non-empty tensor state dict")
    if any(not isinstance(key, str) for key in state):
        raise ValueError("Hibou state-dict keys must be strings")
    if any(
        not bool(getattr(value, "is_floating_point", lambda: False)())
        for value in state.values()
    ):
        raise ValueError("Hibou state dict must contain only floating-point tensors")
    try:
        incompatibility = model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            "Hibou checkpoint keys or tensor shapes do not match the audited architecture"
        ) from error
    if tuple(getattr(incompatibility, "missing_keys", ())) or tuple(
        getattr(incompatibility, "unexpected_keys", ())
    ):
        raise ValueError("Hibou strict state load reported missing or unexpected keys")


def load_local_hibou_b(
    weights_path: str | Path,
    source_path: str | Path,
    *,
    expected_weights_sha256: str | None = None,
    expected_source_commit: str | None = None,
) -> LoadedLocalFoundationModel:
    """Build Hibou-B from an explicit, clean official checkout and local weights.

    This loader performs no network access and never uses ``trust_remote_code``.
    It hashes the weight file, verifies the Git origin/commit/cleanliness, then
    invokes the checked-out official ``build_model`` implementation.
    """

    if expected_weights_sha256 is None or expected_source_commit is None:
        raise ValueError(
            "Hibou loading requires both an approved weights SHA-256 and source commit"
        )
    lexical_weights = Path(weights_path).expanduser()
    lexical_source = Path(source_path).expanduser()
    if lexical_weights.is_symlink() or not lexical_weights.is_file():
        raise ValueError("Hibou weights must be an explicit regular local .pth file")
    if lexical_weights.suffix.casefold() != ".pth":
        raise ValueError("Hibou weights path must end in .pth")
    if lexical_source.is_symlink() or not lexical_source.is_dir():
        raise ValueError("Hibou source must be an explicit local checkout directory")
    weights = lexical_weights.resolve()
    source = lexical_source.resolve()

    expected_digest = expected_weights_sha256.lower()
    if _HEX_SHA256.fullmatch(expected_digest) is None:
        raise ValueError(
            "expected Hibou weights SHA-256 must be 64 lowercase hex characters"
        )
    weights_digest = _sha256_file(weights)
    if weights_digest != expected_digest:
        raise ValueError(
            "Hibou weights SHA-256 mismatch: the selected .pth is not the locked asset"
        )

    commit = _git_output(source, "rev-parse", "HEAD").lower()
    if _HEX_GIT_COMMIT.fullmatch(commit) is None:
        raise ValueError("Hibou source checkout did not resolve to a full Git commit")
    expected_commit = expected_source_commit.lower()
    if _HEX_GIT_COMMIT.fullmatch(expected_commit) is None:
        raise ValueError(
            "expected Hibou source commit must be 40 lowercase hex characters"
        )
    if commit != expected_commit:
        raise ValueError("Hibou source commit does not match the locked commit")
    approved = _HIBOU_B_APPROVED_RELEASES.get(expected_commit)
    if approved is None or approved.get("weights_sha256") != expected_digest:
        raise ValueError(
            "Hibou source/weight identity is not in the approved release allowlist"
        )
    dirty = _git_output(source, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise ValueError(
            "Hibou source checkout has tracked or untracked modifications; "
            "use a clean checkout"
        )
    origin = _git_output(source, "config", "--get", "remote.origin.url")
    normalized_origin = origin.casefold().removesuffix(".git")
    if normalized_origin not in {
        "https://github.com/histai/hibou",
        "git@github.com:histai/hibou",
    }:
        raise ValueError(
            "Hibou checkout origin must be the official HistAI/hibou repository"
        )

    expected_source_hashes = approved.get("source_sha256")
    if not isinstance(expected_source_hashes, Mapping):
        raise TypeError("Approved Hibou release is missing source-file identities")
    source_files = {
        str(relative_name): source / str(relative_name)
        for relative_name in expected_source_hashes
    }
    if any(path.is_symlink() or not path.is_file() for path in source_files.values()):
        raise ValueError(
            "Hibou checkout is missing an audited regular source/license file"
        )
    license_path = source_files["LICENSE"]
    build_path = source_files["hibou/models/__init__.py"]
    vision_path = source_files["hibou/models/vision_transformer.py"]
    readme_path = source_files["README.md"]
    try:
        license_text = license_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError("Hibou LICENSE is unreadable") from error
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Hibou checkout LICENSE is not the declared Apache-2.0 text")

    for relative_name, path in source_files.items():
        if _sha256_file(path) != expected_source_hashes.get(relative_name):
            raise ValueError(f"Hibou audited source hash mismatch: {relative_name}")

    build_model = _import_hibou_build_model(source)
    model = build_model(
        None,
        img_size=224,
        arch="vit_base",
        patch_size=14,
        num_register_tokens=4,
    )
    _load_strict_torch_state_dict(model, weights)
    if not callable(getattr(model, "forward_features", None)):
        raise TypeError("Hibou-B model must expose mapping-style forward_features")
    if int(getattr(model, "patch_size", -1)) != 14:
        raise ValueError("Hibou-B model patch geometry does not match patch14")
    if int(getattr(model, "num_register_tokens", -1)) != 4:
        raise ValueError("Hibou-B model must expose exactly four register tokens")

    provenance = {
        "id": "HistAI/Hibou-B",
        "provider": "HistAI",
        "architecture": "vit_base_patch14_reg4",
        "loader": "audited_checkout_build_plus_strict_weights_only_state_load",
        "weights": {
            "path": str(weights),
            "sha256": weights_digest,
            "size_bytes": weights.stat().st_size,
        },
        "source": {
            "path": str(source),
            "repository": _HIBOU_OFFICIAL_REMOTE,
            "recorded_origin": origin,
            "commit": commit,
            "worktree_clean_including_untracked": True,
            "audited_executable_source_sha256": {
                relative_name: _sha256_file(path)
                for relative_name, path in source_files.items()
                if relative_name.endswith(".py")
            },
            "build_model_sha256": _sha256_file(build_path),
            "vision_transformer_sha256": _sha256_file(vision_path),
            "readme_sha256": _sha256_file(readme_path),
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidence_path": str(license_path),
            "evidence_sha256": _sha256_file(license_path),
        },
        "input": {
            "image_size": [224, 224],
            "patch_size": [14, 14],
            "register_tokens": 4,
            "normalization_mean": list(HIBOU_B_MEAN),
            "normalization_std": list(HIBOU_B_STD),
        },
        "trust_remote_code": False,
        "network_access_allowed": False,
    }
    return LoadedLocalFoundationModel(model=model, provenance=provenance)


def _import_siglip_vision_model() -> Any:
    """Import the standard Transformers SigLIP vision class lazily."""

    try:
        from transformers import SiglipVisionModel
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "SigLIP2 feature extraction requires the project's 'foundation' extra."
        ) from error
    return SiglipVisionModel


def _import_siglip_image_processor() -> Any:
    """Import the standard locked SigLIP image processor lazily."""

    try:
        from transformers import SiglipImageProcessor
    except ImportError as error:  # pragma: no cover - depends on environment
        raise ImportError(
            "SigLIP2 preprocessing requires the project's 'foundation' extra."
        ) from error
    return SiglipImageProcessor


def _locked_siglip2_preprocessor(processor: Any) -> Callable[..., Float32Array]:
    """Wrap the official processor with an explicit semantic-RGB uint8 boundary."""

    def preprocess(
        images: ArrayLike,
        *,
        semantic_channels: Sequence[str] = ("red", "green", "blue"),
    ) -> Float32Array:
        normalized = validate_semantic_rgb(
            images,
            semantic_channels=semantic_channels,
        )
        # The upstream processor's published rescale factor is 1/255. Convert
        # every supported source dtype through one declared 8-bit RGB boundary
        # before invoking its exact resize/rescale/normalize implementation.
        uint8_batch = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
        processed = processor(
            images=[image for image in uint8_batch],
            return_tensors="np",
        )
        values = (
            processed.get("pixel_values")
            if isinstance(processed, Mapping)
            else getattr(processed, "pixel_values", None)
        )
        output = np.asarray(values, dtype=np.float32)
        if output.ndim != 4 or output.shape[1:] != (3, 224, 224):
            raise ValueError(
                "Locked SigLIP2 processor must return NCHW float32 at 224x224"
            )
        if output.shape[0] != uint8_batch.shape[0] or not np.isfinite(output).all():
            raise ValueError("Locked SigLIP2 processor returned invalid values")
        return np.ascontiguousarray(output)

    return preprocess


def load_local_siglip2_base_vision(
    snapshot_path: str | Path,
) -> LoadedLocalFoundationModel:
    """Load the hash-locked public SigLIP2 Base vision tower offline.

    The accepted files are pinned to the exact public Hugging Face revision
    recorded in :data:`_SIGLIP2_BASE_REVISION`.  The loader performs no network
    access, never executes remote code, verifies the Apache-2.0 model-card
    evidence and official preprocessing contract, and loads only the vision
    tower through the standard Transformers implementation.

    Notes
    -----
    The upstream safetensors file also contains the text tower.  Transformers
    ignores those keys when constructing ``SiglipVisionModel``; provenance
    therefore records the hash of the complete official upstream checkpoint.
    """

    lexical_root = Path(snapshot_path).expanduser()
    if lexical_root.is_symlink() or not lexical_root.is_dir():
        raise ValueError("SigLIP2 snapshot must be an explicit local directory")
    root = lexical_root.resolve()
    assets = {name: root / name for name in _SIGLIP2_BASE_ASSET_SHA256}
    if any(path.is_symlink() or not path.is_file() for path in assets.values()):
        raise ValueError(
            "SigLIP2 snapshot is missing a required regular model, configuration, "
            "preprocessor, or license-evidence file"
        )
    observed_digests = {name: _sha256_file(path) for name, path in assets.items()}
    mismatches = sorted(
        name
        for name, expected in _SIGLIP2_BASE_ASSET_SHA256.items()
        if observed_digests[name] != expected
    )
    if mismatches:
        raise ValueError(
            "SigLIP2 snapshot hash mismatch for locked asset(s): "
            + ", ".join(mismatches)
        )

    try:
        configuration = json.loads(assets["config.json"].read_text(encoding="utf-8"))
        preprocessor = json.loads(
            assets["preprocessor_config.json"].read_text(encoding="utf-8")
        )
        readme = assets["README.md"].read_text(encoding="utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("SigLIP2 snapshot metadata is unreadable") from error
    vision_configuration = configuration.get("vision_config")
    if not isinstance(vision_configuration, Mapping):
        raise TypeError("SigLIP2 config does not expose a vision configuration")
    if configuration.get("model_type") != "siglip" or (
        vision_configuration.get("model_type") != "siglip_vision_model"
    ):
        raise ValueError(
            "SigLIP2 config does not identify the locked vision architecture"
        )
    if preprocessor.get("image_mean") != list(SIGLIP2_BASE_MEAN) or (
        preprocessor.get("image_std") != list(SIGLIP2_BASE_STD)
    ):
        raise ValueError("SigLIP2 preprocessor normalization does not match the lock")
    if preprocessor.get("size") != {"height": 224, "width": 224}:
        raise ValueError("SigLIP2 preprocessor image size does not match 224x224")
    if (
        preprocessor.get("resample") != 2
        or preprocessor.get("do_resize") is not True
        or preprocessor.get("do_rescale") is not True
        or preprocessor.get("do_normalize") is not True
        or float(preprocessor.get("rescale_factor", -1.0)) != 1.0 / 255.0
    ):
        raise ValueError("SigLIP2 resize/rescale processor contract does not match")
    if re.search(r"(?mi)^license:\s*apache-2\.0\s*$", readme) is None:
        raise ValueError("SigLIP2 model card does not declare Apache-2.0")

    model_class = _import_siglip_vision_model()
    processor_class = _import_siglip_image_processor()
    image_processor = processor_class.from_pretrained(
        str(root),
        local_files_only=True,
        token=False,
    )
    model = model_class.from_pretrained(
        str(root),
        local_files_only=True,
        trust_remote_code=False,
        token=False,
        use_safetensors=True,
    )
    model_configuration = getattr(model, "config", None)
    geometry = (
        int(getattr(model_configuration, "image_size", -1)),
        int(getattr(model_configuration, "patch_size", -1)),
        int(getattr(model_configuration, "hidden_size", -1)),
    )
    if geometry != (224, 16, 768):
        raise ValueError(
            "Loaded SigLIP2 vision geometry does not match image224/patch16/dim768"
        )

    provenance = {
        "id": _SIGLIP2_BASE_MODEL_ID,
        "provider": "Google",
        "architecture": "siglip2_base_vit_b_patch16_224",
        "loader": "transformers.SiglipVisionModel.from_pretrained",
        "source": {
            "repository": f"https://huggingface.co/{_SIGLIP2_BASE_MODEL_ID}",
            "revision": _SIGLIP2_BASE_REVISION,
        },
        "assets": {
            name: {
                "path": str(path),
                "sha256": observed_digests[name],
                "size_bytes": path.stat().st_size,
            }
            for name, path in assets.items()
        },
        "license": {
            "spdx": "Apache-2.0",
            "evidence_path": str(assets["README.md"]),
            "evidence_sha256": observed_digests["README.md"],
        },
        "input": {
            "image_size": [224, 224],
            "patch_size": [16, 16],
            "normalization_mean": list(SIGLIP2_BASE_MEAN),
            "normalization_std": list(SIGLIP2_BASE_STD),
            "channels": "explicit semantic RGB only",
            "source_dtype_boundary": "round_clipped_unit_RGB_to_uint8",
            "processor": "transformers.SiglipImageProcessor",
            "resample": 2,
            "resample_semantics": "PIL.Image.Resampling.BILINEAR",
            "rescale_factor": 1.0 / 255.0,
        },
        "output": {
            "global_embedding": "pooler_output",
            "spatial_embedding": "last_hidden_state",
            "prefix_tokens": 0,
            "grid_shape": [14, 14],
            "embedding_dim": 768,
        },
        "trust_remote_code": False,
        "token_used": False,
        "network_access_allowed": False,
    }
    return LoadedLocalFoundationModel(
        model=model,
        provenance=provenance,
        preprocessor=_locked_siglip2_preprocessor(image_processor),
    )


@dataclass(frozen=True)
class FoundationFeatures:
    """Frozen encoder output with both global and spatial representations."""

    cls_embedding: Float32Array
    patch_grid: Float32Array
    input_size: tuple[int, int]
    patch_size: tuple[int, int]
    semantic_channels: tuple[str, str, str]

    def __post_init__(self) -> None:
        cls = np.asarray(self.cls_embedding, dtype=np.float32)
        grid = np.asarray(self.patch_grid, dtype=np.float32)
        if cls.ndim != 2:
            raise ValueError("cls_embedding must have shape (N,D)")
        if grid.ndim != 4:
            raise ValueError("patch_grid must have shape (N,grid_h,grid_w,D)")
        if cls.shape[0] != grid.shape[0] or cls.shape[1] != grid.shape[-1]:
            raise ValueError(
                "CLS and patch embeddings must share batch and embedding dimensions"
            )
        if min(grid.shape) <= 0 or min(cls.shape) <= 0:
            raise ValueError("foundation embeddings must be non-empty")
        if not np.isfinite(cls).all() or not np.isfinite(grid).all():
            raise ValueError("foundation embeddings must contain only finite values")
        input_size = tuple(int(value) for value in self.input_size)
        patch_size = tuple(int(value) for value in self.patch_size)
        if len(input_size) != 2 or min(input_size) <= 0:
            raise ValueError("input_size must contain two positive integers")
        if len(patch_size) != 2 or min(patch_size) <= 0:
            raise ValueError("patch_size must contain two positive integers")
        channels = _validate_semantic_channels(self.semantic_channels)
        object.__setattr__(self, "cls_embedding", np.ascontiguousarray(cls))
        object.__setattr__(self, "patch_grid", np.ascontiguousarray(grid))
        object.__setattr__(self, "input_size", input_size)
        object.__setattr__(self, "patch_size", patch_size)
        object.__setattr__(self, "semantic_channels", channels)

    @property
    def batch_size(self) -> int:
        return int(self.cls_embedding.shape[0])

    @property
    def grid_shape(self) -> tuple[int, int]:
        return int(self.patch_grid.shape[1]), int(self.patch_grid.shape[2])

    @property
    def embedding_dim(self) -> int:
        return int(self.cls_embedding.shape[1])


class DINOv2FeatureExtractor:
    """Frozen ViT encoder returning global and spatial patch tokens.

    Parameters
    ----------
    model:
        An already loaded and approved torch model.  Hugging Face-style outputs
        exposing ``last_hidden_state`` and DINOv2 ``forward_features`` mappings
        are supported.
    model_input_name:
        The model keyword receiving the image tensor.  Hugging Face models use
        ``"pixel_values"``.  Set to ``None`` for a positional torch-hub model.
    prefix_tokens:
        Number of non-spatial tokens preceding the patch tokens.  Standard
        DINOv2 uses one CLS token.  Models with register tokens must declare the
        larger value explicitly so token/grid alignment cannot be guessed.  A
        model with no prefix token must set this to zero and explicitly name a
        separate global output through ``global_embedding_name``.
    global_embedding_name:
        Named model output to use as the global embedding when
        ``prefix_tokens=0``.  For example, fixed-resolution SigLIP2 vision
        towers expose ``"pooler_output"``.  It is ignored when a CLS prefix
        token is present.
    normalization_mean, normalization_std:
        Published RGB input normalization.  The defaults are ImageNet and
        preserve the original DINOv2 behavior.
    preprocessor:
        Optional locked callable returning NCHW float32. SigLIP2 uses its
        standard offline ``SiglipImageProcessor`` so resize semantics are not
        silently replaced by the DINOv2 bicubic path.
    """

    def __init__(
        self,
        model: Any,
        *,
        device: str = "auto",
        image_size: int | Sequence[int] = 224,
        patch_size: int | Sequence[int] = 14,
        prefix_tokens: int = 1,
        model_input_name: str | None = "pixel_values",
        global_embedding_name: str | None = None,
        normalization_mean: Sequence[float] = tuple(_IMAGENET_MEAN),
        normalization_std: Sequence[float] = tuple(_IMAGENET_STD),
        preprocessor: Callable[..., Float32Array] | None = None,
        torch_module: Any | None = None,
    ) -> None:
        if model is None:
            raise ValueError("model must be an injected, already-loaded torch model")
        self._torch = _import_torch() if torch_module is None else torch_module
        self.device = select_torch_device(device, torch_module=self._torch)
        self.image_size = self._normalize_pair(image_size, "image_size")
        self.patch_size = self._normalize_pair(patch_size, "patch_size")
        if any(
            image % patch != 0
            for image, patch in zip(self.image_size, self.patch_size, strict=True)
        ):
            raise ValueError("image_size must be divisible by patch_size")
        if prefix_tokens < 0:
            raise ValueError("prefix_tokens cannot be negative")
        if model_input_name is not None and not str(model_input_name).strip():
            raise ValueError("model_input_name must be non-empty or None")
        if global_embedding_name is not None and not str(global_embedding_name).strip():
            raise ValueError("global_embedding_name must be non-empty or None")
        if prefix_tokens == 0 and global_embedding_name is None:
            raise ValueError(
                "prefix_tokens=0 requires an explicit global_embedding_name"
            )
        self.prefix_tokens = int(prefix_tokens)
        self.model_input_name = model_input_name
        self.global_embedding_name = global_embedding_name
        mean, std = _validate_normalization(normalization_mean, normalization_std)
        self.normalization_mean = tuple(float(value) for value in mean.reshape(-1))
        self.normalization_std = tuple(float(value) for value in std.reshape(-1))
        self.preprocessor = preprocessor
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _normalize_pair(value: int | Sequence[int], name: str) -> tuple[int, int]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values = tuple(int(item) for item in value)
            if len(values) != 2:
                raise ValueError(f"{name} must be an integer or a pair")
        else:
            side = int(value)
            values = (side, side)
        if min(values) <= 0:
            raise ValueError(f"{name} values must be positive")
        return values  # type: ignore[return-value]

    @property
    def grid_shape(self) -> tuple[int, int]:
        return tuple(
            image // patch
            for image, patch in zip(
                self.image_size,
                self.patch_size,
                strict=True,
            )
        )  # type: ignore[return-value]

    def _forward(self, tensor: Any) -> Any:
        if self.model_input_name is None:
            forward_features = getattr(self.model, "forward_features", None)
            if callable(forward_features):
                return forward_features(tensor)
            return self.model(tensor)
        return self.model(**{self.model_input_name: tensor})

    def _tokens_from_output(self, output: Any) -> tuple[Any, Any]:
        torch = self._torch
        cls_token = None
        patch_tokens = None
        hidden = None
        named_global = None

        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
            if self.global_embedding_name is not None:
                named_global = getattr(output, self.global_embedding_name, None)
        elif isinstance(output, Mapping):
            if "x_norm_clstoken" in output and "x_norm_patchtokens" in output:
                cls_token = output["x_norm_clstoken"]
                patch_tokens = output["x_norm_patchtokens"]
            elif "last_hidden_state" in output:
                hidden = output["last_hidden_state"]
                if self.global_embedding_name is not None:
                    named_global = output.get(self.global_embedding_name)
        elif isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        elif torch.is_tensor(output):
            hidden = output

        if hidden is not None:
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise TypeError(
                    "last_hidden_state must be a (batch,tokens,embedding) tensor"
                )
            expected_patches = int(np.prod(self.grid_shape))
            expected_tokens = self.prefix_tokens + expected_patches
            if int(hidden.shape[1]) != expected_tokens:
                raise ValueError(
                    f"Model returned {hidden.shape[1]} tokens; expected "
                    f"{expected_tokens} ({self.prefix_tokens} prefix + "
                    f"{expected_patches} spatial). Configure patch_size and "
                    "prefix_tokens explicitly."
                )
            cls_token = hidden[:, 0, :] if self.prefix_tokens else named_global
            patch_tokens = hidden[:, self.prefix_tokens :, :]

        if not torch.is_tensor(cls_token) or not torch.is_tensor(patch_tokens):
            raise TypeError(
                "Model output must expose last_hidden_state with the configured "
                "global embedding, or both x_norm_clstoken and x_norm_patchtokens"
            )
        if cls_token.ndim != 2 or patch_tokens.ndim != 3:
            raise ValueError("Model CLS/patch tensors have incompatible ranks")
        expected_patches = int(np.prod(self.grid_shape))
        if int(patch_tokens.shape[1]) != expected_patches:
            raise ValueError(
                f"Model returned {patch_tokens.shape[1]} patch tokens; expected "
                f"{expected_patches} for grid {self.grid_shape}"
            )
        if int(cls_token.shape[0]) != int(patch_tokens.shape[0]) or int(
            cls_token.shape[-1]
        ) != int(patch_tokens.shape[-1]):
            raise ValueError("Model CLS and patch tokens have incompatible shapes")
        return cls_token, patch_tokens

    def encode(
        self,
        images: ArrayLike,
        *,
        semantic_channels: Sequence[str] = ("red", "green", "blue"),
        batch_size: int = 8,
    ) -> FoundationFeatures:
        """Extract deterministic frozen embeddings in bounded batches."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        channels = _validate_semantic_channels(semantic_channels)
        if self.preprocessor is None:
            preprocessed = preprocess_dinov2_rgb(
                images,
                image_size=self.image_size,
                semantic_channels=channels,
                normalization_mean=self.normalization_mean,
                normalization_std=self.normalization_std,
            )
        else:
            preprocessed = np.asarray(
                self.preprocessor(images, semantic_channels=channels),
                dtype=np.float32,
            )
            if (
                preprocessed.ndim != 4
                or preprocessed.shape[0] <= 0
                or tuple(preprocessed.shape[1:])
                != (3, self.image_size[0], self.image_size[1])
                or not np.isfinite(preprocessed).all()
            ):
                raise ValueError(
                    "Locked foundation preprocessor returned an invalid NCHW batch"
                )
            preprocessed = np.ascontiguousarray(preprocessed)
        cls_batches: list[Float32Array] = []
        patch_batches: list[Float32Array] = []
        torch = self._torch
        with torch.inference_mode():
            for start in range(0, preprocessed.shape[0], batch_size):
                numpy_batch = preprocessed[start : start + batch_size]
                tensor = torch.as_tensor(
                    numpy_batch,
                    dtype=torch.float32,
                    device=self.device,
                )
                output = self._forward(tensor)
                cls_token, patch_tokens = self._tokens_from_output(output)
                cls_batches.append(
                    cls_token.detach().to("cpu").numpy().astype(np.float32, copy=False)
                )
                patch_batches.append(
                    patch_tokens.detach()
                    .to("cpu")
                    .numpy()
                    .astype(np.float32, copy=False)
                )
        cls = np.concatenate(cls_batches, axis=0)
        flat_patches = np.concatenate(patch_batches, axis=0)
        grid = flat_patches.reshape(
            flat_patches.shape[0],
            self.grid_shape[0],
            self.grid_shape[1],
            flat_patches.shape[-1],
        )
        return FoundationFeatures(
            cls_embedding=cls,
            patch_grid=grid,
            input_size=self.image_size,
            patch_size=self.patch_size,
            semantic_channels=channels,
        )


def _as_patch_matrix(
    features: FoundationFeatures | ArrayLike,
) -> tuple[Float64Array, tuple[int, int, int] | None]:
    array = (
        features.patch_grid
        if isinstance(features, FoundationFeatures)
        else np.asarray(features)
    )
    array = np.asarray(array, dtype=np.float64)
    grid_shape: tuple[int, int, int] | None = None
    if array.ndim == 4:
        grid_shape = (int(array.shape[0]), int(array.shape[1]), int(array.shape[2]))
        matrix = array.reshape(-1, array.shape[-1])
    elif array.ndim == 2:
        matrix = array
    else:
        raise ValueError(
            "patch features must have shape (tokens,D) or (N,grid_h,grid_w,D)"
        )
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("patch features must be non-empty")
    if not np.isfinite(matrix).all():
        raise ValueError("patch features must contain only finite values")
    return np.ascontiguousarray(matrix), grid_shape


def _feature_fingerprint(matrix: Float64Array) -> str:
    contiguous = np.ascontiguousarray(matrix, dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


class PatchKNNAnomalyScorer:
    """PatchCore-like nearest-neighbour scorer with held-out calibration.

    ``fit`` builds only the nominal memory bank.  ``calibrate`` is a required,
    separate operation over an independent reviewed-clean split.  Reusing the
    fit bank (including an exact array copy) is rejected.  Explicit split IDs
    provide an additional audit guard when feature arrays are regenerated.
    """

    def __init__(
        self,
        *,
        neighbors: int = 1,
        calibration_quantile: float = 0.995,
        l2_normalize: bool = True,
        distance_chunk_size: int = 2048,
        max_reference_tokens: int | None = None,
    ) -> None:
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if not 0.5 < calibration_quantile < 1.0:
            raise ValueError("calibration_quantile must lie in (0.5, 1)")
        if distance_chunk_size <= 0:
            raise ValueError("distance_chunk_size must be positive")
        if max_reference_tokens is not None and max_reference_tokens <= 0:
            raise ValueError("max_reference_tokens must be positive")
        self.neighbors = int(neighbors)
        self.calibration_quantile = float(calibration_quantile)
        self.l2_normalize = bool(l2_normalize)
        self.distance_chunk_size = int(distance_chunk_size)
        self.max_reference_tokens = max_reference_tokens
        self._fitted = False
        self._calibrated = False

    @staticmethod
    def _normalize(matrix: Float64Array) -> Float64Array:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return np.divide(
            matrix,
            norms,
            out=np.zeros_like(matrix),
            where=norms > np.finfo(np.float64).eps,
        )

    def fit(
        self,
        reference_features: FoundationFeatures | ArrayLike,
        *,
        split_id: str | None = None,
    ) -> PatchKNNAnomalyScorer:
        matrix, _ = _as_patch_matrix(reference_features)
        self.fit_fingerprint_ = _feature_fingerprint(matrix)
        self.fit_split_id_ = None if split_id is None else str(split_id)
        if self.l2_normalize:
            matrix = self._normalize(matrix)
        if self.max_reference_tokens is not None and (
            matrix.shape[0] > self.max_reference_tokens
        ):
            # Deterministic coverage over acquisition order.  A future large
            # benchmark may replace this with an audited k-center coreset.
            indices = np.linspace(
                0,
                matrix.shape[0] - 1,
                num=self.max_reference_tokens,
                dtype=np.int64,
            )
            matrix = matrix[indices]
        if self.neighbors > matrix.shape[0]:
            raise ValueError("neighbors cannot exceed the number of reference tokens")
        self.reference_bank_ = np.ascontiguousarray(matrix, dtype=np.float64)
        self.n_features_in_ = int(matrix.shape[1])
        self._fitted = True
        self._calibrated = False
        return self

    def _raw_scores_from_matrix(self, matrix: Float64Array) -> Float64Array:
        if not self._fitted:
            raise RuntimeError("PatchKNNAnomalyScorer must be fitted before scoring")
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected embedding dimension {self.n_features_in_}, received "
                f"{matrix.shape[1]}"
            )
        if self.l2_normalize:
            matrix = self._normalize(matrix)
        bank = self.reference_bank_
        bank_norm = np.einsum("ij,ij->i", bank, bank)[None, :]
        scores = np.empty(matrix.shape[0], dtype=np.float64)
        for start in range(0, matrix.shape[0], self.distance_chunk_size):
            query = matrix[start : start + self.distance_chunk_size]
            query_norm = np.einsum("ij,ij->i", query, query)[:, None]
            squared = np.maximum(
                query_norm + bank_norm - 2.0 * query @ bank.T,
                0.0,
            )
            nearest = np.partition(
                squared,
                kth=self.neighbors - 1,
                axis=1,
            )[:, : self.neighbors]
            scores[start : start + query.shape[0]] = np.mean(
                np.sqrt(nearest),
                axis=1,
            )
        return scores

    def raw_token_scores(
        self, features: FoundationFeatures | ArrayLike
    ) -> Float64Array:
        """Return uncalibrated token distances in flattened token order."""

        matrix, _ = _as_patch_matrix(features)
        return self._raw_scores_from_matrix(matrix)

    def calibrate(
        self,
        clean_calibration_features: FoundationFeatures | ArrayLike,
        *,
        split_id: str | None = None,
    ) -> PatchKNNAnomalyScorer:
        """Fit an empirical nominal score distribution on an independent bank."""

        if not self._fitted:
            raise RuntimeError("fit must be called before calibrate")
        matrix, _ = _as_patch_matrix(clean_calibration_features)
        fingerprint = _feature_fingerprint(matrix)
        calibration_split = None if split_id is None else str(split_id)
        if fingerprint == self.fit_fingerprint_:
            raise ValueError(
                "Calibration features duplicate the fit bank; use an independent "
                "reviewed-clean calibration split"
            )
        if (
            calibration_split is not None
            and self.fit_split_id_ is not None
            and calibration_split == self.fit_split_id_
        ):
            raise ValueError(
                "Calibration split_id matches the fit split; calibration must be "
                "slide-disjoint"
            )
        scores = self._raw_scores_from_matrix(matrix)
        if scores.size < 2:
            raise ValueError("At least two calibration tokens are required")
        self.calibration_scores_ = np.sort(scores)
        self.threshold_ = float(
            np.quantile(self.calibration_scores_, self.calibration_quantile)
        )
        self.calibration_fingerprint_ = fingerprint
        self.calibration_split_id_ = calibration_split
        self._calibrated = True
        return self

    def calibrated_token_scores(
        self, features: FoundationFeatures | ArrayLike
    ) -> Float64Array:
        """Return empirical nominal percentiles in ``[0, 1]`` for each token."""

        if not self._calibrated:
            raise RuntimeError(
                "calibrate must be called on an independent clean split before "
                "requesting calibrated scores"
            )
        raw = self.raw_token_scores(features)
        ranks = np.searchsorted(self.calibration_scores_, raw, side="right")
        return ranks.astype(np.float64) / float(self.calibration_scores_.size)

    def predict_tokens(
        self, features: FoundationFeatures | ArrayLike
    ) -> NDArray[np.bool_]:
        """Flag raw token distances beyond the held-out clean quantile."""

        if not self._calibrated:
            raise RuntimeError("calibrate must be called before predict_tokens")
        return self.raw_token_scores(features) > self.threshold_

    def score_heatmaps(
        self,
        features: FoundationFeatures | ArrayLike,
        *,
        output_shape: Sequence[int] | None = None,
        calibrated: bool = True,
    ) -> Float64Array:
        """Reconstruct per-image anomaly heatmaps from the spatial token grid."""

        matrix, grid_shape = _as_patch_matrix(features)
        if grid_shape is None:
            raise ValueError(
                "Heatmap reconstruction requires spatial (N,grid_h,grid_w,D) "
                "patch features"
            )
        scores = (
            self.calibrated_token_scores(matrix)
            if calibrated
            else self._raw_scores_from_matrix(matrix)
        )
        token_maps = scores.reshape(grid_shape)
        if output_shape is None:
            if isinstance(features, FoundationFeatures):
                output_shape = features.input_size
            else:
                output_shape = grid_shape[1:]
        reconstructed = reconstruct_anomaly_heatmaps(token_maps, output_shape)
        assert reconstructed.ndim == 3
        return reconstructed


def reconstruct_anomaly_heatmaps(
    token_scores: ArrayLike,
    output_shape: Sequence[int],
) -> Float64Array:
    """Bilinearly resize one or more spatial token maps to image resolution."""

    scores = np.asarray(token_scores, dtype=np.float64)
    single = scores.ndim == 2
    if single:
        scores = scores[None, ...]
    if scores.ndim != 3 or min(scores.shape) <= 0:
        raise ValueError("token_scores must have shape (grid_h,grid_w) or (N,H,W)")
    if not np.isfinite(scores).all():
        raise ValueError("token_scores must contain only finite values")
    shape = tuple(int(value) for value in output_shape)
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("output_shape must contain two positive integers")

    import cv2

    output = np.empty((scores.shape[0], shape[0], shape[1]), dtype=np.float64)
    for index, score_map in enumerate(scores):
        output[index] = cv2.resize(
            score_map,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    return output[0] if single else output


@dataclass(frozen=True)
class FoundationRuntimeDiagnostics:
    """Small, serializable snapshot of the optional torch/MPS runtime."""

    torch_available: bool
    torch_version: str | None
    device: str | None
    mps_built: bool
    mps_available: bool
    mps_current_allocated_bytes: int | None = None
    mps_driver_allocated_bytes: int | None = None
    mps_recommended_max_memory_bytes: int | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, bool | int | str | None]:
        return {
            "torch_available": self.torch_available,
            "torch_version": self.torch_version,
            "device": self.device,
            "mps_built": self.mps_built,
            "mps_available": self.mps_available,
            "mps_current_allocated_bytes": self.mps_current_allocated_bytes,
            "mps_driver_allocated_bytes": self.mps_driver_allocated_bytes,
            "mps_recommended_max_memory_bytes": (self.mps_recommended_max_memory_bytes),
            "error": self.error,
        }


def foundation_runtime_diagnostics(
    *,
    torch_module: Any | None = None,
) -> FoundationRuntimeDiagnostics:
    """Report torch/MPS availability and safe memory counters when supported."""

    if torch_module is None:
        try:
            torch = _import_torch()
        except ImportError as error:
            return FoundationRuntimeDiagnostics(
                torch_available=False,
                torch_version=None,
                device=None,
                mps_built=False,
                mps_available=False,
                error=str(error),
            )
    else:
        torch = torch_module

    backend = getattr(getattr(torch, "backends", None), "mps", None)
    is_built = getattr(backend, "is_built", None)
    is_available = getattr(backend, "is_available", None)
    mps_built = bool(is_built()) if callable(is_built) else False
    mps_available = bool(is_available()) if callable(is_available) else False
    mps = getattr(torch, "mps", None)

    def safe_counter(name: str) -> int | None:
        if not mps_available or mps is None:
            return None
        counter = getattr(mps, name, None)
        if not callable(counter):
            return None
        try:
            return int(counter())
        except (RuntimeError, TypeError, ValueError):
            return None

    return FoundationRuntimeDiagnostics(
        torch_available=True,
        torch_version=str(getattr(torch, "__version__", "unknown")),
        device="mps" if mps_available else "cpu",
        mps_built=mps_built,
        mps_available=mps_available,
        mps_current_allocated_bytes=safe_counter("current_allocated_memory"),
        mps_driver_allocated_bytes=safe_counter("driver_allocated_memory"),
        mps_recommended_max_memory_bytes=safe_counter("recommended_max_memory"),
    )


__all__ = [
    "HIBOU_B_MEAN",
    "HIBOU_B_STD",
    "SIGLIP2_BASE_MEAN",
    "SIGLIP2_BASE_STD",
    "DINOv2FeatureExtractor",
    "FoundationFeatures",
    "FoundationRuntimeDiagnostics",
    "LoadedLocalFoundationModel",
    "PatchKNNAnomalyScorer",
    "foundation_runtime_diagnostics",
    "load_local_hibou_b",
    "load_local_siglip2_base_vision",
    "preprocess_dinov2_rgb",
    "reconstruct_anomaly_heatmaps",
    "select_torch_device",
    "validate_semantic_rgb",
]
