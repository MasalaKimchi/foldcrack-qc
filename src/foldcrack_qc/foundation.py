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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]

_IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
_IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


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
) -> Float32Array:
    """Deterministically resize and ImageNet-normalize semantic RGB patches.

    The returned array is NCHW and ready to convert to a torch tensor.  Direct
    resizing is appropriate here because inputs are already extracted patches;
    no content-dependent crop or per-image intensity normalization is applied.
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
    normalized = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    return np.ascontiguousarray(np.moveaxis(normalized, -1, 1), dtype=np.float32)


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
    """Frozen DINOv2-style encoder returning CLS and spatial patch tokens.

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
        larger value explicitly so token/grid alignment cannot be guessed.
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
        if prefix_tokens < 1:
            raise ValueError("prefix_tokens must be at least one for a CLS encoder")
        if model_input_name is not None and not str(model_input_name).strip():
            raise ValueError("model_input_name must be non-empty or None")
        self.prefix_tokens = int(prefix_tokens)
        self.model_input_name = model_input_name
        self.model = model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def _normalize_pair(
        value: int | Sequence[int], name: str
    ) -> tuple[int, int]:
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
            return self.model(tensor)
        return self.model(**{self.model_input_name: tensor})

    def _tokens_from_output(self, output: Any) -> tuple[Any, Any]:
        torch = self._torch
        cls_token = None
        patch_tokens = None
        hidden = None

        if hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        elif isinstance(output, Mapping):
            if "x_norm_clstoken" in output and "x_norm_patchtokens" in output:
                cls_token = output["x_norm_clstoken"]
                patch_tokens = output["x_norm_patchtokens"]
            elif "last_hidden_state" in output:
                hidden = output["last_hidden_state"]
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
            cls_token = hidden[:, 0, :]
            patch_tokens = hidden[:, self.prefix_tokens :, :]

        if not torch.is_tensor(cls_token) or not torch.is_tensor(patch_tokens):
            raise TypeError(
                "Model output must expose last_hidden_state or both "
                "x_norm_clstoken and x_norm_patchtokens"
            )
        if cls_token.ndim != 2 or patch_tokens.ndim != 3:
            raise ValueError("Model CLS/patch tensors have incompatible ranks")
        expected_patches = int(np.prod(self.grid_shape))
        if int(patch_tokens.shape[1]) != expected_patches:
            raise ValueError(
                f"Model returned {patch_tokens.shape[1]} patch tokens; expected "
                f"{expected_patches} for grid {self.grid_shape}"
            )
        if (
            int(cls_token.shape[0]) != int(patch_tokens.shape[0])
            or int(cls_token.shape[-1]) != int(patch_tokens.shape[-1])
        ):
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
        preprocessed = preprocess_dinov2_rgb(
            images,
            image_size=self.image_size,
            semantic_channels=channels,
        )
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
            raise ValueError(
                "neighbors cannot exceed the number of reference tokens"
            )
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
            "mps_recommended_max_memory_bytes": (
                self.mps_recommended_max_memory_bytes
            ),
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
    "DINOv2FeatureExtractor",
    "FoundationFeatures",
    "FoundationRuntimeDiagnostics",
    "PatchKNNAnomalyScorer",
    "foundation_runtime_diagnostics",
    "preprocess_dinov2_rgb",
    "reconstruct_anomaly_heatmaps",
    "select_torch_device",
    "validate_semantic_rgb",
]
