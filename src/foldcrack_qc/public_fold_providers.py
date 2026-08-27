"""Lazy frozen-encoder providers for the public H&E fold benchmark.

Importing this module does not import PyTorch, Transformers, or any model
implementation.  The selected provider loads its optional dependencies only
when a foundation method actually requires an encoder.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Protocol


class PublicFoldProviderArguments(Protocol):
    """CLI arguments consumed by the frozen-encoder provider builders."""

    allow_download: bool
    cache_dir: Path
    device: str
    hibou_source: Path | None
    hibou_source_commit: str | None
    hibou_weights: Path | None
    hibou_weights_sha256: str | None
    model_id: str
    revision: str
    siglip2_snapshot: Path | None


class PublicFoldEncoder(Protocol):
    """Minimum frozen-encoder surface required by the public benchmark."""

    def encode(
        self,
        images: Any,
        *,
        semantic_channels: Sequence[str],
        batch_size: int,
    ) -> Any: ...


@dataclass(frozen=True)
class BuiltPublicFoldProvider:
    """Validated encoder plus an isolated identity with a read-only root."""

    encoder: PublicFoldEncoder
    model_identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not callable(getattr(self.encoder, "encode", None)):
            raise TypeError("public-fold encoder must provide a callable encode method")
        if not isinstance(self.model_identity, Mapping):
            raise TypeError("public-fold model_identity must be a mapping")
        identity = copy.deepcopy(dict(self.model_identity))
        if not identity:
            raise ValueError("public-fold model_identity must be non-empty")
        for field in ("requested_device", "resolved_device"):
            value = identity.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"public-fold model_identity {field!r} must be a non-empty string"
                )
        object.__setattr__(self, "model_identity", MappingProxyType(identity))


PublicFoldProviderBuilder = Callable[
    [PublicFoldProviderArguments], BuiltPublicFoldProvider
]


@dataclass(frozen=True)
class PublicFoldProviderSpec:
    """Immutable construction and legacy-alias policy for one provider."""

    builder: PublicFoldProviderBuilder
    allows_dinov2_legacy_aliases: bool
    display_name: str

    def __post_init__(self) -> None:
        if not callable(self.builder):
            raise TypeError("public-fold provider builder must be callable")
        if not self.display_name.strip():
            raise ValueError("public-fold provider display_name must be non-empty")


def _build_hibou_b_local(
    args: PublicFoldProviderArguments,
) -> BuiltPublicFoldProvider:
    from .foundation import DINOv2FeatureExtractor

    if args.allow_download:
        raise ValueError(
            "--allow-download is incompatible with the local-only Hibou-B loader"
        )
    if args.hibou_weights is None or args.hibou_source is None:
        raise ValueError(
            "hibou-b-local requires both --hibou-weights and --hibou-source"
        )
    if args.hibou_weights_sha256 is None or args.hibou_source_commit is None:
        raise ValueError(
            "hibou-b-local requires both --hibou-weights-sha256 and "
            "--hibou-source-commit from an approved release"
        )
    from .foundation import HIBOU_B_MEAN, HIBOU_B_STD, load_local_hibou_b

    local = load_local_hibou_b(
        args.hibou_weights,
        args.hibou_source,
        expected_weights_sha256=args.hibou_weights_sha256,
        expected_source_commit=args.hibou_source_commit,
    )
    encoder = DINOv2FeatureExtractor(
        local.model,
        device=args.device,
        image_size=224,
        patch_size=14,
        prefix_tokens=5,
        model_input_name=None,
        normalization_mean=HIBOU_B_MEAN,
        normalization_std=HIBOU_B_STD,
    )
    model_identity = {
        **dict(local.provenance),
        "requested_device": args.device,
        "resolved_device": str(getattr(encoder, "device", args.device)),
        "output_contract": {
            "type": "mapping",
            "cls_key": "x_norm_clstoken",
            "patch_key": "x_norm_patchtokens",
            "prefix_tokens": 5,
        },
    }
    return BuiltPublicFoldProvider(encoder, model_identity)


def _build_siglip2_base_local(
    args: PublicFoldProviderArguments,
) -> BuiltPublicFoldProvider:
    from .foundation import DINOv2FeatureExtractor

    if args.allow_download:
        raise ValueError(
            "--allow-download is incompatible with the local-only SigLIP2 Base loader"
        )
    if args.siglip2_snapshot is None:
        raise ValueError("siglip2-base-local requires --siglip2-snapshot")
    from .foundation import (
        SIGLIP2_BASE_MEAN,
        SIGLIP2_BASE_STD,
        load_local_siglip2_base_vision,
    )

    local = load_local_siglip2_base_vision(args.siglip2_snapshot)
    encoder = DINOv2FeatureExtractor(
        local.model,
        device=args.device,
        image_size=224,
        patch_size=16,
        prefix_tokens=0,
        model_input_name="pixel_values",
        global_embedding_name="pooler_output",
        normalization_mean=SIGLIP2_BASE_MEAN,
        normalization_std=SIGLIP2_BASE_STD,
        preprocessor=local.preprocessor,
    )
    model_identity = {
        **dict(local.provenance),
        "requested_device": args.device,
        "resolved_device": str(getattr(encoder, "device", args.device)),
        "output_contract": {
            "type": "object",
            "global_key": "pooler_output",
            "patch_key": "last_hidden_state",
            "prefix_tokens": 0,
        },
    }
    return BuiltPublicFoldProvider(encoder, model_identity)


def _build_dinov2_hf(
    args: PublicFoldProviderArguments,
) -> BuiltPublicFoldProvider:
    from .foundation import DINOv2FeatureExtractor
    from .foundation_smoke import (
        FoundationSmokeConfig,
        dinov2_model_geometry,
        load_huggingface_model,
    )

    model_config = FoundationSmokeConfig(
        revision=args.revision,
        model_id=args.model_id,
        cache_dir=args.cache_dir,
        device=args.device,
        allow_download=args.allow_download,
        image_size=224,
        steady_runs=1,
    )
    loaded = load_huggingface_model(model_config)
    patch_size, prefix_tokens = dinov2_model_geometry(loaded.model, 224)
    encoder = DINOv2FeatureExtractor(
        loaded.model,
        device=args.device,
        image_size=224,
        patch_size=patch_size,
        prefix_tokens=prefix_tokens,
        model_input_name="pixel_values",
    )
    model_identity = {
        "id": args.model_id,
        "requested_revision": args.revision,
        "resolved_revision": loaded.resolved_revision,
        "weight_files": [item.as_dict() for item in loaded.weight_digests],
        "configuration_files": [
            item.as_dict() for item in loaded.configuration_digests
        ],
        "requested_device": args.device,
        "resolved_device": str(getattr(encoder, "device", args.device)),
        "trust_remote_code": False,
        "token_used": False,
        "network_access_allowed": bool(args.allow_download),
        "input": {
            "normalization": "ImageNet",
            "image_size": [224, 224],
            "patch_size": list(patch_size),
            "prefix_tokens": prefix_tokens,
        },
    }
    return BuiltPublicFoldProvider(encoder, model_identity)


PUBLIC_FOLD_ENCODER_PROVIDERS: Mapping[str, PublicFoldProviderSpec] = MappingProxyType(
    {
        "dinov2-hf": PublicFoldProviderSpec(
            builder=_build_dinov2_hf,
            allows_dinov2_legacy_aliases=True,
            display_name="DINOv2",
        ),
        "hibou-b-local": PublicFoldProviderSpec(
            builder=_build_hibou_b_local,
            allows_dinov2_legacy_aliases=False,
            display_name="Hibou-B",
        ),
        "siglip2-base-local": PublicFoldProviderSpec(
            builder=_build_siglip2_base_local,
            allows_dinov2_legacy_aliases=False,
            display_name="SigLIP2 Base",
        ),
    }
)
PUBLIC_FOLD_ENCODER_NAMES = tuple(PUBLIC_FOLD_ENCODER_PROVIDERS)


def build_public_fold_encoder(
    name: str,
    args: PublicFoldProviderArguments,
    methods: Sequence[str],
) -> BuiltPublicFoldProvider:
    """Build one registered encoder without importing unselected providers."""

    try:
        spec = PUBLIC_FOLD_ENCODER_PROVIDERS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown public-fold foundation encoder {name!r}; expected one of "
            f"{list(PUBLIC_FOLD_ENCODER_NAMES)}"
        ) from error
    legacy_aliases = sorted(
        method for method in methods if method.startswith("dinov2_")
    )
    if legacy_aliases and not spec.allows_dinov2_legacy_aliases:
        raise ValueError(
            f"{spec.display_name} must use encoder-agnostic foundation_patchknn "
            "and/or foundation_linear_probe, not DINOv2 aliases: "
            f"{legacy_aliases}"
        )
    built = spec.builder(args)
    if not isinstance(built, BuiltPublicFoldProvider):
        raise TypeError(
            f"Public-fold provider {name!r} returned an invalid build result"
        )
    return built


__all__ = [
    "PUBLIC_FOLD_ENCODER_NAMES",
    "PUBLIC_FOLD_ENCODER_PROVIDERS",
    "BuiltPublicFoldProvider",
    "PublicFoldEncoder",
    "PublicFoldProviderArguments",
    "PublicFoldProviderSpec",
    "build_public_fold_encoder",
]
