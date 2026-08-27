"""Public-resource registry with explicit license and validation caveats."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .resources import resource_path


def default_registry_path() -> Path:
    return resource_path("datasets.json")


def load_registry(path: str | Path | None = None) -> list[dict[str, Any]]:
    registry_path = Path(path) if path else default_registry_path()
    with registry_path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise TypeError("Dataset registry must contain a JSON list")
    return value


def format_registry(records: list[dict[str, Any]]) -> str:
    lines = [
        "ID                 MODALITY       LABELS                     LICENSE STATUS",
        "-" * 92,
    ]
    for record in records:
        modalities = ",".join(record.get("modalities", []))
        labels = ",".join(record.get("labels", [])) or "nominal only"
        lines.append(
            f"{record['id']:<18} {modalities:<14} {labels[:26]:<26} "
            f"{record.get('license_status', 'unknown')}"
        )
    lines.extend(
        [
            "",
            "Registry entries are discovery aids, not legal approval. Confirm the exact",
            "dataset/model version and terms with Merck Legal before download or reuse.",
        ]
    )
    return "\n".join(lines)
