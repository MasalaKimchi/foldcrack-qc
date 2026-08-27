from __future__ import annotations

import csv
import hashlib
import itertools
import json
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from foldcrack_qc.foundation import FoundationFeatures
from foldcrack_qc.public_fold_benchmark import (
    PUBLIC_FOLD_METHODS,
    PublicFoldBenchmarkConfig,
    PublicFoldRecord,
    PublicFoldValidationError,
    build_public_fold_splits,
    load_public_fold_dataset,
    run_public_fold_benchmark,
)


class _FakeEncoder:
    def __init__(self) -> None:
        self.encode_calls = 0

    def encode(
        self,
        images: np.ndarray,
        *,
        semantic_channels: tuple[str, str, str],
        batch_size: int,
    ) -> FoundationFeatures:
        del batch_size
        self.encode_calls += 1
        batch = np.asarray(images, dtype=np.float32) / 255.0
        rows = []
        for image in batch:
            tokens = []
            for top in (0, 8):
                token_row = []
                for left in (0, 8):
                    patch = image[top : top + 8, left : left + 8]
                    darkness = 1.0 - patch.mean(axis=(0, 1))
                    token_row.append(
                        np.asarray([darkness.mean(), *darkness], dtype=np.float32)
                    )
                tokens.append(token_row)
            rows.append(tokens)
        grid = np.asarray(rows, dtype=np.float32)
        return FoundationFeatures(
            cls_embedding=grid.mean(axis=(1, 2)),
            patch_grid=grid,
            input_size=(16, 16),
            patch_size=(8, 8),
            semantic_channels=semantic_channels,
        )


def _write_mapping(path: Path, rows: list[dict[str, str]]) -> None:
    header = ("image_filename", "organ", "class", "slide_id")
    xml_rows = []
    for row_index, row in enumerate([dict(zip(header, header, strict=True)), *rows], 1):
        cells = []
        for column, name in zip("ABCD", header, strict=True):
            value = row[name]
            cells.append(f'<c r="{column}{row_index}" t="str"><v>{value}</v></c>')
        xml_rows.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    xml = (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(xml_rows)}</sheetData></worksheet>"
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", xml)


def _make_dataset(root: Path, *, organs: tuple[str, ...] = ("Brain", "Liver")) -> Path:
    root.mkdir()
    (root / "LICENSE.txt").write_text("CC BY 4.0", encoding="utf-8")
    (root / "README.source.md").write_text(
        "Real H&E tissue fold images acquired with a microscope; manual QuPath masks.",
        encoding="utf-8",
    )
    metadata: list[dict[str, str]] = []
    mapping: list[dict[str, str]] = []
    for organ_index, organ in enumerate(organs):
        for class_name in ("clean", "tissue_fold"):
            for slide_index in range(4):
                slide_id = f"{organ}_{class_name}_{slide_index}"
                filename = f"{slide_id}.jpg"
                image_path = root / "images" / organ / class_name / filename
                image_path.parent.mkdir(parents=True, exist_ok=True)
                base = 205 - 8 * organ_index - 3 * slide_index
                rgb = np.full((32, 32, 3), base, dtype=np.uint8)
                rgb[..., 0] = np.clip(rgb[..., 0] + organ_index * 12, 0, 255)
                mask_filename = ""
                mask_relative = ""
                if class_name == "tissue_fold":
                    rgb[:, 12:20] = np.asarray((55, 20, 75), dtype=np.uint8)
                    mask = np.zeros((32, 32), dtype=np.uint8)
                    mask[:, 12:20] = 255
                    mask_filename = f"{Path(filename).stem}_mask.png"
                    mask_path = root / "masks" / organ / mask_filename
                    mask_path.parent.mkdir(parents=True, exist_ok=True)
                    assert cv2.imwrite(str(mask_path), mask)
                    mask_relative = str(mask_path.relative_to(root))
                assert cv2.imwrite(
                    str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                )
                metadata.append(
                    {
                        "dataset_version": "1.0",
                        "organ": organ,
                        "class": class_name,
                        "image_filename": filename,
                        "image_relative_path": str(image_path.relative_to(root)),
                        "mask_available": "yes"
                        if class_name == "tissue_fold"
                        else "no",
                        "mask_filename": mask_filename,
                        "mask_relative_path": mask_relative,
                        "pairing_status": "matched"
                        if class_name == "tissue_fold"
                        else "not_applicable",
                    }
                )
                mapping.append(
                    {
                        "image_filename": filename,
                        "organ": organ,
                        "class": class_name,
                        "slide_id": slide_id,
                    }
                )
    with (root / "metadata.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
        writer.writeheader()
        writer.writerows(metadata)
    _write_mapping(root / "slide_image_mapping.xlsx", mapping)
    (root / "masks").mkdir(exist_ok=True)
    return root


def _config(**overrides: object) -> PublicFoldBenchmarkConfig:
    values: dict[str, object] = {
        "strict_public_v1": False,
        "max_dimension": 32,
        "tile_size": 16,
        "tile_stride": 16,
        "encoder_batch_size": 4,
        "max_reference_tokens": 48,
        "max_probe_tokens_per_class": 48,
        "patchknn_neighbors": 1,
        "patchknn_distance_chunk_size": 16,
        "threshold_candidates": 12,
        "calibration_score_sample": 512,
        "bootstrap_resamples": 12,
        "probe_max_iterations": 30,
    }
    values.update(overrides)
    return PublicFoldBenchmarkConfig(**values)  # type: ignore[arg-type]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _valid_run_provenance(config: PublicFoldBenchmarkConfig) -> dict[str, object]:
    foundation_requested = any(method != "classical_fold" for method in config.methods)
    return {
        "schema_version": "public-fold-run-provenance-1.1",
        "capture": {
            "captured_before_scoring": True,
            "validation_status": "structurally_validated",
            "validator_id": "unit-test-approved-provenance-validator-v1",
        },
        "code": {
            "identity_type": "git",
            "commit": "a" * 40,
            "dirty_diff_sha256": "b" * 64,
        },
        "environment": {
            "python_version": "3.13.2",
            "platform": "test-platform-arm64",
            "dependencies": {
                "numpy": "2.2.3",
                "scipy": "1.16.3",
                "opencv": "4.11.0",
                **(
                    {
                        "torch": "2.7.1",
                        "transformers": "4.55.4",
                        "huggingface_hub": "0.34.4",
                    }
                    if foundation_requested
                    else {}
                ),
            },
        },
        "method_model": {
            "selected_methods": list(config.methods),
            "benchmark_configuration_sha256": _canonical_sha256(config.as_dict()),
            "implementation_id": "foldcrack-qc-public-fold-v1",
            "model_id": "fake-frozen-encoder-v1"
            if foundation_requested
            else "classical-fold-v1",
            "model_config_sha256": "c" * 64,
            "weights_sha256": "d" * 64 if foundation_requested else None,
            "weights_not_applicable": not foundation_requested,
            "loader_identity": "approved-test-loader-v1",
            "frozen_evaluation": True,
            "transductive_updates": False,
        },
        "execution": {"device": "cpu", "precision": "float32"},
    }


def _lock_fixture_as_strict_release(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    dataset = load_public_fold_dataset(
        root,
        strict_public_v1=False,
        validate_asset_dimensions=True,
        hash_assets=True,
        empty_positive_mask_policy="exclude_localization",
    )
    counts = Counter((record.organ, record.class_name) for record in dataset.records)
    identity_keys = tuple(module._PUBLIC_V1_RELEASE_IDENTITY)
    monkeypatch.setattr(module, "_PUBLIC_V1_COUNTS", dict(counts))
    monkeypatch.setattr(
        module,
        "_PUBLIC_V1_RELEASE_IDENTITY",
        {key: dataset.audit[key] for key in identity_keys},
    )


def test_official_v1_release_identity_is_pinned_to_audited_hashes() -> None:
    from foldcrack_qc import public_fold_benchmark as module

    assert dict(module._PUBLIC_V1_RELEASE_IDENTITY) == {
        "metadata_sha256": "101ca59ad4505db673253d370698b285f15342c77f590eeee65b0935357b72d4",
        "slide_mapping_sha256": "d3199c431771c8d87ac1d35f178208d1207769a9a048bd748b84808836169a40",
        "license_sha256": "866d89cbf299323640d2ff76a5695e9813fded3a8aeed676c260583763767f17",
        "source_readme_sha256": "6e69e809522c880f093bb8c674351211f939969f594ea8658e64df674371d73f",
        "asset_manifest_sha256": "826202d9951415ea5ffeafe2648b192bccc25f02ad0c3617b3be29bc9a5ab328",
        "localization_exclusion_manifest_sha256": "2002f53e1beb42f8743169d0d023f385b4d7a3cb943d972c5e7a13bb1bf57926",
    }


@pytest.mark.parametrize(
    "component",
    (
        "metadata_sha256",
        "slide_mapping_sha256",
        "license_sha256",
        "source_readme_sha256",
        "asset_manifest_sha256",
        "localization_exclusion_manifest_sha256",
    ),
)
def test_every_release_identity_component_rejects_tampering(component: str) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    observed = dict(module._PUBLIC_V1_RELEASE_IDENTITY)
    observed[component] = "0" * 64
    with pytest.raises(
        PublicFoldValidationError, match="public_v1_release_identity_mismatch"
    ) as caught:
        module._verify_public_v1_release_identity(observed)
    assert component in caught.value.detail


def test_strict_public_v1_requires_full_validation_hashing_and_exclusions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "unused"
    with pytest.raises(
        PublicFoldValidationError, match="strict_public_v1_full_validation_required"
    ):
        load_public_fold_dataset(
            root,
            strict_public_v1=True,
            validate_asset_dimensions=False,
            hash_assets=True,
            empty_positive_mask_policy="exclude_localization",
        )
    with pytest.raises(
        PublicFoldValidationError, match="strict_public_v1_asset_hashes_required"
    ):
        load_public_fold_dataset(
            root,
            strict_public_v1=True,
            validate_asset_dimensions=True,
            hash_assets=False,
            empty_positive_mask_policy="exclude_localization",
        )
    with pytest.raises(
        PublicFoldValidationError, match="strict_public_v1_exclusion_manifest_required"
    ):
        load_public_fold_dataset(
            root,
            strict_public_v1=True,
            validate_asset_dimensions=True,
            hash_assets=True,
            empty_positive_mask_policy="error",
        )


def test_strict_release_audit_is_verified_and_asset_tampering_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_dataset(tmp_path / "strict-release", organs=("Brain",))
    _lock_fixture_as_strict_release(root, monkeypatch)
    dataset = load_public_fold_dataset(
        root,
        strict_public_v1=True,
        empty_positive_mask_policy="exclude_localization",
    )
    assert dataset.audit["release_identity_verified"] is True
    assert dataset.audit["release_identity"]["verified"] is True
    assert dataset.audit["validation"]["release_identity_verified"] is True
    assert set(dataset.audit["release_identity"]["verified_components"]) == {
        "metadata_sha256",
        "slide_mapping_sha256",
        "license_sha256",
        "source_readme_sha256",
        "asset_manifest_sha256",
        "localization_exclusion_manifest_sha256",
    }

    image_path = dataset.records[0].image_path
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert image is not None
    image[0, 0] = 255 - image[0, 0]
    assert cv2.imwrite(str(image_path), image)
    with pytest.raises(
        PublicFoldValidationError, match="public_v1_release_identity_mismatch"
    ) as caught:
        load_public_fold_dataset(
            root,
            strict_public_v1=True,
            empty_positive_mask_policy="exclude_localization",
        )
    assert "asset_manifest_sha256" in caught.value.detail


def test_run_provenance_schema_rejects_foundation_without_weight_identity() -> None:
    from foldcrack_qc import public_fold_benchmark as module

    config = _config(
        methods=("foundation_patchknn",),
        bootstrap_resamples=0,
    )
    provenance = _valid_run_provenance(config)
    method_model = provenance["method_model"]
    assert isinstance(method_model, dict)
    method_model["weights_sha256"] = None
    method_model["weights_not_applicable"] = True
    audit = module._validate_run_provenance(provenance, config)
    assert audit["validated_before_scoring"] is True
    assert audit["valid"] is False
    assert (
        "method_model_foundation_weights_identity_invalid" in audit["validation_errors"]
    )


def test_strict_report_eligibility_requires_valid_pre_scoring_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    root = _make_dataset(tmp_path / "strict-provenance", organs=("Brain",))
    _lock_fixture_as_strict_release(root, monkeypatch)
    config = _config(
        methods=("classical_fold",),
        strict_public_v1=True,
        empty_positive_mask_policy="exclude_localization",
        bootstrap_resamples=0,
    )

    absent = run_public_fold_benchmark(root, config=config)
    assert absent["report_eligible"] is False
    assert "run_provenance_absent" in absent["nonreportable_reasons"]
    assert absent["run_provenance"] == {
        "schema_version": "public-fold-run-provenance-1.1",
        "provided": False,
        "validated_before_scoring": True,
        "valid": False,
        "validation_errors": ["run_provenance_absent"],
        "identity_sha256": None,
        "value": None,
    }

    malformed_value = _valid_run_provenance(config)
    malformed_method = malformed_value["method_model"]
    assert isinstance(malformed_method, dict)
    malformed_method["benchmark_configuration_sha256"] = "e" * 64
    malformed = run_public_fold_benchmark(
        root,
        config=config,
        run_provenance=malformed_value,
    )
    assert malformed["report_eligible"] is False
    assert "run_provenance_invalid" in malformed["nonreportable_reasons"]
    assert malformed["run_provenance"]["valid"] is False
    assert (
        "method_model_benchmark_configuration_mismatch"
        in malformed["run_provenance"]["validation_errors"]
    )

    events: list[str] = []
    original_validate = module._validate_run_provenance
    original_score = module._score_classical

    def tracking_validate(
        value: object, benchmark_config: PublicFoldBenchmarkConfig
    ) -> dict[str, object]:
        events.append("provenance_validation")
        return original_validate(value, benchmark_config)  # type: ignore[arg-type]

    def tracking_score(
        record: PublicFoldRecord, benchmark_config: PublicFoldBenchmarkConfig
    ) -> object:
        events.append("scoring")
        return original_score(record, benchmark_config)

    monkeypatch.setattr(module, "_validate_run_provenance", tracking_validate)
    monkeypatch.setattr(module, "_score_classical", tracking_score)
    valid_value = _valid_run_provenance(config)
    valid = run_public_fold_benchmark(
        root,
        config=config,
        run_provenance=valid_value,
    )
    assert events[0] == "provenance_validation"
    assert "scoring" in events[1:]
    assert valid["schema_version"] == "public-fold-benchmark-1.2"
    assert valid["report_eligible"] is True
    assert valid["status"] == "complete_reportable_real_public_fold_benchmark"
    assert valid["dataset"]["release_identity_verified"] is True
    assert valid["run_provenance"]["valid"] is True
    assert valid["run_provenance"]["value"] == valid_value
    assert valid["run_provenance"]["identity_sha256"] == _canonical_sha256(valid_value)
    outcomes = valid["methods"]["classical_fold"]["locked_test_outcomes"]
    assert outcomes
    assert len(valid["methods"]["classical_fold"]["locked_test_outcomes_sha256"]) == 64


def test_dataset_pairing_provenance_and_slide_disjoint_splits(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "dataset")
    dataset = load_public_fold_dataset(root, strict_public_v1=False)
    assert len(dataset.records) == 16
    assert dataset.audit["asset_content_hashes_computed"] is True
    assert dataset.audit["crack_reference_available"] is False

    splits = build_public_fold_splits(dataset.records, _config())
    slide_sets = {
        role: {record.slide_id for record in records}
        for role, records in splits.items()
    }
    assert not slide_sets["fit"] & slide_sets["calibration"]
    assert not slide_sets["fit"] & slide_sets["locked_test"]
    assert not slide_sets["calibration"] & slide_sets["locked_test"]
    for records in splits.values():
        assert {(item.organ, item.class_name) for item in records} == {
            (organ, class_name)
            for organ in ("Brain", "Liver")
            for class_name in ("clean", "tissue_fold")
        }


def test_split_optimizer_minimizes_field_imbalance_with_fixed_slide_counts() -> None:
    weights = {
        "slide_a": 10,
        "slide_b": 9,
        "slide_c": 8,
        "slide_d": 3,
        "slide_e": 2,
        "slide_f": 1,
    }
    records = tuple(
        PublicFoldRecord(
            image_filename=f"{slide_id}_{field_index}.jpg",
            organ="Brain",
            class_name="clean",
            slide_id=slide_id,
            image_path=Path(f"/{slide_id}_{field_index}.jpg"),
            mask_path=None,
        )
        for slide_id, n_fields in weights.items()
        for field_index in range(n_fields)
    )
    config = _config(methods=("classical_fold",), bootstrap_resamples=0)
    first = build_public_fold_splits(records, config)
    second = build_public_fold_splits(records, config)
    assert {
        role: [item.image_filename for item in values] for role, values in first.items()
    } == {
        role: [item.image_filename for item in values]
        for role, values in second.items()
    }
    assert [
        len({item.slide_id for item in first[role]})
        for role in ("fit", "calibration", "locked_test")
    ] == [4, 1, 1]
    actual = [len(first[role]) for role in ("fit", "calibration", "locked_test")]
    targets = np.asarray((0.6, 0.2, 0.2)) * sum(weights.values())
    observed_objective = float(np.abs(np.asarray(actual) - targets).sum())
    brute_force = []
    slide_ids = tuple(weights)
    for fit in itertools.combinations(slide_ids, 4):
        remaining = tuple(item for item in slide_ids if item not in fit)
        for calibration in itertools.combinations(remaining, 1):
            locked_test = tuple(item for item in remaining if item not in calibration)
            counts = (
                sum(weights[item] for item in fit),
                sum(weights[item] for item in calibration),
                sum(weights[item] for item in locked_test),
            )
            brute_force.append(float(np.abs(np.asarray(counts) - targets).sum()))
    assert observed_objective == pytest.approx(min(brute_force))


def test_missing_mask_and_cross_stratum_slide_are_rejected(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "missing")
    mask = next((root / "masks").rglob("*.png"))
    mask.unlink()
    with pytest.raises(PublicFoldValidationError, match="fold_mask_missing"):
        load_public_fold_dataset(root, strict_public_v1=False)

    root = _make_dataset(tmp_path / "leakage", organs=("Brain",))
    rows = []
    from foldcrack_qc.public_fold_benchmark import _read_xlsx_rows

    rows = _read_xlsx_rows(root / "slide_image_mapping.xlsx")
    rows[-1]["slide_id"] = rows[0]["slide_id"]
    _write_mapping(root / "slide_image_mapping.xlsx", rows)
    with pytest.raises(PublicFoldValidationError, match="slide_id_crosses_strata"):
        load_public_fold_dataset(root, strict_public_v1=False)


def test_empty_positive_mask_requires_audited_localization_exclusion(
    tmp_path: Path,
) -> None:
    root = _make_dataset(tmp_path / "empty", organs=("Brain",))
    mask = next((root / "masks").rglob("*.png"))
    assert cv2.imwrite(str(mask), np.zeros((32, 32), dtype=np.uint8))
    with pytest.raises(PublicFoldValidationError, match="positive_mask_empty"):
        load_public_fold_dataset(root, strict_public_v1=False)
    dataset = load_public_fold_dataset(
        root,
        strict_public_v1=False,
        empty_positive_mask_policy="exclude_localization",
    )
    assert dataset.audit["n_empty_positive_masks"] == 1
    excluded = [
        record for record in dataset.records if not record.localization_reference_valid
    ]
    assert len(excluded) == 1
    assert excluded[0].is_fold


def test_all_methods_produce_locked_real_fold_metrics_and_no_crack_claim(
    tmp_path: Path,
) -> None:
    root = _make_dataset(tmp_path / "benchmark")
    config = _config()
    report = run_public_fold_benchmark(
        root,
        encoder=_FakeEncoder(),
        config=config,
    )
    assert report["status"] == "complete_nonreportable_feasibility_run"
    assert report["execution_status"] == "complete"
    assert report["report_eligible"] is False
    assert (
        "strict_public_v1_expected_counts_and_provenance_not_enforced"
        in report["nonreportable_reasons"]
    )
    assert report["split_protocol"]["full_record_coverage"] is True
    assert len(report["split_protocol"]["assignment_manifest_sha256"]) == 64
    assert report["leakage_audit"]["passed"] is True
    assert report["claim_scope"]["fold_localization"] is True
    assert report["claim_scope"]["crack_localization"] is False
    assert set(report["methods"]) == set(config.methods)
    for method in report["methods"].values():
        assert method["thresholds"]["test_labels_accessed_during_selection"] is False
        assert (
            method["thresholds"]["pixel_localization"]["selected_on"]
            == "calibration_only"
        )
        pixel = method["locked_test"]["pixel"]
        assert 0.0 <= pixel["dice"] <= 1.0
        assert 0.0 <= pixel["iou"] <= 1.0
        assert 0.0 <= pixel["precision"] <= 1.0
        assert 0.0 <= pixel["recall"] <= 1.0
        assert method["locked_test"]["pixel_all_fields_micro"] == pixel
        assert (
            0.0 <= method["locked_test"]["pixel_positive_fields_micro"]["dice"] <= 1.0
        )
        field_macro = method["locked_test"]["positive_field_macro"]
        assert field_macro["dice"]["n"] > 0
        assert field_macro["dice"]["sample_sd"] is not None
        assert field_macro["dice"]["mean_stratified_cluster_bootstrap_ci"] is not None
        assert method["locked_test"]["positive_slide_macro"]["dice"]["n"] > 0
        clean = method["locked_test"]["clean_burden"]
        assert 0.0 <= clean["mean_predicted_area_fraction_per_clean_field"] <= 1.0
        assert 0.0 <= clean["clean_pixel_specificity"] <= 1.0
        assert (
            0.0
            <= clean["fraction_clean_fields_predicted_area_at_least_1_percent"]
            <= 1.0
        )
        assert (
            0.0
            <= clean["fraction_clean_fields_predicted_area_at_least_5_percent"]
            <= 1.0
        )
        image = method["locked_test"]["image"]
        assert image["auroc"] is not None
        assert image["auprc"] is not None
        assert set(method["per_organ"]) == {"Brain", "Liver"}
        assert method["bootstrap_ci"]["cluster_unit"] == "provided_source_slide_id"
        assert method["bootstrap_ci"]["stratification"] == "organ_by_class"
        assert method["bootstrap_ci"]["preserves_cluster_count_per_stratum"] is True


def test_encoder_agnostic_foundation_names_and_dinov2_aliases_are_equivalent(
    tmp_path: Path,
) -> None:
    assert {
        "foundation_patchknn",
        "foundation_linear_probe",
        "dinov2_patchknn",
        "dinov2_linear_probe",
    }.issubset(PUBLIC_FOLD_METHODS)
    root = _make_dataset(tmp_path / "aliases", organs=("Brain",))
    config = _config(
        methods=(
            "foundation_patchknn",
            "classical_fold",
            "dinov2_patchknn",
            "foundation_linear_probe",
            "dinov2_linear_probe",
        ),
        bootstrap_resamples=0,
    )
    encoder = _FakeEncoder()
    report = run_public_fold_benchmark(root, encoder=encoder, config=config)

    generic_knn = report["methods"]["foundation_patchknn"]
    legacy_knn = report["methods"]["dinov2_patchknn"]
    generic_probe = report["methods"]["foundation_linear_probe"]
    legacy_probe = report["methods"]["dinov2_linear_probe"]
    assert list(report["methods"]) == list(config.methods)
    # One organ has four fit records, two calibration records and two locked
    # test records.  Four tiles fit in one fake-encoder batch per record, so
    # joint execution is exactly 4 + 2*2 + 2 = 10 calls, independent of the
    # four generic/legacy report IDs.
    assert encoder.encode_calls == 10
    joint = report["foundation_evaluation"]
    assert joint["unique_heads"] == ["patchknn", "linear_probe"]
    assert joint["record_scoring_traversals"] == {
        "calibration": 4,
        "locked_test": 2,
    }
    assert joint["full_resolution_score_maps_retained_between_records"] is False
    assert joint["method_report_runtime_values_additive"] is False
    assert generic_knn["locked_test"] == legacy_knn["locked_test"]
    assert generic_probe["locked_test"] == legacy_probe["locked_test"]
    assert generic_knn["thresholds"] == legacy_knn["thresholds"]
    assert generic_probe["thresholds"] == legacy_probe["thresholds"]
    assert generic_knn["method_identity"]["algorithm_family"] == "patchknn"
    assert generic_probe["method_identity"]["algorithm_family"] == "linear_probe"
    assert generic_probe["method_identity"]["legacy_encoder_specific_alias"] is False
    assert legacy_probe["method_identity"]["legacy_encoder_specific_alias"] is True
    for name, method in report["methods"].items():
        if name == "classical_fold":
            assert method["runtime"]["shared_joint_foundation_execution"] is False
            assert method["runtime"]["additive_across_method_reports"] is True
            continue
        assert method["runtime"]["shared_joint_foundation_execution"] is True
        assert method["runtime"]["additive_across_method_reports"] is False


def test_joint_foundation_results_match_separate_head_runs(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "joint-equivalence", organs=("Brain",))
    joint = run_public_fold_benchmark(
        root,
        encoder=_FakeEncoder(),
        config=_config(
            methods=("foundation_patchknn", "foundation_linear_probe"),
            bootstrap_resamples=0,
        ),
    )
    separate_knn = run_public_fold_benchmark(
        root,
        encoder=_FakeEncoder(),
        config=_config(methods=("foundation_patchknn",), bootstrap_resamples=0),
    )
    separate_probe = run_public_fold_benchmark(
        root,
        encoder=_FakeEncoder(),
        config=_config(methods=("foundation_linear_probe",), bootstrap_resamples=0),
    )

    for method, separate in (
        ("foundation_patchknn", separate_knn),
        ("foundation_linear_probe", separate_probe),
    ):
        joint_method = joint["methods"][method]
        separate_method = separate["methods"][method]
        for key in ("thresholds", "locked_test", "per_organ", "bootstrap_ci"):
            assert joint_method[key] == separate_method[key]


def test_joint_foundation_scores_calibration_twice_then_locked_test_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    root = _make_dataset(tmp_path / "joint-order", organs=("Brain",))
    config = _config(
        methods=("foundation_patchknn", "foundation_linear_probe"),
        bootstrap_resamples=0,
    )
    dataset = load_public_fold_dataset(root, strict_public_v1=False)
    splits = build_public_fold_splits(dataset.records, config)
    calibration_names = [item.image_filename for item in splits["calibration"]]
    test_names = [item.image_filename for item in splits["locked_test"]]
    scoring_calls: list[str] = []
    original = module._score_foundation

    def tracking_score(
        record: PublicFoldRecord,
        encoder: module.FrozenEncoder,
        benchmark_config: PublicFoldBenchmarkConfig,
        knn: module.PatchKNNAnomalyScorer | None,
        probe: module._LinearTokenProbe | None,
    ) -> dict[str, module._Scored]:
        scoring_calls.append(record.image_filename)
        return original(record, encoder, benchmark_config, knn, probe)

    monkeypatch.setattr(module, "_score_foundation", tracking_score)
    encoder = _FakeEncoder()
    run_public_fold_benchmark(root, encoder=encoder, config=config)

    assert scoring_calls == calibration_names + calibration_names + test_names
    assert encoder.encode_calls == 10


def test_foundation_alias_validation_requires_an_encoder(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "encoder-required", organs=("Brain",))
    config = _config(methods=("foundation_linear_probe",), bootstrap_resamples=0)
    with pytest.raises(ValueError, match="injected frozen encoder"):
        run_public_fold_benchmark(root, config=config)


def test_test_mask_change_cannot_change_calibration_thresholds(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "locking", organs=("Brain",))
    config = _config(methods=("classical_fold",), bootstrap_resamples=0)
    dataset = load_public_fold_dataset(root, strict_public_v1=False, hash_assets=False)
    test_fold = next(
        record
        for record in build_public_fold_splits(dataset.records, config)["locked_test"]
        if record.is_fold
    )
    before = run_public_fold_benchmark(root, config=config)
    replacement = np.zeros((32, 32), dtype=np.uint8)
    replacement[4:12, :] = 255
    assert test_fold.mask_path is not None
    assert cv2.imwrite(str(test_fold.mask_path), replacement)
    after = run_public_fold_benchmark(root, config=config)
    assert (
        before["methods"]["classical_fold"]["thresholds"]
        == after["methods"]["classical_fold"]["thresholds"]
    )
    assert (
        before["methods"]["classical_fold"]["locked_test"]["pixel"]
        != after["methods"]["classical_fold"]["locked_test"]["pixel"]
    )


def test_skipped_integrity_and_limited_cohort_are_nonreportable(tmp_path: Path) -> None:
    root = _make_dataset(tmp_path / "nonreportable", organs=("Brain",))
    report = run_public_fold_benchmark(
        root,
        config=_config(
            methods=("classical_fold",),
            validate_asset_dimensions=False,
            hash_assets=False,
            limit_slides_per_stratum_per_split=1,
            bootstrap_resamples=0,
        ),
    )
    assert report["report_eligible"] is False
    assert {
        "image_mask_dimensions_and_binary_masks_not_strictly_validated",
        "per_asset_content_hashes_not_computed",
        "cohort_limited_smoke_run",
        "split_does_not_cover_full_validated_cohort",
    }.issubset(report["nonreportable_reasons"])
    assert report["split_protocol"]["smoke_limit_applied"] is True


def test_linear_probe_nonconvergence_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    def failed_minimize(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return SimpleNamespace(
            success=False,
            status=1,
            message="iteration limit reached",
            fun=0.5,
            x=np.zeros(3),
            nit=1,
            nfev=2,
            jac=np.zeros(3),
        )

    monkeypatch.setattr(module, "minimize", failed_minimize)
    probe = module._LinearTokenProbe(l2=1e-3, max_iterations=1)
    with pytest.raises(RuntimeError, match="did not converge"):
        probe.fit(
            np.asarray([[0.0, 0.0], [0.1, 0.0]]),
            np.asarray([[1.0, 1.0], [0.9, 1.0]]),
        )


def test_classical_component_filter_is_independent_of_foundation_tile_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from foldcrack_qc import public_fold_benchmark as module

    root = _make_dataset(tmp_path / "component-size", organs=("Brain",))
    dataset = load_public_fold_dataset(root, strict_public_v1=False)
    observed: list[int] = []

    def fake_candidates(
        image: np.ndarray, *, modality: str, min_component_size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        assert modality == "he"
        observed.append(min_component_size)
        shape = image.shape[:2]
        return np.zeros(shape, dtype=bool), np.zeros(shape, dtype=np.float32)

    monkeypatch.setattr(module, "classical_fold_candidates", fake_candidates)
    record = dataset.records[0]
    module._score_classical(
        record,
        _config(tile_size=16, tile_stride=16, classical_min_component_size=7),
    )
    module._score_classical(
        record,
        _config(tile_size=32, tile_stride=32, classical_min_component_size=7),
    )
    assert observed == [7, 7]
