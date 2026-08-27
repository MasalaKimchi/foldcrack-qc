from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pytest

from scripts.compare_hardened_public_fold import (
    COMPARISON_SCHEMA,
    ComparisonValidationError,
    compare_reports,
    main,
)

_RELEASE_COMPONENTS = {
    "asset_manifest_sha256": "826202d9951415ea5ffeafe2648b192bccc25f02ad0c3617b3be29bc9a5ab328",
    "license_sha256": "866d89cbf299323640d2ff76a5695e9813fded3a8aeed676c260583763767f17",
    "localization_exclusion_manifest_sha256": "2002f53e1beb42f8743169d0d023f385b4d7a3cb943d972c5e7a13bb1bf57926",
    "metadata_sha256": "101ca59ad4505db673253d370698b285f15342c77f590eeee65b0935357b72d4",
    "slide_mapping_sha256": "d3199c431771c8d87ac1d35f178208d1207769a9a048bd748b84808836169a40",
    "source_readme_sha256": "6e69e809522c880f093bb8c674351211f939969f594ea8658e64df674371d73f",
}
_CLAIM_SCOPE = {
    "artifact": "tissue_fold",
    "crack_localization": False,
    "crack_presence": False,
    "cross_modality_generalization": False,
    "fold_localization": True,
    "fold_presence": True,
    "modality": "H&E brightfield microscopy",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _record(
    *,
    field_index: int,
    filename: str,
    organ: str,
    class_name: str,
    slide_id: str,
) -> dict[str, object]:
    return {
        "class": class_name,
        "image_filename": filename,
        "image_sha256": f"{field_index + 1:064x}",
        "localization_reference_valid": True,
        "mask_sha256": f"{field_index + 100_000:064x}"
        if class_name == "tissue_fold"
        else None,
        "organ": organ,
        "slide_id": slide_id,
    }


def _locked_manifest(*, positive_slides_per_organ: int = 2) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    field_index = 0
    for organ in ("Brain", "Liver"):
        records.append(
            _record(
                field_index=field_index,
                filename=f"{organ}_clean.jpg",
                organ=organ,
                class_name="clean",
                slide_id=f"{organ}_Clean_S001",
            )
        )
        field_index += 1
        for slide_index in range(1, positive_slides_per_organ + 1):
            for field_on_slide in (1, 2):
                records.append(
                    _record(
                        field_index=field_index,
                        filename=(f"{organ}_fold_s{slide_index}_f{field_on_slide}.jpg"),
                        organ=organ,
                        class_name="tissue_fold",
                        slide_id=f"{organ}_Fold_S{slide_index:03d}",
                    )
                )
                field_index += 1
    return records


def _all_manifests(
    *,
    locked_manifest: list[dict[str, object]] | None = None,
    positive_slides_per_organ: int = 2,
) -> dict[str, list[dict[str, object]]]:
    locked = (
        _locked_manifest(positive_slides_per_organ=positive_slides_per_organ)
        if locked_manifest is None
        else locked_manifest
    )
    locked_slides = {str(item["slide_id"]) for item in locked}
    filler_slide_count = 283 - len(locked_slides)
    assert filler_slide_count > 2
    fit_slide_count = min(140, filler_slide_count - 1)
    manifests: dict[str, list[dict[str, object]]] = {
        "fit": [],
        "calibration": [],
        "locked_test": locked,
    }
    next_field = len(locked) + 1_000
    organs = ("Brain", "Kidney", "Liver", "Small_Intestine", "Testis")
    for slide_index in range(filler_slide_count):
        role = "fit" if slide_index < fit_slide_count else "calibration"
        organ = organs[slide_index % len(organs)]
        manifests[role].append(
            _record(
                field_index=next_field,
                filename=f"filler_{slide_index:03d}_field_0000.jpg",
                organ=organ,
                class_name="clean",
                slide_id=f"Filler_Clean_S{slide_index:03d}",
            )
        )
        next_field += 1
    remaining_fields = 2127 - sum(len(items) for items in manifests.values())
    assert remaining_fields >= 0
    first_slide = str(manifests["fit"][0]["slide_id"])
    first_organ = str(manifests["fit"][0]["organ"])
    for extra_index in range(remaining_fields):
        manifests["fit"].append(
            _record(
                field_index=next_field,
                filename=f"filler_extra_{extra_index:04d}.jpg",
                organ=first_organ,
                class_name="clean",
                slide_id=first_slide,
            )
        )
        next_field += 1
    assert sum(len(items) for items in manifests.values()) == 2127
    assert (
        len({str(item["slide_id"]) for items in manifests.values() for item in items})
        == 283
    )
    return manifests


def _split_report(manifest: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for item in manifest:
        key = f"{item['organ']}/{item['class']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "counts": dict(sorted(counts.items())),
        "localization_exclusions": sorted(
            str(item["image_filename"])
            for item in manifest
            if not item["localization_reference_valid"]
        ),
        "manifest": manifest,
        "manifest_sha256": _canonical_sha256(manifest),
        "n_images": len(manifest),
        "n_localization_references": sum(
            bool(item["localization_reference_valid"]) for item in manifest
        ),
        "n_slides": len({item["slide_id"] for item in manifest}),
        "slide_ids": sorted({str(item["slide_id"]) for item in manifest}),
    }


def _split_protocol(manifests: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    assignment: list[dict[str, object]] = []
    for role in ("fit", "calibration", "locked_test"):
        by_slide: dict[str, list[dict[str, object]]] = defaultdict(list)
        for item in manifests[role]:
            by_slide[str(item["slide_id"])].append(item)
        assignment.extend(
            {
                "organ": records[0]["organ"],
                "class": records[0]["class"],
                "slide_id": slide_id,
                "role": role,
                "n_images": len(records),
            }
            for slide_id, records in by_slide.items()
        )
    assignment.sort(
        key=lambda item: (item["organ"], item["class"], item["slide_id"], item["role"])
    )
    return {
        "protocol": "organ-by-class source-slide group split fixture",
        "group_unit": "provided_source_slide_id",
        "requested_role_fractions": {
            "fit": 0.6,
            "calibration": 0.2,
            "locked_test": 0.2,
        },
        "full_record_coverage": True,
        "smoke_limit_applied": False,
        "assignment_manifest": assignment,
        "assignment_manifest_sha256": _canonical_sha256(assignment),
    }


def _confusion_for_dice(dice: float) -> tuple[int, int, int, int]:
    if dice == 1.0:
        return 10, 0, 0, 90
    if dice == 0.5:
        return 5, 5, 5, 85
    if dice == 0.0:
        return 0, 10, 10, 80
    raise AssertionError(f"unsupported test Dice: {dice}")


def _configuration() -> dict[str, object]:
    return {
        "bootstrap_confidence": 0.95,
        "bootstrap_resamples": 1000,
        "calibration_fraction": 0.2,
        "calibration_score_sample": 250000,
        "classical_min_component_size": 8,
        "empty_positive_mask_policy": "exclude_localization",
        "encoder_batch_size": 8,
        "fit_fraction": 0.6,
        "hash_assets": True,
        "image_score_quantile": 0.995,
        "limit_slides_per_stratum_per_split": None,
        "max_dimension": 896,
        "max_probe_tokens_per_class": 8192,
        "max_reference_tokens": 4096,
        "methods": ["classical_fold"],
        "patchknn_distance_chunk_size": 256,
        "patchknn_neighbors": 3,
        "probe_l2": 0.001,
        "probe_max_iterations": 100,
        "seed": 20260826,
        "strict_public_v1": True,
        "test_fraction": 0.2,
        "threshold_candidates": 96,
        "tile_size": 224,
        "tile_stride": 224,
        "token_positive_fraction": 0.05,
        "validate_asset_dimensions": True,
    }


def _method_report(
    locked_manifest: list[dict[str, object]],
    positive_dice: float | dict[str, float],
) -> dict[str, object]:
    outcomes: list[dict[str, object]] = []
    positive_values: list[float] = []
    for item in locked_manifest:
        is_positive = item["class"] == "tissue_fold"
        if is_positive:
            requested_dice = (
                positive_dice[str(item["image_filename"])]
                if isinstance(positive_dice, dict)
                else positive_dice
            )
            tp, fp, fn, tn = _confusion_for_dice(requested_dice)
            positive_values.append(requested_dice)
        else:
            tp, fp, fn, tn = 0, 0, 0, 100
        outcomes.append(
            {
                "field_key": item["image_filename"],
                "fn": fn,
                "fp": fp,
                "image_prediction": int(is_positive),
                "image_score": 0.9 if is_positive else 0.1,
                "label": int(is_positive),
                "localization_reference_valid": item["localization_reference_valid"],
                "n_valid": 100,
                "organ": item["organ"],
                "runtime_seconds": 0.01,
                "source_slide_id": item["slide_id"],
                "tn": tn,
                "tp": tp,
            }
        )
    return {
        "method": "classical_fold",
        "method_identity": {
            "algorithm_family": "classical_fold_candidates",
            "foundation_encoder_required": False,
            "legacy_encoder_specific_alias": False,
            "reported_method_id": "classical_fold",
        },
        "locked_test": {
            "positive_field_macro": {
                "dice": {
                    "mean": math.fsum(positive_values) / len(positive_values),
                    "n": len(positive_values),
                }
            }
        },
        "locked_test_outcomes": outcomes,
        "locked_test_outcomes_sha256": _canonical_sha256(outcomes),
    }


def _rehash_report(report: dict[str, object]) -> None:
    report["configuration_sha256"] = _canonical_sha256(report["configuration"])
    provenance = report["run_provenance"]
    provenance["value"]["method_model"]["benchmark_configuration_sha256"] = report[
        "configuration_sha256"
    ]
    provenance["identity_sha256"] = _canonical_sha256(provenance["value"])


def _convert_to_foundation(report: dict[str, object]) -> None:
    method = report["methods"].pop("classical_fold")
    method["method"] = "foundation_linear_probe"
    method["method_identity"] = {
        "algorithm_family": "linear_probe",
        "canonical_encoder_agnostic_method_id": "foundation_linear_probe",
        "encoder_identity_location": "top-level model_identity when invoked by CLI",
        "foundation_encoder_required": True,
        "legacy_encoder_specific_alias": False,
        "reported_method_id": "foundation_linear_probe",
    }
    report["methods"]["foundation_linear_probe"] = method
    report["configuration"]["methods"] = ["foundation_linear_probe"]
    model_identity = {
        "id": "fixture/foundation-model",
        "loader": "fixture.StandardVisionModel.from_pretrained",
        "assets": {
            "model.safetensors": {
                "path": "/fixture/model.safetensors",
                "sha256": "3" * 64,
                "size_bytes": 1024,
            },
            "config.json": {
                "path": "/fixture/config.json",
                "sha256": "4" * 64,
                "size_bytes": 128,
            },
        },
        "input": {"image_size": [224, 224], "patch_size": [16, 16]},
        "network_access_allowed": False,
        "requested_device": "cpu",
        "resolved_device": "cpu",
        "token_used": False,
        "trust_remote_code": False,
    }
    report["model_identity"] = model_identity
    method_model = report["run_provenance"]["value"]["method_model"]
    method_model.update(
        {
            "loader_identity": model_identity["loader"],
            "model_config_sha256": _canonical_sha256(model_identity),
            "model_id": model_identity["id"],
            "selected_methods": ["foundation_linear_probe"],
            "weights_not_applicable": False,
            "weights_sha256": "3" * 64,
        }
    )
    dependencies = report["run_provenance"]["value"]["environment"]["dependencies"]
    dependencies.update(
        {"torch": "2.4", "transformers": "4.55", "huggingface_hub": "0.34"}
    )
    _rehash_report(report)


def _convert_to_real_shape_dinov2(report: dict[str, object]) -> None:
    _convert_to_foundation(report)
    revision = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
    model_identity = {
        "configuration_files": [
            {
                "filename": "config.json",
                "sha256": "5" * 64,
                "size_bytes": 547,
            },
            {
                "filename": "preprocessor_config.json",
                "sha256": "6" * 64,
                "size_bytes": 436,
            },
        ],
        "id": "facebook/dinov2-small",
        "input": {
            "image_size": [224, 224],
            "normalization": "ImageNet",
            "patch_size": [14, 14],
            "prefix_tokens": 1,
        },
        "network_access_allowed": False,
        "requested_device": "cpu",
        "requested_revision": revision,
        "resolved_device": "cpu",
        "resolved_revision": revision,
        "token_used": False,
        "trust_remote_code": False,
        "weight_files": [
            {
                "filename": "model.safetensors",
                "sha256": "7" * 64,
                "size_bytes": 88_249_960,
            }
        ],
    }
    report["model_identity"] = model_identity
    method_model = report["run_provenance"]["value"]["method_model"]
    method_model.update(
        {
            "loader_identity": "transformers_pretrained_trust_remote_code_false",
            "model_config_sha256": _canonical_sha256(model_identity),
            "model_id": "facebook/dinov2-small",
            "weights_sha256": "7" * 64,
        }
    )
    _rehash_report(report)


def _write_json(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_report(
    path: Path,
    *,
    positive_dice: float | dict[str, float],
    locked_manifest: list[dict[str, object]] | None = None,
    positive_slides_per_organ: int = 2,
) -> dict[str, object]:
    manifests = _all_manifests(
        locked_manifest=locked_manifest,
        positive_slides_per_organ=positive_slides_per_organ,
    )
    release = {
        "identity_version": "histology-tissue-fold-v1.0-2026-08-26",
        "verified": True,
        "verified_components": sorted(_RELEASE_COMPONENTS),
        "canonical_identity_sha256": _canonical_sha256(_RELEASE_COMPONENTS),
        "identity": _RELEASE_COMPONENTS,
    }
    dataset: dict[str, object] = {
        "dataset_name": "Histology Tissue Fold Dataset",
        "dataset_version": "1.0",
        "license": "CC BY 4.0",
        "claimable_artifacts": ["tissue_fold"],
        "crack_reference_available": False,
        "data_origin": "real microscope-acquired H&E teaching-slide fields",
        "empty_positive_mask_policy": "exclude_localization",
        "n_records": 2127,
        "n_slides": 283,
        "asset_content_hashes_computed": True,
        "release_identity_verified": True,
        "release_identity": release,
        **_RELEASE_COMPONENTS,
    }
    configuration = _configuration()
    configuration_sha256 = _canonical_sha256(configuration)
    classical_identity = {
        "id": "classical-fold-candidates-v1",
        "loader": "in_process_foldcrack_qc.detectors.classical_fold_candidates",
    }
    method_model = {
        "benchmark_configuration_sha256": configuration_sha256,
        "frozen_evaluation": True,
        "implementation_id": "foldcrack_qc.public_fold_benchmark:v1.2",
        "loader_identity": "in_process_foldcrack_qc.detectors.classical_fold_candidates",
        "model_config_sha256": _canonical_sha256(classical_identity),
        "model_id": classical_identity["id"],
        "selected_methods": ["classical_fold"],
        "transductive_updates": False,
        "weights_not_applicable": True,
        "weights_sha256": None,
    }
    provenance_value = {
        "capture": {
            "approval_scope": "reproducibility_structure_not_corporate_model_governance",
            "captured_before_scoring": True,
            "validation_status": "structurally_validated",
            "validator_id": "foldcrack-qc-cli-preflight-v1.1",
        },
        "code": {
            "commit": "1" * 40,
            "dirty_diff_capture": "git_diff_HEAD_plus_untracked_runtime_sources",
            "dirty_diff_sha256": "2" * 64,
            "identity_type": "git",
            "untracked_runtime_sources": [],
        },
        "environment": {
            "dependencies": {"numpy": "2.0", "opencv": "4.10", "scipy": "1.13"},
            "platform": "fixture-platform",
            "python_version": "3.13",
        },
        "execution": {"device": "cpu", "precision": "float32"},
        "method_model": method_model,
        "schema_version": "public-fold-run-provenance-1.1",
    }
    report: dict[str, object] = {
        "schema_version": "public-fold-benchmark-1.2",
        "status": "complete_reportable_real_public_fold_benchmark",
        "claim_scope": dict(_CLAIM_SCOPE),
        "report_eligible": True,
        "execution_status": "complete",
        "nonreportable_reasons": [],
        "leakage_audit": {
            "calibration_test_overlap": 0,
            "fit_calibration_overlap": 0,
            "fit_test_overlap": 0,
            "group_unit": "source_slide_id",
            "passed": True,
        },
        "dataset": dataset,
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "split_protocol": _split_protocol(manifests),
        "splits": {role: _split_report(items) for role, items in manifests.items()},
        "methods": {
            "classical_fold": _method_report(manifests["locked_test"], positive_dice)
        },
        "run_provenance": {
            "schema_version": "public-fold-run-provenance-1.1",
            "provided": True,
            "valid": True,
            "validated_before_scoring": True,
            "validation_errors": [],
            "identity_sha256": _canonical_sha256(provenance_value),
            "value": provenance_value,
        },
    }
    _write_json(path, report)
    return report


def _compare_paths(
    first_path: Path, second_path: Path, *, resamples: int = 40
) -> dict[str, object]:
    return compare_reports(
        [f"first={first_path}", f"second={second_path}"],
        ["first=classical_fold", "second=classical_fold"],
        resamples=resamples,
        seed=7,
    )


def test_comparison_and_cli_output_are_deterministic(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    _write_report(second_path, positive_dice=0.5)
    reports = [f"a={first_path}", f"b={second_path}"]
    methods = ["a=classical_fold", "b=classical_fold"]

    first = compare_reports(reports, methods, resamples=50, seed=19)
    second = compare_reports(reports, methods, resamples=50, seed=19)
    assert first == second
    assert first["schema_version"] == COMPARISON_SCHEMA
    assert (
        first["reports"]["a"]["artifact_sha256"]
        == hashlib.sha256(first_path.read_bytes()).hexdigest()
    )
    assert first["reports"]["a"]["run_provenance"]["identity_sha256"]
    comparator = first["comparator_identity"]
    assert (
        comparator["code_sha256"]
        == hashlib.sha256(Path(comparator["code_path"]).read_bytes()).hexdigest()
    )
    assert (
        first["claim_scope"]["validated_source_benchmark_claim_scope"] == _CLAIM_SCOPE
    )

    output_path = tmp_path / "comparison.json"
    assert (
        main(
            [
                "--report",
                reports[0],
                "--report",
                reports[1],
                "--method",
                methods[0],
                "--method",
                methods[1],
                "--resamples",
                "50",
                "--seed",
                "19",
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == first


def test_identical_methods_use_the_same_cluster_draw(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=0.5)
    _write_report(second_path, positive_dice=0.5)

    contrast = _compare_paths(first_path, second_path)["paired_differences"][0]
    assert contrast["same_cluster_draw_used_for_both_methods"] is True
    assert contrast["point_difference"] == 0.0
    assert contrast["descriptive_bootstrap_ci"]["lower"] == 0.0
    assert contrast["descriptive_bootstrap_ci"]["upper"] == 0.0


def test_foundation_model_and_exact_selected_method_identity_are_linked(
    tmp_path: Path,
) -> None:
    classical_path = tmp_path / "classical.json"
    foundation_path = tmp_path / "foundation.json"
    _write_report(classical_path, positive_dice=1.0)
    foundation = _write_report(foundation_path, positive_dice=0.5)
    _convert_to_foundation(foundation)
    _write_json(foundation_path, foundation)

    result = compare_reports(
        [f"classical={classical_path}", f"foundation={foundation_path}"],
        [
            "classical=classical_fold",
            "foundation=foundation_linear_probe",
        ],
        resamples=20,
    )
    assert result["reports"]["foundation"][
        "model_identity_sha256"
    ] == _canonical_sha256(foundation["model_identity"])

    foundation["model_identity"]["assets"]["model.safetensors"]["sha256"] = "9" * 64
    _write_json(foundation_path, foundation)
    with pytest.raises(ComparisonValidationError, match="weight SHA-256"):
        compare_reports(
            [f"classical={classical_path}", f"foundation={foundation_path}"],
            [
                "classical=classical_fold",
                "foundation=foundation_linear_probe",
            ],
            resamples=20,
        )


def test_loader_is_derived_from_real_loaderless_dinov2_schema(tmp_path: Path) -> None:
    classical_path = tmp_path / "classical.json"
    dinov2_path = tmp_path / "dinov2.json"
    _write_report(classical_path, positive_dice=1.0)
    dinov2 = _write_report(dinov2_path, positive_dice=0.5)
    _convert_to_real_shape_dinov2(dinov2)
    _write_json(dinov2_path, dinov2)

    result = compare_reports(
        [f"classical={classical_path}", f"dinov2={dinov2_path}"],
        ["classical=classical_fold", "dinov2=foundation_linear_probe"],
        resamples=20,
    )
    assert result["reports"]["dinov2"]["model_identity_sha256"] == _canonical_sha256(
        dinov2["model_identity"]
    )

    method_model = dinov2["run_provenance"]["value"]["method_model"]
    method_model["loader_identity"] = "forged_loader"
    dinov2["run_provenance"]["identity_sha256"] = _canonical_sha256(
        dinov2["run_provenance"]["value"]
    )
    _write_json(dinov2_path, dinov2)
    with pytest.raises(ComparisonValidationError, match="model loaders disagree"):
        compare_reports(
            [f"classical={classical_path}", f"dinov2={dinov2_path}"],
            ["classical=classical_fold", "dinov2=foundation_linear_probe"],
            resamples=20,
        )


def test_locked_manifest_mismatch_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    changed_manifest = _locked_manifest()
    changed_manifest[0] = {**changed_manifest[0], "image_sha256": "9" * 64}
    _write_report(
        second_path,
        positive_dice=0.5,
        locked_manifest=changed_manifest,
    )

    with pytest.raises(ComparisonValidationError, match="does not exactly match"):
        _compare_paths(first_path, second_path)


def test_tampered_outcome_hash_and_missing_method_are_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    second = _write_report(second_path, positive_dice=0.5)
    second["methods"]["classical_fold"]["locked_test_outcomes"][1]["tp"] = 4
    _write_json(second_path, second)

    with pytest.raises(ComparisonValidationError, match="outcome SHA-256"):
        _compare_paths(first_path, second_path)

    _write_report(second_path, positive_dice=0.5)
    with pytest.raises(ComparisonValidationError, match="does not exist"):
        compare_reports(
            [f"first={first_path}", f"second={second_path}"],
            ["first=missing", "second=classical_fold"],
            resamples=20,
        )


def test_descriptive_paired_interval_has_expected_direction(tmp_path: Path) -> None:
    high_path = tmp_path / "high.json"
    low_path = tmp_path / "low.json"
    _write_report(high_path, positive_dice=1.0)
    _write_report(low_path, positive_dice=0.0)

    result = _compare_paths(high_path, low_path, resamples=80)
    contrast = result["paired_differences"][0]
    assert contrast["point_difference"] == 1.0
    assert contrast["descriptive_bootstrap_ci"]["lower"] > 0.0
    assert result["claim_scope"]["p_values_computed"] is False
    assert result["claim_scope"]["superiority_claim_made"] is False
    assert result["claim_scope"]["noninferiority_claim_made"] is False


def test_configuration_hash_and_cross_report_contract_are_enforced(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    second = _write_report(second_path, positive_dice=0.5)
    second["configuration"]["max_dimension"] = 672
    _write_json(second_path, second)
    with pytest.raises(ComparisonValidationError, match="configuration SHA-256"):
        _compare_paths(first_path, second_path)

    _rehash_report(second)
    _write_json(second_path, second)
    with pytest.raises(ComparisonValidationError, match="evaluation contract"):
        _compare_paths(first_path, second_path)


def test_per_field_reference_domain_mismatch_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    second = _write_report(second_path, positive_dice=0.5)
    outcome = second["methods"]["classical_fold"]["locked_test_outcomes"][1]
    outcome["tn"] += 1
    outcome["n_valid"] += 1
    method = second["methods"]["classical_fold"]
    method["locked_test_outcomes_sha256"] = _canonical_sha256(
        method["locked_test_outcomes"]
    )
    _write_json(second_path, second)

    with pytest.raises(ComparisonValidationError, match="evaluation domain"):
        _compare_paths(first_path, second_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("claim", "claim scope"),
        ("status", "status"),
        ("dataset", "dataset_name.*canonical"),
        ("provenance", "capture identity"),
        ("code", "Git code identity"),
        ("environment", "dependency identity"),
        ("execution", "execution identity"),
        ("method_identity", "method identity"),
    ],
)
def test_self_asserted_claims_and_incomplete_identities_are_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    second = _write_report(second_path, positive_dice=0.5)
    if mutation == "claim":
        second["claim_scope"]["modality"] = "self-asserted modality"
    elif mutation == "status":
        second["status"] = "complete"
    elif mutation == "dataset":
        second["dataset"]["dataset_name"] = "self-asserted dataset"
    elif mutation == "provenance":
        del second["run_provenance"]["value"]["capture"]["approval_scope"]
        second["run_provenance"]["identity_sha256"] = _canonical_sha256(
            second["run_provenance"]["value"]
        )
    elif mutation == "code":
        second["run_provenance"]["value"]["code"]["commit"] = "not-a-commit"
        second["run_provenance"]["identity_sha256"] = _canonical_sha256(
            second["run_provenance"]["value"]
        )
    elif mutation == "environment":
        del second["run_provenance"]["value"]["environment"]["dependencies"]["numpy"]
        second["run_provenance"]["identity_sha256"] = _canonical_sha256(
            second["run_provenance"]["value"]
        )
    elif mutation == "execution":
        second["run_provenance"]["value"]["execution"]["precision"] = ""
        second["run_provenance"]["identity_sha256"] = _canonical_sha256(
            second["run_provenance"]["value"]
        )
    else:
        second["methods"]["classical_fold"]["method_identity"]["algorithm_family"] = (
            "forged"
        )
    _write_json(second_path, second)

    with pytest.raises(ComparisonValidationError, match=message):
        _compare_paths(first_path, second_path)


def test_manifest_leakage_is_recomputed_instead_of_trusting_flag(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0)
    second = _write_report(second_path, positive_dice=0.5)
    fit_manifest = second["splits"]["fit"]["manifest"]
    calibration_slide = second["splits"]["calibration"]["manifest"][0]["slide_id"]
    fit_manifest[0]["slide_id"] = calibration_slide
    second["splits"]["fit"] = _split_report(fit_manifest)
    _write_json(second_path, second)

    with pytest.raises(
        ComparisonValidationError,
        match="multiple organ/class/split strata|leakage audit",
    ):
        _compare_paths(first_path, second_path)


def test_fewer_than_two_positive_slides_per_organ_is_rejected(tmp_path: Path) -> None:
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    _write_report(first_path, positive_dice=1.0, positive_slides_per_organ=1)
    _write_report(second_path, positive_dice=0.5, positive_slides_per_organ=1)

    with pytest.raises(ComparisonValidationError, match="at least two positive"):
        _compare_paths(first_path, second_path)
