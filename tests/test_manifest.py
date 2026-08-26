from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from foldcrack_qc.manifest import (
    ManifestValidationError,
    load_samples,
    validate_manifest,
)
from foldcrack_qc.schema import Modality
from foldcrack_qc.synthetic import generate_synthetic_sample


class ManifestTests(unittest.TestCase):
    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _write_sample_arrays(
        self,
        directory: Path,
        modality: str,
        index: int,
        *,
        channel_first: bool = False,
    ) -> dict[str, object]:
        sample = generate_synthetic_sample(modality, seed=100 + index, size=(72, 80))
        image = sample.image.data
        channel_axis = -1
        if channel_first:
            image = np.moveaxis(image, -1, 0)
            channel_axis = 0

        image_name = f"image-{index}.npy"
        np.save(directory / image_name, image)
        mask_fields: dict[str, str] = {}
        for name in ("fold", "crack", "tissue"):
            mask_name = f"{name}-{index}.npy"
            np.save(directory / mask_name, sample.masks[name].astype(np.uint8))
            mask_fields[f"{name}_mask_path"] = mask_name

        return {
            "sample_id": f"sample-{index}",
            "patient_id": f"patient-{index}",
            "slide_id": f"slide-{index}",
            "modality": modality,
            "image_path": image_name,
            "fold_mask_path": mask_fields["fold_mask_path"],
            "crack_mask_path": mask_fields["crack_mask_path"],
            "tissue_mask_path": mask_fields["tissue_mask_path"],
            "channel_names": list(sample.image.channel_names),
            "channel_axis": channel_axis,
            "pixel_size_um": [0.5, 0.6],
            "split": ("train", "validation", "test")[index % 3],
        }

    def _lock_record(
        self, directory: Path, record: dict[str, object]
    ) -> dict[str, object]:
        locked = dict(record)
        locked.update(
            block_id=f"block-{record['sample_id']}",
            run_id=f"run-{record['sample_id']}",
            source_id=f"source-{record['sample_id']}",
            cohort="prevalence",
        )
        valid_name = f"valid-{record['sample_id']}.npy"
        tissue_path = directory / str(record["tissue_mask_path"])
        np.save(directory / valid_name, np.load(tissue_path, allow_pickle=False))
        locked["valid_mask_path"] = valid_name
        for path_field in (
            "image_path",
            "fold_mask_path",
            "crack_mask_path",
            "tissue_mask_path",
            "valid_mask_path",
        ):
            checksum_field = path_field.removesuffix("_path") + "_sha256"
            locked[checksum_field] = self._sha256(directory / str(locked[path_field]))
        return locked

    def test_json_loads_all_modalities_and_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [
                self._write_sample_arrays(directory, "he", 0),
                self._write_sample_arrays(directory, "comet", 1, channel_first=True),
                self._write_sample_arrays(directory, "cosmx", 2),
            ]
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps({"samples": records}), encoding="utf-8")

            report = validate_manifest(manifest_path)
            self.assertTrue(report.valid, report.to_dict())
            self.assertEqual(report.record_count, 3)
            self.assertEqual(report.valid_sample_count, 3)

            samples = load_samples(manifest_path)
            self.assertEqual([sample.modality for sample in samples], list(Modality))
            self.assertEqual(
                samples[0].image.source_path, str((directory / "image-0.npy").resolve())
            )
            self.assertEqual(samples[1].image.data.shape, (72, 80, 5))
            self.assertEqual(samples[2].image.data.shape, (72, 80, 4))
            for sample in samples:
                self.assertSetEqual(set(sample.masks), {"fold", "crack", "tissue"})
                self.assertTrue(
                    all(mask.dtype == np.bool_ for mask in sample.masks.values())
                )
                self.assertEqual(sample.image.pixel_size_um, (0.5, 0.6))
                self.assertIn("patient_id", sample.metadata)
                self.assertIn("slide_id", sample.metadata)

    def test_jsonl_is_supported_and_blank_lines_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self._write_sample_arrays(directory, "he", 0)
            second = self._write_sample_arrays(directory, "cosmx", 1)
            path = directory / "manifest.jsonl"
            path.write_text(
                json.dumps(first) + "\n\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )

            report = validate_manifest(path)
            self.assertTrue(report.is_valid, report.to_dict())
            self.assertEqual(report.record_count, 2)
            self.assertEqual(len(load_samples(path)), 2)

    def test_duplicate_and_required_fields_are_structured_issues(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = self._write_sample_arrays(directory, "he", 0)
            duplicate = dict(record)
            duplicate["slide_id"] = "another-slide"
            duplicate["patient_id"] = "another-patient"
            duplicate["image_path"] = "image-0.npy"
            incomplete = {"sample_id": "incomplete", "patient_id": "patient-x"}
            path = directory / "manifest.json"
            path.write_text(
                json.dumps([record, duplicate, incomplete]), encoding="utf-8"
            )

            report = validate_manifest(path)
            codes = {issue.code for issue in report.issues}
            self.assertFalse(report.valid)
            self.assertIn("duplicate_sample_id", codes)
            self.assertIn("missing_required_field", codes)
            self.assertLess(report.valid_sample_count, report.record_count)

    def test_patient_and_slide_leakage_are_detected_without_echoing_phi(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self._write_sample_arrays(directory, "he", 0)
            second = self._write_sample_arrays(directory, "he", 1)
            secret_patient = "PRIVATE-PATIENT-8472"
            secret_slide = "PRIVATE-SLIDE-9331"
            first.update(
                patient_id=secret_patient, slide_id=secret_slide, split="train"
            )
            second.update(
                patient_id=secret_patient, slide_id=secret_slide, split="test"
            )
            path = directory / "manifest.json"
            path.write_text(json.dumps([first, second]), encoding="utf-8")

            report = validate_manifest(path)
            codes = {issue.code for issue in report.issues}
            self.assertIn("patient_split_leakage", codes)
            self.assertIn("slide_split_leakage", codes)
            serialized = json.dumps(report.to_dict())
            self.assertNotIn(secret_patient, serialized)
            self.assertNotIn(secret_slide, serialized)
            self.assertIn("sample-1", serialized)

            with self.assertRaises(ManifestValidationError) as caught:
                load_samples(path)
            self.assertIn(
                caught.exception.issue.code,
                {"patient_split_leakage", "slide_split_leakage"},
            )
            self.assertNotIn(secret_patient, str(caught.exception))
            self.assertNotIn(secret_slide, str(caught.exception))

    def test_mismatched_mask_is_rejected_fail_fast_and_path_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = self._write_sample_arrays(directory, "comet", 0)
            private_mask_name = "PRIVATE-PATIENT-MASK.npy"
            np.save(directory / private_mask_name, np.zeros((12, 13), dtype=np.uint8))
            record["crack_mask_path"] = private_mask_name
            path = directory / "manifest.json"
            path.write_text(json.dumps([record]), encoding="utf-8")

            report = validate_manifest(path)
            self.assertFalse(report.valid)
            self.assertIn(
                "mask_shape_mismatch", {issue.code for issue in report.issues}
            )
            self.assertEqual(report.valid_sample_count, 0)

            with self.assertRaises(ManifestValidationError) as caught:
                load_samples(path)
            self.assertEqual(caught.exception.issue.code, "mask_shape_mismatch")
            self.assertIn("sample-0", str(caught.exception))
            self.assertNotIn(private_mask_name, str(caught.exception))

    def test_missing_group_identifier_prevents_unverifiable_split(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = self._write_sample_arrays(directory, "cosmx", 0)
            record.pop("patient_id")
            record.pop("slide_id")
            path = directory / "manifest.json"
            path.write_text(json.dumps([record]), encoding="utf-8")

            report = validate_manifest(path)
            self.assertIn("missing_group_id", [issue.code for issue in report.issues])
            with self.assertRaises(ManifestValidationError) as caught:
                load_samples(path)
            self.assertEqual(caught.exception.issue.code, "missing_group_id")

    def test_jsonl_parse_error_is_reported_without_source_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.jsonl"
            secret = "PRIVATE-SLIDE-IN-MALFORMED-JSON"
            path.write_text('{"slide_id": "' + secret + '"\n', encoding="utf-8")

            report = validate_manifest(path)
            self.assertFalse(report.valid)
            self.assertEqual(report.issues[0].code, "json_decode_error")
            self.assertNotIn(secret, json.dumps(report.to_dict()))

    def test_empty_manifest_cannot_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text("[]", encoding="utf-8")
            report = validate_manifest(path)
            self.assertFalse(report.valid)
            self.assertEqual(report.issues[0].code, "empty_manifest")
            with self.assertRaises(ManifestValidationError):
                load_samples(path)

    def test_exploratory_warnings_become_strict_lock_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            record = self._write_sample_arrays(directory, "comet", 0)
            record.pop("channel_names")
            record.pop("pixel_size_um")
            path = directory / "manifest.json"
            path.write_text(json.dumps([record]), encoding="utf-8")

            exploratory = validate_manifest(path)
            self.assertTrue(exploratory.valid, exploratory.to_dict())
            warning_codes = {issue.code for issue in exploratory.issues}
            self.assertIn("missing_pixel_size", warning_codes)
            self.assertIn("missing_fluorescence_channel_names", warning_codes)
            self.assertTrue(
                all(issue.severity == "warning" for issue in exploratory.issues)
            )

            strict = validate_manifest(path, strict=True)
            self.assertFalse(strict.valid)
            self.assertGreater(strict.error_count, 0)

    def test_strict_manifest_with_checksums_and_valid_masks_loads_all_modalities(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            records = [
                self._lock_record(
                    directory, self._write_sample_arrays(directory, "he", 0)
                ),
                self._lock_record(
                    directory,
                    self._write_sample_arrays(
                        directory, "comet", 1, channel_first=True
                    ),
                ),
                self._lock_record(
                    directory, self._write_sample_arrays(directory, "cosmx", 2)
                ),
            ]
            path = directory / "manifest.json"
            path.write_text(json.dumps({"samples": records}), encoding="utf-8")

            report = validate_manifest(path, strict=True)
            self.assertTrue(report.valid, report.to_dict())
            samples = load_samples(path, strict=True)
            self.assertEqual(len(samples), 3)
            self.assertTrue(
                all(sample.metadata["strict_manifest"] for sample in samples)
            )
            self.assertTrue(all("valid" in sample.masks for sample in samples))

    def test_strict_checks_block_run_source_and_content_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = self._lock_record(
                directory, self._write_sample_arrays(directory, "he", 0)
            )
            second = self._lock_record(
                directory, self._write_sample_arrays(directory, "he", 1)
            )
            first.update(split="train", cohort="development")
            second.update(split="test", cohort="prevalence")
            second["block_id"] = first["block_id"]
            second["run_id"] = first["run_id"]
            second["source_id"] = first["source_id"]

            # Separate path, identical bytes: exercises file- and decoded-content checks.
            copied_name = "copied-image.npy"
            shutil.copyfile(
                directory / str(first["image_path"]), directory / copied_name
            )
            second["image_path"] = copied_name
            second["image_sha256"] = self._sha256(directory / copied_name)
            path = directory / "manifest.json"
            path.write_text(json.dumps([first, second]), encoding="utf-8")

            report = validate_manifest(path, strict=True)
            codes = {issue.code for issue in report.issues}
            self.assertIn("block_split_leakage", codes)
            self.assertIn("run_split_leakage", codes)
            self.assertIn("source_split_leakage", codes)
            self.assertIn("file_checksum_split_leakage", codes)
            self.assertIn("image_content_split_leakage", codes)

    def test_nonfinite_images_and_nonbinary_masks_are_never_silently_converted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            image_record = self._write_sample_arrays(directory, "comet", 0)
            image = np.load(
                directory / str(image_record["image_path"]), allow_pickle=False
            )
            image.flat[0] = np.nan
            np.save(directory / str(image_record["image_path"]), image)

            mask_record = self._write_sample_arrays(directory, "he", 1)
            bad_mask = np.load(
                directory / str(mask_record["fold_mask_path"]), allow_pickle=False
            ).astype(np.float32)
            bad_mask.flat[0] = 2.0
            np.save(directory / str(mask_record["fold_mask_path"]), bad_mask)
            path = directory / "manifest.json"
            path.write_text(json.dumps([image_record, mask_record]), encoding="utf-8")

            report = validate_manifest(path)
            codes = {issue.code for issue in report.issues}
            self.assertIn("image_nonfinite", codes)
            self.assertIn("mask_non_binary", codes)

    def test_invalid_early_record_does_not_poison_duplicate_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            valid = self._write_sample_arrays(directory, "he", 0)
            invalid = dict(valid)
            invalid.pop("image_path")
            path = directory / "manifest.json"
            path.write_text(json.dumps([invalid, valid]), encoding="utf-8")

            report = validate_manifest(path)
            codes = [issue.code for issue in report.issues]
            self.assertIn("missing_required_field", codes)
            self.assertNotIn("duplicate_sample_id", codes)
            self.assertEqual(report.valid_sample_count, 1)


if __name__ == "__main__":
    unittest.main()
