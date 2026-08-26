from __future__ import annotations

import unittest

import numpy as np

from foldcrack_qc.schema import ChannelRole, Modality
from foldcrack_qc.synthetic import (
    SyntheticConfig,
    generate_synthetic_dataset,
    generate_synthetic_sample,
)


class SyntheticSampleTests(unittest.TestCase):
    def test_modalities_have_expected_channels_and_dynamic_ranges(self) -> None:
        he = generate_synthetic_sample("he", seed=11, size=(96, 112))
        comet = generate_synthetic_sample("comet", seed=11, size=(96, 112))
        cosmx = generate_synthetic_sample("cosmx", seed=11, size=(96, 112))

        self.assertEqual(he.image.data.shape, (96, 112, 3))
        self.assertEqual(he.image.data.dtype, np.uint8)
        self.assertGreater(int(he.image.data.max()), int(he.image.data.min()))

        self.assertEqual(comet.image.data.shape, (96, 112, 5))
        self.assertEqual(comet.image.data.dtype, np.float32)
        self.assertGreaterEqual(float(comet.image.data.min()), 0.0)
        self.assertLessEqual(float(comet.image.data.max()), 1.0)
        self.assertEqual(comet.image.channel_roles[0], ChannelRole.NUCLEAR)
        self.assertIn(ChannelRole.AUTOFLUORESCENCE, comet.image.channel_roles)

        self.assertEqual(cosmx.image.data.shape, (96, 112, 4))
        self.assertEqual(cosmx.image.data.dtype, np.float32)
        self.assertEqual(cosmx.image.indices_for_role(ChannelRole.MEMBRANE), (3,))

    def test_same_seed_is_bitwise_deterministic(self) -> None:
        first = generate_synthetic_sample("comet", seed=29, size=(128, 128))
        second = generate_synthetic_sample("comet", seed=29, size=(128, 128))
        np.testing.assert_array_equal(first.image.data, second.image.data)
        self.assertEqual(first.sample_id, second.sample_id)
        for key in first.masks:
            np.testing.assert_array_equal(first.masks[key], second.masks[key])

    def test_paired_modalities_share_geometry(self) -> None:
        samples = [
            generate_synthetic_sample(modality, seed=7, size=(104, 120))
            for modality in ("he", "comet", "cosmx")
        ]
        for key in ("tissue", "fold", "crack", "hard_negative"):
            np.testing.assert_array_equal(samples[0].masks[key], samples[1].masks[key])
            np.testing.assert_array_equal(samples[0].masks[key], samples[2].masks[key])

    def test_masks_are_nonempty_valid_and_artifacts_do_not_overlap_hard_negatives(
        self,
    ) -> None:
        sample = generate_synthetic_sample("he", seed=3, size=(128, 144))
        tissue = sample.masks["tissue"]
        fold = sample.masks["fold"]
        crack = sample.masks["crack"]
        hard = sample.masks["hard_negative"]
        for mask in (tissue, fold, crack, hard):
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(mask.shape, (128, 144))
            self.assertTrue(mask.any())
        self.assertFalse(np.any((fold | crack | hard) & ~tissue))
        self.assertFalse(np.any(fold & crack))
        self.assertFalse(np.any((fold | crack) & hard))
        np.testing.assert_array_equal(sample.reference_artifact_mask, fold | crack)

    def test_include_flags_create_explicit_empty_masks(self) -> None:
        sample = generate_synthetic_sample(
            "cosmx",
            seed=4,
            size=(80, 96),
            include_fold=False,
            include_crack=False,
            include_hard_negatives=False,
        )
        self.assertFalse(sample.masks["fold"].any())
        self.assertFalse(sample.masks["crack"].any())
        self.assertFalse(sample.masks["hard_negative"].any())
        self.assertFalse(sample.reference_artifact_mask.any())

    def test_different_seed_changes_geometry_and_appearance(self) -> None:
        first = generate_synthetic_sample("he", seed=1, size=(128, 128))
        second = generate_synthetic_sample("he", seed=2, size=(128, 128))
        self.assertFalse(np.array_equal(first.image.data, second.image.data))
        self.assertFalse(np.array_equal(first.masks["fold"], second.masks["fold"]))

    def test_metadata_states_synthetic_limit(self) -> None:
        sample = generate_synthetic_sample("he", seed=0, size=(72, 72))
        self.assertTrue(sample.metadata["synthetic"])
        self.assertIn("not_clinical_validation", sample.metadata["intended_use"])
        self.assertSetEqual(
            set(sample.metadata["artifact_burden_fraction"]),
            {"fold", "crack", "hard_negative"},
        )


class SyntheticDatasetTests(unittest.TestCase):
    def test_dataset_is_balanced_unique_and_reproducible(self) -> None:
        first = generate_synthetic_dataset(n_per_modality=2, seed=10, size=(72, 80))
        second = generate_synthetic_dataset(n_per_modality=2, seed=10, size=(72, 80))
        self.assertEqual(len(first), 6)
        self.assertEqual(len({sample.sample_id for sample in first}), 6)
        counts = {modality: 0 for modality in Modality}
        for sample in first:
            counts[sample.modality] += 1
        self.assertEqual(counts, {Modality.HE: 2, Modality.COMET: 2, Modality.COSMX: 2})
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left.image.data, right.image.data)

    def test_dataset_validates_cardinality(self) -> None:
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(n_per_modality=0)
        with self.assertRaises(ValueError):
            generate_synthetic_dataset(modalities=())
        with self.assertRaises(ValueError):
            SyntheticConfig(size=(32, 32))


if __name__ == "__main__":
    unittest.main()
