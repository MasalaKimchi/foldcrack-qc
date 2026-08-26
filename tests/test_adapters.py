from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from foldcrack_qc.adapters import (
    COMETAdapter,
    CosMxAdapter,
    HEAdapter,
    adapt_image,
    get_adapter,
    load_sample,
    read_image,
)
from foldcrack_qc.schema import (
    ArtifactKind,
    CanonicalImage,
    ChannelRole,
    Modality,
    QCSample,
    QCResult,
)


class SchemaTests(unittest.TestCase):
    def test_modality_aliases_and_canonical_grayscale(self) -> None:
        image = CanonicalImage(np.zeros((12, 14), np.uint8), "H&E", pixel_size_um=0.5)
        self.assertEqual(image.modality, Modality.HE)
        self.assertEqual(image.data.shape, (12, 14, 1))
        self.assertEqual(image.pixel_size_um, (0.5, 0.5))
        self.assertEqual(image.spatial_shape, (12, 14))

    def test_sample_normalizes_masks_and_combines_reference(self) -> None:
        image = CanonicalImage(np.zeros((8, 9, 3), np.uint8), "he")
        fold = np.zeros((8, 9), np.uint8)
        tear = np.zeros((8, 9), np.uint8)
        fold[1:3, 2:5] = 1
        tear[5:7, 6] = 1
        sample = QCSample("case", image, {ArtifactKind.FOLD: fold, "tear": tear})
        self.assertEqual(sample.masks["fold"].dtype, np.bool_)
        self.assertIn("crack", sample.masks)
        np.testing.assert_array_equal(
            sample.reference_artifact_mask, fold.astype(bool) | tear.astype(bool)
        )
        self.assertTrue(sample.tissue_mask.all())

    def test_result_validates_shapes_and_combines_predictions(self) -> None:
        fold_score = np.zeros((8, 9), np.float64)
        crack_mask = np.zeros((8, 9), bool)
        crack_mask[3, 4] = True
        result = QCResult(
            "case",
            "cosmx",
            score_maps={"fold": fold_score},
            masks={"crack": crack_mask},
            summary_scores={"artifact_burden": np.float32(0.1)},
            runtime_seconds=0.03,
        )
        self.assertEqual(result.score_maps["fold"].dtype, np.float32)
        self.assertEqual(result.spatial_shape, (8, 9))
        np.testing.assert_array_equal(result.artifact_mask, crack_mask)

    def test_schema_rejects_inconsistent_metadata(self) -> None:
        with self.assertRaises(ValueError):
            CanonicalImage(
                np.zeros((8, 9, 3), np.uint8),
                "he",
                channel_names=("red", "green"),
            )

    def test_schema_rejects_nonfinite_images_scores_and_nonbinary_masks(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.float32)
        image[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            CanonicalImage(image, modality="he")

        with self.assertRaisesRegex(ValueError, "not binary"):
            QCSample(
                sample_id="bad-mask",
                image=CanonicalImage(np.zeros((8, 8, 3)), modality="he"),
                masks={"fold": np.full((8, 8), 2, dtype=np.uint8)},
            )

        with self.assertRaisesRegex(ValueError, "non-finite"):
            QCResult(
                sample_id="bad-score",
                modality="he",
                score_maps={"fold": np.full((8, 8), np.inf)},
            )
        with self.assertRaises(ValueError):
            QCSample(
                "bad",
                CanonicalImage(np.zeros((8, 9, 1), np.uint8), "he"),
                masks={"fold": np.zeros((9, 8), bool)},
            )


class AdapterTests(unittest.TestCase):
    def test_adapter_dispatch(self) -> None:
        self.assertIsInstance(get_adapter("H&E"), HEAdapter)
        self.assertIsInstance(get_adapter("comet"), COMETAdapter)
        self.assertIsInstance(get_adapter("CosMx SMI"), CosMxAdapter)

    def test_he_preserves_rgb_and_can_convert_bgr(self) -> None:
        bgr = np.zeros((20, 24, 3), dtype=np.uint8)
        bgr[..., 0] = 11
        bgr[..., 1] = 22
        bgr[..., 2] = 33
        image = adapt_image(bgr, "he", color_order="bgr", pixel_size_um=(0.25, 0.3))
        self.assertEqual(tuple(image.data[0, 0]), (33, 22, 11))
        self.assertEqual(image.channel_names, ("red", "green", "blue"))
        self.assertEqual(
            image.channel_roles,
            (
                ChannelRole.BRIGHTFIELD_RED,
                ChannelRole.BRIGHTFIELD_GREEN,
                ChannelRole.BRIGHTFIELD_BLUE,
            ),
        )
        self.assertEqual(image.pixel_size_um, (0.25, 0.3))
        self.assertTrue(image.metadata["raw_intensity_preserved"])

    def test_comet_channel_first_semantic_resolution(self) -> None:
        array = np.zeros((5, 72, 80), np.float32)
        names = ["DAPI", "FITC", "AF_TRITC", "autofluorescence Cy5", "Cy7"]
        image = adapt_image(array, "comet", channel_names=names)
        self.assertEqual(image.data.shape, (72, 80, 5))
        self.assertEqual(image.channel_roles[0], ChannelRole.NUCLEAR)
        self.assertEqual(image.channel_roles[1], ChannelRole.MARKER)
        self.assertEqual(image.channel_roles[2], ChannelRole.AUTOFLUORESCENCE)
        self.assertEqual(image.indices_for_role("autofluorescence"), (2, 3))
        self.assertEqual(image.metadata["structural_channel_indices"], [0, 2, 3])

    def test_cosmx_roles_are_name_based_after_channel_reorder(self) -> None:
        names = ["CD298/B2M", "CD45", "DAPI", "PanCK"]
        array = np.zeros((68, 70, 4), np.uint16)
        image = CosMxAdapter().adapt(array, channel_names=names)
        self.assertEqual(
            image.channel_roles,
            (
                ChannelRole.MEMBRANE,
                ChannelRole.IMMUNE,
                ChannelRole.NUCLEAR,
                ChannelRole.CYTOPLASM,
            ),
        )
        self.assertIs(image.channel("nuclear").base, image.data)

    def test_explicit_axis_handles_small_channel_first_crop(self) -> None:
        array = np.zeros((4, 16, 18), np.float32)
        image = adapt_image(array, "cosmx", channel_axis=0)
        self.assertEqual(image.data.shape, (16, 18, 4))
        self.assertEqual(image.channel_names, ("DAPI", "PanCK", "CD45", "CD298_B2M"))


class ReaderTests(unittest.TestCase):
    def test_npy_npz_and_load_sample(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            array = np.arange(18 * 20 * 3, dtype=np.uint16).reshape(18, 20, 3)
            np.save(directory / "case.npy", array)
            np.savez(directory / "case.npz", metadata=np.ones(2), image=array)
            np.savez(directory / "ambiguous.npz", x=array, y=array + 1)

            np.testing.assert_array_equal(read_image(directory / "case.npy"), array)
            np.testing.assert_array_equal(read_image(directory / "case.npz"), array)
            with self.assertRaises(ValueError):
                read_image(directory / "ambiguous.npz")
            np.testing.assert_array_equal(
                read_image(directory / "ambiguous.npz", key="y"), array + 1
            )

            sample = load_sample(directory / "case.npy", "he", pixel_size_um=0.25)
            self.assertEqual(sample.sample_id, "case")
            self.assertEqual(sample.image.source_path, str(directory / "case.npy"))
            self.assertEqual(sample.image.data.dtype, np.uint16)

    def test_common_image_reader_returns_rgb(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "color.png"
            rgb = np.zeros((10, 12, 3), dtype=np.uint8)
            rgb[..., 0] = 201
            rgb[..., 1] = 102
            rgb[..., 2] = 7
            ok = cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            self.assertTrue(ok)
            decoded = read_image(path)
            np.testing.assert_array_equal(decoded, rgb)

    def test_missing_file_is_clear(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_image("does-not-exist.npy")


if __name__ == "__main__":
    unittest.main()
