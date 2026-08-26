from __future__ import annotations

import unittest

import numpy as np

from foldcrack_qc.wsi import (
    ArrayPyramidSource,
    choose_level,
    iter_tile_windows,
    level_box_to_level0,
    stitch_core_predictions,
)


class WSITilingTests(unittest.TestCase):
    def test_physical_level_selection_and_coordinate_mapping(self) -> None:
        source = ArrayPyramidSource(
            [np.zeros((40, 60)), np.zeros((20, 30)), np.zeros((10, 15))],
            [0.25, 0.5, 1.0],
        )
        self.assertEqual(choose_level(source, 0.6), 1)
        self.assertEqual(
            level_box_to_level0(
                (2, 3, 8, 10),
                level_pixel_size_um=(0.5, 0.5),
                level0_pixel_size_um=(0.25, 0.25),
            ),
            (4, 6, 16, 20),
        )

    def test_halo_reads_stitch_without_seams_or_duplicate_cores(self) -> None:
        yy, xx = np.mgrid[:37, :53]
        score = yy * 1000.0 + xx
        source = ArrayPyramidSource([score], [0.5])
        windows = iter_tile_windows(
            source.level_shape(0),
            pixel_size_um=source.level_pixel_size_um(0),
            tile_size_um=(8.0, 10.0),
            halo_um=(1.5, 2.0),
        )
        predictions = [source.read_region(0, window.read) for window in windows]
        stitched = stitch_core_predictions(windows, predictions, source.level_shape(0))
        np.testing.assert_array_equal(stitched, score)
        self.assertGreater(len(windows), 1)
        self.assertTrue(any(window.read != window.core for window in windows))

    def test_invalid_prediction_geometry_fails_instead_of_silent_resize(self) -> None:
        windows = iter_tile_windows(
            (20, 20), pixel_size_um=1.0, tile_size_um=10.0, halo_um=2.0
        )
        predictions = [np.zeros(window.read_shape) for window in windows]
        predictions[0] = np.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "does not match"):
            stitch_core_predictions(windows, predictions, (20, 20))

        no_halo = iter_tile_windows(
            (20, 20), pixel_size_um=1.0, tile_size_um=10.0, halo_um=0.0
        )
        self.assertTrue(all(window.read == window.core for window in no_halo))


if __name__ == "__main__":
    unittest.main()
