import unittest

import numpy as np

from undistort import fit_line_pca, line_distances, linearity_rows, summarize_linearity


class UndistortionNumericsTest(unittest.TestCase):
    def test_line_fit_has_zero_distance_for_collinear_points(self):
        points = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        distances = line_distances(points, fit_line_pca(points))
        np.testing.assert_allclose(distances, 0.0, atol=1e-12)

    def test_linearity_rows_preserves_grid_counts(self):
        points = np.zeros((77, 2), dtype=np.float64)
        grid = points.reshape(7, 11, 2)
        grid[:, :, 0] = np.arange(11)
        grid[:, :, 1] = np.arange(7)[:, None]
        rows = linearity_rows(points, "L1", (11, 7), "before")
        self.assertEqual(len(rows), 18)
        self.assertEqual(sum(row["direction"] == "horizontal" for row in rows), 7)
        self.assertEqual(sum(row["direction"] == "vertical" for row in rows), 11)
        self.assertTrue(all(row["point_count"] in (7, 11) for row in rows))

    def test_summary_reports_before_and_after(self):
        rows = [
            {"image_name": "L1", "stage": "before", "direction": "horizontal", "rmse_px": 1.0},
            {"image_name": "L1", "stage": "before", "direction": "vertical", "rmse_px": 2.0},
            {"image_name": "L1", "stage": "after", "direction": "horizontal", "rmse_px": 0.5},
            {"image_name": "L1", "stage": "after", "direction": "vertical", "rmse_px": 1.0},
        ]
        summary = summarize_linearity(rows)
        overall = next(row for row in summary if row["direction"] == "all")
        self.assertAlmostEqual(overall["before_rmse_px"], np.sqrt(2.5))
        self.assertAlmostEqual(overall["after_rmse_px"], np.sqrt(0.625))


if __name__ == "__main__":
    unittest.main()
