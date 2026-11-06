"""Regression sheets contain distinct source markers, never user assets."""
import unittest

import numpy as np

from grid_core import split_sheet


def marked_sheet(row_bounds, column_bounds, blank=()):
    image = np.full((row_bounds[-1], column_bounds[0][-1], 3), 255, np.uint8)
    colors = {}
    index = 0
    for row, xb in enumerate(column_bounds):
        y0, y1 = row_bounds[row:row + 2]
        for x in xb[1:-1]:
            image[y0:y1, x - 1:x + 2] = 230
        for col in range(len(xb) - 1):
            index += 1
            if index in blank:
                continue
            color = (20 + index * 7, 40 + index * 5, 180 - index * 3)
            colors[index] = color
            x = (xb[col] + xb[col + 1]) // 2
            y = (y0 + y1) // 2
            image[y - 20:y + 20, x - 20:x + 20] = color
    for y in row_bounds[1:-1]:
        image[y - 1:y + 2] = 230
    return image, colors


class GridRegression(unittest.TestCase):
    def assert_one_marker_per_tile(self, tiles, colors, indices):
        self.assertEqual(len(tiles), len(indices))
        for tile, index in zip(tiles, indices):
            present = [key for key, color in colors.items()
                       if np.any(np.all(tile == color, axis=2))]
            self.assertEqual(present, [index])

    def test_staggered_4_4_5_columns_never_merge_adjacent_assets(self):
        row_bounds = [0, 330, 610, 970]
        columns = [[0, 300, 610, 910, 1220]] * 2 + [
            [0, 245, 515, 742, 910, 1220]]
        image, colors = marked_sheet(row_bounds, columns)
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, list(range(1, 14)))
        self.assertEqual(report["row_columns"], [4, 4, 5])
        self.assertEqual(report["row_x_boundaries"], columns)
        self.assertEqual(report["kept_indices"], list(range(1, 14)))

    def test_uniform_grid_and_blank_filter_keep_source_order(self):
        xb = [0, 300, 600, 900, 1200]
        image, colors = marked_sheet([0, 300, 600], [xb, xb], blank=(3, 8))
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2, 4, 5, 6, 7])
        self.assertEqual(report["x_boundaries"], xb)
        self.assertEqual(report["columns"], 4)
        self.assertEqual(report["skipped_indices"], [3, 8])

    def test_separated_parts_inside_one_cell_remain_one_asset(self):
        xb = [0, 400, 800]
        image, colors = marked_sheet([0, 400, 800], [xb, xb])
        image[180:220, 280:320] = colors[1]
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2, 3, 4])
        self.assertTrue(np.all(tiles[0][180:220, 280:320] == colors[1]))
        self.assertEqual(report["kept_count"], 4)

    def test_manual_grid_remains_authoritative(self):
        columns = [[0, 300, 610, 910, 1220]] * 2 + [
            [0, 245, 515, 742, 910, 1220]]
        image, _ = marked_sheet([0, 330, 610, 970], columns)
        _, report = split_sheet(image, rows=2, columns=2)
        self.assertEqual(report["rows"], 2)
        self.assertEqual(report["columns"], 2)
        self.assertEqual(report["x_boundaries"], [0, 610, 1220])
        self.assertEqual(report["y_boundaries"], [0, 485, 970])
        self.assertEqual(report["detection"], {"x": "manual", "y": "manual"})


if __name__ == "__main__":
    unittest.main()
