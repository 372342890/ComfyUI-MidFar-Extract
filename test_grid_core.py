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

    def test_narrow_blank_cell_does_not_abort_the_other_13_assets(self):
        columns = [
            [0, 300, 625, 910, 1220],
            [0, 300, 580, 625, 910, 1220],
            [0, 245, 515, 742, 910, 1220],
        ]
        image, colors = marked_sheet([0, 330, 610, 970], columns, blank=(7,))
        tiles, report = split_sheet(image)
        indices = [index for index in range(1, 15) if index != 7]
        self.assert_one_marker_per_tile(tiles, colors, indices)
        self.assertEqual(report["row_columns"], [4, 5, 5])
        self.assertEqual(report["skipped_indices"], [7])

    def test_narrow_occupied_edge_cell_is_not_ignored(self):
        columns = [[0, 50, 610, 1220]]
        image, colors = marked_sheet([0, 300], columns)
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2, 3])
        self.assertEqual(report["row_x_boundaries"], columns)

    def test_short_first_row_is_detected_without_a_relative_height_limit(self):
        columns = [[0, 610, 1220]] * 3
        image, colors = marked_sheet([0, 50, 350, 970], columns)
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, list(range(1, 7)))
        self.assertEqual(report["y_boundaries"], [0, 50, 350, 970])

    def test_blank_gutters_do_not_require_similarly_sized_cells(self):
        image = np.full((200, 1000, 3), 255, np.uint8)
        colors = {1: (200, 40, 30), 2: (30, 180, 50), 3: (40, 50, 200)}
        for index, x in enumerate((40, 250, 780), 1):
            image[80:120, x - 20:x + 20] = colors[index]
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2, 3])
        self.assertEqual(report["row_x_detection"], ["gaps"])

    def test_ten_clear_columns_are_within_the_supported_cell_budget(self):
        columns = [list(range(0, 2001, 200))]
        image, colors = marked_sheet([0, 200], columns)
        tiles, report = split_sheet(image)
        self.assert_one_marker_per_tile(tiles, colors, list(range(1, 11)))
        self.assertEqual(report["row_columns"], [10])

    def test_overlapping_gray_walls_do_not_cut_a_marker_in_half(self):
        columns = [[0, 400, 800]]
        image, colors = marked_sheet([0, 300], columns)
        # Different walls overlap in a narrow projection band through asset 1.
        # Neither wall is a separator stroke with background on both sides.
        image[:140, 80:204] = 230
        image[160:, 196:320] = 230
        tiles, report = split_sheet(image)
        self.assertEqual(report["row_x_boundaries"], columns)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2])
        for tile, color in zip(tiles, colors.values()):
            self.assertEqual(np.all(tile == color, axis=2).sum(),
                             np.all(image == color, axis=2).sum())

    def test_tall_gray_part_stopping_before_row_ends_is_not_a_separator(self):
        columns = [[0, 400, 800]]
        image, colors = marked_sheet([0, 300], columns)
        image[30:270, 78:81] = 230
        tiles, report = split_sheet(image)
        self.assertEqual(report["row_x_boundaries"], columns)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2])
        self.assertTrue(np.all(tiles[0][30:270, 78:81] == 230))

    def test_gray_horizontal_shapes_do_not_replace_faint_grid_lines(self):
        columns = [[0, 400, 800]] * 2
        image, colors = marked_sheet([0, 300, 600], columns)
        image[50:154, :380] = 230
        image[146:250, 420:] = 230
        # Genuine low-contrast separators must still be detected on both axes.
        image[299:302] = 247
        image[:, 399:402] = 247
        tiles, report = split_sheet(image)
        self.assertEqual(report["y_boundaries"], [0, 300, 600])
        self.assertEqual(report["row_x_boundaries"], columns)
        self.assert_one_marker_per_tile(tiles, colors, [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
