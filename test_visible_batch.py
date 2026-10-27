import importlib.util
import unittest

import numpy as np

if importlib.util.find_spec("visible_batch"):
    from visible_batch import process_three
else:
    process_three = None


def image(index=0, size=300):
    value = np.full((1, size, size, 3), [1, 0, 1], np.float32)
    colors = ([.2, .5, .8], [.65, .45, .25], [.3, .65, .4])
    value[:, size//3:2*size//3, size//3:2*size//3] = colors[index]
    if index == 2:
        value[:, size//3:size//2, size//3:size//2] = [1, 0, 1]
    return value


class VisibleBatch(unittest.TestCase):
    def call(self, values=None):
        self.assertIsNotNone(process_three, "Shared three-image processor is missing.")
        return process_three([image(i) for i in range(3)] if values is None else values,
                             "#FF00FF")

    def test_tiny_non_key_noise_is_recorded_without_aborting_three_outputs(self):
        noisy = image()
        for row in (25, 27, 29):
            noisy[0, row, row] = [.2, .5, .8]
        outputs, records = self.call([noisy, image(1), image(2)])
        self.assertEqual(len(outputs), 3)
        self.assertEqual(records[0]["cleaned_remote_pixels"], 3)
        self.assertEqual(records[0]["status"], "warning")
        self.assertEqual(records[0]["cleanup_limit_pixels"], 4)
        self.assertFalse(records[0]["warning"] == "")

    def test_remote_five_by_five_object_part_is_still_rejected(self):
        noisy = image()
        noisy[:, 25:30, 25:30] = [.2, .5, .8]
        with self.assertRaisesRegex(ValueError, "第1路"):
            self.call([noisy, image(1), image(2)])

    def test_canvas_border_cut_is_still_rejected(self):
        clipped = image()
        clipped[:, 100:130, :110] = [.2, .5, .8]
        with self.assertRaisesRegex(ValueError, "第1路"):
            self.call([clipped, image(1), image(2)])

    def test_identical_outputs_are_rejected_even_with_different_target_ids(self):
        same = image()
        with self.assertRaisesRegex(ValueError, "重复"):
            self.call([same, same.copy(), image(2)])

    def test_resized_identical_assets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            self.call([image(), image(size=600), image(2)])

    def test_same_shape_with_changed_brightness_is_a_duplicate_variant(self):
        variant = image()
        variant[:, 100:200, 100:200] *= .7
        with self.assertRaisesRegex(ValueError, "重复"):
            self.call([image(), variant, image(2)])

    def test_shared_shape_with_distinct_colors_is_not_a_duplicate(self):
        outputs, records = self.call()
        self.assertEqual(len(outputs), 3)
        self.assertTrue(all(record["status"] == "success" for record in records))

    def test_distinct_asset_proportions_with_same_color_are_not_duplicates(self):
        wide = image()
        wide[:, 100:200, 100:200] = [1, 0, 1]
        wide[:, 120:180, 80:220] = [.2, .5, .8]
        try:
            outputs, _ = self.call([image(), wide, image(2)])
        except ValueError as error:
            self.fail(f"Different proportions must not be flattened into duplicates: {error}")
        self.assertEqual(len(outputs), 3)

    def test_exactly_three_images_are_required(self):
        with self.assertRaisesRegex(ValueError, "三"):
            self.call([image(), image(1)])


if __name__ == "__main__":
    unittest.main()
