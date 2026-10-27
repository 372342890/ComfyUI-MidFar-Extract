import unittest
import numpy as np
from selection import selected_matte


def image():
    value = np.zeros((1, 128, 128, 3), np.float32)
    value[:] = [1, 0, 1]
    value[:, 32:96, 32:96] = [.6, .45, .3]
    return value


class ScreenContext(unittest.TestCase):
    def test_isolated_desaturated_key_pixel_in_background_is_noise(self):
        raw = image()
        raw[0, 0, 0] = np.array([229, 141, 255])/255
        rgba, report = selected_matte(raw, "#FF00FF", strict=False)
        self.assertEqual(report["removed_remote_pixels"], 0,
                         "A corner screen pixel must not become a detached foreground failure.")
        self.assertGreaterEqual(report["suppressed_screen_noise_pixels"], 1)
        self.assertFalse(report["touches_border"])
        self.assertEqual(report["meaningful_components"], 1)
        self.assertEqual(rgba.shape[-1], 4)

    def test_non_key_foreground_fragment_stays_a_blocking_detached_fragment(self):
        raw = image()
        raw[0, 0, 0] = [.6, .45, .3]
        _, report = selected_matte(raw, "#FF00FF", strict=False)
        self.assertEqual(report["removed_remote_pixels"], 1)
        self.assertEqual(report["suppressed_screen_noise_pixels"], 0)

    def test_larger_key_colored_region_is_not_classified_as_screen_noise(self):
        raw = image()
        raw[0, :5, :5] = np.array([229, 141, 255])/255
        _, report = selected_matte(raw, "#FF00FF", strict=False)
        self.assertEqual(report["suppressed_screen_noise_pixels"], 0)
        self.assertGreater(report["removed_remote_pixels"], 0)

    def test_two_real_units_still_fail_strict_check(self):
        raw = image()
        raw[:, 14:28, 14:28] = [.6, .45, .3]
        with self.assertRaisesRegex(ValueError, "Multiple substantial components"):
            selected_matte(raw, "#FF00FF", strict=True)


if __name__ == "__main__":
    unittest.main()
