import json
import unittest
import numpy as np

try:
    from localization_view import localization_view
except ImportError:
    localization_view = None


class LocalizationViewTests(unittest.TestCase):
    def test_marked_target_uses_current_crop_coordinates_without_mutating_source(self):
        self.assertTrue(callable(localization_view))
        source = np.full((1, 100, 120, 3), .4, np.float32)
        original = source.copy()
        record = json.dumps({
            "id": "c01", "source_size": [1000, 500],
            "source_bbox_pixels": [200, 50, 320, 150],
            "bbox": [.21, .12, .31, .28],
            "target_point": [250, 200],
            "connected_parts": ["central peak"], "excluded_objects": ["other depth peak"],
        })
        marked, prompt, note = localization_view(source, record)
        np.testing.assert_array_equal(source, original)
        self.assertEqual(marked.shape, source.shape)
        self.assertFalse(np.array_equal(marked, source))
        self.assertEqual(json.loads(note)["target_point_crop_pixels"], [50, 50])
        self.assertIn("c01", prompt)
        self.assertIn("0..1000", prompt)
        self.assertIn("marker", prompt)

    def test_outside_target_point_is_not_silently_retargeted(self):
        self.assertTrue(callable(localization_view))
        source = np.zeros((1, 100, 120, 3), np.float32)
        record = json.dumps({
            "id": "c01", "source_size": [1000, 500],
            "source_bbox_pixels": [200, 50, 320, 150],
            "bbox": [.21, .12, .31, .28], "target_point": [800, 200],
            "connected_parts": [], "excluded_objects": [],
        })
        with self.assertRaisesRegex(ValueError, "target_point"):
            localization_view(source, record)
