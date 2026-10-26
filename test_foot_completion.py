import unittest
import numpy as np
try:
    from foot_completion import completion_context, complete_foot
except ImportError:
    completion_context = complete_foot = None


def guide():
    image = np.broadcast_to([1., 0., 1.], (1, 160, 160, 3)).copy().astype(np.float32)
    image[:, 30:110, 45:115] = [.3, .5, .7]
    return image


class FootCompletionTests(unittest.TestCase):
    def test_edit_region_covers_occluder_notches_above_fixed_fraction(self):
        image = guide()
        image[:, 76:110, 74:87] = [1, 0, 1]
        _, mask = completion_context(image, "#FF00FF")
        self.assertGreater(float(mask[0, 77, 80]), .25)

    def test_large_foot_notch_is_detected_without_rejecting_rounded_feet(self):
        from foot_completion import foot_profile_problem
        rgba = np.zeros((1, 160, 180, 4), np.float32)
        rgba[:, 30:130, 20:160, 3] = 1
        self.assertEqual(foot_profile_problem(rgba), "")
        rgba[:, 90:130, 75:105, 3] = 0
        self.assertIn("notch", foot_profile_problem(rgba))

    def test_foot_transition_matches_reference_color_without_touching_upper(self):
        source = guide()
        _, mask = completion_context(source, "#FF00FF")
        generated = source.copy()
        generated[:, 70:125, 45:115] = [.4, .6, .8]
        result, _ = complete_foot(generated, source, mask, "#FF00FF")
        np.testing.assert_allclose(result[0, 92, 80], [.3, .5, .7], atol=.04)
        np.testing.assert_array_equal(result[:, :65], source[:, :65])

    def test_api_exists(self):
        self.assertTrue(callable(completion_context))
        self.assertTrue(callable(complete_foot))

    @unittest.skipIf(completion_context is None, "not implemented")
    def test_edit_mask_only_covers_lower_target_and_never_canvas_edge(self):
        marked, mask = completion_context(guide(), "#FF00FF")
        self.assertEqual(marked.shape, guide().shape)
        self.assertEqual(mask.shape, (1, 160, 160))
        self.assertEqual(float(mask[:, :70].max()), 0)
        self.assertEqual(float(mask[:, -1].max()), 0)
        self.assertEqual(float(mask[:, :, 0].max()), 0)
        self.assertGreater(float(mask.sum()), 100)
        np.testing.assert_array_equal(marked[:, :65], guide()[:, :65])

    @unittest.skipIf(complete_foot is None, "not implemented")
    def test_generated_changes_outside_foot_are_discarded_inside_workflow(self):
        source = guide()
        _, mask = completion_context(source, "#FF00FF")
        generated = np.broadcast_to([.6, .7, .8], (1, 320, 320, 3)).copy()
        result, record = complete_foot(generated, source, mask, "#FF00FF")
        np.testing.assert_allclose(result[0, 85, 150], [.3, .5, .7], atol=1e-5)
        np.testing.assert_allclose(result[0, 0, 0], [1, 0, 1])
        self.assertTrue(record["original_upper_geometry_preserved"])
        self.assertGreater(record["generated_foot_pixels"], 100)

    @unittest.skipIf(complete_foot is None, "not implemented")
    def test_empty_guide_and_bad_mask_are_rejected(self):
        empty = np.broadcast_to([1., 0., 1.], (1, 160, 160, 3)).copy()
        with self.assertRaises(ValueError):
            completion_context(empty, "#FF00FF")
        with self.assertRaises(ValueError):
            complete_foot(guide(), guide(), np.ones((1, 160, 160)), "#FF00FF")

if __name__ == "__main__":
    unittest.main(verbosity=2)
