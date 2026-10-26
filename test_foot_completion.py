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
