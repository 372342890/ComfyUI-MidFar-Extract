import importlib.util
import json
import unittest
import numpy as np

try:
    from target_guide import target_guide
except ImportError:
    target_guide = None

from test_autonomous_audit import plan

class TargetGuideTests(unittest.TestCase):
    def test_detailed_model_polygon_preserves_all_vertices(self):
        angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
        points = np.round(np.stack((500+280*np.cos(angles),
                                    500+280*np.sin(angles)), axis=-1)).astype(int).tolist()
        localization = json.dumps({
            "asset_id": "unit_a", "visible_polygon": points,
            "description": "detailed connected outline", "hidden_base": "short foot",
        })
        try:
            image, _, record = target_guide(np.full((1, 100, 120, 3), .4, np.float32),
                                            plan(), localization, "#FF00FF")
        except ValueError as error:
            self.fail(f"A detailed model polygon must remain usable: {error}")
        self.assertEqual(json.loads(record)["visible_polygon"], points)
        self.assertEqual(image.shape, (1, 120, 120, 3))

    def test_guide_is_editable_and_never_freezes_localization_errors(self):
        reference = np.full((1, 100, 120, 3), [.2, .4, .6], np.float32)
        localization = json.dumps({
            "asset_id": "unit_a",
            "visible_polygon": [[200, 800], [300, 500], [500, 100], [700, 500], [800, 800]],
            "description": "central peak and attached shoulder",
            "hidden_base": "complete own foot",
        })
        _, prompt, _ = target_guide(reference, plan(), localization, "#FF00FF")
        self.assertIn("editable", prompt)
        self.assertIn("incorrect fragments", prompt)
        self.assertNotIn("只重绘朱红", prompt)

    def test_api_available(self):
        self.assertTrue(callable(target_guide))

    @unittest.skipIf(target_guide is None, "guide not implemented")
    def test_current_reference_pixels_are_isolated_without_becoming_final_asset(self):
        reference = np.full((1, 100, 120, 3), [.2, .4, .6], np.float32)
        localization = json.dumps({
            "asset_id": "unit_a",
            "visible_polygon": [[200, 800], [300, 500], [500, 100], [700, 500], [800, 800]],
            "description": "main peak and its attached shoulder; separate foreground excluded",
            "hidden_base": "own foot behind foreground, extend naturally",
        })
        image, prompt, record = target_guide(reference, plan(), localization, "#FF00FF")
        report = json.loads(record)
        self.assertEqual(image.shape, (1, 120, 120, 3))
        self.assertTrue(report["guidance_only"])
        self.assertEqual(report["asset_id"], "unit_a")
        np.testing.assert_allclose(image[0, 0, 0], [1, 0, 1])
        self.assertIn("not a finished asset", prompt)
        self.assertIn("hidden", prompt)
        self.assertTrue((np.isclose(image[0], [.2, .4, .6]).all(-1)).any())

    @unittest.skipIf(target_guide is None, "guide not implemented")
    def test_malformed_or_retargeted_polygon_cannot_run(self):
        reference = np.zeros((1, 100, 120, 3), np.float32)
        good = {"asset_id": "unit_a",
                "visible_polygon": [[100, 800], [500, 100], [900, 800]],
                "description": "target", "hidden_base": "foot"}
        for changes in ({"asset_id": "other"}, {"visible_polygon": [[0, 0]]},
                        {"visible_polygon": [[-1, 800], [500, 100], [900, 800]]},
                        {"visible_polygon": [[.1, .8], [.5, .1], [.9, .8]]}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                target_guide(reference, plan(), json.dumps({**good, **changes}), "#FF00FF")

if __name__ == "__main__":
    unittest.main(verbosity=2)
