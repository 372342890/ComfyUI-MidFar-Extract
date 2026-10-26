import importlib
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image


class ImageHandoffTests(unittest.TestCase):
    def test_rejected_inspections_are_saved_with_original_pixels_and_reviews(self):
        module = importlib.import_module("image_handoff")
        raw = np.full((1, 30, 40, 3), .4, np.float32)
        rgba = np.concatenate((raw, np.zeros((1, 30, 40, 1), np.float32)), axis=-1)
        before = raw.copy()
        with tempfile.TemporaryDirectory() as folder:
            prefix = "current-run/background/mountain_a"
            module.save_inspection(raw, raw, rgba, '{"valid":false}', "review this actual image",
                                   prefix, 2, folder)
            with Image.open(Path(folder)/(prefix+"_v2_raw.png")) as image:
                actual = np.asarray(image)
            np.testing.assert_array_equal(actual, np.round(raw[0]*255).astype("uint8"))
            with Image.open(Path(folder)/(prefix+"_v2_rgba.png")) as image:
                self.assertEqual(image.mode, "RGBA")
            self.assertFalse(json.loads((Path(folder)/(prefix+"_v2_numeric.json")).read_text())["valid"])
            module.save_review_records(prefix, ["numeric original"], ["malformed visual JSON"], folder)
            records = json.loads((Path(folder)/(prefix+"_reviews.json")).read_text())
            self.assertEqual(records["visual_reports"], ["malformed visual JSON"])
            self.assertEqual(records["numeric_reports"], ["numeric original"])
        np.testing.assert_array_equal(raw, before)

    def test_diagnostics_cannot_write_outside_the_output_directory(self):
        module = importlib.import_module("image_handoff")
        raw = np.ones((1, 10, 20, 3), np.float32)
        rgba = np.concatenate((raw, np.ones((1, 10, 20, 1), np.float32)), axis=-1)
        with tempfile.TemporaryDirectory() as folder:
            for prefix in ("../escape", "/outside", "C:/outside", "a/../../escape", "a\\escape"):
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    module.save_inspection(raw, raw, rgba, "{}", "", prefix, 1, folder)
                with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                    module.save_review_records(prefix, [], [], folder)

    def test_handoff_preserves_nonblack_pixels_and_content_address(self):
        module = importlib.import_module("image_handoff")
        source = np.zeros((1, 48, 96, 3), np.float32)
        source[:, :, :32] = [1, 0, 1]
        source[:, :, 32:64] = [.2, .4, .8]
        source[:, :, 64:] = [0, 1, 0]
        with tempfile.TemporaryDirectory() as folder:
            name = module.native_image_file(source, folder)
            actual = np.asarray(Image.open(Path(folder)/name).convert("RGB"))/255
            np.testing.assert_allclose(actual, source[0], atol=1/255)
            self.assertEqual(name, module.native_image_file(source, folder))
            changed = source.copy()
            changed[0, 10, 10] = [.4, .3, .2]
            self.assertNotEqual(name, module.native_image_file(changed, folder))

    def test_handoff_rejects_batches_and_nonfinite_data(self):
        module = importlib.import_module("image_handoff")
        with tempfile.TemporaryDirectory() as folder:
            for source in [np.zeros((2, 20, 20, 3)), np.full((1, 20, 20, 3), np.nan)]:
                with self.assertRaises(ValueError):
                    module.native_image_file(source, folder)


if __name__ == "__main__":
    unittest.main(verbosity=2)
