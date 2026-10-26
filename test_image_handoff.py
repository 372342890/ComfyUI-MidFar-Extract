import importlib
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image


class ImageHandoffTests(unittest.TestCase):
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
