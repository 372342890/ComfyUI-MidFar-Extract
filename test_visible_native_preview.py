"""ByteArtist renders declared result slots; a second Comfy UI payload duplicates them."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

from test_visible_batch import image
from test_visible_pe_split import recognition

ROOT = Path(__file__).parent


class Tensor:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


def native_module():
    package = types.ModuleType("_visible_preview_test")
    package.__path__ = [str(ROOT)]
    torch = types.ModuleType("torch")
    torch.from_numpy = Tensor
    spec = importlib.util.spec_from_file_location(
        "_visible_preview_test.visible_nodes", ROOT / "visible_nodes.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"_visible_preview_test": package, "torch": torch}):
        spec.loader.exec_module(module)
    return module


class NativeVisiblePreview(unittest.TestCase):
    def setUp(self):
        self.nodes = native_module()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folders = types.ModuleType("folder_paths")
        folders.get_output_directory = lambda: self.directory.name
        self.folders = patch.dict(sys.modules, {"folder_paths": folders})
        self.folders.start()
        self.addCleanup(self.folders.stop)

    def test_pe_returns_only_declared_slots_without_a_duplicate_text_preview(self):
        pe = {"objects": [{"id": f"asset{i}", "prompt": f"提取第{i}个目标"}
                          for i in (1, 2, 3)]}
        result = self.nodes.BAMidFarVisiblePE().run(
            Tensor(image()), json.dumps(recognition()), json.dumps(pe), "test/run")
        self.assertIsInstance(result, tuple, "Native text slots already provide the preview.")
        self.assertEqual(len(result), 8)
        self.assertEqual(len(list(Path(self.directory.name).rglob("*.json"))), 1)

    def test_single_png_returns_one_image_slot_without_a_duplicate_ui_image(self):
        result = self.nodes.BAMidFarVisiblePNG().run(Tensor(image()), "#FF00FF", "test/one")
        self.assertIsInstance(result, tuple, "Native IMAGE slot already provides the preview.")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(list(Path(self.directory.name).rglob("*.png"))), 1)

    def test_three_png_node_saves_and_returns_each_image_once(self):
        result = self.nodes.BAMidFarVisibleThreePNG().run(
            *[Tensor(image(i)) for i in range(3)], "#FF00FF",
            "test/one", "test/two", "test/three")
        self.assertIsInstance(result, tuple, "Do not add a second generic three-image preview.")
        self.assertEqual(len(result), 4)
        paths = list(Path(self.directory.name).rglob("*.png"))
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(Image.open(path).mode == "RGBA" for path in paths))
        self.assertEqual(json.loads(result[-1])["png_count"], 3)


if __name__ == "__main__":
    unittest.main()
