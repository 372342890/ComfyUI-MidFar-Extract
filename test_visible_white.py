"""White backgrounds are accepted without damaging pale object pixels."""
import json
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from test_visible_batch import image
import test_visible_native_preview as native_helpers
from test_visible_pe_split import recognition
from visible_batch import process_three
from visible_pipeline import split_prompts
from visible_workflow import build_workflow


def white_image(index=0):
    source = image(index)
    magenta = (source[..., 0] == 1) & (source[..., 1] == 0) & (source[..., 2] == 1)
    source[magenta] = 1
    return source


class WhiteBackgroundOutput(unittest.TestCase):
    def test_three_white_images_pass_without_requiring_alpha(self):
        sources = [white_image(i) for i in range(3)]
        outputs, records = process_three(sources, "#FFFFFF")
        for source, output, record in zip(sources, outputs, records):
            np.testing.assert_array_equal(output, source)
            self.assertEqual(output.shape[-1], 3)
            self.assertEqual(record["background"], "white")
            self.assertFalse(record["alpha_required"])

    def test_pure_white_and_near_white_snow_pixels_are_preserved(self):
        snowy = white_image()
        snowy[:, 110:130, 110:130] = 1
        snowy[:, 150:170, 120:140] = [.985, .985, 1]
        outputs, _ = process_three([snowy, white_image(1), white_image(2)], "#FFFFFF")
        np.testing.assert_array_equal(outputs[0], snowy)

    def test_slightly_off_white_background_is_accepted_without_repainting(self):
        source = white_image()
        source[np.all(source == 1, axis=-1)] = [.97, .98, .99]
        outputs, _ = process_three([source, white_image(1), white_image(2)], "#FFFFFF")
        np.testing.assert_array_equal(outputs[0], source)

    def test_white_background_cannot_disguise_duplicate_objects(self):
        source = white_image()
        with self.assertRaisesRegex(ValueError, "重复"):
            process_three([source, source.copy(), white_image(2)], "#FFFFFF")

    def test_white_background_cannot_disguise_a_brightness_variant(self):
        variant = white_image()
        variant[:, 100:200, 100:200] *= .7
        with self.assertRaisesRegex(ValueError, "重复"):
            process_three([white_image(), variant, white_image(2)], "#FFFFFF")

    def test_blank_white_image_is_still_rejected(self):
        with self.assertRaisesRegex(ValueError, "第1路"):
            process_three([np.ones((1, 300, 300, 3), np.float32),
                           white_image(1), white_image(2)], "#FFFFFF")

    def test_border_cut_is_still_rejected(self):
        source = white_image()
        source[:, 100:150, :120] = [.2, .5, .8]
        with self.assertRaisesRegex(ValueError, "第1路"):
            process_three([source, white_image(1), white_image(2)], "#FFFFFF")

    def test_pe_requests_white_background_and_preserves_target_exclusions(self):
        entries = {"objects": [{"id": f"asset{i}", "prompt": "提取，{KEY_COLOR}背景"}
                               for i in (1, 2, 3)]}
        outputs = split_prompts(
            json.dumps(recognition()), json.dumps(entries), white_image(),
            "white/test", background_mode="white")
        self.assertEqual(outputs[3], "#FFFFFF")
        self.assertIn("白底", outputs[0])
        self.assertIn("雪山2", outputs[0])
        self.assertIn("雪山3", outputs[0])
        self.assertNotIn("#00FF00", outputs[0])
        self.assertNotIn("{KEY_COLOR}", outputs[0])
        self.assertEqual(json.loads(outputs[-1])["background_mode"], "white")

    def test_white_workflow_retains_independent_stages_and_one_three_image_output(self):
        graph = build_workflow("current.png", "white/test", background_mode="white")
        self.assertEqual(len(graph["nodes"]), 8)
        self.assertEqual(len(graph["edges"]), 19)
        splitter = graph["nodes"][3]
        self.assertEqual(next(i["value"] for i in splitter["data"]["inputs"]
                              if i["name"] == "background_mode"), "white")
        pe_prompt = next(i["value"] for i in graph["nodes"][2]["data"]["inputs"]
                         if i["name"] == "user_prompt")
        self.assertIn("白底", pe_prompt)
        self.assertIn("白底", graph["nodes"][-1]["data"]["title"])
        self.assertNotIn("透明PNG", graph["nodes"][-1]["data"]["title"])


class NativeWhiteBackgroundOutput(unittest.TestCase):
    setUp = native_helpers.NativeVisiblePreview.setUp

    def test_white_output_is_saved_as_three_rgb_pngs_with_one_native_preview_each(self):
        sources = [white_image(i) for i in range(3)]
        result = self.nodes.BAMidFarVisibleThreePNG().run(
            *[native_helpers.Tensor(source) for source in sources], "#FFFFFF",
            "white/one", "white/two", "white/three")
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 4)
        paths = list(Path(self.directory.name).rglob("*.png"))
        self.assertEqual(len(paths), 3)
        for path in paths:
            with Image.open(path) as output:
                self.assertEqual(output.mode, "RGB")
                self.assertEqual(output.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(json.loads(result[-1])["duplicate_check"], "passed")


if __name__ == "__main__":
    unittest.main()
