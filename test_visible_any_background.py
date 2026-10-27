"""Background color must never repaint or block the three extraction outputs."""
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


def colored_image(index, background):
    pixels = image(index)
    mask = np.all(pixels == [1, 0, 1], axis=-1)
    pixels[mask] = background
    return pixels


class AnyBackgroundOutput(unittest.TestCase):
    def test_white_black_green_and_arbitrary_background_pixels_are_preserved(self):
        for background in ([1, 1, 1], [0, 0, 0], [0, 1, 0], [.4, .2, .7]):
            with self.subTest(background=background):
                sources = [colored_image(i, background) for i in range(3)]
                outputs, records = process_three(sources, "any")
                for source, output, record in zip(sources, outputs, records):
                    np.testing.assert_array_equal(output, source)
                    self.assertTrue(record["source_pixels_preserved"])
                    self.assertFalse(record["alpha_required"])
                    self.assertEqual(record["background"], "any")

    def test_transparent_pixels_and_alpha_are_preserved(self):
        sources = []
        for i in range(3):
            rgb = image(i)
            alpha = (~np.all(rgb == [1, 0, 1], axis=-1)).astype(np.float32)
            sources.append(np.concatenate((rgb, alpha[..., None]), axis=-1))
        outputs, _ = process_three(sources, "any")
        for source, output in zip(sources, outputs):
            np.testing.assert_array_equal(output, source)

    def test_background_variation_and_border_content_do_not_trigger_matte_checks(self):
        sources = [colored_image(i, [.96, .97, .98]) for i in range(3)]
        sources[0][:, 100:150, :120] = [.2, .5, .8]
        sources[1][:, :50] = [.88, .92, .96]
        outputs, _ = process_three(sources, "any")
        for source, output in zip(sources, outputs):
            np.testing.assert_array_equal(output, source)

    def test_different_backgrounds_cannot_disguise_the_same_subject(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            process_three([colored_image(0, [1, 1, 1]), colored_image(0, [0, 0, 0]),
                           colored_image(2, [0, 1, 0])], "any")

    def test_three_actual_images_are_still_required(self):
        with self.assertRaisesRegex(ValueError, "三"):
            process_three([image(0), image(1)], "any")

    def test_pe_has_no_fixed_background_requirement_and_retains_exclusions(self):
        pe = {"objects": [{"id": f"asset{i}", "prompt": f"独立提取{i}，保留原外观。"}
                          for i in (1, 2, 3)]}
        result = split_prompts(json.dumps(recognition()), json.dumps(pe), image(),
                               "any/test", background_mode="any")
        self.assertEqual(result[3], "any")
        self.assertIn("底色不限", result[0])
        self.assertIn("雪山2", result[0])
        self.assertIn("雪山3", result[0])
        self.assertNotIn("全部空白区域必须使用均匀纯色", result[0])
        self.assertNotIn("#FFFFFF", result[0])
        self.assertEqual(json.loads(result[-1])["background_mode"], "any")

    def test_workflow_keeps_independent_recognition_pe_and_three_visible_branches(self):
        graph = build_workflow("current.png", "any/test", background_mode="any")
        self.assertEqual(len(graph["nodes"]), 8)
        self.assertEqual(len(graph["edges"]), 19)
        self.assertEqual(next(i["value"] for i in graph["nodes"][3]["data"]["inputs"]
                              if i["name"] == "background_mode"), "any")
        for node in graph["nodes"][2:3] + graph["nodes"][4:7]:
            prompt = next(i["value"] for i in node["data"]["inputs"]
                          if i["name"] == "user_prompt")
            self.assertIn("底色不限", prompt)
            self.assertNotIn("{KEY_COLOR}", prompt)
            self.assertNotIn("指定纯色键背景", prompt)
        self.assertIn("三张提取", graph["nodes"][-1]["data"]["title"])


class NativeAnyBackgroundOutput(unittest.TestCase):
    setUp = native_helpers.NativeVisiblePreview.setUp

    def test_native_output_saves_each_image_once_without_background_repainting(self):
        sources = [colored_image(0, [1, 1, 1]), colored_image(1, [0, 0, 0]),
                   colored_image(2, [0, 1, 0])]
        result = self.nodes.BAMidFarVisibleThreePNG().run(
            *[native_helpers.Tensor(source) for source in sources], "any",
            "any/one", "any/two", "any/three")
        self.assertEqual(len(result), 4)
        paths = list(Path(self.directory.name).rglob("*.png"))
        self.assertEqual(len(paths), 3)
        for name, source in zip(("one", "two", "three"), sources):
            with Image.open(Path(self.directory.name) / "any" / (name + ".png")) as output:
                np.testing.assert_array_equal(np.asarray(output),
                                              np.round(source[0] * 255).astype(np.uint8))
        self.assertEqual(json.loads(result[-1])["duplicate_check"], "passed")


if __name__ == "__main__":
    unittest.main()
