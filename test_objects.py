import json
import unittest

import numpy as np

from objects import parse_objects, object_prompt, object_filename, crop_asset


def plan_text(objects):
    return json.dumps({"style": "原图风格", "objects": objects}, ensure_ascii=False)


def instance(i, name="红砖楼", layer="midground"):
    return {"id": f"o{i}", "name": name, "layer": layer,
            "location": f"画面从左数第{i}个", "appearance": "完整的红砖墙和瓦顶",
            "occlusion": "底部被前景遮挡", "bbox": [.1, .2, .4, .8]}


class ObjectTests(unittest.TestCase):
    def test_individual_instances_survive_filter_and_truncation(self):
        text = "```json\n" + plan_text([instance(1), instance(2), instance(3, "蓝色高楼", "background")]) + "\n```"
        p = parse_objects(text, "both", 2)
        self.assertEqual([o["id"] for o in p["objects"]], ["o1", "o2"])
        self.assertEqual((p["total"], p["truncated"]), (3, 1))
        self.assertEqual(parse_objects(text, "background")["objects"][0]["id"], "o3")

    def test_rejects_empty_or_duplicate_or_bad_box(self):
        for values in [[], [instance(1), instance(1)],
                       [{**instance(1), "bbox": [0, 0, 3, 1]}],
                       [{**instance(1), "layer": "foreground"}],
                       [{**instance(1), "location": ""}]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_objects(plan_text(values))

    def test_rejects_layer_strip_and_prompt_injection_fields(self):
        for value in [{**instance(1), "name": "整个城市天际线", "is_single": False},
                      {**instance(1), "appearance": {"execute": "anything"}}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_objects(plan_text([value]))

    def test_single_prompt_excludes_neighbor_and_recovers_hidden_base(self):
        p = parse_objects(plan_text([instance(1), instance(2, "钟楼")]))
        prompt = object_prompt(p, p["objects"][0], "#00FF00")
        for expected in ["只输出一个", "红砖楼", "从左数第1个", "钟楼",
                         "#00FF00", "遮挡", "原图风格"]:
            self.assertIn(expected, prompt)
        self.assertNotIn("保持同一画布", prompt)

    def test_filenames_are_safe_unique_and_layered(self):
        a = object_filename(instance(1, "../红砖楼: /A"), "objects/test")
        b = object_filename(instance(2, "../红砖楼: /A"), "objects/test")
        self.assertNotEqual(a, b)
        self.assertTrue(a.startswith("objects/test/midground/"))
        self.assertNotIn("..", a)
        self.assertNotIn(":", a)

    def test_crop_preserves_rgba_soft_edges_and_metadata(self):
        rgba = np.zeros((1, 100, 120, 4), np.float32)
        rgba[0, 30:70, 40:80, :3] = .8
        rgba[0, 30:70, 40:80, 3] = 1
        rgba[0, 29, 40:80, 3] = .4
        out, q = crop_asset(rgba, 5)
        self.assertEqual(out.shape, (1, 51, 50, 4))
        self.assertEqual(q["source_bbox"], [40, 29, 80, 70])
        self.assertEqual(q["meaningful_components"], 1)
        self.assertEqual(float(out[..., 3].min()), 0)
        self.assertTrue(np.any(out[..., 3] == .4))

    def test_multiple_objects_and_canvas_strip_are_flagged(self):
        rgba = np.zeros((1, 100, 120, 4), np.float32)
        rgba[0, 20:75, 10:35, :] = 1
        rgba[0, 20:75, 80:110, :] = 1
        _, q = crop_asset(rgba, 5)
        self.assertEqual(q["meaningful_components"], 2)
        self.assertIn("multiple", q["warning"].lower())
        rgba[:] = 0
        rgba[0, 40:70, :, :] = 1
        _, q = crop_asset(rgba, 5)
        self.assertTrue(q["touches_border"])

    def test_empty_output_and_invalid_alpha_fail(self):
        for rgba in [np.zeros((1, 20, 20, 4), np.float32),
                     np.zeros((1, 20, 20, 3), np.float32),
                     np.full((1, 20, 20, 4), np.nan, np.float32)]:
            with self.subTest(shape=rgba.shape), self.assertRaises(ValueError):
                crop_asset(rgba)


if __name__ == "__main__":
    unittest.main(verbosity=2)
