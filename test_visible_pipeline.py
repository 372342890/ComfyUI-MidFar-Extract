import importlib.util
import json
import unittest
import numpy as np

if importlib.util.find_spec("visible_pipeline"):
    from visible_pipeline import prepare_asset
else:
    prepare_asset = None


def recognition():
    return json.dumps({"style": "浅色卡通雪山", "objects": [
        {"id": "c"+str(i), "unit_id": "u"+str(i), "name": "山体"+str(i),
         "kind": "mountain", "layer": "midground" if i == 1 else "background",
         "location": "当前原图第"+str(i)+"处", "appearance": "一座完整雪山",
         "occlusion": "仅短山脚遮挡", "selection_reason": "可复用完整单元",
         "bbox": [.1+(i-1)*.25, .2, .3+(i-1)*.25, .8],
         "asset_unit": "independent", "connected_parts": ["岩基"], "is_single": True,
         "boundary_cut": False, "suitable": True, "priority": 90-i,
         "excluded_objects": ["树林"]} for i in range(1, 4)]})


def pe():
    return json.dumps({"objects": [{"id": "c"+str(i), "prompt": "PE提取提示词"+str(i)}
                                 for i in range(1, 4)]})


class VisiblePipeline(unittest.TestCase):
    def call(self, plan=None, prompts=None, index=1):
        self.assertIsNotNone(prepare_asset, "Visible PE pipeline is missing.")
        source = np.zeros((1, 100, 200, 3), np.float32)
        source[..., 2] = .8
        return prepare_asset(recognition() if plan is None else plan,
                             pe() if prompts is None else prompts, source,
                             index, "visible/asset")

    def test_pe_text_and_current_crop_are_preserved_and_visible(self):
        crop, prompt, key, prefix, record = self.call()
        record = json.loads(record)
        self.assertIn("PE提取提示词1", prompt)
        self.assertEqual(record["pe_prompt"], "PE提取提示词1")
        self.assertEqual(record["asset_id"], "c1")
        self.assertEqual(record["source_size"], [200, 100])
        self.assertTrue(40 <= crop.shape[2] <= 44)
        self.assertTrue(60 <= crop.shape[1] <= 64)
        self.assertIn(key, ("#00FF00", "#FF00FF", "#0000FF"))
        self.assertIn("c1", prefix)

    def test_three_branches_use_three_distinct_actual_ids(self):
        records = [json.loads(self.call(index=i)[4]) for i in range(1, 4)]
        self.assertEqual([r["asset_id"] for r in records], ["c1", "c2", "c3"])

    def test_pe_cannot_replace_recognized_identity(self):
        value = json.loads(pe())
        value["objects"][0]["id"] = "invented"
        with self.assertRaises(ValueError):
            self.call(prompts=json.dumps(value))

    def test_missing_pe_prompt_is_not_silently_generated(self):
        value = json.loads(pe())
        value["objects"][0]["prompt"] = ""
        with self.assertRaises(ValueError):
            self.call(prompts=json.dumps(value))

    def test_insufficient_source_units_are_explicit(self):
        value = json.loads(recognition())
        value["objects"].pop()
        with self.assertRaises(ValueError):
            self.call(plan=json.dumps(value))


if __name__ == "__main__":
    unittest.main()
