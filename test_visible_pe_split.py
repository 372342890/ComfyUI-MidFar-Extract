import importlib.util
import json
import unittest
import numpy as np

if importlib.util.find_spec("visible_pipeline"):
    from visible_pipeline import __dict__ as api
    split_prompts = api.get("split_prompts")
else:
    split_prompts = None


def recognition():
    return {"style": "浅蓝卡通雪山", "objects": [
        {"id": "asset"+str(i), "unit_id": "unit"+str(i), "name": "雪山"+str(i),
         "kind": "mountain", "layer": "background", "location": "原图中第"+str(i)+"处",
         "appearance": "一个完整独立雪山", "occlusion": "短山脚被树林遮挡",
         "connected_parts": ["岩基"], "excluded_objects": ["树林"],
         "is_single": True, "boundary_cut": False, "suitable": True} for i in range(1, 4)]}


class VisiblePESplit(unittest.TestCase):
    def call(self, rec=None, pe=None):
        self.assertIsNotNone(split_prompts, "Direct PE splitter is missing.")
        raw = recognition() if rec is None else rec
        prompts = {"objects": [{"id": "asset"+str(i), "prompt": "独立PE"+str(i)+" {KEY_COLOR}"}
                               for i in range(1, 4)]} if pe is None else pe
        source = np.full((1, 80, 160, 3), [.2, .4, .8], np.float32)
        return split_prompts(json.dumps(raw), json.dumps(prompts), source, "visible/direct")

    def test_original_source_does_not_require_any_bbox(self):
        outputs = self.call()
        self.assertEqual(len(outputs), 8)
        self.assertIn("独立PE1", outputs[0])
        self.assertIn("独立PE2", outputs[1])
        self.assertIn("独立PE3", outputs[2])
        self.assertNotIn("{KEY_COLOR}", outputs[0])
        self.assertEqual(json.loads(outputs[-1])["source_size"], [160, 80])

    def test_unrelated_coordinates_do_not_drive_or_modify_the_current_source(self):
        raw = recognition()
        raw["objects"][0]["bbox"] = [0.395, 60, 528, 230]
        output = self.call(rec=raw)
        self.assertIn("独立PE1", output[0])
        self.assertFalse(json.loads(output[-1])["source_image_modified"])

    def test_pe_cannot_replace_the_three_recognized_ids(self):
        value = {"objects": [{"id": "other"+str(i), "prompt": "提取"} for i in range(1, 4)]}
        with self.assertRaises(ValueError):
            self.call(pe=value)

    def test_target_identity_precedes_pe_and_explicitly_excludes_other_branches(self):
        outputs = self.call()
        self.assertTrue(outputs[0].startswith("本路唯一目标：雪山1"))
        self.assertIn("本路排除的其他目标", outputs[0])
        self.assertIn("雪山2", outputs[0])
        self.assertIn("雪山3", outputs[0])
        self.assertLess(outputs[0].index("本路唯一目标"), outputs[0].index("独立PE1"))

    def test_same_connected_unit_cannot_be_split_to_fill_three(self):
        raw = recognition()
        raw["objects"][1]["unit_id"] = raw["objects"][0]["unit_id"]
        with self.assertRaises(ValueError):
            self.call(rec=raw)

    def test_case_and_space_do_not_disguise_a_repeated_connected_unit(self):
        raw = recognition()
        raw["objects"][1]["unit_id"] = " UNIT1 "
        with self.assertRaisesRegex(ValueError, "重复"):
            self.call(rec=raw)

    def test_new_ids_do_not_disguise_the_same_location_and_appearance(self):
        raw = recognition()
        raw["objects"][1]["location"] = raw["objects"][0]["location"]
        raw["objects"][1]["appearance"] = raw["objects"][0]["appearance"]
        with self.assertRaisesRegex(ValueError, "重复"):
            self.call(rec=raw)

    def test_fresh_runs_share_one_unique_folder_without_overwriting_previous_run(self):
        first, second = self.call(), self.call()
        first_record, second_record = json.loads(first[-1]), json.loads(second[-1])
        self.assertIn("run_id", first_record)
        self.assertIn("run_id", second_record)
        self.assertNotEqual(first_record["run_id"], second_record["run_id"])
        self.assertTrue(all(first_record["run_id"] in prefix for prefix in first[4:7]))
        self.assertTrue(set(first[4:7]).isdisjoint(second[4:7]))

    def test_real_recognition_occlusion_lists_are_preserved(self):
        raw = recognition()
        raw["objects"][0]["occlusion"] = ["下部被岸边针叶林遮挡山脚"]
        raw["objects"][1]["occlusion"] = []
        output = self.call(rec=raw)
        targets = json.loads(output[-1])["recognized_targets"]
        self.assertEqual(targets[0]["occlusion"], ["下部被岸边针叶林遮挡山脚"])
        self.assertEqual(targets[1]["occlusion"], [])

    def test_invalid_occlusion_payload_is_rejected(self):
        raw = recognition()
        raw["objects"][0]["occlusion"] = {"unexpected": "field"}
        with self.assertRaises(ValueError):
            self.call(rec=raw)


if __name__ == "__main__":
    unittest.main()
