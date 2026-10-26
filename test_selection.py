import copy
import importlib.util
import json
import unittest

import cv2
import numpy as np

if importlib.util.find_spec("selection"):
    import selection
else:
    selection = None


def target(identifier, layer="midground", score=80, unit=None, **extra):
    return {"id": identifier, "unit_id": unit or identifier, "name": "完整连体岩体",
            "kind": "rock", "layer": layer, "location": "参考图中部",
            "appearance": "平帽主峰与侧柱，共享岩台整体保留",
            "occlusion": "岩脚略被沙地遮住", "bbox": [.2, .1, .7, .8],
            "asset_unit": "connected_formation", "connected_parts": ["主峰", "侧柱", "共享岩台"],
            "selection_reason": "轮廓和体量适合中远景", "priority": score,
            "is_single": True, **extra}


def text(values):
    return json.dumps({"style": "原图暖橙卡通岩石与淡紫远景", "objects": values}, ensure_ascii=False)


def node(kind, identifier, inputs=()):
    return {"id": identifier, "position": {"x": 0, "y": 0}, "data": {
        "type": kind, "inputs": [{"name": n, "value": v} for n, v in inputs],
        "outputSlots": [], "extra": {"model": "existing-config"}}}


def seed():
    return {"name": "ba-clipboard", "version": 1, "nodes": [
        node("LoadImage", "ref", [("image", "old.png")]),
        node("BAMidFarSelectedAsset", "prompt", [("plan", ""), ("filename_prefix", "test")]),
        node("BALLMImage", "gen"), node("BAMidFarSelectedMatte", "matte"),
        node("SaveImage", "save"), node("SaveImage", "raw", [("filename_prefix", "test/raw")]),
        node("SaveText|pysssss", "qa", [("file", "test-quality.json")]),
    ], "edges": [
        {"id": "e0", "source": "ref", "target": "prompt", "targetHandle": "reference"},
        {"id": "e1", "source": "prompt", "target": "gen", "targetHandle": "image"},
        {"id": "e2", "source": "prompt", "target": "gen", "targetHandle": "prompt"},
        {"id": "e3", "source": "gen", "target": "matte"},
        {"id": "e4", "source": "matte", "target": "save"},
        {"id": "e5", "source": "gen", "target": "raw"},
        {"id": "e6", "source": "matte", "target": "qa"},
    ]}


class Availability(unittest.TestCase):
    def test_selection_workflow_support_is_available(self):
        self.assertIsNotNone(selection, "Generic connected-unit selection pipeline is not implemented yet.")


@unittest.skipIf(selection is None, "New selection API not available yet")
class SelectionTests(unittest.TestCase):
    def test_ranking_covers_both_distance_layers_without_first_three_truncation(self):
        plan = selection.parse_selection(text([target("poor", score=1), target("mid2", score=80),
                                              target("mid1", score=95), target("far", "background", 90)]), 3)
        self.assertEqual([t["id"] for t in plan["objects"]], ["mid1", "mid2", "far"])
        self.assertEqual(plan["truncated"], 1)

    def test_two_assets_prefer_one_mid_one_far(self):
        plan = selection.parse_selection(text([target("mid1", score=99), target("mid2", score=98),
                                              target("far", "background", 50)]), 2)
        self.assertEqual([o["id"] for o in plan["objects"]], ["mid1", "far"])

    def test_connected_parts_with_same_unit_are_merged_before_selection(self):
        a = target("cap", unit="whole", bbox=[.3, .1, .5, .5], connected_parts=["主峰"])
        b = target("base", unit="whole", bbox=[.1, .4, .8, .9], connected_parts=["岩台", "侧柱"])
        plan = selection.parse_selection(text([a, b, target("far", "background")]))
        whole = plan["objects"][0]
        self.assertEqual(whole["bbox"], [.1, .1, .8, .9])
        self.assertEqual(whole["connected_parts"], ["主峰", "岩台", "侧柱"])
        self.assertEqual(len(plan["objects"]), 2)

    def test_overlapping_independent_units_are_not_automatically_joined(self):
        plan = selection.parse_selection(text([target("first"), target("second")]))
        self.assertEqual(len(plan["objects"]), 2)

    def test_grass_tuft_and_marked_unsuitable_candidates_are_excluded(self):
        plan = selection.parse_selection(text([target("grass", kind="hill", name="独立草堆"),
                                              target("cut", boundary_cut=True),
                                              target("bad", suitable=False), target("good")]))
        self.assertEqual([o["id"] for o in plan["objects"]], ["good"])
        self.assertEqual(len(plan["rejected_objects"]), 3)
        self.assertIn("不足2", plan["warning"])

    def test_empty_invalid_box_and_bad_limit_are_rejected(self):
        for values in [[], [target("bad", bbox=[0, .2, 2, .8])],
                       [target("bad", bbox=[.5, .2, .3, .8])],
                       [target("x"), target("x")], [target("bad", priority=float("nan"))]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                selection.parse_selection(text(values))
        for limit in [0, 1, 4, True, 2.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                selection.parse_selection(text([target("valid")]), limit)

    def test_crop_uses_current_bbox_pixels_and_does_not_rescale(self):
        image = np.arange(1*100*200*3, dtype=np.float32).reshape(1, 100, 200, 3)/60000
        crop, info = selection.crop_reference(image, [.25, .2, .75, .8], 0)
        self.assertEqual(crop.shape, (1, 60, 100, 3))
        np.testing.assert_array_equal(crop, image[:, 20:80, 50:150])
        self.assertEqual(info["source_bbox_pixels"], [50, 20, 150, 80])

    def test_crop_context_is_bounded_by_source_image(self):
        image = np.zeros((1, 100, 200, 3), np.float32)
        crop, _ = selection.crop_reference(image, [0, 0, .4, .5], .1)
        self.assertEqual(crop.shape, (1, 55, 88, 3))
        with self.assertRaises(ValueError):
            selection.crop_reference(np.zeros((2, 10, 10, 3)), [0, 0, 1, 1], 0)

    def test_extraction_prompt_keeps_shared_base_and_never_designs_variants(self):
        plan = selection.parse_selection(text([target("one")]))
        prompt = selection.selected_prompt(plan, plan["objects"][0], "#00FF00")
        for item in ["主峰", "侧柱", "共享岩台", "连体", "局部", "原图", "15%", "#00FF00", "草堆"]:
            self.assertIn(item, prompt)
        self.assertNotIn("其他相连部分全部移除", prompt)

    def test_new_reference_and_per_unit_carriers_are_propagated(self):
        plan = selection.parse_selection(text([target("main"), target("far", "background")]))
        template = seed()
        unchanged = copy.deepcopy(template)
        graph = selection.expand_selection(plan, template, "new-reference.png")
        self.assertEqual(template, unchanged)
        self.assertEqual(len(graph["nodes"]), 13)
        self.assertEqual(len(graph["edges"]), 14)
        ids = {n["id"] for n in graph["nodes"]}
        self.assertEqual(len(ids), 13)
        self.assertTrue(all(e["source"] in ids and e["target"] in ids for e in graph["edges"]))
        self.assertEqual(graph["nodes"][0]["data"]["inputs"][0]["value"], "new-reference.png")
        self.assertEqual(sum(n["data"]["type"] == "BALLMImage" for n in graph["nodes"]), 2)
        for n in graph["nodes"]:
            if n["data"]["type"] == "BAMidFarSelectedAsset":
                p = json.loads(next(i["value"] for i in n["data"]["inputs"] if i["name"] == "plan"))
                self.assertEqual(len(p["objects"]), 1)
        files = [i["value"] for n in graph["nodes"] for i in n["data"]["inputs"] if i["name"] == "file"]
        self.assertEqual(len(set(files)), 2)

    def test_observed_tinted_green_screen_removes_border_and_keeps_alpha(self):
        foreground = np.array([.79, .41, .23], np.float32)
        key = np.array([.065, .94, .05], np.float32)
        alpha = np.zeros((120, 140), np.float32)
        alpha[30:90, 40:100] = 1
        alpha = cv2.GaussianBlur(alpha, (7, 7), .8)
        rgb = foreground*alpha[..., None]+key*(1-alpha[..., None])
        rgba, quality = selection.selected_matte(rgb[None], "#00FF00", .06, 8, True)
        self.assertEqual(quality["meaningful_components"], 1)
        self.assertFalse(quality["touches_border"])
        self.assertGreater(np.count_nonzero((rgba[..., 3] > 0) & (rgba[..., 3] < 1)), 100)
        self.assertEqual(float(rgba[:, 0, :, 3].max()), 0)
        center = rgba[0, rgba.shape[1]//2, rgba.shape[2]//2]
        np.testing.assert_allclose(center[:3], foreground, atol=.01)

    def test_magenta_screen_handles_green_hill_without_erasing_grass_surface(self):
        rgb = np.broadcast_to([.94, .05, .93], (1, 100, 120, 3)).copy().astype(np.float32)
        rgb[:, 25:75, 30:90] = [.12, .7, .16]
        rgba, quality = selection.selected_matte(rgb, "#FF00FF", .06, 8, True)
        self.assertEqual(quality["meaningful_components"], 1)
        self.assertEqual(float(rgba[..., 3].max()), 1)
        self.assertGreater(float(rgba[0, 30, 30, 1]), .6)

    def test_missing_key_and_multiple_objects_fail_in_strict_mode(self):
        with self.assertRaises(ValueError):
            selection.selected_matte(np.full((1, 100, 100, 3), .5), "#00FF00", .06, 8, True)
        rgb = np.broadcast_to([0, 1, 0], (1, 100, 140, 3)).copy().astype(np.float32)
        rgb[:, 25:75, 10:40] = [.8, .4, .2]
        rgb[:, 25:75, 95:125] = [.8, .4, .2]
        with self.assertRaisesRegex(ValueError, "Multiple"):
            selection.selected_matte(rgb, "#00FF00", .06, 8, True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
