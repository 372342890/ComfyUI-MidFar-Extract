import json
import unittest

from terrain import parse_assets, asset_prompt, expand_terrain_workflow


def asset(i, kind="hill", name="矮圆草丘"):
    return {"id": f"a{i}", "name": name, "kind": kind,
            "shape": "扁宽圆润的独立体块，高宽比约0.3，完整底部",
            "surface": "鲜黄绿色短草覆盖自身表面",
            "variation": "一侧稍高，轮廓自然不对称", "is_single": True}


def text(assets):
    return json.dumps({"style": "参考图的明亮卡通3D，柔和暖光，简化块面",
                       "assets": assets}, ensure_ascii=False)


class TerrainTests(unittest.TestCase):
    def test_all_natural_asset_types_are_allowed(self):
        values = [asset(1), asset(2, "grass_tuft", "独立草堆"),
                  asset(3, "rock", "风蚀岩石"), asset(4, "mountain", "独立山体")]
        plan = parse_assets("```json\n" + text(values) + "\n```", 3)
        self.assertEqual([a["kind"] for a in plan["assets"]], ["hill", "grass_tuft", "rock"])
        self.assertEqual((plan["total"], plan["truncated"]), (4, 1))

    def test_duplicate_empty_bad_kind_and_non_single_fail(self):
        for values in [[], [asset(1), asset(1)], [asset(1, "city")],
                       [{**asset(1), "is_single": False}],
                       [{**asset(1), "shape": ""}],
                       [{**asset(1), "surface": {"bad": True}}]]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_assets(text(values))

    def test_group_landscape_and_ground_strip_fail(self):
        for name in ["连续山脉", "整片草地", "岩石群", "地面托盘", "mountain range"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                parse_assets(text([asset(1, name=name)]))

    def test_unsafe_identifier_or_bad_limit_fail(self):
        for entry in [{**asset(1), "id": "../a"}, {**asset(1), "id": ""}]:
            with self.assertRaises(ValueError):
                parse_assets(text([entry]))
        for limit in [0, 33, True, 1.5]:
            with self.assertRaises(ValueError):
                parse_assets(text([asset(1)]), limit)

    def test_hill_keeps_own_grass_volume_and_shape_reference_priority(self):
        plan = parse_assets(text([asset(1)]))
        prompt = asset_prompt(plan, plan["assets"][0], "#FF00FF")
        for term in ["图1", "图2", "高宽比约0.3", "自身", "草面", "厚度", "15%",
                     "#FF00FF", "明亮卡通3D", "一侧稍高"]:
            self.assertIn(term, prompt)
        self.assertNotIn("不附带草地", prompt)

    def test_tuft_preserves_roots_and_other_assets_are_not_a_contact_sheet(self):
        plan = parse_assets(text([asset(1, "grass_tuft", "草堆")]))
        prompt = asset_prompt(plan, plan["assets"][0], "#0000FF")
        self.assertIn("根部", prompt)
        self.assertIn("不输出四视图", prompt)
        self.assertIn("不带天空", prompt)
        with self.assertRaises(ValueError):
            asset_prompt(plan, plan["assets"][0], "#FFFFFF")

    def test_expansion_retains_two_references_and_independent_carriers(self):
        plan = parse_assets(text([asset(1), asset(2, "rock", "单块岩石")]))
        template = {"name": "ba-clipboard", "version": 1, "nodes": [
            {"id": "100", "data": {"type": "LoadImage", "title": "style",
                                  "inputs": [{"name": "image", "value": "style.png"}]}},
            {"id": "110", "data": {"type": "LoadImage", "title": "shape",
                                  "inputs": [{"name": "image", "value": "shape.png"}]}},
            {"id": "102", "data": {"type": "BATerrainAssetPrompt", "inputs": [
                {"name": "plan", "format": "slot"}, {"name": "filename_prefix", "value": "test"}]}},
            {"id": "103", "data": {"type": "BALLMImage", "inputs": [], "extra": {"model": "actual"}}},
            {"id": "104", "data": {"type": "BAMidFarObjectMatte", "inputs": []}},
            {"id": "105", "data": {"type": "SaveImage", "inputs": []}},
        ], "edges": [
            {"source": "100", "target": "102", "targetHandle": "style_reference"},
            {"source": "110", "target": "102", "targetHandle": "shape_reference"},
            {"source": "102", "target": "103"},
            {"source": "103", "target": "104"},
            {"source": "104", "target": "105"}]}
        out = expand_terrain_workflow(plan, template, "new-style.png", "new-shape.png")
        self.assertEqual(len(out["nodes"]), 10)
        self.assertEqual(len(out["edges"]), 10)
        self.assertEqual(len({n["id"] for n in out["nodes"]}), 10)
        self.assertEqual([n["data"]["inputs"][0]["value"] for n in out["nodes"][:2]],
                         ["new-style.png", "new-shape.png"])
        self.assertEqual(len([n for n in out["nodes"] if n["data"]["type"] == "BALLMImage"]), 2)
        for node in out["nodes"]:
            if node["data"]["type"] == "BATerrainAssetPrompt":
                pin = next(i for i in node["data"]["inputs"] if i["name"] == "plan")
                self.assertNotIn("format", pin)
                self.assertEqual(len(json.loads(pin["value"])["assets"]), 1)
        self.assertEqual(template["nodes"][0]["data"]["inputs"][0]["value"], "style.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)
