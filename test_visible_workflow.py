import importlib.util
import unittest
from pathlib import Path

if importlib.util.find_spec("visible_workflow"):
    from visible_workflow import build_workflow
else:
    build_workflow = None


class VisibleWorkflow(unittest.TestCase):
    def graph(self):
        self.assertIsNotNone(build_workflow, "Visible workflow is missing.")
        return build_workflow("current.png", "visible/asset")

    def test_one_input_two_visible_text_stages_three_native_image_stages(self):
        graph = self.graph()
        types = [n["data"]["type"] for n in graph["nodes"]]
        self.assertEqual(len(types), 8)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("BALLMText"), 2)
        self.assertEqual(types.count("BALLMImage"), 3)
        self.assertEqual(types.count("BAMidFarVisiblePE"), 1)
        self.assertEqual(types.count("BAMidFarVisibleAsset"), 0)
        self.assertEqual(types.count("BAMidFarVisiblePNG"), 0)
        self.assertEqual(types.count("BAMidFarVisibleThreePNG"), 1)
        self.assertFalse(any("3D" in t or "GLB" in t or "Compact" in t or
                             "ReferencePlan" in t for t in types))
        text_nodes = [n for n in graph["nodes"] if n["data"]["type"] == "BALLMText"]
        self.assertIn("识别", text_nodes[0]["data"]["title"])
        self.assertIn("PE", text_nodes[1]["data"]["title"])

    def test_single_output_waits_for_three_distinct_native_branches(self):
        graph = self.graph()
        self.assertEqual(len(graph["edges"]), 19)
        wires = {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"])
                 for e in graph["edges"]}
        self.assertEqual(len(wires), len(graph["edges"]))
        finals = [n for n in graph["nodes"]
                  if n["data"]["type"] == "BAMidFarVisibleThreePNG"]
        self.assertEqual(len(finals), 1)
        sources = {e["source"] for e in graph["edges"]
                   if e["target"] == finals[0]["id"] and e["targetHandle"].startswith("image_")}
        self.assertEqual(len(sources), 3)

    def test_current_image_and_native_text_flow_are_connected(self):
        graph = self.graph()
        lookup = {n["id"]: n for n in graph["nodes"]}
        image = next(n for n in graph["nodes"] if n["data"]["type"] == "LoadImage")
        recognition, pe = [n for n in graph["nodes"] if n["data"]["type"] == "BALLMText"]
        wires = {(e["source"], e["sourceHandle"], e["target"], e["targetHandle"])
                 for e in graph["edges"]}
        self.assertIn((image["id"], "IMAGE", recognition["id"], "image"), wires)
        self.assertIn((image["id"], "IMAGE", pe["id"], "image"), wires)
        self.assertIn((recognition["id"], "text", pe["id"], "text"), wires)
        for edge in graph["edges"]:
            output = next(s for s in lookup[edge["source"]]["data"]["outputSlots"]
                          if s["name"] == edge["sourceHandle"])
            target = next(s for s in lookup[edge["target"]]["data"]["inputs"]
                          if s["name"] == edge["targetHandle"])
            self.assertEqual(output["type"], target["type"])

    def test_every_image_stage_exposes_ui_preview(self):
        graph = self.graph()
        for node in graph["nodes"]:
            if node["data"]["type"] != "LoadImage":
                self.assertTrue(node["data"]["extra"]["output_preview_names"],
                                node["data"]["title"])
        source = Path(__file__).with_name("visible_nodes.py")
        self.assertTrue(source.is_file())
        text = source.read_text(encoding="utf8")
        self.assertIn('"ui"', text)
        self.assertIn('"images"', text)
        self.assertIn('"text"', text)
        self.assertNotIn("invoke(", text)


if __name__ == "__main__":
    unittest.main()
