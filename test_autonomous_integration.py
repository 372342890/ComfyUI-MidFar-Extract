import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent
NEW_TYPES = {
    "BAMidFarCandidateReview", "BAMidFarThreePlans",
    "BAMidFarAutonomousReference", "BAMidFarAssetInspect",
    "BAMidFarRepairPrompt", "BAMidFarChooseAsset",
    "BAMidFarTargetGuide",
    "BAMidFarCompleteFoot",
    "BAMidFarNativeImageFile",
}


class IntegrationAvailability(unittest.TestCase):
    def test_new_nodes_registered_with_scalar_three_plans(self):
        tree = ast.parse((ROOT / "__init__.py").read_text(encoding="utf-8"))
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and target.id == "NODE_CLASS_MAPPINGS":
                        names.update(k.value for k in n.value.keys)
        self.assertTrue(NEW_TYPES <= names, f"Missing {NEW_TYPES - names}")
        self.assertTrue((ROOT / "autonomous_nodes.py").is_file())

    def test_complete_public_workflow_exists(self):
        self.assertTrue((ROOT / "midfar-three-autonomous.byteartist.json").is_file())


@unittest.skipUnless((ROOT / "midfar-three-autonomous.byteartist.json").is_file(),
                     "Workflow not generated yet")
class GraphContract(unittest.TestCase):
    def setUp(self):
        self.graph = json.loads((ROOT / "midfar-three-autonomous.byteartist.json")
                                .read_text(encoding="utf-8"))
        self.nodes = {n["id"]: n for n in self.graph["nodes"]}
        self.edges = self.graph["edges"]

    def of_type(self, kind):
        return [n for n in self.nodes.values() if n["data"]["type"] == kind]

    def ancestors(self, identifier):
        visited, pending = set(), [identifier]
        while pending:
            child = pending.pop()
            for e in self.edges:
                if e["target"] == child and e["source"] not in visited:
                    visited.add(e["source"])
                    pending.append(e["source"])
        return visited

    def test_three_independent_accepted_rgba_outputs(self):
        saves = self.of_type("SaveImage")
        self.assertEqual(len(saves), 3)
        self.assertEqual(len(self.of_type("BAMidFarChooseAsset")), 3)
        self.assertEqual(len(self.of_type("BAMidFarThreePlans")), 1)
        sources = [n for n in self.of_type("LoadImage") if any(
            i["name"] == "image" and "value" in i for i in n["data"]["inputs"])]
        self.assertEqual(len(sources), 1)
        source = sources[0]["id"]
        providers = []
        for save in saves:
            self.assertIn(source, self.ancestors(save["id"]))
            wires = [e for e in self.edges
                     if e["target"] == save["id"] and e["targetHandle"] == "images"]
            self.assertEqual(len(wires), 1)
            wire = wires[0]
            self.assertEqual(wire["sourceHandle"], "rgba")
            self.assertEqual(self.nodes[wire["source"]]["data"]["type"], "BAMidFarChooseAsset")
            providers.append(wire["source"])
        self.assertEqual(len(set(providers)), 3)
        self.assertFalse(any("Expand" in n["data"]["type"] for n in self.nodes.values()))

    def test_all_nine_image_calls_use_pro_png_and_feedback_dependencies(self):
        carriers = self.of_type("BALLMImage")
        self.assertEqual(len(carriers), 9)
        self.assertEqual(len(self.of_type("BAMidFarAssetInspect")), 9)
        repairs = self.of_type("BAMidFarRepairPrompt")
        self.assertEqual(len(repairs), 6)
        for node in carriers:
            extra = node["data"]["extra"]
            self.assertEqual(extra["model"]["key"], "Doubao-Seedream-5.0-pro")
            self.assertEqual(extra["modelConfig"]["output_format"]["value"], "png")
            self.assertEqual(extra["modelConfig"]["size"]["value"], "2048x2048")
            self.assertFalse(extra["modelConfig"]["watermark"]["value"])
        for repair in repairs:
            ancestors = [self.nodes[i]["data"]["type"] for i in self.ancestors(repair["id"])]
            self.assertIn("BALLMImage", ancestors)
            self.assertIn("BALLMText", ancestors)
            self.assertIn("BAMidFarAssetInspect", ancestors)
        for choose in self.of_type("BAMidFarChooseAsset"):
            incoming = {e["targetHandle"] for e in self.edges if e["target"] == choose["id"]}
            self.assertEqual(incoming, {f"{k}_{i}" for k in ("rgba", "numeric", "visual")
                                        for i in range(1, 4)} | {"diagnostic_prefix"})

    def test_graph_types_are_valid_acyclic_and_free_of_private_account_data(self):
        pending = set(self.nodes)
        resolved = set()
        for e in self.edges:
            a, b = self.nodes[e["source"]], self.nodes[e["target"]]
            self.assertTrue(any(o["name"] == e["sourceHandle"] for o in a["data"]["outputSlots"]))
            self.assertTrue(any(i["name"] == e["targetHandle"] for i in b["data"]["inputs"]))
        while pending:
            ready = {i for i in pending if all(
                e["source"] in resolved for e in self.edges if e["target"] == i)}
            self.assertTrue(ready, "Cycle or dangling edge")
            resolved.update(ready)
            pending.difference_update(ready)
        raw = json.dumps(self.graph).lower()
        for token in ("client_id", "authorization", "api_key", "bench-", "workspace_id",
                      "user_id", "localhost", "127.0.0.1"):
            self.assertNotIn(token, raw)

    def test_every_generator_uses_current_editable_source_guide_without_frozen_bad_geometry(self):
        self.assertEqual(len(self.of_type("BAMidFarCompleteFoot")), 0)
        self.assertEqual(len(self.of_type("BAMidFarLocalizationView")), 3)
        self.assertEqual(len(self.of_type("BAMidFarTargetGuide")), 3)
        for generator in self.of_type("BALLMImage"):
            images = [e for e in self.edges
                      if e["target"] == generator["id"] and e["targetHandle"] == "image"]
            is_repair = any(self.nodes[e["source"]]["data"]["type"] == "BAMidFarRepairPrompt"
                            for e in self.edges if e["target"] == generator["id"])
            self.assertEqual(len(images), 2 if is_repair else 1)
            loader = self.nodes[images[0]["source"]]
            self.assertEqual(loader["data"]["type"], "LoadImage")
            self.assertEqual(images[0]["sourceHandle"], "IMAGE")
            handoff_id = next(e["source"] for e in self.edges if e["target"] == loader["id"])
            self.assertEqual(self.nodes[handoff_id]["data"]["type"], "BAMidFarNativeImageFile")
            origin = next(e for e in self.edges if e["target"] == handoff_id)
            self.assertEqual(self.nodes[origin["source"]]["data"]["type"], "BAMidFarTargetGuide")
            self.assertEqual(origin["sourceHandle"], "guide_image")
            self.assertTrue(any(self.nodes[a]["data"]["type"] == "BAMidFarAutonomousReference"
                                for a in self.ancestors(origin["source"])))
            if is_repair:
                prior_loader = images[1]["source"]
                prior_handoff = next(e["source"] for e in self.edges if e["target"] == prior_loader)
                prior_origin = next(e for e in self.edges if e["target"] == prior_handoff)
                self.assertEqual(self.nodes[prior_origin["source"]]["data"]["type"], "BALLMImage")
                repair = next(e["source"] for e in self.edges if e["target"] == generator["id"]
                              and e["targetHandle"] == "prompt")
                inspection = next(e["source"] for e in self.edges if e["target"] == repair
                                  and e["targetHandle"] == "numeric_report")
                inspected_generator = next(e["source"] for e in self.edges if e["target"] == inspection
                                           and e["targetHandle"] == "image")
                self.assertEqual(prior_origin["source"], inspected_generator,
                                 "Repair must see the very candidate that its feedback reviewed.")
        for inspector in self.of_type("BAMidFarAssetInspect"):
            image = next(e for e in self.edges
                         if e["target"] == inspector["id"] and e["targetHandle"] == "image")
            self.assertEqual(self.nodes[image["source"]]["data"]["type"], "BALLMImage")
            self.assertEqual(image["sourceHandle"], "image")
            guide = next(e for e in self.edges
                         if e["target"] == inspector["id"] and e["targetHandle"] == "target_guide")
            self.assertEqual(self.nodes[guide["source"]]["data"]["type"], "BAMidFarTargetGuide")
            self.assertTrue(any(e["target"] == inspector["id"] and e["targetHandle"] == "diagnostic_prefix"
                                for e in self.edges), "Rejected raw candidates must remain inspectable.")

    def test_every_model_reference_uses_standard_native_loading(self):
        carriers = {n["id"] for n in self.of_type("BALLMImage")+self.of_type("BALLMText")}
        for edge in self.edges:
            if edge["target"] in carriers and edge["targetHandle"] == "image":
                self.assertEqual(self.nodes[edge["source"]]["data"]["type"], "LoadImage")


if __name__ == "__main__":
    unittest.main(verbosity=2)
