import importlib.util
import unittest

if importlib.util.find_spec("reference_workflow"):
    from reference_workflow import build_workflow
else:
    build_workflow = None


class ReferenceWorkflow(unittest.TestCase):
    def graph(self):
        self.assertIsNotNone(build_workflow, "Reference workflow builder is missing.")
        return build_workflow("current.png", "reference-test/asset")

    def test_one_current_image_and_nine_real_nodes(self):
        graph = self.graph()
        self.assertEqual(graph["name"], "ba-clipboard")
        self.assertEqual(len(graph["nodes"]), 9)
        uploads = [n for n in graph["nodes"] if n["data"]["type"] == "LoadImage"]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["data"]["inputs"][0]["value"], "current.png")

    def test_three_distinct_extract_and_glb_branches(self):
        graph = self.graph()
        types = [n["data"]["type"] for n in graph["nodes"]]
        self.assertEqual(types.count("BAMidFarReferencePlan"), 1)
        self.assertEqual(types.count("BAMidFarReferenceExtract"), 3)
        self.assertEqual(types.count("BAMidFarReferenceGLB"), 3)
        self.assertNotIn("BAMidFarCompactExtract", types)

    def test_shared_model_reaches_all_three_glb_nodes(self):
        graph = self.graph()
        models = [n for n in graph["nodes"] if n["data"]["type"] == "BAMidFar3DModel"]
        self.assertEqual(len(models), 1)
        exporters = {n["id"] for n in graph["nodes"] if n["data"]["type"] == "BAMidFarReferenceGLB"}
        for handle in ("model", "clip_vision", "vae"):
            targets = {e["target"] for e in graph["edges"]
                       if e["source"] == models[0]["id"] and e["sourceHandle"] == handle}
            self.assertEqual(targets, exporters)

    def test_every_connected_port_exists_and_has_matching_types(self):
        graph = self.graph()
        lookup = {n["id"]: n for n in graph["nodes"]}
        self.assertEqual(len(lookup), 9)
        self.assertTrue(graph["edges"])
        targets = set()
        for edge in graph["edges"]:
            source = lookup[edge["source"]]["data"]
            target = lookup[edge["target"]]["data"]
            outputs = {s["name"]: s["type"] for s in source["outputSlots"]}
            inputs = {s["name"]: s["type"] for s in target["inputs"]}
            self.assertIn(edge["sourceHandle"], outputs)
            self.assertIn(edge["targetHandle"], inputs)
            self.assertEqual(outputs[edge["sourceHandle"]], inputs[edge["targetHandle"]])
            socket = (edge["target"], edge["targetHandle"])
            self.assertNotIn(socket, targets)
            targets.add(socket)

    def test_only_planner_exposes_an_output_prefix(self):
        graph = self.graph()
        editable = [(n["data"]["type"], s["value"]) for n in graph["nodes"]
                    for s in n["data"]["inputs"]
                    if s["name"] == "filename_prefix" and "value" in s]
        self.assertEqual(editable, [("BAMidFarReferencePlan", "reference-test/asset")])


if __name__ == "__main__":
    unittest.main()
