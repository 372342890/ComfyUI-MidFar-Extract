import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


class ThreeDIntegrationTests(unittest.TestCase):
    def test_three_classes_are_registered_and_native_execution_is_lazy(self):
        tree = ast.parse((ROOT/"__init__.py").read_text(encoding="utf-8"))
        mappings = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS"
                                for t in n.targets))
        names = {k.value for k in mappings.keys}
        self.assertTrue({"BAMidFar3DModel", "BAMidFar3DGenerate", "BAMidFarGLBExport"}
                        <= names)
        module = ast.parse((ROOT/"three_d_nodes.py").read_text(encoding="utf-8"))
        imports = [n for n in module.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(getattr(n, "module", "") in
                             ("nodes", "folder_paths", "comfy_extras.nodes_hunyuan3d")
                             for n in imports))

    def test_original_png_graph_is_identical_and_only_seven_nodes_are_added(self):
        old = json.loads((ROOT/"midfar-three-autonomous.byteartist.json").read_text(encoding="utf-8"))
        new = json.loads((ROOT/"midfar-three-glb.byteartist.json").read_text(encoding="utf-8"))
        self.assertEqual(len(new["nodes"]), len(old["nodes"])+7)
        original = {n["id"]: n for n in old["nodes"]}
        self.assertEqual([n for n in new["nodes"] if n["id"] in original], old["nodes"])
        self.assertEqual(new["edges"][:len(old["edges"])], old["edges"])

    def test_three_glbs_depend_on_matching_approved_rgba_and_shared_model(self):
        graph = json.loads((ROOT/"midfar-three-glb.byteartist.json").read_text(encoding="utf-8"))
        by_id = {n["id"]: n for n in graph["nodes"]}
        exports = [n for n in graph["nodes"] if n["data"]["type"] == "BAMidFarGLBExport"]
        loaders = [n["id"] for n in graph["nodes"] if n["data"]["type"] == "BAMidFar3DModel"]
        self.assertEqual(len(exports), 3)
        self.assertEqual(len(loaders), 1)
        for export in exports:
            inputs = {e["targetHandle"]: e for e in graph["edges"] if e["target"] == export["id"]}
            self.assertEqual(by_id[inputs["rgba"]["source"]]["data"]["type"], "BAMidFarChooseAsset")
            shape = inputs["mesh"]["source"]
            shape_inputs = {e["targetHandle"]: e for e in graph["edges"] if e["target"] == shape}
            self.assertEqual(shape_inputs["rgba"]["source"], inputs["rgba"]["source"])
            for name in ("model", "clip_vision", "vae"):
                self.assertEqual(shape_inputs[name]["source"], loaders[0])


if __name__ == "__main__":
    unittest.main()
