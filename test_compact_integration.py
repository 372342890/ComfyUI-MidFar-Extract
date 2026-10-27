import json
from pathlib import Path
import unittest


ROOT = Path(__file__).parent


class CompactIntegration(unittest.TestCase):
    def test_single_upload_and_nine_real_nodes(self):
        path = ROOT/"midfar-compact.byteartist.json"
        self.assertTrue(path.is_file(), "Nine-node workflow is missing.")
        graph = json.loads(path.read_text(encoding="utf8"))
        kinds = [n["data"]["type"] for n in graph["nodes"]]
        self.assertEqual(len(kinds), 9)
        self.assertEqual(kinds.count("LoadImage"), 1)
        self.assertEqual(kinds.count("BAMidFarCompactExtract"), 1)
        self.assertEqual(kinds.count("BAMidFar3DModel"), 1)
        self.assertEqual(kinds.count("BAMidFar3DGenerate"), 3)
        self.assertEqual(kinds.count("BAMidFarGLBExport"), 3)
        self.assertNotIn("AIPExecScript", kinds)
        self.assertNotIn("BALLMImage", kinds)

    def test_every_model_exports_its_own_image_and_prefix(self):
        path = ROOT/"midfar-compact.byteartist.json"
        self.assertTrue(path.is_file(), "Nine-node workflow is missing.")
        graph = json.loads(path.read_text(encoding="utf8"))
        for i in range(1, 4):
            generate = str(10+i)
            export = str(20+i)
            incoming = {e["targetHandle"]: (e["source"], e["sourceHandle"])
                        for e in graph["edges"] if e["target"] == export}
            self.assertEqual(incoming["mesh"], (generate, "mesh"))
            self.assertEqual(incoming["rgba"], ("2", f"rgba_{i}"))
            self.assertEqual(incoming["filename_prefix"], ("2", f"prefix_{i}"))
            incoming = {e["targetHandle"]: (e["source"], e["sourceHandle"])
                        for e in graph["edges"] if e["target"] == generate}
            self.assertEqual(incoming["rgba"], ("2", f"rgba_{i}"))
            self.assertEqual(incoming["model"], ("3", "model"))

    def test_native_settings_retain_original_models_and_generation_parameters(self):
        path = ROOT/"compact_settings.json"
        self.assertTrue(path.is_file(), "Native settings are missing.")
        settings = json.loads(path.read_text(encoding="utf8"))
        self.assertEqual(set(settings), {"plan", "selection", "localization",
                                        "generate", "repair", "review"})
        for stage, cfg in settings.items():
            model = json.loads(cfg["model"])
            self.assertEqual(model["key"], "Doubao-Seedream-5.0-pro"
                             if stage in ("generate", "repair") else "Doubao-Seed-2.1-pro")
        self.assertEqual(settings["generate"]["extra_inputs"]["size"], "2048x2048")
        self.assertFalse(settings["generate"]["extra_inputs"]["watermark"])

    def test_plugin_registers_compact_node(self):
        text = (ROOT/"__init__.py").read_text(encoding="utf8")
        self.assertIn('"BAMidFarCompactExtract": BAMidFarCompactExtract', text)


if __name__ == "__main__":
    unittest.main()
