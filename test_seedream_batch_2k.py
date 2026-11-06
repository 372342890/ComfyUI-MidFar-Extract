"""Contract tests use a fake paid carrier, with real PNG input/cache files."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image
import torch

if importlib.util.find_spec("seedream_batch_2k"):
    import seedream_batch_2k as batch
else:
    batch = None


class Availability(unittest.TestCase):
    def test_batch_enhancer_exists(self):
        self.assertIsNotNone(batch, "Seedream 5.0 Pro batch enhancer is missing.")


@unittest.skipIf(batch is None, "Seedream batch enhancer is missing.")
class BatchContract(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calls = []
        self.fail_at = None
        self.reply = None
        owner = self

        class Native:
            INPUT_IS_LIST = True
            FUNCTION = "run_with_tea"

            def run_with_tea(self, **kw):
                file = Path(kw["image0"][0])
                with Image.open(file) as image:
                    pixels = np.asarray(image.convert("RGB")).copy()
                owner.calls.append((kw, dict(self.node_info), pixels))
                if owner.fail_at == len(owner.calls):
                    raise RuntimeError("provider request denied")
                if owner.reply is not None:
                    return {"result": (owner.reply,)}
                size = int(kw["extra_inputs"][0]["size"].split("x")[0])
                color = torch.from_numpy(pixels[0, 0].astype("float32") / 255)
                return {"result": (color.reshape(1, 1, 1, 3).expand(
                    1, size, size, 3),)}

        self.mappings = {"BALLMImage": Native}

    def source(self, count=2):
        result = torch.zeros((count, 12, 16, 3))
        for index in range(count):
            result[index, ..., index % 3] = 1
        return result

    def run_batch(self, images=None, **options):
        return batch.enhance_batch(
            self.source() if images is None else images, self.mappings,
            {"prompt_id": "task-123", "node_id": "81"}, "81",
            {"81": {"class_type": "SeedreamBatch2K"}},
            self.root / "input", self.root / "output", **options)

    def test_isolated_ordered_native_requests_and_exact_1k_outputs(self):
        images, records = self.run_batch()
        self.assertEqual(tuple(images.shape), (2, 1024, 1024, 3))
        self.assertEqual(images.device.type, "cpu")
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(len(self.calls), 2)
        self.assertTrue(torch.equal(images[0, 0, 0], torch.tensor([1., 0., 0.])))
        self.assertTrue(torch.equal(images[1, 0, 0], torch.tensor([0., 1., 0.])))
        for kw, info, _ in self.calls:
            self.assertEqual([key for key in kw if key.startswith("image")], ["image0"])
            self.assertTrue(all(isinstance(value, list) and len(value) == 1
                                for value in kw.values()))
            self.assertEqual(info["prompt_id"], "task-123")
            self.assertEqual(kw["extra_inputs"][0]["size"], "1024x1024")
            self.assertFalse(kw["extra_inputs"][0]["watermark"])
            model = json.loads(kw["model"][0])
            self.assertEqual(model["model_name"], "Doubao-Seedream-5.0-pro")
            self.assertIn("唯一", kw["user_prompt"][0])
        self.assertEqual([item["index"] for item in records], [1, 2])
        for item in records:
            self.assertTrue(item["relative_file"].startswith("seedream-batch-1k/"))
            with Image.open(self.root / "output" / item["relative_file"]) as image:
                self.assertEqual(image.size, (1024, 1024))

    def test_partial_failure_is_indexed_not_retried_and_success_reused(self):
        self.fail_at = 2
        with self.assertRaisesRegex(RuntimeError, "2/2.*provider request denied"):
            self.run_batch()
        self.assertEqual(len(self.calls), 2)
        self.fail_at = None
        images, records = self.run_batch()
        self.assertEqual(len(self.calls), 3)
        self.assertEqual([item["cached"] for item in records], [True, False])
        self.assertEqual(tuple(images.shape), (2, 1024, 1024, 3))

    def test_saved_2048_prompt_is_migrated_without_changing_other_instructions(self):
        for dimensions in ("2048×2048", "2048x2048", "2048 X 2048"):
            self.run_batch(self.source(1), reuse_completed=False,
                           instruction=f"输出{dimensions}，保留木箱和当前视角。")
            instruction = self.calls[-1][0]["user_prompt"][0]
            self.assertEqual(instruction, "输出1024×1024，保留木箱和当前视角。")

    def test_seed_and_instruction_invalidate_completed_cache(self):
        self.run_batch(self.source(1), seed=7)
        self.run_batch(self.source(1), seed=7)
        self.assertEqual(len(self.calls), 1)
        self.run_batch(self.source(1), seed=8)
        self.run_batch(self.source(1), seed=8, instruction="保留唯一参考图的造型。")
        self.assertEqual(len(self.calls), 3)

    def test_empty_invalid_sources_fail_before_native_request(self):
        for source in (torch.empty(0, 12, 16, 3), torch.zeros(12, 16, 3),
                       torch.full((1, 2, 2, 3), float("nan")),
                       torch.full((1, 2, 2, 3), 1.5)):
            with self.subTest(shape=source.shape):
                with self.assertRaises(ValueError):
                    self.run_batch(source)
        self.assertEqual(len(self.calls), 0)

    def test_wrong_size_and_multiple_or_missing_native_outputs_are_rejected(self):
        for reply in (torch.zeros(1, 2048, 2048, 3), torch.zeros(2, 1024, 1024, 3),
                      [], [torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3)],
                      None):
            self.reply = reply
            if reply is None:
                self.reply = "no IMAGE"
            with self.subTest(kind=type(reply).__name__):
                with self.assertRaisesRegex(RuntimeError, "1/1"):
                    self.run_batch(self.source(1))
        self.assertEqual(len(self.calls), 5)

    def test_corrupt_cached_output_fails_without_paid_regeneration(self):
        _, records = self.run_batch(self.source(1))
        path = self.root / "output" / records[0]["relative_file"]
        path.write_bytes(b"broken PNG")
        with self.assertRaisesRegex(RuntimeError, "cache"):
            self.run_batch(self.source(1))
        self.assertEqual(len(self.calls), 1)

    def test_rgba_input_composites_over_white_and_limit_is_explicit(self):
        source = torch.zeros(2, 12, 16, 4)
        source[0, ..., 0] = 1
        source[0, ..., 3] = .5
        images, _ = self.run_batch(source, max_images=1)
        self.assertEqual(images.shape[0], 1)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.calls[0][2][0, 0].tolist(), [255, 128, 128])

    def test_interruption_before_next_image_keeps_completed_cache(self):
        checks = []

        def interrupted():
            checks.append(True)
            if len(checks) == 2:
                raise RuntimeError("cancelled")

        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            self.run_batch(check_interrupt=interrupted)
        self.assertEqual(len(self.calls), 1)
        _, records = self.run_batch()
        self.assertTrue(records[0]["cached"])
        self.assertEqual(len(self.calls), 2)

    def test_single_image_output_port_and_fixed_defaults(self):
        node = batch.SeedreamBatch2K
        self.assertEqual(node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(node.RETURN_NAMES, ("images_2048",))
        inputs = node.INPUT_TYPES()
        self.assertEqual(inputs["required"]["images"], ("IMAGE",))
        self.assertEqual(inputs["required"]["max_images"][1]["default"], 0)
        self.assertIn("1024×1024", inputs["required"]["instruction"][1]["default"])
        self.assertIn("unique_id", inputs["hidden"])
        self.assertIn("prompt", inputs["hidden"])
        registration = Path(__file__).with_name("__init__.py").read_text(encoding="utf8")
        self.assertIn('"SeedreamBatch2K": SeedreamBatch2K', registration)

    def test_selected_resolution_controls_native_request_and_exact_output(self):
        for size in (960, 2048):
            with self.subTest(size=size):
                images, records = self.run_batch(
                    resolution=f"{size}x{size}", reuse_completed=False)
                self.assertEqual(tuple(images.shape), (2, size, size, 3))
                self.assertTrue(torch.equal(images[1, 0, 0],
                                            torch.tensor([0., 1., 0.])))
                settings = self.calls[-1][0]
                self.assertEqual(settings["extra_inputs"][0]["size"], f"{size}x{size}")
                self.assertIn(f"{size}×{size}", settings["user_prompt"][0])
                for record in records:
                    with Image.open(self.root / "output" / record["relative_file"]) as img:
                        self.assertEqual(img.size, (size, size))

    def test_resolution_caches_are_isolated_and_reusable(self):
        paths = []
        for resolution in ("960x960", "1024x1024", "2048x2048"):
            _, first = self.run_batch(self.source(1), resolution=resolution)
            _, second = self.run_batch(self.source(1), resolution=resolution)
            paths.append(first[0]["relative_file"])
            self.assertFalse(first[0]["cached"])
            self.assertTrue(second[0]["cached"])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(set(paths)), 3)
        # Missing resolution and explicit 1024 must reuse the legacy default cache.
        _, records = self.run_batch(self.source(1))
        self.assertTrue(records[0]["cached"])
        self.assertEqual(records[0]["relative_file"], paths[1])

    def test_invalid_resolution_is_rejected_before_native_request(self):
        for resolution in ("1024", "512x512", "768x768", "512x1024", 2048, None):
            with self.subTest(resolution=resolution):
                with self.assertRaisesRegex(ValueError, "resolution"):
                    self.run_batch(resolution=resolution)
        self.assertEqual(len(self.calls), 0)

    def test_wrong_selected_size_is_rejected_without_resizing_or_retrying(self):
        self.reply = torch.zeros(1, 1024, 1024, 3)
        with self.assertRaisesRegex(RuntimeError, "1/2.*2048x2048"):
            self.run_batch(resolution="2048x2048")
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(list((self.root / "output").rglob("*.png")), [])

    def test_known_saved_dimensions_follow_selection(self):
        self.run_batch(self.source(1), resolution="960x960",
                       instruction="输出1024 X 1024，不要拼成2048×2048或512x512或768x768；保留尺寸300x300的标牌。")
        self.assertEqual(self.calls[0][0]["user_prompt"][0],
                         "输出960×960，不要拼成960×960或960×960或960×960；保留尺寸300x300的标牌。")

    def test_resolution_widget_is_optional_with_legacy_default(self):
        options, metadata = batch.SeedreamBatch2K.INPUT_TYPES()["optional"]["resolution"]
        self.assertEqual(options, ["960x960", "1024x1024", "2048x2048"])
        self.assertEqual(metadata["default"], "1024x1024")

    def test_records_distinguish_native_time_from_cache_and_report_completion(self):
        completed = []
        _, records = self.run_batch(
            self.source(1), on_completed=lambda *args: completed.append(args))
        self.assertGreater(records[0]["request_seconds"], 0)
        self.assertEqual(records[0]["resolution"], "1024x1024")
        self.assertEqual(completed, [(1, 1)])
        _, cached = self.run_batch(self.source(1))
        self.assertEqual(cached[0]["request_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
