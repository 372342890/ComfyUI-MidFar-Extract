import json
import unittest
import numpy as np
from core import parse_plan, choose_key, compile_prompt, matte


class PlanTests(unittest.TestCase):
    def test_json_fence_and_surrounding_prose(self):
        plan = parse_plan('分析如下：```json\n{"midground":["拱桥"],"background":["远山"],"style":"水彩"}\n```')
        self.assertEqual(plan["midground"], ["拱桥"])

    def test_invalid_or_missing_plan_rejected(self):
        for text in ["oops", '{"midground":[]}', '{"midground":"桥","background":[]}']:
            with self.assertRaises(ValueError):
                parse_plan(text)

    def test_prompt_has_dynamic_objects_and_no_fixed_scene(self):
        plan = parse_plan('{"midground":["红色拱桥"],"background":["都市天际线"],"style":"写实","occlusion":"桥遮挡了天际线"}')
        text = compile_prompt(plan, "background", "#00FF00")
        self.assertIn("都市天际线", text)
        self.assertIn("红色拱桥", text)
        self.assertIn("遮挡", text)
        for forbidden in ["风车", "小屋", "雪松", "25%", "森林"]:
            self.assertNotIn(forbidden, text)
        self.assertIn("同一", text)

    def test_empty_layer_never_invents_objects(self):
        plan = parse_plan('{"midground":[],"background":["远山"]}')
        self.assertIn("没有", compile_prompt(plan, "midground", "#FF00FF"))

    def test_auto_key_avoids_reference_colors(self):
        green = np.zeros((1, 20, 20, 3), np.float32)
        green[..., 1] = 1
        self.assertNotEqual(choose_key(green, "auto"), "#00FF00")
        self.assertEqual(choose_key(green, "green"), "#00FF00")
        with self.assertRaises(ValueError):
            choose_key(green, "invalid")


class MatteTests(unittest.TestCase):
    @staticmethod
    def composite(key="#00FF00"):
        color = np.array([0.35, 0.22, 0.12], np.float32)
        k = np.array([int(key[i:i+2], 16) / 255 for i in (1, 3, 5)], np.float32)
        alpha = np.zeros((15, 19), np.float32)
        alpha[4:11, 5:14] = 1
        alpha[3, 5:14] = 0.45
        alpha[4:11, 4] = 0.7
        # One pixel wide opaque branch must survive.
        alpha[1:5, 9] = 1
        rgb = color * alpha[..., None] + k * (1-alpha[..., None])
        return rgb[None], alpha[None], color

    def test_known_mixed_alpha_and_foreground_recovery(self):
        rgb, truth, color = self.composite()
        rgba, mask, preview, info = matte(rgb, "#00FF00", 0.04, True)
        np.testing.assert_allclose(mask, truth, atol=0.02)
        np.testing.assert_allclose(rgba[0, 3, 6, :3], color, atol=0.03)
        self.assertEqual(rgba.shape, (1, 15, 19, 4))
        self.assertEqual(preview.shape[-1], 3)
        self.assertIn("foreground", info)

    def test_multiple_keys_and_batches(self):
        for key in ["#00FF00", "#FF00FF", "#0000FF"]:
            rgb, truth, _ = self.composite(key)
            rgba, mask, _, _ = matte(np.repeat(rgb, 2, axis=0), key, 0.04, True)
            np.testing.assert_allclose(mask[0], truth[0], atol=0.03)
            np.testing.assert_allclose(mask[1], truth[0], atol=0.03)
            self.assertEqual(float(rgba[0, 1, 9, 3]), 1)

    def test_all_screen_and_no_screen(self):
        rgb = np.zeros((1, 12, 12, 3), np.float32)
        rgb[..., 1] = 1
        self.assertEqual(matte(rgb, "#00FF00", 0.04, True)[1].max(), 0)
        self.assertEqual(matte(np.full_like(rgb, .3), "#00FF00", 0.04, True)[1].min(), 1)

    def test_existing_alpha_and_bad_input(self):
        rgb, truth, _ = self.composite()
        rgba = np.concatenate([rgb, np.full((*rgb.shape[:-1], 1), .5)], axis=-1)
        np.testing.assert_allclose(matte(rgba, "#00FF00", .04, True)[1], truth * .5, atol=.02)
        for data, key in [(rgb, "red"), (np.full_like(rgb, np.nan), "#00FF00"), (rgb[0], "#00FF00")]:
            with self.assertRaises(ValueError):
                matte(data, key, .04, True)

    def test_generated_screen_brightness_variation_is_transparent(self):
        for key in ["#00FF00", "#FF00FF", "#0000FF"]:
            k = np.array([int(key[i:i+2], 16)/255 for i in (1, 3, 5)], np.float32)
            brightness = np.linspace(.75, 1, 24, dtype=np.float32)[None, :, None]
            rgb = np.repeat((brightness*k)[None], 16, axis=1)
            self.assertEqual(matte(rgb, key, .04, True)[1].max(), 0)

    def test_actual_screen_color_is_used_for_edge_unmixing(self):
        rgb, truth, color = self.composite()
        actual_key = np.array([.018, .84, .018], np.float32)
        rgb = color*truth[..., None] + actual_key*(1-truth[..., None])
        rgba, mask, _, _ = matte(rgb, "#00FF00", .04, True)
        np.testing.assert_allclose(mask, truth, atol=.02)
        np.testing.assert_allclose(rgba[0, 3, 6, :3], color, atol=.03)

    def test_tinted_generated_screen_does_not_leave_outer_halo(self):
        _, truth, color = self.composite()
        actual_key = np.array([.06, .91, .03], np.float32)
        rgb = color*truth[..., None] + actual_key*(1-truth[..., None])
        rgba, mask, _, _ = matte(rgb, "#00FF00", .04, True)
        np.testing.assert_allclose(mask, truth, atol=.02)
        np.testing.assert_allclose(rgba[0, 3, 6, :3], color, atol=.03)

    def test_low_alpha_key_residue_away_from_subject_is_transparent(self):
        rgb, truth, _ = self.composite("#FF00FF")
        # Image editors can leave faint hue streaks in otherwise blank key areas.
        # These must not turn into white translucent cloth on a dark backdrop.
        rgb[0, :2, :2] = np.array([.9, .1, 1], np.float32)
        rgba, mask, _, _ = matte(rgb, "#FF00FF", .04, True)
        np.testing.assert_allclose(mask[0, :2, :2], 0, atol=1e-6)
        self.assertEqual(float(mask[0, 1, 9]), 1)
        np.testing.assert_allclose(mask[0, 3, 6], truth[0, 3, 6], atol=.02)

    def test_residue_cleanup_counts_as_detected_screen(self):
        rgb = np.empty((1, 64, 64, 3), np.float32)
        rgb[:] = [.9, .1, 1]
        rgb[0, 24:40, 24:40] = [.3, .3, .3]
        _, mask, _, info = matte(rgb, "#FF00FF", .04, True)
        self.assertEqual(float(mask[0, 0, 0]), 0)
        self.assertEqual(json.loads(info)[0]["warning"], "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
