import importlib.util
import copy
import json
import unittest

import cv2
import numpy as np


if importlib.util.find_spec("autonomous_audit"):
    import autonomous_audit as audit
else:
    audit = None


class Availability(unittest.TestCase):
    def test_autonomous_audit_api_is_available(self):
        self.assertIsNotNone(audit, "Autonomous asset audit is not implemented yet.")
        for name in ("inspect_asset", "repair_prompt", "choose_asset"):
            self.assertTrue(callable(getattr(audit, name, None)), name + " is missing.")


GATES = ("target_match", "complete_outline", "connected_parts_preserved",
         "natural_base", "no_unrelated_objects", "style_match", "clean_background")


def plan(**extra):
    return json.dumps({
        "style": "warm stylized stone with atmospheric perspective",
        "objects": [{
            "id": "unit_a", "name": "connected rock formation", "kind": "rock",
            "layer": "midground", "location": "center of the current reference",
            "appearance": "broad main peak and smaller attached peak on a shared ledge",
            "occlusion": "short rock foot hidden by unrelated foreground",
            "bbox": [.2, .15, .8, .85], "asset_unit": "connected_formation",
            "connected_parts": ["main peak", "attached peak", "shared natural ledge"],
            "selection_reason": "one complete connected reusable unit", "is_single": True,
            "excluded_objects": ["trees", "separate terrain behind the target"], **extra
        }]
    })


def verdict(**extra):
    return {**dict.fromkeys(GATES, True), "score": 95, "issues": [], **extra}


def generated(cut=False, multiple=False, soft=True):
    alpha = np.zeros((120, 180), np.float32)
    cv2.fillPoly(alpha, [np.array([[30, 92], [42, 55], [65, 26], [90, 60],
                                 [122, 44], [150, 90], [112, 99], [70, 96]])], 1)
    if cut:
        alpha[55:92, :60] = 1
    if multiple:
        alpha[15:40, 150:173] = 1
    if soft:
        alpha = cv2.GaussianBlur(alpha, (7, 7), .8)
    foreground = np.array([.79, .41, .23], np.float32)
    screen = np.array([.065, .94, .05], np.float32)
    image = (foreground * alpha[..., None] + screen * (1-alpha[..., None]))[None]
    return image.astype(np.float32)


def reference():
    y, x = np.indices((70, 210))
    return np.stack((x/210, y/70, np.full_like(x, .25, dtype=np.float32)), -1)[None].astype("float32")


@unittest.skipIf(audit is None, "New autonomous audit API not available yet")
class AuditTests(unittest.TestCase):
    def inspect(self, image=None, **kwargs):
        return audit.inspect_asset(generated() if image is None else image, reference(),
                                   plan(), "#00FF00", **kwargs)

    def good_candidate(self):
        _, _, rgba, numeric = self.inspect()
        return rgba, json.loads(numeric)

    def reject_visual(self, visual, pattern=None):
        rgba, numeric = self.good_candidate()
        with self.assertRaisesRegex(ValueError, pattern or "No eligible asset"):
            audit.choose_asset([rgba], [numeric], [visual])

    def test_clean_screen_recovers_color_and_soft_alpha(self):
        sheet, prompt, rgba, numeric = self.inspect(padding=8)
        report = json.loads(numeric)
        self.assertIs(report["valid"], True)
        self.assertIs(report["diagnostic_only"], False)
        self.assertEqual(report["problems"], [])
        self.assertEqual(report["matte_error"], "")
        self.assertEqual(report["asset_id"], "unit_a")
        self.assertEqual(report["key_color"], "#00FF00")
        self.assertEqual(report["meaningful_components"], 1)
        self.assertIs(report["touches_border"], False)
        self.assertGreater(report["soft_edge_pixels"], 100)
        self.assertEqual(rgba.dtype, np.float32)
        self.assertEqual(rgba.shape[-1], 4)
        self.assertEqual(sheet.shape[0], 1)
        self.assertEqual(sheet.shape[-1], 3)
        self.assertTrue(np.isfinite(sheet).all())
        self.assertGreater(np.count_nonzero((rgba[..., 3] > 0) & (rgba[..., 3] < 1)), 100)
        self.assertEqual(float(rgba[:, 0, :, 3].max()), 0)
        np.testing.assert_allclose(rgba[0, 50, 45, :3], [.79, .41, .23], atol=.02)
        self.assertIn("SOURCE CROP", prompt)

    def test_comparison_sheet_uses_actual_crop_and_preserves_all_aspects(self):
        source, image = reference(), generated()
        unchanged_source, unchanged_image = source.copy(), image.copy()
        sheet, _, rgba, numeric = self.inspect(image)
        panels = json.loads(numeric)["comparison_panels"]
        self.assertEqual([p["label"] for p in panels[:3]],
                         ["SOURCE CROP", "RAW GENERATED", "RECOVERED ALPHA"])
        for panel, original in zip(panels, (source, image, rgba)):
            x0, y0, x1, y1 = panel["display_bbox"]
            self.assertEqual(panel["input_size"], [original.shape[2], original.shape[1]])
            self.assertAlmostEqual((x1-x0)/(y1-y0), original.shape[2]/original.shape[1],
                                   delta=.03)
            if panel["label"] != "RECOVERED ALPHA":
                np.testing.assert_allclose(sheet[:, y0:y1, x0:x1], original[..., :3], atol=1e-6)
        np.testing.assert_array_equal(source, unchanged_source)
        np.testing.assert_array_equal(image, unchanged_image)

    def test_large_comparison_keeps_foot_detail_readable_without_wide_strip_shrink(self):
        source = np.zeros((1, 400, 500, 3), np.float32)
        raw = np.zeros((1, 2048, 2048, 3), np.float32)
        rgba = np.ones((1, 1200, 1800, 4), np.float32)
        rgba[0, 900:, :, :3] = [.2, .4, .7]
        guide = np.zeros((1, 500, 500, 3), np.float32)
        before = rgba.copy()
        sheet, panels = audit._comparison(source, raw, rgba, guide)
        self.assertLessEqual(sheet.shape[2], 1600)
        self.assertLessEqual(sheet.shape[1], 2048)
        self.assertEqual(len(panels), 5)
        detail = panels[-1]
        self.assertEqual(detail["label"], "LOWER FOOT DETAIL")
        self.assertEqual(detail["source_panel"], "RECOVERED ALPHA")
        bx0, by0, bx1, by1 = detail["source_bbox"]
        self.assertEqual((bx0, bx1, by1), (0, 1800, 1200))
        self.assertGreater(by0, 600)
        dx0, dy0, dx1, dy1 = detail["display_bbox"]
        self.assertGreater(dx1-dx0, 1400)
        expected = cv2.resize(rgba[0, by0:by1, bx0:bx1, :3],
                              (dx1-dx0, dy1-dy0), interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(sheet[0, dy0:dy1, dx0:dx1], expected, atol=1e-6)
        np.testing.assert_array_equal(rgba, before)
        boxes = [p["display_bbox"] for p in panels]
        for i, (x0, y0, x1, y1) in enumerate(boxes):
            for ox0, oy0, ox1, oy1 in boxes[i+1:]:
                self.assertFalse(x0 < ox1 and ox0 < x1 and y0 < oy1 and oy0 < y1)

    def test_detail_handles_empty_diagnostic_alpha_without_changing_candidate(self):
        rgba = np.zeros((1, 80, 160, 4), np.float32)
        sheet, panels = audit._comparison(reference(), generated(), rgba)
        self.assertTrue(np.isfinite(sheet).all())
        self.assertEqual(panels[-1]["label"], "LOWER FOOT DETAIL")
        self.assertGreater(panels[-1]["input_size"][1], 0)
        self.assertEqual(float(rgba.max()), 0)

    def test_review_and_repair_explicitly_reject_foreground_forest_and_crop_planes(self):
        _, prompt, _, numeric = self.inspect()
        repair = audit.repair_prompt(plan(), "#00FF00", numeric,
                                    verdict(no_unrelated_objects=False,
                                            issues=["forest silhouette across lower foot"]), 1)
        for phrase in ("LOWER FOOT DETAIL", "forest silhouette", "side cut planes",
                       "even when", "paper-thin"):
            self.assertIn(phrase, prompt)
        self.assertIn("forest silhouette", repair)
        self.assertIn("side cut planes", repair)
        self.assertIn("reconstruct", repair)

    def test_review_prompt_explicitly_demands_fixed_schema_and_reference_fidelity(self):
        _, prompt, _, _ = self.inspect()
        for field in GATES + ("score", "issues"):
            self.assertIn('"' + field + '"', prompt)
        for instruction in ("true booleans", "0..100", "attached peak", "short natural base",
                            "rectangular notches", "flat sliced", "trees", "depth",
                            "reference", "numeric", "blocking"):
            self.assertIn(instruction, prompt)

    def test_review_allows_expected_asset_isolation_and_hidden_base_completion(self):
        _, prompt, _, _ = self.inspect()
        for phrase in ("Expected transformations", "isolation", "hidden base",
                       "not a defect by itself", "actually visible", "not a hypothetical risk"):
            self.assertIn(phrase, prompt)

    def test_repair_prompt_does_not_introduce_unobserved_pillars_or_caps(self):
        one = plan(name="one smooth mountain", connected_parts=["single peak", "low slope"])
        _, numeric = self.good_candidate()
        prompt = audit.repair_prompt(one, "#00FF00", numeric,
                                    verdict(natural_base=False, issues=["flat base"]), 1)
        self.assertNotIn("attached pillar", prompt)
        self.assertNotIn("multiple peaks", prompt)
        self.assertIn("single peak", prompt)
        self.assertIn("natural", prompt)

    def test_canvas_cut_is_feedback_not_an_inspection_exception(self):
        sheet, prompt, rgba, numeric = self.inspect(generated(cut=True))
        report = json.loads(numeric)
        self.assertIs(report["valid"], False)
        self.assertIs(report["diagnostic_only"], True)
        self.assertIs(report["touches_border"], True)
        self.assertIn("border", " ".join(report["problems"]).lower())
        self.assertEqual(rgba.shape[-1], 4)
        self.assertTrue(np.isfinite(sheet).all())
        self.assertIn("false", prompt)
        with self.assertRaisesRegex(ValueError, "border"):
            audit.choose_asset([rgba], [numeric], [verdict()])

    def test_numeric_matte_failure_retains_reason_and_diagnostic_tensor(self):
        image = np.full((1, 120, 180, 3), .5, np.float32)
        sheet, _, rgba, numeric = self.inspect(image)
        report = json.loads(numeric)
        self.assertIs(report["valid"], False)
        self.assertIs(report["diagnostic_only"], True)
        self.assertIn("Color key not found", report["matte_error"])
        self.assertIn(report["matte_error"], report["problems"])
        self.assertEqual(rgba.shape, (1, 120, 180, 4))
        np.testing.assert_array_equal(rgba[..., :3], image)
        self.assertTrue(np.isfinite(sheet).all())
        with self.assertRaisesRegex(ValueError, "Color key"):
            audit.choose_asset([rgba], [numeric], [verdict()])

    def test_empty_screen_is_rejected_without_interrupting_review(self):
        image = np.broadcast_to([0., 1., 0.], (1, 60, 90, 3)).copy()
        _, _, rgba, numeric = self.inspect(image)
        self.assertFalse(json.loads(numeric)["valid"])
        self.assertIn("stable", json.loads(numeric)["matte_error"])
        with self.assertRaisesRegex(ValueError, "No eligible asset"):
            audit.choose_asset([rgba], [numeric], [verdict()])

    def test_multiple_substantial_units_remain_rejected(self):
        _, _, rgba, numeric = self.inspect(generated(multiple=True))
        report = json.loads(numeric)
        self.assertFalse(report["valid"])
        self.assertGreater(report["meaningful_components"], 1)
        with self.assertRaisesRegex(ValueError, "Multiple"):
            audit.choose_asset([rgba], [numeric], [verdict()])

    def test_tiny_detached_pixels_are_not_silently_accepted_after_cleanup(self):
        image = generated()
        image[:, 5:8, 5:8] = [.79, .41, .23]
        _, _, rgba, numeric = self.inspect(image)
        report = json.loads(numeric)
        self.assertGreater(report["removed_remote_pixels"], 0)
        self.assertFalse(report["valid"])
        with self.assertRaisesRegex(ValueError, "removed"):
            audit.choose_asset([rgba], [numeric], [verdict()])

    def test_remote_key_colored_compression_noise_is_classified_before_matte(self):
        image = generated(soft=False)
        foreground = image[..., 0] > .1
        image[~foreground] = [.97, .02, .96]
        image[:, 7:9, 7:9] = [143/255, 0., 201/255]
        _, _, rgba, numeric = audit.inspect_asset(image, reference(), plan(), "#FF00FF")
        report = json.loads(numeric)
        self.assertTrue(report["valid"], report["problems"])
        self.assertEqual(report["removed_remote_pixels"], 0)
        self.assertEqual(report["suppressed_screen_noise_pixels"], 4)
        audit.choose_asset([rgba], [numeric], [verdict()])

    def test_existing_input_alpha_and_magenta_key_are_preserved(self):
        image = generated(soft=False)
        image[..., 1] = np.where(image[..., 0] < .1, .05, image[..., 1])
        image[..., 0] = np.where(image[..., 0] < .1, .94, image[..., 0])
        image[..., 2] = np.where(image[..., 2] < .1, .93, image[..., 2])
        image = np.concatenate((image, np.full(image.shape[:-1] + (1,), .6)), axis=-1)
        _, _, rgba, numeric = audit.inspect_asset(image, reference(), plan(), "#FF00FF")
        self.assertTrue(json.loads(numeric)["valid"])
        self.assertAlmostEqual(float(rgba[..., 3].max()), .6, places=5)

    def test_invalid_inspection_inputs_are_programming_errors(self):
        for kwargs in ({"softness": float("nan")}, {"softness": float("inf")},
                       {"softness": True}, {"softness": -.01}, {"softness": .3},
                       {"padding": True}, {"padding": -1}, {"padding": 257}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.inspect(**kwargs)
        for image in (np.zeros((2, 10, 10, 3)), np.zeros((10, 10, 3)),
                      np.full((1, 10, 10, 3), np.nan)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.inspect(image)

    def test_one_target_plan_is_required(self):
        data = json.loads(plan())
        data["objects"].append({**data["objects"][0], "id": "unit_b"})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            audit.inspect_asset(generated(), reference(), data, "#00FF00")
        with self.assertRaises(ValueError):
            audit.repair_prompt(plan(boundary_cut=True), "#00FF00", {}, verdict(), 1)

    def test_planning_contract_keeps_structured_excluded_objects_and_connected_parts(self):
        exclusions = ["ochre cloud bank", {
            "id": "separate_ridge", "name": "unconnected ivory arch",
            "location": "behind the left peak", "appearance": "cool distant ridge",
            "reason": "overlapping at a different depth",
        }]
        parts = ["lower attached spire", "shared sloping stone foot"]
        one_plan = json.loads(plan(excluded_objects=exclusions, connected_parts=parts))
        one_plan.update(schema_version=1, source_size=[900, 600], edge_margin=.01)
        before = copy.deepcopy(one_plan)
        _, prompt, _, numeric = audit.inspect_asset(generated(), reference(), one_plan, "#00FF00")
        repair = audit.repair_prompt(one_plan, "#00FF00", numeric, verdict(), 1)
        self.assertIn('"excluded_objects"', prompt)
        for text in ("ochre cloud bank", "unconnected ivory arch", "behind the left peak",
                     "cool distant ridge", "overlapping at a different depth", *parts):
            self.assertIn(text, prompt)
            self.assertIn(text, repair)
        self.assertEqual(json.loads(numeric)["excluded_objects"], exclusions)
        self.assertEqual(json.loads(numeric)["connected_parts"], parts)
        self.assertEqual(one_plan, before)

    def test_bad_or_contradictory_exclusions_and_connected_part_types_fail_closed(self):
        for exclusions in ("trees", [None], [{"name": ""}], [{"name": "other", "reason": False}],
                           ["attached peak"], [{"id": "unit_a", "name": "this unit"}]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                audit.inspect_asset(generated(), reference(), plan(excluded_objects=exclusions), "#00FF00")
        with self.assertRaisesRegex(ValueError, "excluded_objects"):
            audit.repair_prompt(plan(exclude_objects=["legacy alias"]), "#00FF00", {}, verdict(), 1)
        for parts in ("attached peak", [False], [{"name": "attached peak"}]):
            with self.subTest(parts=parts), self.assertRaises(ValueError):
                audit.inspect_asset(generated(), reference(), plan(connected_parts=parts), "#00FF00")

    def test_each_semantic_gate_rejects_a_numeric_pass(self):
        for field in GATES:
            with self.subTest(field=field):
                self.reject_visual(verdict(**{field: False}), field)

    def test_missing_fields_and_aliases_are_not_accepted(self):
        for field in GATES + ("score", "issues"):
            visual = verdict()
            del visual[field]
            with self.subTest(field=field):
                self.reject_visual(visual, field)
        self.reject_visual({**verdict(), "complete": True}, "schema")

    def test_string_boolean_integer_boolean_and_null_are_rejected(self):
        for field in GATES:
            for value in ("true", "false", 1, 0, None):
                with self.subTest(field=field, value=value):
                    self.reject_visual(verdict(**{field: value}), field)

    def test_scores_must_be_finite_numbers_in_range_not_boolean(self):
        for value in ("99", None, True, False, -1, 101, float("nan"),
                      float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.reject_visual(verdict(score=value), "score")

    def test_contradictory_issues_reject_all_true_high_score_verdict(self):
        self.reject_visual(verdict(issues=["missing attached peak"]), "missing attached peak")
        self.reject_visual(verdict(issues=["trees merged into the natural base"]), "trees")
        self.reject_visual(verdict(issues=["rectangular notch in a flat sliced base"]), "notch")

    def test_issues_require_a_concrete_list_of_nonempty_strings(self):
        for issues in ("none", {}, [None], [False], [""], ["   "], [12]):
            with self.subTest(issues=issues):
                self.reject_visual(verdict(issues=issues), "issues")

    def test_malformed_nested_duplicate_and_prose_json_are_rejected(self):
        valid = json.dumps(verdict())
        cases = ("not json", "[]", '{"target_match":true}', '{"review":' + valid + '}',
                 "Here is a review: " + valid, valid[:-1] + ',"score":100}',
                 valid.replace('"score": 95', '"score": NaN'),
                 valid.replace('"score": 95', '"score": 1e309'))
        for visual in cases:
            with self.subTest(visual=visual):
                self.reject_visual(visual)

    def test_json_arrays_dicts_and_single_json_fences_have_same_verdict(self):
        rgba, numeric = self.good_candidate()
        for visual in (verdict(), json.dumps(verdict()), "```json\n" + json.dumps(verdict()) + "\n```"):
            with self.subTest(visual=visual):
                selected, record = audit.choose_asset([rgba], [json.dumps(numeric)], [visual])
                np.testing.assert_array_equal(selected, rgba)
                self.assertEqual(json.loads(record)["selected_index"], 0)
        selected, _ = audit.choose_asset([rgba], json.dumps([numeric]), json.dumps([verdict()]))
        np.testing.assert_array_equal(selected, rgba)

    def test_natural_base_repair_reconstructs_short_rock_foot_not_notches(self):
        rgba, numeric = self.good_candidate()
        visual = verdict(natural_base=False, connected_parts_preserved=False,
                         issues=["missing attached peak", "rectangular notch and flat sliced base"])
        prompt = audit.repair_prompt(plan(), "#00FF00", numeric, visual, 1)
        for phrase in ("short natural base", "rectangular notches", "flat sliced bases",
                       "missing connected peak", "attached peak", "shared natural ledge",
                       "trees", "depth-mismatched terrain", "reference fidelity",
                       "15%", "70%", "#00FF00", "Repair attempt 1"):
            self.assertIn(phrase, prompt)
        self.assertIn("missing attached peak", prompt)

    def test_repair_can_correct_upper_geometry_instead_of_freezing_a_bad_guide(self):
        _, numeric = self.good_candidate()
        visual = verdict(target_match=False, connected_parts_preserved=False,
                         issues=["unrelated curled upper fragment", "missing side peak"])
        prompt = audit.repair_prompt(plan(), "#00FF00", numeric, visual, 1)
        self.assertIn("actual original reference crop", prompt)
        self.assertIn("Correct the entire target silhouette", prompt)
        self.assertNotIn("Only that lower hidden region", prompt)
        self.assertNotIn("只编辑朱红范围", prompt)
        self.assertIn("unrelated curled upper fragment", prompt)
        self.assertIn("missing side peak", prompt)

    def test_repair_identifies_actual_previous_candidate_as_editable_second_image(self):
        _, numeric = self.good_candidate()
        prompt = audit.repair_prompt(plan(), "#00FF00", numeric,
                                     verdict(no_unrelated_objects=False, issues=["forest ribbon"]), 1)
        for phrase in ("Image 1", "Image 2", "previous candidate", "not immutable",
                       "actual visible defects", "forest ribbon"):
            self.assertIn(phrase, prompt)

    def test_clipping_repair_calls_for_scale_down_and_all_side_margins(self):
        _, _, _, numeric = self.inspect(generated(cut=True))
        prompt = audit.repair_prompt(plan(), "#00FF00", numeric, verdict(), 2)
        for phrase in ("scale down", "clipping", "all four sides", "15%", "border"):
            self.assertIn(phrase, prompt)

    def test_repair_retains_malformed_review_as_blocking_feedback(self):
        _, numeric = self.good_candidate()
        for visual in ("broken JSON", verdict(complete_outline="true"),
                       verdict(issues=["missing attached peak"])):
            with self.subTest(visual=visual):
                prompt = audit.repair_prompt(plan(), "#00FF00", numeric, visual, 1)
                self.assertIn("blocking", prompt)
                self.assertIn("not accepted", prompt)
                self.assertIn("reference fidelity", prompt)
        prompt = audit.repair_prompt(plan(), "#00FF00", '{"valid":true}', verdict(), 1)
        self.assertIn("numeric", prompt)
        self.assertIn("not accepted", prompt)

    def test_repair_prompt_ignores_filenames_and_non_contract_metadata(self):
        _, numeric = self.good_candidate()
        prompt = audit.repair_prompt(plan(source_filename="secret-lake-reference.png"),
                                    "#00FF00", numeric, verdict(), 1)
        self.assertNotIn("secret-lake-reference.png", prompt)
        for attempt in (0, -1, True, 1.5, "1"):
            with self.subTest(attempt=attempt), self.assertRaises(ValueError):
                audit.repair_prompt(plan(), "#00FF00", numeric, verdict(), attempt)

    def test_choose_keeps_earlier_good_candidate_when_repairs_are_worse(self):
        rgba, numeric = self.good_candidate()
        _, _, cut_rgba, cut_numeric = self.inspect(generated(cut=True))
        selected, record = audit.choose_asset(
            [rgba, rgba.copy(), cut_rgba], [numeric, numeric, cut_numeric],
            [verdict(score=90), verdict(score=99, issues=["missing attached peak"]),
             verdict(score=100)])
        np.testing.assert_array_equal(selected, rgba)
        report = json.loads(record)
        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["selected_index"], 0)
        self.assertEqual(report["eligible_indices"], [0])
        self.assertEqual(report["score"], 90)
        self.assertEqual(report["asset_id"], "unit_a")
        self.assertEqual([r["index"] for r in report["rejected_candidates"]], [1, 2])

    def test_choose_highest_eligible_score_with_earliest_tie_break(self):
        rgba, numeric = self.good_candidate()
        for scores, expected in (([85, 99, 92], 1), ([95, 95, 90], 0)):
            with self.subTest(scores=scores):
                selected, record = audit.choose_asset([rgba] * 3, [numeric] * 3,
                                                     [verdict(score=s) for s in scores])
                np.testing.assert_array_equal(selected, rgba)
                self.assertEqual(json.loads(record)["selected_index"], expected)
                self.assertEqual(json.loads(record)["eligible_indices"], [0, 1, 2])

    def test_score_threshold_is_inclusive_but_never_overrides_hard_gates(self):
        rgba, numeric = self.good_candidate()
        audit.choose_asset([rgba], [numeric], [verdict(score=85)])
        self.reject_visual(verdict(score=84.99), "score")
        with self.assertRaisesRegex(ValueError, "natural_base"):
            audit.choose_asset([rgba], [numeric], [verdict(score=100, natural_base=False)], 0)
        for value in (-1, 101, True, "85", float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                audit.choose_asset([rgba], [numeric], [verdict()], value)

    def test_numeric_schema_and_inconsistent_pass_flags_are_rejected(self):
        rgba, numeric = self.good_candidate()
        changes = ({"valid": "true"}, {"valid": False}, {"diagnostic_only": True},
                   {"problems": ["border cut"]}, {"warning": "Multiple units"},
                   {"touches_border": True}, {"meaningful_components": 2},
                   {"foreground_coverage": float("nan")}, {"matte_error": "no stable interior"},
                   {"removed_remote_pixels": 4}, {"output_size": [1, 1]},
                   {"source_bbox": [0, 0, 180, 120]})
        for change in changes:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "No eligible asset"):
                audit.choose_asset([rgba], [{**numeric, **change}], [verdict()])
        for report in ({}, {"valid": True}, "not json"):
            with self.subTest(report=report), self.assertRaisesRegex(ValueError, "numeric"):
                audit.choose_asset([rgba], [report], [verdict()])

    def test_malformed_candidate_tensor_cannot_be_selected_by_reports(self):
        rgba, numeric = self.good_candidate()
        for tensor in (rgba[..., :3], np.repeat(rgba, 2, axis=0),
                       np.full_like(rgba, np.nan), np.full_like(rgba, 1.5),
                       np.zeros_like(rgba), np.ones_like(rgba)):
            with self.subTest(shape=tensor.shape), self.assertRaisesRegex(ValueError, "RGBA"):
                audit.choose_asset([tensor], [numeric], [verdict()])

    def test_malformed_later_review_does_not_block_an_earlier_good_version(self):
        rgba, numeric = self.good_candidate()
        _, record = audit.choose_asset([rgba, rgba], [numeric, numeric], [verdict(), "invalid"])
        self.assertEqual(json.loads(record)["selected_index"], 0)
        self.assertIn("visual", " ".join(json.loads(record)["rejected_candidates"][0]["reasons"]))

    def test_selection_record_retains_all_three_original_numeric_and_visual_records(self):
        rgba, numeric = self.good_candidate()
        _, _, cut_rgba, cut_numeric = self.inspect(generated(cut=True))
        numerics = [json.dumps(numeric, indent=2), numeric, cut_numeric]
        visuals = [json.dumps(verdict(score=91), indent=2),
                   verdict(natural_base=False, issues=["flat sliced base"]), "invalid visual JSON"]
        _, text = audit.choose_asset([rgba, rgba, cut_rgba], numerics, visuals)
        record = json.loads(text)
        self.assertEqual(record["numeric_reports"], numerics)
        self.assertEqual(record["visual_reports"], visuals)
        self.assertEqual(record["selected_index"], 0)
        self.assertEqual([r["index"] for r in record["rejected_candidates"]], [1, 2])
        _, text = audit.choose_asset([rgba, rgba, cut_rgba], json.dumps(numerics), json.dumps(visuals))
        self.assertEqual(json.loads(text)["numeric_reports"], numerics)
        self.assertEqual(json.loads(text)["visual_reports"], visuals)

    def test_nonfinite_later_records_are_rejected_individually_and_retained_as_text(self):
        rgba, numeric = self.good_candidate()
        numerics = [numeric, {**numeric, "foreground_coverage": float("nan")}, numeric]
        visuals = [verdict(score=90), verdict(), verdict(score=float("inf"))]
        _, text = audit.choose_asset([rgba] * 3, numerics, visuals)
        record = json.loads(text, parse_constant=lambda token: self.fail("Nonfinite output: " + token))
        self.assertEqual(record["selected_index"], 0)
        self.assertEqual(record["eligible_indices"], [0])
        self.assertIsInstance(record["numeric_reports"][1], str)
        self.assertIn("NaN", record["numeric_reports"][1])
        self.assertIsInstance(record["visual_reports"][2], str)
        self.assertIn("Infinity", record["visual_reports"][2])
        self.assertEqual([r["index"] for r in record["rejected_candidates"]], [1, 2])

    def test_all_failed_is_explicit_and_has_no_fallback(self):
        rgba, numeric = self.good_candidate()
        _, _, cut_rgba, cut_numeric = self.inspect(generated(cut=True))
        with self.assertRaisesRegex(ValueError, "No eligible asset") as raised:
            audit.choose_asset([cut_rgba, rgba], [cut_numeric, numeric],
                               [verdict(), verdict(natural_base=False)])
        self.assertIn("border", str(raised.exception))
        self.assertIn("natural_base", str(raised.exception))

    def test_empty_and_unaligned_lists_are_errors_not_silent_zip_truncation(self):
        rgba, numeric = self.good_candidate()
        for index, inputs in enumerate((([], [], []), ([rgba], [], [verdict()]),
                                       ([rgba], [numeric, numeric], [verdict()]),
                                       ([rgba], [numeric], "not an array"))):
            with self.subTest(case=index), self.assertRaises(ValueError):
                audit.choose_asset(*inputs)

    def test_selection_and_feedback_do_not_mutate_inputs(self):
        rgba, numeric = self.good_candidate()
        visual = verdict()
        before_rgba, before_numeric, before_visual = rgba.copy(), copy.deepcopy(numeric), copy.deepcopy(visual)
        audit.repair_prompt(plan(), "#00FF00", numeric, visual, 1)
        selected, _ = audit.choose_asset([rgba], [numeric], [visual])
        selected[:] = 0
        np.testing.assert_array_equal(rgba, before_rgba)
        self.assertEqual(numeric, before_numeric)
        self.assertEqual(visual, before_visual)


if __name__ == "__main__":
    unittest.main(verbosity=2)
