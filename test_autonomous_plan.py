import copy
import importlib.util
import json
import unittest

import cv2
import numpy as np

from selection import crop_reference

if importlib.util.find_spec("autonomous_plan"):
    import autonomous_plan
else:
    autonomous_plan = None


def candidate(identifier, **extra):
    return {
        "id": identifier, "unit_id": identifier, "name": "Weathered joined formation",
        "kind": "rock", "layer": "midground", "location": "Central middle distance",
        "appearance": "Ochre twin caps attached to a narrow stone shelf",
        "occlusion": "Lower left foot hidden by a separate distant ridge",
        "selection_reason": "Visible connected silhouette with a short natural base",
        "bbox": [.2, .2, .6, .8], "asset_unit": "connected_formation",
        "connected_parts": ["tall cap", "attached low cap", "shared stone shelf"],
        "excluded_objects": ["separate distant ridge", "foreground grass tufts"],
        "priority": 80, "is_single": True, "boundary_cut": False, "suitable": True,
        **extra,
    }


def plan(objects):
    return json.dumps({"style": "Soft painted ochre rocks and cool aerial perspective",
                       "objects": objects}, ensure_ascii=True)


def scene():
    y, x = np.indices((120, 240), dtype=np.float32)
    return np.stack((x / 240, y / 120, (x + y) / 360), axis=-1)[None]


class Availability(unittest.TestCase):
    def test_autonomous_planning_api_is_available(self):
        self.assertIsNotNone(
            autonomous_plan, "Autonomous planning module is not implemented yet.")
        for name in ("candidate_review", "split_three", "asset_reference"):
            self.assertTrue(callable(getattr(autonomous_plan, name, None)), name)


@unittest.skipIf(autonomous_plan is None, "Autonomous planning API unavailable")
class CandidateReviewTests(unittest.TestCase):
    def review(self, objects, image=None, **kwargs):
        sheet, manifest = autonomous_plan.candidate_review(
            plan(objects), scene() if image is None else image, **kwargs)
        self.assertIsInstance(manifest, str)
        return sheet, json.loads(manifest)

    def test_actual_source_edges_rejected_even_with_boundary_cut_false(self):
        boxes = [[0, .2, .6, .8], [.2, 0, .6, .8], [.2, .2, 1, .8],
                 [.2, .2, .6, 1], [.005, .2, .6, .8]]
        objects = [candidate("cut_" + str(i), bbox=b) for i, b in enumerate(boxes)]
        _, report = self.review(objects + [candidate("safe")])
        self.assertEqual(set(report["rejected_ids"]), {o["id"] for o in objects})
        self.assertEqual([o["id"] for o in report["accepted_candidates"]], ["safe"])
        self.assertTrue(all("edge" in o["reason"].lower()
                            for o in report["rejected_objects"]))
        self.assertEqual(report["original_objects"], objects + [candidate("safe")])

    def test_grass_scenery_and_explicit_unsuitable_or_cut_are_excluded(self):
        objects = [candidate("tuft", name="Grass tuft", kind="hill"),
                   candidate("scenery", name="Entire scene", kind="mountain"),
                   candidate("forest", name="Forest scenery layer"),
                   candidate("cut", boundary_cut=True), candidate("bad", suitable=False),
                   candidate("safe")]
        _, report = self.review(objects)
        self.assertEqual(report["rejected_ids"], ["tuft", "scenery", "forest", "cut", "bad"])
        self.assertEqual(len(report["accepted_candidates"]), 1)

    def test_invalid_candidate_fields_are_diagnostics_not_silently_accepted(self):
        invalid = [
            {"bbox": [float("nan"), .2, .6, .8]},
            {"bbox": [.2, .2, float("inf"), .8]},
            {"bbox": [.6, .2, .2, .8]}, {"bbox": [.2, .2, .6, 1.1]},
            {"bbox": [True, .2, .6, .8]}, {"priority": float("nan")},
            {"boundary_cut": "false"}, {"suitable": 1}, {"is_single": "true"},
            {"is_single": False}, {"connected_parts": "one peak"},
        ]
        for fields in invalid:
            with self.subTest(fields=fields):
                _, report = self.review([candidate("bad", **fields), candidate("safe")])
                self.assertIn("bad", report["rejected_ids"])
                self.assertEqual([o["id"] for o in report["accepted_candidates"]], ["safe"])
                json.dumps(report, allow_nan=False)

    def test_json_input_and_parameter_types_are_strict(self):
        for value in ({"style": "x", "objects": [candidate("x")]},
                      '{"style":"x","objects":[{"id":"x","bbox":(.2,.2,.6,.8)}]}',
                      plan([candidate("same"), candidate("same")]),
                      '{"style":"","objects":[]}'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                autonomous_plan.candidate_review(value, scene())
        for margin in (True, "0.01", float("nan"), float("inf"), -.1, .5):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                self.review([candidate("x")], edge_margin=margin)

    def test_six_to_nine_candidates_are_not_truncated_to_three(self):
        objects = [candidate("arbitrary_" + str(i), priority=i) for i in range(9)]
        _, report = self.review(objects)
        self.assertEqual(len(report["accepted_candidates"]), 9)
        self.assertEqual(report["status"], "ready")
        self.assertFalse(report["diagnostics"])

    def test_insufficient_or_zero_safe_candidates_produce_structured_diagnostics(self):
        for objects, count in (([], 0), ([candidate("edge", bbox=[0, .2, .5, .8])], 0),
                               ([candidate("one"), candidate("two")], 2)):
            with self.subTest(count=count):
                sheet, report = self.review(objects)
                self.assertEqual(report["status"], "insufficient_safe_candidates")
                self.assertEqual(report["safe_unit_count"], count)
                self.assertTrue(report["diagnostics"])
                self.assertEqual(len(report["accepted_candidates"]), count)
                self.assertEqual(sheet.shape[0], 1)
                self.assertEqual(sheet.shape[-1], 3)
                self.assertEqual(sheet.dtype, np.float32)

    def test_overlap_alone_does_not_merge_but_explicit_shared_unit_preserves_parts(self):
        a = candidate("cap", unit_id="joined", bbox=[.3, .1, .5, .6],
                      connected_parts=["tall cap"])
        b = candidate("shelf", unit_id="joined", bbox=[.2, .4, .7, .85],
                      connected_parts=["attached low cap", "shared stone shelf"])
        _, report = self.review([a, b, candidate("behind"), candidate("front")])
        self.assertEqual(report["safe_unit_count"], 3)
        whole = next(o for o in report["accepted_candidates"] if o["unit_id"] == "joined")
        self.assertEqual(whole["source_ids"], ["cap", "shelf"])
        self.assertEqual(whole["bbox"], [.2, .1, .7, .85])
        self.assertEqual(whole["connected_parts"],
                         ["tall cap", "attached low cap", "shared stone shelf"])

    def test_cut_connected_part_disqualifies_whole_unit(self):
        _, report = self.review([
            candidate("edge", unit_id="whole", bbox=[0, .1, .4, .8]),
            candidate("interior", unit_id="whole"), candidate("other")])
        self.assertEqual(set(report["rejected_ids"]), {"edge", "interior"})
        self.assertEqual([o["id"] for o in report["accepted_candidates"]], ["other"])

    def test_sheet_contains_actual_aspect_preserving_crops_and_ascii_labels(self):
        image = scene()
        before = image.copy()
        sheet, report = self.review([candidate("ridge_7", bbox=[.25, .25, .75, .75])],
                                   image=image)
        self.assertEqual(sheet.dtype, np.float32)
        self.assertTrue(np.isfinite(sheet).all())
        self.assertTrue(np.all((sheet >= 0) & (sheet <= 1)))
        tile = report["contact_sheet"]["tiles"][0]
        self.assertEqual(tile["label"], "ridge_7")
        x0, y0, x1, y1 = tile["sheet_crop_pixels"]
        crop, info = crop_reference(image, [.25, .25, .75, .75], 0)
        expected = cv2.resize(crop[0, ..., :3], (x1 - x0, y1 - y0),
                              interpolation=cv2.INTER_AREA)
        np.testing.assert_allclose(sheet[0, y0:y1, x0:x1], expected, atol=1e-6)
        self.assertAlmostEqual((x1 - x0) / (y1 - y0), 2, delta=.025)
        self.assertEqual(tile["source_bbox_pixels"], info["source_bbox_pixels"])
        lx0, ly0, lx1, ly1 = tile["sheet_label_pixels"]
        self.assertGreater(float(np.ptp(sheet[0, ly0:ly1, lx0:lx1])), .1)
        np.testing.assert_array_equal(image, before)

    def test_image_validation_and_rgba_sheet_are_consistent(self):
        rgba = np.concatenate((scene(), np.ones((1, 120, 240, 1), np.float32)), axis=-1)
        sheet, _ = self.review([candidate("a")], image=rgba)
        self.assertEqual(sheet.shape[-1], 3)
        for image in (scene()[0], np.repeat(scene(), 2, axis=0),
                      np.full((1, 20, 20, 3), np.nan)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.review([candidate("a")], image=image)

    def test_tuple_bbox_is_not_a_normalized_json_array(self):
        with self.assertRaises(ValueError):
            autonomous_plan._candidate(candidate("tuple", bbox=(.2, .2, .6, .8)), "style", .01)

    def test_long_ascii_ids_are_fully_rendered_not_clipped(self):
        identifier = "wide_" + "W" * 35
        sheet, report = self.review([candidate(identifier)])
        x0, y0, x1, y1 = report["contact_sheet"]["tiles"][0]["sheet_label_pixels"]
        expected = np.full((y1-y0, x1-x0, 3), 20, np.uint8)
        width = cv2.getTextSize(identifier, cv2.FONT_HERSHEY_SIMPLEX, .6, 1)[0][0]
        scale = min(.6, .6 * (x1-x0-18) / width)
        cv2.putText(expected, identifier, (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (242, 242, 242), 1, cv2.LINE_AA)
        np.testing.assert_allclose(sheet[0, y0:y1, x0:x1], expected.astype(np.float32)/255)

    def test_ambiguous_duplicate_boolean_keys_are_not_accepted(self):
        text = plan([candidate("duplicate")]).replace(
            '"boundary_cut": false', '"boundary_cut": true, "boundary_cut": false')
        with self.assertRaises(ValueError):
            autonomous_plan.candidate_review(text, scene())

    def test_integer_thousand_grid_keeps_raw_evidence_and_exact_source_crop(self):
        raw = candidate("grid", bbox=[200, 200, 600, 800])
        sheet, report = self.review([raw, candidate("b"), candidate("c")])
        self.assertEqual(report["safe_unit_count"], 3)
        target = next(o for o in report["accepted_candidates"] if o["id"] == "grid")
        self.assertEqual(target["bbox"], [.2, .2, .6, .8])
        self.assertEqual(target["original_description"], raw)
        tile = report["contact_sheet"]["tiles"][0]
        expected, info = crop_reference(scene(), [.2, .2, .6, .8], 0)
        self.assertEqual(tile["source_bbox_pixels"], info["source_bbox_pixels"])
        values = autonomous_plan.split_three(
            plan([{"id": "grid", "bbox": [210, 210, 610, 790]}, {"id": "b"}, {"id": "c"}]),
            json.dumps(report))
        _, crop, _, _, source = autonomous_plan.asset_reference(values[0], scene(), "test", 0)
        refined, _ = crop_reference(scene(), [.21, .21, .61, .79], 0)
        np.testing.assert_array_equal(crop, refined)
        self.assertEqual(json.loads(source)["original_description"], raw)

    def test_thousand_grid_edges_and_mixed_scales_stay_rejected(self):
        boxes = [[0, 200, 600, 800], [200, 200, 1000, 800],
                 [.2, .2, 600, 800], [0, 0, 292, .245], [200, 200, 1200, 800]]
        objects = [candidate("bad_"+str(i), bbox=box) for i, box in enumerate(boxes)]
        _, report = self.review(objects+[candidate("safe")])
        self.assertEqual(report["rejected_ids"], [o["id"] for o in objects])
        self.assertEqual([o["id"] for o in report["accepted_candidates"]], ["safe"])


@unittest.skipIf(autonomous_plan is None, "Autonomous planning API unavailable")
class SplitThreeTests(unittest.TestCase):
    def manifest(self, objects=None):
        _, manifest = autonomous_plan.candidate_review(
            plan(objects or [candidate(x) for x in ("a", "b", "c", "d", "e", "f")]), scene())
        return manifest

    def test_exactly_three_scalar_json_plans_keep_original_context_and_exclusions(self):
        manifest = self.manifest()
        result = autonomous_plan.split_three(
            plan([candidate(x) for x in ("c", "a", "b")]), manifest)
        self.assertEqual(len(result), 4)
        self.assertTrue(all(isinstance(value, str) for value in result))
        for identifier, value in zip(("c", "a", "b"), result[:3]):
            single = json.loads(value)
            self.assertEqual(len(single["objects"]), 1)
            obj = single["objects"][0]
            self.assertEqual(obj["id"], identifier)
            self.assertEqual(obj["original_description"]["appearance"],
                             candidate(identifier)["appearance"])
            self.assertEqual(obj["excluded_objects"], candidate(identifier)["excluded_objects"])
            self.assertEqual(obj["connected_parts"], candidate(identifier)["connected_parts"])
        self.assertEqual(json.loads(result[3])["selected_ids"], ["c", "a", "b"])

    def test_only_actual_safe_ids_can_be_selected_not_inward_relabelled_edge(self):
        manifest = self.manifest([candidate("edge", bbox=[0, .2, .6, .8])] +
                                 [candidate(x) for x in ("a", "b", "c", "d", "e")])
        for obj in (candidate("edge", bbox=[.1, .2, .6, .8]),
                    candidate("invented"), candidate("a", source_ids=["edge"]),
                    candidate("a", unit_id="edge")):
            with self.subTest(obj=obj), self.assertRaises(ValueError):
                autonomous_plan.split_three(plan([obj, candidate("b"), candidate("c")]),
                                            manifest)

    def test_three_distinct_ids_and_units_are_mandatory(self):
        manifest = self.manifest()
        for ids in ([], ["a"], ["a", "b"], ["a", "b", "c", "d"], ["a", "a", "b"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                autonomous_plan.split_three(plan([candidate(x) for x in ids]), manifest)
        manifest = self.manifest([candidate("a", unit_id="joined"),
                                  candidate("b", unit_id="joined"),
                                  candidate("c"), candidate("d")])
        with self.assertRaises(ValueError):
            autonomous_plan.split_three(plan([{"id": x} for x in ("a", "b", "c")]), manifest)

    def test_small_safe_bbox_refinement_is_allowed_but_retargeting_is_rejected(self):
        manifest = self.manifest()
        result = autonomous_plan.split_three(
            plan([{"id": "a", "bbox": [.21, .21, .61, .79]}, {"id": "b"}, {"id": "c"}]),
            manifest)
        self.assertEqual(json.loads(result[0])["objects"][0]["bbox"], [.21, .21, .61, .79])
        for box in ([0, .2, .6, .8], [.65, .1, .9, .3], [.3, .3, .4, .4],
                    [float("nan"), .2, .6, .8], [True, .2, .6, .8]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                autonomous_plan.split_three(
                    plan([{"id": "a", "bbox": box}, {"id": "b"}, {"id": "c"}]), manifest)

    def test_shared_connected_parts_cannot_be_lost_in_visual_review(self):
        manifest = self.manifest()
        result = autonomous_plan.split_three(
            plan([{"id": "a", "connected_parts": ["tall cap"],
                   "excluded_objects": ["separate blue hill"]}, {"id": "b"}, {"id": "c"}]),
            manifest)
        target = json.loads(result[0])["objects"][0]
        self.assertEqual(target["connected_parts"], candidate("a")["connected_parts"])
        self.assertIn("separate blue hill", target["excluded_objects"])
        self.assertIn("foreground grass tufts", target["excluded_objects"])

    def test_two_overlapping_safe_objects_remain_two_separate_targets(self):
        result = autonomous_plan.split_three(
            plan([{"id": x} for x in ("a", "b", "c")]), self.manifest())
        self.assertEqual([json.loads(v)["objects"][0]["unit_id"] for v in result[:3]],
                         ["a", "b", "c"])

    def test_review_booleans_malformed_parts_and_insufficient_manifest_fail_closed(self):
        manifest = self.manifest()
        for fields in ({"boundary_cut": "false"}, {"suitable": "yes"},
                       {"is_single": 1}, {"boundary_cut": True}, {"suitable": False},
                       {"connected_parts": "a"}, {"excluded_objects": "trees"}):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                autonomous_plan.split_three(
                    plan([{"id": "a", **fields}, {"id": "b"}, {"id": "c"}]), manifest)
        manifest = self.manifest([candidate("a"), candidate("b"),
                                  candidate("edge", bbox=[0, .2, .6, .8])])
        with self.assertRaisesRegex(ValueError, "insufficient_safe_candidates"):
            autonomous_plan.split_three(plan([{"id": x} for x in ("a", "b", "c")]), manifest)

    def test_fenced_model_json_is_supported_without_mutating_source(self):
        manifest = self.manifest()
        before = manifest
        result = autonomous_plan.split_three(
            "Visual review:\n```json\n" + plan([{"id": x} for x in ("a", "b", "c")]) +
            "\n```", manifest)
        self.assertEqual(manifest, before)
        self.assertEqual(json.loads(result[3])["status"], "ready")

    def test_exclusions_cannot_contradict_connected_parts_or_own_candidate_ids(self):
        manifest = self.manifest()
        for exclusions in (["attached low cap"], [{"id": "a", "name": "The whole target"}]):
            with self.subTest(exclusions=exclusions), self.assertRaises(ValueError):
                autonomous_plan.split_three(
                    plan([{"id": "a", "excluded_objects": exclusions},
                          {"id": "b"}, {"id": "c"}]), manifest)

    def test_source_margin_cannot_be_weakened_and_manifest_ids_must_be_arrays(self):
        _, manifest = autonomous_plan.candidate_review(
            plan([candidate("near_edge", bbox=[.02, .2, .6, .8])] +
                 [candidate(x) for x in ("a", "b", "c")]), scene(), .03)
        with self.assertRaises(ValueError):
            autonomous_plan.split_three(
                plan([{"id": x} for x in ("near_edge", "a", "b")]), manifest, 0)
        manifest = json.loads(self.manifest())
        for ids in ("not-an-array", [None], [["a"]]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                autonomous_plan.split_three(
                    plan([{"id": x} for x in ("a", "b", "c")]),
                    json.dumps({**manifest, "rejected_ids": ids}))

    def test_shared_unit_alias_keeps_every_original_source_description(self):
        objects = [
            candidate("cap", unit_id="whole", bbox=[.3, .1, .5, .6],
                      connected_parts=["tall cap"], appearance="Original tall orange tower",
                      excluded_objects=[{"id": "occluder", "name": "Blue distant ridge",
                                         "location": "Behind the attached cap"}]),
            candidate("base", unit_id="whole", bbox=[.2, .4, .7, .85],
                      connected_parts=["attached low cap", "shared stone shelf"],
                      appearance="Original low shelf across the tower foot"),
            candidate("b"), candidate("c")]
        result = autonomous_plan.split_three(
            plan([{"id": x} for x in ("base", "b", "c")]), self.manifest(objects))
        one = json.loads(result[0])["objects"][0]
        self.assertEqual(one["source_ids"], ["cap", "base"])
        self.assertEqual(one["bbox"], [.2, .1, .7, .85])
        prompt, _, _, _, record = autonomous_plan.asset_reference(result[0], scene(), "test")
        for phrase in ("Original tall orange tower", "Original low shelf across the tower foot",
                       "Blue distant ridge", "Behind the attached cap", "shared stone shelf"):
            self.assertIn(phrase, prompt)
        self.assertEqual(len(json.loads(record)["original_candidates"]), 2)


@unittest.skipIf(autonomous_plan is None, "Autonomous planning API unavailable")
class AssetReferenceTests(unittest.TestCase):
    def single(self):
        _, manifest = autonomous_plan.candidate_review(
            plan([candidate(x) for x in ("a", "b", "c")]), scene())
        return autonomous_plan.split_three(
            plan([{"id": x} for x in ("a", "b", "c")]), manifest)[0]

    def test_real_current_source_crop_key_filename_and_record_types(self):
        image = scene()
        before = image.copy()
        value = self.single()
        prompt, crop, key, prefix, record_text = autonomous_plan.asset_reference(
            value, image, "autonomous/run_9", 0)
        self.assertTrue(all(isinstance(v, str) for v in (prompt, key, prefix, record_text)))
        expected, info = crop_reference(image, candidate("a")["bbox"], 0)
        np.testing.assert_array_equal(crop, expected)
        self.assertEqual(crop.dtype, np.float32)
        self.assertIn(key, ("#00FF00", "#FF00FF", "#0000FF"))
        self.assertTrue(prefix.startswith("autonomous/run_9/midground/a_"))
        record = json.loads(record_text)
        self.assertEqual(record["source_bbox_pixels"], info["source_bbox_pixels"])
        self.assertEqual(record["original_description"], candidate("a"))
        self.assertEqual(record["connected_parts"], candidate("a")["connected_parts"])
        self.assertFalse(record["rescaled"])
        np.testing.assert_array_equal(image, before)

    def test_prompt_preserves_original_full_context_parts_exclusions_base_and_canvas(self):
        value = json.loads(self.single())
        value["objects"][0]["appearance"] = "Review observes a chipped rim"
        prompt, _, key, _, _ = autonomous_plan.asset_reference(
            json.dumps(value), scene(), "test")
        for phrase in (candidate("a")["appearance"], candidate("a")["location"],
                       candidate("a")["occlusion"], candidate("a")["selection_reason"],
                       "tall cap", "attached low cap", "shared stone shelf",
                       "separate distant ridge", "foreground grass tufts",
                       "short", "natural", "complete", "15%", "70%", key,
                       "Review observes a chipped rim"):
            self.assertIn(phrase, prompt)

    def test_context_uses_existing_crop_helper_and_key_avoids_green_subject(self):
        green = np.broadcast_to([.1, .8, .1], scene().shape).copy().astype(np.float32)
        prompt, crop, key, _, record = autonomous_plan.asset_reference(
            self.single(), green, "test", .025)
        expected, _ = crop_reference(green, candidate("a")["bbox"], .025)
        np.testing.assert_array_equal(crop, expected)
        self.assertNotEqual(key, "#00FF00")
        self.assertIn(key, prompt)
        self.assertEqual(json.loads(record)["key_color"], key)

    def test_rejects_multi_target_bad_context_prefix_and_nonfinite_image(self):
        for text in (plan([]), plan([candidate("a"), candidate("b")])):
            with self.subTest(text=text), self.assertRaises(ValueError):
                autonomous_plan.asset_reference(text, scene(), "test")
        for context in (True, float("nan"), float("inf"), -.01, .16, "0.025"):
            with self.subTest(context=context), self.assertRaises(ValueError):
                autonomous_plan.asset_reference(self.single(), scene(), "test", context)
        for prefix in ("", "  ", None):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                autonomous_plan.asset_reference(self.single(), scene(), prefix)
        with self.assertRaises(ValueError):
            autonomous_plan.asset_reference(
                self.single(), np.full((1, 10, 10, 3), np.inf), "test")

    def test_original_parts_and_exclusions_survive_refined_plan_omissions(self):
        single = json.loads(self.single())
        single["objects"][0]["connected_parts"] = ["tall cap"]
        single["objects"][0]["excluded_objects"] = []
        prompt, _, _, _, record = autonomous_plan.asset_reference(
            json.dumps(single), scene(), "test")
        value = json.loads(record)
        self.assertEqual(value["connected_parts"], candidate("a")["connected_parts"])
        self.assertEqual(value["excluded_objects"], candidate("a")["excluded_objects"])
        self.assertIn("separate distant ridge", prompt)

    def test_original_identity_and_original_bbox_cannot_be_substituted(self):
        for originals in ([None], [candidate("different")],
                          [candidate("a", unit_id="different")]):
            single = json.loads(self.single())
            single["objects"][0]["original_candidates"] = originals
            with self.subTest(originals=originals), self.assertRaises(ValueError):
                autonomous_plan.asset_reference(json.dumps(single), scene(), "test")
        single = json.loads(self.single())
        single["objects"][0]["bbox"] = [.7, .1, .9, .3]
        with self.assertRaises(ValueError):
            autonomous_plan.asset_reference(json.dumps(single), scene(), "test")

    def test_rgba_crop_is_an_independent_copy_and_wrong_reference_size_is_rejected(self):
        image = np.concatenate((scene(), np.full((1, 120, 240, 1), .7, np.float32)), axis=-1)
        before = image.copy()
        _, crop, _, _, _ = autonomous_plan.asset_reference(self.single(), image, "test", 0)
        self.assertEqual(crop.shape[-1], 4)
        crop[:] = 0
        np.testing.assert_array_equal(image, before)
        with self.assertRaises(ValueError):
            autonomous_plan.asset_reference(self.single(), scene()[:, :100], "test")


if __name__ == "__main__":
    unittest.main(verbosity=2)
