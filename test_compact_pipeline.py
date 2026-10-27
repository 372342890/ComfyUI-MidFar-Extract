import importlib.util
import json
import unittest

import numpy as np
from test_autonomous_audit import generated, reference, plan, verdict
from autonomous_audit import inspect_asset

if importlib.util.find_spec("compact_pipeline"):
    import compact_pipeline as pipeline
else:
    pipeline = None


class Availability(unittest.TestCase):
    def test_compact_api_exists(self):
        self.assertIsNotNone(pipeline, "Compact pipeline is missing.")


@unittest.skipIf(pipeline is None, "Compact pipeline is missing.")
class ConditionalRepair(unittest.TestCase):
    def execute(self, reports):
        calls = []
        raw = generated()

        def generate(version, previous, feedback):
            calls.append(("generate", version))
            if version > 1:
                self.assertIsNotNone(previous)
                self.assertIn("blocking", feedback)
            return raw

        def review(sheet, prompt, version):
            calls.append(("review", version))
            return json.dumps(reports[version-1])

        result = pipeline.extract_until_accepted(
            generate, review, reference(), plan(), "#00FF00")
        return calls, result

    def test_accepted_first_version_makes_no_repair_calls(self):
        calls, (rgba, record) = self.execute([verdict()])
        self.assertEqual(calls, [("generate", 1), ("review", 1)])
        self.assertEqual(json.loads(record)["attempts"], 1)
        self.assertEqual(rgba.shape[-1], 4)

    def test_first_failed_then_second_accepted_stops_before_third(self):
        calls, (_, record) = self.execute([
            verdict(natural_base=False, issues=["blocking flat foot"]), verdict()])
        self.assertEqual(calls, [("generate", 1), ("review", 1),
                                 ("generate", 2), ("review", 2)])
        self.assertEqual(json.loads(record)["attempts"], 2)
        self.assertEqual(len(json.loads(record)["visual_reports"]), 2)

    def test_three_failed_versions_raise_with_real_last_feedback(self):
        with self.assertRaisesRegex(ValueError, "No eligible asset"):
            self.execute([verdict(target_match=False, issues=["blocking wrong target"])]*3)

    def test_generation_service_error_is_not_quality_retry(self):
        calls = []

        def broken(version, previous, feedback):
            calls.append(version)
            raise RuntimeError("native service unavailable")

        with self.assertRaisesRegex(RuntimeError, "native service unavailable"):
            pipeline.extract_until_accepted(broken, lambda *a: "", reference(),
                                            plan(), "#00FF00")
        self.assertEqual(calls, [1])

    def test_numeric_failure_cannot_pass_even_with_perfect_visual_report(self):
        calls = []

        def clipped(version, previous, feedback):
            calls.append(version)
            return generated(cut=True)

        with self.assertRaisesRegex(ValueError, "No eligible asset"):
            pipeline.extract_until_accepted(clipped, lambda *a: json.dumps(verdict()),
                                            reference(), plan(), "#00FF00")
        self.assertEqual(calls, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
