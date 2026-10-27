import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

if importlib.util.find_spec("compact_native"):
    import compact_native as adapter
else:
    adapter = None


class Availability(unittest.TestCase):
    def test_native_adapter_exists(self):
        self.assertIsNotNone(adapter, "Compact native adapter is missing.")


@unittest.skipIf(adapter is None, "Compact native adapter is missing.")
class NativeContract(unittest.TestCase):
    def test_calls_installed_function_with_list_inputs_and_parent_task(self):
        observed = {}

        class Native:
            INPUT_IS_LIST = True
            FUNCTION = "run_with_tea"

            def run_with_tea(self, **kw):
                observed.update(kw)
                observed["node_info"] = self.node_info
                return ("actual native response",)

        with TemporaryDirectory() as directory:
            file = Path(directory)/"reference.png"
            file.write_bytes(b"actual file")
            result = adapter.invoke(
                {"BALLMText": Native}, "BALLMText",
                {"node_id": "2", "prompt_id": "task", "class_type": "Compact"},
                "2", {"2": {"class_type": "Compact"}},
                {"model": "native-config", "user_prompt": "observe image",
                 "extra_inputs": {"max_tokens": 6000}},
                [str(file)], ["target record"])
        self.assertEqual(result, ("actual native response",))
        self.assertEqual(observed["image0"], [str(file)])
        self.assertEqual(observed["text0"], ["target record"])
        self.assertEqual(observed["model"], ["native-config"])
        self.assertEqual(observed["extra_inputs"], [{"max_tokens": 6000}])
        self.assertEqual(observed["prompt"][0]["2"]["class_type"], "Compact")
        self.assertEqual(observed["node_info"]["prompt_id"], "task")
        self.assertEqual(observed["node_info"]["node_id"], "2")

    def test_invalid_image_path_fails_before_service_call(self):
        with self.assertRaisesRegex(ValueError, "image file"):
            adapter.invoke({}, "BALLMImage", {}, "2", {},
                           {"model": "x"}, ["missing-reference.png"], [])

    def test_missing_native_class_is_explicit_error(self):
        with self.assertRaisesRegex(RuntimeError, "BALLMText"):
            adapter.invoke({}, "BALLMText", {}, "2", {}, {"model": "x"}, [], [])

    def test_does_not_swallow_native_error_or_repeat_request(self):
        class Broken:
            INPUT_IS_LIST = True
            FUNCTION = "run"

            def run(self, **kw):
                raise RuntimeError("account request denied")

        with self.assertRaisesRegex(RuntimeError, "account request denied"):
            adapter.invoke({"BALLMText": Broken}, "BALLMText", {}, "2", {},
                           {"model": "x"}, [], [])

    def test_invalid_json_requests_one_format_correction(self):
        self.assertTrue(callable(getattr(adapter, "request_json", None)),
                        "Automatic JSON format correction is missing.")
        feedbacks = []
        replies = iter(['{"asset_id":"c02","visible_polygon":[[100,200],…]}',
                        '{"asset_id":"c02","visible_polygon":[[100,200],[200,100],[300,200]]}'])

        def request(feedback):
            feedbacks.append(feedback)
            return next(replies)

        result = adapter.request_json(request)
        self.assertIn('"asset_id":"c02"', result)
        self.assertEqual(len(feedbacks), 2)
        self.assertEqual(feedbacks[0], "")
        self.assertIn("JSON", feedbacks[1])
        self.assertIn("visible_polygon", feedbacks[1])

    def test_json_retry_is_bounded_and_never_accepts_invalid_reply(self):
        self.assertTrue(callable(getattr(adapter, "request_json", None)),
                        "Automatic JSON format correction is missing.")
        attempts = []

        def request(feedback):
            attempts.append(feedback)
            return '{"visible_polygon":[[100,200],…]}'

        with self.assertRaisesRegex(ValueError, "JSON"):
            adapter.request_json(request)
        self.assertEqual(len(attempts), 2)

    def test_valid_json_does_not_trigger_an_extra_model_call(self):
        self.assertTrue(callable(getattr(adapter, "request_json", None)),
                        "Automatic JSON format correction is missing.")
        calls = []

        def request(feedback):
            calls.append(feedback)
            return '{"score":95,"issues":[]}'

        self.assertEqual(adapter.request_json(request), '{"score":95,"issues":[]}')
        self.assertEqual(calls, [""])

    def test_json_correction_does_not_retry_service_failure(self):
        self.assertTrue(callable(getattr(adapter, "request_json", None)),
                        "Automatic JSON format correction is missing.")
        calls = []

        def request(feedback):
            calls.append(feedback)
            raise RuntimeError("native account error")

        with self.assertRaisesRegex(RuntimeError, "native account error"):
            adapter.request_json(request)
        self.assertEqual(calls, [""])


if __name__ == "__main__":
    unittest.main()
