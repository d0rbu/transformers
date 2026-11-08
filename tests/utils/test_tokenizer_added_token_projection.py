"""Current native token-table reads and original JSON fallback semantics."""

import io
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch


try:
    import pydantic_core
except ImportError:
    pydantic_core = None

from transformers import tokenization_utils_base as module


class AddedTokenProjectionTest(unittest.TestCase):
    def read(self, payload):
        return module._read_tokenizer_added_tokens(io.StringIO(payload))

    def test_current_descriptors_are_fresh_and_independent(self):
        token = {"id": 7, "content": "雪", "special": True}
        for flag in ["normalized", "lstrip", "rstrip", "single_word"]:
            for value in [False, True]:
                token[flag] = value
                payload = json.dumps({"model": {"unused": [1, 2, 3]}, "added_tokens": [token]})
                actual = self.read(payload)
                self.assertEqual(actual, [token])
                actual[0]["content"] = "changed result"
                self.assertEqual(self.read(payload), [token])

    @unittest.skipUnless(pydantic_core is not None, "optional field parser is unavailable")
    def test_native_table_does_not_materialize_vocabulary_with_stdlib(self):
        local_json = types.SimpleNamespace(loads=Mock(wraps=json.loads))
        payload = '{"model":{"vocab":{"雪":1}},"added_tokens":[{"id":1,"content":"雪","special":true}]}'
        with patch.object(module, "json", local_json):
            self.assertEqual(self.read(payload), [{"id": 1, "content": "雪", "special": True}])
        local_json.loads.assert_not_called()

    def test_optional_parser_absence_and_partial_initialization_use_stdlib(self):
        payload = '{"added_tokens":[{"id":7,"content":"x"}]}'
        for core in [None, types.ModuleType("partial-pydantic-core")]:
            with self.subTest(core=core), patch.dict(sys.modules, {"pydantic_core": core}):
                self.assertEqual(self.read(payload), json.loads(payload)["added_tokens"])

    def test_unusual_values_keep_stdlib_values(self):
        payloads = [
            '{"added_tokens":null}',
            '{"added_tokens":42}',
            '{"added_tokens":[{"id":1.234567890123456789,"content":"x"}]}',
            '{"added_tokens":[{"id":7,"content":"x","future":{"value":1}}]}',
            '{"added_tokens":[{"content":"\\ud800"}]}',
            '{"added_tokens":[-0.0,1e400,-1e400]}',
            '{"added_tokens":[1],"added_tokens":[2]}',
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                self.assertEqual(self.read(payload), json.loads(payload)["added_tokens"])

    @unittest.skipUnless(pydantic_core is not None, "optional field parser is unavailable")
    def test_float_values_use_the_original_rounding_path(self):
        payload = '{"added_tokens":[{"id":1.234567890123456789}]}'
        local_json = types.SimpleNamespace(loads=Mock(wraps=json.loads))
        with patch.object(module, "json", local_json):
            self.assertEqual(self.read(payload), json.loads(payload)["added_tokens"])
        local_json.loads.assert_called_once_with(payload)

    def test_original_errors_preserve_stdlib_context(self):
        payloads = [
            "{}",
            "[]",
            "null",
            "42",
            '{"unused":"\\uZZZZ","added_tokens":[]}',
            '{"unused":01,"added_tokens":[]}',
            '{"added_tokens":[',
            '\ufeff{"added_tokens":[]}',
            '{"added_tokens":[' + "9" * 5000 + "]}",
        ]
        for payload in payloads:
            with self.subTest(payload=payload):
                try:
                    json.loads(payload).pop("added_tokens")
                except Exception as expected:
                    expected_type, expected_message = type(expected), str(expected)
                    expected_context = expected.__context__
                    expected_cause = expected.__cause__
                else:
                    self.fail("control input did not produce its original error")
                with self.assertRaises(expected_type) as actual:
                    self.read(payload)
                self.assertEqual(str(actual.exception), expected_message)
                for observed, reference in [
                    (actual.exception.__context__, expected_context),
                    (actual.exception.__cause__, expected_cause),
                ]:
                    self.assertIs(type(observed), type(reference))
                    self.assertEqual(str(observed), str(reference))


if __name__ == "__main__":
    unittest.main()
