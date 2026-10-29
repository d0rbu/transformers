# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from transformers.utils.import_utils import fetch__all__


class FetchAllSuffixTest(unittest.TestCase):
    def test_last_declaration_after_each_supported_line_separator(self):
        separators = ("\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
        for left in separators:
            for right in separators:
                with self.subTest(left=repr(left), right=repr(right)):
                    text = f'__all__ = ["Earlier"]{left}noise{right}__all__ = ["Last"]{left}'
                    self.assertEqual(fetch__all__(text), ["Last"])

    def test_multiline_declaration_and_non_declarations(self):
        self.assertEqual(fetch__all__('noise\n__all__ = [\n    "First",\n    "Second",\n]\n'), ["First", "Second"])
        self.assertEqual(fetch__all__('  __all__ = ["Indented"]\nx__all__ = ["Suffix"]\n'), [])
        self.assertEqual(fetch__all__('__all__ = ["First"]\n# __all__ comment\n'), ["First"])

    def test_legacy_malformed_final_declaration_errors(self):
        with self.assertRaises(IndexError):
            fetch__all__('__all__ = ["First"]\n__all__ ]')
        self.assertEqual(fetch__all__('__all__ = [\n    "Unclosed",\n'), ["Unclosed"])

    def test_external_string_subclass_keeps_method_dispatch(self):
        class CustomLines(str):
            def splitlines(self, *args, **kwargs):
                return ['__all__ = ["Override"]']

        self.assertEqual(fetch__all__(CustomLines('__all__ = ["Ignored"]')), ["Override"])


if __name__ == "__main__":
    unittest.main()
