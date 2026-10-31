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

from transformers.utils.import_utils import spread_import_structure


class ImportStructurePropagationTest(unittest.TestCase):
    def test_group_and_module_order_preserves_mutable_leaf_aliases(self):
        plain, torch = frozenset(), frozenset({"torch"})
        first, second = {"First"}, {"Second"}
        tree = {
            "one": {torch: {"first": first}, plain: {"second": first}},
            "two": {plain: {"third": second}, torch: {"fourth": second}},
        }
        result = spread_import_structure(tree)
        self.assertEqual(list(result), [torch, plain])
        self.assertEqual(list(result[torch]), ["one.first", "two.fourth"])
        self.assertEqual(list(result[plain]), ["one.second", "two.third"])
        self.assertIs(result[torch]["one.first"], first)
        self.assertIs(result[plain]["one.second"], first)
        self.assertIs(result[torch]["two.fourth"], second)
        self.assertIs(result[plain]["two.third"], second)
        first.add("Added")
        self.assertEqual(result[plain]["one.second"], {"First", "Added"})
        self.assertEqual(list(tree), ["one", "two"])
        self.assertIs(tree["one"][torch]["first"], first)

    def test_six_pass_depth_limit_and_original_input_are_preserved(self):
        backend = frozenset({"torch"})
        for depth in range(10):
            with self.subTest(depth=depth):
                leaf = {"Export"}
                tree = {backend: {"module": leaf}}
                for level in range(depth):
                    tree = {f"level{level}": tree}
                result = spread_import_structure(tree)
                module = ".".join([*(f"level{i}" for i in reversed(range(depth))), "module"])
                self.assertEqual(result, {backend: {module: leaf}} if depth <= 6 else {})
                if depth <= 6:
                    self.assertIs(result[backend][module], leaf)
                original = tree
                for level in reversed(range(depth)):
                    self.assertEqual(list(original), [f"level{level}"])
                    original = original[f"level{level}"]
                self.assertIs(original[backend]["module"], leaf)

    def test_empty_groups_and_unconditional_entries_keep_original_behavior(self):
        backend = frozenset()
        tree = {"unconditional": {"module": {"Ignored"}}, backend: {}}
        self.assertEqual(spread_import_structure(tree), {backend: {}})
        self.assertEqual(list(tree), ["unconditional", backend])
        self.assertEqual(tree["unconditional"], {"module": {"Ignored"}})
        self.assertEqual(tree[backend], {})

    def test_invalid_nested_exports_keep_the_original_failure(self):
        backend = frozenset()
        tree = {"outer": {backend: {"bad"}}}
        with self.assertRaisesRegex(ValueError, "dictionary update sequence element"):
            spread_import_structure(tree)
        self.assertEqual(tree, {"outer": {backend: {"bad"}}})


if __name__ == "__main__":
    unittest.main()
