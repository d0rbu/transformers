import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers.utils.import_utils import create_import_structure_from_path


class RequiresPresenceSearchTest(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "modeling_example.py").write_text(content, encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            return create_import_structure_from_path(root)

    def test_absent_marker_keeps_ordinary_exports(self):
        self.assertEqual(
            self.parse('__all__ = ["Ordinary"]\n'),
            {frozenset({"torch"}): {"modeling_example": {"Ordinary"}}},
        )

    def test_present_marker_keeps_required_backend(self):
        self.assertEqual(
            self.parse('@requires(backends=("vision",))\nclass Exported:\n    pass\n__all__ = ["Exported"]\n'),
            {frozenset({"torch", "vision"}): {"modeling_example": {"Exported"}}},
        )

    def test_malformed_marker_keeps_failure(self):
        with self.assertRaisesRegex(ValueError, "Couldn't parse backends"):
            self.parse("@requires(backends=broken)\nclass Exported:\n")

    def test_trailing_all_is_read_after_required_definition(self):
        for definition in ("class Exported:", "def Exported():"):
            with self.subTest(definition=definition):
                self.assertEqual(
                    self.parse(
                        '@requires(backends=("vision",))\n'
                        + definition
                        + '\n    pass\n\nclass Ordinary:\n    pass\n__all__ = ["Exported", "Ordinary"]\n'
                    ),
                    {
                        frozenset({"torch", "vision"}): {"modeling_example": {"Exported"}},
                        frozenset({"torch"}): {"modeling_example": {"Ordinary"}},
                    },
                )

    def test_required_definition_without_final_newline(self):
        self.assertEqual(
            self.parse('@requires(backends=("vision",))\ndef Exported():'),
            {frozenset({"torch", "vision"}): {"modeling_example": {"Exported"}}},
        )

    def test_ignored_trailing_marker_keeps_all_exports(self):
        for tail in ("    # @requires\n", "# @requires\n", 'text = "@requires"\n'):
            with self.subTest(tail=tail):
                self.assertEqual(
                    self.parse(
                        '@requires(backends=("vision",))\nclass Exported:\n    pass\n'
                        + tail
                        + '__all__ = ["Exported", "Ordinary"]\n'
                    ),
                    {
                        frozenset({"torch", "vision"}): {"modeling_example": {"Exported"}},
                        frozenset({"torch"}): {"modeling_example": {"Ordinary"}},
                    },
                )

    def test_marker_without_following_definition(self):
        self.assertEqual(
            self.parse('@requires(backends=("vision",))\n'),
            {frozenset({"torch", "vision"}): {"modeling_example": set()}},
        )

    def test_subclass_split_dispatch_with_present_marker(self):
        calls = []

        class Content(str):
            def find(self, *args):
                raise AssertionError("subclass find must not be called")

            def rfind(self, *args):
                raise AssertionError("subclass rfind must not be called")

            def split(self, separator=None, maxsplit=-1):
                calls.append((separator, maxsplit))
                return super().split(separator, maxsplit)

        class Reader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return Content('@requires(backends=("vision",))\nclass Exported:\n    pass\n')

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "modeling_example.py").touch()
            create_import_structure_from_path.cache_clear()
            with patch("builtins.open", return_value=Reader()):
                self.assertEqual(
                    create_import_structure_from_path(root),
                    {frozenset({"torch", "vision"}): {"modeling_example": {"Exported"}}},
                )
        self.assertEqual(calls, [("\n", -1)])

    def test_subclass_contains_dispatch_and_exception_are_preserved(self):
        for raises in (False, True):
            with self.subTest(raises=raises):
                calls = []

                class Content(str):
                    def __contains__(self, value):
                        calls.append(value)
                        if raises and value == "@requires":
                            raise RuntimeError("contains sentinel")
                        return super().__contains__(value)

                    def rfind(self, *args):
                        raise AssertionError("subclass rfind must not be called")

                class Reader:
                    def __enter__(self):
                        return self

                    def __exit__(self, *args):
                        return False

                    def read(self):
                        return Content('__all__ = ["Ordinary"]\n')

                with tempfile.TemporaryDirectory() as root:
                    (Path(root) / "modeling_example.py").touch()
                    create_import_structure_from_path.cache_clear()
                    with patch("builtins.open", return_value=Reader()):
                        if raises:
                            with self.assertRaisesRegex(RuntimeError, "contains sentinel"):
                                create_import_structure_from_path(root)
                        else:
                            self.assertEqual(
                                create_import_structure_from_path(root),
                                {frozenset({"torch"}): {"modeling_example": {"Ordinary"}}},
                            )
                self.assertEqual(calls, ["@requires"] if raises else ["@requires", "__all__", "__all__"])
