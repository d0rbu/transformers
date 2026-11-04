import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers.utils.import_utils import create_import_structure_from_path


class ImportStructureAllScanTest(unittest.TestCase):
    def parse(self, content):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "modeling_example.py").write_text(content, encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            return create_import_structure_from_path(root)

    def test_no_all_marker_keeps_empty_structure(self):
        self.assertEqual(self.parse("class Ordinary:\n    pass\n"), {})

    def test_last_top_level_all_assignment_keeps_existing_precedence(self):
        actual = self.parse(
            '__all__ = ["Earlier"]\nclass Ordinary:\n    note = "__all__"\n'
            '__all__ = ["Later"]\n# another mention of __all__\n'
        )
        self.assertEqual(actual, {frozenset({"torch"}): {"modeling_example": {"Later"}}})

    def test_multiline_all_with_crlf_keeps_exports(self):
        actual = self.parse('__all__ = [\r\n    "First",\r\n    "Second",\r\n]\r\n')
        self.assertEqual(actual, {frozenset({"torch"}): {"modeling_example": {"First", "Second"}}})

    def test_malformed_assignment_keeps_index_error(self):
        with self.assertRaises(IndexError):
            self.parse('__all__ ["Broken"]\n')

    def test_decorated_export_keeps_required_backend(self):
        actual = self.parse(
            '@requires(backends=("vision",))\nclass Decorated:\n    pass\n__all__ = ["Ordinary", "Decorated"]\n'
        )
        self.assertEqual(
            actual,
            {
                frozenset({"torch", "vision"}): {"modeling_example": {"Decorated"}},
                frozenset({"torch"}): {"modeling_example": {"Ordinary"}},
            },
        )

    def test_string_subclass_preserves_contains_and_splitlines_dispatch(self):
        calls = []

        class Content(str):
            def __contains__(self, value):
                calls.append(("contains", value))
                return super().__contains__(value)

            def splitlines(self, keepends=False):
                calls.append(("splitlines", keepends))
                return super().splitlines(keepends)

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
                actual = create_import_structure_from_path(root)
        self.assertEqual(actual, {frozenset({"torch"}): {"modeling_example": {"Ordinary"}}})
        self.assertEqual(
            calls, [("contains", "@requires"), ("contains", "__all__"), ("contains", "__all__"), ("splitlines", False)]
        )


if __name__ == "__main__":
    unittest.main()
