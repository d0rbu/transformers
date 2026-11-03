import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transformers.utils.import_utils import create_import_structure_from_path


class ImportStructureLineFilterTest(unittest.TestCase):
    def parse(self, contents):
        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "modeling_example.py").write_text(contents, encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            return create_import_structure_from_path(root)

    def test_multiline_decorators_and_indented_declarations_keep_export_order(self):
        actual = self.parse(
            "@requires(\n"
            "    backends=(\n"
            '        "vision",\n'
            "    )\n"
            ")\n"
            "@other\n"
            "\n"
            "# intervening comment\n"
            "class Decorated:\n"
            '    @requires(backends=("audio",))\n'
            "    def nested(self):\n"
            "        pass\n"
            '__all__ = ["Ordinary", "Decorated"]\n'
        )
        vision, torch = frozenset({"torch", "vision"}), frozenset({"torch"})
        self.assertEqual(
            actual,
            {
                vision: {"modeling_example": {"Decorated"}},
                torch: {"modeling_example": {"Ordinary"}},
            },
        )
        self.assertEqual(list(actual), [vision, torch])

    def test_supplied_string_subclasses_keep_their_startswith_dispatch(self):
        calls = []

        class Line(str):
            def startswith(self, *args, **kwargs):
                calls.append(args[0])
                return super().startswith(*args, **kwargs)

        class Content(str):
            def split(self, sep=None, maxsplit=-1):
                return [Line(line) for line in super().split(sep, maxsplit)]

        class Reader:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return Content('@requires(backends=("vision",))\nclass Decorated:\n    pass\n')

        with tempfile.TemporaryDirectory() as root:
            (Path(root) / "modeling_example.py").touch()
            create_import_structure_from_path.cache_clear()
            with patch("builtins.open", return_value=Reader()):
                actual = create_import_structure_from_path(root)
        self.assertEqual(actual, {frozenset({"torch", "vision"}): {"modeling_example": {"Decorated"}}})
        self.assertIn((" ", "\t", "@", ")"), calls)

    def test_malformed_backend_declaration_keeps_its_failure(self):
        with self.assertRaisesRegex(
            ValueError, "Couldn't parse backends for @requires decorator in file modeling_example"
        ):
            self.parse("@requires(backends=broken)\nclass Broken:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
