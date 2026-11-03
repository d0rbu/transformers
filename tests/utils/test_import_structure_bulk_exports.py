import tempfile
import unittest
from pathlib import Path

from transformers.utils.import_utils import create_import_structure_from_path


class BulkExportsTest(unittest.TestCase):
    def parse(self, contents):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "modeling_example.py"
            path.write_text(contents, encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            return create_import_structure_from_path(root)

    def test_decorated_names_keep_their_backend_group(self):
        actual = self.parse(
            '@requires(backends=("vision",))\n'
            "class Decorated:\n    pass\n"
            '__all__ = ["Ordinary", "Decorated", "Ordinary"]\n'
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

    def test_matching_backend_declarations_merge_names(self):
        actual = self.parse(
            '@requires(backends=("torch",))\n'
            "def decorated():\n    pass\n"
            '__all__ = ["ordinary", "decorated", "ordinary"]\n'
        )
        self.assertEqual(
            actual,
            {frozenset({"torch"}): {"modeling_example": {"decorated", "ordinary"}}},
        )

    def test_already_decorated_exports_do_not_add_an_empty_group(self):
        actual = self.parse(
            '@requires(backends=("vision",))\nclass Decorated:\n    pass\n__all__ = ["Decorated", "Decorated"]\n'
        )
        self.assertEqual(
            actual,
            {frozenset({"torch", "vision"}): {"modeling_example": {"Decorated"}}},
        )


if __name__ == "__main__":
    unittest.main()
