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


class DiscoveryCacheReuseTest(unittest.TestCase):
    def tearDown(self):
        create_import_structure_from_path.cache_clear()

    def test_parent_discovery_retains_a_broad_package_tree(self):
        # Nested lazy modules should reuse the package tree discovered by their parent.
        with tempfile.TemporaryDirectory() as root:
            for index in range(140):
                package = Path(root) / f"package_{index}"
                package.mkdir()
                (package / "configuration_example.py").write_text('__all__ = ["Example"]\n', encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            parent = create_import_structure_from_path(root)
            before = create_import_structure_from_path.cache_info()
            for name, structure in parent.items():
                self.assertIs(create_import_structure_from_path(str(Path(root) / name)), structure)
                self.assertEqual(structure, {frozenset(): {"configuration_example": {"Example"}}})
            after = create_import_structure_from_path.cache_info()
            self.assertEqual(after.misses, before.misses)
            self.assertEqual(after.hits - before.hits, 140)
            self.assertLessEqual(after.currsize, after.maxsize)

    def test_explicit_clear_rediscovers_changed_exports(self):
        with tempfile.TemporaryDirectory() as root:
            source = Path(root) / "configuration_example.py"
            source.write_text('__all__ = ["Before"]\n', encoding="utf-8")
            create_import_structure_from_path.cache_clear()
            before = create_import_structure_from_path(root)
            source.write_text('__all__ = ["After"]\n', encoding="utf-8")
            self.assertIs(create_import_structure_from_path(root), before)
            create_import_structure_from_path.cache_clear()
            self.assertEqual(
                create_import_structure_from_path(root),
                {frozenset(): {"configuration_example": {"After"}}},
            )


if __name__ == "__main__":
    unittest.main()
