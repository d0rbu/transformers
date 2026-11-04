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

import importlib.metadata
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock, patch

from transformers.utils import import_utils


# The integrated fork owns an equivalent one-name-read map builder. Mock its
# producer seam while retaining the original map identity, ordering and errors.
_MISSING = object()


class PackageDistributionDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.original_mapping = import_utils.__dict__.pop("PACKAGE_DISTRIBUTION_MAPPING", _MISSING)

    def tearDown(self):
        import_utils.__dict__.pop("PACKAGE_DISTRIBUTION_MAPPING", None)
        if self.original_mapping is not _MISSING:
            import_utils.PACKAGE_DISTRIBUTION_MAPPING = self.original_mapping

    def test_declared_matching_names_keep_version_without_full_discovery(self):
        for package, name in [("torch", "torch"), ("foo_bar", "foo-bar")]:
            with self.subTest(package=package, name=name):
                distribution = SimpleNamespace(metadata={"Name": name}, read_text=lambda _: f"other\n{package}\n")
                with (
                    patch.object(import_utils.importlib.util, "find_spec", return_value=object()),
                    patch.object(importlib.metadata, "distribution", return_value=distribution) as find,
                    patch.object(importlib.metadata, "version", return_value="1.2.3") as version,
                    patch.object(
                        import_utils, "_get_package_distribution_mapping", side_effect=AssertionError("full scan")
                    ),
                ):
                    self.assertEqual(import_utils._is_package_available(package, True), (True, "1.2.3"))
                    find.assert_called_once_with(package.replace("_", "-"))
                    version.assert_called_once_with(name)
                    self.assertNotIn("PACKAGE_DISTRIBUTION_MAPPING", import_utils.__dict__)

    def test_undeclared_or_differently_named_distribution_keeps_complete_resolution(self):
        for name, declared in [
            ("foo-bar", None),
            ("foo-bar", "another"),
            ("another", "foo_bar"),
            ("foo_bar", "foo_bar"),
        ]:
            for owners, chosen in [
                (["other", "foo_bar", "foo-bar"], "foo-bar"),
                (["other", "foo_bar"], "foo_bar"),
                (["other", "third"], "other"),
            ]:
                with self.subTest(name=name, declared=declared, owners=owners):
                    import_utils.__dict__.pop("PACKAGE_DISTRIBUTION_MAPPING", None)
                    distribution = SimpleNamespace(metadata={"Name": name}, read_text=lambda _: declared)
                    with (
                        patch.object(import_utils.importlib.util, "find_spec", return_value=object()),
                        patch.object(importlib.metadata, "distribution", return_value=distribution),
                        patch.object(importlib.metadata, "version", return_value="2.0") as version,
                        patch.object(
                            import_utils, "_get_package_distribution_mapping", return_value={"foo_bar": owners}
                        ) as full,
                    ):
                        self.assertEqual(import_utils._is_package_available("foo_bar", True), (True, "2.0"))
                        version.assert_called_once_with(chosen)
                        full.assert_called_once_with()

    def test_missing_direct_distribution_still_finds_alias(self):
        with (
            patch.object(import_utils.importlib.util, "find_spec", return_value=object()),
            patch.object(importlib.metadata, "distribution", side_effect=importlib.metadata.PackageNotFoundError),
            patch.object(import_utils, "_get_package_distribution_mapping", return_value={"PIL": ["pillow"]}),
            patch.object(importlib.metadata, "version", return_value="12.0") as version,
        ):
            self.assertEqual(import_utils._is_package_available("PIL", True), (True, "12.0"))
            version.assert_called_once_with("pillow")

    def test_explicit_mapping_keeps_priority_and_import_fallback(self):
        for mapping, missing_version in [({"custom": ["vendor"]}, False), ({}, False), ({"custom": ["vendor"]}, True)]:
            with self.subTest(mapping=mapping, missing_version=missing_version):
                import_utils.PACKAGE_DISTRIBUTION_MAPPING = mapping
                version = Mock(return_value="1.0")
                if missing_version:
                    version.side_effect = importlib.metadata.PackageNotFoundError
                with (
                    patch.object(import_utils.importlib.util, "find_spec", return_value=object()),
                    patch.object(importlib.metadata, "distribution", side_effect=AssertionError("explicit mapping")),
                    patch.object(importlib.metadata, "version", version),
                    patch.object(
                        import_utils.importlib, "import_module", return_value=SimpleNamespace(__version__="editable")
                    ) as load,
                ):
                    expected = "editable" if missing_version or not mapping else "1.0"
                    self.assertEqual(import_utils._is_package_available("custom", True), (True, expected))
                    self.assertEqual(load.call_count, int(expected == "editable"))

    def test_absent_and_boolean_only_queries_do_not_resolve_metadata(self):
        with (
            patch.object(import_utils.importlib.util, "find_spec", return_value=None),
            patch.object(importlib.metadata, "distribution", side_effect=AssertionError("metadata")),
        ):
            self.assertEqual(import_utils._is_package_available("absent", True), (False, "N/A"))
            self.assertEqual(import_utils._is_package_available("absent"), (False, None))
        with (
            patch.object(import_utils.importlib.util, "find_spec", return_value=object()),
            patch.object(importlib.metadata, "distribution", side_effect=AssertionError("metadata")),
        ):
            self.assertEqual(import_utils._is_package_available("present"), (True, None))

    def test_complete_mapping_exports_same_dictionary_once(self):
        mapping = {"flash_attn": ["flash-attn-4"], "namespace": ["first", "second"]}
        with patch.object(import_utils, "_get_package_distribution_mapping", return_value=mapping) as full:
            self.assertIn("PACKAGE_DISTRIBUTION_MAPPING", dir(import_utils))
            self.assertNotIn("PACKAGE_DISTRIBUTION_MAPPING", vars(import_utils))
            namespace = {}
            exec("from transformers.utils.import_utils import *", namespace)
            self.assertIs(namespace["PACKAGE_DISTRIBUTION_MAPPING"], mapping)
            self.assertIs(import_utils.PACKAGE_DISTRIBUTION_MAPPING, mapping)
            mapping["added"] = ["later"]
            self.assertEqual(import_utils._package_distributions_for("added"), ["later"])
            full.assert_called_once_with()
        with self.assertRaises(AttributeError):
            getattr(import_utils, "unknown_package_mapping_attribute")

    def test_concurrent_full_mapping_consumers_share_one_object(self):
        mapping = {"namespace": ["first", "second"]}
        with patch.object(import_utils, "_get_package_distribution_mapping", return_value=mapping) as full:
            with ThreadPoolExecutor(max_workers=8) as pool:
                observed = list(pool.map(lambda _: import_utils._package_distribution_mapping(), range(32)))
            self.assertTrue(all(value is mapping for value in observed))
            full.assert_called_once_with()

    def test_installed_torch_version_and_complete_mapping_match_standard_library(self):
        if not import_utils.is_torch_available():
            self.skipTest("requires an installed torch distribution")
        mapping = importlib.metadata.packages_distributions()
        distribution = importlib.metadata.distribution("torch")
        if "torch" not in (distribution.read_text("top_level.txt") or "").split():
            self.skipTest("torch installation does not declare a top-level import")
        with patch.object(
            import_utils, "_get_package_distribution_mapping", wraps=importlib.metadata.packages_distributions
        ) as full:
            self.assertEqual(
                import_utils._is_package_available("torch", True), (True, importlib.metadata.version("torch"))
            )
            full.assert_not_called()
            self.assertEqual(import_utils.PACKAGE_DISTRIBUTION_MAPPING, mapping)
            full.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
