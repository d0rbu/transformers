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

import importlib.machinery
import importlib.metadata
import subprocess
import sys
import textwrap
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from transformers.utils import import_utils
from transformers.utils.import_utils import _get_package_distribution_mapping


class DistributionFixture:
    def __init__(self, name, declared=None, files=None):
        self.name = name
        self.declared = declared
        self.files = None if files is None else [importlib.metadata.PackagePath(file) for file in files]
        self.metadata_reads = 0

    def read_text(self, filename):
        assert filename == "top_level.txt"
        return self.declared

    @property
    def metadata(self):
        self.metadata_reads += 1
        return {"Name": self.name}


class PackageDistributionMappingTest(unittest.TestCase):
    def setUp(self):
        self.saved_mapping = vars(import_utils).pop("PACKAGE_DISTRIBUTION_MAPPING", None)

    def tearDown(self):
        vars(import_utils).pop("PACKAGE_DISTRIBUTION_MAPPING", None)
        if self.saved_mapping is not None:
            import_utils.PACKAGE_DISTRIBUTION_MAPPING = self.saved_mapping

    def assert_matches_stdlib(self, distributions):
        with patch("importlib.metadata.distributions", return_value=distributions):
            expected = importlib.metadata.packages_distributions()
            for distribution in distributions:
                distribution.metadata_reads = 0
            actual = _get_package_distribution_mapping()
        # Dict equality alone does not compare key iteration order or duplicate owners.
        self.assertEqual(list(actual.items()), list(expected.items()))
        self.assertIs(type(actual), dict)
        for owners in actual.values():
            self.assertIs(type(owners), list)
        return actual

    def test_declared_aliases_duplicates_and_owner_order(self):
        distributions = [
            DistributionFixture("first-provider", "shared own shared\nalias"),
            DistributionFixture("shared", "shared"),
            DistributionFixture("other-provider", "shared other"),
        ]
        actual = self.assert_matches_stdlib(distributions)
        self.assertEqual(actual["shared"], ["first-provider", "first-provider", "shared", "other-provider"])
        self.assertEqual([distribution.metadata_reads for distribution in distributions], [1, 1, 1])

    def test_inferred_packages_modules_extensions_and_nonimportable_files(self):
        files = [
            "namespace/subpackage/file.py",
            "namespace/another.py",
            "module.py",
            "module.pyc",
            "resources.dist-info/METADATA",
            "../bin/tool",
            "/absolute/file.py",
            "not_a_module.txt",
            "nested.name/file.py",
            "ordinary_package/config.json",
            ".hidden/file",
            "__init__.py",
        ]
        files.extend("extension" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES)
        distribution = DistributionFixture("inferred", files=files)
        actual = self.assert_matches_stdlib([distribution])
        self.assertIn("namespace", actual)
        self.assertIn("extension", actual)
        self.assertIn("ordinary_package", actual)
        self.assertNotIn("resources.dist-info", actual)
        self.assertEqual(distribution.metadata_reads, 1)

    def test_declared_packages_override_file_inference(self):
        distribution = DistributionFixture("declared", "alias\n", ["ignored.py"])
        self.assertEqual(self.assert_matches_stdlib([distribution]), {"alias": ["declared"]})
        self.assertEqual(distribution.metadata_reads, 1)

    def test_empty_metadata_uses_files_and_empty_distributions_do_not_read_names(self):
        distributions = [
            DistributionFixture("inferred", " \n\t", ["actual.py"]),
            DistributionFixture("no-files"),
            DistributionFixture("empty-files", "", []),
            DistributionFixture("nonimportable", files=["METADATA", "top_level.txt"]),
        ]
        self.assertEqual(self.assert_matches_stdlib(distributions), {"actual": ["inferred"]})
        self.assertEqual([distribution.metadata_reads for distribution in distributions], [1, 0, 0, 0])

    def test_each_build_observes_distribution_changes(self):
        distribution = DistributionFixture("before", "old")
        self.assertEqual(self.assert_matches_stdlib([distribution]), {"old": ["before"]})
        distribution.name = "after"
        distribution.declared = "new other"
        self.assertEqual(self.assert_matches_stdlib([distribution]), {"new": ["after"], "other": ["after"]})

    def test_installed_environment_matches_stdlib(self):
        self.assertEqual(
            list(_get_package_distribution_mapping().items()),
            list(importlib.metadata.packages_distributions().items()),
        )

    def test_preferred_distribution_is_resolved_without_the_global_map(self):
        for name in ("some-package", "some_package"):
            with self.subTest(name=name):
                distribution = DistributionFixture(name, "some_package other")
                with (
                    patch("importlib.metadata.distribution", return_value=distribution),
                    patch.object(
                        import_utils, "_get_package_distribution_mapping", side_effect=AssertionError("scan")
                    ),
                ):
                    self.assertEqual(import_utils._package_distribution_name("some_package"), name)
                self.assertNotIn("PACKAGE_DISTRIBUTION_MAPPING", vars(import_utils))

    def test_nonpreferred_owner_and_unowned_package_fall_back_to_the_original_order(self):
        for distribution in (
            DistributionFixture("Some-Package", "some_package"),
            DistributionFixture("some-package", "different_package"),
        ):
            with self.subTest(distribution=distribution.name):
                vars(import_utils).pop("PACKAGE_DISTRIBUTION_MAPPING", None)
                with (
                    patch("importlib.metadata.distribution", return_value=distribution),
                    patch.object(
                        import_utils,
                        "_get_package_distribution_mapping",
                        return_value={"some_package": ["first-provider", "Some-Package"]},
                    ),
                ):
                    self.assertEqual(import_utils._package_distribution_name("some_package"), "first-provider")

    def test_absent_distribution_uses_alias_and_missing_alias_still_raises(self):
        with (
            patch("importlib.metadata.distribution", side_effect=importlib.metadata.PackageNotFoundError("alias")),
            patch.object(import_utils, "_get_package_distribution_mapping", return_value={"alias": ["owner"]}),
        ):
            self.assertEqual(import_utils._package_distribution_name("alias"), "owner")
            with self.assertRaises(KeyError):
                import_utils._package_distribution_name("missing")

    def test_explicit_public_map_is_the_same_mutable_dict_and_has_priority(self):
        mapping = {"some_package": ["other", "some_package", "some-package"]}
        with patch.object(import_utils, "_get_package_distribution_mapping", return_value=mapping) as build:
            self.assertIn("PACKAGE_DISTRIBUTION_MAPPING", dir(import_utils))
            self.assertIs(import_utils.PACKAGE_DISTRIBUTION_MAPPING, mapping)
            self.assertIs(vars(import_utils)["PACKAGE_DISTRIBUTION_MAPPING"], mapping)
            self.assertEqual(import_utils._package_distribution_name("some_package"), "some-package")
            mapping["some_package"] = ["changed"]
            self.assertEqual(import_utils._package_distribution_name("some_package"), "changed")
            build.assert_called_once_with()

    def test_fresh_tokenizer_import_leaves_global_map_deferred(self):
        script = textwrap.dedent("""
            import importlib.metadata
            import transformers
            from transformers import AutoTokenizer
            from transformers.utils import import_utils
            assert import_utils.is_torch_available()
            assert "PACKAGE_DISTRIBUTION_MAPPING" not in vars(import_utils)
            assert "PACKAGE_DISTRIBUTION_MAPPING" in dir(import_utils)
            expected = importlib.metadata.packages_distributions()
            from transformers.utils.import_utils import PACKAGE_DISTRIBUTION_MAPPING
            assert list(PACKAGE_DISTRIBUTION_MAPPING.items()) == list(expected.items())
            assert import_utils.PACKAGE_DISTRIBUTION_MAPPING is PACKAGE_DISTRIBUTION_MAPPING
            namespace = {}
            exec("from transformers.utils.import_utils import *", namespace)
            assert namespace["PACKAGE_DISTRIBUTION_MAPPING"] is PACKAGE_DISTRIBUTION_MAPPING
            assert "_getmodulename" not in namespace
        """)
        subprocess.run([sys.executable, "-c", script], check=True)

    def test_concurrent_public_lookups_share_one_mutable_map(self):
        mapping = {"package": ["owner"]}

        def build():
            time.sleep(0.01)
            return mapping

        with patch.object(import_utils, "_get_package_distribution_mapping", side_effect=build) as factory:
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda _: import_utils.PACKAGE_DISTRIBUTION_MAPPING, range(8)))
        self.assertTrue(all(result is mapping for result in results))
        factory.assert_called_once_with()

    def test_installed_version_queries_match_the_original_owner_preference(self):
        mapping = importlib.metadata.packages_distributions()
        for package, owners in mapping.items():
            with self.subTest(package=package):
                vars(import_utils).pop("PACKAGE_DISTRIBUTION_MAPPING", None)
                normalized = package.replace("_", "-")
                owner = next((name for name in (normalized, package) if name in owners), owners[0])
                expected = importlib.metadata.version(owner)
                with patch("importlib.util.find_spec", return_value=object()):
                    self.assertEqual(
                        import_utils._is_package_available(package, return_version=True), (True, expected)
                    )

    def test_absent_or_unversioned_packages_do_not_initialize_metadata(self):
        for spec, version, expected in (
            (None, False, (False, None)),
            (None, True, (False, "N/A")),
            (object(), False, (True, None)),
        ):
            with self.subTest(expected=expected):
                with (
                    patch("importlib.util.find_spec", return_value=spec),
                    patch("importlib.metadata.distribution", side_effect=AssertionError("metadata")),
                    patch.object(
                        import_utils, "_get_package_distribution_mapping", side_effect=AssertionError("scan")
                    ),
                ):
                    self.assertEqual(import_utils._is_package_available("package", version), expected)

    def test_unmapped_editable_packages_keep_import_fallback_and_missing_version(self):
        for module, expected in ((SimpleNamespace(__version__="dev"), "dev"), (SimpleNamespace(), "N/A")):
            with self.subTest(expected=expected):
                vars(import_utils).pop("PACKAGE_DISTRIBUTION_MAPPING", None)
                with (
                    patch("importlib.util.find_spec", return_value=object()),
                    patch("importlib.metadata.distribution", side_effect=importlib.metadata.PackageNotFoundError),
                    patch.object(import_utils, "_get_package_distribution_mapping", return_value={}),
                    patch("importlib.import_module", return_value=module) as load,
                ):
                    self.assertEqual(import_utils._is_package_available("editable", True), (True, expected))
                    load.assert_called_once_with("editable")

    def test_metadata_version_failure_keeps_import_fallback_and_import_errors(self):
        import_utils.PACKAGE_DISTRIBUTION_MAPPING = {"package": ["owner"]}
        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError),
            patch("importlib.import_module", side_effect=ImportError("broken dependency")),
        ):
            with self.assertRaisesRegex(ImportError, "broken dependency"):
                import_utils._is_package_available("package", True)


if __name__ == "__main__":
    unittest.main()
