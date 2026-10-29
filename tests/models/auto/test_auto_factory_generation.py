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

import importlib.util
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is required")
class AutoFactoryGenerationTest(unittest.TestCase):
    def run_fresh(self, source: str) -> None:
        package = Path(__file__).resolve().parents[3] / "src"
        script = f"import sys; sys.path.insert(0, {str(package)!r})\n" + textwrap.dedent(source)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_tokenizer_and_nongenerative_classes_do_not_import_generation(self) -> None:
        self.run_fresh("""
            from transformers import AutoTokenizer
            from transformers.models.auto import auto_factory
            import torch

            class Plain:
                pass

            class NonGenerative(torch.nn.Module):
                pass

            for original in (Plain, NonGenerative):
                assert auto_factory.add_generation_mixin_to_remote_model(original) is original
            assert "GenerationMixin" in dir(auto_factory)
            assert "transformers.generation.utils" not in sys.modules
        """)

    def test_first_remote_generate_or_prepare_loads_the_compatible_mixin(self) -> None:
        self.run_fresh("""
            from transformers.models.auto import auto_factory
            import torch

            class Generate(torch.nn.Module):
                def generate(self, value):
                    return value + 1

            class Prepare(torch.nn.Module):
                def prepare_inputs_for_generation(self, value):
                    return {"value": value + 2}

            assert "GenerationMixin" not in vars(auto_factory)
            generated = auto_factory.add_generation_mixin_to_remote_model(Generate)
            from transformers.generation import GenerationMixin
            assert generated.__bases__ == (Generate, GenerationMixin)
            assert generated.__name__ == Generate.__name__
            assert generated.__module__ == Generate.__module__
            assert generated.generate is Generate.generate
            assert generated().generate(4) == 5
            assert isinstance(generated(), GenerationMixin)

            prepared = auto_factory.add_generation_mixin_to_remote_model(Prepare)
            assert prepared.__bases__ == (Prepare, GenerationMixin)
            assert prepared.prepare_inputs_for_generation is Prepare.prepare_inputs_for_generation
            assert prepared().prepare_inputs_for_generation(4) == {"value": 6}

            class Direct(torch.nn.Module, GenerationMixin):
                pass

            class Indirect(Direct):
                pass

            assert auto_factory.add_generation_mixin_to_remote_model(Direct) is Direct
            assert auto_factory.add_generation_mixin_to_remote_model(Indirect) is Indirect
        """)

    def test_legacy_attribute_access_and_override(self) -> None:
        self.run_fresh("""
            from transformers.models.auto import auto_factory
            import torch

            class Generate(torch.nn.Module):
                def generate(self, value):
                    return value

            class Replacement:
                pass

            # An override before the first generation import keeps its original meaning.
            auto_factory.GenerationMixin = Replacement
            wrapped = auto_factory.add_generation_mixin_to_remote_model(Generate)
            assert wrapped.__bases__ == (Generate, Replacement)
            assert "transformers.generation.utils" not in sys.modules
            del auto_factory.GenerationMixin
            from transformers.models.auto.auto_factory import GenerationMixin
            from transformers.generation import GenerationMixin as expected
            assert GenerationMixin is expected
            assert vars(auto_factory)["GenerationMixin"] is expected
            try:
                auto_factory._missing_factory_attribute
            except AttributeError as error:
                assert str(error) == "module 'transformers.models.auto.auto_factory' has no attribute '_missing_factory_attribute'"
            else:
                raise AssertionError("Missing factory attribute did not fail")
        """)

    def test_wildcard_import_preserves_explicit_exports(self) -> None:
        self.run_fresh("""
            namespace = {}
            exec("from transformers.models.auto.auto_factory import *", namespace)
            from transformers.models.auto import auto_factory
            assert sorted(name for name in namespace if not name.startswith("_")) == ["get_values"]
            assert namespace["get_values"] is auto_factory.get_values
        """)

    def test_torch_unavailable_keeps_generation_attribute_unavailable(self) -> None:
        self.run_fresh("""
            import importlib.util
            find_spec = importlib.util.find_spec

            def without_torch(name, package=None):
                if name == "torch" or name.startswith("torch."):
                    return None
                return find_spec(name, package)

            importlib.util.find_spec = without_torch
            from transformers.models.auto import auto_factory
            assert not auto_factory.is_torch_available()
            assert "GenerationMixin" not in dir(auto_factory)
            assert not hasattr(auto_factory, "GenerationMixin")

            class Plain:
                pass

            assert auto_factory.add_generation_mixin_to_remote_model(Plain) is Plain
            assert "transformers.generation.utils" not in sys.modules
        """)


if __name__ == "__main__":
    unittest.main()
