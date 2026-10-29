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
class GenericDebuggerImportTest(unittest.TestCase):
    def run_fresh(self, source: str) -> None:
        package = Path(__file__).resolve().parents[2] / "src"
        script = f"import sys; sys.path.insert(0, {str(package)!r})\n" + textwrap.dedent(source)
        result = subprocess.run([sys.executable, "-I", "-B", "-c", script], capture_output=True, text=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_ordinary_decorated_forward_keeps_debugger_unloaded(self) -> None:
        self.run_fresh("""
            from types import SimpleNamespace
            import torch
            from transformers.utils import generic

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = SimpleNamespace(is_causal=False)

                @generic.merge_with_config_defaults
                def forward(self, value, **kwargs):
                    return value + 1

            model = Model()
            assert model(torch.tensor([2])).tolist() == [3]
            assert "model_addition_debugger_context" in dir(generic)
            assert "transformers.model_debugging_utils" not in sys.modules
            assert "torch" in vars(generic)
            assert "_torch_pytree" in vars(generic)
        """)

    def test_first_debug_forward_loads_context_and_restores_config_on_failure(self) -> None:
        self.run_fresh("""
            from types import SimpleNamespace
            import torch
            from transformers.utils import generic

            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.config = SimpleNamespace(is_causal=False)
                    self.layer = torch.nn.Identity()

                @generic.merge_with_config_defaults
                def forward(self, value, **kwargs):
                    assert self.config.is_causal is True
                    value = self.layer(value)
                    if kwargs.get("fail"):
                        raise ValueError("forward failed")
                    return value + 1

            model = Model()
            assert "model_addition_debugger_context" not in vars(generic)
            assert model(torch.tensor([2]), debug_io=True, debug_io_dir=None, is_causal=True).tolist() == [3]
            from transformers.model_debugging_utils import model_addition_debugger_context
            assert generic.model_addition_debugger_context is model_addition_debugger_context
            assert model.config.is_causal is False
            before = model.layer.forward
            try:
                model(torch.tensor([2]), debug_io=True, debug_io_dir=None, is_causal=True, fail=True)
            except ValueError as error:
                assert str(error) == "forward failed"
            else:
                raise AssertionError("Forward failure did not propagate")
            assert model.config.is_causal is False
            assert model.layer.forward == before
        """)

    def test_legacy_context_override_before_first_use(self) -> None:
        self.run_fresh("""
            from contextlib import contextmanager
            from types import SimpleNamespace
            from transformers.utils import generic

            events = []
            @contextmanager
            def replacement(model, path, prune):
                events.append((path, prune))
                yield model
                events.append("exit")

            class Model:
                config = SimpleNamespace()
                @generic.merge_with_config_defaults
                def forward(self, value, **kwargs):
                    return value + 1

            generic.model_addition_debugger_context = replacement
            assert Model().forward(2, debug_io=True, debug_io_dir="chosen", prune_layers=False) == 3
            assert events == [("chosen", False), "exit"]
            assert "transformers.model_debugging_utils" not in sys.modules
            del generic.model_addition_debugger_context
            from transformers.utils.generic import model_addition_debugger_context
            from transformers.model_debugging_utils import model_addition_debugger_context as expected
            assert model_addition_debugger_context is expected
            assert vars(generic)["model_addition_debugger_context"] is expected
        """)

    def test_wildcard_import_preserves_debugger_and_hides_private_helper(self) -> None:
        self.run_fresh("""
            namespace = {}
            exec("from transformers.utils.generic import *", namespace)
            from transformers.model_debugging_utils import model_addition_debugger_context
            assert namespace["model_addition_debugger_context"] is model_addition_debugger_context
            assert "importlib" not in namespace
            assert "_importlib" not in namespace
        """)

    def test_torch_unavailable_keeps_debugger_attribute_unavailable(self) -> None:
        self.run_fresh("""
            import importlib.util
            original = importlib.util.find_spec
            def without_torch(name, package=None):
                return None if name == "torch" or name.startswith("torch.") else original(name, package)
            importlib.util.find_spec = without_torch
            from transformers.utils import generic
            assert not generic._is_torch_available
            assert "model_addition_debugger_context" not in dir(generic)
            assert not hasattr(generic, "model_addition_debugger_context")
            assert "transformers.model_debugging_utils" not in sys.modules
        """)


if __name__ == "__main__":
    unittest.main()
