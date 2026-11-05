"""Import the initialization module without demanding the full lazy model catalog."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "first_import",
    [
        "import transformers",
        "import transformers.initialization",
        "from transformers.models.llama.configuration_llama import LlamaConfig",
    ],
)
def test_modeling_initialization_uses_its_module_owner(first_import):
    subprocess.run(
        [
            sys.executable,
            "-c",
            first_import
            + """
import importlib
import transformers as root
from transformers.utils.import_utils import _LazyModule

factory = root._deferred_import_structure
assert factory is not None
modeling = importlib.import_module('transformers.modeling_utils')
initialization = importlib.import_module('transformers.initialization')
assert modeling.init is initialization
assert root.initialization is initialization
assert root._deferred_import_structure is factory
assert modeling.init.no_init_weights is initialization.no_init_weights
assert modeling.init.guard_torch_init_functions is initialization.guard_torch_init_functions

# A later complete-catalog reader still receives every original export/backend.
expected = _LazyModule(root.__name__, root.__file__, factory(), root.__spec__, root._objects)
assert root.__all__ == expected.__all__
assert root._class_to_module == expected._class_to_module
assert root._object_missing_backend == expected._object_missing_backend
assert root._import_structure == expected._import_structure
""",
        ],
        check=True,
    )
