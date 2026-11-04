import pickle
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from transformers.utils.import_utils import _DeferredLazyModule, _LazyModule


def module(factory):
    return _DeferredLazyModule(
        "example", "/example/__init__.py", {frozenset(): {"core": {"Core"}}}, deferred_import_structure=factory
    )


def test_known_export_and_metadata_do_not_discover_model_tree():
    calls = []
    value = object()
    m = module(lambda: calls.append(1) or {frozenset(): {"extra": {"Extra"}}})
    m._get_module = lambda name: SimpleNamespace(Core=value)
    assert m.Core is value
    assert m.__path__ == ["/example"]
    assert m.__file__ == "/example/__init__.py"
    with pytest.raises(AttributeError):
        m.__not_a_real_dunder__
    assert calls == []


def test_unknown_export_expands_once_and_preserves_loaded_attributes():
    calls = []
    structure = {frozenset(): {"core": {"Core"}, "extra": {"Extra"}}}
    m = module(lambda: calls.append(1) or structure)
    core, extra = object(), object()
    m._get_module = lambda name: SimpleNamespace(Core=core, Extra=extra)
    assert m.Core is core
    assert m.Extra is extra
    assert m.Extra is extra
    assert m.Core is core
    expected = _LazyModule("example", "/example/__init__.py", structure)
    assert m.__all__ == expected.__all__
    assert m._class_to_module == expected._class_to_module
    assert m._import_structure == expected._import_structure
    assert m._object_missing_backend == expected._object_missing_backend
    assert calls == [1]


@pytest.mark.parametrize("consumer", [lambda m: m.__all__, dir, pickle.dumps])
@pytest.mark.parametrize("backend_keys", [False, True])
def test_export_introspection_and_pickle_expand(consumer, backend_keys):
    calls = []
    exports = {"extra": {"Extra"}, "other": {"Other", "Another"}}
    structure = {frozenset(): exports} if backend_keys else exports
    expected = _LazyModule("example", "/example/__init__.py", structure)
    m = module(lambda: calls.append(1) or structure)
    consumer(m)
    assert calls == [1]
    assert m.__all__ == expected.__all__
    restored = pickle.loads(pickle.dumps(m))
    expected_restored = pickle.loads(pickle.dumps(expected))
    assert restored._class_to_module == m._class_to_module
    assert restored._class_to_module == expected_restored._class_to_module
    # Ordinary lazy modules flatten backend-keyed exports when pickled, which
    # can change __all__ ordering. Match that existing round-trip contract.
    assert restored.__all__ == expected_restored.__all__
    assert restored._modules == expected_restored._modules
    assert restored._import_structure == expected_restored._import_structure
    assert restored._deferred_import_structure is None


def test_failed_discovery_keeps_original_maps():
    calls = []

    def factory():
        calls.append(1)
        if len(calls) == 1:
            raise OSError("unreadable source")
        return {frozenset(): {"extra": {"Extra"}}}

    m = module(factory)
    before = m._class_to_module.copy()
    with pytest.raises(OSError, match="unreadable source"):
        m.Extra
    assert m._class_to_module == before
    assert "Extra" in m.__all__
    assert len(calls) == 2


def test_concurrent_discovery_publishes_once():
    entered, release = Event(), Event()
    calls = []

    def factory():
        calls.append(1)
        entered.set()
        assert release.wait(5)
        return {frozenset(): {"extra": {"Extra"}}}

    m = module(factory)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lambda: m.__all__)
        assert entered.wait(5)
        second = executor.submit(lambda: m.__all__)
        release.set()
        assert first.result() == second.result()
    assert calls == [1]


def test_known_export_failure_does_not_trigger_model_discovery():
    calls = []
    m = module(lambda: calls.append(1) or {})

    def failure(name):
        raise RuntimeError("known module failed")

    m._get_module = failure
    with pytest.raises(ModuleNotFoundError):
        m.Core
    assert calls == []


def test_actual_owner_import_and_complete_model_exports():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            """
import sys
from reward_model_conditional_pretraining import _pretrain_encoding
import transformers as root
from transformers.utils.import_utils import _LazyModule, create_import_structure_from_path
models = sys.modules['transformers.models']
assert root._deferred_import_structure is not None
assert models._deferred_import_structure is not None
assert create_import_structure_from_path.cache_info().misses < 100
expected_root = _LazyModule(root.__name__, root.__file__, root._deferred_import_structure(), root.__spec__, root._objects)
expected_models = _LazyModule(models.__name__, models.__file__, models._deferred_import_structure(), models.__spec__)
for name, module in root._class_to_module.items():
    assert expected_root._class_to_module[name] == module, name
    assert expected_root._object_missing_backend.get(name) == root._object_missing_backend.get(name), name
for actual, expected in [(root, expected_root), (models, expected_models)]:
    assert actual.__all__ == expected.__all__
    assert actual._class_to_module == expected._class_to_module
    assert actual._modules == expected._modules
    assert actual._object_missing_backend == expected._object_missing_backend
    assert actual._import_structure == expected._import_structure
    assert set(dir(expected)) <= set(dir(actual))
from transformers import BertConfig, GPT2Config, GPT2Tokenizer
from transformers.models.bert.configuration_bert import BertConfig as expected_bert
from transformers.models.gpt2.configuration_gpt2 import GPT2Config as expected_gpt2
assert BertConfig is expected_bert
assert GPT2Config is expected_gpt2
assert GPT2Tokenizer is _pretrain_encoding.GPT2Tokenizer
""",
        ],
        check=True,
    )


def test_shared_gpt2_tokenizer_resolution_keeps_root_discovery_deferred():
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            """
import sys
import transformers as root
from transformers.models.auto.tokenization_auto import tokenizer_class_from_name
from transformers.utils.import_utils import create_import_structure_from_path
initial_factory = root._deferred_import_structure
assert initial_factory is not None
resolved = tokenizer_class_from_name('GPT2Tokenizer')
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
assert resolved is GPT2Tokenizer
assert root.GPT2Tokenizer is GPT2Tokenizer
assert root._deferred_import_structure is initial_factory
assert sys.modules['transformers.models']._deferred_import_structure is not None
assert create_import_structure_from_path.cache_info().misses < 100
assert 'torch' not in sys.modules
# The full catalog must preserve backend declarations and all GPT2 exports.
from transformers.utils.import_utils import _LazyModule
expected = _LazyModule(root.__name__, root.__file__, initial_factory(), root.__spec__, root._objects)
assert root.__all__ == expected.__all__
assert root._class_to_module == expected._class_to_module
assert root._object_missing_backend == expected._object_missing_backend
assert root._import_structure == expected._import_structure
assert 'GPT2Model' in root._class_to_module
assert 'BertConfig' in root._class_to_module
assert root.GPT2Tokenizer is resolved
""",
        ],
        check=True,
    )
