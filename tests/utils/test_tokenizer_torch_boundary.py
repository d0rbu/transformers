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

import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


class TokenizerTorchBoundaryTest(unittest.TestCase):
    def run_fresh(self, source: str) -> None:
        root = Path(__file__).resolve().parents[2] / "src"
        command = "import sys\nsys.path.insert(0, " + repr(str(root)) + ")\n" + textwrap.dedent(source)
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", command],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1"),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_plain_tokenizer_and_config_do_not_load_torch(self) -> None:
        self.run_fresh("""
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers.tokenization_utils_tokenizers import TokenizersBackend
            from transformers.configuration_utils import PreTrainedConfig

            backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")
            assert tokenizer("hello hello")["input_ids"] == [1, 1]
            assert tokenizer.decode([1, 1]) == "hello hello"
            assert PreTrainedConfig().to_dict()["model_type"] == ""
            assert "torch" not in sys.modules

            config = PreTrainedConfig(dtype="float32")
            import torch
            assert config.dtype is torch.float32
            assert tokenizer("hello", return_tensors="pt")["input_ids"].tolist() == [[1]]
        """)

    def test_saved_auto_tokenizer_does_not_load_torch(self) -> None:
        self.run_fresh("""
            from tempfile import TemporaryDirectory
            from tokenizers import Tokenizer, models, pre_tokenizers
            from transformers import AutoTokenizer
            from transformers.configuration_utils import PreTrainedConfig
            from transformers.tokenization_utils_tokenizers import TokenizersBackend
            backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = TokenizersBackend(tokenizer_object=backend, unk_token="[UNK]")
            with TemporaryDirectory() as path:
                tokenizer.save_pretrained(path)
                restored = AutoTokenizer.from_pretrained(path, config=PreTrainedConfig(),
                                                        local_files_only=True, trust_remote_code=False)
                assert type(restored) is TokenizersBackend
                assert restored("hello hello")["input_ids"] == [1, 1]
                assert restored.decode([1, 1]) == "hello hello"
            assert "torch" not in sys.modules
        """)

    def test_gguf_aliases_and_original_patch_targets(self) -> None:
        self.run_fresh("""
            from pathlib import Path
            from tempfile import TemporaryDirectory
            from unittest.mock import patch
            from transformers import configuration_utils as config
            from transformers import tokenization_utils_tokenizers as tokenizer
            from transformers.models.auto import tokenization_auto as auto
            from transformers.modeling_gguf_pytorch_utils import load_gguf_checkpoint
            from transformers.convert_slow_tokenizer import SpmConverter
            from transformers.integrations.ggml import convert_gguf_tokenizer

            assert config.load_gguf_checkpoint is load_gguf_checkpoint
            assert auto.load_gguf_checkpoint is load_gguf_checkpoint
            assert tokenizer.load_gguf_checkpoint is load_gguf_checkpoint
            assert tokenizer.SpmConverter is SpmConverter
            assert tokenizer.convert_gguf_tokenizer is convert_gguf_tokenizer

            with TemporaryDirectory() as path:
                source = Path(path) / "model.gguf"
                source.write_bytes(b"loader-dispatch-only")
                with patch.object(config, "load_gguf_checkpoint", return_value={"config": {"marker": 17}}) as load:
                    value, _ = config.PreTrainedConfig._get_config_dict(
                        path, gguf_file=source.name, local_files_only=True
                    )
                    assert value["marker"] == 17
                    load.assert_called_once_with(str(source), return_tensors=False)

                for module, call, options in (
                    (auto, lambda: auto.AutoTokenizer.from_pretrained(
                        path, gguf_file=source.name, local_files_only=True
                    ), {"return_tensors": False}),
                    (tokenizer, lambda: tokenizer.TokenizersBackend(
                        name_or_path=path, gguf_file=source.name, local_files_only=True
                    ), {}),
                ):
                    error = RuntimeError("original-loader-error")
                    with patch.object(module, "load_gguf_checkpoint", side_effect=error) as load:
                        try:
                            call()
                        except RuntimeError as caught:
                            assert caught is error
                        else:
                            raise AssertionError("Loader error did not propagate")
                        load.assert_called_once_with(str(source), **options)
        """)

    def test_cold_model_output_retains_annotations_and_pytree(self) -> None:
        self.run_fresh("""
            from dataclasses import dataclass
            import pickle
            from typing import get_type_hints
            from transformers.utils import ModelOutput, TransformersKwargs
            from transformers.utils import generic
            import torch

            assert generic.torch is torch
            assert get_type_hints(TransformersKwargs)["num_items_in_batch"] == torch.Tensor | None
            assert get_type_hints(generic.maybe_autocast)["dtype"] == torch.dtype | None
            assert ModelOutput.__init_subclass__.__func__.__module__ == "transformers.utils.generic"
            assert pickle.loads(pickle.dumps(ModelOutput)) is ModelOutput
            @dataclass
            class Output(ModelOutput):
                logits: torch.Tensor = None
            value = torch.tensor([1.0, 2.0], requires_grad=True)
            leaves, spec = torch.utils._pytree.tree_flatten(Output(value))
            restored = torch.utils._pytree.tree_unflatten(leaves, spec)
            assert type(restored) is Output and restored.logits is value
            restored.logits.square().sum().backward()
            torch.testing.assert_close(value.grad, torch.tensor([2.0, 4.0]), rtol=0, atol=0)
        """)

    def test_optional_torch_absence_keeps_tokenizer_and_model_output(self) -> None:
        self.run_fresh("""
            import importlib.util
            import builtins
            original = importlib.util.find_spec
            def no_torch(name, *args, **kwargs):
                return None if name == "torch" else original(name, *args, **kwargs)
            importlib.util.find_spec = no_torch
            original_import = builtins.__import__
            def without_torch(name, *args, **kwargs):
                if name == "torch" or name.startswith("torch."):
                    raise ModuleNotFoundError("No module named 'torch'", name="torch")
                return original_import(name, *args, **kwargs)
            builtins.__import__ = without_torch
            from dataclasses import dataclass
            from transformers.configuration_utils import PreTrainedConfig
            from transformers.utils import generic
            assert not generic.is_torch_tensor(1)
            assert not generic.is_torch_device("cpu")
            assert not generic.is_torch_dtype("float32")
            assert generic.torch_int(2.0) == 2
            @dataclass
            class Output(generic.ModelOutput):
                logits: int = None
            assert Output(2).to_tuple() == (2,)
            assert PreTrainedConfig().model_type == ""
            assert "torch" not in sys.modules
            assert "Tensor" not in dir(__import__("transformers.utils.chat_template_utils", fromlist=["_"]))
        """)

    def test_watermark_annotations_and_original_patch_target(self) -> None:
        self.run_fresh("""
            import pickle
            from typing import get_type_hints
            from unittest.mock import patch
            from transformers.generation import configuration_utils as config
            cls = config.WatermarkingConfig
            assert get_type_hints(cls.construct_processor)["return"] is config.WatermarkLogitsProcessor
            assert pickle.loads(pickle.dumps(cls)) is cls
            assert cls.from_dict.__func__.__module__ == config.__name__
            with patch.object(config, "WatermarkLogitsProcessor") as make_processor:
                value = cls().construct_processor(16, "cpu")
                assert value is make_processor.return_value
                assert make_processor.call_args.kwargs["vocab_size"] == 16
            value = config.GenerationConfig(watermarking_config={"bias": 2.5})
            assert isinstance(value.watermarking_config, cls)
            assert value.watermarking_config.bias == 2.5
        """)

    def test_dynamic_rope_observes_replaced_public_registry(self) -> None:
        self.run_fresh("""
            from unittest.mock import patch
            import torch
            import transformers.modeling_rope_utils as rope
            from transformers.models.llama.configuration_llama import LlamaConfig
            from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
            config = LlamaConfig(hidden_size=16, num_attention_heads=2, max_position_embeddings=4,
                                 rope_parameters={"rope_type": "dynamic", "factor": 2.0, "rope_theta": 10000.0})
            layer = LlamaRotaryEmbedding(config)
            calls = []
            original = rope.ROPE_INIT_FUNCTIONS["dynamic"]
            def counted(*args, **kwargs):
                calls.append(kwargs.get("seq_len"))
                return original(*args, **kwargs)
            with patch.object(rope, "ROPE_INIT_FUNCTIONS", {**rope.ROPE_INIT_FUNCTIONS, "dynamic": counted}):
                cosine, sine = layer(torch.zeros(1, 8, 16), torch.arange(8)[None, :])
            assert len(calls) == 1 and calls[0] == 8
            assert cosine.shape == sine.shape == (1, 8, 8)
            assert torch.isfinite(cosine).all() and torch.isfinite(sine).all()
        """)
