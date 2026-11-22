# Copyright 2026 The HuggingFace Inc. team.
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

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenizers import AddedToken

from transformers import GPT2Tokenizer
from transformers.tokenization_utils_tokenizers import TokenizersBackend


class TokenizersBackendConversionTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.scratch = Path(temporary.name)

    def save_tokenizer(self, directory, left="A", right="B"):
        joined = left + right
        tokenizer = GPT2Tokenizer(
            vocab={left: 0, right: 1, joined: 2, "<|endoftext|>": 3},
            merges=[(left, right)],
        )
        tokenizer.add_tokens([AddedToken("<extra>", lstrip=True, normalized=False)])
        tokenizer.save_pretrained(directory)
        config_path = directory / "tokenizer_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.pop("added_tokens_decoder", None)
        config_path.write_text(json.dumps(config), encoding="utf-8")
        return tokenizer

    def load_with_read_count(self, directory, **kwargs):
        with patch("builtins.open", wraps=open) as observed:
            loaded = GPT2Tokenizer.from_pretrained(directory, local_files_only=True, **kwargs)
        reads = sum(
            str(call.args[0]) == str(directory / "tokenizer.json") for call in observed.call_args_list if call.args
        )
        return loaded, reads

    def test_legacy_added_tokens_use_the_same_serialization_as_conversion(self):
        original = self.save_tokenizer(self.scratch)
        loaded, reads = self.load_with_read_count(self.scratch)

        self.assertEqual(reads, 1)
        self.assertEqual(loaded.get_vocab(), original.get_vocab())
        self.assertEqual(loaded.added_tokens_decoder, original.added_tokens_decoder)
        self.assertEqual(loaded.encode("AB <extra>"), original.encode("AB <extra>"))

    def test_explicit_conversion_file_is_loaded_separately(self):
        source = self.scratch / "source"
        override = self.scratch / "override"
        self.save_tokenizer(source)
        self.save_tokenizer(override, "C", "D")

        loaded, reads = self.load_with_read_count(source, tokenizer_file=str(override / "tokenizer.json"))

        self.assertEqual(reads, 1)
        self.assertEqual(loaded.encode("CD"), [2])
        self.assertNotIn("AB", loaded.get_vocab())
        self.assertIn("<extra>", loaded.get_added_vocab())

    def test_replaced_converter_receives_its_original_arguments(self):
        self.save_tokenizer(self.scratch)
        original = TokenizersBackend.convert_to_native_format
        supplied = []

        def convert(cls, **kwargs):
            supplied.append(kwargs["tokenizer_file"])
            self.assertNotIn("tokenizer_json", kwargs)
            return original.__func__(cls, **kwargs)

        with patch.object(TokenizersBackend, "convert_to_native_format", classmethod(convert)):
            loaded, reads = self.load_with_read_count(self.scratch)

        self.assertEqual(list(map(Path, supplied)), [self.scratch / "tokenizer.json"])
        self.assertEqual(reads, 2)
        self.assertEqual(loaded.encode("AB"), [2])


if __name__ == "__main__":
    unittest.main()
