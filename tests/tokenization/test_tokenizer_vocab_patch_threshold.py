# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from tokenizers import Tokenizer, models, pre_tokenizers

from transformers.tokenization_utils_tokenizers import TokenizersBackend


class TokenizerVocabularyPatchThresholdTest(unittest.TestCase):
    def test_patch_consumer_uses_complete_vocabulary_threshold(self):
        class ObservedTokenizer(TokenizersBackend):
            patch_calls = []

            @classmethod
            def _patch_mistral_regex(cls, tokenizer, *args, **kwargs):
                cls.patch_calls.append(tokenizer.get_vocab_size())
                return tokenizer

        for base_size in (0, 99998, 99999, 100000, 100001):
            for extra_size in (0, 1, 3):
                for use_pre_tokenizer in (False, True):
                    with self.subTest(base=base_size, extra=extra_size, pre_tokenizer=use_pre_tokenizer):
                        backend = Tokenizer(models.WordLevel({f"t{i}": i for i in range(base_size)}))
                        if use_pre_tokenizer:
                            backend.pre_tokenizer = pre_tokenizers.Whitespace()
                        backend.add_tokens([f"added{i}" for i in range(extra_size)])
                        if base_size:
                            backend.add_tokens(["t0"])
                        expected_total = backend.get_vocab_size()
                        ObservedTokenizer.patch_calls.clear()
                        tokenizer = ObservedTokenizer(
                            tokenizer_object=backend,
                            bos_token=None,
                            eos_token=None,
                            unk_token=None,
                        )
                        expected_calls = [expected_total] if use_pre_tokenizer and expected_total > 100000 else []
                        self.assertEqual(ObservedTokenizer.patch_calls, expected_calls)
                        self.assertEqual(tokenizer.vocab_size, base_size)
                        self.assertEqual(len(tokenizer), expected_total)
                        self.assertEqual(tokenizer.get_vocab(), backend.get_vocab())
                        if base_size:
                            self.assertEqual(tokenizer.convert_tokens_to_ids("t0"), 0)


if __name__ == "__main__":
    unittest.main()
