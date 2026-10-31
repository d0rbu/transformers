# Copyright 2022 The HuggingFace Inc. team.
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
"""Watermark configuration consumers with their original processor type globals."""

import copy
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..utils import is_torch_available
from . import configuration_utils as _owner


if is_torch_available():
    from .logits_process import SynthIDTextWatermarkLogitsProcessor as SynthIDTextWatermarkLogitsProcessor
    from .logits_process import WatermarkLogitsProcessor


@dataclass
class BaseWatermarkingConfig(ABC):
    """Generic watermarking config"""

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """
        Constructs a BaseWatermarkingConfig instance from a dictionary of parameters.

        Args:
            config_dict (dict[str, Any]): Dictionary containing configuration parameters.
            **kwargs: Additional keyword arguments to override dictionary values.

        Returns:
            BaseWatermarkingConfig: Instance of BaseWatermarkingConfig constructed from the dictionary.
        """
        config = cls(**config_dict)
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config

    def to_json_file(self, json_file_path: str | os.PathLike):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (Union[str, os.PathLike]): Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

            writer.write(json_string)

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            dict[str, Any]: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output

    def __iter__(self):
        yield from copy.deepcopy(self.__dict__).items()

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_json_string(self):
        """
        Serializes this instance to a JSON formatted string.

        Returns:
            str: JSON formatted string representing the configuration instance.
        """
        return json.dumps(self.__dict__, indent=2) + "\n"

    def update(self, **kwargs):
        """
        Update the configuration attributes with new values.

        Args:
            **kwargs: Keyword arguments representing configuration attributes and their new values.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

    @abstractmethod
    def validate(self): ...

    @abstractmethod
    def construct_processor(self, vocab_size): ...


@dataclass
class WatermarkingConfig(BaseWatermarkingConfig):
    """
    Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
    See [this paper](https://huggingface.co/papers/2306.04634) for more details on the arguments.

    Accepts the following keys:
        - greenlist_ratio (`float`):
            Used for watermarking. The ratio of "green" tokens used to the vocabulary size. Defaults to 0.25.
        - bias (`float`):
            Used with watermarking. The bias added to the selected "green" tokens' logits. Defaults to 2.0.
        - hashing_key (`int`):
            Hashing key used for watermarking. Defaults to 15485863 (the millionth prime).
        - seeding_scheme (`str`):
            Algorithm to use for watermarking. Accepts values:
                - "lefthash" (default): "green" tokens selection depend on the last token (Algorithm 2 from the paper)
                - "selfhash": "green" tokens selection depends on the current token itself (Algorithm 3 from the paper)
                    The downside of this scheme is that it considers all possible next tokens and can be slower than "lefthash".
        - context_width(`int`):
            The context length of previous tokens to use in seeding. Higher context length makes watermarking more robust.
    """

    def __init__(
        self,
        greenlist_ratio: float = 0.25,
        bias: float = 2.0,
        hashing_key: int = 15485863,
        seeding_scheme: str = "lefthash",
        context_width: int = 1,
    ):
        self.greenlist_ratio = greenlist_ratio
        self.bias = bias
        self.hashing_key = hashing_key
        self.seeding_scheme = seeding_scheme
        self.context_width = context_width

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.seeding_scheme not in ["selfhash", "lefthash"]:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="seeding_scheme",
                    correct_value="[`selfhash`, `lefthash`]",
                    found_value=self.seeding_scheme,
                ),
            )
        if not 0.0 <= self.greenlist_ratio <= 1.0:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="greenlist_ratio",
                    correct_value="in range between 0.0 and 1.0",
                    found_value=self.seeding_scheme,
                ),
            )
        if not self.context_width >= 1:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="context_width",
                    correct_value="a positive integer",
                    found_value=self.context_width,
                ),
            )

    def construct_processor(self, vocab_size: int, device) -> "WatermarkLogitsProcessor":
        return _owner.__getattr__("WatermarkLogitsProcessor")(
            vocab_size=vocab_size,
            device=device,
            greenlist_ratio=self.greenlist_ratio,
            bias=self.bias,
            hashing_key=self.hashing_key,
            seeding_scheme=self.seeding_scheme,
            context_width=self.context_width,
        )


@dataclass
class SynthIDTextWatermarkingConfig(BaseWatermarkingConfig):
    """
    Class that holds arguments for watermark generation and should be passed into `GenerationConfig` during `generate`.
    See [this paper](https://www.nature.com/articles/s41586-024-08025-4) for more details on the arguments.

    Args:
        ngram_len (`int`):
            Ngram length.
        keys (`list[int]`):
            A sequence of watermarking keys, one for each depth.
        context_history_size (`int`, *optional*, defaults to 1024):
            Size of the tensor to keep track of seen contexts.
        sampling_table_seed (`int`, *optional*, defaults to 0):
            Random seed to generate the sampling table.
        sampling_table_size (`int`, *optional*, defaults to 65536):
            Size of the sampling table.
        skip_first_ngram_calls (`bool`, *optional*, defaults to `False`):
            Whether to skip first ngram calls.
        debug_mode (`bool`, optional, *optional*, defaults to `False`):
            Logits are modified to uniform one got before watermarking modification is applied. This is to test the
            implementation.

    Examples:
    ```python
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer, SynthIDTextWatermarkingConfig

    >>> tokenizer = AutoTokenizer.from_pretrained('google/gemma-2-2b', padding_side="left")
    >>> model = AutoModelForCausalLM.from_pretrained('google/gemma-2-2b')

    >>> # SynthID Text configuration
    >>> watermarking_config = SynthIDTextWatermarkingConfig(
    ...     keys=[654, 400, 836, 123, 340, 443, 597, 160, 57],
    ...     ngram_len=5,
    ... )

    >>> # Generation with watermarking
    >>> tokenized_prompts = tokenizer(["Once upon a time, "], return_tensors="pt", padding=True)
    >>> output_sequences = model.generate(
    ...     **tokenized_prompts, watermarking_config=watermarking_config, do_sample=True, max_new_tokens=10
    ... )
    >>> watermarked_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
    ```
    """

    def __init__(
        self,
        ngram_len: int,
        keys: list[int],
        context_history_size: int = 1024,
        sampling_table_seed: int = 0,
        sampling_table_size: int = 2**16,
        skip_first_ngram_calls: bool = False,
        debug_mode: bool = False,
    ):
        self.ngram_len = ngram_len
        self.keys = keys
        self.sampling_table_size = sampling_table_size
        self.sampling_table_seed = sampling_table_seed
        self.context_history_size = context_history_size
        self.skip_first_ngram_calls = skip_first_ngram_calls
        self.debug_mode = debug_mode

    def validate(self):
        watermark_missing_arg_msg = (
            "Some of the keys in `watermarking_config` are defined incorrectly. `{key}` should be {correct_value}` "
            "but found {found_value}"
        )
        if self.sampling_table_size > 2**24:
            raise ValueError(
                watermark_missing_arg_msg.format(
                    key="sampling_table_size",
                    correct_value="< 2**24",
                    found_value=self.sampling_table_size,
                ),
            )

    def construct_processor(self, vocab_size: int, device) -> "WatermarkLogitsProcessor":
        return _owner.__getattr__("SynthIDTextWatermarkLogitsProcessor")(
            ngram_len=self.ngram_len,
            keys=self.keys,
            sampling_table_size=self.sampling_table_size,
            sampling_table_seed=self.sampling_table_seed,
            context_history_size=self.context_history_size,
            device=device,
            skip_first_ngram_calls=self.skip_first_ngram_calls,
            debug_mode=self.debug_mode,
        )


for _name in _owner._WATERMARK_CONFIG_NAMES:
    _class = globals()[_name]
    _class.__module__ = _owner.__name__
    for _method in vars(_class).values():
        if isinstance(_method, (classmethod, staticmethod)):
            _method = _method.__func__
        if callable(_method) and hasattr(_method, "__module__"):
            _method.__module__ = _owner.__name__
    setattr(_owner, _name, _class)
