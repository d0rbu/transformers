# Copyright 2024 The HuggingFace Team. All rights reserved.
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

import importlib as _importlib
import math as math
import warnings
from functools import wraps as wraps
from typing import TYPE_CHECKING, TypedDict
from typing import Any as _Any
from typing import Optional as Optional

from .utils import is_torch_available, logging


logger = logging.get_logger(__name__)


if TYPE_CHECKING:
    from ._rope_tensors import ROPE_INIT_FUNCTIONS as ROPE_INIT_FUNCTIONS
    from ._rope_tensors import _compute_dynamic_ntk_parameters as _compute_dynamic_ntk_parameters
    from ._rope_tensors import _compute_linear_scaling_rope_parameters as _compute_linear_scaling_rope_parameters
    from ._rope_tensors import _compute_llama3_parameters as _compute_llama3_parameters
    from ._rope_tensors import _compute_longrope_parameters as _compute_longrope_parameters
    from ._rope_tensors import _compute_yarn_parameters as _compute_yarn_parameters
    from ._rope_tensors import dynamic_rope_update as dynamic_rope_update
    from .configuration_utils import PreTrainedConfig


# This maps the "rope_type" string field in rope config to the corresponding function to compute the RoPE parameters
# from the model config. You can append new {'rope_type': callable} pairs to this rope_parameters to enable custom RoPE
# parameterizations, as long as the callable has the same signature.


class RopeParameters(TypedDict, total=False):
    """
    Args:
        rope_theta (`float`):
            The base period of the RoPE embeddings.
        rope_type (`str`, *optional*, defaults to "default"):
            The sub-variant of RoPE to use. Can be one of ['default', 'linear', 'dynamic', 'yarn', 'longrope',
            'llama3'], with 'default' being the original RoPE implementation.
        partial_rotary_factor (`float`, *optional*):
            The percentage of the query and key head embedding on which RoPE will be applied.
        factor (`float`, *optional*):
            Used with all rope types except 'default'. The scaling factor to apply to the RoPE embeddings. In
            most scaling types, a `factor` of x will enable the model to handle sequences of length x *
            original maximum pre-trained length.
        original_max_position_embeddings (`int`, *optional*):
            Used with 'yarn', 'longrope' and 'llama3'. The original max position embeddings used during
            pretraining.
        attention_factor (`float`, *optional*):
            Used with 'yarn' and 'longrope'. The scaling factor to be applied on the attention
            computation. If unspecified, it defaults to value recommended by the implementation, using the
            `factor` field to infer the suggested value.
        beta_fast (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for extrapolation (only) in the linear
            ramp function. If unspecified, it defaults to 32.
        beta_slow (`float`, *optional*):
            Only used with 'yarn'. Parameter to set the boundary for interpolation (only) in the linear
            ramp function. If unspecified, it defaults to 1.
        short_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to short contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        long_factor (`list[float]`, *optional*):
            Only used with 'longrope'. The scaling factor to be applied to long contexts (<
            `original_max_position_embeddings`). Must be a list of numbers with the same length as the hidden
            size divided by the number of attention heads divided by 2
        low_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to low frequency components of the RoPE
        high_freq_factor (`float`, *optional*):
            Only used with 'llama3'. Scaling factor applied to high frequency components of the RoPE
    """

    rope_theta: float
    rope_type: str | None
    partial_rotary_factor: float | None
    factor: float | None
    original_max_position_embeddings: int | None
    attention_factor: float | None
    beta_fast: float | None
    beta_slow: float | None
    short_factor: list[float] | None
    long_factor: list[float] | None
    low_freq_factor: float | None
    high_freq_factor: float | None


class RotaryEmbeddingConfigMixin:
    """
    A Mixin containing the functionality to standardize and validate RoPE parameters.
    """

    default_theta = 10_000.0
    ignore_keys_at_rope_validation = set()

    def convert_rope_params_to_dict(self, **kwargs):
        rope_scaling = kwargs.pop("rope_scaling", None)
        self.rope_parameters = rope_scaling or self.rope_parameters
        self.rope_parameters = self.rope_parameters if self.rope_parameters is not None else {}

        # Standardize and validate the correctness of rotary position embeddings parameters. Priority for these parameters is:
        # 1. Values in `rope_parameters` dict (where they should be after standardization)
        # 2. Values in `kwargs` (i.e. it's in config.json but not MyConfig.__init__'s args)
        # 3. Values in the config's attributes (i.e. it's in MyConfig.__init__'s args)
        # 4. Default values (i.e. not present at all but other RoPE parameters are present)
        rope_theta = kwargs.pop("rope_theta", getattr(self, "rope_theta", self.default_theta))
        self.rope_parameters.setdefault("rope_theta", rope_theta)

        partial_rotary_factor = kwargs.get("partial_rotary_factor", getattr(self, "partial_rotary_factor", None))
        if partial_rotary_factor is not None:
            self.rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
            self.ignore_keys_at_rope_validation = self.ignore_keys_at_rope_validation | {"partial_rotary_factor"}

        self.standardize_rope_params()
        return kwargs

    def standardize_rope_params(self):
        """
        Helper to standardize the config's rope params field by ensuring the params are defined for each
        later type. For old model the fn will duplicate a single rope param in each layer type (backward compatibility)
        """
        # Move `rope_theta` and `partial_rotary_factor` to the `rope_parameters`, if not there yet
        rope_theta = getattr(self, "rope_theta", None)
        partial_rotary_factor = getattr(self, "partial_rotary_factor", None)
        rope_parameters = getattr(self, "rope_parameters", None) or {}
        layer_types = getattr(self, "layer_types", None)

        # Case 0: no RoPE params defined
        if not (rope_parameters or rope_theta):
            # partial_rotary_factor without rope_theta is invalid, so we don't check for it here
            logger.warning("`standardize_rope_params` was called but no RoPE parameters were found.")
            return
        # Case 1: RoPE param keys do not intersect with possible `layer_types` -> one global dict
        elif layer_types is None or rope_parameters == {} or not set(rope_parameters.keys()).issubset(layer_types):
            rope_parameters.setdefault("rope_type", rope_parameters.get("type", "default"))
            rope_parameters.setdefault("rope_theta", rope_theta)
            if partial_rotary_factor is not None:
                rope_parameters["partial_rotary_factor"] = partial_rotary_factor

            # Move pretraining-time maximum length to rope parameter dict for RoPE types with scaling
            if rope_parameters["rope_type"] in ["llama3", "yarn", "longrope"]:
                if hasattr(self, "original_max_position_embeddings"):
                    # NOTE: Phi3 (and potentially other models) save `original_max_position_embeddings` field
                    # containing the pretrained value outside rope parameters. This is an exception case where we
                    # give priority to `self.original_max_position_embeddings
                    self.rope_parameters["original_max_position_embeddings"] = self.original_max_position_embeddings
                else:
                    self.rope_parameters.setdefault("original_max_position_embeddings", self.max_position_embeddings)

        # Case 2: different RoPE for each layer -> several params as nested dict
        else:
            for layer_type in set(layer_types):
                rope_parameters[layer_type].setdefault("rope_type", rope_parameters[layer_type].get("type", "default"))
                rope_parameters[layer_type].setdefault("rope_theta", rope_theta)
                if partial_rotary_factor is not None:
                    rope_parameters[layer_type]["partial_rotary_factor"] = partial_rotary_factor

                if rope_parameters[layer_type]["rope_type"] in ["llama3", "yarn", "longrope"]:
                    self.rope_parameters[layer_type].setdefault(
                        "original_max_position_embeddings", self.max_position_embeddings
                    )

        self.rope_parameters = rope_parameters

    def validate_rope(self: "PreTrainedConfig"):
        """
        Validate the RoPE config arguments, given a `"PreTrainedConfig"` object
        """
        # Don't validate if no rope_parameters found (`None`) or if it's an empty dict
        # Note that validation runs every time a new config is created, even if config is non-RoPE
        rope_parameters_dict = getattr(self, "rope_parameters", None)
        if not rope_parameters_dict:
            return

        if getattr(self, "layer_types", None) is not None and set(rope_parameters_dict.keys()).issubset(
            self.layer_types
        ):
            pass
        else:
            rope_parameters_dict = {"full_attention": rope_parameters_dict}

        for rope_parameters in rope_parameters_dict.values():
            rope_type = rope_parameters.get("rope_type", rope_parameters.get("type", "default"))
            validation_fn = getattr(self, f"_validate_{rope_type}_rope_parameters", None)
            rope_parameters["rope_type"] = rope_type

            if validation_fn is not None:
                validation_fn(rope_parameters, ignore_keys=self.ignore_keys_at_rope_validation)
            else:
                logger.warning(
                    f"Missing validation function in 'RotaryEmbeddingConfigMixin' for 'rope_type'='{rope_type}'"
                )

    def _validate_default_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

    def _validate_linear_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor", "rope_theta"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    def _validate_dynamic_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

    def _validate_yarn_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "factor", "rope_theta", "original_max_position_embeddings"}
        optional_keys = {
            "attention_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        }
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )
        beta_fast = rope_parameters.get("beta_fast")
        if beta_fast is not None and not isinstance(beta_fast, float):
            logger.warning(f"`rope_parameters`'s beta_fast field must be a float, got {beta_fast}")
        beta_slow = rope_parameters.get("beta_slow")
        if beta_slow is not None and not isinstance(beta_slow, float):
            logger.warning(f"`rope_parameters`'s beta_slow field must be a float, got {beta_slow}")

        if (beta_fast or 32) < (beta_slow or 1):
            logger.warning(
                f"`rope_parameters`'s beta_fast field must be greater than beta_slow, got beta_fast={beta_fast} "
                f"(defaults to 32 if None) and beta_slow={beta_slow} (defaults to 1 if None)"
            )

        # Double-check: `factor` should be the ratio between the pre-yarn and post-yarn context lengths.
        # NOTE: we might get `implicit_factor == 1` if config's `original_max_position_embeddings` was
        # inferred from `max_position_embeddings` during standardization
        original_max_position_embeddings = self.rope_parameters["original_max_position_embeddings"]
        implicit_factor = self.max_position_embeddings / original_max_position_embeddings
        if implicit_factor != factor and implicit_factor != 1:
            logger.warning_once(
                f"The explicitly set RoPE scaling factor (config.rope_parameters['factor'] = {factor}) does not match "
                "the ratio implicitly set by other parameters (implicit factor = "
                "post-yarn context length / pre-yarn context length = "
                "config.max_position_embeddings / config.rope_parameters['original_max_position_embeddings'] = "
                f"{implicit_factor}). Using the explicit factor ({factor}) in YaRN. This may cause unexpected "
                "behaviour in model usage, please correct the 'original_max_position_embeddings' fields in the model config."
            )

    def _validate_longrope_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {"rope_type", "short_factor", "long_factor", "rope_theta", "original_max_position_embeddings"}
        optional_keys = {"attention_factor", "factor"}
        received_keys = set(rope_parameters.keys())
        rope_type = rope_parameters["rope_type"]
        self._check_received_keys(rope_type, received_keys, required_keys, optional_keys, ignore_keys=ignore_keys)

        partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
        head_dim = getattr(self, "head_dim", self.hidden_size // self.num_attention_heads)
        dim = int(head_dim * partial_rotary_factor)

        short_factor = rope_parameters.get("short_factor")
        if not isinstance(short_factor, list) and all(isinstance(x, (int, float)) for x in short_factor):
            logger.warning(f"`rope_parameters`'s short_factor field must be a list of numbers, got {short_factor}")
        if len(short_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s short_factor field must have length {dim // 2}, got {len(short_factor)}"
            )

        long_factor = rope_parameters.get("long_factor")
        if not isinstance(long_factor, list) and all(isinstance(x, (int, float)) for x in long_factor):
            logger.warning(f"`rope_parameters`'s long_factor field must be a list of numbers, got {long_factor}")
        if len(long_factor) != dim // 2:
            logger.warning(
                f"`rope_parameters`'s long_factor field must have length {dim // 2}, got {len(long_factor)}"
            )

        factor = rope_parameters.get("factor")
        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]

        # Handle Phi3 divergence: we prefer the use of `attention_factor` and/or `factor` over
        # `original_max_position_embeddings` to compute internal variables. The latter is undesirable
        if factor is None and original_max_position_embeddings is not None:
            logger.warning_once(
                "This model config has set a `rope_parameters['original_max_position_embeddings']` field, to be used together with "
                "`max_position_embeddings` to determine a scaling factor. Please set the `factor` field of `rope_parameters`"
                "with this ratio instead -- we recommend the use of this field over `original_max_position_embeddings`, "
                "as it is compatible with most model architectures."
            )
        elif factor is None and original_max_position_embeddings is None:
            logger.warning("Missing required keys in `rope_parameters`: 'factor'")
        elif not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        attention_factor = rope_parameters.get("attention_factor")
        if attention_factor is not None and (not isinstance(attention_factor, float) or attention_factor < 0.0):
            logger.warning(
                f"`rope_parameters`'s attention_factor field must be a float greater than 0, got {attention_factor}"
            )

    def _validate_llama3_rope_parameters(self, rope_parameters: dict, ignore_keys: set | None = None):
        required_keys = {
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "low_freq_factor",
            "high_freq_factor",
            "rope_theta",
        }
        rope_type = rope_parameters["rope_type"]
        received_keys = set(rope_parameters.keys())
        self._check_received_keys(rope_type, received_keys, required_keys, ignore_keys=ignore_keys)

        factor = rope_parameters["factor"]
        if factor is None or not isinstance(factor, float) or factor < 1.0:
            logger.warning(f"`rope_parameters`'s factor field must be a float >= 1, got {factor}")

        low_freq_factor = rope_parameters["low_freq_factor"]
        high_freq_factor = rope_parameters["high_freq_factor"]
        if low_freq_factor is None or not isinstance(low_freq_factor, float):
            logger.warning(f"`rope_parameters`'s low_freq_factor field must be a float, got {low_freq_factor}")
        if high_freq_factor is None or not isinstance(high_freq_factor, float):
            logger.warning(f"`rope_parameters`'s high_freq_factor field must be a float, got {high_freq_factor}")
        if high_freq_factor <= low_freq_factor:
            logger.warning(
                "`rope_parameters`'s high_freq_factor field must be greater than low_freq_factor, got high_freq_factor="
                f"{high_freq_factor} and low_freq_factor={low_freq_factor}"
            )

        original_max_position_embeddings = rope_parameters["original_max_position_embeddings"]
        if original_max_position_embeddings is None or not isinstance(original_max_position_embeddings, int):
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be an integer, got "
                f"{original_max_position_embeddings}"
            )
        if original_max_position_embeddings >= self.max_position_embeddings:
            logger.warning(
                "`rope_parameters`'s original_max_position_embeddings field must be less than max_position_embeddings, got "
                f"{original_max_position_embeddings} and max_position_embeddings={self.max_position_embeddings}"
            )

    @staticmethod
    def _check_received_keys(
        rope_type: str,
        received_keys: set,
        required_keys: set,
        optional_keys: set | None = None,
        ignore_keys: set | None = None,
    ):
        """Compare the received keys in `config.rope_parameters` against the expected and optional keys"""
        # BC: "rope_type" was originally "type" -- let's check for "rope_type" when "type" is present
        if "type" in received_keys:
            received_keys -= {"type"}
            required_keys.add("rope_type")

        optional_keys = optional_keys or set()
        if "partial_rotary_factor" not in optional_keys:
            optional_keys.add("partial_rotary_factor")

        # Some models need to store model-specific keys, and we don't want to throw warning at them
        if ignore_keys is not None:
            received_keys -= ignore_keys

        missing_keys = required_keys - received_keys
        if missing_keys:
            raise KeyError(f"Missing required keys in `rope_parameters` for 'rope_type'='{rope_type}': {missing_keys}")

        unused_keys = received_keys - required_keys - optional_keys
        if unused_keys:
            logger.warning(f"Unrecognized keys in `rope_parameters` for 'rope_type'='{rope_type}': {unused_keys}")


def rope_config_validation(config: RotaryEmbeddingConfigMixin, ignore_keys: set | None = None):
    """
    This is a deprecated function.
    It has been kept for backward compatibility with custom code models.
    """
    warnings.warn(
        "`rope_config_validation` is deprecated and has been removed. "
        "Its functionality has been moved to RotaryEmbeddingConfigMixin.validate_rope method. "
        "PreTrainedConfig inherits this class, so please call self.validate_rope() instead. "
        "Also, make sure to use the new rope_parameters syntax. "
        "You can call self.standardize_rope_params() in the meantime.",
        FutureWarning,
    )
    config.standardize_rope_params()
    config.validate_rope()


_LAZY_ATTRIBUTES = {
    "dynamic_rope_update": ("._rope_tensors", "dynamic_rope_update"),
    "_compute_linear_scaling_rope_parameters": ("._rope_tensors", "_compute_linear_scaling_rope_parameters"),
    "_compute_dynamic_ntk_parameters": ("._rope_tensors", "_compute_dynamic_ntk_parameters"),
    "_compute_yarn_parameters": ("._rope_tensors", "_compute_yarn_parameters"),
    "_compute_longrope_parameters": ("._rope_tensors", "_compute_longrope_parameters"),
    "_compute_llama3_parameters": ("._rope_tensors", "_compute_llama3_parameters"),
    "ROPE_INIT_FUNCTIONS": ("._rope_tensors", "ROPE_INIT_FUNCTIONS"),
}


def __getattr__(name: str) -> _Any:
    if name == "__all__":
        return [key for key in __dir__() if not key.startswith("_")]
    if name in globals():
        return globals()[name]
    if name == "torch" and is_torch_available():
        value = _importlib.import_module("torch")
        globals()[name] = value
        return value
    if name in _LAZY_ATTRIBUTES:
        module, attribute = _LAZY_ATTRIBUTES[name]
        value = getattr(_importlib.import_module(module, __package__), attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    names = set(globals()) | set(_LAZY_ATTRIBUTES)
    if is_torch_available():
        names.add("torch")
    return sorted(names)
