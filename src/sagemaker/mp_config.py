"""Parses and validates configurations for modelparallel arguments.

This validation is done in Python SDK so that validation errors are
immediately raised, before launching the training job.
"""
"""Rubik config."""

# Standard Library
import json
import os
import re
from pydoc import locate

# Third Party
import yaml

# NOTE: this file is duplicated across smp and SageMaker Python SDK.
# any changes to this file requires an update of the corresponding file
# in SageMaker Python SDK upon release.

try:
    import smdistributed.modelparallel.torch as smp

    py_sdk = False
except ImportError:
    # the case where this file runs on Python SDK for config validation
    info_func = print
    warn_func = print
    py_sdk = True
    SMPInvalidArgumentError = ValueError

_TYPE = "type"
_OPTIONS = "options"


def int_or_float_if_possible(s):
    """Attempt to convert to int, or float."""
    try:
        float_s = float(s)
        int_s = int(float_s)
        return int_s if int_s == float_s else float_s
    except ValueError:
        return s


class ConfigParam:
    """Represents and validates a single modelparallel configuration parameter."""

    def __init__(self, name, input_value, cfg_dict, existing_params, provided=False):
        """Initialize ConfigParam."""
        self.name = name
        self.cfg_dict = cfg_dict
        self.existing_params = existing_params
        self.provided = provided
        # Assigning to a property uses its property setter.
        # Here, the property setter of `value` is called
        # which writes to the hidden attribute `self._value`.
        self.value = input_value

    @property
    def value(self):
        """Return the underlying value of the configuration parameter."""
        # self._value is created by property.
        return self._value  # pylint: disable=no-member

    def _get_default(self):
        """Get default value for the parameter."""
        default = self._handle_dependencies(self.cfg_dict["default"])
        if isinstance(default, (float, int)):
            # should explicitly enforce upper and lower bounds for dynamic formulas in case
            # the default evaluates out of bounds
            if "upper_bound" in self.cfg_dict:
                default = min(default, self._handle_dependencies(self.cfg_dict["upper_bound"]))
            if "lower_bound" in self.cfg_dict:
                default = max(default, self._handle_dependencies(self.cfg_dict["lower_bound"]))

        return default

    @value.setter
    def value(self, input_value):
        """Set the value for the parameter."""
        if not self.provided:
            if "default" not in self.cfg_dict:
                raise SMPInvalidArgumentError(f"Config parameter {self.name} is required.")

            self._value = self._get_default()
            return

        self._check_type(input_value, self.name, self.cfg_dict)
        self._check_options(input_value, self.name, self.cfg_dict)

        self._check_bounds_in_config(input_value, "lower_bound", self.name, self.cfg_dict)
        self._check_bounds_in_config(input_value, "upper_bound", self.name, self.cfg_dict)

        self._check_requires(input_value, "requires", self.name, self.cfg_dict)
        self._check_requires(input_value, "requires_not", self.name, self.cfg_dict)
        self._check_requires(input_value, "requires_either", self.name, self.cfg_dict)

        self._value = input_value

    def _check_type(self, input_value, name, config_dict):
        """Check if input_value type is valid."""
        if _TYPE in config_dict:
            expected_types = self._parse_allowed_types(config_dict[_TYPE])
            if type(input_value) not in expected_types:
                raise TypeError(
                    f"Config parameter {name} type needs to be one of {[e.__name__ for e in expected_types]}. Found: {type(input_value).__name__}."
                )

    def _check_options(self, input_value, name, config_dict):
        """Check if input_value is an available option."""
        if _OPTIONS in config_dict:
            options = self._handle_dependencies(config_dict[_OPTIONS])
            if input_value not in options:
                raise SMPInvalidArgumentError(
                    f"Config parameter {name} must be one of {config_dict[_OPTIONS]}. Found: {input_value}."
                )

    def _check_bounds_in_config(self, input_value, bound_type, name, config_dict):
        """Check input_value is within bounds specified in config."""
        if bound_type not in config_dict:
            return
        bound = self._handle_dependencies(config_dict[bound_type])
        if bound_type == "upper_bound":
            if input_value > bound:
                raise SMPInvalidArgumentError(
                    f"Config parameter {name} ({input_value}) cannot be larger than {config_dict[bound_type]} ({bound})."
                )
        elif bound_type == "lower_bound":
            if input_value < bound:
                raise SMPInvalidArgumentError(
                    f"Config parameter {name} ({input_value}) cannot be less than {config_dict[bound_type]} ({bound})."
                )
        else:
            raise SMPInvalidArgumentError(
                f"Error: the only inputs to this `check_bounds_in_config()` should be 'upper_bound' or 'lower_bound'. This is a bug and should be fixed."
            )

    def _check_requires(self, input_value, requires_type, name, config_dict):
        """Check given requires type for the input value."""
        if requires_type not in config_dict:
            return
        default = self._get_default()

        if requires_type not in ("requires", "requires_not", "requires_either"):
            raise SMPInvalidArgumentError(
                f"Error: the only inputs to this `_check_requires()` should be 'requires', 'requires_not', or 'requires_either'. This is a bug and should be fixed."
            )

        if requires_type in ("requires", "requires_not"):
            for k, v in config_dict[requires_type].items():
                if requires_type == "requires":
                    if self.existing_params[k].value != v and input_value != default:
                        raise SMPInvalidArgumentError(
                            f"Setting config parameter {name} to non-default value {input_value} requires {k} to be set to {v}. Found: {self.existing_params[k].value}"
                        )
                else:
                    # requires_not
                    if self.existing_params[k].value == v and input_value != default:
                        raise SMPInvalidArgumentError(
                            f"Setting config parameter {name} to non-default value {input_value} requires {k} to not be {v}."
                        )
            return

        # requires_type == requires_either
        if input_value != default:
            provided_configs = {}
            requirement_satisfied = False
            for k, v in config_dict["requires_either"].items():
                if self.existing_params[k].value == v:
                    requirement_satisfied = True
                    break
                provided_configs[k] = self.existing_params[k].value
            if not requirement_satisfied:
                raise SMPInvalidArgumentError(
                    f"Setting config parameter {name} to non-default value {input_value} requires either of following configs: {config_dict['requires_either']} But the configs found are: {provided_configs}"
                )

    def _maybe_convert(self, value):
        """Convert the value to int or float if possible."""
        if value[0] == "(" and value[-1] == ")" and value[1:-1] in self.existing_params:
            value = self.existing_params[value[1:-1]].value

        return int_or_float_if_possible(value)

    def _handle_dependencies(self, value):
        """If value depends on another one, parse the formula."""
        if isinstance(value, str):
            tokens = re.split("\\+|\\-|\\*|\\/", value)
            ops = [c for c in value if c in ["+", "-", "*", "/"]]

            assert len(tokens) == len(ops) + 1, f"Malformed formula: {value}"

            # if there are operations, all terms must be convertible to float or int
            # if not, cur_value can be a string
            tokens = [self._maybe_convert(token.strip()) for token in tokens]
            cur_value = tokens[0]
            for op, val in zip(ops, tokens[1:]):
                if op == "+":
                    cur_value += val
                elif op == "-":
                    cur_value -= val
                elif op == "*":
                    cur_value *= val
                elif op == "/":
                    cur_value /= val
            return cur_value
        return value

    def _parse_allowed_types(self, types):
        """Parse allowed types."""
        def _parse(t):
            if t is None:
                return type(None)
            if isinstance(t, str):
                return locate(t)
            raise ValueError(f"Invalid type {t} in config schema.")

        if isinstance(types, list):
            return [_parse(typ) for typ in types]
        return [_parse(types)]


class DependencyIterator:
    """An iterator that traverses the configuration parameters in topological sort.

    If a parameter a has dependency to another parameter b, then b is guaranteed to
    be returned before a.
    """

    def __init__(self, config):
        """Initialize DependencyIterator."""
        self.config = config
        self.seen = set()

    def __iter__(self):
        """Create the iterator."""
        return self

    def __next__(self):
        """Return the next element in topological sort."""
        for k in self.config:
            if k not in self.seen:
                if "dependencies" not in self.config[k]:
                    self.seen.add(k)
                    return k

                if all([(d in self.seen) for d in self.config[k]["dependencies"]]):
                    self.seen.add(k)
                    return k

        raise StopIteration


class ModelParallelConfig:
    """Structure that holds the user-defined parameters for SMP."""

    def __init__(self, config):
        """Initialize ModelParallelConfig."""
        if not py_sdk:
            SM_CONFIG = json.loads(os.environ.get("SM_HP_MP_PARAMETERS", default="{}"))
            for each_sm_config in SM_CONFIG:
                config[each_sm_config] = SM_CONFIG[each_sm_config]

        # Load the schema from yaml file.
        dir_path = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(dir_path, "config.yaml")
        schema = self.load_schema(config_path)

        # Handle aliases.
        config = self.handle_aliases(schema, config)

        # Check no invalid config parameters.
        # Parse and validate the inputs.
        params = self.parse_and_validate(schema, config)

        # Set the attributes.
        for k in params:
            setattr(self, k, params[k].value)

        self._input_config = config
        self._config_dict = params

        # Enable fp16 and fp16_param backward compatibility.
        self._fp16_param_init = self.fp16 or self.fp16_params

        # Enforce additional constraints.
        self.enforce_additional_constraints()

        self._zero2d_config_dict = {}

    def load_schema(self, path):
        """Load schema from given path."""
        with open(path, "r") as f:  # pylint: disable=invalid-name,unspecified-encoding
            schema = yaml.safe_load(f)
        return schema

    def handle_aliases(self, schema, config):
        """Handle aliases."""
        # Note: below logic assumes at most one alias per config parameter.
        aliases = {v["alias"]: k for k, v in schema.items() if "alias" in v}
        for alias, orig in aliases.items():
            if alias in config:
                if orig in config and config[alias] != config[orig]:
                    raise SMPInvalidArgumentError(
                        f"Conflicting values {config[orig]} and {config[alias]} are provided for config parameter {orig} and its alias {alias}."
                    )
                config[orig] = config[alias]
                del config[alias]
        return config

    def parse_and_validate(self, schema, config):
        """Parse and validate given config with schema."""
        # Make sure there are no invalid config parameters.
        for k in config:
            if k not in schema:
                raise SMPInvalidArgumentError(f"Unrecognized config parameter {k}.")

        params = {}
        for k in DependencyIterator(schema):
            provided = k in config
            input_value = config[k] if provided else None
            params[k] = ConfigParam(k, input_value, schema[k], params, provided)
        return params

    def enforce_additional_constraints(self):
        """Enforce additional constraints."""
        # Need to be careful here to make sure these do not conflict with the
        # existing constraints.
        if self.active_microbatches != self.microbatches and self.pipeline != "interleaved":
            # PT limitation right now
            self.pipeline = "interleaved"
            info_func(
                "Simple pipeline is only supported when 'active_microbatches' is equal to 'microbatches'. Using interleaved pipeline instead."
            )

        if self.pipeline_parallel_degree > 1 and self.checkpoint_attentions:
            warn_func(
                "Cannot checkpoint attentions when pipeline-parallel degree is more than 1, disabling attention checkpointing."
            )
            self.checkpoint_attentions = False

        if (
            self.sharded_data_parallel_degree > 1
            and self.tensor_parallel_degree > 1
        ):
            # If this is not valid, then we need additional logic to handle non TP distributed
            # parameters in ZeRO-2D
            if not self.prescaled_batch:
                raise SMPInvalidArgumentError(
                    "When using combination of SDP and TP, prescaled_batch needs to be True, meaning all GPUs in a tp_group receive the same data"
                )

    def zero2d_enabled(self):
        """Return if zero2d is enabled."""
        return self.sharded_data_parallel_degree > 1

    def zero2d_config_dict(self):
        """Get the zero2d config dictionary."""
        return self._zero2d_config_dict

    def construct_zero2d_config_dict(self, smp_core):
        """Construct the zero2d config dictionary."""
        if not py_sdk:
            from smdistributed.modelparallel.backend.zero_config import construct_zero2d_config_dict

            self._zero2d_config_dict = construct_zero2d_config_dict(self, smp_core)

    def get_config_dict(self):
        """Get the model parallel config dictionary."""
        if not hasattr(self, "_config_dict"):
            raise SMPInvalidArgumentError("ModelParallelConfig should contain _config_dict attr")
        return {key: val.value for key, val in self._config_dict.items()}

    def display_config(self):
        """Display the parsed configuration."""
        assert hasattr(self, "_config_dict")

        deprecated_configs = {}
        if py_sdk:
            return

        info_func("Configuration parameters:")
        for k, v in sorted(self._config_dict.items()):
            if "internal" not in v.cfg_dict or not v.cfg_dict["internal"]:
                info_func(f"  {k}: {v.value}")
            if "deprecated" in v.cfg_dict and v.cfg_dict["deprecated"]:
                deprecated_configs[k] = v

        for k, v in sorted(deprecated_configs.items()):
            if "replacement" in v.cfg_dict:
                warn_func(
                    f"WARNING: \"{k}\" is a deprecated config key, please use \"{v.cfg_dict['replacement']}\" instead"
                )
            else:
                warn_func(f'WARNING: "{k}" is a deprecated config key')
