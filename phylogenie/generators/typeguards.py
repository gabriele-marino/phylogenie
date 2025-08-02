from typing import Any, TypeGuard

import phylogenie.generators.configs as cfg
import phylogenie.typings as pgt


def is_list(x: Any) -> TypeGuard[list[Any]]:
    return isinstance(x, list)


def is_list_of_scalar_configs(x: Any) -> TypeGuard[list[cfg.Scalar]]:
    return is_list(x) and all(isinstance(v, cfg.Scalar) for v in x)


def is_list_of_skyline_parameter_configs(
    x: Any,
) -> TypeGuard[list[cfg.SkylineParameter]]:
    return is_list(x) and all(isinstance(v, cfg.SkylineParameter) for v in x)


def is_skyline_vector_config(x: Any) -> TypeGuard[cfg.SkylineVector]:
    return isinstance(
        x, str | pgt.Scalar | cfg.SkylineVectorModel
    ) or is_list_of_skyline_parameter_configs(x)


def is_list_of_skyline_vector_configs(x: Any) -> TypeGuard[list[cfg.SkylineVector]]:
    return is_list(x) and all(is_skyline_vector_config(v) for v in x)
