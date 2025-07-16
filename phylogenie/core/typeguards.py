from typing import TypeGuard

import phylogenie.core.configs as cfg
import phylogenie.typings as pgt


def is_list(x: object) -> TypeGuard[list[object]]:
    return isinstance(x, list)


def is_list_of_scalar_configs(x: object) -> TypeGuard[list[cfg.ScalarConfig]]:
    return is_list(x) and all(isinstance(v, cfg.ScalarConfig) for v in x)


def is_list_of_skyline_parameter_like_configs(
    x: object,
) -> TypeGuard[list[cfg.SkylineParameterLikeConfig]]:
    return is_list(x) and all(isinstance(v, cfg.SkylineParameterLikeConfig) for v in x)


def is_skyline_vector_coercible_config(
    x: object,
) -> TypeGuard[cfg.SkylineVectorCoercibleConfig]:
    return isinstance(
        x, str | pgt.Scalar | cfg.SkylineVectorValueModel
    ) or is_list_of_skyline_parameter_like_configs(x)


def is_list_of_skyline_vector_coercible_configs(
    x: object,
) -> TypeGuard[list[cfg.SkylineVectorCoercibleConfig]]:
    return is_list(x) and all(is_skyline_vector_coercible_config(v) for v in x)
