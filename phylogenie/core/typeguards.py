from typing import TypeGuard

import phylogenie.core.configs as cfg


def is_list(x: object) -> TypeGuard[list[object]]:
    return isinstance(x, list)


def is_list_of_scalar_configs(x: object) -> TypeGuard[list[cfg.ScalarConfig]]:
    return is_list(x) and all(isinstance(v, cfg.ScalarConfig) for v in x)


def is_list_of_skyline_parameter_like_configs(
    x: object,
) -> TypeGuard[list[cfg.SkylineParameterLikeConfig]]:
    return is_list(x) and all(isinstance(v, cfg.SkylineParameterLikeConfig) for v in x)


def is_list_of_skyline_vector_like_configs(
    x: object,
) -> TypeGuard[list[cfg.SkylineVectorLikeConfig]]:
    return is_list(x) and all(
        isinstance(v, str | cfg.SkylineVectorValueModel)
        or is_list_of_skyline_parameter_like_configs(v)
        for v in x
    )
