from typing import TypeGuard

import phylogenie.core.configs as cfg
import phylogenie.typeguards as tg
import phylogenie.typings as pgt


def is_many_scalar_configs(x: object) -> TypeGuard[pgt.Many[cfg.ScalarConfig]]:
    return tg.is_many(x) and all(isinstance(v, cfg.ScalarConfig) for v in x)


def is_many_2D_scalar_configs(x: object) -> TypeGuard[pgt.Many2D[cfg.ScalarConfig]]:
    return tg.is_many(x) and all(is_many_scalar_configs(v) for v in x)


def is_many_3D_scalar_configs(x: object) -> TypeGuard[pgt.Many3D[cfg.ScalarConfig]]:
    return tg.is_many(x) and all(is_many_2D_scalar_configs(v) for v in x)


def is_many_skyline_parameter_like_configs(
    x: object,
) -> TypeGuard[pgt.Many[cfg.SkylineParameterLikeConfig]]:
    return tg.is_many(x) and all(
        isinstance(v, cfg.SkylineParameterLikeConfig) for v in x
    )


def is_many_skyline_vector_like_configs(
    x: object,
) -> TypeGuard[pgt.Many[cfg.SkylineVectorLikeConfig]]:
    return tg.is_many(x) and all(
        isinstance(v, str | pgt.Scalar | cfg.SkylineVectorValueModel)
        or is_many_skyline_parameter_like_configs(v)
        for v in x
    )
