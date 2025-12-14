from typing import Any, TypeGuard

import phylogenie.generators.configs as cfg
import phylogenie.typeguards as tg
import phylogenie.typings as pgt


def is_many_scalar_configs(x: Any) -> TypeGuard[pgt.Many[cfg.Scalar]]:
    return tg.is_many(x) and all(isinstance(v, cfg.Scalar) for v in x)


def is_many_skyline_parameter_configs(
    x: Any,
) -> TypeGuard[pgt.Many[cfg.SkylineParameter]]:
    return tg.is_many(x) and all(isinstance(v, cfg.SkylineParameter) for v in x)


def is_skyline_vector_config(x: Any) -> TypeGuard[cfg.SkylineVector]:
    return isinstance(
        x, str | pgt.Scalar | cfg.SkylineVectorModel
    ) or is_many_skyline_parameter_configs(x)


def is_many_skyline_vector_configs(x: Any) -> TypeGuard[pgt.Many[cfg.SkylineVector]]:
    return tg.is_many(x) and all(is_skyline_vector_config(v) for v in x)
