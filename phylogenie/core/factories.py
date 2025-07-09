from typing import Any

import numpy as np

import phylogenie.core.configs as cfg
import phylogenie.core.typeguards as ctg
import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.core.typings import Data
from phylogenie.skyline import (
    SkylineMatrix,
    SkylineMatrixLike,
    SkylineParameter,
    SkylineParameterLike,
    SkylineVector,
    SkylineVectorLike,
)


def _eval_expression(expression: str, data: Data) -> Any:
    return np.array(
        eval(
            expression,
            {"__builtins__": __builtins__},
            {k: np.array(v) for k, v in data.items()},
        )
    ).tolist()


def int_factory(x: cfg.IntConfig, data: Data) -> int:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, int):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int."
        )
    return x


def scalar_factory(x: cfg.ScalarConfig, data: Data) -> pgt.Scalar:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar."
        )
    return x


def one_or_many_scalars_factory(
    x: cfg.OneOrManyScalarsConfig, data: Data
) -> pgt.OneOrMany[pgt.Scalar]:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar or a sequence of them."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return [scalar_factory(v, data) for v in x]


def one_or_many_ints_factory(
    x: cfg.OneOrManyIntsConfig, data: Data
) -> pgt.OneOrMany[int]:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_ints(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int or a sequence of them."
        )
    if isinstance(x, int):
        return x
    return [int_factory(v, data) for v in x]


def skyline_parameter_like_factory(
    x: cfg.SkylineParameterLikeConfig, data: Data
) -> SkylineParameterLike:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineParameterLike (e.g., a scalar)."
        )
    if isinstance(x, pgt.Scalar):
        return x

    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which is not a valid value for a SkylineParameter (expected a sequence of scalars)."
            )
    else:
        value = [scalar_factory(v, data) for v in x.value]
    return SkylineParameter(
        value=value,
        change_times=one_or_many_scalars_factory(x.change_times, data),
    )


def skyline_vector_like_factory(
    x: cfg.SkylineVectorLikeConfig, data: Data
) -> SkylineVectorLike:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineVectorLike object (e.g., a scalar or a sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_parameter_like_configs(x):
        return [skyline_parameter_like_factory(p, data) for p in x]

    assert isinstance(x, cfg.SkylineVectorValueModel)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which is not a valid value for a SkylineVector (expected a sequence containing scalars and/or sequences of them)."
            )
    else:
        value = [one_or_many_scalars_factory(x, data) for x in x.value]
    return SkylineVector(
        value=value,
        change_times=one_or_many_scalars_factory(x.change_times, data),
    )


def skyline_matrix_like_factory(
    x: cfg.SkylineMatrixLikeConfig, data: Data
) -> SkylineMatrixLike:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_scalars(e) or tg.is_many_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineMatrixLike object (e.g., a scalar, a sequence of them, or a sequence containing scalars and/or sequences of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_vector_like_configs(x):
        return [skyline_vector_like_factory(v, data) for v in x]

    assert isinstance(x, cfg.SkylineMatrixValueModel)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_2d_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which is not a valid value for a SkylineMatrix (expected a sequence containing scalars and/or nested (2D) sequences of them)."
            )
    else:
        value = [
            (
                scalar_factory(v, data)
                if isinstance(v, cfg.ScalarConfig)
                else [[scalar_factory(row, data) for row in m] for m in v]
            )
            for v in x.value
        ]
    return SkylineMatrix(
        value=value, change_times=one_or_many_scalars_factory(x.change_times, data)
    )
