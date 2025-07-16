from typing import Any

import numpy as np

import phylogenie.core.configs as cfg
import phylogenie.core.typeguards as ctg
import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.skyline import (
    SkylineMatrix,
    SkylineMatrixCoercible,
    SkylineParameter,
    SkylineParameterLike,
    SkylineVector,
    SkylineVectorCoercible,
)


def _eval_expression(expression: str, data: pgt.Data) -> Any:
    return np.array(
        eval(
            expression,
            {
                "__builtins__": __builtins__,
                "np": np,
                **{k: np.array(v) for k, v in data.items()},
            },
        )
    ).tolist()


def int_factory(x: cfg.IntConfig, data: pgt.Data) -> int:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, int):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int."
        )
    return x


def scalar_factory(x: cfg.ScalarConfig, data: pgt.Data) -> pgt.Scalar:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar."
        )
    return x


def many_ints_factory(x: cfg.ManyIntsConfig, data: pgt.Data) -> pgt.Many[int]:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_many_ints(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a sequence of integers."
        )
    return [int_factory(v, data) for v in x]


def many_scalars_factory(x: cfg.ManyScalarsConfig, data: pgt.Data) -> pgt.ManyScalars:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a sequence of scalars."
        )
    return [scalar_factory(v, data) for v in x]


def one_or_many_scalars_factory(
    x: cfg.OneOrManyScalarsConfig, data: pgt.Data
) -> pgt.OneOrManyScalars:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar or a sequence of them."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return many_scalars_factory(x, data)


def skyline_parameter_like_factory(
    x: cfg.SkylineParameterLikeConfig, data: pgt.Data
) -> SkylineParameterLike:
    if isinstance(x, cfg.ScalarConfig):
        return scalar_factory(x, data)
    return SkylineParameter(
        value=many_scalars_factory(x.value, data),
        change_times=many_scalars_factory(x.change_times, data),
    )


def skyline_vector_coercible_factory(
    x: cfg.SkylineVectorCoercibleConfig, data: pgt.Data
) -> SkylineVectorCoercible:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineVectorCoercible object (e.g., a scalar or a sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_list_of_skyline_parameter_like_configs(x):
        return [skyline_parameter_like_factory(p, data) for p in x]

    assert isinstance(x, cfg.SkylineVectorValueModel)

    change_times = many_scalars_factory(x.change_times, data)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineVector (expected a sequence composed of scalars and/or sequences of scalars)."
            )
    else:
        value = [one_or_many_scalars_factory(v, data) for v in x.value]

    if tg.is_many_scalars(value):
        return SkylineParameter(value=value, change_times=change_times)

    Ns = {len(elem) for elem in value if tg.is_many(elem)}
    if len(Ns) > 1:
        raise ValueError(
            f"All elements in the value of a SkylineVector config must be scalars or have the same length (config {x.value} yielded value={value} with inconsistent lengths {Ns})."
        )
    (N,) = Ns
    value = [[p] * N if isinstance(p, pgt.Scalar) else p for p in value]

    return SkylineVector(value=value, change_times=change_times)


def one_or_many_2D_scalars_factory(
    x: cfg.OneOrMany2DScalarsConfig, data: pgt.Data
) -> pgt.OneOrMany2DScalars:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_2D_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a nested (2D) sequence of scalars."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return [many_scalars_factory(v, data) for v in x]


def skyline_matrix_coercible_factory(
    x: cfg.SkylineMatrixCoercibleConfig, data: pgt.Data
) -> SkylineMatrixCoercible:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_2D_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineMatrixCoercible object (e.g., a scalar or a nested (2D) sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_list_of_skyline_vector_coercible_configs(x):
        return [skyline_vector_coercible_factory(v, data) for v in x]

    assert isinstance(x, cfg.SkylineMatrixValueModel)

    change_times = many_scalars_factory(x.change_times, data)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_2D_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineMatrix (expected a sequence composed of scalars and/or nested (2D) sequences of scalars)."
            )
    else:
        value = [one_or_many_2D_scalars_factory(v, data) for v in x.value]

    if tg.is_many_scalars(value):
        return SkylineParameter(value=value, change_times=change_times)

    Ns: set[int] = set()
    for elem in value:
        if tg.is_many_2D_scalars(elem):
            n_rows = len(elem)
            if any(len(row) != n_rows for row in elem):
                raise ValueError(
                    f"All elements in the value of a SkylineMatrix config must be scalars or square matrices (config {x.value} yeilded a non-square matrix: {elem})."
                )
            Ns.add(n_rows)

    if len(Ns) > 1:
        raise ValueError(
            f"All elements in the value of a SkylineMatrix config must be scalars or have the same square shape (config {x.value} yielded value={value} with inconsistent lengths {Ns})."
        )
    (N,) = Ns
    value = [[[p] * N] * N if isinstance(p, pgt.Scalar) else p for p in value]

    return SkylineMatrix(value=value, change_times=change_times)
