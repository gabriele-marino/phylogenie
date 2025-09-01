import re
from typing import Any

import numpy as np
from numpy.random import Generator

import phylogenie.generators.configs as cfg
import phylogenie.generators.typeguards as ctg
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
from phylogenie.utils import Distribution


def _eval_expression(expression: str, data: dict[str, Any]) -> Any:
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


def integer(x: cfg.Integer, data: dict[str, Any]) -> int:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, int):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int."
        )
    return x


def scalar(x: cfg.Scalar, data: dict[str, Any]) -> pgt.Scalar:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar."
        )
    return x


def string(s: Any, data: dict[str, Any]) -> str:
    if not isinstance(s, str):
        return str(s)
    return re.sub(
        r"\{([^{}]+)\}", lambda match: str(_eval_expression(match.group(1), data)), s
    )  # Match content inside curly braces


def many_scalars(x: cfg.ManyScalars, data: dict[str, Any]) -> pgt.ManyScalars:
    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a sequence of scalars."
        )
    return [scalar(v, data) for v in x]


def one_or_many_scalars(
    x: cfg.OneOrManyScalars, data: dict[str, Any]
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
    return many_scalars(x, data)


def skyline_parameter(
    x: cfg.SkylineParameter, data: dict[str, Any]
) -> SkylineParameterLike:
    if isinstance(x, cfg.Scalar):
        return scalar(x, data)
    return SkylineParameter(
        value=many_scalars(x.value, data),
        change_times=many_scalars(x.change_times, data),
    )


def skyline_vector(
    x: cfg.SkylineVector, data: dict[str, Any]
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
    if ctg.is_many_skyline_parameter_configs(x):
        return [skyline_parameter(p, data) for p in x]

    assert isinstance(x, cfg.SkylineVectorModel)

    change_times = many_scalars(x.change_times, data)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineVector (expected a sequence composed of scalars and/or sequences of scalars)."
            )
    else:
        value = [one_or_many_scalars(v, data) for v in x.value]

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


def one_or_many_2D_scalars(
    x: cfg.OneOrMany2DScalars, data: dict[str, Any]
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
    return [many_scalars(v, data) for v in x]


def skyline_matrix(
    x: cfg.SkylineMatrix, data: dict[str, Any]
) -> SkylineMatrixCoercible | None:
    if x is None:
        return None

    if isinstance(x, str):
        e = _eval_expression(x, data)
        if tg.is_one_or_many_2D_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineMatrixCoercible object (e.g., a scalar or a nested (2D) sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_vector_configs(x):
        return [skyline_vector(v, data) for v in x]

    assert isinstance(x, cfg.SkylineMatrixModel)

    change_times = many_scalars(x.change_times, data)
    if isinstance(x.value, str):
        e = _eval_expression(x.value, data)
        if tg.is_many_one_or_many_2D_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineMatrix (expected a sequence composed of scalars and/or nested (2D) sequences of scalars)."
            )
    else:
        value = [one_or_many_2D_scalars(v, data) for v in x.value]

    if tg.is_many_scalars(value):
        return SkylineParameter(value=value, change_times=change_times)

    shapes: set[tuple[int, int]] = set()
    for elem in value:
        if tg.is_many_2D_scalars(elem):
            Ms = len(elem)
            Ns = {len(row) for row in elem}
            if len(Ns) > 1:
                raise ValueError(
                    f"The values of a SkylineMatrix config must be scalars or nested (2D) lists of them with a consistent row length (config {x.value} yielded element {elem} with row lengths {Ns})."
                )
            shapes.add((Ms, Ns.pop()))

    if len(shapes) > 1:
        raise ValueError(
            f"All elements in the value of a SkylineMatrix config must be scalars or nested (2D) lists of them with the same shape (config {x.value} yielded value={value} with inconsistent shapes {shapes})."
        )
    ((M, N),) = shapes
    value = [[[e] * N] * M if isinstance(e, pgt.Scalar) else e for e in value]

    return SkylineMatrix(value=value, change_times=change_times)


def distribution(x: Distribution, data: dict[str, Any]) -> Distribution:
    args = x.args
    for arg_name, arg_value in args.items():
        if isinstance(arg_value, str):
            args[arg_name] = _eval_expression(arg_value, data)
    return Distribution(type=x.type, **args)


def data(context: dict[str, Distribution] | None, rng: Generator) -> dict[str, Any]:
    if context is None:
        return {}
    data: dict[str, Any] = {}
    for k, v in context.items():
        dist = distribution(v, data)
        data[k] = np.array(getattr(rng, dist.type)(**dist.args)).tolist()
    return data
