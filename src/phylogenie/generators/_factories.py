import re
from typing import Any

import numpy as np
from numpy.random import default_rng

import phylogenie._typeguards as tg
import phylogenie._typings as pgt
import phylogenie.generators._configs as cfg
import phylogenie.generators._typeguards as ctg
from phylogenie.skyline import (
    SkylineMatrix,
    SkylineMatrixCoercible,
    SkylineParameter,
    SkylineParameterLike,
    SkylineVector,
    SkylineVectorCoercible,
)
from phylogenie.treesimulator import TimedEvent
from phylogenie.treesimulator.parameterizations.common import TimedSampling


def eval_expression(
    expression: str,
    context: dict[str, Any],
    extra_globals: dict[str, Any] | None = None,
) -> Any:
    if extra_globals is None:
        extra_globals = {}
    return np.array(
        eval(
            expression,
            {"np": np, **{k: np.array(v) for k, v in context.items()}, **extra_globals},
        )
    ).tolist()


def integer(x: cfg.Integer, context: dict[str, Any]) -> int:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if isinstance(e, int):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int."
        )
    return x


def scalar(x: cfg.Scalar, context: dict[str, Any]) -> pgt.Scalar:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar."
        )
    return x


def string(s: Any, context: dict[str, Any]) -> str:
    if not isinstance(s, str):
        return str(s)
    return re.sub(
        r"\{([^{}]+)\}", lambda match: str(eval_expression(match.group(1), context)), s
    )  # Match content inside curly braces


def many_scalars(x: cfg.ManyScalars, context: dict[str, Any]) -> pgt.ManyScalars:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if tg.is_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a sequence of scalars."
        )
    return [scalar(v, context) for v in x]


def one_or_many_scalars(
    x: cfg.OneOrManyScalars, context: dict[str, Any]
) -> pgt.OneOrManyScalars:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar or a sequence of them."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return many_scalars(x, context)


def skyline_parameter(
    x: cfg.SkylineParameter, context: dict[str, Any]
) -> SkylineParameterLike:
    if isinstance(x, cfg.Scalar):
        return scalar(x, context)
    return SkylineParameter(
        value=many_scalars(x.value, context),
        change_times=many_scalars(x.change_times, context),
    )


def skyline_vector(
    x: cfg.SkylineVector, context: dict[str, Any]
) -> SkylineVectorCoercible:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineVectorCoercible object (e.g., a scalar or a sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_parameter_configs(x):
        return [skyline_parameter(p, context) for p in x]

    assert isinstance(x, cfg.SkylineVectorModel)

    change_times = many_scalars(x.change_times, context)
    if isinstance(x.value, str):
        e = eval_expression(x.value, context)
        if tg.is_many_one_or_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineVector (expected a sequence composed of scalars and/or sequences of scalars)."
            )
    else:
        value = [one_or_many_scalars(v, context) for v in x.value]

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
    x: cfg.OneOrMany2DScalars, context: dict[str, Any]
) -> pgt.OneOrMany2DScalars:
    if isinstance(x, str):
        e = eval_expression(x, context)
        if tg.is_one_or_many_2D_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a nested (2D) sequence of scalars."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return [many_scalars(v, context) for v in x]


def skyline_matrix(
    x: cfg.SkylineMatrix, context: dict[str, Any]
) -> SkylineMatrixCoercible | None:
    if x is None:
        return None

    if isinstance(x, str):
        e = eval_expression(x, context)
        if tg.is_one_or_many_2D_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a SkylineMatrixCoercible object (e.g., a scalar or a nested (2D) sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_vector_configs(x):
        return [skyline_vector(v, context) for v in x]

    assert isinstance(x, cfg.SkylineMatrixModel)

    change_times = many_scalars(x.change_times, context)
    if isinstance(x.value, str):
        e = eval_expression(x.value, context)
        if tg.is_many_one_or_many_2D_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, which cannot be coerced to a valid value for a SkylineMatrix (expected a sequence composed of scalars and/or nested (2D) sequences of scalars)."
            )
    else:
        value = [one_or_many_2D_scalars(v, context) for v in x.value]

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


def distribution(
    x: cfg.Distribution, context: dict[str, Any], seed: int | None
) -> pgt.Distribution:
    args = x.args
    for arg_name, arg_value in args.items():
        if isinstance(arg_value, str):
            args[arg_name] = eval_expression(arg_value, context)
    rng = default_rng(seed)
    return lambda: getattr(rng, x.type)(**args)


def context(x: cfg.Context, seed: int | None) -> dict[str, Any]:
    context: dict[str, Any] = {}
    for k, v in x.items():
        context[k] = np.array(distribution(v, context, seed)()).tolist()
    return context


def timed_event(timed_event: cfg.TimedEvent, context: dict[str, Any]) -> TimedEvent:
    state = None if timed_event.state is None else timed_event.state.format(**context)
    return TimedSampling(
        state=state,
        times=many_scalars(timed_event.times, context),
        proportion=scalar(timed_event.proportion, context),
        removal=timed_event.removal,
    )
