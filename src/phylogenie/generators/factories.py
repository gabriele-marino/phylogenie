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


def eval_expression(
    expression: str,
    ctx: dict[str, Any],
    **kwargs: Any,
) -> Any:
    return np.array(
        eval(
            expression,
            {"np": np, **{k: np.array(v) for k, v in ctx.items()}, **kwargs},
        )
    ).tolist()


def integer(x: cfg.Integer, ctx: dict[str, Any]) -> int:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if isinstance(e, int):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected an int."
        )
    return x


def scalar(x: cfg.Scalar, ctx: dict[str, Any]) -> pgt.Scalar:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if isinstance(e, pgt.Scalar):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a scalar."
        )
    return x


def string(s: Any, ctx: dict[str, Any]) -> str:
    if not isinstance(s, str):
        return str(s)
    return re.sub(
        r"\{([^{}]+)\}", lambda match: str(eval_expression(match.group(1), ctx)), s
    )  # Match content inside curly braces


def many_scalars(x: cfg.ManyScalars, ctx: dict[str, Any]) -> pgt.ManyScalars:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if tg.is_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a sequence of scalars."
        )
    return [scalar(v, ctx) for v in x]


def one_or_many_scalars(
    x: cfg.OneOrManyScalars, ctx: dict[str, Any]
) -> pgt.OneOrManyScalars:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a "
            "scalar or a sequence of them."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return many_scalars(x, ctx)


def skyline_parameter(
    x: cfg.SkylineParameter, ctx: dict[str, Any]
) -> SkylineParameterLike:
    if isinstance(x, cfg.Scalar):
        return scalar(x, ctx)
    return SkylineParameter(
        value=many_scalars(x.value, ctx),
        change_times=many_scalars(x.change_times, ctx),
    )


def skyline_vector(x: cfg.SkylineVector, ctx: dict[str, Any]) -> SkylineVectorCoercible:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if tg.is_one_or_many_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a "
            "SkylineVectorCoercible object (e.g., a scalar or a sequence of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_parameter_configs(x):
        return [skyline_parameter(p, ctx) for p in x]

    assert isinstance(x, cfg.SkylineVectorModel)

    change_times = many_scalars(x.change_times, ctx)
    if isinstance(x.value, str):
        e = eval_expression(x.value, ctx)
        if tg.is_many_one_or_many_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, "
                "which cannot be coerced to a valid value for a SkylineVector "
                "(expected a sequence composed of scalars and/or sequences of "
                "scalars)."
            )
    else:
        value = [one_or_many_scalars(v, ctx) for v in x.value]

    if tg.is_many_scalars(value):
        return SkylineParameter(value=value, change_times=change_times)

    sizes = {len(elem) for elem in value if tg.is_many(elem)}
    if len(sizes) > 1:
        raise ValueError(
            "All elements in the value of a SkylineVector config must be scalars or "
            f"have the same length (config {x.value} yielded value={value} with "
            f"inconsistent sizes {sizes})."
        )
    (size,) = sizes
    value = [[p] * size if isinstance(p, pgt.Scalar) else p for p in value]

    return SkylineVector(value=value, change_times=change_times)


def one_or_many_2d_scalars(
    x: cfg.OneOrMany2DScalars, ctx: dict[str, Any]
) -> pgt.OneOrMany2DScalars:
    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if tg.is_one_or_many_2d_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a "
            "nested (2D) sequence of scalars."
        )
    if isinstance(x, pgt.Scalar):
        return x
    return [many_scalars(v, ctx) for v in x]


def skyline_matrix(
    x: cfg.SkylineMatrix, ctx: dict[str, Any]
) -> SkylineMatrixCoercible | None:
    if x is None:
        return None

    if isinstance(x, str):
        e = eval_expression(x, ctx)
        if tg.is_one_or_many_2d_scalars(e):
            return e
        raise ValueError(
            f"Expression '{x}' evaluated to {e} of type {type(e)}, expected a "
            "SkylineMatrixCoercible object (e.g., a scalar or a nested (2D) sequence "
            "of them)."
        )
    if isinstance(x, pgt.Scalar):
        return x
    if ctg.is_many_skyline_vector_configs(x):
        return [skyline_vector(v, ctx) for v in x]

    assert isinstance(x, cfg.SkylineMatrixModel)

    change_times = many_scalars(x.change_times, ctx)
    if isinstance(x.value, str):
        e = eval_expression(x.value, ctx)
        if tg.is_many_one_or_many_2d_scalars(e):
            value = e
        else:
            raise ValueError(
                f"Expression '{x.value}' evaluated to {e} of type {type(e)}, "
                "which cannot be coerced to a valid value for a SkylineMatrix "
                "(expected a sequence composed of scalars and/or nested (2D) "
                "sequences of scalars)."
            )
    else:
        value = [one_or_many_2d_scalars(v, ctx) for v in x.value]

    if tg.is_many_scalars(value):
        return SkylineParameter(value=value, change_times=change_times)

    shapes: set[tuple[int, int]] = set()
    for elem in value:
        if tg.is_many_2d_scalars(elem):
            n_rows = len(elem)
            n_cols = {len(row) for row in elem}
            if len(n_cols) > 1:
                raise ValueError(
                    "The values of a SkylineMatrix config must be scalars or nested "
                    "(2D) lists of them with a consistent row length (config "
                    f"{x.value} yielded element {elem} with row lengths {n_cols})."
                )
            shapes.add((n_rows, n_cols.pop()))

    if len(shapes) > 1:
        raise ValueError(
            "All elements in the value of a SkylineMatrix config must be scalars or "
            f"nested (2D) lists of them with the same shape (config {x.value} yielded "
            f"value={value} with inconsistent shapes {shapes})."
        )
    ((n_rows, n_cols),) = shapes
    value = [[[e] * n_cols] * n_rows if isinstance(e, pgt.Scalar) else e for e in value]

    return SkylineMatrix(value=value, change_times=change_times)


def distribution(
    x: cfg.Distribution, ctx: dict[str, Any], rng: Generator
) -> pgt.Distribution:
    args = x.args
    for arg_name, arg_value in args.items():
        if isinstance(arg_value, str):
            args[arg_name] = eval_expression(arg_value, ctx)
    return lambda: getattr(rng, x.type)(**args)


def context(x: cfg.Context, rng: Generator) -> dict[str, Any]:
    ctx: dict[str, Any] = {}
    for k, v in x.items():
        ctx[k] = (
            eval_expression(v, ctx)
            if isinstance(v, str)
            else np.array(distribution(v, ctx, rng)()).tolist()
        )
    return ctx
