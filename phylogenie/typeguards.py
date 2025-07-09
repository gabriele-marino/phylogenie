from collections.abc import Sequence
from typing import TypeGuard

import phylogenie.typings as pgt


def is_many(x: object) -> TypeGuard[pgt.Many[object]]:
    return isinstance(x, Sequence) and not isinstance(x, str)


def is_many_scalars(x: object) -> TypeGuard[pgt.Many[pgt.Scalar]]:
    return is_many(x) and all(isinstance(i, pgt.Scalar) for i in x)


def is_many_ints(x: object) -> TypeGuard[pgt.Many[int]]:
    return is_many(x) and all(isinstance(i, int) for i in x)


def is_one_or_many_scalars(x: object) -> TypeGuard[pgt.OneOrManyScalars]:
    return isinstance(x, pgt.Scalar) or is_many_scalars(x)


def is_many_one_or_many_scalars(x: object) -> TypeGuard[pgt.Many[pgt.OneOrManyScalars]]:
    return is_many(x) and all(is_one_or_many_scalars(i) for i in x)


def is_one_or_many_ints(x: object) -> TypeGuard[pgt.OneOrMany[int]]:
    return isinstance(x, int) or is_many_ints(x)


def is_many_2D_scalars(x: object) -> TypeGuard[pgt.Many2DScalars]:
    return is_many(x) and all(is_many_scalars(i) for i in x)


def is_many_one_or_many_2d_scalars(
    x: object,
) -> TypeGuard[pgt.Many[pgt.OneOrMany2DScalars]]:
    return is_many(x) and all(
        isinstance(i, pgt.Scalar) or is_many_2D_scalars(i) for i in x
    )
