from collections.abc import Callable, Iterator
from typing import TypeGuard, Union, overload

import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.skyline.parameter import SkylineParameter, is_skyline_parameter_like
from phylogenie.skyline.vector import (
    SkylineVector,
    SkylineVectorCoercible,
    SkylineVectorLike,
    SkylineVectorOperand,
    is_many_skyline_vectors_coercible,
    is_many_skyline_vectors_like,
    is_skyline_vector_coercible,
    is_skyline_vector_like,
    is_skyline_vector_operand,
    skyline_vector,
)

SkylineMatrixOperand = Union[SkylineVectorOperand, "SkylineMatrix"]
SkylineMatrixCoercible = Union[pgt.OneOrMany[SkylineVectorCoercible], "SkylineMatrix"]


def is_skyline_matrix_operand(x: object) -> TypeGuard[SkylineMatrixOperand]:
    return isinstance(x, SkylineMatrix) or is_skyline_vector_operand(x)


class SkylineMatrix:
    def __init__(
        self,
        params: pgt.Many[SkylineVectorLike] | None = None,
        value: pgt.Many3DScalars | None = None,
        change_times: pgt.ManyScalars | None = None,
    ):
        if params is not None and value is None and change_times is None:
            if is_many_skyline_vectors_like(params):
                self.params = [skyline_vector(p, len(params)) for p in params]
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineMatrix from `params` {params} of type {type(params)}. Please provide a sequence composed of SkylineVectorLike objects (a SkylineVectorLike object can either be a SkylineVector or a sequence of scalars and/or SkylineParameters)."
                )
        elif params is None and value is not None and change_times is not None:
            if tg.is_many_3D_scalars(value):
                matrix_lengths = {len(matrix) for matrix in value}
                if any(ml != len(value[0]) for ml in matrix_lengths):
                    raise ValueError(
                        f"All matrices in the `value` of a SkylineMatrix must have the same length (got value={value} with matrix lengths={matrix_lengths})."
                    )
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineMatrix from `value` {value} of type {type(value)}. Please provide a nested (3D) sequence of scalar values."
                )
            self.params = [
                SkylineVector(
                    value=[matrix[i] for matrix in value], change_times=change_times
                )
                for i in range(len(value[0]))
            ]
        else:
            raise ValueError(
                "Either `params` or both `value` and `change_times` must be provided to create a SkylineMatrix."
            )

    @property
    def N(self) -> int:
        return len(self.params)

    @property
    def change_times(self) -> pgt.Vector1D:
        return sorted(set([t for row in self.params for t in row.change_times]))

    @property
    def value(self) -> pgt.Vector3D:
        return [self.get_value_at_time(t) for t in (0, *self.change_times)]

    def get_value_at_time(self, time: pgt.Scalar) -> pgt.Vector2D:
        return [param.get_value_at_time(time) for param in self.params]

    def _operate(
        self,
        other: SkylineMatrixOperand,
        func: Callable[
            [SkylineVector, SkylineVector | SkylineParameter], SkylineVector
        ],
    ) -> "SkylineMatrix":
        if is_skyline_vector_operand(other):
            other = skyline_vector(other, N=self.N)
        elif isinstance(other, SkylineMatrix):
            if other.N != self.N:
                raise ValueError(
                    f"Expected a SkylineMatrix with the same size as self (N={self.N}), but got {other} with N={other.N}."
                )
        else:
            return NotImplemented
        return SkylineMatrix(
            [func(p1, p2) for p1, p2 in zip(self.params, other.params)]
        )

    def __add__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: x + y)

    def __radd__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: y + x)

    def __sub__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: x - y)

    def __rsub__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: y - x)

    def __mul__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: x * y)

    def __rmul__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: y * x)

    def __truediv__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: x / y)

    def __rtruediv__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self._operate(operand, lambda x, y: y / x)

    @property
    def T(self) -> "SkylineMatrix":
        return SkylineMatrix([[v[i] for v in self] for i in range(self.N)])

    def __bool__(self) -> bool:
        return any(self.params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkylineMatrix) and self.params == other.params

    def __repr__(self) -> str:
        return f"SkylineMatrix(value={list(self.value)}, change_times={list(self.change_times)})"

    def __iter__(self) -> Iterator[SkylineVector]:
        return iter(self.params)

    def __len__(self) -> int:
        return self.N

    @overload
    def __getitem__(self, item: int) -> SkylineVector: ...
    @overload
    def __getitem__(self, item: slice) -> list[SkylineVector]: ...
    def __getitem__(
        self, item: int | slice
    ) -> Union[SkylineVector, list[SkylineVector]]:
        return self.params[item]

    def __setitem__(self, item: int, value: SkylineVectorLike) -> None:
        if not is_skyline_vector_like(value):
            raise TypeError(
                f"It is impossible to set item {item} of SkylineMatrix with value {value} of type {type(value)}. Please provide a SkylineVectorLike object (i.e., a SkylineVector or a sequence of scalars and/or SkylineParameters)."
            )
        self.params[item] = skyline_vector(value, N=self.N)


def skyline_matrix(
    x: SkylineMatrixCoercible, N: int, zero_diagonal: bool = False
) -> SkylineMatrix:
    if N <= 0:
        raise ValueError(
            f"N must be a positive integer for SkylineMatrix construction (got N={N})."
        )
    if is_skyline_vector_coercible(x):
        x = SkylineMatrix([[p] * N for p in skyline_vector(x, N)])
        if zero_diagonal:
            for i in range(N):
                x[i][i] = 0
        return x
    elif is_many_skyline_vectors_coercible(x):
        x = SkylineMatrix(
            [
                [
                    (
                        0
                        if i == j and is_skyline_parameter_like(v) and zero_diagonal
                        else p
                    )
                    for j, p in enumerate(skyline_vector(v, N))
                ]
                for i, v in enumerate(x)
            ]
        )
    if not isinstance(x, SkylineMatrix):
        raise TypeError(
            f"It is impossible to coerce {x} of type {type(x)} into a SkylineMatrix. Please provide either:\n"
            "- a SkylineMatrix,\n"
            "- a SkylineVectorCoercible object (i.e., a scalar, a SkylineParameter, a SkylineVector, or a sequence of scalars and/or SkylineParameters),\n"
            "- a sequence of SkylineVectorCoercible objects."
        )

    if x.N != N:
        raise ValueError(
            f"Expected an {N}x{N} SkylineMatrix, got {x} of shape {x.N}x{x.N}."
        )

    if zero_diagonal and any(x[i][i] for i in range(x.N)):
        raise ValueError(f"Expected a SkylineMatrix with zero diagonal, but got {x}.")

    return x
