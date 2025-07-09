from collections.abc import Callable, Iterator
from typing import TypeGuard, Union, overload

import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.skyline.parameter import (
    SkylineParameterLike,
    is_many_skyline_parameters_like,
    is_skyline_parameter_like,
)
from phylogenie.skyline.vector import (
    SkylineVector,
    SkylineVectorLike,
    SkylineVectorOperand,
    is_many_skyline_vectors_like,
    is_skyline_vector_like,
    is_skyline_vector_operand,
    skyline_vector,
)

SkylineVector2D = pgt.Many[pgt.Many[SkylineParameterLike] | SkylineVector]
SkylineMatrixParams = SkylineParameterLike | SkylineVector | SkylineVector2D
SkylineMatrixOperand = Union[SkylineVectorOperand, "SkylineMatrix"]
SkylineMatrixLike = Union[pgt.OneOrMany[SkylineVectorLike], "SkylineMatrix"]


def is_skyline_vector2D(value: object) -> TypeGuard[SkylineVector2D]:
    return tg.is_many(value) and all(
        isinstance(v, SkylineVector) or is_many_skyline_parameters_like(v)
        for v in value
    )


class SkylineMatrix:
    def __init__(
        self,
        params: SkylineMatrixParams | None = None,
        value: pgt.Many[pgt.OneOrMany2DScalars] | None = None,
        change_times: pgt.OneOrManyScalars | None = None,
    ):
        if params is not None and value is None and change_times is None:
            if is_skyline_parameter_like(params) or isinstance(params, SkylineVector):
                self.params = [skyline_vector(params, 1)]
            elif is_skyline_vector2D(params):
                self.params = [skyline_vector(param, N=len(params)) for param in params]
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineMatrix from `params` {params} of type {type(params)}.Please provide either:\n"
                    "- a SkylineParameterLike object (i.e., a SkylineParameter or a scalar),\n"
                    "- a SkylineVector,\n"
                    "- or a SkylineVector2D: a sequence containing SkylineVectors and/or sequences of SkylineParameterLike objects."
                )
        elif value is not None and change_times is not None:
            Ns = {len(matrix) for matrix in value if tg.is_many(matrix)}
            if len(Ns) > 1:
                raise ValueError(
                    f"All matrices in the `value` must be scalars or have the same length to create a SkylineMatrix (got value={value} with row lengths={Ns})."
                )
            N = Ns.pop() if Ns else 1
            value = [[[x] * N] * N if isinstance(x, pgt.Scalar) else x for x in value]
            self.params = [
                SkylineVector(
                    value=[matrix[i] for matrix in value], change_times=change_times
                )
                for i in range(N)
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
        return tuple(sorted(set([t for row in self.params for t in row.change_times])))

    @property
    def value(self) -> pgt.Vector3D:
        return tuple(self.get_value_at_time(t) for t in (0, *self.change_times))

    def get_value_at_time(self, time: pgt.Scalar) -> pgt.Vector2D:
        return tuple(param.get_value_at_time(time) for param in self.params)

    def operate(
        self,
        other: SkylineMatrixOperand,
        func: Callable[[SkylineVector, SkylineVector], SkylineVector],
    ) -> "SkylineMatrix":
        if not is_skyline_matrix_operand(other):
            return NotImplemented
        other = skyline_matrix(other, self.N)
        return SkylineMatrix(
            [func(p1, p2) for p1, p2 in zip(self.params, other.params)]
        )

    def __add__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: x + y)

    def __radd__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: y + x)

    def __sub__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: x - y)

    def __rsub__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: y - x)

    def __mul__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: x * y)

    def __rmul__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: y * x)

    def __truediv__(self, operand: SkylineMatrixOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: x / y)

    def __rtruediv__(self, operand: SkylineVectorOperand) -> "SkylineMatrix":
        return self.operate(operand, lambda x, y: y / x)

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
    def __getitem__(self, item: slice) -> "SkylineMatrix": ...
    def __getitem__(self, item: int | slice) -> "SkylineVector | SkylineMatrix":
        if isinstance(item, slice):
            return SkylineMatrix(self.params[item])
        return self.params[item]

    def __setitem__(self, item: int, value: SkylineVectorLike) -> None:
        if not is_skyline_vector_like(value):
            raise TypeError(f"Expected a SkylineVectorLike, got {type(value)}.")
        self.params[item] = skyline_vector(value, N=self.N)


def skyline_matrix(
    x: SkylineMatrixLike, N: int, zero_diagonal: bool = False
) -> SkylineMatrix:
    if is_skyline_vector_like(x):
        x = SkylineMatrix([skyline_vector(x, N)] * N)
        if zero_diagonal:
            for i in range(N):
                x[i][i] = 0
        return x.T

    if is_many_skyline_vectors_like(x):
        x = SkylineMatrix([skyline_vector(v, N) for v in x])

    if not isinstance(x, SkylineMatrix):
        raise TypeError(
            f"It is impossible to coerce {x} of type {type(x)} into a SkylineMatrix. Please provide either:\n"
            "- a SkylineMatrix,\n"
            "- a SkylineVectorLike: a SkylineParameterLike (i.e., a SkylineParameter or a scalar) or a sequence of them,\n"
            "- or a sequence of SkylineVectorLike objects."
        )

    if x.N != N:
        raise ValueError(
            f"Expected an {N}x{N} SkylineMatrix, got {x} of shape {x.N}x{x.N}."
        )

    if zero_diagonal and any(x[i][i] for i in range(x.N)):
        raise ValueError(f"Expected a SkylineMatrix with zero diagonal, but got {x}.")

    return x


def is_skyline_matrix_operand(x: object) -> TypeGuard[SkylineMatrixOperand]:
    return isinstance(x, SkylineMatrix) or is_skyline_vector_operand(x)
