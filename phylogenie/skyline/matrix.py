from collections.abc import Callable, Iterator
from typing import Any, TypeGuard, Union, overload

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
    is_skyline_vector_like,
    is_skyline_vector_operand,
    skyline_vector,
)

SkylineMatrixOperand = Union[SkylineVectorOperand, "SkylineMatrix"]
SkylineMatrixCoercible = Union[pgt.OneOrMany[SkylineVectorCoercible], "SkylineMatrix"]


def is_skyline_matrix_operand(x: Any) -> TypeGuard[SkylineMatrixOperand]:
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
                self.params = [
                    p if isinstance(p, SkylineVector) else SkylineVector(p)
                    for p in params
                ]
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineMatrix from `params` {params} of type {type(params)}. Please provide a sequence composed of SkylineVectorLike objects (a SkylineVectorLike object can either be a SkylineVector or a sequence of scalars and/or SkylineParameters)."
                )
            lengths = {len(p) for p in self.params}
            if len(lengths) > 1:
                raise ValueError(
                    f"All `params` must have the same length to create a SkylineMatrix (got params={params} with lengths {lengths})."
                )
        elif params is None and value is not None and change_times is not None:
            if tg.is_many_3D_scalars(value):
                lengths = {len(matrix) for matrix in value}
                if len(lengths) > 1:
                    raise ValueError(
                        f"All matrices in the `value` of a SkylineMatrix must have the same number of rows (got matrices={value} with row lengths {lengths})."
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
    def n_rows(self) -> int:
        return len(self.params)

    @property
    def n_cols(self) -> int:
        return len(self.params[0])

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

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
        if is_skyline_matrix_operand(other):
            other = skyline_matrix(other, self.n_rows, self.n_cols)
        elif isinstance(other, SkylineMatrix):
            if other.shape != self.shape:
                raise ValueError(
                    f"It is impossible to operate on SkylineMatrices of different shapes (got self={self.shape} and other={other.shape})."
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
        return SkylineMatrix([[v[i] for v in self] for i in range(self.n_cols)])

    def __bool__(self) -> bool:
        return any(self.params)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, SkylineMatrix) and self.params == other.params

    def __repr__(self) -> str:
        return f"SkylineMatrix(value={list(self.value)}, change_times={list(self.change_times)})"

    def __iter__(self) -> Iterator[SkylineVector]:
        return iter(self.params)

    def __len__(self) -> int:
        return self.n_rows

    @overload
    def __getitem__(self, item: int) -> SkylineVector: ...
    @overload
    def __getitem__(self, item: slice) -> "SkylineMatrix": ...
    @overload
    def __getitem__(self, item: tuple[int, int]) -> SkylineParameter: ...
    @overload
    def __getitem__(self, item: tuple[int, slice]) -> SkylineVector: ...
    @overload
    def __getitem__(self, item: tuple[slice, int]) -> SkylineVector: ...
    @overload
    def __getitem__(self, item: tuple[slice, slice]) -> "SkylineMatrix": ...
    def __getitem__(
        self, item: int | slice | tuple[int | slice, int | slice]
    ) -> Union[SkylineParameter | SkylineVector, "SkylineMatrix"]:
        if isinstance(item, int):
            return self.params[item]
        if isinstance(item, slice):
            return SkylineMatrix(self.params[item])
        row_idx, col_idx = item
        if isinstance(row_idx, int):
            return self.params[row_idx][col_idx]
        if isinstance(col_idx, int):
            return SkylineVector([row[col_idx] for row in self.params[row_idx]])
        return SkylineMatrix([row[col_idx] for row in self.params[row_idx]])

    def __setitem__(self, item: int, value: SkylineVectorLike) -> None:
        if not is_skyline_vector_like(value):
            raise TypeError(
                f"It is impossible to set item of SkylineMatrix to value {value} of type {type(value)}. Please provide a SkylineVectorLike object (i.e., a SkylineVector or a sequence of scalars and/or SkylineParameters)."
            )
        self.params[item] = skyline_vector(value, self.n_cols)


def skyline_matrix(
    x: SkylineMatrixCoercible, n_rows: int, n_cols: int
) -> SkylineMatrix:
    if n_rows <= 0 or n_cols <= 0:
        raise ValueError(
            f" n_rows and n_cols must be positive integers to create a SkylineMatrix (got n_rows={n_rows} and n_cols={n_cols})."
        )

    if is_skyline_parameter_like(x):
        return SkylineMatrix([[x] * n_cols] * n_rows)
    if is_skyline_vector_like(x) or is_many_skyline_vectors_coercible(x):
        if len(x) == n_rows:
            return SkylineMatrix([skyline_vector(p, n_cols) for p in x])
        elif len(x) == n_cols:
            return SkylineMatrix([skyline_vector(p, n_rows) for p in x]).T
        raise ValueError(
            f"Expected a SkylineVectorLike of size {n_rows} or {n_cols}, got {x} of size {len(x)}."
        )

    if not isinstance(x, SkylineMatrix):
        raise TypeError(
            f"It is impossible to coerce {x} of type {type(x)} into a SkylineMatrix. Please provide either:\n"
            "- a SkylineMatrix,\n"
            "- a SkylineVectorCoercible object (i.e., a scalar, a SkylineParameter, a SkylineVector, or a sequence of scalars and/or SkylineParameters),\n"
            "- a sequence of SkylineVectorCoercible objects."
        )

    if x.shape != (n_rows, n_cols):
        raise ValueError(
            f"Expected an SkylineMatrix of shape ({n_rows}, {n_cols}), got {x} of shape {x.shape}."
        )

    return x
