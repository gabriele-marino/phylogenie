from typing import Callable, Iterator, Sequence, Union

from kitpy.type_hints import Numeric

from phylogenie.skyline.core.parameter import (
    ParameterOperand,
    SerialSkylineParameter,
    SkylineParameter,
)
from phylogenie.skyline.core.vector import SkylineVector, VectorOperand

MatrixOperand = Union[VectorOperand, "SkylineMatrix"]

SerialSkylineMatrix = SerialSkylineParameter | dict[str, SerialSkylineParameter]


class SkylineMatrix:
    def __init__(self, params: Sequence[Sequence[SkylineParameter | Numeric]]):
        self.params = [
            [
                (
                    param
                    if isinstance(param, SkylineParameter)
                    else SkylineParameter(param)
                )
                for param in row
            ]
            for row in params
        ]
        self.N = len(params)
        if not all(len(row) == self.N - 1 for row in params):
            raise ValueError(
                "The matrix should be a square matrix with no diagonal elements "
                f"(got {len(params)} rows with lengths {[len(row) for row in params]})"
            )

    def _operate(
        self,
        other: MatrixOperand,
        func: Callable[[SkylineParameter, ParameterOperand], SkylineParameter],
    ) -> "SkylineMatrix":
        if isinstance(other, SkylineMatrix):
            if self.N != other.N:
                raise ValueError("Matrix dimensions must match")
            params = [
                [func(p1, p2) for p1, p2 in zip(row1, row2)]
                for row1, row2 in zip(self.params, other.params)
            ]
        elif isinstance(other, SkylineVector):
            if len(other.params) == self.N:
                params = [
                    [func(cell, other.params[i]) for cell in row]
                    for i, row in enumerate(self.params)
                ]
            elif len(other.params) == self.N - 1:
                params = [
                    [func(cell, other.params[j]) for j, cell in enumerate(row)]
                    for row in self.params
                ]
            else:
                raise ValueError(
                    f"Cannot broadcast SkylineVector of length {len(other.params)} "
                    f"to matrix shape {self.N}x{self.N-1}"
                )
        else:
            params = [[func(cell, other) for cell in row] for row in self.params]
        return SkylineMatrix(params)

    def get_value_at_time(self, time: Numeric) -> list[list[Numeric]]:
        return [[param.get_value_at_time(time) for param in row] for row in self.params]

    @property
    def change_times(self) -> list[Numeric]:
        return sorted(
            set([t for row in self.params for param in row for t in param.change_times])
        )

    def __add__(self, other: MatrixOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: x + y)

    def __radd__(self, other: VectorOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: y + x)

    def __sub__(self, other: MatrixOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: x - y)

    def __rsub__(self, other: VectorOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: y - x)

    def __mul__(self, other: MatrixOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: x * y)

    def __rmul__(self, other: VectorOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: y * x)

    def __truediv__(self, other: MatrixOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: x / y)

    def __rtruediv__(self, other: VectorOperand) -> "SkylineMatrix":
        return self._operate(other, lambda x, y: y / x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkylineMatrix) and self.params == other.params

    def __bool__(self) -> bool:
        return any(any(row) for row in self.params)

    def __repr__(self) -> str:
        return f"SkylineMatrix(params={self.params})"

    def __iter__(self) -> Iterator[SkylineVector]:
        return iter([SkylineVector(row) for row in self.params])

    def serialize(self, keys: list[str] | None = None) -> SerialSkylineMatrix:
        if all(p == self.params[0][0] for row in self.params for p in row):
            return self.params[0][0].serialize()
        if keys is None:
            keys = [str(i + 1) for i in range(len(self.params))]
        return {
            f"{k1}->{k2}": param.serialize()
            for row, k1 in zip(self.params, keys)
            for param, k2 in zip(row, [k for k in keys if k != k1])
        }
