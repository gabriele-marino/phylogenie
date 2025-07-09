from collections.abc import Callable, Iterator
from typing import TypeGuard, Union, overload

import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.skyline.parameter import (
    SkylineParameter,
    SkylineParameterLike,
    is_many_skyline_parameters_like,
    is_skyline_parameter_like,
    skyline_parameter,
)

SkylineVectorParams = pgt.OneOrMany[SkylineParameterLike]
SkylineVectorOperand = Union[SkylineParameterLike, "SkylineVector"]
SkylineVectorLike = Union[SkylineVectorParams, "SkylineVector"]


class SkylineVector:
    def __init__(
        self,
        params: SkylineVectorParams | None = None,
        value: pgt.Many[pgt.OneOrManyScalars] | None = None,
        change_times: pgt.OneOrManyScalars | None = None,
    ) -> None:
        if params is not None and value is None and change_times is None:
            if is_skyline_parameter_like(params):
                self.params = [skyline_parameter(params)]
            elif is_many_skyline_parameters_like(params):
                self.params = [skyline_parameter(param) for param in params]
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineVector from `params` {params} of type {type(params)}. Please provide a SkylineParameterLike object (i.e., a SkylineParameter or a scalar) or a sequence of them."
                )
        elif value is not None and change_times is not None:
            Ns = {len(row) for row in value if tg.is_many(row)}
            if len(Ns) > 1:
                raise ValueError(
                    f"All rows in the `value` must be scalars or have the same length to create a SkylineVector (got value={value} with row lengths={Ns})."
                )
            N = Ns.pop() if Ns else 1
            value = [[x] * N if isinstance(x, pgt.Scalar) else x for x in value]
            self.params = [
                SkylineParameter([row[i] for row in value], change_times)
                for i in range(N)
            ]
        else:
            raise ValueError(
                "Either `params` or both `value` and `change_times` must be provided to create a SkylineVector."
            )

    @property
    def change_times(self) -> pgt.Vector1D:
        return tuple(
            sorted(set(t for param in self.params for t in param.change_times))
        )

    @property
    def value(self) -> pgt.Vector2D:
        return tuple(self.get_value_at_time(t) for t in (0, *self.change_times))

    @property
    def N(self) -> int:
        return len(self.params)

    def get_value_at_time(self, t: pgt.Scalar) -> pgt.Vector1D:
        return tuple(param.get_value_at_time(t) for param in self.params)

    def operate(
        self,
        other: SkylineVectorOperand,
        func: Callable[[SkylineParameter, SkylineParameter], SkylineParameter],
    ) -> "SkylineVector":
        if not is_skyline_vector_operand(other):
            return NotImplemented
        other = skyline_vector(other, self.N)
        return SkylineVector(
            [func(p1, p2) for p1, p2 in zip(self.params, other.params)]
        )

    def __add__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self.operate(operand, lambda x, y: x + y)

    def __radd__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self.operate(operand, lambda x, y: y + x)

    def __sub__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self.operate(operand, lambda x, y: x - y)

    def __rsub__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self.operate(operand, lambda x, y: y - x)

    def __mul__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self.operate(operand, lambda x, y: x * y)

    def __rmul__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self.operate(operand, lambda x, y: y * x)

    def __truediv__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self.operate(operand, lambda x, y: x / y)

    def __rtruediv__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self.operate(operand, lambda x, y: y / x)

    def __len__(self) -> int:
        return self.N

    def __bool__(self) -> bool:
        return any(self.params)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkylineVector) and self.params == other.params

    def __repr__(self) -> str:
        return f"SkylineVector(value={list(self.value)}, change_times={list(self.change_times)})"

    def __iter__(self) -> Iterator[SkylineParameter]:
        return iter(self.params)

    @overload
    def __getitem__(self, item: int) -> SkylineParameter: ...
    @overload
    def __getitem__(self, item: slice) -> "SkylineVector": ...
    def __getitem__(self, item: int | slice) -> "SkylineParameter | SkylineVector":
        if isinstance(item, slice):
            return SkylineVector(self.params[item])
        return self.params[item]

    def __setitem__(self, item: int, value: SkylineParameterLike) -> None:
        if not is_skyline_parameter_like(value):
            raise TypeError(
                f"`value` must be a SkylineParameterLike (got {type(value)})."
            )
        self.params[item] = skyline_parameter(value)


def skyline_vector(x: SkylineVectorLike, N: int) -> SkylineVector:
    if is_skyline_parameter_like(x):
        return SkylineVector([skyline_parameter(x)] * N)
    if not isinstance(x, SkylineVector):
        x = SkylineVector(x)
    if x.N != N:
        raise ValueError(
            f"Expected a SkylineVector of size {N}, got {x} of size {x.N}."
        )
    return x


def is_skyline_vector_operand(value: object) -> TypeGuard[SkylineVectorOperand]:
    return isinstance(value, SkylineVector) or is_skyline_parameter_like(value)


def is_skyline_vector_like(value: object) -> TypeGuard[SkylineVectorLike]:
    return (
        isinstance(value, SkylineVector)
        or is_skyline_parameter_like(value)
        or is_many_skyline_parameters_like(value)
    )


def is_many_skyline_vectors_like(
    value: object,
) -> TypeGuard[pgt.Many[SkylineVectorLike]]:
    return tg.is_many(value) and all(is_skyline_vector_like(v) for v in value)
