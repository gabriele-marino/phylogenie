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

SkylineVectorOperand = Union[SkylineParameterLike, "SkylineVector"]
SkylineVectorLike = Union[pgt.Many[SkylineParameterLike], "SkylineVector"]
SkylineVectorCoercible = Union[pgt.OneOrMany[SkylineParameterLike], "SkylineVector"]


def is_skyline_vector_operand(x: object) -> TypeGuard[SkylineVectorOperand]:
    return isinstance(x, SkylineVector) or is_skyline_parameter_like(x)


def is_skyline_vector_like(x: object) -> TypeGuard[SkylineVectorLike]:
    return isinstance(x, SkylineVector) or is_many_skyline_parameters_like(x)


def is_skyline_vector_coercible(
    x: object,
) -> TypeGuard[SkylineVectorCoercible]:
    return is_skyline_parameter_like(x) or is_skyline_vector_like(x)


def is_many_skyline_vectors_like(x: object) -> TypeGuard[pgt.Many[SkylineVectorLike]]:
    return tg.is_many(x) and all(is_skyline_vector_like(v) for v in x)


def is_many_skyline_vectors_coercible(
    x: object,
) -> TypeGuard[pgt.Many[SkylineVectorCoercible]]:
    return tg.is_many(x) and all(is_skyline_vector_coercible(v) for v in x)


class SkylineVector:
    def __init__(
        self,
        params: pgt.Many[SkylineParameterLike] | None = None,
        value: pgt.Many2DScalars | None = None,
        change_times: pgt.ManyScalars | None = None,
    ):
        if params is not None and value is None and change_times is None:
            if is_many_skyline_parameters_like(params):
                self.params = [skyline_parameter(param) for param in params]
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineVector from `params` {params} of type {type(params)}. Please provide a sequence of SkylineParameterLike objects (a SkylineParameterLike object can either be a SkylineParameter or a scalar)."
                )
        elif params is None and value is not None and change_times is not None:
            if tg.is_many_2D_scalars(value):
                vector_lengths = {len(vector) for vector in value}
                if any(vl != len(value[0]) for vl in vector_lengths):
                    raise ValueError(
                        f"All rows in the `value` of a SkylineVector must have the same length (got value={value} with vector lengths={vector_lengths})."
                    )
            else:
                raise TypeError(
                    f"It is impossible to create a SkylineVector from `value` {value} of type {type(value)}. Please provide a nested (2D) sequence of scalar values."
                )
            self.params = [
                SkylineParameter([vector[i] for vector in value], change_times)
                for i in range(len(value[0]))
            ]
        else:
            raise ValueError(
                "Either `params` or both `value` and `change_times` must be provided to create a SkylineVector."
            )

    @property
    def change_times(self) -> pgt.Vector1D:
        return sorted(set(t for param in self.params for t in param.change_times))

    @property
    def value(self) -> pgt.Vector2D:
        return [self.get_value_at_time(t) for t in (0, *self.change_times)]

    @property
    def N(self) -> int:
        return len(self.params)

    def get_value_at_time(self, t: pgt.Scalar) -> pgt.Vector1D:
        return [param.get_value_at_time(t) for param in self.params]

    def _operate(
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
        return self._operate(operand, lambda x, y: x + y)

    def __radd__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self._operate(operand, lambda x, y: y + x)

    def __sub__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self._operate(operand, lambda x, y: x - y)

    def __rsub__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self._operate(operand, lambda x, y: y - x)

    def __mul__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self._operate(operand, lambda x, y: x * y)

    def __rmul__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self._operate(operand, lambda x, y: y * x)

    def __truediv__(self, operand: SkylineVectorOperand) -> "SkylineVector":
        return self._operate(operand, lambda x, y: x / y)

    def __rtruediv__(self, operand: SkylineParameterLike) -> "SkylineVector":
        return self._operate(operand, lambda x, y: y / x)

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
    def __getitem__(self, item: slice) -> list[SkylineParameter]: ...
    def __getitem__(
        self, item: int | slice
    ) -> Union[SkylineParameter, list[SkylineParameter]]:
        return self.params[item]

    def __setitem__(self, item: int, value: SkylineParameterLike) -> None:
        if not is_skyline_parameter_like(value):
            raise TypeError(
                f"It is impossible to set item {item} of SkylineVector with value {value} of type {type(value)}. Please provide a SkylineParameterLike object (i.e., a scalar or a SkylineParameter)."
            )
        self.params[item] = skyline_parameter(value)


def skyline_vector(x: SkylineVectorCoercible, N: int) -> SkylineVector:
    if N <= 0:
        raise ValueError(
            f"N must be a positive integer for SkylineVector construction (got N={N})."
        )
    if is_skyline_parameter_like(x):
        return SkylineVector([skyline_parameter(x)] * N)
    elif is_many_skyline_parameters_like(x):
        x = SkylineVector(x)
    elif not isinstance(x, SkylineVector):
        raise TypeError(
            f"It is impossible to coerce {x} of type {type(x)} into a SkylineVector. Please provide a SkylineParameterLike object (i.e., a scalar or a SkylineParameter), or a sequence of them."
        )
    if x.N != N:
        raise ValueError(
            f"Expected a SkylineVector of size {N}, got {x} of size {x.N}."
        )
    return x
