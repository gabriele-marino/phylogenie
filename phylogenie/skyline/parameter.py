from bisect import bisect_right
from collections.abc import Callable
from typing import TypeGuard, Union

import phylogenie.typeguards as tg
import phylogenie.typings as pgt

SkylineParameterLike = Union[pgt.Scalar, "SkylineParameter"]


def is_skyline_parameter_like(x: object) -> TypeGuard[SkylineParameterLike]:
    return isinstance(x, pgt.Scalar | SkylineParameter)


def is_many_skyline_parameters_like(
    x: object,
) -> TypeGuard[pgt.Many[SkylineParameterLike]]:
    return tg.is_many(x) and all(is_skyline_parameter_like(v) for v in x)


class SkylineParameter:
    def __init__(
        self,
        value: pgt.OneOrManyScalars,
        change_times: pgt.ManyScalars | None = None,
    ):
        if isinstance(value, pgt.Scalar):
            value = [value]
        elif not tg.is_many_scalars(value):
            raise TypeError(
                f"It is impossible to create a SkylineParameter from `value` {value} of type {type(value)}. Please provide a scalar or a sequence of scalars."
            )

        if change_times is None:
            change_times = []
        elif not tg.is_many_scalars(change_times):
            raise TypeError(
                f"It is impossible to create a SkylineParameter from `change_times` {change_times} of type {type(change_times)}. Please provide a sequence of scalars."
            )

        if len(value) != len(change_times) + 1:
            raise ValueError(
                f"`value` must have exactly one more element than `change_times` (got value={value} of length {len(value)} and change_times={change_times} of length {len(change_times)})."
            )
        if any(t1 >= t2 for t1, t2 in zip(change_times, change_times[1:])):
            raise ValueError(
                f"`change_times` must be sorted in strictly increasing order "
                f"(got change_times={change_times})."
            )
        if any(t < 0 for t in change_times):
            raise ValueError(
                f"`change_times` must be non-negative (got change_times={change_times})."
            )

        self.value = [value[0]]
        self.change_times: list[pgt.Scalar] = []
        for i in range(1, len(value)):
            if value[i] != value[i - 1]:
                self.value.append(value[i])
                self.change_times.append(change_times[i - 1])

    def get_value_at_time(self, t: pgt.Scalar) -> pgt.Scalar:
        if t < 0:
            raise ValueError(f"Time cannot be negative (got t={t}).")
        return self.value[bisect_right(self.change_times, t)]

    def _operate(
        self,
        other: SkylineParameterLike,
        f: Callable[[pgt.Scalar, pgt.Scalar], pgt.Scalar],
    ) -> "SkylineParameter":
        if is_skyline_parameter_like(other):
            other = skyline_parameter(other)
        else:
            return NotImplemented
        change_times = sorted(set(self.change_times + other.change_times))
        value = [
            f(self.get_value_at_time(t), other.get_value_at_time(t))
            for t in (0, *change_times)
        ]
        return SkylineParameter(value, change_times)

    def __add__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self._operate(other, lambda x, y: x + y)

    def __radd__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self._operate(other, lambda x, y: y + x)

    def __sub__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self._operate(other, lambda x, y: x - y)

    def __rsub__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self._operate(other, lambda x, y: y - x)

    def __mul__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self._operate(other, lambda x, y: x * y)

    def __rmul__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self._operate(other, lambda x, y: y * x)

    def __truediv__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self._operate(other, lambda x, y: x / y)

    def __rtruediv__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self._operate(other, lambda x, y: y / x)

    def __bool__(self) -> bool:
        return any(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkylineParameter) and (
            self.value == other.value and self.change_times == other.change_times
        )

    def __repr__(self) -> str:
        return f"SkylineParameter(value={list(self.value)}, change_times={list(self.change_times)})"


def skyline_parameter(x: SkylineParameterLike) -> SkylineParameter:
    return SkylineParameter(x) if isinstance(x, pgt.Scalar) else x
