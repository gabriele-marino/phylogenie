from bisect import bisect_right
from collections.abc import Callable
from typing import TypeGuard, Union

import phylogenie.typeguards as tg
import phylogenie.typings as pgt
from phylogenie.utils import vectorify1D

SkylineParameterLike = Union[pgt.Scalar, "SkylineParameter"]


class SkylineParameter:
    def __init__(
        self,
        value: pgt.OneOrManyScalars,
        change_times: pgt.OneOrManyScalars | None = None,
    ) -> None:
        value = vectorify1D(value)
        change_times = vectorify1D(change_times)
        if len(value) != len(change_times) + 1:
            raise ValueError(
                f"`value` must have exactly one more element than `change_times` (got value={value} of length {len(value)} and change_times={change_times} of length {len(change_times)})."
            )

        value_ = [value[0]]
        change_times_: list[pgt.Scalar] = []
        for i in range(1, len(value)):
            if value[i] != value[i - 1]:
                value_.append(value[i])
                change_times_.append(change_times[i - 1])
        self.value = tuple(value_)
        self.change_times = tuple(change_times_)

    def get_value_at_time(self, t: pgt.Scalar) -> pgt.Scalar:
        return self.value[bisect_right(self.change_times, t)]

    def operate(
        self,
        other: SkylineParameterLike,
        f: Callable[[pgt.Scalar, pgt.Scalar], pgt.Scalar],
    ) -> "SkylineParameter":
        other = skyline_parameter(other)
        change_times = sorted(set(self.change_times + other.change_times))
        value = [
            f(self.get_value_at_time(t), other.get_value_at_time(t))
            for t in (0, *change_times)
        ]
        return SkylineParameter(value, change_times)

    def __add__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self.operate(other, lambda x, y: x + y)

    def __radd__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self.operate(other, lambda x, y: y + x)

    def __sub__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self.operate(other, lambda x, y: x - y)

    def __rsub__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self.operate(other, lambda x, y: y - x)

    def __mul__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self.operate(other, lambda x, y: x * y)

    def __rmul__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self.operate(other, lambda x, y: y * x)

    def __truediv__(self, other: SkylineParameterLike) -> "SkylineParameter":
        return self.operate(other, lambda x, y: x / y)

    def __rtruediv__(self, other: pgt.Scalar) -> "SkylineParameter":
        return self.operate(other, lambda x, y: y / x)

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


def is_skyline_parameter_like(x: object) -> TypeGuard[SkylineParameterLike]:
    return isinstance(x, pgt.Scalar | SkylineParameter)


def is_many_skyline_parameters_like(
    x: object,
) -> TypeGuard[pgt.Many[SkylineParameterLike]]:
    return tg.is_many(x) and all(is_skyline_parameter_like(v) for v in x)
