from bisect import bisect_right
from typing import Callable, Union

from phylogenie.utils.type_hints import Numeric, Vector
from phylogenie.utils.validators import ensure_list

ParameterOperand = Union[Numeric, "SkylineParameter"]

SerialSkylineParameter = Numeric | dict[str, Vector]


class SkylineParameter:
    def __init__(
        self,
        value: Vector,
        change_times: Vector | None = None,
    ):
        self.value = ensure_list(value)
        self.change_times = ensure_list(change_times)
        if len(self.value) != len(self.change_times) + 1:
            raise ValueError(
                f"`value` must have exactly one more element than `change_times` "
                f"(got value={self.value} and change_times={self.change_times})"
            )

    def get_value_at_time(self, time: Numeric) -> Numeric:
        return self.value[bisect_right(self.change_times, time)]

    def operate(
        self,
        other: ParameterOperand,
        func: Callable[[Numeric, Numeric], Numeric],
    ) -> "SkylineParameter":
        if isinstance(other, Numeric):
            other = SkylineParameter(other)
        change_times = sorted(set(self.change_times + other.change_times))
        value = [
            func(self.get_value_at_time(t), other.get_value_at_time(t))
            for t in [0] + change_times
        ]
        return SkylineParameter(value, change_times)

    def __add__(self, other: ParameterOperand) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x + y)

    def __radd__(self, other: Numeric) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x + y)

    def __sub__(self, other: ParameterOperand) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x - y)

    def __rsub__(self, other: Numeric) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: y - x)

    def __mul__(self, other: ParameterOperand) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x * y)

    def __rmul__(self, other: Numeric) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x * y)

    def __truediv__(self, other: ParameterOperand) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: x / y)

    def __rtruediv__(self, other: Numeric) -> "SkylineParameter":
        return self.operate(other, func=lambda x, y: y / x)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, SkylineParameter)
            and self.value == other.value
            and self.change_times == other.change_times
        )

    def __bool__(self) -> bool:
        return any(self.value)

    def __repr__(self) -> str:
        return f"SkylineParameter(value={self.value}, change_times={self.change_times})"

    def serialize(self) -> SerialSkylineParameter:
        if len(self.value) == 1:
            return self.value[0]
        return {
            "value": self.value,
            "change_times": (
                self.change_times
                if len(self.change_times) > 1
                else self.change_times[0]
            ),
        }
