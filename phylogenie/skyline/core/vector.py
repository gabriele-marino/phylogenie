from typing import Callable, Iterator, Union

from kitpy.type_hints import Numeric, OneOrSequence
from kitpy.validators import ensure_list

from phylogenie.skyline.core.parameter import (
    ParameterOperand,
    SerialSkylineParameter,
    SkylineParameter,
)

VectorOperand = Union[ParameterOperand, "SkylineVector"]

SerialSkylineVector = SerialSkylineParameter | dict[str, SerialSkylineParameter]


class SkylineVector:
    def __init__(self, params: OneOrSequence[SkylineParameter | Numeric]):
        params = ensure_list(params)
        self.params = [
            (param if isinstance(param, SkylineParameter) else SkylineParameter(param))
            for param in params
        ]

    def get_value_at_time(self, time: Numeric) -> list[Numeric]:
        return [param.get_value_at_time(time) for param in self.params]

    @property
    def change_times(self) -> list[Numeric]:
        return sorted(set([t for param in self.params for t in param.change_times]))

    def _operate(
        self,
        other: VectorOperand,
        func: Callable[[SkylineParameter, ParameterOperand], SkylineParameter],
    ) -> "SkylineVector":
        if isinstance(other, SkylineVector):
            if len(self.params) != len(other.params):
                raise ValueError(
                    f"Cannot operate on SkylineVectors of different lengths: "
                    f"{len(self.params)} vs {len(other.params)}"
                )
            params = [func(p1, p2) for p1, p2 in zip(self.params, other.params)]
        else:
            params = [func(p, other) for p in self.params]
        return SkylineVector(params)

    def __add__(self, other: VectorOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: x + y)

    def __radd__(self, other: ParameterOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: y + x)

    def __sub__(self, other: VectorOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: x - y)

    def __rsub__(self, other: ParameterOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: y - x)

    def __mul__(self, other: VectorOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: x * y)

    def __rmul__(self, other: ParameterOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: y * x)

    def __truediv__(self, other: VectorOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: x / y)

    def __rtruediv__(self, other: ParameterOperand) -> "SkylineVector":
        return self._operate(other, lambda x, y: y / x)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SkylineVector) and self.params == other.params

    def __bool__(self) -> bool:
        return any(self.params)

    def __repr__(self) -> str:
        return f"SkylineVector(params={self.params})"

    def __len__(self) -> int:
        return len(self.params)

    def __iter__(self) -> Iterator[SkylineParameter]:
        return iter(self.params)

    def serialize(self, keys: list[str] | None = None) -> SerialSkylineVector:
        if len(self.params) == 1 or all(p == self.params[0] for p in self.params):
            return self.params[0].serialize()
        if keys is None:
            keys = [str(i + 1) for i in range(len(self.params))]
        return {key: param.serialize() for key, param in zip(keys, self.params)}
