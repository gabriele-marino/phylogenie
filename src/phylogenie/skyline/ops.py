from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from operator import add, mul, sub, truediv
from typing import Any, Generic, TypeGuard, TypeVar

OperandT = TypeVar("OperandT")
ElemT = TypeVar("ElemT")
ResultT = TypeVar("ResultT")


class SkylineBinaryOpsMixin(Generic[OperandT, ElemT, ResultT], ABC):
    """Shared binary operator wiring for Skyline types."""

    @abstractmethod
    def _operate(
        self, other: OperandT, func: Callable[[ElemT, ElemT], ElemT]
    ) -> ResultT:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def is_valid_operand(cls, other: Any) -> TypeGuard[OperandT]: ...

    def _binary(
        self,
        other: OperandT,
        op: Callable[[ElemT, ElemT], ElemT],
        reverse: bool = False,
    ) -> ResultT:
        if not self.is_valid_operand(other):
            return NotImplemented
        if reverse:
            return self._operate(other, lambda x, y: op(y, x))
        return self._operate(other, op)

    def __add__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=add)

    def __radd__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=add, reverse=True)

    def __sub__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=sub)

    def __rsub__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=sub, reverse=True)

    def __mul__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=mul)

    def __rmul__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=mul, reverse=True)

    def __truediv__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=truediv)

    def __rtruediv__(self, other: OperandT) -> ResultT:
        return self._binary(other, op=truediv, reverse=True)
