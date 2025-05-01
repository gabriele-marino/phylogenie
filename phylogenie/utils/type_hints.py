from typing import Sequence, TypeVar, Union

T = TypeVar("T")

Numeric = float | int

OneOrSequence = Union[T | Sequence[T]]
OneOrMatrix = Union[T | Sequence[Sequence[T]]]

Vector = OneOrSequence[Numeric]
