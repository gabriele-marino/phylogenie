from typing import Annotated, Sequence

from pydantic import BeforeValidator

from phylogenie.utils.type_hints import OneOrSequence, T


def ensure_list(
    value: OneOrSequence[T] | None, default: list[T] | None = None
) -> list[T]:
    return (
        list(value)
        if isinstance(value, Sequence)
        else [value] if value is not None else default if default is not None else []
    )


EnsuredList = Annotated[list[T], BeforeValidator(ensure_list)]
