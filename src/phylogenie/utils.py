from collections.abc import Callable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, BeforeValidator

import phylogenie.typeguards as tg

T = TypeVar("T")
BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


class OrderedSet(Generic[T]):
    def __init__(self):
        self._dict: dict[T, None] = {}

    def add(self, item: T):
        self._dict[item] = None

    def remove(self, item: T):
        del self._dict[item]

    def __contains__(self, item: T) -> bool:
        return item in self._dict

    def __iter__(self) -> Iterator[T]:
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)


class MetadataMixin:
    def __init__(self):
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(self._metadata)

    def set(self, key: str, value: Any):
        self._metadata[key] = value

    def update(self, metadata: Mapping[str, Any]):
        self._metadata.update(metadata)

    def get(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def delete(self, key: str):
        del self._metadata[key]

    def clear(self):
        self._metadata.clear()

    def __getitem__(self, key: str) -> Any:
        return self._metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._metadata[key] = value


class Registry(Generic[BaseModelT]):
    def __init__(self, base_type: type[BaseModelT], discriminator: str = "spec"):
        self.name = base_type.__name__
        self.base_type = base_type
        self.discriminator = discriminator
        self._registry: dict[str, type[BaseModelT]] = {}

    def register(self, key: str) -> Callable[[type[BaseModelT]], type[BaseModelT]]:
        def decorator(cls: type[BaseModelT]) -> type[BaseModelT]:
            if not issubclass(cls, self.base_type):
                raise TypeError(
                    f"{cls.__name__} must be a subclass of {self.base_type.__name__}"
                )
            if key in self._registry:
                raise RuntimeError(f"{self.name}: '{key}' already registered")
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[BaseModelT]:
        if key not in self._registry:
            raise ValueError(
                f"Unknown {self.name} '{key}'. Available: {list(self._registry.keys())}"
            )
        return self._registry[key]

    def registered(self) -> Mapping[str, type[BaseModelT]]:
        return MappingProxyType(self._registry)

    def load(self, value: Any) -> BaseModelT:
        if isinstance(value, self.base_type):
            return value

        if not tg.is_dictionary(value):
            raise TypeError(
                f"Expected a dict or a {self.base_type.__name__} instance, got {type(value)}"
            )

        if self.discriminator not in value:
            raise ValueError(
                f"Missing '{self.discriminator}' field for {self.base_type.__name__}"
            )
        key = value.pop(self.discriminator)

        cls = self.get(key)
        return cls.model_validate(value)

    @property
    def Validator(self) -> BeforeValidator:
        return BeforeValidator(self.load)
