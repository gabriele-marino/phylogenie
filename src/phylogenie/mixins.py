from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class MetadataMixin:
    def __init__(self) -> None:
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        return MappingProxyType(self._metadata)

    def set(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def update(self, metadata: Mapping[str, Any]) -> None:
        self._metadata.update(metadata)

    def get(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def delete(self, key: str) -> None:
        self._metadata.pop(key, None)

    def clear(self) -> None:
        self._metadata.clear()

    def __getitem__(self, key: str) -> Any:
        return self._metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._metadata[key] = value
