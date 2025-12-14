from types import MappingProxyType
from typing import Any, Mapping, Optional


class MetadataMixin:
    """A mixin that provides metadata management with dictionary-like access."""

    def __init__(self) -> None:
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        """Return a read-only view of all metadata."""
        return MappingProxyType(self._metadata)

    def set(self, key: str, value: Any) -> None:
        """Set or update a metadata value."""
        self._metadata[key] = value

    def update(self, metadata: Mapping[str, Any]) -> None:
        """Bulk update metadata values."""
        self._metadata.update(metadata)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a metadata value, returning `default` if not found."""
        return self._metadata.get(key, default)

    def delete(self, key: str) -> None:
        """Delete a metadata if it exists, else do nothing."""
        self._metadata.pop(key, None)

    def clear(self) -> None:
        """Remove all metadata."""
        self._metadata.clear()

    # Dict-like behavior
    def __getitem__(self, key: str) -> Any:
        return self._metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._metadata[key] = value
