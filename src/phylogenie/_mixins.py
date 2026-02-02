from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


class MetadataMixin:
    """
    Provide a dictionary-backed metadata interface.

    This mixin exposes read-only access to metadata and helper methods for
    mutating key/value annotations in a consistent way.
    """

    def __init__(self):
        self._metadata: dict[str, Any] = {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        """
        Return a read-only view of metadata.

        Returns
        --------
        Mapping[str, Any]
            An immutable view of the metadata dictionary.
        """
        return MappingProxyType(self._metadata)

    def set(self, key: str, value: Any):
        """
        Set a metadata key to a value.

        Parameters
        -----------
        key : str
            Metadata key to set.
        value : Any
            Value to store under the key.
        """
        self._metadata[key] = value

    def update(self, metadata: Mapping[str, Any]):
        """
        Merge a mapping of metadata into the existing metadata.

        Parameters
        -----------
        metadata : Mapping[str, Any]
            Key/value pairs to merge into metadata.
        """
        self._metadata.update(metadata)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Fetch a metadata value with an optional default.

        Parameters
        -----------
        key : str
            Metadata key to retrieve.
        default : Any, optional
            Value to return if the key is missing.
        """
        return self._metadata.get(key, default)

    def delete(self, key: str):
        """
        Delete a metadata key.

        Parameters
        -----------
        key : str
            Metadata key to delete.
        """
        del self._metadata[key]

    def clear(self):
        """
        Remove all metadata entries.
        """
        self._metadata.clear()

    def __getitem__(self, key: str) -> Any:
        return self._metadata[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._metadata[key] = value
