class VectorStoreError(Exception):
    """Base exception for vector-store operations."""


class VectorStoreConfigurationError(VectorStoreError):
    """The collection is incompatible with the configured embedding model."""
