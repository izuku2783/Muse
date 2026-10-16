from abc import ABC, abstractmethod
from collections.abc import Sequence

from app.vector_store.schemas import VectorRecord, VectorSearchFilter, VectorSearchResult


class VectorStore(ABC):
    @abstractmethod
    def ensure_collection(
        self,
        vector_size: int,
        embedding_model: str,
        embedding_version: str,
        recreate: bool = False,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def upsert(self, records: Sequence[VectorRecord]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        vector: list[float],
        embedding_model: str,
        embedding_version: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        raise NotImplementedError

    @abstractmethod
    def list_document_ids(
        self, embedding_model: str, embedding_version: str
    ) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_ids: Sequence[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        """Release vector-store resources, when applicable."""
