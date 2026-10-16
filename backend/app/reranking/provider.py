from abc import ABC, abstractmethod
from collections.abc import Sequence


class Reranker(ABC):
    """Provider-independent contract for query-document relevance scoring."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        raise NotImplementedError

    @property
    def signature(self) -> str:
        return f"{self.model_name}:{self.version}"
