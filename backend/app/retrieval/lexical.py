import math
import re
import unicodedata
from collections import Counter

from app.knowledge.schemas import KnowledgeDocument
from app.retrieval.filters import document_matches_filter
from app.retrieval.schemas import LexicalSearchResponse, LexicalSearchResult
from app.vector_store.schemas import VectorSearchFilter


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(value: str) -> list[str]:
    return _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", value).casefold())


class BM25LexicalRetriever:
    """Deterministic in-process BM25 over Stage 3 knowledge documents."""

    def __init__(
        self,
        documents: list[KnowledgeDocument],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("BM25 requires k1 > 0 and b between 0 and 1.")
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Lexical documents must have unique document IDs.")
        self.documents = tuple(documents)
        self.k1 = k1
        self.b = b
        self._tokens = [tokenize(document.content) for document in documents]
        self._term_frequencies = [Counter(tokens) for tokens in self._tokens]
        self._average_length = (
            sum(len(tokens) for tokens in self._tokens) / len(self._tokens)
            if self._tokens
            else 0.0
        )
        self._document_frequency = Counter(
            token for tokens in self._tokens for token in set(tokens)
        )

    def _score(self, query_tokens: list[str], index: int) -> float:
        if not self._average_length:
            return 0.0
        frequencies = self._term_frequencies[index]
        document_length = len(self._tokens[index])
        score = 0.0
        for token in set(query_tokens):
            frequency = frequencies.get(token, 0)
            if not frequency:
                continue
            document_frequency = self._document_frequency[token]
            inverse_document_frequency = math.log(
                1
                + (
                    len(self.documents) - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / self._average_length
            )
            score += inverse_document_frequency * (
                frequency * (self.k1 + 1) / denominator
            )
        return score

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
    ) -> LexicalSearchResponse:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("Lexical search query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("Lexical search limit must be between 1 and 100.")
        query_tokens = tokenize(cleaned_query)
        scored = [
            (self._score(query_tokens, index), document)
            for index, document in enumerate(self.documents)
            if document_matches_filter(document, filters)
        ]
        ranked = sorted(
            ((score, document) for score, document in scored if score > 0),
            key=lambda item: (-item[0], item[1].document_id),
        )[:limit]
        return LexicalSearchResponse(
            query=cleaned_query,
            results=tuple(
                LexicalSearchResult(
                    rank=rank,
                    document_id=document.document_id,
                    entity_type=document.document_type,
                    entity_id=document.metadata.entity_id,
                    score=score,
                    content=document.content,
                    metadata=document.metadata,
                )
                for rank, (score, document) in enumerate(ranked, start=1)
            ),
        )
