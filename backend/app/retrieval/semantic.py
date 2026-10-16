from app.embeddings.provider import EmbeddingProvider
from app.retrieval.schemas import SemanticSearchResponse, SemanticSearchResult
from app.vector_store.provider import VectorStore
from app.vector_store.schemas import VectorSearchFilter


class SemanticRetriever:
    """Embed a natural-language query and retrieve ranked knowledge entities."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> SemanticSearchResponse:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("Semantic search query must not be empty.")
        if not 1 <= limit <= 100:
            raise ValueError("Semantic search limit must be between 1 and 100.")

        provider = self.embedding_provider
        query_vector = provider.embed_query(cleaned_query)
        matches = self.vector_store.search(
            vector=query_vector,
            embedding_model=provider.model_name,
            embedding_version=provider.version,
            limit=limit,
            filters=filters,
            score_threshold=score_threshold,
        )
        return SemanticSearchResponse(
            query=cleaned_query,
            results=tuple(
                SemanticSearchResult(
                    rank=rank,
                    document_id=match.document_id,
                    entity_type=match.metadata.document_type,
                    entity_id=match.metadata.entity_id,
                    score=match.score,
                    content=match.content,
                    metadata=match.metadata,
                )
                for rank, match in enumerate(matches, start=1)
            ),
        )
