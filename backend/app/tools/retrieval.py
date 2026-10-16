from typing import Protocol
from uuid import UUID

from app.query_understanding.schemas import PopularityPreference
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.schemas import RerankedSearchResponse, SemanticSearchResponse
from app.retrieval.semantic import SemanticRetriever
from app.tools.base import BaseTool
from app.tools.schemas import RerankedSearchInput, SemanticSearchInput
from app.vector_store.schemas import VectorSearchFilter


class EvidenceRetriever(Protocol):
    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        semantic_score_threshold: float | None = None,
        user_id: UUID | None = None,
        query_popularity: PopularityPreference | None = None,
    ) -> RerankedSearchResponse: ...


class SemanticSearchTool(BaseTool[SemanticSearchInput, SemanticSearchResponse]):
    name = "semantic_search"
    description = "Search indexed music knowledge by meaning and optional hard filters."
    input_type = SemanticSearchInput
    output_type = SemanticSearchResponse

    def __init__(self, retriever: SemanticRetriever):
        self.retriever = retriever

    def run(self, tool_input: SemanticSearchInput) -> SemanticSearchResponse:
        return self.retriever.search(
            query=tool_input.query,
            limit=tool_input.limit,
            filters=tool_input.filters,
            score_threshold=tool_input.score_threshold,
        )


class RetrieveMusicEvidenceTool(
    BaseTool[RerankedSearchInput, RerankedSearchResponse]
):
    name = "retrieve_music_evidence"
    description = (
        "Retrieve filtered music knowledge with hybrid search and cross-encoder reranking."
    )
    input_type = RerankedSearchInput
    output_type = RerankedSearchResponse

    def __init__(self, retriever: EvidenceRetriever):
        self.retriever = retriever

    def run(self, tool_input: RerankedSearchInput) -> RerankedSearchResponse:
        arguments = dict(
            query=tool_input.query,
            limit=tool_input.limit,
            filters=tool_input.filters,
            semantic_score_threshold=tool_input.score_threshold,
        )
        if tool_input.user_id is not None or tool_input.query_popularity is not None:
            arguments.update(
                user_id=tool_input.user_id,
                query_popularity=tool_input.query_popularity,
            )
        return self.retriever.search(**arguments)

