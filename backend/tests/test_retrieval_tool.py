from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.retrieval.schemas import (
    RerankedSearchResponse,
    RerankedSearchResult,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.tools.retrieval import RetrieveMusicEvidenceTool, SemanticSearchTool


class RecordingRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, limit=10, filters=None, score_threshold=None):
        self.calls.append((query, limit, filters, score_threshold))
        return SemanticSearchResponse(
            query=query,
            results=(
                SemanticSearchResult(
                    rank=1,
                    document_id="track:7",
                    entity_type=DocumentType.TRACK,
                    entity_id=7,
                    score=0.9,
                    content="A dark synthwave track.",
                    metadata=DocumentMetadata(
                        document_type=DocumentType.TRACK,
                        entity_id=7,
                        track_id=7,
                        genres=("synthwave",),
                    ),
                ),
            ),
        )


def test_semantic_search_tool_forwards_typed_filters_and_returns_evidence():
    retriever = RecordingRetriever()

    output = SemanticSearchTool(retriever).invoke(
        {
            "query": " dark   electronic music ",
            "limit": 4,
            "filters": {"genres": ["Synthwave"], "release_year_min": 2010},
            "score_threshold": 0.25,
        }
    )

    query, limit, filters, threshold = retriever.calls[0]
    assert query == "dark electronic music"
    assert limit == 4
    assert filters.genres == ("synthwave",)
    assert threshold == 0.25
    assert output.results[0].entity_id == 7


class RecordingRerankedRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, limit=10, filters=None, semantic_score_threshold=None):
        self.calls.append((query, limit, filters, semantic_score_threshold))
        metadata = DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=7,
            track_id=7,
        )
        return RerankedSearchResponse(
            query=query,
            candidate_count=1,
            reranker_model="test-reranker",
            reranker_version="1",
            results=(
                RerankedSearchResult(
                    rank=1,
                    hybrid_rank=1,
                    document_id="track:7",
                    entity_type=DocumentType.TRACK,
                    entity_id=7,
                    reranker_score=0.8,
                    fusion_score=0.03,
                    semantic_rank=1,
                    semantic_score=0.9,
                    content="A dark synthwave track.",
                    metadata=metadata,
                ),
            ),
        )


def test_reranked_evidence_tool_preserves_filter_and_ranking_provenance():
    retriever = RecordingRerankedRetriever()

    output = RetrieveMusicEvidenceTool(retriever).invoke(
        {"query": "dark music", "filters": {"genres": ["synthwave"]}}
    )

    assert retriever.calls[0][2].genres == ("synthwave",)
    assert output.reranker_model == "test-reranker"
    assert output.results[0].reranker_score == 0.8

