import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.reranking.provider import Reranker
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.schemas import HybridSearchResponse, HybridSearchResult
from app.vector_store.schemas import VectorSearchFilter


def result(document_id: str, rank: int) -> HybridSearchResult:
    entity_id = int(document_id.split(":")[1])
    return HybridSearchResult(
        rank=rank,
        document_id=document_id,
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        fusion_score=1 / (60 + rank),
        semantic_rank=rank,
        semantic_score=1 / rank,
        lexical_rank=rank,
        lexical_score=2 / rank,
        content=f"content for {document_id}",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=entity_id,
            track_id=entity_id,
        ),
    )


class StubHybridRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, limit, filters=None, semantic_score_threshold=None):
        self.calls.append((query, limit, filters, semantic_score_threshold))
        return HybridSearchResponse(
            query=" ".join(query.split()),
            semantic_candidate_count=len(self.results),
            lexical_candidate_count=len(self.results),
            results=self.results[:limit],
        )


class StubReranker(Reranker):
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    @property
    def model_name(self):
        return "stub-model"

    @property
    def version(self):
        return "1"

    def score(self, query, documents):
        self.calls.append((query, documents))
        return list(self.scores)


def test_reranked_retriever_reorders_hybrid_candidates_and_preserves_provenance():
    hybrid = StubHybridRetriever(
        (result("track:1", 1), result("track:2", 2), result("track:3", 3))
    )
    reranker = StubReranker((0.2, 0.9, 0.4))

    response = RerankedRetriever(hybrid, reranker, candidate_limit=3).search(
        " dark   music ", limit=2
    )

    assert [item.document_id for item in response.results] == ["track:2", "track:3"]
    assert [item.rank for item in response.results] == [1, 2]
    assert [item.hybrid_rank for item in response.results] == [2, 3]
    assert response.results[0].reranker_score == 0.9
    assert response.results[0].fusion_score == pytest.approx(1 / 62)
    assert response.candidate_count == 3
    assert response.reranker_model == "stub-model"
    assert reranker.calls == [
        (
            "dark music",
            ["content for track:1", "content for track:2", "content for track:3"],
        )
    ]


def test_reranked_retriever_forwards_filters_threshold_and_candidate_limit():
    hybrid = StubHybridRetriever(())
    reranker = StubReranker(())
    filters = VectorSearchFilter(genres=("ambient",))
    RerankedRetriever(hybrid, reranker, candidate_limit=20).search(
        "ambient",
        limit=5,
        filters=filters,
        semantic_score_threshold=0.3,
    )
    assert hybrid.calls == [("ambient", 20, filters, 0.3)]


def test_reranked_retriever_breaks_equal_score_ties_by_hybrid_rank():
    hybrid = StubHybridRetriever((result("track:2", 1), result("track:1", 2)))
    response = RerankedRetriever(
        hybrid, StubReranker((0.5, 0.5)), candidate_limit=2
    ).search("music", limit=2)
    assert [item.document_id for item in response.results] == ["track:2", "track:1"]


@pytest.mark.parametrize("scores", [(0.5,), (0.5, float("inf"))])
def test_reranked_retriever_rejects_invalid_provider_output(scores):
    hybrid = StubHybridRetriever((result("track:1", 1), result("track:2", 2)))
    with pytest.raises(RuntimeError):
        RerankedRetriever(hybrid, StubReranker(scores), candidate_limit=2).search(
            "music", limit=2
        )


@pytest.mark.parametrize("candidate_limit", [0, 101])
def test_reranked_retriever_validates_candidate_limit(candidate_limit):
    with pytest.raises(ValueError, match="candidate limit"):
        RerankedRetriever(StubHybridRetriever(()), StubReranker(()), candidate_limit)


@pytest.mark.parametrize("limit", [0, 21])
def test_reranked_retriever_validates_result_limit(limit):
    retriever = RerankedRetriever(
        StubHybridRetriever(()), StubReranker(()), candidate_limit=20
    )
    with pytest.raises(ValueError, match="result limit"):
        retriever.search("music", limit=limit)
