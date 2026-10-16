import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schemas import (
    LexicalSearchResponse,
    LexicalSearchResult,
    SemanticSearchResponse,
    SemanticSearchResult,
)
from app.vector_store.schemas import VectorSearchFilter


def metadata(entity_id: int) -> DocumentMetadata:
    return DocumentMetadata(
        document_type=DocumentType.TRACK,
        entity_id=entity_id,
        track_id=entity_id,
    )


def semantic_result(document_id: str, rank: int, score: float):
    entity_id = int(document_id.split(":")[1])
    return SemanticSearchResult(
        rank=rank,
        document_id=document_id,
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        score=score,
        content=f"content for {document_id}",
        metadata=metadata(entity_id),
    )


def lexical_result(document_id: str, rank: int, score: float):
    entity_id = int(document_id.split(":")[1])
    return LexicalSearchResult(
        rank=rank,
        document_id=document_id,
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        score=score,
        content=f"content for {document_id}",
        metadata=metadata(entity_id),
    )


class StubSemanticRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, limit=10, filters=None, score_threshold=None):
        self.calls.append((query, limit, filters, score_threshold))
        return SemanticSearchResponse(query=" ".join(query.split()), results=self.results)


class StubLexicalRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, limit=10, filters=None):
        self.calls.append((query, limit, filters))
        return LexicalSearchResponse(query=" ".join(query.split()), results=self.results)


def hybrid(semantic_results=(), lexical_results=(), **kwargs):
    semantic = StubSemanticRetriever(semantic_results)
    lexical = StubLexicalRetriever(lexical_results)
    return HybridRetriever(semantic, lexical, **kwargs), semantic, lexical


def test_hybrid_uses_exact_unweighted_reciprocal_rank_fusion():
    retriever, _, _ = hybrid(
        (
            semantic_result("track:1", 1, 0.9),
            semantic_result("track:2", 2, 0.8),
        ),
        (
            lexical_result("track:2", 1, 2.5),
            lexical_result("track:3", 2, 1.5),
        ),
    )

    response = retriever.search("dark music")

    assert [result.document_id for result in response.results] == [
        "track:2",
        "track:1",
        "track:3",
    ]
    combined = response.results[0]
    assert combined.fusion_score == pytest.approx(1 / 62 + 1 / 61)
    assert combined.semantic_rank == 2
    assert combined.semantic_score == 0.8
    assert combined.lexical_rank == 1
    assert combined.lexical_score == 2.5
    assert response.semantic_candidate_count == 2
    assert response.lexical_candidate_count == 2


def test_hybrid_keeps_candidates_found_by_only_one_retriever():
    retriever, _, _ = hybrid(
        (semantic_result("track:1", 1, 0.9),),
        (lexical_result("track:2", 1, 3.0),),
    )
    response = retriever.search("music")
    assert {result.document_id for result in response.results} == {"track:1", "track:2"}
    semantic_only = next(item for item in response.results if item.document_id == "track:1")
    lexical_only = next(item for item in response.results if item.document_id == "track:2")
    assert semantic_only.lexical_rank is None
    assert lexical_only.semantic_rank is None


def test_hybrid_forwards_filters_threshold_and_candidate_limit():
    retriever, semantic, lexical = hybrid(candidate_limit=20)
    filters = VectorSearchFilter(genres=("ambient",))

    retriever.search(
        "ambient",
        limit=5,
        filters=filters,
        semantic_score_threshold=0.4,
    )

    assert semantic.calls == [("ambient", 20, filters, 0.4)]
    assert lexical.calls == [("ambient", 20, filters)]


def test_hybrid_fetches_enough_candidates_when_result_limit_is_larger():
    retriever, semantic, lexical = hybrid(candidate_limit=5)
    retriever.search("music", limit=8)
    assert semantic.calls[0][1] == 8
    assert lexical.calls[0][1] == 8


def test_hybrid_has_deterministic_document_id_tie_breaking():
    retriever, _, _ = hybrid(
        (semantic_result("track:2", 1, 0.9),),
        (lexical_result("track:1", 1, 3.0),),
    )
    response = retriever.search("music")
    assert [result.document_id for result in response.results] == ["track:1", "track:2"]


@pytest.mark.parametrize("limit", [0, 101])
def test_hybrid_validates_limit(limit):
    retriever, _, _ = hybrid()
    with pytest.raises(ValueError, match="between 1 and 100"):
        retriever.search("music", limit=limit)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rrf_k": 0}, "RRF k must be positive"),
        ({"candidate_limit": 0}, "Candidate limit"),
        ({"candidate_limit": 101}, "Candidate limit"),
    ],
)
def test_hybrid_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        hybrid(**kwargs)
