import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.rag.context import ContextBuilder
from app.retrieval.schemas import RerankedSearchResponse, RerankedSearchResult


def result(entity_id: int, rank: int, content: str) -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=rank,
        hybrid_rank=rank,
        document_id=f"track:{entity_id}",
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        reranker_score=1 / rank,
        fusion_score=1 / (60 + rank),
        content=content,
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=entity_id,
            track_id=entity_id,
        ),
    )


def response(*results):
    return RerankedSearchResponse(
        query="dark music",
        candidate_count=len(results),
        reranker_model="test",
        reranker_version="1",
        results=results,
    )


def test_context_builder_creates_stable_structured_evidence():
    context = ContextBuilder(max_evidence=2).build(
        response(
            result(1, 1, "First grounded document."),
            result(2, 2, "Second grounded document."),
            result(3, 3, "Third document is outside the evidence limit."),
        )
    )
    assert [item.evidence_id for item in context.evidence] == ["E1", "E2"]
    assert [item.document_id for item in context.evidence] == ["track:1", "track:2"]
    assert context.evidence[0].content == "First grounded document."
    assert context.evidence[0].metadata.track_id == 1


def test_context_builder_respects_character_budget_after_first_document():
    context = ContextBuilder(max_characters=500).build(
        response(
            result(1, 1, "a" * 400),
            result(2, 2, "b" * 200),
        )
    )
    assert [item.document_id for item in context.evidence] == ["track:1"]


def test_context_builder_keeps_first_document_when_it_exceeds_budget():
    context = ContextBuilder(max_characters=500).build(
        response(result(1, 1, "a" * 600))
    )
    assert len(context.evidence) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_evidence": 0},
        {"max_evidence": 21},
        {"max_characters": 499},
    ],
)
def test_context_builder_validates_configuration(kwargs):
    with pytest.raises(ValueError):
        ContextBuilder(**kwargs)
