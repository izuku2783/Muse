import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.rag.context import ContextBuilder
from app.rag.schemas import (
    GeneratedGroundedAnswer,
    GroundedStatement,
    RAGGenerationMetadata,
    RAGGenerationResult,
)
from app.rag.service import GroundingError, RAGService
from app.retrieval.schemas import RerankedSearchResponse, RerankedSearchResult
from app.vector_store.schemas import VectorSearchFilter


def result() -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=1,
        hybrid_rank=2,
        document_id="track:1",
        entity_type=DocumentType.TRACK,
        entity_id=1,
        reranker_score=0.8,
        fusion_score=0.03,
        content="Night Drive is a synthwave track by Neon Waves.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=1,
            track_id=1,
            artist_name="Neon Waves",
            genres=("synthwave",),
        ),
    )


class StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, limit, filters=None, semantic_score_threshold=None):
        self.calls.append((query, limit, filters, semantic_score_threshold))
        return RerankedSearchResponse(
            query=" ".join(query.split()),
            candidate_count=len(self.results),
            reranker_model="test",
            reranker_version="1",
            results=self.results,
        )


class RecordingGenerator:
    def __init__(self, answer):
        self.answer_value = answer
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        return RAGGenerationResult(
            output=self.answer_value,
            metadata=RAGGenerationMetadata(
                provider="test",
                model="test-model",
            ),
        )


def generated(citation="E1"):
    return GeneratedGroundedAnswer(
        answer="Night Drive matches the request.",
        retrieved_facts=(
            GroundedStatement(
                text="Night Drive is by Neon Waves.",
                evidence_ids=(citation,),
            ),
        ),
        recommendation_reasoning=(
            GroundedStatement(
                text="Its synthwave metadata matches the requested style.",
                evidence_ids=(citation,),
            ),
        ),
        uncertainty=("The evidence does not include popularity.",),
    )


def test_rag_service_passes_structured_evidence_and_returns_sources():
    retriever = StubRetriever((result(),))
    generator = RecordingGenerator(generated())
    filters = VectorSearchFilter(genres=("synthwave",))
    response = RAGService(
        retriever,
        generator,
        ContextBuilder(max_evidence=3),
    ).answer("dark synthwave", filters=filters, semantic_score_threshold=0.2)

    assert response.answer == "Night Drive matches the request."
    assert response.sources[0].evidence_id == "E1"
    assert response.retrieved_facts[0].evidence_ids == ("E1",)
    assert response.generation.provider == "test"
    assert response.generation.model == "test-model"
    assert retriever.calls == [("dark synthwave", 3, filters, 0.2)]
    request = generator.requests[0]
    assert request.evidence[0].content == result().content
    assert "Never invent music metadata" in request.system_instructions
    assert "must introduce no additional claims" in request.system_instructions


def test_rag_service_rejects_unknown_citations():
    service = RAGService(
        StubRetriever((result(),)),
        RecordingGenerator(generated("E99")),
    )
    with pytest.raises(GroundingError, match="E99"):
        service.answer("music")


def test_rag_service_returns_explicit_uncertainty_without_evidence():
    generator = RecordingGenerator(generated())
    response = RAGService(StubRetriever(()), generator).answer("unknown music")
    assert "not have enough evidence" in response.answer
    assert response.retrieved_facts == ()
    assert response.sources == ()
    assert response.generation is None
    assert generator.requests == []
