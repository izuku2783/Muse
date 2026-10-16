import pytest

from app.api.routes.rag import get_rag_service
from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.main import app
from app.llm.errors import LLMConfigurationError, LLMProviderError
from app.llm.schemas import TokenUsage
from app.rag.schemas import (
    EvidenceItem,
    GroundedResponse,
    GroundedStatement,
    RAGGenerationMetadata,
)
from app.rag.service import GroundingError


def grounded_response() -> GroundedResponse:
    source = EvidenceItem(
        evidence_id="E1",
        document_id="track:1",
        entity_type=DocumentType.TRACK,
        entity_id=1,
        content="Night Drive is a synthwave track by Neon Waves.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=1,
            track_id=1,
            artist_name="Neon Waves",
        ),
        retrieval_rank=1,
        reranker_score=0.9,
    )
    citation = GroundedStatement(
        text="Night Drive is by Neon Waves.",
        evidence_ids=("E1",),
    )
    return GroundedResponse(
        answer="Night Drive matches the request.",
        retrieved_facts=(citation,),
        recommendation_reasoning=(citation,),
        uncertainty=(),
        sources=(source,),
        generation=RAGGenerationMetadata(
            provider="test",
            model="test-model",
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=25,
                total_tokens=125,
            ),
        ),
    )


class FakeRAGService:
    def __init__(self, response=None, error=None):
        self.response = response or grounded_response()
        self.error = error
        self.calls = []

    def answer(self, query, filters=None, semantic_score_threshold=None):
        self.calls.append((query, filters, semantic_score_threshold))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def override_rag_service():
    services = []

    def install(service):
        services.append(service)
        app.dependency_overrides[get_rag_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_rag_service, None)


def test_answer_endpoint_returns_grounded_sections_and_sources(
    client,
    override_rag_service,
):
    service = override_rag_service(FakeRAGService())
    response = client.post(
        "/answer",
        json={
            "query": "dark synthwave",
            "filters": {"genres": ["Synthwave"]},
            "semantic_score_threshold": 0.2,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Night Drive matches the request."
    assert body["retrieved_facts"][0]["evidence_ids"] == ["E1"]
    assert body["sources"][0]["document_id"] == "track:1"
    assert body["generation"]["provider"] == "test"
    assert body["generation"]["usage"]["total_tokens"] == 125
    query, filters, threshold = service.calls[0]
    assert query == "dark synthwave"
    assert filters.genres == ("synthwave",)
    assert threshold == 0.2


@pytest.mark.parametrize(
    "payload",
    [
        {"query": ""},
        {"query": "   "},
        {"query": "music", "semantic_score_threshold": 1.1},
        {"query": "music", "unexpected": True},
        {"query": "music", "filters": {"genres": [""]}},
    ],
)
def test_answer_endpoint_validates_request(client, override_rag_service, payload):
    override_rag_service(FakeRAGService())
    assert client.post("/answer", json=payload).status_code == 422


@pytest.mark.parametrize(
    "error,expected_status,expected_detail",
    [
        (
            LLMConfigurationError("missing key"),
            503,
            "RAG generation is not configured.",
        ),
        (LLMProviderError("provider failed"), 502, "RAG generation failed."),
        (GroundingError("bad citation"), 502, "RAG generation failed."),
    ],
)
def test_answer_endpoint_translates_generation_failures(
    client,
    override_rag_service,
    error,
    expected_status,
    expected_detail,
):
    override_rag_service(FakeRAGService(error=error))
    response = client.post("/answer", json={"query": "music"})
    assert response.status_code == expected_status
    assert response.json() == {"detail": expected_detail}
