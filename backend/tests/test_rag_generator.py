import json

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.llm.schemas import (
    ChatCompletionResponse,
    StructuredCompletion,
    TokenUsage,
)
from app.rag.generator import LLMRAGGenerator
from app.rag.schemas import (
    EvidenceItem,
    GeneratedGroundedAnswer,
    GroundedStatement,
    RAGGenerationRequest,
)


def generation_request() -> RAGGenerationRequest:
    return RAGGenerationRequest(
        system_instructions="Use only the supplied evidence.",
        query="dark synthwave",
        evidence=(
            EvidenceItem(
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
            ),
        ),
    )


class FakeProvider:
    provider_name = "fake"
    default_model = "fake-default"

    def __init__(self):
        self.calls = []

    def complete_structured(self, request, response_model, schema_name):
        self.calls.append((request, response_model, schema_name))
        output = GeneratedGroundedAnswer(
            answer="Night Drive matches the request.",
            retrieved_facts=(
                GroundedStatement(
                    text="Night Drive is by Neon Waves.",
                    evidence_ids=("E1",),
                ),
            ),
            recommendation_reasoning=(),
            uncertainty=(),
        )
        return StructuredCompletion(
            output=output,
            response=ChatCompletionResponse(
                content=output.model_dump_json(),
                model="fake-returned",
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=100,
                    completion_tokens=25,
                    total_tokens=125,
                ),
            ),
        )


def test_rag_generator_sends_structured_evidence_and_returns_metadata():
    provider = FakeProvider()
    result = LLMRAGGenerator(provider)(generation_request())

    assert result.output.answer == "Night Drive matches the request."
    assert result.metadata.provider == "fake"
    assert result.metadata.model == "fake-returned"
    assert result.metadata.usage.total_tokens == 125
    request, response_model, schema_name = provider.calls[0]
    assert response_model is GeneratedGroundedAnswer
    assert schema_name == "muse_grounded_answer"
    assert request.messages[0].content == "Use only the supplied evidence."
    supplied = json.loads(request.messages[1].content)
    assert supplied["query"] == "dark synthwave"
    assert supplied["evidence"][0]["evidence_id"] == "E1"
