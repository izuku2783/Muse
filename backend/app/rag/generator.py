import json
from typing import Protocol

from app.llm.provider import LLMProvider
from app.llm.schemas import ChatCompletionRequest, ChatMessage, ChatRole
from app.rag.schemas import (
    GeneratedGroundedAnswer,
    RAGGenerationMetadata,
    RAGGenerationRequest,
    RAGGenerationResult,
)


class RAGGenerator(Protocol):
    def __call__(self, request: RAGGenerationRequest) -> RAGGenerationResult: ...


class LLMRAGGenerator:
    """Adapt the provider-neutral LLM interface to the RAG response schema."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def __call__(self, request: RAGGenerationRequest) -> RAGGenerationResult:
        evidence_payload = {
            "query": request.query,
            "evidence": [
                item.model_dump(mode="json", exclude_none=True)
                for item in request.evidence
            ],
        }
        completion = self.provider.complete_structured(
            ChatCompletionRequest(
                messages=(
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content=request.system_instructions,
                    ),
                    ChatMessage(
                        role=ChatRole.USER,
                        content=json.dumps(
                            evidence_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    ),
                ),
                temperature=0,
            ),
            GeneratedGroundedAnswer,
            "muse_grounded_answer",
        )
        return RAGGenerationResult(
            output=completion.output,
            metadata=RAGGenerationMetadata(
                provider=self.provider.provider_name,
                model=completion.response.model,
                finish_reason=completion.response.finish_reason,
                usage=completion.response.usage,
            ),
        )
