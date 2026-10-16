import json
from typing import Protocol

from app.agent.schemas import AgentEvidence
from app.llm.provider import LLMProvider
from app.llm.schemas import ChatCompletionRequest, ChatMessage, ChatRole
from app.rag.schemas import (
    GeneratedGroundedAnswer,
    RAGGenerationMetadata,
    RAGGenerationResult,
)


AGENT_GENERATION_INSTRUCTIONS = """You are Muse, a grounded music assistant.
Use only the supplied evidence. Never invent music metadata or listening claims.
Keep retrieved facts, recommendation reasoning, and uncertainty separate.
Every factual or recommendation statement must cite one or more supplied evidence IDs.
The answer may summarize cited statements but must introduce no additional claims.
State uncertainty when external provider metadata is sparse or evidence is incomplete."""


class AgentGenerator(Protocol):
    def __call__(
        self, query: str, evidence: tuple[AgentEvidence, ...]
    ) -> RAGGenerationResult: ...


class LLMAgentGenerator:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def __call__(
        self, query: str, evidence: tuple[AgentEvidence, ...]
    ) -> RAGGenerationResult:
        payload = {
            "query": query,
            "evidence": [
                item.model_dump(mode="json", exclude_none=True) for item in evidence
            ],
        }
        completion = self.provider.complete_structured(
            ChatCompletionRequest(
                messages=(
                    ChatMessage(
                        role=ChatRole.SYSTEM,
                        content=AGENT_GENERATION_INSTRUCTIONS,
                    ),
                    ChatMessage(
                        role=ChatRole.USER,
                        content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                    ),
                ),
                temperature=0,
            ),
            GeneratedGroundedAnswer,
            "muse_agent_grounded_answer",
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

