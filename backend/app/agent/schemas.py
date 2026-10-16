from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.conversation.schemas import ConversationState
from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.query_understanding.schemas import QueryUnderstandingResult
from app.rag.schemas import GeneratedGroundedAnswer, RAGGenerationMetadata
from app.retrieval.schemas import PersonalizationBreakdown
from app.services.music.schemas import TrackData
from app.vector_store.schemas import VectorSearchFilter


class AgentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AgentStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FALLBACK = "fallback"


class EvidenceSource(str, Enum):
    KNOWLEDGE = "knowledge"
    EXTERNAL_PROVIDER = "external_provider"


class ToolExecutionStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AgentLimits(AgentModel):
    max_iterations: int = Field(default=2, ge=1, le=5)
    max_tool_calls: int = Field(default=2, ge=1, le=10)
    timeout_seconds: float = Field(default=60.0, gt=0, le=300)
    max_evidence: int = Field(default=5, ge=1, le=20)
    minimum_evidence: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def evidence_bounds_are_consistent(self):
        if self.minimum_evidence > self.max_evidence:
            raise ValueError("minimum_evidence cannot exceed max_evidence")
        return self


class RetrievalPlan(AgentModel):
    semantic_query: str = Field(min_length=1)
    filters: VectorSearchFilter
    primary_tool: str = "retrieve_music_evidence"
    allow_external_fallback: bool
    external_tool: str = "search_external_music_provider"


class ToolExecutionTrace(AgentModel):
    sequence: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    status: ToolExecutionStatus
    result_count: int = Field(default=0, ge=0)
    error_code: str | None = None
    error_message: str | None = None

    @model_validator(mode="after")
    def failure_requires_error(self):
        if self.status == ToolExecutionStatus.FAILED:
            if not self.error_code or not self.error_message:
                raise ValueError("failed tool traces require an error code and message")
        elif self.error_code is not None or self.error_message is not None:
            raise ValueError("successful tool traces cannot contain an error")
        return self


class AgentEvidence(AgentModel):
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    source: EvidenceSource
    content: str = Field(min_length=1)
    document_id: str | None = None
    entity_type: DocumentType | None = None
    entity_id: int | None = Field(default=None, gt=0)
    metadata: DocumentMetadata | None = None
    rank: int = Field(ge=1)
    score: float | None = None
    provider: str | None = None
    provider_id: str | None = None
    track: TrackData | None = None
    personalization: PersonalizationBreakdown | None = None

    @model_validator(mode="after")
    def validate_source_shape(self):
        if self.source == EvidenceSource.KNOWLEDGE:
            if not all((self.document_id, self.entity_type, self.entity_id, self.metadata)):
                raise ValueError("knowledge evidence requires canonical document metadata")
            if self.track is not None:
                raise ValueError("knowledge evidence cannot contain an external track")
        else:
            if self.track is None or not self.provider or not self.provider_id:
                raise ValueError("external evidence requires provider track provenance")
            if any((self.document_id, self.entity_type, self.entity_id, self.metadata)):
                raise ValueError("external evidence cannot claim a canonical document identity")
        return self


class AgentResponse(GeneratedGroundedAnswer):
    status: AgentStatus
    understood_query: QueryUnderstandingResult
    plan: RetrievalPlan
    tool_executions: tuple[ToolExecutionTrace, ...]
    sources: tuple[AgentEvidence, ...]
    generation: RAGGenerationMetadata | None = None
    conversation: ConversationState | None = None

