from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.llm.schemas import TokenUsage


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    document_id: str
    entity_type: DocumentType
    entity_id: int = Field(gt=0)
    content: str = Field(min_length=1)
    metadata: DocumentMetadata
    retrieval_rank: int = Field(ge=1)
    reranker_score: float


class EvidenceContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...]


class GroundedStatement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str = Field(min_length=1)
    evidence_ids: tuple[str, ...] = Field(min_length=1)


class GeneratedGroundedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer: str = Field(min_length=1)
    retrieved_facts: tuple[GroundedStatement, ...]
    recommendation_reasoning: tuple[GroundedStatement, ...]
    uncertainty: tuple[str, ...]


class RAGGenerationMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class RAGGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    output: GeneratedGroundedAnswer
    metadata: RAGGenerationMetadata


class GroundedResponse(GeneratedGroundedAnswer):
    sources: tuple[EvidenceItem, ...]
    generation: RAGGenerationMetadata | None = None


class RAGGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    system_instructions: str = Field(min_length=1)
    query: str = Field(min_length=1)
    evidence: tuple[EvidenceItem, ...] = Field(min_length=1)
