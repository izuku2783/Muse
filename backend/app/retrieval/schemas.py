from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.schemas import DocumentMetadata, DocumentType


class PersonalizationBreakdown(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relevance_score: float = Field(ge=0, le=1)
    explicit_preference_score: float = Field(ge=-1, le=1)
    provider_affinity_score: float = Field(ge=0, le=1)
    preference_score: float = Field(ge=-1, le=1)
    novelty_score: float = Field(ge=0, le=1)
    diversity_penalty: float = Field(ge=0, le=1)
    final_score: float
    reasons: tuple[str, ...] = ()


class SemanticSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    document_id: str
    entity_type: DocumentType
    entity_id: int = Field(gt=0)
    score: float
    content: str
    metadata: DocumentMetadata


class SemanticSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    results: tuple[SemanticSearchResult, ...]


class LexicalSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    document_id: str
    entity_type: DocumentType
    entity_id: int = Field(gt=0)
    score: float = Field(gt=0)
    content: str
    metadata: DocumentMetadata


class LexicalSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    results: tuple[LexicalSearchResult, ...]


class HybridSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    document_id: str
    entity_type: DocumentType
    entity_id: int = Field(gt=0)
    fusion_score: float = Field(gt=0)
    semantic_rank: int | None = Field(default=None, ge=1)
    semantic_score: float | None = None
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = Field(default=None, gt=0)
    content: str
    metadata: DocumentMetadata


class HybridSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    semantic_candidate_count: int = Field(ge=0)
    lexical_candidate_count: int = Field(ge=0)
    results: tuple[HybridSearchResult, ...]


class RerankedSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: int = Field(ge=1)
    hybrid_rank: int = Field(ge=1)
    document_id: str
    entity_type: DocumentType
    entity_id: int = Field(gt=0)
    reranker_score: float
    fusion_score: float = Field(gt=0)
    semantic_rank: int | None = Field(default=None, ge=1)
    semantic_score: float | None = None
    lexical_rank: int | None = Field(default=None, ge=1)
    lexical_score: float | None = Field(default=None, gt=0)
    content: str
    metadata: DocumentMetadata
    personalization: PersonalizationBreakdown | None = None


class RerankedSearchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    candidate_count: int = Field(ge=0)
    reranker_model: str
    reranker_version: str
    results: tuple[RerankedSearchResult, ...]
    personalization_applied: bool = False
    personalization_sources: tuple[str, ...] = ()
