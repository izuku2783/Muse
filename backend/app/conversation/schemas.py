from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.knowledge.schemas import DocumentType
from app.query_understanding.schemas import (
    MusicQueryIntent,
    QueryContextEntity,
    QueryConversationContext,
    QueryUnderstandingFilters,
)


class ConversationEntitySource(str, Enum):
    KNOWLEDGE = "knowledge"
    EXTERNAL_PROVIDER = "external_provider"


class ConversationEntityReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source: ConversationEntitySource
    label: str = Field(min_length=1, max_length=500)
    position: int = Field(ge=1)
    entity_type: DocumentType | None = None
    entity_id: int | None = Field(default=None, gt=0)
    provider: str | None = None
    provider_id: str | None = None

    @model_validator(mode="after")
    def validate_identity(self):
        if self.source == ConversationEntitySource.KNOWLEDGE:
            if self.entity_type is None or self.entity_id is None:
                raise ValueError("knowledge references require a canonical identity")
        elif not self.provider or not self.provider_id:
            raise ValueError("external references require provider provenance")
        return self

    def to_query_entity(self) -> QueryContextEntity:
        return QueryContextEntity(
            label=self.label,
            position=self.position,
            entity_type=self.entity_type,
            entity_id=self.entity_id,
            provider=self.provider,
            provider_id=self.provider_id,
        )


class ConversationState(BaseModel):
    """Bounded, short-term state; this is not durable user preference memory."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    conversation_id: UUID
    turn_count: int = Field(default=0, ge=0)
    current_intent: MusicQueryIntent | None = None
    previous_query: str | None = None
    previous_semantic_query: str | None = None
    previous_filters: QueryUnderstandingFilters | None = None
    previous_entities: tuple[ConversationEntityReference, ...] = Field(
        default=(), max_length=20
    )
    previous_recommendations: tuple[ConversationEntityReference, ...] = Field(
        default=(), max_length=20
    )
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def timestamps_are_ordered(self):
        if self.updated_at < self.created_at:
            raise ValueError("updated_at cannot precede created_at")
        return self

    def to_query_context(self) -> QueryConversationContext:
        return QueryConversationContext(
            current_intent=self.current_intent,
            previous_query=self.previous_query,
            previous_semantic_query=self.previous_semantic_query,
            previous_filters=self.previous_filters,
            previous_entities=tuple(
                entity.to_query_entity() for entity in self.previous_entities
            ),
            previous_recommendations=tuple(
                entity.to_query_entity() for entity in self.previous_recommendations
            ),
        )
