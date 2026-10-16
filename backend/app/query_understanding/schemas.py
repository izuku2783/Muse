from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.knowledge.schemas import DocumentType
from app.llm.schemas import TokenUsage
from app.vector_store.schemas import VectorSearchFilter


class MusicQueryIntent(str, Enum):
    DISCOVER_MUSIC = "discover_music"
    RESEARCH_MUSIC = "research_music"
    FIND_SIMILAR_MUSIC = "find_similar_music"
    UNKNOWN = "unknown"


class PopularityPreference(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class QueryPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    popularity: PopularityPreference | None


class QueryUnderstandingFilters(BaseModel):
    """Natural-language constraints that can safely become retrieval filters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_types: tuple[DocumentType, ...]
    genres: tuple[str, ...]
    exclude_artists: tuple[str, ...]
    year_min: int | None
    year_max: int | None
    duration_ms_min: int | None
    duration_ms_max: int | None
    popularity_min: float | None
    popularity_max: float | None
    instrumental: bool | None
    countries: tuple[str, ...]
    album_types: tuple[str, ...]

    @field_validator("genres", "exclude_artists", "countries", "album_types")
    @classmethod
    def normalize_string_filters(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(" ".join(value.split()).casefold() for value in values)
        )
        if any(not value for value in normalized):
            raise ValueError("filter values must not be blank")
        return normalized

    @model_validator(mode="after")
    def validate_retrieval_constraints(self):
        self.to_vector_filter()
        return self

    def to_vector_filter(self) -> VectorSearchFilter:
        return VectorSearchFilter(
            document_types=self.document_types,
            genres=self.genres,
            exclude_artists=self.exclude_artists,
            release_year_min=self.year_min,
            release_year_max=self.year_max,
            duration_ms_min=self.duration_ms_min,
            duration_ms_max=self.duration_ms_max,
            popularity_min=self.popularity_min,
            popularity_max=self.popularity_max,
            instrumental=self.instrumental,
            countries=self.countries,
            album_types=self.album_types,
        )


class UnderstoodMusicQuery(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: MusicQueryIntent
    semantic_query: str = Field(min_length=1, pattern=r".*\S.*")
    filters: QueryUnderstandingFilters
    preferences: QueryPreferences


class QueryUnderstandingResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    original_query: str = Field(min_length=1, pattern=r".*\S.*")
    understood: UnderstoodMusicQuery
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    finish_reason: str | None = None
    usage: TokenUsage | None = None
    conversation_context_applied: bool = False


class QueryContextEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = Field(min_length=1)
    position: int | None = Field(default=None, ge=1)
    entity_type: DocumentType | None = None
    entity_id: int | None = Field(default=None, gt=0)
    provider: str | None = None
    provider_id: str | None = None


class QueryConversationContext(BaseModel):
    """Compact previous-turn context supplied only for reference resolution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    current_intent: MusicQueryIntent | None = None
    previous_query: str | None = None
    previous_semantic_query: str | None = None
    previous_filters: QueryUnderstandingFilters | None = None
    previous_entities: tuple[QueryContextEntity, ...] = ()
    previous_recommendations: tuple[QueryContextEntity, ...] = ()
