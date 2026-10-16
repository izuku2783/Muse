from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.memory.schemas import PreferenceCategory
from app.query_understanding.schemas import PopularityPreference
from app.services.music.schemas import AlbumData, ArtistData, TrackData
from app.vector_store.schemas import VectorSearchFilter


class ToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SearchMusicInput(ToolModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    offset: int = Field(default=0, ge=0)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class SearchExternalMusicProviderInput(SearchMusicInput):
    pass


class EntityIdInput(ToolModel):
    entity_id: int = Field(gt=0)


class GetUserPreferencesInput(ToolModel):
    user_id: UUID
    category: PreferenceCategory | None = None


class SemanticSearchInput(ToolModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    filters: VectorSearchFilter | None = None
    score_threshold: float | None = None

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must not be blank")
        return normalized


class RerankedSearchInput(SemanticSearchInput):
    user_id: UUID | None = None
    query_popularity: PopularityPreference | None = None


class TrackRecord(ToolModel):
    id: int = Field(gt=0)
    data: TrackData


class ArtistRecord(ToolModel):
    id: int = Field(gt=0)
    data: ArtistData


class AlbumRecord(ToolModel):
    id: int = Field(gt=0)
    data: AlbumData


class SearchMusicOutput(ToolModel):
    query: str
    results: tuple[TrackRecord, ...]


class SearchExternalMusicProviderOutput(ToolModel):
    query: str
    results: tuple[TrackData, ...]


class GetTrackOutput(ToolModel):
    track: TrackRecord


class GetArtistOutput(ToolModel):
    artist: ArtistRecord


class GetAlbumOutput(ToolModel):
    album: AlbumRecord

