from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DocumentType(StrEnum):
    TRACK = "track"
    ARTIST = "artist"
    ALBUM = "album"
    GENRE = "genre"


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_type: DocumentType
    entity_id: int = Field(gt=0)
    track_id: int | None = Field(default=None, gt=0)
    artist_id: int | None = Field(default=None, gt=0)
    album_id: int | None = Field(default=None, gt=0)
    genre_id: int | None = Field(default=None, gt=0)
    artist_name: str | None = None
    provider: str | None = None
    provider_id: str | None = None
    genres: tuple[str, ...] = ()
    release_year: int | None = Field(default=None, ge=1, le=9999)
    duration_ms: int | None = Field(default=None, ge=0)
    popularity: float | None = Field(default=None, ge=0, le=100)
    instrumental: bool | None = None
    country: str | None = None
    album_type: str | None = None
    isrc: str | None = None
    source_url: str | None = None

    @model_validator(mode="after")
    def validate_entity_identifier(self):
        expected_field = f"{self.document_type.value}_id"
        if getattr(self, expected_field) != self.entity_id:
            raise ValueError(f"{expected_field} must equal entity_id")
        return self


class KnowledgeDocument(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str = Field(min_length=1)
    document_type: DocumentType
    content: str = Field(min_length=1)
    metadata: DocumentMetadata

    @model_validator(mode="after")
    def validate_identity(self):
        expected_id = f"{self.document_type.value}:{self.metadata.entity_id}"
        if self.document_id != expected_id:
            raise ValueError(f"document_id must be {expected_id}")
        if self.metadata.document_type != self.document_type:
            raise ValueError("document type and metadata type must match")
        return self
