import math

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.knowledge.schemas import DocumentMetadata, DocumentType


class VectorRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    vector: list[float] = Field(min_length=1)
    content: str = Field(min_length=1)
    metadata: DocumentMetadata
    embedding_model: str = Field(min_length=1)
    embedding_version: str = Field(min_length=1)

    @field_validator("vector")
    @classmethod
    def vector_must_be_finite(cls, value: list[float]) -> list[float]:
        if not all(math.isfinite(item) for item in value):
            raise ValueError("vector values must be finite")
        return value

    @model_validator(mode="after")
    def document_identity_must_match_metadata(self):
        expected = f"{self.metadata.document_type.value}:{self.metadata.entity_id}"
        if self.document_id != expected:
            raise ValueError(f"document_id must be {expected}")
        return self


class VectorSearchFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_types: tuple[DocumentType, ...] = ()
    entity_ids: tuple[int, ...] = ()
    genres: tuple[str, ...] = ()
    exclude_artist_ids: tuple[int, ...] = ()
    exclude_artists: tuple[str, ...] = ()
    release_year_min: int | None = Field(default=None, ge=1, le=9999)
    release_year_max: int | None = Field(default=None, ge=1, le=9999)
    duration_ms_min: int | None = Field(default=None, ge=0)
    duration_ms_max: int | None = Field(default=None, ge=0)
    popularity_min: float | None = Field(default=None, ge=0, le=100)
    popularity_max: float | None = Field(default=None, ge=0, le=100)
    instrumental: bool | None = None
    countries: tuple[str, ...] = ()
    album_types: tuple[str, ...] = ()
    provider: str | None = None

    @field_validator(
        "genres",
        "exclude_artists",
        "countries",
        "album_types",
    )
    @classmethod
    def normalize_string_filters(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(" ".join(value.split()).casefold() for value in values)
        )
        if any(not value for value in normalized):
            raise ValueError("filter values must not be blank")
        return normalized

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = " ".join(value.split()).casefold()
        if not normalized:
            raise ValueError("provider must not be blank")
        return normalized

    @field_validator("entity_ids", "exclude_artist_ids")
    @classmethod
    def identifiers_must_be_positive(
        cls, values: tuple[int, ...]
    ) -> tuple[int, ...]:
        if any(value <= 0 for value in values):
            raise ValueError("filter identifiers must be positive")
        return tuple(dict.fromkeys(values))

    @model_validator(mode="after")
    def ranges_must_be_ordered(self):
        ranges = (
            ("release_year", self.release_year_min, self.release_year_max),
            ("duration_ms", self.duration_ms_min, self.duration_ms_max),
            ("popularity", self.popularity_min, self.popularity_max),
        )
        for name, minimum, maximum in ranges:
            if minimum is not None and maximum is not None and minimum > maximum:
                raise ValueError(f"{name}_min cannot exceed {name}_max")
        return self


class VectorSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    document_id: str
    score: float
    content: str
    metadata: DocumentMetadata
    embedding_model: str
    embedding_version: str
