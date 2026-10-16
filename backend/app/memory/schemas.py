from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PreferenceCategory(str, Enum):
    ARTIST = "artist"
    GENRE = "genre"
    POPULARITY = "popularity"
    CONTEXT = "context"
    MOOD = "mood"


class PreferenceStance(str, Enum):
    LIKE = "like"
    DISLIKE = "dislike"
    PREFER = "prefer"
    AVOID = "avoid"
    FREQUENT = "frequent"


class PreferenceSource(str, Enum):
    EXPLICIT_USER = "explicit_user"
    PROVIDER_IMPORT = "provider_import"
    INFERRED_BEHAVIOR = "inferred_behavior"


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserPreferenceCreate(MemoryModel):
    category: PreferenceCategory
    value: str = Field(min_length=1, max_length=255)
    stance: PreferenceStance

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: str) -> str:
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("preference value must not be blank")
        return cleaned

    @model_validator(mode="after")
    def validate_category_value(self):
        if self.category == PreferenceCategory.POPULARITY:
            if self.value.casefold() not in {"low", "moderate", "high"}:
                raise ValueError("popularity preference must be low, moderate, or high")
            if self.stance != PreferenceStance.PREFER:
                raise ValueError("popularity preference must use the prefer stance")
        return self


class UserPreferenceUpdate(MemoryModel):
    expected_version: int = Field(ge=1)
    value: str | None = Field(default=None, min_length=1, max_length=255)
    stance: PreferenceStance | None = None

    @field_validator("value")
    @classmethod
    def clean_value(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = " ".join(value.split())
        if not cleaned:
            raise ValueError("preference value must not be blank")
        return cleaned

    @model_validator(mode="after")
    def require_change(self):
        if self.value is None and self.stance is None:
            raise ValueError("preference update requires value or stance")
        return self


class UserPreferenceRecord(MemoryModel):
    id: int = Field(gt=0)
    user_id: UUID
    category: PreferenceCategory
    value: str
    stance: PreferenceStance
    source: PreferenceSource
    source_detail: str | None = None
    confidence: float = Field(ge=0, le=1)
    version: int = Field(ge=1)
    created_at: datetime
    updated_at: datetime


class UserPreferenceProfile(MemoryModel):
    user_id: UUID
    preferences: tuple[UserPreferenceRecord, ...]
