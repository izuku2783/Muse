from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.memory.schemas import UserPreferenceRecord
from app.query_understanding.schemas import PopularityPreference


class PersonalizationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PersonalizationWeights(PersonalizationModel):
    relevance: float = Field(default=0.5, ge=0, le=1)
    preference: float = Field(default=0.2, ge=0, le=1)
    novelty: float = Field(default=0.15, ge=0, le=1)
    diversity: float = Field(default=0.15, ge=0, le=1)

    @model_validator(mode="after")
    def weights_sum_to_one(self):
        total = self.relevance + self.preference + self.novelty + self.diversity
        if abs(total - 1.0) > 1e-9:
            raise ValueError("Personalization weights must sum to 1.0.")
        return self


class PersonalizationProfile(PersonalizationModel):
    preferences: tuple[UserPreferenceRecord, ...] = ()
    known_track_ids: frozenset[int] = frozenset()
    known_isrcs: frozenset[str] = frozenset()
    familiar_artists: frozenset[str] = frozenset()
    sources: tuple[str, ...] = ()


class ProviderListeningSignals(PersonalizationModel):
    provider: str = Field(min_length=1)
    provider_track_ids: frozenset[str] = frozenset()
    isrcs: frozenset[str] = frozenset()
    artist_names: frozenset[str] = frozenset()


class PersonalizationRequest(PersonalizationModel):
    profile: PersonalizationProfile
    query_popularity: PopularityPreference | None = None
