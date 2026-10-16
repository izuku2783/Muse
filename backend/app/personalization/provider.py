from collections.abc import Callable
from typing import Protocol
from uuid import UUID

from sqlalchemy.orm import Session

from app.memory.service import UserMemoryService
from app.personalization.schemas import (
    PersonalizationProfile,
    ProviderListeningSignals,
)
from app.repositories.spotify import MusicProviderMappingRepository


class ListeningSignalSource(Protocol):
    def collect(
        self, user_id: UUID, session: Session
    ) -> ProviderListeningSignals | None: ...


class PersonalizationProfileProvider(Protocol):
    def get_profile(self, user_id: UUID) -> PersonalizationProfile: ...


class DatabasePersonalizationProfileProvider:
    """Builds a request-local profile without persisting provider listening data."""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        listening_sources: tuple[ListeningSignalSource, ...] = (),
    ):
        self.session_factory = session_factory
        self.listening_sources = listening_sources

    def get_profile(self, user_id: UUID) -> PersonalizationProfile:
        with self.session_factory() as session:
            preferences = UserMemoryService(session).get_profile(user_id).preferences
            known_track_ids: set[int] = set()
            known_isrcs: set[str] = set()
            familiar_artists: set[str] = set()
            sources = ["explicit_preferences"] if preferences else []
            mappings = MusicProviderMappingRepository(session)
            for source in self.listening_sources:
                signals = source.collect(user_id, session)
                if signals is None:
                    continue
                sources.append(signals.provider)
                known_track_ids.update(
                    mappings.resolve_entity_ids(
                        "track", signals.provider, signals.provider_track_ids
                    )
                )
                known_isrcs.update(value.casefold() for value in signals.isrcs)
                familiar_artists.update(
                    " ".join(value.casefold().split())
                    for value in signals.artist_names
                    if value.strip()
                )
            return PersonalizationProfile(
                preferences=preferences,
                known_track_ids=frozenset(known_track_ids),
                known_isrcs=frozenset(known_isrcs),
                familiar_artists=frozenset(familiar_artists),
                sources=tuple(dict.fromkeys(sources)),
            )
