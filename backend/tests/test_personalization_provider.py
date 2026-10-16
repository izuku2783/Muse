from datetime import datetime, timezone
from uuid import UUID

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.database.session import Base
from app.memory.schemas import PreferenceCategory, PreferenceStance, UserPreferenceCreate
from app.memory.service import UserMemoryService
from app.models.music import Track
from app.personalization.provider import DatabasePersonalizationProfileProvider
from app.personalization.schemas import ProviderListeningSignals
from app.repositories.spotify import MusicProviderMappingRepository


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


class StubListeningSource:
    def collect(self, user_id, session):
        return ProviderListeningSignals(
            provider="spotify",
            provider_track_ids=frozenset({"spotify-track"}),
            isrcs=frozenset({"USABC1234567"}),
            artist_names=frozenset({"Neon Waves"}),
        )


def test_profile_combines_explicit_memory_provider_signals_and_canonical_mappings():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    with factory() as session:
        track = Track(provider="musicbrainz", provider_id="mb-track", title="Night Drive")
        session.add(track)
        session.flush()
        MusicProviderMappingRepository(session).add(
            "track", track.id, "spotify", "spotify-track"
        )
        UserMemoryService(session).create(
            USER_ID,
            UserPreferenceCreate(
                category=PreferenceCategory.GENRE,
                value="Synthwave",
                stance=PreferenceStance.LIKE,
            ),
        )
        session.commit()
        track_id = track.id

    profile = DatabasePersonalizationProfileProvider(
        factory, listening_sources=(StubListeningSource(),)
    ).get_profile(USER_ID)

    assert profile.preferences[0].value == "Synthwave"
    assert profile.known_track_ids == frozenset({track_id})
    assert profile.known_isrcs == frozenset({"usabc1234567"})
    assert profile.familiar_artists == frozenset({"neon waves"})
    assert profile.sources == ("explicit_preferences", "spotify")


def test_profile_without_memory_or_provider_connection_is_empty():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    profile = DatabasePersonalizationProfileProvider(factory).get_profile(USER_ID)
    assert profile.preferences == ()
    assert profile.sources == ()
