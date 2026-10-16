import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from app.database.session import Base
from app.models import Album, Artist, Genre, Track
from app.services.music.errors import MusicProviderError
from app.services.music.ingestion import MusicIngestionService
from app.services.music.provider import MusicProvider
from app.services.music.schemas import AlbumData, ArtistData, TrackData


class StubProvider(MusicProvider):
    def __init__(self, pages=None, failures=0):
        self.pages = pages or {}
        self.failures = failures
        self.calls = []

    def search_tracks(self, query, limit=10, offset=0):
        self.calls.append((query, limit, offset))
        if self.failures:
            self.failures -= 1
            raise MusicProviderError("temporary failure")
        return self.pages.get(offset, [])

    def get_track(self, track_id):
        return None

    def get_artist(self, artist_id):
        return None

    def get_album(self, album_id):
        return None

    def get_related_artists(self, artist_id):
        return []


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def track(provider_id="track-1", genres=None):
    return TrackData(
        provider_id=provider_id,
        provider="musicbrainz",
        title=f"Dark Signal {provider_id}",
        artist=ArtistData(provider_id="artist-1", name="Neon Waves"),
        album=AlbumData(provider_id="album-1", title="Night Music", artist_provider_id="artist-1"),
        genres=genres if genres is not None else ["Synthwave"],
    )


def counts(session):
    return {
        model.__name__: session.scalar(select(func.count(model.id)))
        for model in (Artist, Album, Track, Genre)
    }


def test_ingestion_persists_complete_entity_graph(session):
    service = MusicIngestionService(session, StubProvider())
    service.ingest_tracks([track()])
    assert counts(session) == {"Artist": 1, "Album": 1, "Track": 1, "Genre": 1}
    stored = session.scalar(select(Track))
    assert stored.artist.name == "Neon Waves"
    assert stored.album.title == "Night Music"
    assert [genre.name for genre in stored.genres] == ["synthwave"]


def test_ingestion_is_idempotent(session):
    service = MusicIngestionService(session, StubProvider())
    service.ingest_tracks([track()])
    service.ingest_tracks([track()])
    assert counts(session) == {"Artist": 1, "Album": 1, "Track": 1, "Genre": 1}


def test_partial_metadata_does_not_erase_existing_relationships(session):
    service = MusicIngestionService(session, StubProvider())
    service.ingest_tracks([track()])
    partial = TrackData(provider_id="track-1", provider="musicbrainz", title="Dark Signal track-1")
    service.ingest_tracks([partial])
    stored = session.scalar(select(Track))
    assert stored.artist.name == "Neon Waves"
    assert stored.album.title == "Night Music"
    assert [genre.name for genre in stored.genres] == ["synthwave"]


def test_batch_ingestion_handles_missing_optional_entities(session):
    bare = TrackData(provider_id="bare", provider="musicbrainz", title="Unknown Signal")
    MusicIngestionService(session, StubProvider()).ingest_tracks([bare])
    assert counts(session) == {"Artist": 0, "Album": 0, "Track": 1, "Genre": 0}


def test_paginated_ingestion_stops_on_short_page(session):
    provider = StubProvider(pages={0: [track("one"), track("two")], 2: [track("three")]})
    summary = MusicIngestionService(session, provider).ingest_search("dark synthwave", page_size=2, max_pages=5)
    assert summary.fetched == 3
    assert summary.persisted == 3
    assert summary.pages == 2
    assert provider.calls == [("dark synthwave", 2, 0), ("dark synthwave", 2, 2)]


def test_ingestion_retries_provider_errors(session):
    provider = StubProvider(pages={0: [track()]}, failures=1)
    summary = MusicIngestionService(session, provider, max_retries=1, retry_delay=0).ingest_search("dark", page_size=2)
    assert summary.persisted == 1
    assert len(provider.calls) == 2


def test_ingestion_propagates_provider_error_after_retries(session):
    provider = StubProvider(failures=3)
    with pytest.raises(MusicProviderError):
        MusicIngestionService(session, provider, max_retries=1, retry_delay=0).ingest_search("dark")
    assert counts(session)["Track"] == 0
