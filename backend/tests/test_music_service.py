from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.session import Base
from app.models import Track
from app.services.music.provider import MusicProvider
from app.services.music.schemas import ArtistData, TrackData
from app.services.music.service import MusicService


class StubProvider(MusicProvider):
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search_tracks(self, query, limit=10, offset=0):
        self.calls.append((query, limit, offset))
        return self.results

    def get_track(self, track_id): return None
    def get_artist(self, artist_id): return None
    def get_album(self, album_id): return None
    def get_related_artists(self, artist_id): return []


def test_search_music_prefers_database_and_falls_back_to_provider():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    provider_track = TrackData(provider_id="one", provider="musicbrainz", title="Dark Signal", artist=ArtistData(provider_id="artist", name="Neon Waves"), genres=["synthwave"])
    provider = StubProvider([provider_track])
    with Session(engine) as session:
        service = MusicService(session, provider)
        first = service.search_music("dark synthwave")
        second = service.search_music("dark synthwave")
        assert first[0].title == "Dark Signal"
        assert second[0].title == "Dark Signal"
        assert provider.calls == [("dark synthwave", 10, 0)]
        assert session.scalar(select(Track)).provider_id == "one"
