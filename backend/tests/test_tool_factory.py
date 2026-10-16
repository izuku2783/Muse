from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.services.music.provider import MusicProvider
from app.tools.factory import build_tool_registry


class StubRetriever:
    pass


class StubProvider(MusicProvider):
    def search_tracks(self, query, limit=10, offset=0): return []
    def get_track(self, track_id): return None
    def get_artist(self, artist_id): return None
    def get_album(self, album_id): return None
    def get_related_artists(self, artist_id): return []


def test_factory_builds_allow_list_and_adds_external_tool_only_when_configured():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        local = build_tool_registry(session, StubRetriever())
        configured = build_tool_registry(
            session,
            StubRetriever(),
            StubProvider(),
            reranked_retriever=StubRetriever(),
        )

    assert {item.name for item in local.definitions()} == {
        "search_music",
        "get_track",
        "get_artist",
        "get_album",
        "get_user_preferences",
        "semantic_search",
    }
    assert {item.name for item in configured.definitions()} == {
        *(item.name for item in local.definitions()),
        "search_external_music_provider",
        "retrieve_music_evidence",
    }
