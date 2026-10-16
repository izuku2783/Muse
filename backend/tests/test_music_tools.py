import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.repositories.music import AlbumRepository, ArtistRepository, GenreRepository, TrackRepository
from app.services.music.errors import MusicProviderError
from app.services.music.provider import MusicProvider
from app.services.music.schemas import AlbumData, ArtistData, TrackData
from app.tools.errors import ToolEntityNotFoundError, ToolExecutionError, ToolInputError
from app.tools.music import (
    GetAlbumTool,
    GetArtistTool,
    GetTrackTool,
    SearchExternalMusicProviderTool,
    SearchMusicTool,
)


class StubProvider(MusicProvider):
    def __init__(self, results=(), fail=False):
        self.results = list(results)
        self.fail = fail
        self.calls = []

    def search_tracks(self, query, limit=10, offset=0):
        self.calls.append((query, limit, offset))
        if self.fail:
            raise MusicProviderError("provider detail")
        return self.results

    def get_track(self, track_id): return None
    def get_artist(self, artist_id): return None
    def get_album(self, album_id): return None
    def get_related_artists(self, artist_id): return []


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def persist_track(session):
    artist_data = ArtistData(provider_id="artist-1", name="Neon Waves")
    album_data = AlbumData(
        provider_id="album-1",
        title="After Dark",
        artist_provider_id="artist-1",
    )
    track_data = TrackData(
        provider_id="track-1",
        provider="musicbrainz",
        title="Night Drive",
        artist=artist_data,
        album=album_data,
        genres=["synthwave"],
    )
    artist = ArtistRepository(session).upsert(artist_data, "musicbrainz")
    album = AlbumRepository(session).upsert(album_data, "musicbrainz", artist)
    genre = GenreRepository(session).get_or_create("synthwave")
    track = TrackRepository(session).upsert(track_data, artist, album, [genre])
    session.commit()
    return track


def test_search_music_returns_local_canonical_records(session):
    track = persist_track(session)
    tool = SearchMusicTool(TrackRepository(session))

    output = tool.invoke({"query": "  neon   synthwave ", "limit": 5})

    assert output.query == "neon synthwave"
    assert output.results[0].id == track.id
    assert output.results[0].data.title == "Night Drive"


@pytest.mark.parametrize("query", ["", "   "])
def test_search_music_rejects_blank_queries(session, query):
    with pytest.raises(ToolInputError):
        SearchMusicTool(TrackRepository(session)).invoke({"query": query})


def test_entity_lookup_tools_return_typed_canonical_records(session):
    track = persist_track(session)

    track_output = GetTrackTool(TrackRepository(session)).invoke({"entity_id": track.id})
    artist_output = GetArtistTool(ArtistRepository(session)).invoke(
        {"entity_id": track.artist_id}
    )
    album_output = GetAlbumTool(AlbumRepository(session)).invoke(
        {"entity_id": track.album_id}
    )

    assert track_output.track.data.title == "Night Drive"
    assert artist_output.artist.data.name == "Neon Waves"
    assert album_output.album.data.title == "After Dark"


@pytest.mark.parametrize(
    ("tool_factory", "label"),
    [
        (lambda session: GetTrackTool(TrackRepository(session)), "Track"),
        (lambda session: GetArtistTool(ArtistRepository(session)), "Artist"),
        (lambda session: GetAlbumTool(AlbumRepository(session)), "Album"),
    ],
)
def test_entity_lookup_tools_have_deterministic_not_found_errors(
    session, tool_factory, label
):
    with pytest.raises(ToolEntityNotFoundError) as error:
        tool_factory(session).invoke({"entity_id": 999})
    assert error.value.code == "entity_not_found"
    assert str(error.value) == f"{label} 999 was not found."


def test_external_search_does_not_persist_results(session):
    result = TrackData(
        provider_id="external-1", provider="musicbrainz", title="Remote Signal"
    )
    provider = StubProvider([result])

    output = SearchExternalMusicProviderTool(provider).invoke(
        {"query": " remote signal ", "limit": 3, "offset": 2}
    )

    assert output.results == (result,)
    assert provider.calls == [("remote signal", 3, 2)]
    assert TrackRepository(session).search("remote") == []


def test_external_search_normalizes_provider_failures():
    with pytest.raises(ToolExecutionError) as error:
        SearchExternalMusicProviderTool(StubProvider(fail=True)).invoke(
            {"query": "music"}
        )
    assert error.value.code == "tool_execution_error"
    assert "provider detail" not in str(error.value)

