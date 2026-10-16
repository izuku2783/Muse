import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.session import Base
from app.models import Album, Artist, Genre, Track
from app.repositories.music import AlbumRepository, ArtistRepository, GenreRepository, TrackRepository
from app.services.music.normalization import normalize_text
from app.services.music.schemas import AlbumData, ArtistData, TrackData


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def make_track(provider_id="track-1", title="Night Drive", isrc="USABC1234567"):
    artist = ArtistData(provider_id="artist-1", name="Neon Waves")
    album = AlbumData(provider_id="album-1", title="After Dark", artist_provider_id="artist-1")
    return TrackData(provider_id=provider_id, provider="musicbrainz", title=title, artist=artist, album=album, isrc=isrc, genres=["Synthwave", "Electronic"])


def persist_track(session, data):
    artist = ArtistRepository(session).upsert(data.artist, data.provider)
    album = AlbumRepository(session).upsert(data.album, data.provider, artist)
    genres = [GenreRepository(session).get_or_create(name) for name in data.genres]
    return TrackRepository(session).upsert(data, artist, album, genres)


def test_normalize_text_is_case_and_punctuation_insensitive():
    assert normalize_text("  Street-Spirit (Fade Out) ") == "street spirit fade out"


def test_repositories_persist_models_and_relationships(session):
    track = persist_track(session, make_track())
    session.commit()
    loaded = TrackRepository(session).get_by_provider("musicbrainz", "track-1")
    assert loaded.id == track.id
    assert loaded.artist.name == "Neon Waves"
    assert loaded.album.title == "After Dark"
    assert {genre.name for genre in loaded.genres} == {"synthwave", "electronic"}


def test_repositories_load_canonical_entities_by_database_id(session):
    track = persist_track(session, make_track())

    loaded_track = TrackRepository(session).get_by_id(track.id)
    loaded_artist = ArtistRepository(session).get_by_id(track.artist.id)
    loaded_album = AlbumRepository(session).get_by_id(track.album.id)

    assert TrackRepository.to_data(loaded_track).title == "Night Drive"
    assert ArtistRepository.to_data(loaded_artist).name == "Neon Waves"
    assert AlbumRepository.to_data(loaded_album).artist_provider_id == "artist-1"


def test_provider_identity_is_composite(session):
    session.add_all([
        Artist(provider="musicbrainz", provider_id="same-id", name="One"),
        Artist(provider="another", provider_id="same-id", name="Two"),
    ])
    session.commit()
    assert session.scalar(select(func.count(Artist.id))) == 2


def test_duplicate_provider_identity_is_rejected(session):
    session.add_all([
        Artist(provider="musicbrainz", provider_id="same-id", name="One"),
        Artist(provider="musicbrainz", provider_id="same-id", name="Two"),
    ])
    with pytest.raises(IntegrityError):
        session.commit()


def test_track_deduplicates_by_provider_id_and_updates_metadata(session):
    first = persist_track(session, make_track())
    updated_data = make_track(title="Night Drive Updated")
    second = persist_track(session, updated_data)
    assert second.id == first.id
    assert second.title == "Night Drive Updated"
    assert session.scalar(select(func.count(Track.id))) == 1


def test_track_deduplicates_by_isrc_across_provider_ids(session):
    first = persist_track(session, make_track())
    second = persist_track(session, make_track(provider_id="other-recording"))
    assert second.id == first.id
    assert session.scalar(select(func.count(Track.id))) == 1


def test_track_deduplicates_by_normalized_title_and_artist(session):
    first = persist_track(session, make_track(isrc=None))
    second = persist_track(session, make_track(provider_id="other", title="night-drive", isrc=None))
    assert second.id == first.id


def test_track_does_not_merge_distinct_versions(session):
    persist_track(session, make_track(isrc=None))
    persist_track(session, make_track(provider_id="live", title="Night Drive (Live)", isrc=None))
    assert session.scalar(select(func.count(Track.id))) == 2


def test_track_does_not_merge_same_title_from_different_albums(session):
    persist_track(session, make_track(isrc=None))
    other = make_track(provider_id="other", isrc=None)
    other.album = AlbumData(
        provider_id="album-2",
        title="A Different Release",
        artist_provider_id="artist-1",
    )
    persist_track(session, other)
    assert session.scalar(select(func.count(Track.id))) == 2


def test_genres_are_case_insensitively_deduplicated(session):
    repository = GenreRepository(session)
    assert repository.get_or_create("Synthwave").id == repository.get_or_create(" synthwave ").id
    assert session.scalar(select(func.count(Genre.id))) == 1


def test_database_search_matches_across_title_artist_album_and_genre(session):
    persist_track(session, make_track())
    repository = TrackRepository(session)
    assert repository.search("night drive")[0].title == "Night Drive"
    assert repository.search("neon synthwave")[0].artist.name == "Neon Waves"
    assert repository.search("after dark")[0].album.title == "After Dark"


def test_database_search_paginates(session):
    persist_track(session, make_track(provider_id="one", title="Alpha Night", isrc="ONE"))
    persist_track(session, make_track(provider_id="two", title="Beta Night", isrc="TWO"))
    results = TrackRepository(session).search("night", limit=1, offset=1)
    assert [track.title for track in results] == ["Beta Night"]


def test_database_search_can_return_canonical_models(session):
    persisted = persist_track(session, make_track())

    results = TrackRepository(session).search_models("night drive")

    assert [track.id for track in results] == [persisted.id]
