from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.knowledge.builders import KnowledgeDocumentBuilder
from app.knowledge.cleaning import clean_text, natural_join, sorted_unique
from app.knowledge.schemas import DocumentMetadata, DocumentType, KnowledgeDocument
from app.models import Album, Artist, Genre, Track


@pytest.fixture
def music_graph():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        artist = Artist(
            provider="musicbrainz",
            provider_id="artist-1",
            name="  Neon\nWaves ",
            artist_type="Group",
            country="US",
            begin_date=date(2010, 1, 1),
        )
        album = Album(
            provider="musicbrainz",
            provider_id="album-1",
            title="After Dark",
            release_date=date(2020, 3, 4),
            album_type="Album",
            artist=artist,
        )
        synthwave = Genre(name="synthwave")
        electronic = Genre(name="Electronic")
        track = Track(
            provider="musicbrainz",
            provider_id="track-1",
            title="Night\tDrive",
            artist=artist,
            album=album,
            duration_ms=245_000,
            popularity=72.5,
            instrumentalness=0.8,
            isrc="USABC1234567",
            external_url="https://musicbrainz.org/recording/track-1",
            genres=[synthwave, electronic],
        )
        session.add(track)
        session.commit()
        yield session, artist, album, track, synthwave


def test_cleaning_helpers_are_deterministic():
    assert clean_text("  Café\n\tMusic\x00 ") == "Café Music"
    assert sorted_unique(["Synthwave", " synthwave ", "Ambient"]) == [
        "Ambient",
        "Synthwave",
    ]
    assert natural_join(["one", "two", "three"]) == "one, two, and three"


def test_build_track_document_has_natural_content_and_separate_metadata(music_graph):
    _, _, _, track, _ = music_graph
    document = KnowledgeDocumentBuilder().build_track(track)
    assert document.content == (
        "Night Drive is a track by Neon Waves from the 2020 album After Dark. "
        "It is associated with the genres Electronic and synthwave. "
        "Its duration is 4 minutes and 5 seconds."
    )
    assert document.metadata.model_dump(mode="json", exclude_none=True) == {
        "document_type": "track",
        "entity_id": track.id,
        "track_id": track.id,
        "artist_id": track.artist_id,
        "album_id": track.album_id,
        "artist_name": "Neon Waves",
        "provider": "musicbrainz",
        "provider_id": "track-1",
        "genres": ["Electronic", "synthwave"],
        "release_year": 2020,
        "duration_ms": 245000,
        "popularity": 72.5,
        "instrumental": True,
        "country": "US",
        "album_type": "Album",
        "isrc": "USABC1234567",
        "source_url": "https://musicbrainz.org/recording/track-1",
    }


def test_build_track_document_handles_missing_fields():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        track = Track(provider="local", provider_id="bare", title="Untitled Signal")
        session.add(track)
        session.commit()
        document = KnowledgeDocumentBuilder().build_track(track)
        assert document.content == "Untitled Signal is a track."
        assert document.metadata.artist_id is None
        assert document.metadata.album_id is None
        assert document.metadata.genres == ()
        assert "None" not in document.content


def test_build_artist_album_and_genre_documents(music_graph):
    _, artist, album, _, genre = music_graph
    builder = KnowledgeDocumentBuilder()
    artist_document = builder.build_artist(artist)
    album_document = builder.build_album(album)
    genre_document = builder.build_genre(genre)
    assert artist_document.content == (
        "Neon Waves is a musical artist. The artist is classified as Group and "
        "associated with US. The documented active period begins 2010-01-01. "
        "It is associated with the genres Electronic and synthwave."
    )
    assert album_document.content == (
        "After Dark is a 2020 album by Neon Waves. "
        "Muse currently catalogs 1 track on this release. "
        "It is associated with the genres Electronic and synthwave."
    )
    assert genre_document.content == (
        "synthwave is a music genre. Associated artists in Muse include Neon Waves. "
        "Cataloged tracks in this genre include Night Drive."
    )


def test_document_output_is_deterministic(music_graph):
    _, _, _, track, _ = music_graph
    builder = KnowledgeDocumentBuilder()
    first = builder.build_track(track).model_dump_json()
    second = builder.build_track(track).model_dump_json()
    assert first == second


def test_instrumental_classification_uses_documented_threshold(music_graph):
    _, _, _, track, _ = music_graph
    track.instrumentalness = 0.49
    assert KnowledgeDocumentBuilder().build_track(track).metadata.instrumental is False
    track.instrumentalness = 0.5
    assert KnowledgeDocumentBuilder().build_track(track).metadata.instrumental is True


def test_document_schema_rejects_mismatched_identity():
    metadata = DocumentMetadata(
        document_type=DocumentType.TRACK,
        entity_id=1,
        track_id=1,
    )
    with pytest.raises(ValidationError, match="document_id must be track:1"):
        KnowledgeDocument(
            document_id="track:2",
            document_type=DocumentType.TRACK,
            content="A valid document.",
            metadata=metadata,
        )


def test_builder_requires_persisted_entities():
    with pytest.raises(ValueError, match="must be persisted"):
        KnowledgeDocumentBuilder().build_genre(Genre(name="ambient"))
