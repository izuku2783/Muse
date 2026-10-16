import json
from datetime import date

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.knowledge.corpus import KnowledgeCorpusService
from app.models import Album, Artist, Genre, Track


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def seed_music(session):
    artist = Artist(
        provider="musicbrainz",
        provider_id="artist-1",
        name="Neon Waves",
    )
    album = Album(
        provider="musicbrainz",
        provider_id="album-1",
        title="After Dark",
        release_date=date(2020, 1, 1),
        album_type="Album",
        artist=artist,
    )
    genre = Genre(name="synthwave")
    track = Track(
        provider="musicbrainz",
        provider_id="track-1",
        title="Night Drive",
        artist=artist,
        album=album,
        genres=[genre],
    )
    session.add(track)
    session.commit()
    return artist, album, track, genre


def test_corpus_builds_one_document_per_entity_in_stable_order(session):
    artist, album, track, genre = seed_music(session)
    documents = KnowledgeCorpusService(session).build_documents()
    assert [document.document_id for document in documents] == [
        f"artist:{artist.id}",
        f"album:{album.id}",
        f"track:{track.id}",
        f"genre:{genre.id}",
    ]


def test_corpus_export_is_valid_deterministic_jsonl(session, tmp_path):
    seed_music(session)
    destination = tmp_path / "nested" / "documents.jsonl"
    service = KnowledgeCorpusService(session)

    first_summary = service.export_jsonl(destination)
    first_bytes = destination.read_bytes()
    second_summary = service.export_jsonl(destination)

    assert first_summary.document_count == 4
    assert second_summary.output_path == destination
    assert destination.read_bytes() == first_bytes
    records = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 4
    assert records[2]["document_type"] == "track"
    assert records[2]["metadata"]["genres"] == ["synthwave"]


def test_empty_database_exports_empty_corpus(session, tmp_path):
    destination = tmp_path / "documents.jsonl"
    summary = KnowledgeCorpusService(session).export_jsonl(destination)
    assert summary.document_count == 0
    assert destination.read_text(encoding="utf-8") == ""
