from datetime import date

import pytest
from pydantic import ValidationError

from app.services.music.schemas import AlbumData, ArtistData, TrackData


def test_track_schema_normalizes_nested_music_data():
    track = TrackData(
        provider_id=" recording-1 ",
        title=" Night Drive ",
        artist=ArtistData(provider_id="artist-1", name="Neon Waves"),
        album=AlbumData(
            provider_id="album-1",
            title="After Dark",
            release_date=date(2020, 1, 2),
        ),
        duration_ms=240_000,
        popularity=72.5,
        genres=["synthwave"],
        provider="musicbrainz",
    )

    assert track.provider_id == "recording-1"
    assert track.title == "Night Drive"
    assert track.album.release_date == date(2020, 1, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [("duration_ms", -1), ("popularity", -0.1), ("popularity", 100.1)],
)
def test_track_schema_rejects_invalid_numeric_values(field, value):
    values = {"provider_id": "recording-1", "title": "Track", field: value}

    with pytest.raises(ValidationError):
        TrackData(**values)


@pytest.mark.parametrize("field", ["provider_id", "title"])
def test_track_schema_rejects_blank_required_text(field):
    values = {"provider_id": "recording-1", "title": "Track", field: "  "}

    with pytest.raises(ValidationError):
        TrackData(**values)


def test_track_schema_uses_independent_genre_lists():
    first = TrackData(provider_id="one", title="One")
    second = TrackData(provider_id="two", title="Two")

    first.genres.append("ambient")

    assert second.genres == []
