import pytest
from pydantic import ValidationError

from app.vector_store.schemas import VectorSearchFilter


def test_filter_normalizes_and_deduplicates_structured_values():
    filters = VectorSearchFilter(
        entity_ids=(2, 2),
        exclude_artist_ids=(4, 4),
        genres=(" Synthwave ", "SYNTHWAVE"),
        exclude_artists=(" Neon   Waves ",),
        countries=("US", "us"),
        album_types=(" Album ",),
        provider=" MusicBrainz ",
    )
    assert filters.entity_ids == (2,)
    assert filters.exclude_artist_ids == (4,)
    assert filters.genres == ("synthwave",)
    assert filters.exclude_artists == ("neon waves",)
    assert filters.countries == ("us",)
    assert filters.album_types == ("album",)
    assert filters.provider == "musicbrainz"


@pytest.mark.parametrize(
    "values",
    [
        {"release_year_min": 2020, "release_year_max": 2010},
        {"duration_ms_min": 300_000, "duration_ms_max": 200_000},
        {"popularity_min": 80, "popularity_max": 20},
    ],
)
def test_filter_rejects_inverted_ranges(values):
    with pytest.raises(ValidationError, match="cannot exceed"):
        VectorSearchFilter(**values)


@pytest.mark.parametrize(
    "values",
    [
        {"entity_ids": (0,)},
        {"exclude_artist_ids": (-1,)},
        {"genres": (" ",)},
        {"provider": " "},
        {"popularity_min": -1},
        {"popularity_max": 101},
        {"duration_ms_min": -1},
    ],
)
def test_filter_rejects_invalid_values(values):
    with pytest.raises(ValidationError):
        VectorSearchFilter(**values)
