import pytest
from pydantic import ValidationError

from app.query_understanding.schemas import (
    MusicQueryIntent,
    PopularityPreference,
    QueryPreferences,
    QueryUnderstandingFilters,
    UnderstoodMusicQuery,
)


def test_understood_query_contains_validated_retrieval_filters_and_preferences():
    understood = UnderstoodMusicQuery(
        intent=MusicQueryIntent.DISCOVER_MUSIC,
        semantic_query="dark electronic music",
        filters=QueryUnderstandingFilters(
            document_types=(),
            genres=("Electronic",),
            exclude_artists=(),
            year_min=2010,
            year_max=2019,
            duration_ms_min=None,
            duration_ms_max=None,
            popularity_min=None,
            popularity_max=None,
            instrumental=None,
            countries=(),
            album_types=(),
        ),
        preferences=QueryPreferences(popularity=PopularityPreference.LOW),
    )

    assert understood.semantic_query == "dark electronic music"
    assert understood.filters.genres == ("electronic",)
    assert understood.filters.year_min == 2010
    assert understood.filters.to_vector_filter().release_year_max == 2019
    assert understood.preferences.popularity == PopularityPreference.LOW


@pytest.mark.parametrize(
    "payload",
    [
        {
            "intent": "invalid",
            "semantic_query": "music",
            "filters": {},
            "preferences": {"popularity": None},
        },
        {
            "intent": "discover_music",
            "semantic_query": "   ",
            "filters": {},
            "preferences": {"popularity": None},
        },
        {
            "intent": "discover_music",
            "semantic_query": "music",
            "filters": {
                "document_types": [],
                "genres": [],
                "exclude_artists": [],
                "year_min": 2020,
                "year_max": 2010,
                "duration_ms_min": None,
                "duration_ms_max": None,
                "popularity_min": None,
                "popularity_max": None,
                "instrumental": None,
                "countries": [],
                "album_types": [],
            },
            "preferences": {"popularity": None},
        },
        {
            "intent": "discover_music",
            "semantic_query": "music",
            "filters": {},
            "preferences": {"popularity": "viral"},
        },
        {
            "intent": "discover_music",
            "semantic_query": "music",
            "filters": {},
            "preferences": {"popularity": None},
            "unexpected": True,
        },
    ],
)
def test_understood_query_rejects_malformed_output(payload):
    with pytest.raises(ValidationError):
        UnderstoodMusicQuery.model_validate(payload)
