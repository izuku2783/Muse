import json

import pytest

from app.knowledge.schemas import DocumentType
from app.llm.errors import LLMResponseError
from app.llm.schemas import (
    ChatCompletionResponse,
    StructuredCompletion,
    TokenUsage,
)
from app.query_understanding.schemas import (
    MusicQueryIntent,
    PopularityPreference,
    QueryPreferences,
    QueryContextEntity,
    QueryConversationContext,
    QueryUnderstandingFilters,
    UnderstoodMusicQuery,
)
from app.query_understanding.service import QueryUnderstandingService


class FakeProvider:
    provider_name = "fake"
    default_model = "fake-default"

    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def complete_structured(self, request, response_model, schema_name):
        self.calls.append((request, response_model, schema_name))
        if self.error:
            raise self.error
        return StructuredCompletion(
            output=self.output,
            response=ChatCompletionResponse(
                content=self.output.model_dump_json(),
                model="fake-returned",
                finish_reason="stop",
                usage=TokenUsage(
                    prompt_tokens=200,
                    completion_tokens=80,
                    total_tokens=280,
                ),
            ),
        )


def understood_query():
    return UnderstoodMusicQuery(
        intent=MusicQueryIntent.DISCOVER_MUSIC,
        semantic_query="dark electronic music",
        filters=QueryUnderstandingFilters(
            document_types=(),
            genres=(),
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


def test_service_returns_validated_intent_filters_preferences_and_usage():
    provider = FakeProvider(understood_query())
    result = QueryUnderstandingService(provider).understand(
        "  Give me dark electronic music from the 2010s   but nothing too popular. "
    )

    assert result.original_query == (
        "Give me dark electronic music from the 2010s but nothing too popular."
    )
    assert result.understood.intent == MusicQueryIntent.DISCOVER_MUSIC
    assert result.understood.semantic_query == "dark electronic music"
    assert result.understood.filters.year_max == 2019
    assert result.understood.preferences.popularity == PopularityPreference.LOW
    assert result.provider == "fake"
    assert result.model == "fake-returned"
    assert result.usage.total_tokens == 280

    request, response_model, schema_name = provider.calls[0]
    assert response_model is UnderstoodMusicQuery
    assert schema_name == "muse_understood_music_query"
    assert request.messages[1].content == result.original_query
    assert "Do not invent metadata or IDs" in request.messages[0].content


def test_service_rejects_blank_query_before_calling_provider():
    provider = FakeProvider(understood_query())
    with pytest.raises(ValueError, match="must not be empty"):
        QueryUnderstandingService(provider).understand("   ")
    assert provider.calls == []


def test_malformed_provider_output_cannot_be_returned_as_understood_query():
    provider = FakeProvider(error=LLMResponseError("invalid structured output"))
    with pytest.raises(LLMResponseError):
        QueryUnderstandingService(provider).understand("find some jazz")
    assert len(provider.calls) == 1


def test_service_supplies_typed_previous_turn_context_for_follow_up_resolution():
    resolved = understood_query().model_copy(
        update={
            "semantic_query": "albums similar to Radiohead",
            "filters": understood_query().filters.model_copy(
                update={
                    "document_types": (DocumentType.ALBUM,),
                    "year_min": 2010,
                    "year_max": None,
                }
            ),
        }
    )
    provider = FakeProvider(resolved)
    context = QueryConversationContext(
        current_intent=MusicQueryIntent.FIND_SIMILAR_MUSIC,
        previous_query="Give me Radiohead-like albums.",
        previous_semantic_query="albums similar to Radiohead",
        previous_filters=QueryUnderstandingFilters(
            document_types=(DocumentType.ALBUM,),
            genres=(),
            exclude_artists=(),
            year_min=None,
            year_max=None,
            duration_ms_min=None,
            duration_ms_max=None,
            popularity_min=None,
            popularity_max=None,
            instrumental=None,
            countries=(),
            album_types=("album",),
        ),
        previous_recommendations=(
            QueryContextEntity(
                label="Example Album",
                entity_type=DocumentType.ALBUM,
                entity_id=4,
            ),
        ),
    )

    result = QueryUnderstandingService(provider).understand(
        "Only ones after 2010.", conversation_context=context
    )

    payload = json.loads(provider.calls[0][0].messages[1].content)
    assert payload["current_query"] == "Only ones after 2010."
    assert payload["conversation_context"]["previous_filters"]["document_types"] == [
        "album"
    ]
    assert result.understood.semantic_query == "albums similar to Radiohead"
    assert result.understood.filters.year_min == 2010
    assert result.conversation_context_applied is True
