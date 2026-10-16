from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.agent.runtime import get_agent_service
from app.agent.schemas import AgentResponse, AgentStatus, RetrievalPlan
from app.knowledge.schemas import DocumentType
from app.conversation.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
)
from app.conversation.schemas import ConversationState
from app.llm.errors import LLMConfigurationError, LLMProviderError
from app.main import app
from app.query_understanding.schemas import (
    MusicQueryIntent,
    QueryPreferences,
    QueryUnderstandingFilters,
    QueryUnderstandingResult,
    UnderstoodMusicQuery,
)


def agent_response():
    understood = QueryUnderstandingResult(
        original_query="dark synthwave",
        understood=UnderstoodMusicQuery(
            intent=MusicQueryIntent.DISCOVER_MUSIC,
            semantic_query="dark synthwave",
            filters=QueryUnderstandingFilters(
                document_types=(DocumentType.TRACK,),
                genres=("synthwave",),
                exclude_artists=(),
                year_min=None,
                year_max=None,
                duration_ms_min=None,
                duration_ms_max=None,
                popularity_min=None,
                popularity_max=None,
                instrumental=None,
                countries=(),
                album_types=(),
            ),
            preferences=QueryPreferences(popularity=None),
        ),
        provider="test",
        model="test-model",
    )
    return AgentResponse(
        status=AgentStatus.INSUFFICIENT_EVIDENCE,
        answer="I do not have enough evidence to answer that reliably.",
        retrieved_facts=(),
        recommendation_reasoning=(),
        uncertainty=("No evidence.",),
        understood_query=understood,
        plan=RetrievalPlan(
            semantic_query="dark synthwave",
            filters=understood.understood.filters.to_vector_filter(),
            allow_external_fallback=True,
        ),
        tool_executions=(),
        sources=(),
        conversation=ConversationState(
            conversation_id=UUID("00000000-0000-0000-0000-000000000001"),
            turn_count=1,
            current_intent=MusicQueryIntent.DISCOVER_MUSIC,
            previous_query="dark synthwave",
            previous_semantic_query="dark synthwave",
            previous_filters=understood.understood.filters,
            created_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
            updated_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
        ),
    )


class FakeAgentService:
    def __init__(self, response=None, error=None):
        self.response = response or agent_response()
        self.error = error
        self.calls = []

    def run(self, query, conversation_id=None, user_id=None):
        self.calls.append((query, conversation_id, user_id))
        if self.error:
            raise self.error
        return self.response

    def get_conversation(self, conversation_id):
        if self.error:
            raise self.error
        return self.response.conversation

    def delete_conversation(self, conversation_id):
        if self.error:
            raise self.error


@pytest.fixture
def override_agent_service():
    services = []

    def install(service):
        services.append(service)
        app.dependency_overrides[get_agent_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_agent_service, None)


def test_agent_endpoint_returns_bounded_graph_result(client, override_agent_service):
    service = override_agent_service(FakeAgentService())

    response = client.post("/agent", json={"query": "dark synthwave"})

    assert response.status_code == 200
    assert response.json()["status"] == "insufficient_evidence"
    assert response.json()["plan"]["allow_external_fallback"] is True
    assert response.json()["conversation"]["turn_count"] == 1
    assert service.calls == [("dark synthwave", None, None)]


def test_agent_endpoint_accepts_existing_conversation_id(client, override_agent_service):
    service = override_agent_service(FakeAgentService())
    identifier = "00000000-0000-0000-0000-000000000001"

    response = client.post(
        "/agent",
        json={"query": "Only ones after 2010.", "conversation_id": identifier},
    )

    assert response.status_code == 200
    assert service.calls == [("Only ones after 2010.", UUID(identifier), None)]


def test_agent_endpoint_accepts_optional_user_for_personalized_retrieval(
    client, override_agent_service
):
    service = override_agent_service(FakeAgentService())
    response = client.post(
        "/agent",
        json={"query": "dark synthwave", "user_id": str(UUID(int=2))},
    )
    assert response.status_code == 200
    assert service.calls == [("dark synthwave", None, UUID(int=2))]


@pytest.mark.parametrize("payload", [{"query": ""}, {"query": "   "}, {"query": "music", "extra": True}])
def test_agent_endpoint_validates_requests(client, override_agent_service, payload):
    override_agent_service(FakeAgentService())
    assert client.post("/agent", json=payload).status_code == 422


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (LLMConfigurationError("missing"), 503, "Agent reasoning is not configured."),
        (LLMProviderError("failed"), 502, "Agent reasoning failed."),
    ],
)
def test_agent_endpoint_translates_reasoning_errors(
    client, override_agent_service, error, status, detail
):
    override_agent_service(FakeAgentService(error=error))
    response = client.post("/agent", json={"query": "music"})
    assert response.status_code == status
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ConversationNotFoundError("Conversation was not found or expired."), 404),
        (ConversationConflictError("Conversation changed."), 409),
    ],
)
def test_agent_endpoint_translates_conversation_errors(
    client, override_agent_service, error, status
):
    override_agent_service(FakeAgentService(error=error))
    response = client.post("/agent", json={"query": "music"})
    assert response.status_code == status


def test_conversation_state_can_be_inspected_and_deleted(client, override_agent_service):
    service = override_agent_service(FakeAgentService())
    identifier = "00000000-0000-0000-0000-000000000001"

    loaded = client.get(f"/agent/conversations/{identifier}")
    deleted = client.delete(f"/agent/conversations/{identifier}")

    assert loaded.status_code == 200
    assert loaded.json()["conversation_id"] == identifier
    assert deleted.status_code == 204
    assert deleted.content == b""

