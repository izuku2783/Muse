from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.api.routes.memory import get_user_memory_service
from app.main import app
from app.memory.errors import (
    UserPreferenceConflictError,
    UserPreferenceNotFoundError,
)
from app.memory.schemas import (
    PreferenceCategory,
    PreferenceSource,
    PreferenceStance,
    UserPreferenceProfile,
    UserPreferenceRecord,
)


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def preference_record():
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return UserPreferenceRecord(
        id=1,
        user_id=USER_ID,
        category=PreferenceCategory.ARTIST,
        value="Radiohead",
        stance=PreferenceStance.LIKE,
        source=PreferenceSource.EXPLICIT_USER,
        confidence=1.0,
        version=1,
        created_at=now,
        updated_at=now,
    )


class FakeMemoryService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def get_profile(self, user_id, category=None):
        self.calls.append(("get", user_id, category))
        return UserPreferenceProfile(
            user_id=user_id,
            preferences=(preference_record(),),
        )

    def create(self, user_id, request):
        self.calls.append(("create", user_id, request))
        if self.error:
            raise self.error
        return preference_record()

    def update(self, user_id, preference_id, request):
        self.calls.append(("update", user_id, preference_id, request))
        if self.error:
            raise self.error
        return preference_record().model_copy(
            update={"stance": request.stance or PreferenceStance.LIKE}
        )

    def delete(self, user_id, preference_id, expected_version):
        self.calls.append(("delete", user_id, preference_id, expected_version))
        if self.error:
            raise self.error


@pytest.fixture
def override_memory_service():
    def install(service):
        app.dependency_overrides[get_user_memory_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_user_memory_service, None)


def test_memory_api_lists_and_filters_preferences(client, override_memory_service):
    service = override_memory_service(FakeMemoryService())
    response = client.get(f"/users/{USER_ID}/preferences?category=artist")
    assert response.status_code == 200
    assert response.json()["preferences"][0]["source"] == "explicit_user"
    assert service.calls == [("get", USER_ID, PreferenceCategory.ARTIST)]


def test_memory_api_explicitly_creates_updates_and_deletes(
    client, override_memory_service
):
    service = override_memory_service(FakeMemoryService())
    created = client.post(
        f"/users/{USER_ID}/preferences",
        json={"category": "artist", "value": "Radiohead", "stance": "like"},
    )
    updated = client.patch(
        f"/users/{USER_ID}/preferences/1",
        json={"expected_version": 1, "stance": "prefer"},
    )
    deleted = client.delete(
        f"/users/{USER_ID}/preferences/1?expected_version=1"
    )

    assert created.status_code == 201
    assert updated.status_code == 200
    assert updated.json()["stance"] == "prefer"
    assert deleted.status_code == 204
    assert [call[0] for call in service.calls] == ["create", "update", "delete"]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (UserPreferenceNotFoundError("not found"), 404),
        (UserPreferenceConflictError("conflict"), 409),
    ],
)
def test_memory_api_translates_update_errors(
    client, override_memory_service, error, expected_status
):
    override_memory_service(FakeMemoryService(error=error))
    response = client.patch(
        f"/users/{USER_ID}/preferences/1",
        json={"expected_version": 1, "stance": "prefer"},
    )
    assert response.status_code == expected_status


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "artist", "value": "Radiohead", "stance": "like", "source": "explicit_user"},
        {"category": "popularity", "value": "obscure", "stance": "prefer"},
        {"category": "genre", "value": " ", "stance": "like"},
    ],
)
def test_memory_api_rejects_uncontrolled_or_invalid_writes(
    client, override_memory_service, payload
):
    override_memory_service(FakeMemoryService())
    assert client.post(f"/users/{USER_ID}/preferences", json=payload).status_code == 422
