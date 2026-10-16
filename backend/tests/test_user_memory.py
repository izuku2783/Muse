from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.memory.errors import (
    UserPreferenceConflictError,
    UserPreferenceNotFoundError,
)
from app.memory.schemas import (
    PreferenceCategory,
    PreferenceSource,
    PreferenceStance,
    UserPreferenceCreate,
    UserPreferenceUpdate,
)
from app.memory.service import UserMemoryService
from app.tools.memory import GetUserPreferencesTool


USER_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_USER_ID = UUID("00000000-0000-0000-0000-000000000002")


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 16, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


@pytest.fixture
def session():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db


def create_request(value="Radiohead", stance=PreferenceStance.LIKE):
    return UserPreferenceCreate(
        category=PreferenceCategory.ARTIST,
        value=value,
        stance=stance,
    )


def test_explicit_preference_records_provenance_confidence_and_timestamps(session):
    clock = Clock()
    service = UserMemoryService(session, clock=clock)

    created = service.create(USER_ID, create_request())

    assert created.user_id == USER_ID
    assert created.source == PreferenceSource.EXPLICIT_USER
    assert created.confidence == 1.0
    assert created.version == 1
    assert created.created_at.replace(tzinfo=timezone.utc) == clock.now
    assert created.updated_at.replace(tzinfo=timezone.utc) == clock.now


def test_profile_is_user_scoped_filterable_and_deterministically_ordered(session):
    service = UserMemoryService(session)
    service.create(
        USER_ID,
        UserPreferenceCreate(
            category=PreferenceCategory.GENRE,
            value="Shoegaze",
            stance=PreferenceStance.LIKE,
        ),
    )
    service.create(USER_ID, create_request())
    service.create(OTHER_USER_ID, create_request("Portishead"))

    profile = service.get_profile(USER_ID)
    artists = service.get_profile(USER_ID, PreferenceCategory.ARTIST)

    assert [(item.category, item.value) for item in profile.preferences] == [
        (PreferenceCategory.ARTIST, "Radiohead"),
        (PreferenceCategory.GENRE, "Shoegaze"),
    ]
    assert [item.value for item in artists.preferences] == ["Radiohead"]


def test_duplicate_identity_is_case_and_whitespace_insensitive(session):
    service = UserMemoryService(session)
    service.create(USER_ID, create_request("Radiohead"))

    with pytest.raises(UserPreferenceConflictError, match="already exists"):
        service.create(USER_ID, create_request("  RADIOHEAD "))

    assert len(service.get_profile(USER_ID).preferences) == 1


def test_update_changes_value_or_stance_and_increments_version(session):
    clock = Clock()
    service = UserMemoryService(session, clock=clock)
    created = service.create(USER_ID, create_request())
    clock.now += timedelta(minutes=1)

    updated = service.update(
        USER_ID,
        created.id,
        UserPreferenceUpdate(
            expected_version=1,
            value="Thom Yorke",
            stance=PreferenceStance.PREFER,
        ),
    )

    assert updated.value == "Thom Yorke"
    assert updated.stance == PreferenceStance.PREFER
    assert updated.version == 2
    assert updated.source == PreferenceSource.EXPLICIT_USER
    assert updated.updated_at.replace(tzinfo=timezone.utc) == clock.now


def test_stale_update_and_delete_are_rejected(session):
    service = UserMemoryService(session)
    created = service.create(USER_ID, create_request())
    updated = service.update(
        USER_ID,
        created.id,
        UserPreferenceUpdate(expected_version=1, stance=PreferenceStance.PREFER),
    )

    with pytest.raises(UserPreferenceConflictError):
        service.update(
            USER_ID,
            created.id,
            UserPreferenceUpdate(expected_version=1, stance=PreferenceStance.DISLIKE),
        )
    with pytest.raises(UserPreferenceConflictError):
        service.delete(USER_ID, created.id, expected_version=1)

    service.delete(USER_ID, created.id, expected_version=updated.version)
    assert service.get_profile(USER_ID).preferences == ()


def test_preferences_cannot_be_read_or_changed_through_another_user(session):
    service = UserMemoryService(session)
    created = service.create(USER_ID, create_request())

    with pytest.raises(UserPreferenceNotFoundError):
        service.update(
            OTHER_USER_ID,
            created.id,
            UserPreferenceUpdate(expected_version=1, stance=PreferenceStance.PREFER),
        )
    with pytest.raises(UserPreferenceNotFoundError):
        service.delete(OTHER_USER_ID, created.id, expected_version=1)


@pytest.mark.parametrize(
    "payload",
    [
        {"category": "popularity", "value": "obscure", "stance": "prefer"},
        {"category": "popularity", "value": "low", "stance": "like"},
        {"category": "artist", "value": "   ", "stance": "like"},
    ],
)
def test_invalid_preference_shapes_are_rejected(payload):
    with pytest.raises(ValidationError):
        UserPreferenceCreate.model_validate(payload)


def test_read_only_preference_tool_returns_typed_user_profile(session):
    service = UserMemoryService(session)
    service.create(USER_ID, create_request())

    profile = GetUserPreferencesTool(service).invoke(
        {"user_id": str(USER_ID), "category": "artist"}
    )

    assert profile.user_id == USER_ID
    assert [item.value for item in profile.preferences] == ["Radiohead"]
