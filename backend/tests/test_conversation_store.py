from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.conversation.errors import ConversationConflictError
from app.conversation.store import InMemoryConversationStore


class Clock:
    def __init__(self):
        self.now = datetime(2026, 10, 15, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_store_creates_saves_and_deletes_short_term_state():
    clock = Clock()
    store = InMemoryConversationStore(
        clock=clock,
        id_factory=lambda: UUID("00000000-0000-0000-0000-000000000001"),
    )
    created = store.create()
    updated = created.model_copy(update={"turn_count": 1, "previous_query": "music"})
    clock.now += timedelta(seconds=1)

    saved = store.save(updated, expected_turn_count=0)

    assert saved.updated_at == clock.now
    assert store.get(created.conversation_id).previous_query == "music"
    assert store.delete(created.conversation_id) is True
    assert store.get(created.conversation_id) is None


def test_store_expires_inactive_conversations():
    clock = Clock()
    store = InMemoryConversationStore(ttl_seconds=30, clock=clock)
    created = store.create()
    clock.now += timedelta(seconds=30)
    assert store.get(created.conversation_id) is None


def test_store_evicts_least_recently_used_conversation_at_capacity():
    ids = iter(
        (
            UUID("00000000-0000-0000-0000-000000000001"),
            UUID("00000000-0000-0000-0000-000000000002"),
            UUID("00000000-0000-0000-0000-000000000003"),
        )
    )
    store = InMemoryConversationStore(max_conversations=2, id_factory=ids.__next__)
    first = store.create()
    second = store.create()
    store.get(first.conversation_id)
    third = store.create()
    assert store.get(first.conversation_id) is not None
    assert store.get(second.conversation_id) is None
    assert store.get(third.conversation_id) is not None


def test_store_rejects_stale_concurrent_updates():
    store = InMemoryConversationStore()
    created = store.create()
    first_update = created.model_copy(update={"turn_count": 1})
    store.save(first_update, expected_turn_count=0)
    with pytest.raises(ConversationConflictError):
        store.save(first_update, expected_turn_count=0)
