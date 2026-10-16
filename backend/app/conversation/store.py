from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from threading import RLock
from uuid import UUID, uuid4

from app.conversation.errors import (
    ConversationConflictError,
    ConversationNotFoundError,
)
from app.conversation.schemas import ConversationState


class ConversationStore(ABC):
    @abstractmethod
    def create(self) -> ConversationState:
        raise NotImplementedError

    @abstractmethod
    def get(self, conversation_id: UUID) -> ConversationState | None:
        raise NotImplementedError

    @abstractmethod
    def save(
        self, state: ConversationState, expected_turn_count: int
    ) -> ConversationState:
        raise NotImplementedError

    @abstractmethod
    def delete(self, conversation_id: UUID) -> bool:
        raise NotImplementedError


class InMemoryConversationStore(ConversationStore):
    """Thread-safe, bounded short-term storage with sliding expiration."""

    def __init__(
        self,
        ttl_seconds: float = 1_800,
        max_conversations: int = 1_000,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[], UUID] = uuid4,
    ):
        if ttl_seconds <= 0:
            raise ValueError("Conversation TTL must be positive.")
        if max_conversations <= 0:
            raise ValueError("Maximum conversations must be positive.")
        self.ttl = timedelta(seconds=ttl_seconds)
        self.max_conversations = max_conversations
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.id_factory = id_factory
        self._items: OrderedDict[UUID, ConversationState] = OrderedDict()
        self._lock = RLock()

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            key
            for key, state in self._items.items()
            if now - state.updated_at >= self.ttl
        ]
        for key in expired:
            del self._items[key]

    def create(self) -> ConversationState:
        with self._lock:
            now = self.clock()
            self._purge_expired(now)
            while len(self._items) >= self.max_conversations:
                self._items.popitem(last=False)
            state = ConversationState(
                conversation_id=self.id_factory(),
                created_at=now,
                updated_at=now,
            )
            self._items[state.conversation_id] = state
            return state

    def get(self, conversation_id: UUID) -> ConversationState | None:
        with self._lock:
            now = self.clock()
            self._purge_expired(now)
            state = self._items.get(conversation_id)
            if state is not None:
                self._items.move_to_end(conversation_id)
            return state

    def save(
        self, state: ConversationState, expected_turn_count: int
    ) -> ConversationState:
        with self._lock:
            now = self.clock()
            self._purge_expired(now)
            current = self._items.get(state.conversation_id)
            if current is None:
                raise ConversationNotFoundError("Conversation was not found or expired.")
            if current.turn_count != expected_turn_count:
                raise ConversationConflictError(
                    "Conversation changed while the request was running."
                )
            saved = state.model_copy(update={"updated_at": now})
            self._items[state.conversation_id] = saved
            self._items.move_to_end(state.conversation_id)
            return saved

    def delete(self, conversation_id: UUID) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None
