import base64
import hashlib
import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from uuid import UUID

from app.spotify.errors import SpotifyAuthorizationError


@dataclass(frozen=True)
class PendingSpotifyAuthorization:
    user_id: UUID
    code_verifier: str
    expires_at: datetime


class SpotifyOAuthStateStore:
    """One-use, process-local OAuth state and PKCE verifier storage."""

    def __init__(
        self,
        ttl_seconds: float = 600,
        clock: Callable[[], datetime] | None = None,
    ):
        self.ttl_seconds = ttl_seconds
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self._pending: dict[str, PendingSpotifyAuthorization] = {}
        self._lock = threading.Lock()

    def create(self, user_id: UUID) -> tuple[str, str, str, datetime]:
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)[:96]
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        expires_at = self.clock() + timedelta(seconds=self.ttl_seconds)
        with self._lock:
            self._remove_expired()
            self._pending[state] = PendingSpotifyAuthorization(
                user_id=user_id,
                code_verifier=verifier,
                expires_at=expires_at,
            )
        return state, verifier, challenge, expires_at

    def consume(self, state: str) -> PendingSpotifyAuthorization:
        with self._lock:
            pending = self._pending.pop(state, None)
        if pending is None or pending.expires_at <= self.clock():
            raise SpotifyAuthorizationError("Spotify authorization state is invalid or expired.")
        return pending

    def _remove_expired(self) -> None:
        now = self.clock()
        for state in [key for key, item in self._pending.items() if item.expires_at <= now]:
            del self._pending[state]

