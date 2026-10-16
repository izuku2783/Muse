from datetime import datetime, timedelta, timezone
from uuid import UUID

import pytest

from app.spotify.errors import SpotifyAuthorizationError
from app.spotify.oauth_state import SpotifyOAuthStateStore


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def test_oauth_state_is_one_use_and_pkce_values_are_well_formed():
    store = SpotifyOAuthStateStore()
    state, verifier, challenge, _ = store.create(USER_ID)

    assert len(verifier) >= 43
    assert "=" not in challenge
    assert store.consume(state).user_id == USER_ID
    with pytest.raises(SpotifyAuthorizationError):
        store.consume(state)


def test_expired_oauth_state_is_rejected():
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    store = SpotifyOAuthStateStore(ttl_seconds=10, clock=lambda: now)
    state, _, _, _ = store.create(USER_ID)
    now += timedelta(seconds=11)

    with pytest.raises(SpotifyAuthorizationError, match="expired"):
        store.consume(state)
