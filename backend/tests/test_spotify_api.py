from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.api.routes.spotify import get_spotify_service
from app.main import app
from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyConnectionNotFoundError,
    SpotifyProviderError,
    SpotifyRateLimitError,
)
from app.spotify.schemas import (
    SpotifyAuthorizationStart,
    SpotifyConnectionStatus,
    SpotifyRecentPlayback,
    SpotifyTopTracks,
)


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeSpotifyService:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def start_authorization(self, user_id):
        self.calls.append(("start", user_id))
        return SpotifyAuthorizationStart(
            authorization_url="https://accounts.spotify.com/authorize?state=safe",
            expires_at=datetime(2026, 10, 16, tzinfo=timezone.utc),
        )

    def complete_authorization(self, *, code, state, error=None):
        self.calls.append(("complete", code, state, error))
        if self.error:
            raise self.error
        return self.get_status(USER_ID)

    def get_status(self, user_id):
        self.calls.append(("status", user_id))
        return SpotifyConnectionStatus(user_id=user_id, connected=True)

    def disconnect(self, user_id):
        self.calls.append(("disconnect", user_id))
        if self.error:
            raise self.error

    def get_top_tracks(self, user_id, *, limit, time_range):
        self.calls.append(("top", user_id, limit, time_range))
        if self.error:
            raise self.error
        return SpotifyTopTracks(items=(), time_range=time_range)

    def get_recent_playback(self, user_id, *, limit):
        self.calls.append(("recent", user_id, limit))
        if self.error:
            raise self.error
        return SpotifyRecentPlayback(items=())


@pytest.fixture
def override_spotify_service():
    def install(service):
        app.dependency_overrides[get_spotify_service] = lambda: service
        return service

    yield install
    app.dependency_overrides.pop(get_spotify_service, None)


def test_spotify_api_supports_authorize_callback_status_and_disconnect(
    client, override_spotify_service
):
    service = override_spotify_service(FakeSpotifyService())

    started = client.post(f"/users/{USER_ID}/integrations/spotify/authorize")
    completed = client.get("/integrations/spotify/callback?code=code&state=state")
    status = client.get(f"/users/{USER_ID}/integrations/spotify")
    disconnected = client.delete(f"/users/{USER_ID}/integrations/spotify")

    assert started.status_code == 200
    assert "accounts.spotify.com" in started.json()["authorization_url"]
    assert completed.status_code == 200
    assert status.json()["connected"] is True
    assert disconnected.status_code == 204
    assert [call[0] for call in service.calls] == [
        "start", "complete", "status", "status", "disconnect"
    ]


def test_spotify_api_fetches_bounded_read_only_user_data(client, override_spotify_service):
    service = override_spotify_service(FakeSpotifyService())
    top = client.get(
        f"/users/{USER_ID}/integrations/spotify/top-tracks"
        "?limit=50&time_range=long_term"
    )
    recent = client.get(
        f"/users/{USER_ID}/integrations/spotify/recently-played?limit=5"
    )

    assert top.status_code == 200
    assert recent.status_code == 200
    assert service.calls == [
        ("top", USER_ID, 50, "long_term"),
        ("recent", USER_ID, 5),
    ]


@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (SpotifyConnectionNotFoundError("not found"), 404),
        (SpotifyAuthorizationError("reauthorize"), 401),
        (SpotifyProviderError("provider detail"), 502),
        (SpotifyRateLimitError("limited", retry_after_seconds=8), 429),
    ],
)
def test_spotify_data_api_translates_errors(
    client, override_spotify_service, error, expected_status
):
    override_spotify_service(FakeSpotifyService(error=error))
    response = client.get(f"/users/{USER_ID}/integrations/spotify/top-tracks")
    assert response.status_code == expected_status
    if expected_status == 502:
        assert "provider detail" not in response.text
    if expected_status == 429:
        assert response.headers["Retry-After"] == "8"


def test_spotify_api_validates_provider_bounds_without_calling_service(
    client, override_spotify_service
):
    service = override_spotify_service(FakeSpotifyService())
    assert client.get(f"/users/{USER_ID}/integrations/spotify/top-tracks?limit=51").status_code == 422
    assert client.get(
        f"/users/{USER_ID}/integrations/spotify/top-tracks?time_range=forever"
    ).status_code == 422
    assert service.calls == []


def test_spotify_callback_denial_is_a_safe_bad_request(client, override_spotify_service):
    override_spotify_service(
        FakeSpotifyService(error=SpotifyAuthorizationError("authorization denied"))
    )
    response = client.get(
        "/integrations/spotify/callback?error=access_denied&state=state"
    )
    assert response.status_code == 400


def test_unconfigured_spotify_does_not_affect_core_muse(client):
    app.dependency_overrides.pop(get_spotify_service, None)
    assert client.get("/health").status_code == 200
    response = client.post(f"/users/{USER_ID}/integrations/spotify/authorize")
    assert response.status_code == 503
    assert response.json()["detail"] == "Spotify integration is not configured."
