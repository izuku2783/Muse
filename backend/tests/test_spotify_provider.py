from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyProviderError,
    SpotifyRateLimitError,
)
from app.spotify.provider import SpotifyIntegration
from app.spotify.schemas import SPOTIFY_SCOPES


def integration(handler):
    return SpotifyIntegration(
        client_id="client-id",
        redirect_uri="http://127.0.0.1:8000/integrations/spotify/callback",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_authorization_url_uses_pkce_state_exact_redirect_and_minimal_scopes():
    provider = integration(lambda request: httpx.Response(500))
    query = parse_qs(urlparse(provider.build_authorization_url("state", "challenge")).query)

    assert query["response_type"] == ["code"]
    assert query["state"] == ["state"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["code_challenge"] == ["challenge"]
    assert set(query["scope"][0].split()) == SPOTIFY_SCOPES
    assert query["redirect_uri"] == [provider.redirect_uri]


def test_code_exchange_and_refresh_use_pkce_without_client_secret():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "refresh_token": "refresh",
                "token_type": "Bearer",
                "scope": "user-top-read",
                "expires_in": 3600,
            },
        )

    provider = integration(handler)
    provider.exchange_code("code", "verifier")
    provider.refresh_access_token("refresh")

    first = parse_qs(requests[0].content.decode())
    second = parse_qs(requests[1].content.decode())
    assert first["code_verifier"] == ["verifier"]
    assert first["redirect_uri"] == [provider.redirect_uri]
    assert second == {
        "client_id": ["client-id"],
        "grant_type": ["refresh_token"],
        "refresh_token": ["refresh"],
    }
    assert all(b"client_secret" not in request.content for request in requests)


def test_profile_top_tracks_and_recent_playback_are_typed():
    def handler(request):
        if request.url.path == "/v1/me":
            return httpx.Response(200, json={"account_id": "stable", "display_name": "Muse User"})
        track = {
            "id": "spotify-track",
            "name": "Night Drive",
            "duration_ms": 200000,
            "artists": [{"id": "artist", "name": "Neon Waves"}],
            "album": {"id": "album", "name": "After Dark"},
            "external_ids": {"isrc": "USABC1234567"},
        }
        if request.url.path == "/v1/me/top/tracks":
            return httpx.Response(200, json={"items": [track]})
        return httpx.Response(
            200,
            json={"items": [{"track": track, "played_at": "2026-10-16T10:00:00Z"}]},
        )

    provider = integration(handler)
    assert provider.get_profile("token").account_id == "stable"
    top_track = provider.get_top_tracks("token", limit=10, time_range="medium_term")[0]
    assert top_track.spotify_id == "spotify-track"
    assert "id" not in top_track.model_dump()
    assert provider.get_recently_played("token", limit=10)[0].track.external_ids["isrc"] == "USABC1234567"


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (httpx.Response(401, json={"error": "invalid_grant"}), SpotifyAuthorizationError),
        (httpx.Response(500, json={}), SpotifyProviderError),
    ],
)
def test_token_errors_are_normalized(response, error_type):
    provider = integration(lambda request: response)
    with pytest.raises(error_type):
        provider.exchange_code("code", "verifier")


def test_api_rate_limit_preserves_retry_after_without_leaking_body():
    provider = integration(
        lambda request: httpx.Response(
            429,
            headers={"Retry-After": "7"},
            json={"error": {"message": "private provider detail"}},
        )
    )
    with pytest.raises(SpotifyRateLimitError) as raised:
        provider.get_profile("secret-token")
    assert raised.value.retry_after_seconds == 7
    assert "private" not in str(raised.value)


def test_malformed_spotify_content_is_normalized_as_provider_error():
    provider = integration(
        lambda request: httpx.Response(200, json={"items": [{"name": "Missing ID"}]})
    )
    with pytest.raises(SpotifyProviderError, match="invalid top tracks"):
        provider.get_top_tracks("token", limit=10, time_range="medium_term")
