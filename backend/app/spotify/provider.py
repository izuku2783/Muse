from collections.abc import Mapping
from typing import Any
from urllib.parse import urlencode

import httpx
from pydantic import ValidationError

from app.spotify.errors import (
    SpotifyAuthorizationError,
    SpotifyProviderError,
    SpotifyRateLimitError,
)
from app.spotify.schemas import (
    SPOTIFY_SCOPES,
    SpotifyPlaybackItem,
    SpotifyProfile,
    SpotifyToken,
    SpotifyTrackSummary,
)


class SpotifyIntegration:
    """Spotify wire adapter. Core Muse services never depend on Spotify payloads."""

    authorization_url = "https://accounts.spotify.com/authorize"
    token_url = "https://accounts.spotify.com/api/token"
    api_base_url = "https://api.spotify.com/v1"

    def __init__(
        self,
        client_id: str,
        redirect_uri: str,
        timeout_seconds: float = 10.0,
        client: httpx.Client | None = None,
    ):
        self.client_id = client_id
        self.redirect_uri = redirect_uri
        self._owns_client = client is None
        self.client = client or httpx.Client(timeout=timeout_seconds)

    def build_authorization_url(self, state: str, code_challenge: str) -> str:
        query = urlencode(
            {
                "client_id": self.client_id,
                "response_type": "code",
                "redirect_uri": self.redirect_uri,
                "state": state,
                "scope": " ".join(sorted(SPOTIFY_SCOPES)),
                "code_challenge_method": "S256",
                "code_challenge": code_challenge,
            }
        )
        return f"{self.authorization_url}?{query}"

    def exchange_code(self, code: str, code_verifier: str) -> SpotifyToken:
        return self._token_request(
            {
                "client_id": self.client_id,
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": code_verifier,
            }
        )

    def refresh_access_token(self, refresh_token: str) -> SpotifyToken:
        return self._token_request(
            {
                "client_id": self.client_id,
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
            }
        )

    def get_profile(self, access_token: str) -> SpotifyProfile:
        try:
            return SpotifyProfile.model_validate(
                self._api_get("/me", access_token=access_token)
            )
        except ValidationError as exc:
            raise SpotifyProviderError("Spotify returned an invalid profile response.") from exc

    def get_top_tracks(
        self,
        access_token: str,
        *,
        limit: int,
        time_range: str,
    ) -> tuple[SpotifyTrackSummary, ...]:
        payload = self._api_get(
            "/me/top/tracks",
            access_token=access_token,
            params={"limit": limit, "time_range": time_range},
        )
        try:
            return tuple(
                SpotifyTrackSummary.model_validate(item)
                for item in payload.get("items", [])
            )
        except (TypeError, ValidationError) as exc:
            raise SpotifyProviderError("Spotify returned invalid top tracks.") from exc

    def get_recently_played(
        self,
        access_token: str,
        *,
        limit: int,
    ) -> tuple[SpotifyPlaybackItem, ...]:
        payload = self._api_get(
            "/me/player/recently-played",
            access_token=access_token,
            params={"limit": limit},
        )
        try:
            return tuple(
                SpotifyPlaybackItem.model_validate(item)
                for item in payload.get("items", [])
            )
        except (TypeError, ValidationError) as exc:
            raise SpotifyProviderError("Spotify returned invalid playback history.") from exc

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def _token_request(self, data: Mapping[str, Any]) -> SpotifyToken:
        try:
            response = self.client.post(
                self.token_url,
                data=data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        except httpx.RequestError as exc:
            raise SpotifyProviderError("Spotify authorization service is unavailable.") from exc
        if response.status_code >= 400:
            error = self._error_code(response)
            if error in {"invalid_grant", "invalid_client"}:
                raise SpotifyAuthorizationError(
                    "Spotify authorization is invalid or has expired."
                )
            raise SpotifyProviderError("Spotify authorization request failed.")
        try:
            return SpotifyToken.model_validate(response.json())
        except (ValueError, TypeError) as exc:
            raise SpotifyProviderError("Spotify returned an invalid token response.") from exc

    def _api_get(
        self,
        endpoint: str,
        *,
        access_token: str,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            response = self.client.get(
                f"{self.api_base_url}/{endpoint.lstrip('/')}",
                params=params,
                headers={"Authorization": f"Bearer {access_token}"},
            )
        except httpx.RequestError as exc:
            raise SpotifyProviderError("Spotify Web API is unavailable.") from exc
        if response.status_code == 401:
            raise SpotifyAuthorizationError("Spotify access token is invalid or expired.")
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            raise SpotifyRateLimitError(
                "Spotify rate limit was exceeded.",
                int(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        if response.status_code >= 400:
            raise SpotifyProviderError("Spotify Web API request failed.")
        try:
            payload = response.json()
        except ValueError as exc:
            raise SpotifyProviderError("Spotify returned an invalid API response.") from exc
        if not isinstance(payload, dict):
            raise SpotifyProviderError("Spotify returned an invalid API response.")
        return payload

    @staticmethod
    def _error_code(response: httpx.Response) -> str | None:
        try:
            payload = response.json()
        except ValueError:
            return None
        return payload.get("error") if isinstance(payload, dict) else None
