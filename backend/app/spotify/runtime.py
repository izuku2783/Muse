from dataclasses import dataclass
from functools import lru_cache

from app.config.settings import get_settings
from app.spotify.crypto import SpotifyTokenCipher
from app.spotify.errors import SpotifyConfigurationError
from app.spotify.oauth_state import SpotifyOAuthStateStore
from app.spotify.provider import SpotifyIntegration


@dataclass
class SpotifyRuntime:
    integration: SpotifyIntegration
    state_store: SpotifyOAuthStateStore
    token_cipher: SpotifyTokenCipher

    def close(self) -> None:
        self.integration.close()


@lru_cache
def get_spotify_runtime() -> SpotifyRuntime:
    settings = get_settings()
    if not settings.spotify_client_id or not settings.spotify_token_encryption_key:
        raise SpotifyConfigurationError(
            "Spotify integration requires SPOTIFY_CLIENT_ID and "
            "SPOTIFY_TOKEN_ENCRYPTION_KEY."
        )
    return SpotifyRuntime(
        integration=SpotifyIntegration(
            client_id=settings.spotify_client_id,
            redirect_uri=settings.spotify_redirect_uri,
            timeout_seconds=settings.spotify_http_timeout_seconds,
        ),
        state_store=SpotifyOAuthStateStore(
            ttl_seconds=settings.spotify_oauth_state_ttl_seconds
        ),
        token_cipher=SpotifyTokenCipher(
            settings.spotify_token_encryption_key.get_secret_value()
        ),
    )


def close_spotify_runtime() -> None:
    if get_spotify_runtime.cache_info().currsize:
        runtime = get_spotify_runtime()
        runtime.close()
        get_spotify_runtime.cache_clear()
