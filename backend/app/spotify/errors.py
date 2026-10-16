class SpotifyError(Exception):
    """Base error for the optional Spotify integration."""


class SpotifyConfigurationError(SpotifyError):
    pass


class SpotifyAuthorizationError(SpotifyError):
    pass


class SpotifyProviderError(SpotifyError):
    pass


class SpotifyRateLimitError(SpotifyProviderError):
    def __init__(self, message: str, retry_after_seconds: int | None = None):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class SpotifyConnectionNotFoundError(SpotifyError):
    pass


class SpotifyConnectionConflictError(SpotifyError):
    pass

