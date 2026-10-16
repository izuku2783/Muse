class MusicError(Exception):
    """Base exception for the Stage 2 music data layer."""


class MusicProviderError(MusicError):
    """An external music provider could not complete a request."""


class MusicProviderNotFoundError(MusicProviderError):
    """The requested provider entity does not exist."""
