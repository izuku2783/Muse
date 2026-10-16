from abc import ABC, abstractmethod

from app.services.music.schemas import (
    ArtistData,
    AlbumData,
    TrackData,
)


class MusicProvider(ABC):
    """
    Provider-independent interface for music metadata sources.

    Implementations may use MusicBrainz, Spotify, or another
    external music metadata provider.
    """

    @abstractmethod
    def search_tracks(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[TrackData]:
        """Search for tracks."""
        raise NotImplementedError

    @abstractmethod
    def get_track(self, track_id: str) -> TrackData | None:
        """Retrieve a track by provider-specific ID."""
        raise NotImplementedError

    @abstractmethod
    def get_artist(self, artist_id: str) -> ArtistData | None:
        """Retrieve an artist by provider-specific ID."""
        raise NotImplementedError

    @abstractmethod
    def get_album(self, album_id: str) -> AlbumData | None:
        """Retrieve an album by provider-specific ID."""
        raise NotImplementedError

    @abstractmethod
    def get_related_artists(
        self,
        artist_id: str,
    ) -> list[ArtistData]:
        """Retrieve related artists where supported."""
        raise NotImplementedError

    def close(self) -> None:
        """Release provider resources, when applicable."""
