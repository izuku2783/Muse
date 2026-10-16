from uuid import UUID

from sqlalchemy.orm import Session

from app.personalization.schemas import ProviderListeningSignals
from app.spotify.errors import SpotifyError
from app.spotify.runtime import SpotifyRuntime
from app.spotify.service import SpotifyService


class SpotifyListeningSignalSource:
    """Reads permitted signals on demand; provider failure disables only this source."""

    def __init__(self, runtime: SpotifyRuntime, limit: int = 20):
        if not 1 <= limit <= 50:
            raise ValueError("Spotify personalization signal limit must be 1 to 50.")
        self.runtime = runtime
        self.limit = limit

    def collect(
        self, user_id: UUID, session: Session
    ) -> ProviderListeningSignals | None:
        service = SpotifyService(
            session,
            integration=self.runtime.integration,
            state_store=self.runtime.state_store,
            token_cipher=self.runtime.token_cipher,
        )
        tracks = []
        try:
            top = service.get_top_tracks(
                user_id, limit=self.limit, time_range="medium_term"
            )
            tracks.extend(top.items)
        except SpotifyError:
            pass
        try:
            recent = service.get_recent_playback(user_id, limit=self.limit)
            tracks.extend(item.track for item in recent.items)
        except SpotifyError:
            pass
        if not tracks:
            return None
        return ProviderListeningSignals(
            provider="spotify",
            provider_track_ids=frozenset(track.spotify_id for track in tracks),
            isrcs=frozenset(
                track.external_ids["isrc"]
                for track in tracks
                if track.external_ids.get("isrc")
            ),
            artist_names=frozenset(
                artist.name for track in tracks for artist in track.artists
            ),
        )
