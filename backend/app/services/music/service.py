from sqlalchemy.orm import Session

from app.repositories.music import TrackRepository
from app.services.music.factory import get_music_provider
from app.services.music.ingestion import MusicIngestionService
from app.services.music.provider import MusicProvider
from app.services.music.schemas import TrackData


class MusicService:
    def __init__(self, session: Session, provider: MusicProvider | None = None):
        self.session = session
        self.provider = provider or get_music_provider()
        self.tracks = TrackRepository(session)
        self.ingestion = MusicIngestionService(session, self.provider)

    def search_music(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> list[TrackData]:
        query = query.strip()
        if not query:
            return []
        stored = self.tracks.search(query, limit=limit, offset=offset)
        if stored:
            return stored
        provider_results = self.provider.search_tracks(query, limit=limit, offset=offset)
        persisted = self.ingestion.ingest_tracks(provider_results)
        return [self.tracks.to_data(track) for track in persisted]

    def close(self) -> None:
        self.provider.close()
