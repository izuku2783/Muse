from dataclasses import dataclass
from time import sleep

from sqlalchemy.orm import Session

from app.models.music import Track
from app.repositories.music import (
    AlbumRepository,
    ArtistRepository,
    GenreRepository,
    TrackRepository,
)
from app.services.music.errors import MusicProviderError
from app.services.music.provider import MusicProvider
from app.services.music.schemas import TrackData


@dataclass(frozen=True)
class IngestionSummary:
    fetched: int
    persisted: int
    pages: int


class MusicIngestionService:
    def __init__(
        self,
        session: Session,
        provider: MusicProvider,
        max_retries: int = 2,
        retry_delay: float = 0.25,
    ):
        self.session = session
        self.provider = provider
        self.max_retries = max(0, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.artists = ArtistRepository(session)
        self.albums = AlbumRepository(session)
        self.genres = GenreRepository(session)
        self.tracks = TrackRepository(session)

    def ingest_track(self, data: TrackData) -> Track:
        artist = (
            self.artists.upsert(data.artist, data.provider) if data.artist else None
        )
        album_artist = artist
        if data.album and data.album.artist_provider_id:
            album_artist = self.artists.get_by_provider(
                data.provider, data.album.artist_provider_id
            ) or artist
        album = (
            self.albums.upsert(data.album, data.provider, album_artist)
            if data.album
            else None
        )
        genres = [self.genres.get_or_create(name) for name in data.genres if name.strip()]
        return self.tracks.upsert(data, artist, album, genres)

    def ingest_tracks(self, tracks: list[TrackData]) -> list[Track]:
        try:
            persisted = [self.ingest_track(track) for track in tracks]
            self.session.commit()
            return persisted
        except Exception:
            self.session.rollback()
            raise

    def _search_page(self, query: str, limit: int, offset: int) -> list[TrackData]:
        for attempt in range(self.max_retries + 1):
            try:
                return self.provider.search_tracks(query, limit=limit, offset=offset)
            except MusicProviderError:
                if attempt >= self.max_retries:
                    raise
                if self.retry_delay:
                    sleep(self.retry_delay * (2**attempt))
        return []

    def ingest_search(
        self,
        query: str,
        page_size: int = 25,
        max_pages: int = 10,
    ) -> IngestionSummary:
        page_size = min(max(page_size, 1), 100)
        max_pages = max(max_pages, 0)
        fetched = 0
        persisted_ids: set[int] = set()
        pages = 0
        try:
            for page in range(max_pages):
                results = self._search_page(query, page_size, page * page_size)
                pages += 1
                fetched += len(results)
                for result in results:
                    track = self.ingest_track(result)
                    persisted_ids.add(track.id)
                if len(results) < page_size:
                    break
            self.session.commit()
        except Exception:
            self.session.rollback()
            raise
        return IngestionSummary(fetched=fetched, persisted=len(persisted_ids), pages=pages)
