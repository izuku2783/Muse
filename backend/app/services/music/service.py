from app.services.music.factory import get_music_provider
from app.services.music.schemas import TrackData


class MusicService:

    def __init__(self):
        self.provider = get_music_provider()

    def search_music(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[TrackData]:

        return self.provider.search_tracks(
            query=query,
            limit=limit,
            offset=offset,
        )