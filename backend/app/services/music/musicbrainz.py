import time
from datetime import date

import httpx

from app.services.music.provider import MusicProvider
from app.services.music.schemas import (
    AlbumData,
    ArtistData,
    TrackData,
)


class MusicBrainzProvider(MusicProvider):
    BASE_URL = "https://musicbrainz.org/ws/2"

    def __init__(
        self,
        user_agent: str,
        timeout: float = 10.0,
    ):
        self.user_agent = user_agent
        self.timeout = timeout
        self._last_request_time = 0.0

    def _request(
        self,
        endpoint: str,
        params: dict,
    ) -> dict:
        self._respect_rate_limit()

        headers = {
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }

        response = httpx.get(
            f"{self.BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            headers=headers,
            timeout=self.timeout,
        )

        response.raise_for_status()

        self._last_request_time = time.monotonic()

        return response.json()

    def _respect_rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_time

        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

    @staticmethod
    def _parse_date(value: str | None) -> date | None:
        if not value:
            return None

        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None

    @staticmethod
    def _extract_artist(
        recording: dict,
    ) -> ArtistData | None:

        credits = recording.get("artist-credit", [])

        if not credits:
            return None

        artist = credits[0].get("artist")

        if not artist:
            return None

        artist_id = artist.get("id")

        if not artist_id:
            return None

        return ArtistData(
            provider_id=artist_id,
            name=artist.get("name", "Unknown Artist"),
            sort_name=artist.get("sort-name"),
            artist_type=artist.get("type"),
            external_url=(
                f"https://musicbrainz.org/artist/{artist_id}"
            ),
        )

    @staticmethod
    def _extract_album(
        recording: dict,
    ) -> AlbumData | None:

        releases = recording.get("releases", [])

        if not releases:
            return None

        release = releases[0]

        artist = None

        credits = release.get("artist-credit", [])

        if credits:
            artist = credits[0].get("artist")

        release_id = release.get("id")

        if not release_id:
            return None

        return AlbumData(
            provider_id=release_id,
            title=release.get("title", "Unknown Album"),
            release_date=MusicBrainzProvider._parse_date(
                release.get("date")
            ),
            album_type=release.get("status"),
            artist_provider_id=(
                artist.get("id") if artist else None
            ),
            external_url=(
                f"https://musicbrainz.org/release/{release_id}"
            ),
        )

    @classmethod
    def _extract_track(
        cls,
        recording: dict,
    ) -> TrackData:

        recording_id = recording.get("id")

        if not recording_id:
            raise ValueError("MusicBrainz recording is missing an ID.")

        artist = cls._extract_artist(recording)
        album = cls._extract_album(recording)

        genres = [
            item.get("name")
            for item in recording.get("genres", [])
            if item.get("name")
        ]

        isrcs = recording.get("isrcs", [])

        return TrackData(
            provider_id=recording_id,
            title=recording.get("title", "Unknown Track"),
            artist=artist,
            album=album,
            release_date=(
                album.release_date if album else None
            ),
            duration_ms=recording.get("length"),
            isrc=isrcs[0] if isrcs else None,
            genres=genres,
            provider="musicbrainz",
            external_url=(
                f"https://musicbrainz.org/recording/{recording_id}"
            ),
        )

    def search_tracks(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
    ) -> list[TrackData]:

        query = query.strip()

        if not query:
            return []

        response = self._request(
            "/recording",
            params={
                "query": f'artist:"{query}"',
                "limit": min(limit, 100),
                "offset": offset,
                "fmt": "json",
            },
        )

        recordings = response.get("recordings", [])

        results = []

        for recording in recordings:
            try:
                results.append(
                    self._extract_track(recording)
                )
            except ValueError:
                continue

        return results

    def get_track(
        self,
        track_id: str,
    ) -> TrackData | None:

        data = self._request(
            f"recording/{track_id}",
            {
                "inc": (
                    "artist-credits+releases+isrcs+genres+tags"
                ),
                "fmt": "json",
            },
        )

        return self._extract_track(data)

    def get_artist(
        self,
        artist_id: str,
    ) -> ArtistData | None:

        data = self._request(
            f"artist/{artist_id}",
            {
                "inc": "genres+tags",
                "fmt": "json",
            },
        )

        artist_id = data.get("id")

        if not artist_id:
            return None

        return ArtistData(
            provider_id=artist_id,
            name=data.get("name", "Unknown Artist"),
            sort_name=data.get("sort-name"),
            country=data.get("country"),
            artist_type=data.get("type"),
            begin_date=self._parse_date(
                data.get("life-span", {}).get("begin")
            ),
            end_date=self._parse_date(
                data.get("life-span", {}).get("end")
            ),
            external_url=(
                f"https://musicbrainz.org/artist/{artist_id}"
            ),
        )

    def get_album(
        self,
        album_id: str,
    ) -> AlbumData | None:

        data = self._request(
            f"release/{album_id}",
            {
                "inc": "artist-credits+release-groups",
                "fmt": "json",
            },
        )

        album_id = data.get("id")

        if not album_id:
            return None

        credits = data.get("artist-credit", [])

        artist_id = None

        if credits:
            artist = credits[0].get("artist")

            if artist:
                artist_id = artist.get("id")

        release_group = data.get("release-group", {})

        return AlbumData(
            provider_id=album_id,
            title=data.get("title", "Unknown Album"),
            release_date=self._parse_date(
                data.get("date")
            ),
            album_type=release_group.get("primary-type"),
            artist_provider_id=artist_id,
            external_url=(
                f"https://musicbrainz.org/release/{album_id}"
            ),
        )

    def get_related_artists(
        self,
        artist_id: str,
    ) -> list[ArtistData]:

        # MusicBrainz does not provide a Spotify-style
        # "related artists" endpoint.
        #
        # This will be implemented later using relationships,
        # tags, genres, and our own similarity system.

        return []