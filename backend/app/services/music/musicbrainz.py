import time
from datetime import date
from typing import Any

import httpx

from app.services.music.errors import MusicProviderError, MusicProviderNotFoundError
from app.services.music.provider import MusicProvider
from app.services.music.schemas import AlbumData, ArtistData, TrackData


class MusicBrainzProvider(MusicProvider):
    BASE_URL = "https://musicbrainz.org/ws/2"
    MAX_PAGE_SIZE = 100
    _RELEASE_TYPE_RANK = {"Album": 0, "Single": 1, "EP": 2}
    _NON_CANONICAL_SECONDARY_TYPES = {
        "Compilation", "DJ-mix", "Live", "Mixtape/Street", "Remix"
    }

    def __init__(
        self,
        user_agent: str,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ):
        if not user_agent.strip():
            raise ValueError("A descriptive MusicBrainz User-Agent is required.")
        self.user_agent = user_agent
        self.timeout = timeout
        self.client = client or httpx.Client()
        self._last_request_time = 0.0

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        self._respect_rate_limit()
        try:
            response = self.client.get(
                f"{self.BASE_URL}/{endpoint.lstrip('/')}",
                params=params,
                headers={"User-Agent": self.user_agent, "Accept": "application/json"},
                timeout=self.timeout,
            )
            self._last_request_time = time.monotonic()
            if response.status_code == 404:
                raise MusicProviderNotFoundError("MusicBrainz entity was not found.")
            response.raise_for_status()
            data = response.json()
        except MusicProviderError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise MusicProviderError("MusicBrainz request failed.") from exc
        if not isinstance(data, dict):
            raise MusicProviderError("MusicBrainz returned an invalid response.")
        return data

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
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_artist(recording: dict[str, Any]) -> ArtistData | None:
        credits = recording.get("artist-credit") or []
        artist = credits[0].get("artist") if credits else None
        if not artist or not artist.get("id"):
            return None
        artist_id = artist["id"]
        return ArtistData(
            provider_id=artist_id,
            name=artist.get("name") or "Unknown Artist",
            sort_name=artist.get("sort-name"),
            country=artist.get("country"),
            artist_type=artist.get("type"),
            external_url=f"https://musicbrainz.org/artist/{artist_id}",
        )

    @classmethod
    def _select_release(cls, recording: dict[str, Any]) -> dict[str, Any] | None:
        recording_artist_ids = {
            credit.get("artist", {}).get("id")
            for credit in recording.get("artist-credit") or []
            if credit.get("artist", {}).get("id")
        }
        candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        for release in recording.get("releases") or []:
            if not release.get("id") or release.get("status") != "Official":
                continue
            release_group = release.get("release-group") or {}
            secondary_types = set(release_group.get("secondary-types") or [])
            if secondary_types & cls._NON_CANONICAL_SECONDARY_TYPES:
                continue
            release_artist_ids = {
                credit.get("artist", {}).get("id")
                for credit in release.get("artist-credit") or []
                if credit.get("artist", {}).get("id")
            }
            if (
                recording_artist_ids
                and release_artist_ids
                and recording_artist_ids.isdisjoint(release_artist_ids)
            ):
                continue
            release_date = cls._parse_date(release.get("date"))
            score = (
                cls._RELEASE_TYPE_RANK.get(release_group.get("primary-type"), 3),
                release_date is None,
                release_date or date.max,
                release["id"],
            )
            candidates.append((score, release))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @classmethod
    def _extract_album(cls, recording: dict[str, Any]) -> AlbumData | None:
        release = cls._select_release(recording)
        if not release:
            return None
        credits = release.get("artist-credit") or []
        artist = credits[0].get("artist") if credits else None
        release_id = release["id"]
        return AlbumData(
            provider_id=release_id,
            title=release.get("title") or "Unknown Album",
            release_date=cls._parse_date(release.get("date")),
            album_type=(release.get("release-group") or {}).get("primary-type"),
            artist_provider_id=artist.get("id") if artist else None,
            external_url=f"https://musicbrainz.org/release/{release_id}",
        )

    @classmethod
    def _extract_track(cls, recording: dict[str, Any]) -> TrackData:
        recording_id = recording.get("id")
        if not recording_id:
            raise ValueError("MusicBrainz recording is missing an ID.")
        artist = cls._extract_artist(recording)
        album = cls._extract_album(recording)
        genres = sorted(
            {
                item["name"].strip()
                for item in recording.get("genres") or []
                if item.get("name") and item["name"].strip()
            },
            key=str.casefold,
        )
        isrcs = [value for value in recording.get("isrcs") or [] if value]
        return TrackData(
            provider_id=recording_id,
            title=recording.get("title") or "Unknown Track",
            artist=artist,
            album=album,
            release_date=album.release_date if album else None,
            duration_ms=recording.get("length"),
            isrc=isrcs[0] if isrcs else None,
            genres=genres,
            provider="musicbrainz",
            external_url=f"https://musicbrainz.org/recording/{recording_id}",
        )

    def search_tracks(
        self, query: str, limit: int = 10, offset: int = 0
    ) -> list[TrackData]:
        query = query.strip()
        if not query or limit <= 0:
            return []
        response = self._request(
            "/recording",
            {
                "query": query,
                "limit": min(limit, self.MAX_PAGE_SIZE),
                "offset": max(offset, 0),
                "fmt": "json",
            },
        )
        results = []
        for recording in response.get("recordings") or []:
            try:
                results.append(self._extract_track(recording))
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
        return results

    def get_track(self, track_id: str) -> TrackData | None:
        data = self._request(
            f"recording/{track_id}",
            {"inc": "artist-credits+releases+isrcs+genres+tags", "fmt": "json"},
        )
        return self._extract_track(data)

    def get_artist(self, artist_id: str) -> ArtistData | None:
        data = self._request(
            f"artist/{artist_id}", {"inc": "genres+tags", "fmt": "json"}
        )
        provider_id = data.get("id")
        if not provider_id:
            return None
        life_span = data.get("life-span") or {}
        return ArtistData(
            provider_id=provider_id,
            name=data.get("name") or "Unknown Artist",
            sort_name=data.get("sort-name"),
            country=data.get("country"),
            artist_type=data.get("type"),
            begin_date=self._parse_date(life_span.get("begin")),
            end_date=self._parse_date(life_span.get("end")),
            external_url=f"https://musicbrainz.org/artist/{provider_id}",
        )

    def get_album(self, album_id: str) -> AlbumData | None:
        data = self._request(
            f"release/{album_id}",
            {"inc": "artist-credits+release-groups", "fmt": "json"},
        )
        provider_id = data.get("id")
        if not provider_id:
            return None
        credits = data.get("artist-credit") or []
        artist = credits[0].get("artist") if credits else None
        return AlbumData(
            provider_id=provider_id,
            title=data.get("title") or "Unknown Album",
            release_date=self._parse_date(data.get("date")),
            album_type=(data.get("release-group") or {}).get("primary-type"),
            artist_provider_id=artist.get("id") if artist else None,
            external_url=f"https://musicbrainz.org/release/{provider_id}",
        )

    def get_related_artists(self, artist_id: str) -> list[ArtistData]:
        return []

    def close(self) -> None:
        self.client.close()
