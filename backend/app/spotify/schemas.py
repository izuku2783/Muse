from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


SPOTIFY_SCOPES = frozenset(
    {"user-top-read", "user-read-recently-played"}
)


class SpotifyToken(BaseModel):
    model_config = ConfigDict(extra="ignore")

    access_token: str = Field(min_length=1)
    token_type: str = "Bearer"
    scope: str = ""
    expires_in: int = Field(gt=0)
    refresh_token: str | None = None

    @property
    def scopes(self) -> frozenset[str]:
        return frozenset(self.scope.split())


class SpotifyProfile(BaseModel):
    model_config = ConfigDict(extra="ignore")

    account_id: str = Field(min_length=1)
    id: str | None = None
    display_name: str | None = None
    external_urls: dict[str, str] = Field(default_factory=dict)


class SpotifyArtistSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    spotify_id: str = Field(min_length=1, validation_alias="id")
    name: str = Field(min_length=1)
    external_urls: dict[str, str] = Field(default_factory=dict)


class SpotifyAlbumSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    spotify_id: str = Field(min_length=1, validation_alias="id")
    name: str = Field(min_length=1)
    release_date: str | None = None
    external_urls: dict[str, str] = Field(default_factory=dict)


class SpotifyTrackSummary(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    spotify_id: str = Field(min_length=1, validation_alias="id")
    name: str = Field(min_length=1)
    duration_ms: int | None = Field(default=None, ge=0)
    popularity: int | None = Field(default=None, ge=0, le=100)
    artists: tuple[SpotifyArtistSummary, ...] = ()
    album: SpotifyAlbumSummary | None = None
    external_ids: dict[str, str] = Field(default_factory=dict)
    external_urls: dict[str, str] = Field(default_factory=dict)


class SpotifyPlaybackItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    track: SpotifyTrackSummary
    played_at: datetime


class SpotifyAuthorizationStart(BaseModel):
    authorization_url: HttpUrl
    expires_at: datetime


class SpotifyConnectionStatus(BaseModel):
    user_id: UUID
    connected: bool
    reauthorization_required: bool = False
    spotify_display_name: str | None = None
    granted_scopes: tuple[str, ...] = ()
    connected_at: datetime | None = None


class SpotifyTopTracks(BaseModel):
    items: tuple[SpotifyTrackSummary, ...]
    time_range: Literal["short_term", "medium_term", "long_term"]
    attribution: str = "Spotify"


class SpotifyRecentPlayback(BaseModel):
    items: tuple[SpotifyPlaybackItem, ...]
    attribution: str = "Spotify"

