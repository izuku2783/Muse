from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class ArtistData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider_id: str
    name: str
    sort_name: str | None = None
    country: str | None = None
    artist_type: str | None = None
    begin_date: date | None = None
    end_date: date | None = None
    external_url: str | None = None


class AlbumData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider_id: str
    title: str
    release_date: date | None = None
    album_type: str | None = None
    artist_provider_id: str | None = None
    external_url: str | None = None


class TrackData(BaseModel):
    model_config = ConfigDict(extra="ignore")

    provider_id: str
    title: str

    artist: ArtistData | None = None
    album: AlbumData | None = None

    release_date: date | None = None
    duration_ms: int | None = Field(default=None, ge=0)

    isrc: str | None = None

    popularity: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    tempo: float | None = None
    energy: float | None = None
    danceability: float | None = None
    valence: float | None = None
    acousticness: float | None = None
    instrumentalness: float | None = None
    speechiness: float | None = None

    genres: list[str] = Field(default_factory=list)

    external_url: str | None = None

    provider: str = "unknown"