from datetime import date

from pydantic import BaseModel, ConfigDict, Field, field_validator


class MusicData(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    @field_validator("provider_id", "name", "title", check_fields=False)
    @classmethod
    def required_text_must_not_be_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be blank")
        return value


class ArtistData(MusicData):

    provider_id: str
    name: str
    sort_name: str | None = None
    country: str | None = None
    artist_type: str | None = None
    begin_date: date | None = None
    end_date: date | None = None
    external_url: str | None = None


class AlbumData(MusicData):

    provider_id: str
    title: str
    release_date: date | None = None
    album_type: str | None = None
    artist_provider_id: str | None = None
    external_url: str | None = None


class TrackData(MusicData):

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
