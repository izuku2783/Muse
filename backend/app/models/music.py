from datetime import date

from sqlalchemy import (
    Date,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Column,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


track_genres = Table(
    "track_genres",
    Base.metadata,

    Column(
        "track_id",
        ForeignKey("tracks.id"),
        primary_key=True,
    ),

    Column(
        "genre_id",
        ForeignKey("genres.id"),
        primary_key=True,
    ),
)


artist_genres = Table(
    "artist_genres",
    Base.metadata,

    Column(
        "artist_id",
        ForeignKey("artists.id"),
        primary_key=True,
    ),

    Column(
        "genre_id",
        ForeignKey("genres.id"),
        primary_key=True,
    ),
)


class Artist(Base):
    __tablename__ = "artists"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    provider_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    sort_name: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
    )

    country: Mapped[str | None] = mapped_column(
        String(10),
        nullable=True,
    )

    artist_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    begin_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    end_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    external_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    albums: Mapped[list["Album"]] = relationship(
        back_populates="artist",
    )

    tracks: Mapped[list["Track"]] = relationship(
        back_populates="artist",
    )

    genres: Mapped[list["Genre"]] = relationship(
        secondary=artist_genres,
        back_populates="artists",
    )


class Album(Base):
    __tablename__ = "albums"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    provider_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    title: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    release_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    album_type: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    artist_id: Mapped[int | None] = mapped_column(
        ForeignKey("artists.id"),
        nullable=True,
    )

    external_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    artist: Mapped[Artist | None] = relationship(
        back_populates="albums",
    )

    tracks: Mapped[list["Track"]] = relationship(
        back_populates="album",
    )


class Genre(Base):
    __tablename__ = "genres"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        unique=True,
    )

    artists: Mapped[list[Artist]] = relationship(
        secondary=artist_genres,
        back_populates="genres",
    )

    tracks: Mapped[list["Track"]] = relationship(
        secondary=track_genres,
        back_populates="genres",
    )


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    provider: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    provider_id: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        unique=True,
    )

    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    artist_id: Mapped[int | None] = mapped_column(
        ForeignKey("artists.id"),
        nullable=True,
    )

    album_id: Mapped[int | None] = mapped_column(
        ForeignKey("albums.id"),
        nullable=True,
    )

    release_date: Mapped[date | None] = mapped_column(
        Date,
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    isrc: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    popularity: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    tempo: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    energy: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    danceability: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    valence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    acousticness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    instrumentalness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    speechiness: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    external_url: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    artist: Mapped[Artist | None] = relationship(
        back_populates="tracks",
    )

    album: Mapped[Album | None] = relationship(
        back_populates="tracks",
    )

    genres: Mapped[list[Genre]] = relationship(
        secondary=track_genres,
        back_populates="tracks",
    )