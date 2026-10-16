from collections.abc import Iterable

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session, selectinload

from app.models.music import Album, Artist, Genre, Track
from app.services.music.normalization import normalize_text
from app.services.music.schemas import AlbumData, ArtistData, TrackData


def _update_non_null(instance, values: dict) -> None:
    for field, value in values.items():
        if value is not None:
            setattr(instance, field, value)


class ArtistRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_provider(self, provider: str, provider_id: str) -> Artist | None:
        return self.session.scalar(
            select(Artist).where(
                Artist.provider == provider, Artist.provider_id == provider_id
            )
        )

    def get_by_id(self, artist_id: int) -> Artist | None:
        return self.session.get(Artist, artist_id)

    def upsert(self, data: ArtistData, provider: str) -> Artist:
        artist = self.get_by_provider(provider, data.provider_id)
        if artist is None:
            artist = Artist(provider=provider, provider_id=data.provider_id, name=data.name)
            self.session.add(artist)
        _update_non_null(
            artist,
            data.model_dump(exclude={"provider_id"}),
        )
        self.session.flush()
        return artist

    @staticmethod
    def to_data(artist: Artist) -> ArtistData:
        return ArtistData(
            provider_id=artist.provider_id,
            name=artist.name,
            sort_name=artist.sort_name,
            country=artist.country,
            artist_type=artist.artist_type,
            begin_date=artist.begin_date,
            end_date=artist.end_date,
            external_url=artist.external_url,
        )


class AlbumRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_by_provider(self, provider: str, provider_id: str) -> Album | None:
        return self.session.scalar(
            select(Album).where(
                Album.provider == provider, Album.provider_id == provider_id
            )
        )

    def get_by_id(self, album_id: int) -> Album | None:
        return self.session.scalar(
            select(Album).options(selectinload(Album.artist)).where(Album.id == album_id)
        )

    def upsert(
        self, data: AlbumData, provider: str, artist: Artist | None = None
    ) -> Album:
        album = self.get_by_provider(provider, data.provider_id)
        if album is None:
            album = Album(provider=provider, provider_id=data.provider_id, title=data.title)
            self.session.add(album)
        _update_non_null(
            album,
            data.model_dump(exclude={"provider_id", "artist_provider_id"}),
        )
        if artist is not None:
            album.artist = artist
        self.session.flush()
        return album

    @staticmethod
    def to_data(album: Album) -> AlbumData:
        return AlbumData(
            provider_id=album.provider_id,
            title=album.title,
            release_date=album.release_date,
            album_type=album.album_type,
            artist_provider_id=album.artist.provider_id if album.artist else None,
            external_url=album.external_url,
        )


class GenreRepository:
    def __init__(self, session: Session):
        self.session = session

    def get_or_create(self, name: str) -> Genre:
        cleaned = " ".join(name.split())
        normalized = normalize_text(cleaned)
        genre = self.session.scalar(
            select(Genre).where(func.lower(Genre.name) == normalized)
        )
        if genre is None:
            genre = Genre(name=normalized)
            self.session.add(genre)
            self.session.flush()
        return genre


class TrackRepository:
    _LOAD_OPTIONS = (
        selectinload(Track.artist),
        selectinload(Track.album),
        selectinload(Track.genres),
    )

    def __init__(self, session: Session):
        self.session = session

    def get_by_provider(self, provider: str, provider_id: str) -> Track | None:
        return self.session.scalar(
            select(Track)
            .options(*self._LOAD_OPTIONS)
            .where(Track.provider == provider, Track.provider_id == provider_id)
        )

    def get_by_id(self, track_id: int) -> Track | None:
        return self.session.scalar(
            select(Track)
            .options(*self._LOAD_OPTIONS)
            .where(Track.id == track_id)
        )

    def find_duplicate(
        self,
        data: TrackData,
        artist: Artist | None,
        album: Album | None,
    ) -> Track | None:
        provider_match = self.get_by_provider(data.provider, data.provider_id)
        if provider_match is not None:
            return provider_match
        if data.isrc:
            isrc_match = self.session.scalar(
                select(Track).options(*self._LOAD_OPTIONS).where(Track.isrc == data.isrc)
            )
            if isrc_match is not None:
                return isrc_match
        if artist is None:
            return None
        candidates = self.session.scalars(
            select(Track)
            .options(*self._LOAD_OPTIONS)
            .where(Track.artist_id == artist.id)
        ).all()
        normalized_title = normalize_text(data.title)
        return next(
            (
                candidate
                for candidate in candidates
                if normalize_text(candidate.title) == normalized_title
                and not (
                    album is not None
                    and candidate.album is not None
                    and normalize_text(candidate.album.title)
                    != normalize_text(album.title)
                )
            ),
            None,
        )

    def upsert(
        self,
        data: TrackData,
        artist: Artist | None,
        album: Album | None,
        genres: Iterable[Genre],
    ) -> Track:
        track = self.find_duplicate(data, artist, album)
        if track is None:
            track = Track(
                provider=data.provider,
                provider_id=data.provider_id,
                title=data.title,
            )
            self.session.add(track)
        _update_non_null(
            track,
            data.model_dump(
                exclude={"provider_id", "provider", "artist", "album", "genres"}
            ),
        )
        if artist is not None:
            track.artist = artist
        if album is not None:
            track.album = album
        incoming_genres = list(dict.fromkeys(genres))
        if incoming_genres or track.id is None:
            track.genres = incoming_genres
        self.session.flush()
        return track

    def search_models(self, query: str, limit: int = 10, offset: int = 0) -> list[Track]:
        terms = [term for term in normalize_text(query).split() if term]
        if not terms or limit <= 0:
            return []
        predicates = []
        for term in terms:
            pattern = f"%{term}%"
            predicates.append(
                or_(
                    func.lower(Track.title).like(pattern),
                    func.lower(Artist.name).like(pattern),
                    func.lower(Album.title).like(pattern),
                    func.lower(Genre.name).like(pattern),
                )
            )
        statement = (
            select(Track)
            .outerjoin(Track.artist)
            .outerjoin(Track.album)
            .outerjoin(Track.genres)
            .options(*self._LOAD_OPTIONS)
            .where(and_(*predicates))
            .order_by(Track.title, Track.id)
            .distinct()
            .offset(max(offset, 0))
            .limit(min(limit, 100))
        )
        return list(self.session.scalars(statement).all())

    def search(self, query: str, limit: int = 10, offset: int = 0) -> list[TrackData]:
        return [
            self.to_data(track)
            for track in self.search_models(query, limit=limit, offset=offset)
        ]

    @staticmethod
    def to_data(track: Track) -> TrackData:
        artist = (
            ArtistData(
                provider_id=track.artist.provider_id,
                name=track.artist.name,
                sort_name=track.artist.sort_name,
                country=track.artist.country,
                artist_type=track.artist.artist_type,
                begin_date=track.artist.begin_date,
                end_date=track.artist.end_date,
                external_url=track.artist.external_url,
            )
            if track.artist
            else None
        )
        album = (
            AlbumData(
                provider_id=track.album.provider_id,
                title=track.album.title,
                release_date=track.album.release_date,
                album_type=track.album.album_type,
                artist_provider_id=(track.album.artist.provider_id if track.album.artist else None),
                external_url=track.album.external_url,
            )
            if track.album
            else None
        )
        return TrackData(
            provider_id=track.provider_id,
            provider=track.provider,
            title=track.title,
            artist=artist,
            album=album,
            release_date=track.release_date,
            duration_ms=track.duration_ms,
            isrc=track.isrc,
            popularity=track.popularity,
            tempo=track.tempo,
            energy=track.energy,
            danceability=track.danceability,
            valence=track.valence,
            acousticness=track.acousticness,
            instrumentalness=track.instrumentalness,
            speechiness=track.speechiness,
            genres=sorted((genre.name for genre in track.genres), key=str.casefold),
            external_url=track.external_url,
        )
