from datetime import date

from app.knowledge.cleaning import clean_text, natural_join, sorted_unique
from app.knowledge.schemas import DocumentMetadata, DocumentType, KnowledgeDocument
from app.models.music import Album, Artist, Genre, Track


INSTRUMENTALNESS_THRESHOLD = 0.5


def _require_identity(entity, entity_type: str) -> int:
    if entity.id is None:
        raise ValueError(f"{entity_type} must be persisted before document generation")
    return entity.id


def _year(value: date | None) -> int | None:
    return value.year if value else None


def _genre_sentence(genres: list[str]) -> str | None:
    if not genres:
        return None
    noun = "genre" if len(genres) == 1 else "genres"
    return f"It is associated with the {noun} {natural_join(genres)}."


def _duration_sentence(duration_ms: int | None) -> str | None:
    if duration_ms is None:
        return None
    total_seconds = round(duration_ms / 1000)
    minutes, seconds = divmod(total_seconds, 60)
    if minutes and seconds:
        return f"Its duration is {minutes} minutes and {seconds} seconds."
    if minutes:
        unit = "minute" if minutes == 1 else "minutes"
        return f"Its duration is {minutes} {unit}."
    unit = "second" if seconds == 1 else "seconds"
    return f"Its duration is {seconds} {unit}."


class KnowledgeDocumentBuilder:
    """Build deterministic natural-language documents from canonical ORM entities."""

    def build_track(self, track: Track) -> KnowledgeDocument:
        entity_id = _require_identity(track, "track")
        title = clean_text(track.title) or "Unknown track"
        artist_name = clean_text(track.artist.name) if track.artist else None
        album_title = clean_text(track.album.title) if track.album else None
        release_year = _year(track.release_date) or (
            _year(track.album.release_date) if track.album else None
        )
        genres = sorted_unique(genre.name for genre in track.genres)

        opening = f"{title} is a track"
        if artist_name:
            opening += f" by {artist_name}"
        if album_title:
            year_text = f"{release_year} " if release_year else ""
            opening += f" from the {year_text}album {album_title}"
        opening += "."

        sentences = [opening]
        if release_year and not album_title:
            sentences.append(f"It was released in {release_year}.")
        genre_sentence = _genre_sentence(genres)
        if genre_sentence:
            sentences.append(genre_sentence)
        duration_sentence = _duration_sentence(track.duration_ms)
        if duration_sentence:
            sentences.append(duration_sentence)

        return KnowledgeDocument(
            document_id=f"track:{entity_id}",
            document_type=DocumentType.TRACK,
            content=" ".join(sentences),
            metadata=DocumentMetadata(
                document_type=DocumentType.TRACK,
                entity_id=entity_id,
                track_id=entity_id,
                artist_id=track.artist_id,
                album_id=track.album_id,
                artist_name=artist_name,
                provider=clean_text(track.provider),
                provider_id=clean_text(track.provider_id),
                genres=tuple(genres),
                release_year=release_year,
                duration_ms=track.duration_ms,
                popularity=track.popularity,
                instrumental=(
                    track.instrumentalness >= INSTRUMENTALNESS_THRESHOLD
                    if track.instrumentalness is not None
                    else None
                ),
                country=(
                    clean_text(track.artist.country) if track.artist else None
                ),
                album_type=(
                    clean_text(track.album.album_type) if track.album else None
                ),
                isrc=clean_text(track.isrc),
                source_url=clean_text(track.external_url),
            ),
        )

    def build_artist(self, artist: Artist) -> KnowledgeDocument:
        entity_id = _require_identity(artist, "artist")
        name = clean_text(artist.name) or "Unknown artist"
        genres = sorted_unique(
            [genre.name for genre in artist.genres]
            + [genre.name for track in artist.tracks for genre in track.genres]
        )
        sentences = [f"{name} is a musical artist."]
        details = []
        artist_type = clean_text(artist.artist_type)
        country = clean_text(artist.country)
        if artist_type:
            details.append(f"classified as {artist_type}")
        if country:
            details.append(f"associated with {country}")
        if details:
            sentences.append(f"The artist is {natural_join(details)}.")
        if artist.begin_date and artist.end_date:
            sentences.append(
                f"The documented active period is {artist.begin_date.isoformat()} "
                f"to {artist.end_date.isoformat()}."
            )
        elif artist.begin_date:
            sentences.append(
                f"The documented active period begins {artist.begin_date.isoformat()}."
            )
        elif artist.end_date:
            sentences.append(
                f"The documented active period ends {artist.end_date.isoformat()}."
            )
        genre_sentence = _genre_sentence(genres)
        if genre_sentence:
            sentences.append(genre_sentence)

        return KnowledgeDocument(
            document_id=f"artist:{entity_id}",
            document_type=DocumentType.ARTIST,
            content=" ".join(sentences),
            metadata=DocumentMetadata(
                document_type=DocumentType.ARTIST,
                entity_id=entity_id,
                artist_id=entity_id,
                artist_name=name,
                provider=clean_text(artist.provider),
                provider_id=clean_text(artist.provider_id),
                genres=tuple(genres),
                country=country,
                source_url=clean_text(artist.external_url),
            ),
        )

    def build_album(self, album: Album) -> KnowledgeDocument:
        entity_id = _require_identity(album, "album")
        title = clean_text(album.title) or "Unknown album"
        artist_name = clean_text(album.artist.name) if album.artist else None
        release_year = _year(album.release_date)
        release_type = (clean_text(album.album_type) or "music release").casefold()
        genres = sorted_unique(
            genre.name for track in album.tracks for genre in track.genres
        )
        year_text = f"{release_year} " if release_year else ""
        opening = f"{title} is a {year_text}{release_type}"
        if artist_name:
            opening += f" by {artist_name}"
        opening += "."
        sentences = [opening]
        if album.tracks:
            count = len(album.tracks)
            noun = "track" if count == 1 else "tracks"
            sentences.append(f"Muse currently catalogs {count} {noun} on this release.")
        genre_sentence = _genre_sentence(genres)
        if genre_sentence:
            sentences.append(genre_sentence)

        return KnowledgeDocument(
            document_id=f"album:{entity_id}",
            document_type=DocumentType.ALBUM,
            content=" ".join(sentences),
            metadata=DocumentMetadata(
                document_type=DocumentType.ALBUM,
                entity_id=entity_id,
                album_id=entity_id,
                artist_id=album.artist_id,
                artist_name=artist_name,
                provider=clean_text(album.provider),
                provider_id=clean_text(album.provider_id),
                genres=tuple(genres),
                release_year=release_year,
                country=(
                    clean_text(album.artist.country) if album.artist else None
                ),
                album_type=clean_text(album.album_type),
                source_url=clean_text(album.external_url),
            ),
        )

    def build_genre(self, genre: Genre) -> KnowledgeDocument:
        entity_id = _require_identity(genre, "genre")
        name = clean_text(genre.name) or "Unknown genre"
        artists = sorted_unique(
            [artist.name for artist in genre.artists]
            + [track.artist.name for track in genre.tracks if track.artist]
        )
        tracks = sorted_unique(track.title for track in genre.tracks)
        sentences = [f"{name} is a music genre."]
        if artists:
            sentences.append(
                f"Associated artists in Muse include {natural_join(artists[:5])}."
            )
        if tracks:
            sentences.append(
                f"Cataloged tracks in this genre include {natural_join(tracks[:5])}."
            )
        return KnowledgeDocument(
            document_id=f"genre:{entity_id}",
            document_type=DocumentType.GENRE,
            content=" ".join(sentences),
            metadata=DocumentMetadata(
                document_type=DocumentType.GENRE,
                entity_id=entity_id,
                genre_id=entity_id,
                genres=(name,),
            ),
        )
