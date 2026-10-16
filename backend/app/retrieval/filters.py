from app.knowledge.schemas import KnowledgeDocument
from app.vector_store.schemas import VectorSearchFilter


def document_matches_filter(
    document: KnowledgeDocument,
    filters: VectorSearchFilter | None,
) -> bool:
    if filters is None:
        return True
    metadata = document.metadata
    if filters.document_types and document.document_type not in filters.document_types:
        return False
    if filters.entity_ids and metadata.entity_id not in filters.entity_ids:
        return False
    if (
        filters.exclude_artist_ids
        and metadata.artist_id in filters.exclude_artist_ids
    ):
        return False
    if filters.exclude_artists and metadata.artist_name:
        if metadata.artist_name.casefold() in filters.exclude_artists:
            return False
    if filters.genres:
        document_genres = {genre.casefold() for genre in metadata.genres}
        requested_genres = {genre.casefold() for genre in filters.genres}
        if document_genres.isdisjoint(requested_genres):
            return False
    if filters.provider and (
        metadata.provider is None or metadata.provider.casefold() != filters.provider
    ):
        return False
    if filters.release_year_min is not None and (
        metadata.release_year is None
        or metadata.release_year < filters.release_year_min
    ):
        return False
    if filters.release_year_max is not None and (
        metadata.release_year is None
        or metadata.release_year > filters.release_year_max
    ):
        return False
    if filters.duration_ms_min is not None and (
        metadata.duration_ms is None
        or metadata.duration_ms < filters.duration_ms_min
    ):
        return False
    if filters.duration_ms_max is not None and (
        metadata.duration_ms is None
        or metadata.duration_ms > filters.duration_ms_max
    ):
        return False
    if filters.popularity_min is not None and (
        metadata.popularity is None
        or metadata.popularity < filters.popularity_min
    ):
        return False
    if filters.popularity_max is not None and (
        metadata.popularity is None
        or metadata.popularity > filters.popularity_max
    ):
        return False
    if filters.instrumental is not None and metadata.instrumental != filters.instrumental:
        return False
    if filters.countries and (
        metadata.country is None
        or metadata.country.casefold() not in filters.countries
    ):
        return False
    if filters.album_types and (
        metadata.album_type is None
        or metadata.album_type.casefold() not in filters.album_types
    ):
        return False
    return True
