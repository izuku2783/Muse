import argparse
import json

from app.embeddings.factory import get_embedding_provider
from app.knowledge.schemas import DocumentType
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.factory import get_vector_store
from app.vector_store.schemas import VectorSearchFilter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the indexed Muse knowledge corpus semantically."
    )
    parser.add_argument("query", help="Natural-language semantic query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument(
        "--type",
        action="append",
        choices=[item.value for item in DocumentType],
        dest="document_types",
    )
    parser.add_argument("--genre", action="append", dest="genres")
    parser.add_argument("--entity-id", action="append", type=int, dest="entity_ids")
    parser.add_argument(
        "--exclude-artist-id",
        action="append",
        type=int,
        dest="exclude_artist_ids",
    )
    parser.add_argument(
        "--exclude-artist", action="append", dest="exclude_artists"
    )
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--duration-min-ms", type=int)
    parser.add_argument("--duration-max-ms", type=int)
    parser.add_argument("--popularity-min", type=float)
    parser.add_argument("--popularity-max", type=float)
    parser.add_argument(
        "--instrumental",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--country", action="append", dest="countries")
    parser.add_argument("--album-type", action="append", dest="album_types")
    parser.add_argument("--provider")
    parser.add_argument("--score-threshold", type=float)
    args = parser.parse_args()

    filters = VectorSearchFilter(
        document_types=tuple(
            DocumentType(value) for value in (args.document_types or [])
        ),
        entity_ids=tuple(args.entity_ids or []),
        genres=tuple(args.genres or []),
        exclude_artist_ids=tuple(args.exclude_artist_ids or []),
        exclude_artists=tuple(args.exclude_artists or []),
        release_year_min=args.year_min,
        release_year_max=args.year_max,
        duration_ms_min=args.duration_min_ms,
        duration_ms_max=args.duration_max_ms,
        popularity_min=args.popularity_min,
        popularity_max=args.popularity_max,
        instrumental=args.instrumental,
        countries=tuple(args.countries or []),
        album_types=tuple(args.album_types or []),
        provider=args.provider,
    )
    vector_store = get_vector_store()
    try:
        response = SemanticRetriever(
            get_embedding_provider(), vector_store
        ).search(
            args.query,
            limit=args.limit,
            filters=filters,
            score_threshold=args.score_threshold,
        )
    finally:
        vector_store.close()

    print(json.dumps(response.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
