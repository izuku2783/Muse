import argparse

from app.database.init_db import initialize_database
from app.database.session import SessionLocal
from app.services.music.factory import get_music_provider
from app.services.music.ingestion import MusicIngestionService


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest MusicBrainz search results.")
    parser.add_argument("query", help="Music search query")
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--max-pages", type=int, default=1)
    args = parser.parse_args()

    initialize_database()
    provider = get_music_provider()
    try:
        with SessionLocal() as session:
            summary = MusicIngestionService(session, provider).ingest_search(
                args.query,
                page_size=args.page_size,
                max_pages=args.max_pages,
            )
    finally:
        provider.close()
    print(
        f"Fetched {summary.fetched} records across {summary.pages} page(s); "
        f"persisted {summary.persisted} canonical tracks."
    )


if __name__ == "__main__":
    main()
