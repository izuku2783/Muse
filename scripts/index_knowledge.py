import argparse

from app.database.init_db import initialize_database
from app.database.session import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.indexing.service import IndexingService
from app.knowledge.corpus import KnowledgeCorpusService
from app.vector_store.factory import get_vector_store


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed and index the deterministic knowledge corpus in Qdrant."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Explicitly replace an incompatible or existing collection.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    initialize_database()
    embedding_provider = get_embedding_provider()
    vector_store = get_vector_store()
    try:
        with SessionLocal() as session:
            summary = IndexingService(
                embedding_provider,
                vector_store,
                batch_size=args.batch_size,
            ).index_corpus(
                KnowledgeCorpusService(session),
                recreate=args.recreate,
            )
    finally:
        vector_store.close()

    print(
        f"Indexed {summary.document_count} documents with "
        f"{summary.embedding_model}:{summary.embedding_version} "
        f"({summary.vector_dimension} dimensions); removed "
        f"{summary.deleted_count} stale documents."
    )


if __name__ == "__main__":
    main()
