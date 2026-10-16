import argparse
from pathlib import Path

from app.database.init_db import initialize_database
from app.database.session import SessionLocal
from app.knowledge.corpus import KnowledgeCorpusService


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic knowledge documents from the music database."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/knowledge/documents.jsonl"),
        help="Destination JSONL path",
    )
    args = parser.parse_args()

    initialize_database()
    with SessionLocal() as session:
        summary = KnowledgeCorpusService(session).export_jsonl(args.output)

    print(
        f"Exported {summary.document_count} deterministic documents to "
        f"{summary.output_path}."
    )


if __name__ == "__main__":
    main()
