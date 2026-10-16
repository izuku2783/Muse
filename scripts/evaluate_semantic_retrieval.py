import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.embeddings.factory import get_embedding_provider
from app.indexing.service import IndexingService
from app.retrieval.evaluation import (
    SemanticRetrievalEvaluator,
    load_retrieval_dataset,
)
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.qdrant import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic retrieval on the curated Stage 5 dataset."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/semantic_retrieval.json"),
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    dataset = load_retrieval_dataset(args.dataset)
    provider = get_embedding_provider()
    store = QdrantVectorStore(QdrantClient(":memory:"), "semantic_evaluation")
    try:
        IndexingService(provider, store).index_documents(
            list(dataset.documents), recreate=True
        )
        result = SemanticRetrievalEvaluator(
            SemanticRetriever(provider, store)
        ).evaluate(dataset.cases, k=args.k)
    finally:
        store.close()

    print(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
