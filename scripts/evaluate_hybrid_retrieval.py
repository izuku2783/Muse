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
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import BM25LexicalRetriever
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.qdrant import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare semantic and Stage 6 hybrid retrieval."
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
    store = QdrantVectorStore(QdrantClient(":memory:"), "hybrid_evaluation")
    try:
        IndexingService(provider, store).index_documents(
            list(dataset.documents), recreate=True
        )
        semantic = SemanticRetriever(provider, store)
        hybrid = HybridRetriever(
            semantic,
            BM25LexicalRetriever(list(dataset.documents)),
        )
        semantic_result = SemanticRetrievalEvaluator(semantic).evaluate(
            dataset.cases, k=args.k
        )
        hybrid_result = SemanticRetrievalEvaluator(hybrid).evaluate(
            dataset.cases, k=args.k
        )
    finally:
        store.close()

    output = {
        "dataset": str(args.dataset),
        "embedding_model": provider.model_name,
        "embedding_version": provider.version,
        "vector_dimension": provider.dimension,
        "fusion": {"method": "reciprocal_rank_fusion", "rrf_k": 60},
        "semantic": semantic_result.model_dump(mode="json"),
        "hybrid": hybrid_result.model_dump(mode="json"),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
