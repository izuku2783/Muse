import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from app.embeddings.factory import get_embedding_provider
from app.indexing.service import IndexingService
from app.reranking.factory import get_reranker
from app.retrieval.evaluation import (
    RetrievalEvaluator,
    load_retrieval_dataset,
)
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import BM25LexicalRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.qdrant import QdrantVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare hybrid retrieval with cross-encoder reranking."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evaluation/semantic_retrieval.json"),
    )
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    dataset = load_retrieval_dataset(args.dataset)
    embedding_provider = get_embedding_provider()
    reranker = get_reranker()
    store = QdrantVectorStore(QdrantClient(":memory:"), "reranking_evaluation")
    try:
        IndexingService(embedding_provider, store).index_documents(
            list(dataset.documents), recreate=True
        )
        hybrid = HybridRetriever(
            SemanticRetriever(embedding_provider, store),
            BM25LexicalRetriever(list(dataset.documents)),
            candidate_limit=100,
        )
        reranked = RerankedRetriever(hybrid, reranker, candidate_limit=20)
        hybrid_result = RetrievalEvaluator(hybrid).evaluate(
            dataset.cases, k=args.k
        )
        reranked_result = RetrievalEvaluator(reranked).evaluate(
            dataset.cases, k=args.k
        )
    finally:
        store.close()

    recall_delta = (
        reranked_result.mean_recall_at_k - hybrid_result.mean_recall_at_k
    )
    mrr_delta = (
        reranked_result.mean_reciprocal_rank
        - hybrid_result.mean_reciprocal_rank
    )
    output = {
        "dataset": str(args.dataset),
        "k": args.k,
        "reranker_model": reranker.model_name,
        "reranker_version": reranker.version,
        "hybrid": hybrid_result.model_dump(mode="json"),
        "reranked": reranked_result.model_dump(mode="json"),
        "comparison": {
            "recall_at_k_delta": recall_delta,
            "mrr_delta": mrr_delta,
            "measurable_improvement": recall_delta > 0 or mrr_delta > 0,
            "adopted_as_default": False,
        },
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
