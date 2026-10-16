from app.reranking.provider import Reranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.schemas import RerankedSearchResponse, RerankedSearchResult
from app.vector_store.schemas import VectorSearchFilter


class RerankedRetriever:
    """Rerank the leading hybrid candidates with a cross-encoder."""

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        candidate_limit: int = 20,
    ):
        if not 1 <= candidate_limit <= 100:
            raise ValueError("Reranking candidate limit must be between 1 and 100.")
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        semantic_score_threshold: float | None = None,
    ) -> RerankedSearchResponse:
        if not 1 <= limit <= self.candidate_limit:
            raise ValueError(
                "Reranked result limit must be between 1 and the candidate limit."
            )
        hybrid = self.hybrid_retriever.search(
            query,
            limit=self.candidate_limit,
            filters=filters,
            semantic_score_threshold=semantic_score_threshold,
        )
        scores = self.reranker.score(
            hybrid.query,
            [result.content for result in hybrid.results],
        )
        if len(scores) != len(hybrid.results):
            raise RuntimeError("Reranker returned an unexpected number of scores.")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Reranker returned a non-finite score.")
        ranked = sorted(
            zip(hybrid.results, scores, strict=True),
            key=lambda item: (-item[1], item[0].rank, item[0].document_id),
        )[:limit]
        return RerankedSearchResponse(
            query=hybrid.query,
            candidate_count=len(hybrid.results),
            reranker_model=self.reranker.model_name,
            reranker_version=self.reranker.version,
            results=tuple(
                RerankedSearchResult(
                    rank=rank,
                    hybrid_rank=result.rank,
                    document_id=result.document_id,
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    reranker_score=score,
                    fusion_score=result.fusion_score,
                    semantic_rank=result.semantic_rank,
                    semantic_score=result.semantic_score,
                    lexical_rank=result.lexical_rank,
                    lexical_score=result.lexical_score,
                    content=result.content,
                    metadata=result.metadata,
                )
                for rank, (result, score) in enumerate(ranked, start=1)
            ),
        )
import math
