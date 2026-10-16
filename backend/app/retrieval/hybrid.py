from dataclasses import dataclass

from app.knowledge.schemas import DocumentMetadata
from app.retrieval.lexical import BM25LexicalRetriever
from app.retrieval.schemas import HybridSearchResponse, HybridSearchResult
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.schemas import VectorSearchFilter


@dataclass
class _FusedCandidate:
    content: str
    metadata: DocumentMetadata
    fusion_score: float = 0.0
    semantic_rank: int | None = None
    semantic_score: float | None = None
    lexical_rank: int | None = None
    lexical_score: float | None = None


class HybridRetriever:
    """Fuse semantic and BM25 candidate lists with unweighted RRF."""

    def __init__(
        self,
        semantic_retriever: SemanticRetriever,
        lexical_retriever: BM25LexicalRetriever,
        rrf_k: int = 60,
        candidate_limit: int = 50,
    ):
        if rrf_k < 1:
            raise ValueError("RRF k must be positive.")
        if not 1 <= candidate_limit <= 100:
            raise ValueError("Candidate limit must be between 1 and 100.")
        self.semantic_retriever = semantic_retriever
        self.lexical_retriever = lexical_retriever
        self.rrf_k = rrf_k
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        semantic_score_threshold: float | None = None,
    ) -> HybridSearchResponse:
        if not 1 <= limit <= 100:
            raise ValueError("Hybrid search limit must be between 1 and 100.")
        candidate_limit = max(limit, self.candidate_limit)
        semantic = self.semantic_retriever.search(
            query,
            limit=candidate_limit,
            filters=filters,
            score_threshold=semantic_score_threshold,
        )
        lexical = self.lexical_retriever.search(
            query,
            limit=candidate_limit,
            filters=filters,
        )
        candidates: dict[str, _FusedCandidate] = {}
        for result in semantic.results:
            candidate = candidates.setdefault(
                result.document_id,
                _FusedCandidate(content=result.content, metadata=result.metadata),
            )
            candidate.fusion_score += 1 / (self.rrf_k + result.rank)
            candidate.semantic_rank = result.rank
            candidate.semantic_score = result.score
        for result in lexical.results:
            candidate = candidates.setdefault(
                result.document_id,
                _FusedCandidate(content=result.content, metadata=result.metadata),
            )
            candidate.fusion_score += 1 / (self.rrf_k + result.rank)
            candidate.lexical_rank = result.rank
            candidate.lexical_score = result.score

        ranked = sorted(
            candidates.items(),
            key=lambda item: (
                -item[1].fusion_score,
                min(
                    rank
                    for rank in (
                        item[1].semantic_rank,
                        item[1].lexical_rank,
                    )
                    if rank is not None
                ),
                item[0],
            ),
        )[:limit]
        return HybridSearchResponse(
            query=semantic.query,
            semantic_candidate_count=len(semantic.results),
            lexical_candidate_count=len(lexical.results),
            results=tuple(
                HybridSearchResult(
                    rank=rank,
                    document_id=document_id,
                    entity_type=candidate.metadata.document_type,
                    entity_id=candidate.metadata.entity_id,
                    fusion_score=candidate.fusion_score,
                    semantic_rank=candidate.semantic_rank,
                    semantic_score=candidate.semantic_score,
                    lexical_rank=candidate.lexical_rank,
                    lexical_score=candidate.lexical_score,
                    content=candidate.content,
                    metadata=candidate.metadata,
                )
                for rank, (document_id, candidate) in enumerate(ranked, start=1)
            ),
        )
