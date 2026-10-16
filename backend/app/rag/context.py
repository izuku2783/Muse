from app.rag.schemas import EvidenceContext, EvidenceItem
from app.retrieval.schemas import RerankedSearchResponse


class ContextBuilder:
    """Build bounded, structured evidence from ranked retrieval results."""

    def __init__(self, max_evidence: int = 5, max_characters: int = 8_000):
        if not 1 <= max_evidence <= 20:
            raise ValueError("Maximum evidence count must be between 1 and 20.")
        if max_characters < 500:
            raise ValueError("Context character budget must be at least 500.")
        self.max_evidence = max_evidence
        self.max_characters = max_characters

    def build(self, response: RerankedSearchResponse) -> EvidenceContext:
        evidence = []
        used_characters = 0
        for result in response.results:
            if len(evidence) >= self.max_evidence:
                break
            if evidence and used_characters + len(result.content) > self.max_characters:
                break
            evidence.append(
                EvidenceItem(
                    evidence_id=f"E{len(evidence) + 1}",
                    document_id=result.document_id,
                    entity_type=result.entity_type,
                    entity_id=result.entity_id,
                    content=result.content,
                    metadata=result.metadata,
                    retrieval_rank=result.rank,
                    reranker_score=result.reranker_score,
                )
            )
            used_characters += len(result.content)
        return EvidenceContext(query=response.query, evidence=tuple(evidence))
