import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from app.knowledge.schemas import KnowledgeDocument
from app.retrieval.semantic import SemanticRetriever


class RetrievalEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str = Field(min_length=1)
    relevant_document_ids: tuple[str, ...] = Field(min_length=1)


class RetrievalEvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    documents: tuple[KnowledgeDocument, ...]
    cases: tuple[RetrievalEvaluationCase, ...] = Field(min_length=1)


class RetrievalCaseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    query: str
    retrieved_document_ids: tuple[str, ...]
    relevant_document_ids: tuple[str, ...]
    recall_at_k: float = Field(ge=0, le=1)
    reciprocal_rank: float = Field(ge=0, le=1)


class RetrievalEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    k: int = Field(gt=0)
    query_count: int = Field(ge=0)
    mean_recall_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    cases: tuple[RetrievalCaseResult, ...]


def load_retrieval_dataset(path: str | Path) -> RetrievalEvaluationDataset:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RetrievalEvaluationDataset.model_validate(data)


class RetrievalEvaluator:
    def __init__(self, retriever: SemanticRetriever):
        self.retriever = retriever

    def evaluate(
        self,
        cases: tuple[RetrievalEvaluationCase, ...],
        k: int = 3,
    ) -> RetrievalEvaluationResult:
        if k < 1:
            raise ValueError("Evaluation k must be positive.")
        results = []
        for case in cases:
            response = self.retriever.search(case.query, limit=k)
            retrieved = tuple(item.document_id for item in response.results)
            relevant = set(case.relevant_document_ids)
            recall = len(relevant.intersection(retrieved)) / len(relevant)
            first_rank = next(
                (
                    rank
                    for rank, document_id in enumerate(retrieved, start=1)
                    if document_id in relevant
                ),
                None,
            )
            results.append(
                RetrievalCaseResult(
                    query=case.query,
                    retrieved_document_ids=retrieved,
                    relevant_document_ids=case.relevant_document_ids,
                    recall_at_k=recall,
                    reciprocal_rank=1 / first_rank if first_rank else 0.0,
                )
            )
        count = len(results)
        return RetrievalEvaluationResult(
            k=k,
            query_count=count,
            mean_recall_at_k=(
                sum(item.recall_at_k for item in results) / count if count else 0.0
            ),
            mean_reciprocal_rank=(
                sum(item.reciprocal_rank for item in results) / count
                if count
                else 0.0
            ),
            cases=tuple(results),
        )


# Backward-compatible name retained for the Stage 5 semantic evaluation script.
SemanticRetrievalEvaluator = RetrievalEvaluator
