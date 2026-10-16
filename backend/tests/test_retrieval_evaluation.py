from pathlib import Path

import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.retrieval.evaluation import (
    RetrievalEvaluationCase,
    SemanticRetrievalEvaluator,
    load_retrieval_dataset,
)
from app.retrieval.schemas import SemanticSearchResponse, SemanticSearchResult


class StubRetriever:
    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query, limit=10):
        document_ids = self.rankings[query][:limit]
        return SemanticSearchResponse(
            query=query,
            results=tuple(
                SemanticSearchResult(
                    rank=rank,
                    document_id=document_id,
                    entity_type=DocumentType.TRACK,
                    entity_id=int(document_id.split(":")[1]),
                    score=1.0 / rank,
                    content=document_id,
                    metadata=DocumentMetadata(
                        document_type=DocumentType.TRACK,
                        entity_id=int(document_id.split(":")[1]),
                        track_id=int(document_id.split(":")[1]),
                    ),
                )
                for rank, document_id in enumerate(document_ids, start=1)
            ),
        )


def test_evaluator_calculates_recall_and_mrr():
    cases = (
        RetrievalEvaluationCase(query="first", relevant_document_ids=("track:1",)),
        RetrievalEvaluationCase(query="second", relevant_document_ids=("track:4",)),
    )
    retriever = StubRetriever(
        {
            "first": ["track:1", "track:2"],
            "second": ["track:3", "track:4"],
        }
    )
    result = SemanticRetrievalEvaluator(retriever).evaluate(cases, k=2)
    assert result.query_count == 2
    assert result.mean_recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 0.75
    assert result.cases[1].reciprocal_rank == 0.5


def test_evaluator_handles_misses():
    cases = (
        RetrievalEvaluationCase(query="miss", relevant_document_ids=("track:9",)),
    )
    result = SemanticRetrievalEvaluator(
        StubRetriever({"miss": ["track:1"]})
    ).evaluate(cases, k=1)
    assert result.mean_recall_at_k == 0.0
    assert result.mean_reciprocal_rank == 0.0


def test_evaluator_rejects_invalid_k():
    with pytest.raises(ValueError, match="positive"):
        SemanticRetrievalEvaluator(StubRetriever({})).evaluate((), k=0)


def test_curated_retrieval_dataset_is_valid():
    path = Path(__file__).parents[2] / "evaluation" / "semantic_retrieval.json"
    dataset = load_retrieval_dataset(path)
    assert len(dataset.documents) == 6
    assert len(dataset.cases) == 5
    document_ids = {document.document_id for document in dataset.documents}
    assert all(
        set(case.relevant_document_ids) <= document_ids for case in dataset.cases
    )
