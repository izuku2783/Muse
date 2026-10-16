from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.personalization.ranker import PersonalizedRanker
from app.personalization.schemas import PersonalizationRequest
from app.retrieval.schemas import RerankedSearchResult


@dataclass(frozen=True)
class PersonalizationEvaluationCase:
    name: str
    candidates: tuple[RerankedSearchResult, ...]
    request: PersonalizationRequest
    preferred_document_ids: frozenset[str]
    avoided_document_ids: frozenset[str]
    k: int = 3


class PersonalizationEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_count: int = Field(ge=0)
    preference_hit_rate: float = Field(ge=0, le=1)
    avoidance_success_rate: float = Field(ge=0, le=1)
    mean_novelty_at_k: float = Field(ge=0, le=1)
    mean_artist_diversity_at_k: float = Field(ge=0, le=1)
    mean_relevance_retention_at_k: float = Field(ge=0, le=1)


def evaluate_personalization(
    ranker: PersonalizedRanker,
    cases: tuple[PersonalizationEvaluationCase, ...],
) -> PersonalizationEvaluationResult:
    preference_hits = []
    avoidance_successes = []
    novelties = []
    diversities = []
    relevance_retentions = []
    for case in cases:
        ranked = ranker.rank(case.candidates, case.request, limit=case.k)
        ids = tuple(item.document_id for item in ranked)
        if case.preferred_document_ids:
            preference_hits.append(float(ids[0] in case.preferred_document_ids))
        if case.avoided_document_ids:
            avoidance_successes.append(float(ids[0] not in case.avoided_document_ids))
        novelties.append(
            sum(item.personalization.novelty_score for item in ranked) / len(ranked)
            if ranked
            else 0.0
        )
        artists = {
            " ".join((item.metadata.artist_name or "unknown").casefold().split())
            for item in ranked
        }
        diversities.append(len(artists) / len(ranked) if ranked else 0.0)
        original_by_id = {
            item.document_id: (len(case.candidates) - index) / len(case.candidates)
            for index, item in enumerate(case.candidates)
        }
        achieved = sum(original_by_id[item.document_id] for item in ranked)
        ideal = sum(sorted(original_by_id.values(), reverse=True)[: case.k])
        relevance_retentions.append(achieved / ideal if ideal else 1.0)

    def mean(values: list[float], empty: float = 1.0) -> float:
        return sum(values) / len(values) if values else empty

    return PersonalizationEvaluationResult(
        case_count=len(cases),
        preference_hit_rate=mean(preference_hits),
        avoidance_success_rate=mean(avoidance_successes),
        mean_novelty_at_k=mean(novelties, empty=0.0),
        mean_artist_diversity_at_k=mean(diversities, empty=0.0),
        mean_relevance_retention_at_k=mean(relevance_retentions),
    )
