import json
from datetime import datetime, timezone
from uuid import UUID

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.memory.schemas import (
    PreferenceCategory,
    PreferenceSource,
    PreferenceStance,
    UserPreferenceRecord,
)
from app.personalization.evaluation import (
    PersonalizationEvaluationCase,
    evaluate_personalization,
)
from app.personalization.ranker import PersonalizedRanker
from app.personalization.schemas import (
    PersonalizationProfile,
    PersonalizationRequest,
    PersonalizationWeights,
)
from app.retrieval.schemas import RerankedSearchResult


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def candidate(identifier: int, rank: int, artist: str, genre: str) -> RerankedSearchResult:
    return RerankedSearchResult(
        rank=rank,
        hybrid_rank=rank,
        document_id=f"track:{identifier}",
        entity_type=DocumentType.TRACK,
        entity_id=identifier,
        reranker_score=1 / rank,
        fusion_score=1 / (60 + rank),
        content=f"Track {identifier} by {artist}; genre: {genre}.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=identifier,
            track_id=identifier,
            artist_name=artist,
            genres=(genre,),
            isrc=f"ISRC{identifier}",
        ),
    )


def preference(identifier, category, value, stance):
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return UserPreferenceRecord(
        id=identifier,
        user_id=USER_ID,
        category=category,
        value=value,
        stance=stance,
        source=PreferenceSource.EXPLICIT_USER,
        confidence=1,
        version=1,
        created_at=now,
        updated_at=now,
    )


def cases() -> tuple[PersonalizationEvaluationCase, ...]:
    preference_candidates = tuple(
        candidate(index, index, "Avoided" if index == 1 else f"Artist {index}", "synthwave" if index == 2 else "electronic")
        for index in range(1, 11)
    )
    diversity_candidates = tuple(
        candidate(index + 20, index, "Repeated" if index <= 3 else f"Artist {index}", "ambient" if index == 4 else "electronic")
        for index in range(1, 11)
    )
    novelty_candidates = tuple(
        candidate(index + 40, index, f"Artist {index}", "electronic")
        for index in range(1, 11)
    )
    return (
        PersonalizationEvaluationCase(
            name="explicit preference and avoidance",
            candidates=preference_candidates,
            request=PersonalizationRequest(
                profile=PersonalizationProfile(
                    preferences=(
                        preference(1, PreferenceCategory.ARTIST, "Avoided", PreferenceStance.AVOID),
                        preference(2, PreferenceCategory.GENRE, "synthwave", PreferenceStance.LIKE),
                    ),
                    sources=("explicit_preferences",),
                )
            ),
            preferred_document_ids=frozenset({"track:2"}),
            avoided_document_ids=frozenset({"track:1"}),
        ),
        PersonalizationEvaluationCase(
            name="artist diversity",
            candidates=diversity_candidates,
            request=PersonalizationRequest(
                profile=PersonalizationProfile(sources=("explicit_preferences",))
            ),
            preferred_document_ids=frozenset(),
            avoided_document_ids=frozenset(),
        ),
        PersonalizationEvaluationCase(
            name="known versus novel",
            candidates=novelty_candidates,
            request=PersonalizationRequest(
                profile=PersonalizationProfile(
                    known_track_ids=frozenset({41, 42}),
                    known_isrcs=frozenset({"isrc41", "isrc42"}),
                    sources=("spotify",),
                )
            ),
            preferred_document_ids=frozenset(),
            avoided_document_ids=frozenset(),
        ),
    )


def main() -> None:
    presets = {
        "relevance_only": PersonalizationWeights(
            relevance=1, preference=0, novelty=0, diversity=0
        ),
        "stage17_balanced": PersonalizationWeights(
            relevance=0.5, preference=0.2, novelty=0.15, diversity=0.15
        ),
    }
    output = {
        name: evaluate_personalization(PersonalizedRanker(weights), cases()).model_dump()
        for name, weights in presets.items()
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
