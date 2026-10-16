from datetime import datetime, timezone
from uuid import UUID

import pytest

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.memory.schemas import (
    PreferenceCategory,
    PreferenceSource,
    PreferenceStance,
    UserPreferenceRecord,
)
from app.personalization.ranker import PersonalizedRanker
from app.personalization.schemas import (
    PersonalizationProfile,
    PersonalizationRequest,
    PersonalizationWeights,
)
from app.retrieval.schemas import RerankedSearchResult


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def candidate(
    entity_id,
    rank,
    *,
    artist,
    genres=("electronic",),
    popularity=50,
    isrc=None,
):
    return RerankedSearchResult(
        rank=rank,
        hybrid_rank=rank,
        document_id=f"track:{entity_id}",
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        reranker_score=1 / rank,
        fusion_score=1 / (60 + rank),
        content=f"Track {entity_id} by {artist} is atmospheric electronic music.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=entity_id,
            track_id=entity_id,
            artist_name=artist,
            genres=genres,
            popularity=popularity,
            isrc=isrc,
        ),
    )


def preference(category, value, stance, confidence=1.0):
    now = datetime(2026, 10, 16, tzinfo=timezone.utc)
    return UserPreferenceRecord(
        id=1,
        user_id=USER_ID,
        category=category,
        value=value,
        stance=stance,
        source=PreferenceSource.EXPLICIT_USER,
        confidence=confidence,
        version=1,
        created_at=now,
        updated_at=now,
    )


def test_explicit_dislike_can_demote_a_slightly_more_relevant_candidate():
    candidates = (
        candidate(1, 1, artist="Avoid Me"),
        candidate(2, 2, artist="Preferred", genres=("synthwave",)),
    )
    profile = PersonalizationProfile(
        preferences=(
            preference(PreferenceCategory.ARTIST, "Avoid Me", PreferenceStance.AVOID),
            preference(PreferenceCategory.GENRE, "synthwave", PreferenceStance.LIKE),
        ),
        sources=("explicit_preferences",),
    )

    ranked = PersonalizedRanker().rank(
        candidates, PersonalizationRequest(profile=profile), limit=2
    )

    assert [item.entity_id for item in ranked] == [2, 1]
    assert ranked[0].personalization.explicit_preference_score > 0
    assert ranked[1].personalization.explicit_preference_score < 0


def test_provider_signals_balance_affinity_with_novelty():
    candidates = (
        candidate(1, 1, artist="Known Artist", isrc="KNOWN"),
        candidate(2, 2, artist="Known Artist", isrc="NEW"),
        candidate(3, 3, artist="New Artist", isrc="NEWER"),
    )
    profile = PersonalizationProfile(
        known_track_ids=frozenset({1}),
        known_isrcs=frozenset({"known"}),
        familiar_artists=frozenset({"known artist"}),
        sources=("spotify",),
    )

    ranked = PersonalizedRanker().rank(
        candidates, PersonalizationRequest(profile=profile), limit=3
    )

    by_id = {item.entity_id: item.personalization for item in ranked}
    assert by_id[1].provider_affinity_score == 1
    assert by_id[1].novelty_score == 0
    assert by_id[2].provider_affinity_score == 0.5
    assert by_id[2].novelty_score == 0.5
    assert by_id[3].novelty_score == 1


def test_greedy_selection_penalizes_repeated_artist_for_diversity():
    candidates = (
        candidate(1, 1, artist="Same"),
        candidate(2, 2, artist="Same"),
        candidate(3, 3, artist="Different", genres=("ambient",)),
    )
    weights = PersonalizationWeights(
        relevance=0.4, preference=0.0, novelty=0.1, diversity=0.5
    )
    ranked = PersonalizedRanker(weights).rank(
        candidates,
        PersonalizationRequest(profile=PersonalizationProfile()),
        limit=3,
    )
    assert [item.entity_id for item in ranked][:2] == [1, 3]
    assert ranked[2].personalization.diversity_penalty == 1


def test_current_query_popularity_is_scored_without_mutating_memory():
    candidates = (
        candidate(1, 1, artist="One", popularity=90),
        candidate(2, 2, artist="Two", popularity=10),
    )
    from app.query_understanding.schemas import PopularityPreference

    ranked = PersonalizedRanker().rank(
        candidates,
        PersonalizationRequest(
            profile=PersonalizationProfile(),
            query_popularity=PopularityPreference.LOW,
        ),
        limit=2,
    )
    assert ranked[1].personalization.explicit_preference_score == 1


@pytest.mark.parametrize(
    "weights",
    [
        {"relevance": 0.5, "preference": 0.2, "novelty": 0.1, "diversity": 0.1},
        {"relevance": -0.1, "preference": 0.5, "novelty": 0.3, "diversity": 0.3},
    ],
)
def test_weights_are_bounded_and_must_sum_to_one(weights):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PersonalizationWeights(**weights)
