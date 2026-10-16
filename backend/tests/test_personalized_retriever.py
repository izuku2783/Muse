from uuid import UUID

from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.personalization.ranker import PersonalizedRanker
from app.personalization.retriever import PersonalizedRetriever
from app.personalization.schemas import PersonalizationProfile
from app.retrieval.schemas import RerankedSearchResponse, RerankedSearchResult


USER_ID = UUID("00000000-0000-0000-0000-000000000001")


def result(entity_id, rank, artist):
    return RerankedSearchResult(
        rank=rank,
        hybrid_rank=rank,
        document_id=f"track:{entity_id}",
        entity_type=DocumentType.TRACK,
        entity_id=entity_id,
        reranker_score=1 / rank,
        fusion_score=1 / (60 + rank),
        content=f"Track by {artist}",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=entity_id,
            track_id=entity_id,
            artist_name=artist,
        ),
    )


class StubBaseRetriever:
    candidate_limit = 20

    def __init__(self):
        self.calls = []
        self.items = (result(1, 1, "Known"), result(2, 2, "New"))

    def search(self, query, limit=10, filters=None, semantic_score_threshold=None):
        self.calls.append((query, limit, filters, semantic_score_threshold))
        return RerankedSearchResponse(
            query=query,
            candidate_count=len(self.items),
            reranker_model="test",
            reranker_version="1",
            results=self.items[:limit],
        )


class StubProfileProvider:
    def __init__(self, profile):
        self.profile = profile
        self.calls = []

    def get_profile(self, user_id):
        self.calls.append(user_id)
        return self.profile


def test_anonymous_search_preserves_existing_retriever_behavior_exactly():
    base = StubBaseRetriever()
    profiles = StubProfileProvider(PersonalizationProfile())
    retriever = PersonalizedRetriever(base, profiles, PersonalizedRanker())

    response = retriever.search("music", limit=1)

    assert response == base.search("music", limit=1)
    assert profiles.calls == []


def test_user_search_fetches_more_candidates_and_returns_scoring_provenance():
    base = StubBaseRetriever()
    profiles = StubProfileProvider(
        PersonalizationProfile(
            familiar_artists=frozenset({"known"}),
            sources=("spotify",),
        )
    )
    retriever = PersonalizedRetriever(base, profiles, PersonalizedRanker(), candidate_limit=2)

    response = retriever.search("music", limit=1, user_id=USER_ID)

    assert base.calls == [("music", 2, None, None)]
    assert profiles.calls == [USER_ID]
    assert response.personalization_applied is True
    assert response.personalization_sources == ("spotify",)
    assert response.results[0].personalization is not None


def test_user_without_signals_keeps_base_order_without_claiming_personalization():
    base = StubBaseRetriever()
    retriever = PersonalizedRetriever(
        base,
        StubProfileProvider(PersonalizationProfile()),
        PersonalizedRanker(),
        candidate_limit=2,
    )
    response = retriever.search("music", limit=1, user_id=USER_ID)
    assert response.results[0].entity_id == 1
    assert response.personalization_applied is False


def test_current_query_preference_can_apply_without_a_durable_user_profile():
    from app.query_understanding.schemas import PopularityPreference

    base = StubBaseRetriever()
    profiles = StubProfileProvider(PersonalizationProfile())
    retriever = PersonalizedRetriever(
        base, profiles, PersonalizedRanker(), candidate_limit=2
    )
    response = retriever.search(
        "less popular music",
        limit=1,
        query_popularity=PopularityPreference.LOW,
    )
    assert response.personalization_applied is True
    assert response.personalization_sources == ("query_preferences",)
    assert profiles.calls == []
