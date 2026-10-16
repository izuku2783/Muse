from uuid import UUID

from app.personalization.provider import PersonalizationProfileProvider
from app.personalization.ranker import PersonalizedRanker
from app.personalization.schemas import PersonalizationProfile, PersonalizationRequest
from app.query_understanding.schemas import PopularityPreference
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.schemas import RerankedSearchResponse
from app.vector_store.schemas import VectorSearchFilter


class PersonalizedRetriever:
    """Optional personalization wrapper over the established relevance pipeline."""

    def __init__(
        self,
        base_retriever: RerankedRetriever,
        profile_provider: PersonalizationProfileProvider,
        ranker: PersonalizedRanker,
        candidate_limit: int = 20,
    ):
        if not 1 <= candidate_limit <= base_retriever.candidate_limit:
            raise ValueError(
                "Personalization candidate limit must fit the reranker candidate limit."
            )
        self.base_retriever = base_retriever
        self.profile_provider = profile_provider
        self.ranker = ranker
        self.candidate_limit = candidate_limit

    def search(
        self,
        query: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        semantic_score_threshold: float | None = None,
        user_id: UUID | None = None,
        query_popularity: PopularityPreference | None = None,
    ) -> RerankedSearchResponse:
        if user_id is None and query_popularity is None:
            return self.base_retriever.search(
                query,
                limit=limit,
                filters=filters,
                semantic_score_threshold=semantic_score_threshold,
            )
        base = self.base_retriever.search(
            query,
            limit=self.candidate_limit,
            filters=filters,
            semantic_score_threshold=semantic_score_threshold,
        )
        profile = (
            self.profile_provider.get_profile(user_id)
            if user_id is not None
            else PersonalizationProfile()
        )
        has_signal = bool(profile.preferences or profile.sources or query_popularity)
        if not has_signal:
            return base.model_copy(update={"results": base.results[:limit]})
        ranked = self.ranker.rank(
            base.results,
            PersonalizationRequest(
                profile=profile,
                query_popularity=query_popularity,
            ),
            limit=limit,
        )
        return base.model_copy(
            update={
                "results": ranked,
                "personalization_applied": True,
                "personalization_sources": (
                    *profile.sources,
                    *(("query_preferences",) if query_popularity else ()),
                ),
            }
        )
