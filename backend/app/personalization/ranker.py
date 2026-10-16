import re
from dataclasses import dataclass
from math import isfinite

from app.memory.schemas import PreferenceCategory, PreferenceStance
from app.personalization.schemas import (
    PersonalizationProfile,
    PersonalizationRequest,
    PersonalizationWeights,
)
from app.query_understanding.schemas import PopularityPreference
from app.retrieval.schemas import (
    PersonalizationBreakdown,
    RerankedSearchResult,
)


_TOKEN_PATTERN = re.compile(r"[\w]+", re.UNICODE)
_POSITIVE_STANCES = {
    PreferenceStance.LIKE,
    PreferenceStance.PREFER,
    PreferenceStance.FREQUENT,
}
_NEGATIVE_STANCES = {PreferenceStance.DISLIKE, PreferenceStance.AVOID}


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _tokens(value: str) -> frozenset[str]:
    return frozenset(_TOKEN_PATTERN.findall(_normalize(value)))


@dataclass(frozen=True)
class _CandidateScore:
    result: RerankedSearchResult
    relevance: float
    explicit: float
    provider: float
    preference: float
    novelty: float
    base_score: float
    reasons: tuple[str, ...]


class PersonalizedRanker:
    """Reorder relevant candidates with bounded, explainable user signals."""

    def __init__(self, weights: PersonalizationWeights | None = None):
        self.weights = weights or PersonalizationWeights()

    def rank(
        self,
        candidates: tuple[RerankedSearchResult, ...],
        request: PersonalizationRequest,
        limit: int,
    ) -> tuple[RerankedSearchResult, ...]:
        if limit < 1:
            raise ValueError("Personalized result limit must be positive.")
        if not candidates:
            return ()
        scored = tuple(
            self._score(candidate, rank, len(candidates), request)
            for rank, candidate in enumerate(candidates, start=1)
        )
        if not all(isfinite(item.base_score) for item in scored):
            raise RuntimeError("Personalization produced a non-finite score.")

        selected: list[_CandidateScore] = []
        remaining = list(scored)
        output: list[RerankedSearchResult] = []
        while remaining and len(output) < limit:
            ranked = []
            for item in remaining:
                penalty = self._diversity_penalty(item, selected)
                final = item.base_score - self.weights.diversity * penalty
                ranked.append((final, penalty, item))
            final, penalty, chosen = sorted(
                ranked,
                key=lambda value: (
                    -value[0],
                    value[2].result.rank,
                    value[2].result.document_id,
                ),
            )[0]
            selected.append(chosen)
            remaining.remove(chosen)
            reasons = chosen.reasons
            if penalty > 0:
                reasons = (*reasons, "diversity adjustment applied")
            output.append(
                chosen.result.model_copy(
                    update={
                        "rank": len(output) + 1,
                        "personalization": PersonalizationBreakdown(
                            relevance_score=chosen.relevance,
                            explicit_preference_score=chosen.explicit,
                            provider_affinity_score=chosen.provider,
                            preference_score=chosen.preference,
                            novelty_score=chosen.novelty,
                            diversity_penalty=penalty,
                            final_score=final,
                            reasons=reasons,
                        ),
                    }
                )
            )
        return tuple(output)

    def _score(
        self,
        candidate: RerankedSearchResult,
        rank: int,
        count: int,
        request: PersonalizationRequest,
    ) -> _CandidateScore:
        relevance = (count - rank + 1) / count
        explicit, explicit_reasons = self._explicit_score(
            candidate, request.profile, request.query_popularity
        )
        provider, provider_reasons = self._provider_affinity(
            candidate, request.profile
        )
        has_explicit_signal = bool(request.profile.preferences) or request.query_popularity is not None
        has_provider_signal = bool(
            request.profile.known_track_ids
            or request.profile.known_isrcs
            or request.profile.familiar_artists
        )
        if has_explicit_signal and has_provider_signal:
            preference = 0.75 * explicit + 0.25 * provider
        elif has_explicit_signal:
            preference = explicit
        else:
            preference = provider
        novelty, novelty_reason = self._novelty(candidate, request.profile)
        base_score = (
            self.weights.relevance * relevance
            + self.weights.preference * preference
            + self.weights.novelty * novelty
        )
        reasons = (*explicit_reasons, *provider_reasons)
        if novelty_reason:
            reasons = (*reasons, novelty_reason)
        return _CandidateScore(
            result=candidate,
            relevance=relevance,
            explicit=explicit,
            provider=provider,
            preference=preference,
            novelty=novelty,
            base_score=base_score,
            reasons=reasons,
        )

    @staticmethod
    def _explicit_score(
        candidate: RerankedSearchResult,
        profile: PersonalizationProfile,
        query_popularity: PopularityPreference | None,
    ) -> tuple[float, tuple[str, ...]]:
        metadata = candidate.metadata
        artist = _normalize(metadata.artist_name or "")
        genres = {_normalize(value) for value in metadata.genres}
        content_tokens = _tokens(candidate.content)
        total = 0.0
        denominator = 0.0
        reasons = []
        for preference in profile.preferences:
            value = _normalize(preference.value)
            matched = False
            if preference.category == PreferenceCategory.ARTIST:
                matched = value == artist
            elif preference.category == PreferenceCategory.GENRE:
                matched = value in genres
            elif preference.category == PreferenceCategory.POPULARITY:
                matched = PersonalizedRanker._popularity_matches(
                    metadata.popularity, PopularityPreference(value)
                )
            elif preference.category in {
                PreferenceCategory.CONTEXT,
                PreferenceCategory.MOOD,
            }:
                matched = bool(_tokens(value)) and _tokens(value) <= content_tokens
            if not matched:
                continue
            denominator += preference.confidence
            direction = 1.0 if preference.stance in _POSITIVE_STANCES else -1.0
            if preference.stance not in _POSITIVE_STANCES | _NEGATIVE_STANCES:
                continue
            total += direction * preference.confidence
            reasons.append(
                f"explicit {preference.category.value} {preference.stance.value}: {preference.value}"
            )
        if query_popularity is not None:
            denominator += 1.0
            if PersonalizedRanker._popularity_matches(
                metadata.popularity, query_popularity
            ):
                total += 1.0
                reasons.append(f"current popularity preference: {query_popularity.value}")
        return (total / denominator if denominator else 0.0), tuple(reasons)

    @staticmethod
    def _popularity_matches(
        popularity: float | None, preference: PopularityPreference
    ) -> bool:
        if popularity is None:
            return False
        if preference == PopularityPreference.LOW:
            return popularity < 34
        if preference == PopularityPreference.MODERATE:
            return 34 <= popularity <= 66
        return popularity > 66

    @staticmethod
    def _provider_affinity(
        candidate: RerankedSearchResult,
        profile: PersonalizationProfile,
    ) -> tuple[float, tuple[str, ...]]:
        metadata = candidate.metadata
        isrc = _normalize(metadata.isrc or "")
        artist = _normalize(metadata.artist_name or "")
        if metadata.track_id in profile.known_track_ids or (
            isrc and isrc in profile.known_isrcs
        ):
            return 1.0, ("known from connected listening history",)
        if artist and artist in profile.familiar_artists:
            return 0.5, ("artist appears in connected listening history",)
        return 0.0, ()

    @staticmethod
    def _novelty(
        candidate: RerankedSearchResult,
        profile: PersonalizationProfile,
    ) -> tuple[float, str | None]:
        affinity, _ = PersonalizedRanker._provider_affinity(candidate, profile)
        if affinity == 1.0:
            return 0.0, "already familiar track"
        if affinity == 0.5:
            return 0.5, "new track from a familiar artist"
        return 1.0, "novel relative to available listening signals"

    @staticmethod
    def _diversity_penalty(
        candidate: _CandidateScore,
        selected: list[_CandidateScore],
    ) -> float:
        if not selected:
            return 0.0
        artist = _normalize(candidate.result.metadata.artist_name or "")
        genres = {_normalize(value) for value in candidate.result.metadata.genres}
        penalty = 0.0
        for other in selected:
            other_artist = _normalize(other.result.metadata.artist_name or "")
            if artist and artist == other_artist:
                penalty = max(penalty, 1.0)
            other_genres = {
                _normalize(value) for value in other.result.metadata.genres
            }
            union = genres | other_genres
            if union:
                penalty = max(penalty, 0.5 * len(genres & other_genres) / len(union))
        return penalty
