from scripts.evaluate_personalization import cases

from app.personalization.evaluation import evaluate_personalization
from app.personalization.ranker import PersonalizedRanker
from app.personalization.schemas import PersonalizationWeights


def test_balanced_personalization_improves_preferences_and_diversity_with_bounded_relevance_loss():
    relevance_only = evaluate_personalization(
        PersonalizedRanker(
            PersonalizationWeights(
                relevance=1, preference=0, novelty=0, diversity=0
            )
        ),
        cases(),
    )
    balanced = evaluate_personalization(
        PersonalizedRanker(
            PersonalizationWeights(
                relevance=0.5, preference=0.2, novelty=0.15, diversity=0.15
            )
        ),
        cases(),
    )

    assert balanced.preference_hit_rate > relevance_only.preference_hit_rate
    assert balanced.avoidance_success_rate > relevance_only.avoidance_success_rate
    assert balanced.mean_artist_diversity_at_k > relevance_only.mean_artist_diversity_at_k
    assert balanced.mean_relevance_retention_at_k >= 0.9
