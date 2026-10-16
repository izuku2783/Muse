from app.config.settings import Settings, get_settings
from app.reranking.fastembed import FastEmbedCrossEncoderReranker
from app.reranking.provider import Reranker


def get_reranker(settings: Settings | None = None) -> Reranker:
    configured = settings or get_settings()
    if configured.reranker_provider.casefold() == "fastembed":
        return FastEmbedCrossEncoderReranker(
            model_name=configured.reranker_model,
            version=configured.reranker_version,
            cache_dir=configured.reranker_cache_dir,
            batch_size=configured.reranker_batch_size,
        )
    raise ValueError(f"Unsupported reranker provider: {configured.reranker_provider}")
