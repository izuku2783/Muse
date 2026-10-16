from app.config.settings import Settings, get_settings
from app.embeddings.hashing import HashingEmbeddingProvider
from app.embeddings.fastembed import FastEmbedEmbeddingProvider
from app.embeddings.provider import EmbeddingProvider


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    configured = settings or get_settings()
    if configured.embedding_provider.casefold() == "hashing":
        return HashingEmbeddingProvider(
            dimension=configured.embedding_dimension,
            model_name=configured.embedding_model,
            version=configured.embedding_version,
        )
    if configured.embedding_provider.casefold() == "fastembed":
        return FastEmbedEmbeddingProvider(
            model_name=configured.embedding_model,
            version=configured.embedding_version,
            cache_dir=configured.embedding_cache_dir,
            expected_dimension=configured.embedding_dimension,
        )
    raise ValueError(
        f"Unsupported embedding provider: {configured.embedding_provider}"
    )
