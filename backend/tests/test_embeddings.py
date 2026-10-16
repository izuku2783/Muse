import math

import pytest

from app.config.settings import Settings
from app.embeddings.factory import get_embedding_provider
from app.embeddings.hashing import HashingEmbeddingProvider


def test_hashing_embeddings_are_deterministic_and_normalized():
    provider = HashingEmbeddingProvider(dimension=32)
    first = provider.embed_text("Dark atmospheric electronic music")
    second = provider.embed_text("Dark atmospheric electronic music")
    assert first == second
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)


def test_hashing_embedding_is_unicode_and_case_normalized():
    provider = HashingEmbeddingProvider(dimension=32)
    assert provider.embed_text("CAFÉ MUSIC") == provider.embed_text("cafe\u0301 music")


def test_document_and_query_embedding_methods_use_same_contract():
    provider = HashingEmbeddingProvider(dimension=16)
    texts = ["first track", "second track"]
    assert provider.embed_documents(texts) == [
        provider.embed_text(text) for text in texts
    ]
    assert provider.embed_query(texts[0]) == provider.embed_text(texts[0])


def test_embedding_identity_is_explicit():
    provider = HashingEmbeddingProvider(
        dimension=16,
        model_name="test-model",
        version="2026-10-09",
    )
    assert provider.model_name == "test-model"
    assert provider.version == "2026-10-09"
    assert provider.signature == "test-model:2026-10-09"


@pytest.mark.parametrize("text", ["", "  ", "\n\t"])
def test_embedding_rejects_empty_text(text):
    with pytest.raises(ValueError, match="empty text"):
        HashingEmbeddingProvider().embed_text(text)


def test_embedding_factory_uses_configuration():
    settings = Settings(
        _env_file=None,
        embedding_provider="hashing",
        embedding_model="configured-model",
        embedding_version="2",
        embedding_dimension=24,
    )
    provider = get_embedding_provider(settings)
    assert provider.signature == "configured-model:2"
    assert provider.dimension == 24


def test_embedding_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, embedding_provider="unknown")
    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        get_embedding_provider(settings)
