import pytest

from app.config.settings import Settings
from app.embeddings.factory import get_embedding_provider
from app.embeddings.fastembed import FastEmbedEmbeddingProvider


class FakeVector(list):
    def astype(self, _type):
        return self

    def tolist(self):
        return list(self)


class FakeModel:
    def __init__(self):
        self.passage_calls = []
        self.query_calls = []

    def passage_embed(self, texts):
        self.passage_calls.append(texts)
        return [FakeVector([float(index), 1.0, 0.0]) for index, _ in enumerate(texts)]

    def query_embed(self, texts):
        self.query_calls.append(texts)
        return [FakeVector([0.0, 1.0, 0.0])]


def test_fastembed_uses_separate_passage_and_query_paths():
    model = FakeModel()
    provider = FastEmbedEmbeddingProvider(
        model_name="test-model",
        version="test-version",
        model=model,
        dimension=3,
        expected_dimension=3,
    )
    assert provider.embed_documents(["one", "two"]) == [
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    assert provider.embed_query("query") == [0.0, 1.0, 0.0]
    assert model.passage_calls == [["one", "two"]]
    assert model.query_calls == [["query"]]


@pytest.mark.parametrize("text", ["", "   "])
def test_fastembed_rejects_empty_text(text):
    provider = FastEmbedEmbeddingProvider(
        model_name="test-model",
        version="test-version",
        model=FakeModel(),
        dimension=3,
    )
    with pytest.raises(ValueError, match="empty text"):
        provider.embed_query(text)


def test_fastembed_validates_configured_dimension():
    with pytest.raises(ValueError, match="does not match"):
        FastEmbedEmbeddingProvider(
            model_name="test-model",
            version="test-version",
            model=FakeModel(),
            dimension=3,
            expected_dimension=4,
        )


def test_fastembed_factory_configuration_is_lazy():
    settings = Settings(
        _env_file=None,
        embedding_provider="fastembed",
        embedding_model="BAAI/bge-small-en-v1.5",
        embedding_version="test-version",
        embedding_dimension=384,
        embedding_cache_dir="data/test-models",
    )
    provider = get_embedding_provider(settings)
    assert provider.model_name == "BAAI/bge-small-en-v1.5"
    assert provider.dimension == 384
    assert provider.version == "test-version"
