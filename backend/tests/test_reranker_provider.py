import pytest

from app.config.settings import Settings
from app.reranking.factory import get_reranker
from app.reranking.fastembed import FastEmbedCrossEncoderReranker


class FakeCrossEncoder:
    def __init__(self, scores=(0.3, 0.8)):
        self.scores = scores
        self.calls = []

    def rerank(self, query, documents, batch_size):
        self.calls.append((query, documents, batch_size))
        return iter(self.scores)


def provider(model=None, **kwargs):
    return FastEmbedCrossEncoderReranker(
        model_name="test-reranker",
        version="test-version",
        model=model or FakeCrossEncoder(),
        **kwargs,
    )


def test_fastembed_reranker_scores_query_document_pairs_in_order():
    model = FakeCrossEncoder()
    reranker = provider(model=model, batch_size=8)
    assert reranker.score("  dark   music ", ["first", "second"]) == [0.3, 0.8]
    assert model.calls == [("dark music", ["first", "second"], 8)]
    assert reranker.signature == "test-reranker:test-version"


def test_fastembed_reranker_handles_empty_document_list_without_model_call():
    model = FakeCrossEncoder(scores=())
    reranker = provider(model=model)
    assert reranker.score("music", []) == []
    assert model.calls == []


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_fastembed_reranker_rejects_empty_query(query):
    with pytest.raises(ValueError, match="query must not be empty"):
        provider().score(query, ["document"])


def test_fastembed_reranker_rejects_empty_documents():
    with pytest.raises(ValueError, match="documents must not be empty"):
        provider().score("music", [" "])


@pytest.mark.parametrize("scores", [(0.5,), (0.5, float("nan"))])
def test_fastembed_reranker_rejects_invalid_model_output(scores):
    reranker = provider(model=FakeCrossEncoder(scores=scores))
    with pytest.raises(RuntimeError):
        reranker.score("music", ["first", "second"])


def test_fastembed_reranker_validates_configuration():
    with pytest.raises(ValueError, match="batch size"):
        provider(batch_size=0)
    with pytest.raises(ValueError, match="name and version"):
        FastEmbedCrossEncoderReranker(" ", "version", model=FakeCrossEncoder())


def test_reranker_factory_uses_configuration_without_loading_model():
    settings = Settings(
        _env_file=None,
        reranker_provider="fastembed",
        reranker_model="Xenova/ms-marco-MiniLM-L-6-v2",
        reranker_version="test-version",
        reranker_cache_dir="data/test-models",
        reranker_batch_size=4,
    )
    reranker = get_reranker(settings)
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"
    assert reranker.version == "test-version"


def test_reranker_factory_rejects_unknown_provider():
    settings = Settings(_env_file=None, reranker_provider="unknown")
    with pytest.raises(ValueError, match="Unsupported reranker provider"):
        get_reranker(settings)
