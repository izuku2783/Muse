from app.config.settings import Settings
from app.llm.groq import GroqLLMProvider
from app.rag import runtime
from app.rag.generator import LLMRAGGenerator
from app.rag.service import RAGService
from app.retrieval.reranked import RerankedRetriever


class FakeCloseable:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeCorpus:
    def __init__(self, session):
        self.session = session

    def build_documents(self):
        return []


class FakeEmbeddingProvider:
    model_name = "fake-embedding"
    version = "1"


class FakeReranker:
    model_name = "fake-reranker"
    version = "1"

    def score(self, query, documents):
        return [0.0 for _ in documents]


def test_build_rag_runtime_wires_retrieval_and_generation(monkeypatch):
    store = FakeCloseable()
    monkeypatch.setattr(runtime, "get_vector_store", lambda settings: store)
    monkeypatch.setattr(runtime, "SessionLocal", FakeSession)
    monkeypatch.setattr(runtime, "KnowledgeCorpusService", FakeCorpus)
    monkeypatch.setattr(
        runtime,
        "get_embedding_provider",
        lambda settings: FakeEmbeddingProvider(),
    )
    monkeypatch.setattr(runtime, "get_reranker", lambda settings: FakeReranker())
    provider = GroqLLMProvider(api_key="test-key", model="test-model")
    monkeypatch.setattr(runtime, "get_llm_provider", lambda settings: provider)
    settings = Settings(
        _env_file=None,
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
    )

    built = runtime.build_rag_runtime(settings)

    assert isinstance(built.service, RAGService)
    assert isinstance(built.service.retriever, RerankedRetriever)
    assert isinstance(built.service.generator, LLMRAGGenerator)
    assert built.llm_provider is provider
    assert built.llm_provider.default_model == "test-model"
    built.close()
    assert store.closed is True


def test_cached_rag_runtime_is_reused_and_closed(monkeypatch):
    runtime.close_rag_runtime()
    built = runtime.RAGRuntime(
        service=object(),
        vector_store=FakeCloseable(),
        llm_provider=FakeCloseable(),
    )
    monkeypatch.setattr(runtime, "build_rag_runtime", lambda: built)

    assert runtime.get_rag_runtime() is built
    assert runtime.get_rag_runtime() is built

    runtime.close_rag_runtime()
    assert built.vector_store.closed is True
    assert built.llm_provider.closed is True
