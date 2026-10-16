from app.agent import runtime
from app.agent.generator import LLMAgentGenerator
from app.agent.service import AgentService
from app.config.settings import Settings
from app.llm.groq import GroqLLMProvider


class FakeCloseable:
    def __init__(self, *args, **kwargs):
        self.closed = False

    def close(self):
        self.closed = True


class FakeMusicProvider(FakeCloseable):
    def search_tracks(self, query, limit=10, offset=0): return []
    def get_track(self, track_id): return None
    def get_artist(self, artist_id): return None
    def get_album(self, album_id): return None
    def get_related_artists(self, artist_id): return []


class FakeSession:
    def __enter__(self): return self
    def __exit__(self, *args): pass


class FakeCorpus:
    def __init__(self, session): pass
    def build_documents(self): return []


class FakeEmbeddingProvider:
    model_name = "fake-embedding"
    version = "1"


class FakeReranker:
    model_name = "fake-reranker"
    version = "1"
    def score(self, query, documents): return [0.0 for _ in documents]


def test_build_agent_runtime_wires_graph_tools_and_limits(monkeypatch):
    store = FakeCloseable()
    music_provider = FakeMusicProvider()
    llm_provider = GroqLLMProvider(api_key="test-key", model="test-model")
    monkeypatch.setattr(runtime, "get_vector_store", lambda settings: store)
    monkeypatch.setattr(runtime, "SessionLocal", FakeSession)
    monkeypatch.setattr(runtime, "KnowledgeCorpusService", FakeCorpus)
    monkeypatch.setattr(runtime, "get_embedding_provider", lambda settings: FakeEmbeddingProvider())
    monkeypatch.setattr(runtime, "get_reranker", lambda settings: FakeReranker())
    monkeypatch.setattr(runtime, "get_llm_provider", lambda settings: llm_provider)
    monkeypatch.setattr(runtime, "MusicBrainzProvider", lambda **kwargs: music_provider)
    settings = Settings(
        _env_file=None,
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        AGENT_MAX_ITERATIONS=3,
        AGENT_MAX_TOOL_CALLS=4,
        AGENT_TIMEOUT_SECONDS=45,
        AGENT_MAX_EVIDENCE=6,
        CONVERSATION_TTL_SECONDS=900,
        CONVERSATION_MAX_ACTIVE=200,
    )

    built = runtime.build_agent_runtime(settings)

    assert isinstance(built.service, AgentService)
    assert isinstance(built.service.generator, LLMAgentGenerator)
    assert built.service.limits.max_iterations == 3
    assert built.service.limits.max_tool_calls == 4
    assert built.service.limits.timeout_seconds == 45
    assert built.service.limits.max_evidence == 6
    assert built.service.conversation_store.ttl.total_seconds() == 900
    assert built.service.conversation_store.max_conversations == 200
    assert {item.name for item in built.service.tools.definitions()} == {
        "retrieve_music_evidence",
        "search_external_music_provider",
    }
    built.close()
    assert store.closed is True
    assert music_provider.closed is True


def test_cached_agent_runtime_is_reused_and_closed(monkeypatch):
    runtime.close_agent_runtime()
    built = runtime.AgentRuntime(
        service=object(),
        vector_store=FakeCloseable(),
        llm_provider=FakeCloseable(),
        music_provider=FakeMusicProvider(),
    )
    monkeypatch.setattr(runtime, "build_agent_runtime", lambda: built)

    assert runtime.get_agent_runtime() is built
    assert runtime.get_agent_runtime() is built

    runtime.close_agent_runtime()
    assert built.vector_store.closed is True
    assert built.llm_provider.closed is True
    assert built.music_provider.closed is True

