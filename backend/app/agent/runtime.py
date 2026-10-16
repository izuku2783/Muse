from dataclasses import dataclass
from threading import Lock
from uuid import UUID

from app.agent.generator import LLMAgentGenerator
from app.agent.schemas import AgentLimits, AgentResponse
from app.agent.service import AgentService
from app.config.settings import Settings, get_settings
from app.conversation.store import InMemoryConversationStore
from app.database.session import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.knowledge.corpus import KnowledgeCorpusService
from app.llm.factory import get_llm_provider
from app.llm.provider import LLMProvider
from app.personalization.provider import DatabasePersonalizationProfileProvider
from app.personalization.ranker import PersonalizedRanker
from app.personalization.retriever import PersonalizedRetriever
from app.personalization.schemas import PersonalizationWeights
from app.query_understanding.service import QueryUnderstandingService
from app.reranking.factory import get_reranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import BM25LexicalRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.semantic import SemanticRetriever
from app.services.music.musicbrainz import MusicBrainzProvider
from app.services.music.provider import MusicProvider
from app.spotify.personalization import SpotifyListeningSignalSource
from app.spotify.runtime import get_spotify_runtime
from app.tools.music import SearchExternalMusicProviderTool
from app.tools.registry import ToolRegistry
from app.tools.retrieval import RetrieveMusicEvidenceTool
from app.vector_store.factory import get_vector_store
from app.vector_store.qdrant import QdrantVectorStore


@dataclass
class AgentRuntime:
    service: AgentService
    vector_store: QdrantVectorStore
    llm_provider: LLMProvider
    music_provider: MusicProvider

    def run(
        self, query: str, conversation_id: UUID | str | None = None
    ) -> AgentResponse:
        return self.service.run(query, conversation_id=conversation_id)

    def close(self) -> None:
        try:
            self.music_provider.close()
        finally:
            try:
                self.llm_provider.close()
            finally:
                self.vector_store.close()


def build_agent_runtime(settings: Settings | None = None) -> AgentRuntime:
    configured = settings or get_settings()
    vector_store = get_vector_store(configured)
    llm_provider = None
    music_provider = None
    try:
        with SessionLocal() as session:
            documents = KnowledgeCorpusService(session).build_documents()
        semantic = SemanticRetriever(get_embedding_provider(configured), vector_store)
        hybrid = HybridRetriever(
            semantic,
            BM25LexicalRetriever(documents),
            candidate_limit=100,
        )
        reranked = RerankedRetriever(
            hybrid,
            get_reranker(configured),
            candidate_limit=20,
        )
        listening_sources = ()
        if configured.spotify_client_id and configured.spotify_token_encryption_key:
            listening_sources = (
                SpotifyListeningSignalSource(
                    get_spotify_runtime(),
                    limit=configured.personalization_spotify_signal_limit,
                ),
            )
        personalized = PersonalizedRetriever(
            reranked,
            DatabasePersonalizationProfileProvider(
                SessionLocal,
                listening_sources=listening_sources,
            ),
            PersonalizedRanker(
                PersonalizationWeights(
                    relevance=configured.personalization_relevance_weight,
                    preference=configured.personalization_preference_weight,
                    novelty=configured.personalization_novelty_weight,
                    diversity=configured.personalization_diversity_weight,
                )
            ),
            candidate_limit=20,
        )
        llm_provider = get_llm_provider(configured)
        music_provider = MusicBrainzProvider(
            user_agent=configured.musicbrainz_user_agent,
        )
        tools = ToolRegistry(
            (
                RetrieveMusicEvidenceTool(personalized),
                SearchExternalMusicProviderTool(music_provider),
            )
        )
        service = AgentService(
            QueryUnderstandingService(llm_provider),
            tools,
            LLMAgentGenerator(llm_provider),
            limits=AgentLimits(
                max_iterations=configured.agent_max_iterations,
                max_tool_calls=configured.agent_max_tool_calls,
                timeout_seconds=configured.agent_timeout_seconds,
                max_evidence=configured.agent_max_evidence,
            ),
            conversation_store=InMemoryConversationStore(
                ttl_seconds=configured.conversation_ttl_seconds,
                max_conversations=configured.conversation_max_active,
            ),
        )
        return AgentRuntime(
            service=service,
            vector_store=vector_store,
            llm_provider=llm_provider,
            music_provider=music_provider,
        )
    except Exception:
        if music_provider is not None:
            music_provider.close()
        if llm_provider is not None:
            llm_provider.close()
        vector_store.close()
        raise


_runtime: AgentRuntime | None = None
_runtime_lock = Lock()


def get_agent_runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_agent_runtime()
    return _runtime


def get_agent_service() -> AgentService:
    return get_agent_runtime().service


def close_agent_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None
