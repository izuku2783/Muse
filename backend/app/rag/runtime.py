from dataclasses import dataclass
from threading import Lock

from app.config.settings import Settings, get_settings
from app.database.session import SessionLocal
from app.embeddings.factory import get_embedding_provider
from app.knowledge.corpus import KnowledgeCorpusService
from app.llm.factory import get_llm_provider
from app.llm.provider import LLMProvider
from app.rag.generator import LLMRAGGenerator
from app.rag.service import RAGService
from app.reranking.factory import get_reranker
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lexical import BM25LexicalRetriever
from app.retrieval.reranked import RerankedRetriever
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.factory import get_vector_store
from app.vector_store.qdrant import QdrantVectorStore


@dataclass
class RAGRuntime:
    service: RAGService
    vector_store: QdrantVectorStore
    llm_provider: LLMProvider

    def close(self) -> None:
        try:
            self.llm_provider.close()
        finally:
            self.vector_store.close()


def build_rag_runtime(settings: Settings | None = None) -> RAGRuntime:
    configured = settings or get_settings()
    vector_store = get_vector_store(configured)
    try:
        with SessionLocal() as session:
            documents = KnowledgeCorpusService(session).build_documents()
        hybrid_retriever = HybridRetriever(
            SemanticRetriever(get_embedding_provider(configured), vector_store),
            BM25LexicalRetriever(documents),
            candidate_limit=100,
        )
        retriever = RerankedRetriever(
            hybrid_retriever,
            get_reranker(configured),
            candidate_limit=20,
        )
        llm_provider = get_llm_provider(configured)
        generator = LLMRAGGenerator(llm_provider)
        return RAGRuntime(
            service=RAGService(retriever, generator),
            vector_store=vector_store,
            llm_provider=llm_provider,
        )
    except Exception:
        vector_store.close()
        raise


_runtime: RAGRuntime | None = None
_runtime_lock = Lock()


def get_rag_runtime() -> RAGRuntime:
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = build_rag_runtime()
    return _runtime


def get_rag_service() -> RAGService:
    return get_rag_runtime().service


def close_rag_runtime() -> None:
    global _runtime
    with _runtime_lock:
        if _runtime is not None:
            _runtime.close()
            _runtime = None
