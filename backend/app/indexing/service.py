from dataclasses import dataclass

from app.embeddings.provider import EmbeddingProvider
from app.knowledge.corpus import KnowledgeCorpusService
from app.knowledge.schemas import KnowledgeDocument
from app.vector_store.provider import VectorStore
from app.vector_store.schemas import VectorRecord


@dataclass(frozen=True)
class IndexingSummary:
    document_count: int
    deleted_count: int
    embedding_model: str
    embedding_version: str
    vector_dimension: int


class IndexingService:
    """Turn deterministic knowledge documents into versioned vector records."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        batch_size: int = 64,
    ):
        if batch_size < 1:
            raise ValueError("Indexing batch size must be positive.")
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.batch_size = batch_size

    def index_corpus(
        self,
        corpus: KnowledgeCorpusService,
        recreate: bool = False,
    ) -> IndexingSummary:
        return self.index_documents(corpus.build_documents(), recreate=recreate)

    def index_documents(
        self,
        documents: list[KnowledgeDocument],
        recreate: bool = False,
    ) -> IndexingSummary:
        provider = self.embedding_provider
        document_ids = [document.document_id for document in documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Knowledge documents must have unique document IDs.")
        self.vector_store.ensure_collection(
            vector_size=provider.dimension,
            embedding_model=provider.model_name,
            embedding_version=provider.version,
            recreate=recreate,
        )
        existing_ids = self.vector_store.list_document_ids(
            provider.model_name, provider.version
        )
        records: list[VectorRecord] = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start : start + self.batch_size]
            vectors = provider.embed_documents(
                [document.content for document in batch]
            )
            if len(vectors) != len(batch):
                raise ValueError(
                    "Embedding provider returned a different number of vectors."
                )
            if any(len(vector) != provider.dimension for vector in vectors):
                raise ValueError(
                    "Embedding provider returned an unexpected vector dimension."
                )
            records.extend(
                VectorRecord(
                    document_id=document.document_id,
                    vector=vector,
                    content=document.content,
                    metadata=document.metadata,
                    embedding_model=provider.model_name,
                    embedding_version=provider.version,
                )
                for document, vector in zip(batch, vectors)
            )

        for start in range(0, len(records), self.batch_size):
            self.vector_store.upsert(records[start : start + self.batch_size])

        current_ids = {document.document_id for document in documents}
        stale_ids = sorted(existing_ids - current_ids)
        self.vector_store.delete(stale_ids)
        return IndexingSummary(
            document_count=len(documents),
            deleted_count=len(stale_ids),
            embedding_model=provider.model_name,
            embedding_version=provider.version,
            vector_dimension=provider.dimension,
        )
