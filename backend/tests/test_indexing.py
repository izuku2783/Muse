from qdrant_client import QdrantClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.database.session import Base
from app.embeddings.hashing import HashingEmbeddingProvider
from app.indexing.service import IndexingService
from app.knowledge.schemas import DocumentMetadata, DocumentType, KnowledgeDocument
from app.knowledge.corpus import KnowledgeCorpusService
from app.models import Album, Artist, Genre, Track
from app.vector_store.errors import VectorStoreConfigurationError
from app.vector_store.qdrant import QdrantVectorStore


def document(entity_id, content=None):
    return KnowledgeDocument(
        document_id=f"track:{entity_id}",
        document_type=DocumentType.TRACK,
        content=content or f"Track {entity_id} is synthwave music.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=entity_id,
            track_id=entity_id,
            genres=("synthwave",),
        ),
    )


def make_service(version="1", batch_size=2):
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(client, "knowledge")
    provider = HashingEmbeddingProvider(
        dimension=16,
        model_name="test-hashing",
        version=version,
    )
    return IndexingService(provider, store, batch_size=batch_size), store


def test_indexing_pipeline_is_repeatable_and_idempotent():
    service, store = make_service()
    documents = [document(1), document(2), document(3)]
    try:
        first = service.index_documents(documents)
        second = service.index_documents(documents)
        assert first.document_count == 3
        assert second.deleted_count == 0
        assert store.count() == 3
        assert store.list_document_ids("test-hashing", "1") == {
            "track:1",
            "track:2",
            "track:3",
        }
    finally:
        store.close()


def test_reindex_updates_content_and_removes_stale_documents():
    service, store = make_service()
    try:
        service.index_documents([document(1), document(2)])
        summary = service.index_documents([document(1, "Updated track content.")])
        assert summary.deleted_count == 1
        assert store.count() == 1
        result = store.search(
            service.embedding_provider.embed_query("Updated track content."),
            "test-hashing",
            "1",
        )[0]
        assert result.document_id == "track:1"
        assert result.content == "Updated track content."
    finally:
        store.close()


def test_empty_reindex_removes_all_existing_documents():
    service, store = make_service()
    try:
        service.index_documents([document(1)])
        summary = service.index_documents([])
        assert summary.document_count == 0
        assert summary.deleted_count == 1
        assert store.count() == 0
    finally:
        store.close()


def test_embedding_version_change_requires_explicit_recreation():
    service, store = make_service(version="1")
    try:
        service.index_documents([document(1)])
        upgraded = IndexingService(
            HashingEmbeddingProvider(
                dimension=16,
                model_name="test-hashing",
                version="2",
            ),
            store,
        )
        try:
            upgraded.index_documents([document(1)])
            raise AssertionError("incompatible version should fail")
        except VectorStoreConfigurationError:
            pass
        summary = upgraded.index_documents([document(1)], recreate=True)
        assert summary.embedding_version == "2"
        assert store.count() == 1
        assert store.list_document_ids("test-hashing", "2") == {"track:1"}
    finally:
        store.close()


def test_indexing_rejects_invalid_batch_size():
    provider = HashingEmbeddingProvider(dimension=16)
    store = QdrantVectorStore(QdrantClient(":memory:"), "knowledge")
    try:
        try:
            IndexingService(provider, store, batch_size=0)
            raise AssertionError("invalid batch size should fail")
        except ValueError as exc:
            assert "positive" in str(exc)
    finally:
        store.close()


def test_indexing_rejects_duplicate_document_ids():
    service, store = make_service()
    try:
        try:
            service.index_documents([document(1), document(1)])
            raise AssertionError("duplicate document IDs should fail")
        except ValueError as exc:
            assert "unique" in str(exc)
    finally:
        store.close()


def test_database_to_document_to_qdrant_pipeline_is_idempotent():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    service, store = make_service()
    try:
        with Session(engine) as session:
            artist = Artist(
                provider="musicbrainz",
                provider_id="artist-1",
                name="Neon Waves",
            )
            album = Album(
                provider="musicbrainz",
                provider_id="album-1",
                title="After Dark",
                artist=artist,
            )
            genre = Genre(name="synthwave")
            track = Track(
                provider="musicbrainz",
                provider_id="track-1",
                title="Night Drive",
                artist=artist,
                album=album,
                genres=[genre],
            )
            session.add(track)
            session.commit()
            corpus = KnowledgeCorpusService(session)
            first = service.index_corpus(corpus)
            second = service.index_corpus(corpus)
            assert first.document_count == 4
            assert second.document_count == 4
            assert second.deleted_count == 0
            assert store.count() == 4
    finally:
        store.close()
