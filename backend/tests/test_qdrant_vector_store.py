import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from app.config.settings import Settings
from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.vector_store.errors import VectorStoreConfigurationError
from app.vector_store.factory import get_vector_store
from app.vector_store.qdrant import QdrantVectorStore
from app.vector_store.schemas import VectorRecord, VectorSearchFilter


@pytest.fixture
def store():
    client = QdrantClient(":memory:")
    vector_store = QdrantVectorStore(client, "test_knowledge")
    vector_store.ensure_collection(4, "test-model", "1")
    yield vector_store
    vector_store.close()


def record(
    document_id="track:1",
    vector=None,
    document_type=DocumentType.TRACK,
    entity_id=1,
    genres=("synthwave",),
    release_year=2020,
    provider="musicbrainz",
    version="1",
    artist_id=10,
    artist_name="Neon Waves",
    duration_ms=240_000,
    popularity=80,
    instrumental=True,
    country="US",
    album_type="Album",
):
    id_fields = {f"{document_type.value}_id": entity_id}
    return VectorRecord(
        document_id=document_id,
        vector=vector or [1.0, 0.0, 0.0, 0.0],
        content=f"Content for {document_id}",
        metadata=DocumentMetadata(
            document_type=document_type,
            entity_id=entity_id,
            artist_id=artist_id,
            artist_name=artist_name,
            genres=genres,
            release_year=release_year,
            provider=provider,
            duration_ms=duration_ms,
            popularity=popularity,
            instrumental=instrumental,
            country=country,
            album_type=album_type,
            **id_fields,
        ),
        embedding_model="test-model",
        embedding_version=version,
    )


def test_vector_record_rejects_non_finite_values():
    with pytest.raises(ValidationError, match="finite"):
        record(vector=[float("nan"), 0.0, 0.0, 0.0])


def test_vector_filter_rejects_inverted_year_range():
    with pytest.raises(ValidationError, match="cannot exceed"):
        VectorSearchFilter(release_year_min=2020, release_year_max=1990)


def test_upsert_is_idempotent_and_payload_is_searchable(store):
    item = record()
    store.upsert([item])
    store.upsert([item])
    assert store.count() == 1
    results = store.search([1.0, 0.0, 0.0, 0.0], "test-model", "1")
    assert results[0].document_id == "track:1"
    assert results[0].metadata.genres == ("synthwave",)
    assert results[0].embedding_version == "1"


def test_search_supports_metadata_filters(store):
    store.upsert(
        [
            record(),
            record(
                document_id="album:2",
                vector=[0.9, 0.1, 0.0, 0.0],
                document_type=DocumentType.ALBUM,
                entity_id=2,
                genres=("ambient",),
                release_year=1995,
                provider="local",
            ),
        ]
    )
    results = store.search(
        [1.0, 0.0, 0.0, 0.0],
        "test-model",
        "1",
        filters=VectorSearchFilter(
            document_types=(DocumentType.TRACK,),
            genres=("synthwave",),
            release_year_min=2010,
            provider="musicbrainz",
        ),
    )
    assert [result.document_id for result in results] == ["track:1"]
    assert store.search(
        [1.0, 0.0, 0.0, 0.0],
        "test-model",
        "1",
        filters=VectorSearchFilter(release_year_max=1999),
    )[0].document_id == "album:2"


def test_search_supports_expanded_stage_7_filters(store):
    store.upsert(
        [
            record(),
            record(
                document_id="track:2",
                entity_id=2,
                vector=[0.9, 0.1, 0.0, 0.0],
                artist_id=20,
                artist_name="Other Artist",
                duration_ms=360_000,
                popularity=40,
                instrumental=False,
                country="GB",
                album_type="Single",
            ),
        ]
    )
    expanded = VectorSearchFilter(
        duration_ms_max=300_000,
        popularity_min=70,
        instrumental=True,
        countries=("US",),
        album_types=("ALBUM",),
    )
    results = store.search(
        [1.0, 0.0, 0.0, 0.0], "test-model", "1", filters=expanded
    )
    assert [result.document_id for result in results] == ["track:1"]

    by_id = store.search(
        [1.0, 0.0, 0.0, 0.0],
        "test-model",
        "1",
        filters=VectorSearchFilter(exclude_artist_ids=(10,)),
    )
    assert [result.document_id for result in by_id] == ["track:2"]
    by_name = store.search(
        [1.0, 0.0, 0.0, 0.0],
        "test-model",
        "1",
        filters=VectorSearchFilter(exclude_artists=("NEON WAVES",)),
    )
    assert [result.document_id for result in by_name] == ["track:2"]
    non_instrumental = store.search(
        [1.0, 0.0, 0.0, 0.0],
        "test-model",
        "1",
        filters=VectorSearchFilter(instrumental=False),
    )
    assert [result.document_id for result in non_instrumental] == ["track:2"]


def test_delete_and_list_document_ids(store):
    store.upsert([record(), record(document_id="track:2", entity_id=2)])
    assert store.list_document_ids("test-model", "1") == {"track:1", "track:2"}
    store.delete(["track:1"])
    assert store.list_document_ids("test-model", "1") == {"track:2"}


@pytest.mark.parametrize(
    ("size", "model", "version"),
    [(8, "test-model", "1"), (4, "other-model", "1"), (4, "test-model", "2")],
)
def test_collection_rejects_incompatible_embedding_configuration(
    store, size, model, version
):
    with pytest.raises(VectorStoreConfigurationError, match="does not match"):
        store.ensure_collection(size, model, version)


def test_explicit_recreation_allows_embedding_version_change(store):
    store.upsert([record()])
    store.ensure_collection(8, "new-model", "2", recreate=True)
    assert store.count() == 0
    info = store.client.get_collection(store.collection_name)
    assert info.config.metadata["embedding_model"] == "new-model"
    assert info.config.metadata["embedding_version"] == "2"


def test_upsert_rejects_mixed_embedding_versions(store):
    with pytest.raises(VectorStoreConfigurationError, match="cannot mix"):
        store.upsert([record(), record(document_id="track:2", entity_id=2, version="2")])


def test_vector_store_factory_supports_in_memory_mode():
    settings = Settings(
        _env_file=None,
        qdrant_url=":memory:",
        qdrant_collection="factory_test",
    )
    vector_store = get_vector_store(settings)
    try:
        vector_store.ensure_collection(4, "test-model", "1")
        assert vector_store.count() == 0
    finally:
        vector_store.close()
