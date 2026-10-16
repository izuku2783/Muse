import pytest

from app.embeddings.hashing import HashingEmbeddingProvider
from app.knowledge.schemas import DocumentMetadata, DocumentType
from app.retrieval.semantic import SemanticRetriever
from app.vector_store.provider import VectorStore
from app.vector_store.schemas import VectorSearchFilter, VectorSearchResult


class RecordingVectorStore(VectorStore):
    def __init__(self, results=None):
        self.results = results or []
        self.search_calls = []

    def search(self, vector, embedding_model, embedding_version, limit=10, filters=None, score_threshold=None):
        self.search_calls.append(
            {
                "vector": vector,
                "embedding_model": embedding_model,
                "embedding_version": embedding_version,
                "limit": limit,
                "filters": filters,
                "score_threshold": score_threshold,
            }
        )
        return self.results

    def ensure_collection(self, *args, **kwargs): pass
    def upsert(self, records): pass
    def list_document_ids(self, embedding_model, embedding_version): return set()
    def delete(self, document_ids): pass
    def count(self): return len(self.results)


def match(document_id="track:1", score=0.91):
    return VectorSearchResult(
        document_id=document_id,
        score=score,
        content="Night Drive is a dark synthwave track.",
        metadata=DocumentMetadata(
            document_type=DocumentType.TRACK,
            entity_id=1,
            track_id=1,
            genres=("synthwave",),
        ),
        embedding_model="semantic-test",
        embedding_version="1",
    )


def test_semantic_retrieval_returns_ranked_entities_and_metadata():
    provider = HashingEmbeddingProvider(
        dimension=16, model_name="semantic-test", version="1"
    )
    store = RecordingVectorStore([match()])
    response = SemanticRetriever(provider, store).search(
        "  dark   atmospheric music ", limit=5, score_threshold=0.5
    )
    assert response.query == "dark atmospheric music"
    assert response.results[0].rank == 1
    assert response.results[0].entity_type == DocumentType.TRACK
    assert response.results[0].entity_id == 1
    assert response.results[0].score == 0.91
    assert response.results[0].metadata.genres == ("synthwave",)
    assert store.search_calls[0]["embedding_model"] == "semantic-test"
    assert store.search_calls[0]["limit"] == 5
    assert store.search_calls[0]["score_threshold"] == 0.5


def test_semantic_retrieval_forwards_structured_filters():
    provider = HashingEmbeddingProvider(dimension=16)
    store = RecordingVectorStore()
    filters = VectorSearchFilter(
        document_types=(DocumentType.ALBUM,),
        genres=("ambient",),
        release_year_min=2000,
    )
    SemanticRetriever(provider, store).search("ambient albums", filters=filters)
    assert store.search_calls[0]["filters"] == filters


@pytest.mark.parametrize("query", ["", " ", "\n\t"])
def test_semantic_retrieval_rejects_empty_query(query):
    with pytest.raises(ValueError, match="must not be empty"):
        SemanticRetriever(HashingEmbeddingProvider(), RecordingVectorStore()).search(query)


@pytest.mark.parametrize("limit", [0, 101])
def test_semantic_retrieval_validates_limit(limit):
    with pytest.raises(ValueError, match="between 1 and 100"):
        SemanticRetriever(HashingEmbeddingProvider(), RecordingVectorStore()).search(
            "music", limit=limit
        )


def test_semantic_retrieval_returns_empty_results_without_error():
    response = SemanticRetriever(
        HashingEmbeddingProvider(), RecordingVectorStore()
    ).search("unknown music")
    assert response.results == ()
