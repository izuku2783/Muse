from qdrant_client import QdrantClient

from app.config.settings import Settings, get_settings
from app.vector_store.qdrant import QdrantVectorStore


def get_vector_store(settings: Settings | None = None) -> QdrantVectorStore:
    configured = settings or get_settings()
    if configured.qdrant_url == ":memory:":
        client = QdrantClient(":memory:")
    else:
        client = QdrantClient(
            url=configured.qdrant_url,
            api_key=configured.qdrant_api_key,
        )
    return QdrantVectorStore(client, configured.qdrant_collection)
