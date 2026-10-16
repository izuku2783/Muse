from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.knowledge.schemas import DocumentMetadata
from app.vector_store.errors import VectorStoreConfigurationError
from app.vector_store.provider import VectorStore
from app.vector_store.schemas import VectorRecord, VectorSearchFilter, VectorSearchResult


class QdrantVectorStore(VectorStore):
    def __init__(self, client: QdrantClient, collection_name: str):
        if not collection_name.strip():
            raise ValueError("Qdrant collection name is required.")
        self.client = client
        self.collection_name = collection_name

    @staticmethod
    def _point_id(document_id: str) -> str:
        return str(uuid5(NAMESPACE_URL, f"muse://knowledge/{document_id}"))

    @staticmethod
    def _collection_metadata(
        vector_size: int, embedding_model: str, embedding_version: str
    ) -> dict[str, str | int]:
        return {
            "embedding_model": embedding_model,
            "embedding_version": embedding_version,
            "embedding_dimension": vector_size,
        }

    @staticmethod
    def _filter_facets(
        metadata: DocumentMetadata,
    ) -> dict[str, str | list[str] | None]:
        return {
            "genres": [genre.casefold() for genre in metadata.genres],
            "artist_name": (
                metadata.artist_name.casefold() if metadata.artist_name else None
            ),
            "provider": metadata.provider.casefold() if metadata.provider else None,
            "country": metadata.country.casefold() if metadata.country else None,
            "album_type": (
                metadata.album_type.casefold() if metadata.album_type else None
            ),
        }

    def ensure_collection(
        self,
        vector_size: int,
        embedding_model: str,
        embedding_version: str,
        recreate: bool = False,
    ) -> None:
        exists = self.client.collection_exists(self.collection_name)
        if exists and recreate:
            self.client.delete_collection(self.collection_name)
            exists = False
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                metadata=self._collection_metadata(
                    vector_size, embedding_model, embedding_version
                ),
            )
            return
        self._validate_collection(vector_size, embedding_model, embedding_version)

    def _validate_collection(
        self, vector_size: int, embedding_model: str, embedding_version: str
    ) -> None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        actual_size = getattr(vectors, "size", None)
        actual_metadata = info.config.metadata or {}
        expected_metadata = self._collection_metadata(
            vector_size, embedding_model, embedding_version
        )
        if actual_size != vector_size or any(
            actual_metadata.get(key) != value
            for key, value in expected_metadata.items()
        ):
            raise VectorStoreConfigurationError(
                "Qdrant collection embedding configuration does not match. "
                "Use explicit recreation to reindex with a new model or dimension."
            )

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        first = records[0]
        vector_size = len(first.vector)
        self._validate_collection(
            vector_size, first.embedding_model, first.embedding_version
        )
        for record in records:
            if (
                len(record.vector) != vector_size
                or record.embedding_model != first.embedding_model
                or record.embedding_version != first.embedding_version
            ):
                raise VectorStoreConfigurationError(
                    "A batch cannot mix embedding models, versions, or dimensions."
                )
        points = [
            models.PointStruct(
                id=self._point_id(record.document_id),
                vector=record.vector,
                payload={
                    "document_id": record.document_id,
                    "document_type": record.metadata.document_type.value,
                    "entity_id": record.metadata.entity_id,
                    "content": record.content,
                    "metadata": record.metadata.model_dump(
                        mode="json", exclude_none=True
                    ),
                    "filter_facets": self._filter_facets(record.metadata),
                    "embedding_model": record.embedding_model,
                    "embedding_version": record.embedding_version,
                },
            )
            for record in records
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )

    @staticmethod
    def _build_filter(
        embedding_model: str,
        embedding_version: str,
        filters: VectorSearchFilter | None = None,
    ) -> models.Filter:
        must: list[models.Condition] = [
            models.FieldCondition(
                key="embedding_model",
                match=models.MatchValue(value=embedding_model),
            ),
            models.FieldCondition(
                key="embedding_version",
                match=models.MatchValue(value=embedding_version),
            ),
        ]
        must_not: list[models.Condition] = []
        if filters:
            if filters.document_types:
                must.append(
                    models.FieldCondition(
                        key="document_type",
                        match=models.MatchAny(
                            any=[item.value for item in filters.document_types]
                        ),
                    )
                )
            if filters.entity_ids:
                must.append(
                    models.FieldCondition(
                        key="entity_id",
                        match=models.MatchAny(any=list(filters.entity_ids)),
                    )
                )
            if filters.genres:
                must.append(
                    models.FieldCondition(
                        key="filter_facets.genres",
                        match=models.MatchAny(any=list(filters.genres)),
                    )
                )
            if filters.provider:
                must.append(
                    models.FieldCondition(
                        key="filter_facets.provider",
                        match=models.MatchValue(value=filters.provider),
                    )
                )
            if filters.exclude_artist_ids:
                must_not.append(
                    models.FieldCondition(
                        key="metadata.artist_id",
                        match=models.MatchAny(any=list(filters.exclude_artist_ids)),
                    )
                )
            if filters.exclude_artists:
                must_not.append(
                    models.FieldCondition(
                        key="filter_facets.artist_name",
                        match=models.MatchAny(any=list(filters.exclude_artists)),
                    )
                )
            if (
                filters.release_year_min is not None
                or filters.release_year_max is not None
            ):
                must.append(
                    models.FieldCondition(
                        key="metadata.release_year",
                        range=models.Range(
                            gte=filters.release_year_min,
                            lte=filters.release_year_max,
                        ),
                    )
                )
            if filters.duration_ms_min is not None or filters.duration_ms_max is not None:
                must.append(
                    models.FieldCondition(
                        key="metadata.duration_ms",
                        range=models.Range(
                            gte=filters.duration_ms_min,
                            lte=filters.duration_ms_max,
                        ),
                    )
                )
            if filters.popularity_min is not None or filters.popularity_max is not None:
                must.append(
                    models.FieldCondition(
                        key="metadata.popularity",
                        range=models.Range(
                            gte=filters.popularity_min,
                            lte=filters.popularity_max,
                        ),
                    )
                )
            if filters.instrumental is not None:
                must.append(
                    models.FieldCondition(
                        key="metadata.instrumental",
                        match=models.MatchValue(value=filters.instrumental),
                    )
                )
            if filters.countries:
                must.append(
                    models.FieldCondition(
                        key="filter_facets.country",
                        match=models.MatchAny(any=list(filters.countries)),
                    )
                )
            if filters.album_types:
                must.append(
                    models.FieldCondition(
                        key="filter_facets.album_type",
                        match=models.MatchAny(any=list(filters.album_types)),
                    )
                )
        return models.Filter(must=must, must_not=must_not or None)

    def search(
        self,
        vector: list[float],
        embedding_model: str,
        embedding_version: str,
        limit: int = 10,
        filters: VectorSearchFilter | None = None,
        score_threshold: float | None = None,
    ) -> list[VectorSearchResult]:
        self._validate_collection(len(vector), embedding_model, embedding_version)
        if limit <= 0:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            query_filter=self._build_filter(
                embedding_model, embedding_version, filters
            ),
            limit=min(limit, 100),
            with_payload=True,
            score_threshold=score_threshold,
        )
        results = []
        for point in response.points:
            payload = point.payload or {}
            results.append(
                VectorSearchResult(
                    document_id=payload["document_id"],
                    score=point.score,
                    content=payload["content"],
                    metadata=payload["metadata"],
                    embedding_model=payload["embedding_model"],
                    embedding_version=payload["embedding_version"],
                )
            )
        return results

    def list_document_ids(
        self, embedding_model: str, embedding_version: str
    ) -> set[str]:
        document_ids: set[str] = set()
        offset = None
        query_filter = self._build_filter(embedding_model, embedding_version)
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=query_filter,
                limit=256,
                offset=offset,
                with_payload=["document_id"],
                with_vectors=False,
            )
            document_ids.update(
                record.payload["document_id"]
                for record in records
                if record.payload and record.payload.get("document_id")
            )
            if offset is None:
                return document_ids

    def delete(self, document_ids: Sequence[str]) -> None:
        if not document_ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=[self._point_id(document_id) for document_id in document_ids]
            ),
            wait=True,
        )

    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def close(self) -> None:
        self.client.close()
