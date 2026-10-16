from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding

from app.embeddings.provider import EmbeddingProvider


class FastEmbedEmbeddingProvider(EmbeddingProvider):
    """Local learned text embeddings powered by FastEmbed and ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        version: str = "fastembed-0.8.1-query-passage-v1",
        cache_dir: str | Path | None = "data/models",
        expected_dimension: int | None = None,
        model: Any | None = None,
        dimension: int | None = None,
    ):
        self._model_name = model_name.strip()
        self._version = version.strip()
        if not self._model_name or not self._version:
            raise ValueError("Embedding model name and version are required.")
        supported_dimension = dimension or self._find_dimension(self._model_name)
        if expected_dimension is not None and expected_dimension != supported_dimension:
            raise ValueError(
                f"Configured dimension {expected_dimension} does not match model "
                f"dimension {supported_dimension}."
            )
        self._dimension = supported_dimension
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = TextEmbedding(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                lazy_load=True,
            )
        return self._model

    @staticmethod
    def _find_dimension(model_name: str) -> int:
        match = next(
            (
                item
                for item in TextEmbedding.list_supported_models()
                if item["model"] == model_name
            ),
            None,
        )
        if not match:
            raise ValueError(f"Unsupported FastEmbed model: {model_name}")
        return int(match["dim"])

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def dimension(self) -> int:
        return self._dimension

    @staticmethod
    def _validate_texts(texts: Sequence[str]) -> None:
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text.")

    @staticmethod
    def _to_lists(vectors: Iterable[Any]) -> list[list[float]]:
        return [
            vector.astype(float).tolist()
            if hasattr(vector, "astype")
            else [float(value) for value in vector]
            for vector in vectors
        ]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self._validate_texts(texts)
        if not texts:
            return []
        return self._to_lists(self._get_model().passage_embed(list(texts)))

    def embed_query(self, text: str) -> list[float]:
        self._validate_texts([text])
        return self._to_lists(self._get_model().query_embed([text]))[0]
