import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.reranking.provider import Reranker


class FastEmbedCrossEncoderReranker(Reranker):
    """Local cross-encoder scoring through FastEmbed and ONNX Runtime."""

    def __init__(
        self,
        model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2",
        version: str = "fastembed-0.8.1",
        cache_dir: str | Path | None = "data/models",
        batch_size: int = 32,
        model: Any | None = None,
    ):
        self._model_name = model_name.strip()
        self._version = version.strip()
        if not self._model_name or not self._version:
            raise ValueError("Reranker model name and version are required.")
        if batch_size < 1:
            raise ValueError("Reranker batch size must be positive.")
        supported = {
            item["model"].casefold()
            for item in TextCrossEncoder.list_supported_models()
        }
        if model is None and self._model_name.casefold() not in supported:
            raise ValueError(f"Unsupported FastEmbed reranker: {self._model_name}")
        self._cache_dir = str(cache_dir) if cache_dir is not None else None
        self._batch_size = batch_size
        self._model = model

    def _get_model(self):
        if self._model is None:
            self._model = TextCrossEncoder(
                model_name=self._model_name,
                cache_dir=self._cache_dir,
                lazy_load=True,
            )
        return self._model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        cleaned_query = " ".join(query.split())
        if not cleaned_query:
            raise ValueError("Reranking query must not be empty.")
        if any(not document.strip() for document in documents):
            raise ValueError("Reranking documents must not be empty.")
        if not documents:
            return []
        scores = [
            float(score)
            for score in self._get_model().rerank(
                cleaned_query,
                list(documents),
                batch_size=self._batch_size,
            )
        ]
        if len(scores) != len(documents):
            raise RuntimeError("Reranker returned an unexpected number of scores.")
        if not all(math.isfinite(score) for score in scores):
            raise RuntimeError("Reranker returned a non-finite score.")
        return scores
