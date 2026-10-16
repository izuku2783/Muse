import hashlib
import math
import re
import unicodedata

from app.embeddings.provider import EmbeddingProvider


_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


class HashingEmbeddingProvider(EmbeddingProvider):
    """Small deterministic baseline embedder for development and pipeline tests.

    It uses signed feature hashing over tokens and adjacent token pairs. It is not
    intended to replace a learned semantic model for Stage 5 quality evaluation.
    """

    def __init__(
        self,
        dimension: int = 128,
        model_name: str = "muse-hashing",
        version: str = "1",
    ):
        if dimension < 8:
            raise ValueError("Embedding dimension must be at least 8.")
        if not model_name.strip() or not version.strip():
            raise ValueError("Embedding model name and version are required.")
        self._dimension = dimension
        self._model_name = model_name.strip()
        self._version = version.strip()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def version(self) -> str:
        return self._version

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens = _TOKEN_PATTERN.findall(normalized)
        if not tokens:
            raise ValueError("Cannot embed empty text.")
        features = tokens + [
            f"{first}\u241f{second}" for first, second in zip(tokens, tokens[1:])
        ]
        vector = [0.0] * self.dimension
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector]
