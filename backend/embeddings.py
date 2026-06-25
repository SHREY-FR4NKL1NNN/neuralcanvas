"""Text-embedding backends for the memory layer, behind a common interface.

Two backends are supported and hot-swappable via :class:`EmbeddingRouter`:

* :class:`SentenceTransformerBackend` — local ``all-MiniLM-L6-v2`` (384-dim),
  small and fast, loaded lazily on first use.
* :class:`OllamaEmbeddingBackend` — Ollama's ``mistral`` embeddings (4096-dim),
  reached over HTTP, with graceful handling when Ollama is unreachable.

Both cache embeddings by a hash of the input text. Switching backends changes the
embedding dimension, which means the network's memory-augmented input layer must
be rebuilt — the trainer handles that on the next training start.
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

import httpx
import numpy as np

from log_config import get_logger

logger = get_logger("embeddings")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


def _text_hash(text: str) -> str:
    """Return a stable hex digest for caching an embedding by its text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class EmbeddingBackend(ABC):
    """Common interface: turn a list of strings into an ``(n, dim)`` array."""

    dim: int

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` and return a ``(len(texts), dim)`` float32 array."""

    @abstractmethod
    def get_dim(self) -> int:
        """Return the embedding dimension this backend produces."""


class SentenceTransformerBackend(EmbeddingBackend):
    """Local sentence-transformers backend (``all-MiniLM-L6-v2``, 384-dim)."""

    MODEL_NAME = "all-MiniLM-L6-v2"
    dim = 384

    def __init__(self) -> None:
        """Create the backend; the model is loaded lazily on first ``embed``."""
        self._model = None
        self._cache: dict[str, np.ndarray] = {}

    @property
    def loaded(self) -> bool:
        """Whether the underlying model has been loaded into memory yet."""
        return self._model is not None

    def _ensure_model(self) -> None:
        """Load the sentence-transformers model on first use."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy import

            logger.info("Loading sentence-transformers model %s", self.MODEL_NAME)
            self._model = SentenceTransformer(self.MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` (cached per text), loading the model if needed."""
        self._ensure_model()
        missing = [t for t in texts if _text_hash(t) not in self._cache]
        if missing:
            vecs = self._model.encode(
                missing, convert_to_numpy=True, normalize_embeddings=False
            )
            for t, v in zip(missing, vecs):
                self._cache[_text_hash(t)] = v.astype(np.float32)
        return np.stack([self._cache[_text_hash(t)] for t in texts]).astype(np.float32)

    def get_dim(self) -> int:
        """Return 384 (the ``all-MiniLM-L6-v2`` dimension)."""
        return self.dim


class OllamaEmbeddingBackend(EmbeddingBackend):
    """Ollama embeddings backend using the ``mistral`` model (4096-dim)."""

    MODEL_NAME = "mistral"
    dim = 4096

    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        """Create the backend pointed at ``base_url`` (Ollama's API root)."""
        self._url = f"{base_url}/api/embeddings"
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` one request at a time, caching and degrading gracefully.

        On a connection error or unexpected response a zero vector of the right
        dimension is returned for that text (and logged) so the training loop is
        never crashed by an unreachable Ollama.
        """
        out: list[np.ndarray] = []
        with httpx.Client(timeout=30.0) as client:
            for text in texts:
                key = _text_hash(text)
                if key in self._cache:
                    out.append(self._cache[key])
                    continue
                try:
                    resp = client.post(
                        self._url, json={"model": self.MODEL_NAME, "prompt": text}
                    )
                    resp.raise_for_status()
                    vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
                    self.dim = int(vec.shape[0])
                except (httpx.HTTPError, KeyError, ValueError) as exc:
                    logger.warning("Ollama embedding failed, using zero vector: %s", exc)
                    vec = np.zeros(self.dim, dtype=np.float32)
                self._cache[key] = vec
                out.append(vec)
        return np.stack(out).astype(np.float32)

    def get_dim(self) -> int:
        """Return the embedding dimension (4096 for ``mistral``)."""
        return self.dim


class EmbeddingRouter:
    """Holds both backends and routes to the currently selected one."""

    def __init__(self, backend: str = "sentence-transformers") -> None:
        """Create the router with both backends; ``backend`` is the active one."""
        self._sentence = SentenceTransformerBackend()
        self._ollama = OllamaEmbeddingBackend()
        self.current_backend = backend

    def _active(self) -> EmbeddingBackend:
        """Return the currently selected backend instance."""
        return self._ollama if self.current_backend == "ollama" else self._sentence

    def switch(self, backend: str) -> None:
        """Switch the active backend ("sentence-transformers" | "ollama")."""
        if backend not in ("sentence-transformers", "ollama"):
            raise ValueError(f"Unknown embedding backend: {backend}")
        self.current_backend = backend
        logger.info("Embedding backend switched to %s", backend)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` with the active backend."""
        return self._active().embed(texts)

    def get_dim(self) -> int:
        """Return the active backend's embedding dimension."""
        return self._active().get_dim()

    @property
    def sentence_transformers_loaded(self) -> bool:
        """Whether the sentence-transformers model has been loaded."""
        return self._sentence.loaded
