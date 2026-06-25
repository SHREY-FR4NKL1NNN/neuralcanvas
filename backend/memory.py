"""A persistent in-process memory store for the memory-augmentation layer.

**Conceptual honesty.** This is a *simplified approximation* of memory-augmented
neural networks (MANNs). Real MANNs — Neural Turing Machines, the Differentiable
Neural Computer — use **differentiable** memory with learned read/write heads, so
gradients flow through the addressing mechanism and the network learns *how* to
use memory. NeuralCanvas instead uses plain **cosine retrieval** as a
*non-differentiable* approximation: text memories are embedded once, the most
similar entries to a query are retrieved, and a similarity-weighted average of
their embeddings is concatenated onto the network input. The memory therefore
*influences* what the network learns (observable in the loss curve and activation
patterns), but gradients do **not** flow back through the retrieval step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from log_config import get_logger

logger = get_logger("memory")

MAX_ENTRIES = 50


@dataclass
class MemoryEntry:
    """One stored memory: its text, embedding, timestamp, and retrieval count."""

    text: str
    embedding: np.ndarray
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    retrieval_count: int = 0

    def to_dict(self) -> dict:
        """Serialise for the API (omits the raw embedding for bandwidth)."""
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "retrieval_count": self.retrieval_count,
            "embedding_dim": int(self.embedding.shape[0]),
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors (0.0 if either is zero)."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class MemoryStore:
    """A bounded FIFO store of text memories with cosine-similarity retrieval."""

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        """Create an empty store holding at most ``max_entries`` memories."""
        self.entries: list[MemoryEntry] = []
        self._max = max_entries

    def add(self, text: str, embedding: np.ndarray) -> MemoryEntry:
        """Add a memory, evicting the oldest entry if the store is full (FIFO)."""
        entry = MemoryEntry(text=text, embedding=np.asarray(embedding, dtype=np.float32))
        self.entries.append(entry)
        if len(self.entries) > self._max:
            self.entries.pop(0)
        logger.info("Memory added (%d entries): %.60s", len(self.entries), text)
        return entry

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> list[MemoryEntry]:
        """Return the ``top_k`` entries most cosine-similar to ``query_embedding``.

        Increments each returned entry's ``retrieval_count``. Returns an empty
        list when the store is empty.
        """
        if not self.entries:
            return []
        scored = sorted(
            self.entries,
            key=lambda e: _cosine(query_embedding, e.embedding),
            reverse=True,
        )
        top = scored[: min(top_k, len(scored))]
        for entry in top:
            entry.retrieval_count += 1
        return top

    def retrieve_with_scores(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> list[tuple[MemoryEntry, float]]:
        """Like :meth:`retrieve` but also returns each entry's similarity score."""
        if not self.entries:
            return []
        pairs = [
            (e, _cosine(query_embedding, e.embedding)) for e in self.entries
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        top = pairs[: min(top_k, len(pairs))]
        for entry, _ in top:
            entry.retrieval_count += 1
        return top

    def get_context_vector(
        self, query_embedding: np.ndarray, top_k: int = 3
    ) -> np.ndarray:
        """Return the similarity-weighted average of the top-k entry embeddings.

        This vector is what gets concatenated onto the network input. Returns a
        zero vector of the query's dimension when the store is empty (so memory
        augmentation is a no-op until memories exist).
        """
        dim = int(np.asarray(query_embedding).shape[0])
        if not self.entries:
            return np.zeros(dim, dtype=np.float32)
        pairs = [
            (e.embedding, max(0.0, _cosine(query_embedding, e.embedding)))
            for e in self.entries
        ]
        pairs.sort(key=lambda p: p[1], reverse=True)
        top = pairs[: min(top_k, len(pairs))]
        total = sum(score for _, score in top)
        if total == 0.0:
            # All non-positive similarity — fall back to a plain average.
            return np.mean([emb for emb, _ in top], axis=0).astype(np.float32)
        weighted = sum(emb * score for emb, score in top) / total
        return np.asarray(weighted, dtype=np.float32)

    def get_all(self) -> list[dict]:
        """Return all entries serialised for the API (newest last)."""
        return [e.to_dict() for e in self.entries]

    def clear(self) -> None:
        """Remove every memory entry."""
        self.entries.clear()
        logger.info("Memory cleared")

    def get_retrieval_stats(self) -> dict:
        """Return ``{top_entry, avg_retrieval_count}`` over all entries."""
        if not self.entries:
            return {"top_entry": None, "avg_retrieval_count": 0.0}
        top = max(self.entries, key=lambda e: e.retrieval_count)
        avg = float(np.mean([e.retrieval_count for e in self.entries]))
        return {"top_entry": top.text, "avg_retrieval_count": round(avg, 2)}
