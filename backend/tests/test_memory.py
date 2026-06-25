"""Tests for the cosine-retrieval memory store."""

import numpy as np

from memory import MemoryStore


def test_add_and_fifo_eviction():
    store = MemoryStore(max_entries=3)
    for i in range(4):
        store.add(f"t{i}", np.full(4, float(i), dtype=np.float32))
    assert len(store.entries) == 3
    assert store.entries[0].text == "t1"  # t0 evicted first


def test_retrieve_ranks_by_cosine():
    store = MemoryStore()
    store.add("a", np.array([1, 0, 0, 0], dtype=np.float32))
    store.add("b", np.array([0, 1, 0, 0], dtype=np.float32))
    top = store.retrieve(np.array([1, 0.1, 0, 0], dtype=np.float32), top_k=1)
    assert top[0].text == "a"
    assert top[0].retrieval_count == 1


def test_context_vector_empty_is_zero():
    store = MemoryStore()
    v = store.get_context_vector(np.zeros(8, dtype=np.float32))
    assert v.shape == (8,)
    assert np.all(v == 0)


def test_context_vector_weighted_toward_similar():
    store = MemoryStore()
    store.add("a", np.array([1, 0], dtype=np.float32))
    store.add("b", np.array([0, 1], dtype=np.float32))
    v = store.get_context_vector(np.array([1, 0], dtype=np.float32), top_k=2)
    assert v[0] > v[1]


def test_retrieval_stats():
    store = MemoryStore()
    store.add("a", np.ones(2, dtype=np.float32))
    store.retrieve(np.ones(2, dtype=np.float32))
    stats = store.get_retrieval_stats()
    assert stats["top_entry"] == "a"
    assert stats["avg_retrieval_count"] >= 1.0


def test_clear():
    store = MemoryStore()
    store.add("a", np.ones(2, dtype=np.float32))
    store.clear()
    assert store.entries == []
