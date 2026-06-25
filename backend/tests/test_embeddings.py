"""Tests for the embedding backends and router (models/HTTP mocked)."""

import httpx
import numpy as np
import pytest

from embeddings import (
    EmbeddingRouter,
    OllamaEmbeddingBackend,
    SentenceTransformerBackend,
)


def test_sentence_transformer_caches(monkeypatch):
    backend = SentenceTransformerBackend()

    class FakeModel:
        def __init__(self):
            self.calls = 0

        def encode(self, texts, **kwargs):
            self.calls += len(texts)
            return np.ones((len(texts), 384), dtype=np.float32)

    fake = FakeModel()
    monkeypatch.setattr(backend, "_ensure_model", lambda: setattr(backend, "_model", fake))

    v1 = backend.embed(["hello"])
    v2 = backend.embed(["hello"])  # cached
    assert v1.shape == (1, 384)
    assert np.array_equal(v1, v2)
    assert fake.calls == 1  # second call served from cache


def test_router_dim_and_switch():
    router = EmbeddingRouter()
    assert router.get_dim() == 384
    router.switch("ollama")
    assert router.get_dim() == 4096
    with pytest.raises(ValueError):
        router.switch("bogus")


def test_ollama_backend_degrades_on_error(monkeypatch):
    backend = OllamaEmbeddingBackend()

    def boom(*args, **kwargs):
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(httpx.Client, "post", boom)
    v = backend.embed(["x"])
    assert v.shape == (1, 4096)
    assert np.all(v == 0)  # zero vector fallback, no crash
