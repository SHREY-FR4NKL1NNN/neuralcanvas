"""Integration tests for the FastAPI app via httpx ASGITransport.

These avoid spawning a real training thread (covered by the live UI run) and mock
embeddings so no model is downloaded; CUDA/Ollama simply report unavailable in CI.
"""

import httpx
import numpy as np
from httpx import ASGITransport

import main
from embeddings import EmbeddingRouter


def _client():
    return httpx.AsyncClient(transport=ASGITransport(app=main.app), base_url="http://test")


async def test_health_shape():
    async with _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    for key in (
        "status",
        "cuda_available",
        "cuda_device",
        "vram_total_gb",
        "vram_free_gb",
        "sentence_transformers_loaded",
        "ollama_reachable",
    ):
        assert key in body


async def test_memory_flow(monkeypatch):
    monkeypatch.setattr(
        EmbeddingRouter, "embed", lambda self, texts: np.ones((len(texts), 8), dtype=np.float32)
    )
    monkeypatch.setattr(EmbeddingRouter, "get_dim", lambda self: 8)

    async with _client() as client:
        add = await client.post("/memory/add", json={"text": "hello", "session_id": "t1"})
        assert add.status_code == 200
        assert add.json()["embedding_dim"] == 8

        listed = await client.get("/memory/list?session_id=t1")
        assert listed.status_code == 200
        assert any(e["text"] == "hello" for e in listed.json()["entries"])

        switched = await client.post(
            "/embedding/switch", json={"backend": "ollama", "session_id": "t1"}
        )
        assert switched.status_code == 200

        cleared = await client.post("/memory/clear", json={"session_id": "t1"})
        assert cleared.status_code == 200


async def test_unknown_session_404():
    async with _client() as client:
        resp = await client.get("/session/state?session_id=does-not-exist")
    assert resp.status_code == 404
