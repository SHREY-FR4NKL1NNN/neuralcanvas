"""FastAPI application for NeuralCanvas.

Exposes session lifecycle, live SSE event streaming, the memory layer, and a
health probe. A *session* bundles an event streamer, an embedding router, a
memory store, and (once started) a trainer; the frontend generates a session id
on load and reuses it across calls so memories added before training share the
same session.
"""

from __future__ import annotations

import asyncio
import os
import uuid

from dotenv import load_dotenv

load_dotenv()

import httpx  # noqa: E402
import torch  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import StreamingResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
from starlette.middleware.base import BaseHTTPMiddleware  # noqa: E402

from embeddings import EmbeddingRouter  # noqa: E402
from log_config import get_logger  # noqa: E402
from memory import MemoryStore  # noqa: E402
from network import NetworkConfig  # noqa: E402
from streamer import EventStreamer  # noqa: E402
from trainer import Trainer, TrainingConfig  # noqa: E402

logger = get_logger("api")

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")


# ---- session registry ------------------------------------------------------
class Session:
    """One observatory session: streamer + embeddings + memory + (a) trainer."""

    def __init__(self, session_id: str) -> None:
        """Create an empty session with its own streamer/embeddings/memory."""
        self.id = session_id
        self.streamer = EventStreamer()
        self.embeddings = EmbeddingRouter()
        self.memory = MemoryStore()
        self.trainer: Trainer | None = None


SESSIONS: dict[str, Session] = {}


def get_or_create_session(session_id: str | None) -> Session:
    """Return the session for ``session_id``, creating it if necessary."""
    sid = session_id or uuid.uuid4().hex
    if sid not in SESSIONS:
        SESSIONS[sid] = Session(sid)
        logger.info("Session created: %s", sid)
    return SESSIONS[sid]


def require_session(session_id: str) -> Session:
    """Return an existing session or raise 404."""
    session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return session


# ---- request/response models ----------------------------------------------
class NetworkConfigModel(BaseModel):
    """Network shape (input/output sizes are re-derived from the dataset)."""

    input_size: int = 2
    hidden_layers: list[int] = Field(default_factory=lambda: [16, 8])
    output_size: int = 1
    activation: str = "relu"
    dropout: float = 0.0
    memory_dim: int = 0


class StartRequest(BaseModel):
    """Body for ``POST /session/start`` — a full training configuration."""

    network_config: NetworkConfigModel
    dataset: str = "xor"
    dataset_params: dict = Field(default_factory=dict)
    optimizer: str = "adam"
    learning_rate: float = 0.01
    epochs: int = 20
    batch_size: int = 32
    loss_fn: str = "cross_entropy"
    use_memory: bool = False
    memory_backend: str = "sentence-transformers"
    session_id: str | None = None


class SessionRequest(BaseModel):
    """Body for pause/resume/stop/clear — identifies the session."""

    session_id: str


class MemoryAddRequest(BaseModel):
    """Body for ``POST /memory/add``."""

    text: str = Field(..., min_length=1)
    session_id: str


class EmbeddingSwitchRequest(BaseModel):
    """Body for ``POST /embedding/switch``."""

    backend: str
    session_id: str


# ---- app + middleware ------------------------------------------------------
app = FastAPI(title="NeuralCanvas", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_origin_regex=r"https://([a-z0-9-]+\.)*(vercel\.app|ngrok-free\.(app|dev))",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class NgrokHeaderMiddleware(BaseHTTPMiddleware):
    """Echo ``ngrok-skip-browser-warning`` so ngrok doesn't serve its interstitial."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["ngrok-skip-browser-warning"] = "true"
        return response


app.add_middleware(NgrokHeaderMiddleware)


# ---- helpers ---------------------------------------------------------------
def _to_training_config(req: StartRequest) -> TrainingConfig:
    """Convert the request model into the trainer's dataclass config."""
    nc = req.network_config
    net_cfg = NetworkConfig(
        input_size=nc.input_size,
        hidden_layers=list(nc.hidden_layers),
        output_size=nc.output_size,
        activation=nc.activation,
        dropout=nc.dropout,
        memory_dim=nc.memory_dim,
    )
    return TrainingConfig(
        network_config=net_cfg,
        dataset=req.dataset,
        dataset_params=req.dataset_params,
        optimizer=req.optimizer,
        learning_rate=req.learning_rate,
        epochs=req.epochs,
        batch_size=req.batch_size,
        loss_fn=req.loss_fn,
        use_memory=req.use_memory,
        memory_backend=req.memory_backend,
    )


# ---- session endpoints -----------------------------------------------------
@app.post("/session/start")
async def session_start(req: StartRequest) -> dict:
    """Create (or reuse) a session, build a trainer, and start training."""
    session = get_or_create_session(req.session_id)
    session.streamer.bind_loop(asyncio.get_running_loop())
    config = _to_training_config(req)
    if config.use_memory:
        session.embeddings.switch(config.memory_backend)
    session.trainer = Trainer(
        config, session.streamer, session.embeddings, session.memory
    )
    session.trainer.start()
    logger.info("Session started: %s (dataset=%s)", session.id, config.dataset)
    return {"session_id": session.id}


@app.post("/session/pause")
async def session_pause(req: SessionRequest) -> dict:
    """Pause the session's training run."""
    session = require_session(req.session_id)
    if session.trainer:
        session.trainer.pause()
    return {"status": "paused"}


@app.post("/session/resume")
async def session_resume(req: SessionRequest) -> dict:
    """Resume a paused training run."""
    session = require_session(req.session_id)
    if session.trainer:
        session.trainer.resume()
    return {"status": "training"}


@app.post("/session/stop")
async def session_stop(req: SessionRequest) -> dict:
    """Stop the session's training run."""
    session = require_session(req.session_id)
    if session.trainer:
        session.trainer.stop()
    return {"status": "stopped"}


@app.get("/session/state")
async def session_state(session_id: str) -> dict:
    """Return the current trainer state for ``session_id``."""
    session = require_session(session_id)
    if session.trainer is None:
        return {"status": "idle", "epoch": 0, "step": 0}
    from dataclasses import asdict

    return asdict(session.trainer.get_state())


# ---- event endpoints -------------------------------------------------------
@app.get("/events/stream")
async def events_stream(session_id: str) -> StreamingResponse:
    """Stream this session's training events as Server-Sent Events."""
    session = require_session(session_id)
    session.streamer.bind_loop(asyncio.get_running_loop())
    return StreamingResponse(
        session.streamer.subscribe(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/events/history")
async def events_history(session_id: str) -> dict:
    """Return the last 500 buffered events (for reconnecting clients)."""
    session = require_session(session_id)
    return {"events": session.streamer.get_history()}


# ---- memory endpoints ------------------------------------------------------
@app.post("/memory/add")
async def memory_add(req: MemoryAddRequest) -> dict:
    """Embed ``text`` and add it to the session's memory store."""
    session = get_or_create_session(req.session_id)
    try:
        embedding = session.embeddings.embed([req.text])[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Embedding failed")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}") from exc
    entry = session.memory.add(req.text, embedding)
    return {"entry": entry.to_dict(), "embedding_dim": int(embedding.shape[0])}


@app.post("/memory/clear")
async def memory_clear(req: SessionRequest) -> dict:
    """Clear the session's memory store."""
    session = get_or_create_session(req.session_id)
    session.memory.clear()
    return {"cleared": True}


@app.get("/memory/list")
async def memory_list(session_id: str) -> dict:
    """List the session's memory entries."""
    session = get_or_create_session(session_id)
    return {
        "entries": session.memory.get_all(),
        "stats": session.memory.get_retrieval_stats(),
    }


@app.post("/embedding/switch")
async def embedding_switch(req: EmbeddingSwitchRequest) -> dict:
    """Switch the session's embedding backend (changes the memory dimension)."""
    session = get_or_create_session(req.session_id)
    old_dim = session.embeddings.get_dim()
    try:
        session.embeddings.switch(req.backend)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    new_dim = session.embeddings.get_dim()
    session.streamer.bind_loop(asyncio.get_running_loop())
    if new_dim != old_dim:
        session.streamer.emit(
            "status",
            {
                "status": session.trainer.get_state().status if session.trainer else "idle",
                "message": (
                    f"Embedding dim changed {old_dim}→{new_dim}; the network will be "
                    "rebuilt on the next training start if memory is enabled."
                ),
            },
        )
    return {"backend": req.backend, "embedding_dim": new_dim, "rebuild_needed": new_dim != old_dim}


# ---- health ----------------------------------------------------------------
@app.get("/health")
async def health() -> dict:
    """Report CUDA/VRAM status, embedding readiness, and Ollama reachability."""
    cuda = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda else None
    vram_total = vram_free = 0.0
    if cuda:
        free, total = torch.cuda.mem_get_info()
        vram_total = round(total / 1024**3, 2)
        vram_free = round(free / 1024**3, 2)

    st_loaded = any(s.embeddings.sentence_transformers_loaded for s in SESSIONS.values())

    ollama_ok = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_ok = resp.status_code == 200
    except httpx.HTTPError:
        ollama_ok = False

    return {
        "status": "ok",
        "cuda_available": cuda,
        "cuda_device": device_name,
        "vram_total_gb": vram_total,
        "vram_free_gb": vram_free,
        "sentence_transformers_loaded": st_loaded,
        "ollama_reachable": ollama_ok,
        "active_sessions": len(SESSIONS),
    }
