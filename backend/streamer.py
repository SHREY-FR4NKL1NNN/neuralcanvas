"""Server-Sent-Events fan-out for a training session.

The trainer runs in a background thread and calls :meth:`EventStreamer.emit`;
each connected SSE client has its own :class:`asyncio.Queue`. Because ``emit`` is
called from a non-async thread, events are handed to the event loop with
``loop.call_soon_threadsafe`` (the thread-safe way to push onto an asyncio
queue). The last 500 events are kept in a ring buffer so a reconnecting client
can catch up via ``/events/history``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator

import numpy as np

from log_config import get_logger

logger = get_logger("streamer")

MAX_CLIENTS = 10
RING_SIZE = 500


def _json_default(obj: object) -> object:
    """JSON fallback that converts numpy scalars/arrays to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


class EventStreamer:
    """Fan-out of training events to SSE subscribers, with a replay ring buffer."""

    def __init__(self, max_clients: int = MAX_CLIENTS) -> None:
        """Create an empty streamer (the event loop is bound later)."""
        self._clients: set[asyncio.Queue] = set()
        self._ring: deque[dict] = deque(maxlen=RING_SIZE)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()
        self._max_clients = max_clients

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the asyncio loop emit() will schedule onto (call from an endpoint)."""
        self._loop = loop

    @staticmethod
    def _format(event_type: str, data: dict) -> str:
        """Render an event in SSE wire format ``event: <type>\\ndata: <json>\\n\\n``."""
        payload = json.dumps(data, default=_json_default)
        return f"event: {event_type}\ndata: {payload}\n\n"

    def emit(self, event_type: str, data: dict) -> None:
        """Emit an event to the ring buffer and every connected client (thread-safe)."""
        event = {"type": event_type, "data": data, "ts": time.time() * 1000.0}
        try:
            sse = self._format(event_type, data)
        except TypeError as exc:  # serialisation guard — never crash the trainer
            logger.error("Failed to serialise %s event: %s", event_type, exc)
            return
        with self._lock:
            self._ring.append(event)
            clients = list(self._clients)
        if self._loop is None:
            return
        for queue in clients:
            self._loop.call_soon_threadsafe(self._safe_put, queue, sse)

    @staticmethod
    def _safe_put(queue: asyncio.Queue, sse: str) -> None:
        """Push onto a client queue, dropping the event if the client is too slow."""
        try:
            queue.put_nowait(sse)
        except asyncio.QueueFull:
            pass

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Yield SSE strings for one client until it disconnects.

        Rejects (with a single SSE error event) when ``max_clients`` are already
        connected. Sends an initial comment line so the connection opens promptly.
        """
        with self._lock:
            if len(self._clients) >= self._max_clients:
                yield self._format("error", {"message": "Too many SSE clients"})
                return
            queue: asyncio.Queue = asyncio.Queue(maxsize=2000)
            self._clients.add(queue)
        logger.debug("SSE client subscribed (%d total)", len(self._clients))
        try:
            yield ": connected\n\n"  # SSE comment to flush headers
            while True:
                sse = await queue.get()
                yield sse
        finally:
            with self._lock:
                self._clients.discard(queue)
            logger.debug("SSE client left (%d remaining)", len(self._clients))

    def get_history(self) -> list[dict]:
        """Return the buffered events (oldest first) for reconnecting clients."""
        with self._lock:
            return list(self._ring)
