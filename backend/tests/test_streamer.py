"""Tests for the SSE streamer (formatting, ring buffer, numpy serialisation)."""

import numpy as np

from streamer import EventStreamer, _json_default


def test_json_default_handles_numpy():
    assert _json_default(np.float32(1.5)) == 1.5
    assert _json_default(np.int64(3)) == 3
    assert _json_default(np.array([1, 2])) == [1, 2]


def test_emit_buffers_and_serialises_numpy():
    s = EventStreamer()
    s.emit("step", {"x": np.float32(1.0), "arr": np.arange(3)})
    history = s.get_history()
    assert len(history) == 1
    assert history[0]["type"] == "step"


def test_emit_without_loop_does_not_crash():
    s = EventStreamer()
    s.emit("status", {"a": 1})  # no loop bound yet
    assert len(s.get_history()) == 1


def test_format_is_sse():
    sse = EventStreamer._format("done", {"ok": True})
    assert sse.startswith("event: done\n")
    assert "data: " in sse
    assert sse.endswith("\n\n")
