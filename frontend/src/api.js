// Thin client for the NeuralCanvas backend. Every request carries the
// ngrok-skip-browser-warning header (so an ngrok-tunnelled backend returns JSON,
// not its interstitial) and throws a clear error on non-2xx / network failure.

// Default to the NeuralCanvas backend on 8001 (LocalMind owns 8000). Override
// with VITE_API_URL in production.
export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8001'

const BASE_HEADERS = { 'ngrok-skip-browser-warning': 'true' }

async function jsonRequest(path, options = {}) {
  let res
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: { ...BASE_HEADERS, ...(options.headers || {}) },
    })
  } catch (err) {
    throw new Error(`Network error reaching ${path}: ${err.message || err}`)
  }
  if (!res.ok) {
    let detail = ''
    try {
      detail = (await res.json())?.detail || ''
    } catch {
      detail = ''
    }
    throw new Error(`Request to ${path} failed (${res.status})${detail ? `: ${detail}` : ''}`)
  }
  return res.json()
}

function postJSON(path, body) {
  return jsonRequest(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export function startSession(config) {
  return postJSON('/session/start', config)
}

export function pauseSession(sessionId) {
  return postJSON('/session/pause', { session_id: sessionId })
}

export function resumeSession(sessionId) {
  return postJSON('/session/resume', { session_id: sessionId })
}

export function stopSession(sessionId) {
  return postJSON('/session/stop', { session_id: sessionId })
}

export function getState(sessionId) {
  return jsonRequest(`/session/state?session_id=${encodeURIComponent(sessionId)}`)
}

export function addMemory(text, sessionId) {
  return postJSON('/memory/add', { text, session_id: sessionId })
}

export function clearMemory(sessionId) {
  return postJSON('/memory/clear', { session_id: sessionId })
}

export function listMemory(sessionId) {
  return jsonRequest(`/memory/list?session_id=${encodeURIComponent(sessionId)}`)
}

export function switchEmbeddingBackend(backend, sessionId) {
  return postJSON('/embedding/switch', { backend, session_id: sessionId })
}

export function getHealth() {
  return jsonRequest('/health')
}

// Parse one raw SSE event block ("event: x\ndata: {...}") into { event, data }.
function parseSSEEvent(raw) {
  let event = 'message'
  const dataLines = []
  for (const line of raw.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) }
  } catch {
    return null
  }
}

// Open the SSE stream via fetch ReadableStream (EventSource can't set headers).
// `onEvent(type, data)` fires per event; resolves when the stream closes, throws
// on connection failure.
export async function streamEvents(sessionId, onEvent, signal) {
  const res = await fetch(
    `${API_BASE}/events/stream?session_id=${encodeURIComponent(sessionId)}`,
    { headers: BASE_HEADERS, signal },
  )
  if (!res.ok || !res.body) {
    throw new Error(`Event stream failed (${res.status})`)
  }
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)
      const parsed = parseSSEEvent(rawEvent)
      if (parsed) onEvent(parsed.event, parsed.data)
    }
  }
}
