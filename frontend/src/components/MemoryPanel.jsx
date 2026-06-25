import { useEffect, useState } from 'react'
import { addMemory, clearMemory, listMemory, switchEmbeddingBackend } from '../api'

const BACKENDS = [
  ['sentence-transformers', 'sentence-transformers', 384],
  ['ollama', 'Ollama', 4096],
]

function formatTime(iso) {
  const d = new Date(iso)
  return Number.isNaN(d.getTime()) ? '' : d.toLocaleTimeString([], { hour12: false })
}

export default function MemoryPanel({ sessionId, config, onChange, retrieval }) {
  const [text, setText] = useState('')
  const [entries, setEntries] = useState([])
  const [error, setError] = useState('')
  const [warning, setWarning] = useState('')
  const [flash, setFlash] = useState(null)
  const [busy, setBusy] = useState(false)

  async function refresh() {
    try {
      const data = await listMemory(sessionId)
      setEntries(data.entries || [])
    } catch (err) {
      setError(err.message)
    }
  }

  useEffect(() => {
    refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Flash the most recent retrieval for 3 seconds.
  useEffect(() => {
    if (!retrieval) return undefined
    setFlash(retrieval)
    const id = setTimeout(() => setFlash(null), 3000)
    return () => clearTimeout(id)
  }, [retrieval?.ts])

  async function handleAdd() {
    const t = text.trim()
    if (!t || busy) return
    setBusy(true)
    setError('')
    try {
      await addMemory(t, sessionId)
      setText('')
      await refresh()
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleClear() {
    setError('')
    try {
      await clearMemory(sessionId)
      await refresh()
    } catch (err) {
      setError(err.message)
    }
  }

  async function handleSwitch(backend) {
    if (backend === config.memory_backend) return
    setError('')
    try {
      const res = await switchEmbeddingBackend(backend, sessionId)
      onChange({
        ...config,
        memory_backend: backend,
        network_config: {
          ...config.network_config,
          memory_dim: config.use_memory ? res.embedding_dim : 0,
        },
      })
      setWarning(
        res.rebuild_needed
          ? `Embedding dim is now ${res.embedding_dim}; the network rebuilds on next start.`
          : '',
      )
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="card">
      <h2 className="card__title">Memory</h2>

      <div className="seg">
        {BACKENDS.map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={`seg__btn${config.memory_backend === value ? ' seg__btn--on' : ''}`}
            onClick={() => handleSwitch(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {warning && <div className="warn-inline">{warning}</div>}

      <div className="mem-add">
        <input
          className="input"
          placeholder="Add a text memory…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleAdd()}
        />
        <button type="button" className="btn btn--sm" onClick={handleAdd} disabled={busy}>
          Add
        </button>
      </div>

      {flash && (
        <div className="retrieval-flash">
          <span className="retrieval-flash__label">Retrieved</span>
          <span className="retrieval-flash__text">{flash.retrieved_texts?.[0]}</span>
          {flash.scores?.[0] != null && (
            <span className="retrieval-flash__score">score {flash.scores[0]}</span>
          )}
        </div>
      )}

      {error && <span className="error-text">{error}</span>}

      {entries.length === 0 ? (
        <p className="placeholder">Add text memories to augment the network's input context</p>
      ) : (
        <div className="mem-list">
          {entries
            .slice()
            .reverse()
            .map((e, i) => (
              <div className="mem-row" key={`${e.timestamp}-${i}`}>
                <span className="mem-row__text">{e.text}</span>
                <span className="mem-row__meta">
                  <span className="badge-count" title="retrieval count">{e.retrieval_count}×</span>
                  <span className="muted">{formatTime(e.timestamp)}</span>
                </span>
              </div>
            ))}
          <button type="button" className="btn btn--ghost btn--sm" onClick={handleClear}>
            Clear all
          </button>
        </div>
      )}
    </div>
  )
}
