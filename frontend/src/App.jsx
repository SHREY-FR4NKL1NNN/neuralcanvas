import { useEffect, useReducer, useRef, useState } from 'react'
import ArchitectureConfig from './components/ArchitectureConfig'
import MemoryPanel from './components/MemoryPanel'
import TrainingControls from './components/TrainingControls'
import NetworkDiagram from './components/NetworkDiagram'
import LossCurve from './components/LossCurve'
import GradientPanel from './components/GradientPanel'
import WeightHeatmap from './components/WeightHeatmap'
import ActivationHistogram from './components/ActivationHistogram'
import { getHealth, startSession, pauseSession, resumeSession, stopSession, streamEvents } from './api'

const DEFAULT_CONFIG = {
  dataset: 'xor',
  dataset_params: { n_samples: 1000, noise: 0.1 },
  network_config: {
    input_size: 2,
    hidden_layers: [16, 8],
    output_size: 1,
    activation: 'relu',
    dropout: 0.0,
    memory_dim: 0,
  },
  optimizer: 'adam',
  learning_rate: 0.01,
  epochs: 20,
  batch_size: 32,
  loss_fn: 'cross_entropy',
  use_memory: false,
  memory_backend: 'sentence-transformers',
}

const INITIAL = {
  status: 'idle',
  epoch: 0,
  step: 0,
  stepsPerEpoch: 0,
  metrics: { train_loss: null, val_loss: null, train_acc: null, val_acc: null },
  lossHistory: [],
  gradients: null,
  epochInternals: null,
  stepPulse: 0,
  stepTimes: [],
  memoryRetrieval: null,
  error: null,
  done: null,
}

function reducer(state, action) {
  switch (action.type) {
    case 'reset':
      return { ...INITIAL, status: 'training' }
    case 'status':
      return { ...state, status: action.data.status || state.status }
    case 'step': {
      const d = action.data
      const times = [...state.stepTimes, { t: performance.now(), g: d.global_step }].slice(-40)
      return {
        ...state,
        status: 'training',
        epoch: d.epoch,
        step: d.step,
        gradients: d.internals_summary,
        stepPulse: state.stepPulse + 1,
        stepTimes: times,
        metrics: {
          ...state.metrics,
          train_loss: d.train_loss,
          train_acc: d.train_acc,
        },
      }
    }
    case 'epoch_end': {
      const d = action.data
      return {
        ...state,
        epoch: d.epoch,
        epochInternals: d.full_internals,
        metrics: {
          train_loss: d.train_loss,
          val_loss: d.val_loss,
          train_acc: d.train_acc,
          val_acc: d.val_acc,
        },
        lossHistory: [
          ...state.lossHistory,
          {
            epoch: d.epoch,
            train_loss: d.train_loss,
            val_loss: d.val_loss,
            train_acc: d.train_acc,
            val_acc: d.val_acc,
          },
        ],
      }
    }
    case 'memory_retrieval':
      return { ...state, memoryRetrieval: { ...action.data, ts: Date.now() } }
    case 'done':
      return { ...state, status: 'done', done: action.data }
    case 'error':
      return { ...state, status: 'error', error: action.data.message }
    default:
      return state
  }
}

export default function App() {
  const sessionId = useRef(
    (typeof crypto !== 'undefined' && crypto.randomUUID
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2)),
  ).current

  const [config, setConfig] = useState(DEFAULT_CONFIG)
  const [health, setHealth] = useState(null)
  const [state, dispatch] = useReducer(reducer, INITIAL)
  const abortRef = useRef(null)

  // Poll backend health every 5s.
  useEffect(() => {
    let active = true
    async function load() {
      try {
        const data = await getHealth()
        if (active) setHealth(data)
      } catch {
        if (active) setHealth({ status: 'error', cuda_available: false })
      }
    }
    load()
    const id = setInterval(load, 5000)
    return () => {
      active = false
      clearInterval(id)
    }
  }, [])

  // Clean up the SSE stream on unmount.
  useEffect(() => () => abortRef.current?.abort(), [])

  function onEvent(type, data) {
    dispatch({ type, data })
  }

  async function handleStart() {
    abortRef.current?.abort()
    dispatch({ type: 'reset' })
    try {
      await startSession({ ...config, session_id: sessionId })
    } catch (err) {
      dispatch({ type: 'error', data: { message: err.message } })
      return
    }
    const controller = new AbortController()
    abortRef.current = controller
    streamEvents(sessionId, onEvent, controller.signal).catch((err) => {
      if (!controller.signal.aborted) dispatch({ type: 'error', data: { message: err.message } })
    })
  }

  const handlePause = () => pauseSession(sessionId).catch(() => {})
  const handleResume = () => resumeSession(sessionId).catch(() => {})
  const handleStop = () => stopSession(sessionId).catch(() => {})

  const reachable = health && health.cuda_available !== undefined && health.status === 'ok'
  const stepsPerEpoch =
    Math.max(1, Math.ceil(((config.dataset_params.n_samples || 1000) * 0.8) / config.batch_size))

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar__brand">
          <span className="wordmark">NeuralCanvas</span>
          <span className="topbar__sub">real-time training observatory</span>
        </div>
        <div className="topbar__right">
          {health?.cuda_device && (
            <span className="cuda-badge" title={`VRAM ${health.vram_free_gb}/${health.vram_total_gb} GB free`}>
              ⚡ {health.cuda_device}
            </span>
          )}
          <span
            className={`dot ${reachable ? 'dot--ok' : 'dot--bad'}`}
            title={reachable ? 'Backend online' : 'Backend unreachable'}
          />
        </div>
      </header>

      <main className="layout">
        <aside className="col col--left">
          <ArchitectureConfig
            config={config}
            onChange={setConfig}
            disabled={state.status === 'training' || state.status === 'paused'}
          />
          <MemoryPanel
            sessionId={sessionId}
            config={config}
            onChange={setConfig}
            retrieval={state.memoryRetrieval}
          />
          <TrainingControls
            state={state}
            epochs={config.epochs}
            stepsPerEpoch={stepsPerEpoch}
            onStart={handleStart}
            onPause={handlePause}
            onResume={handleResume}
            onStop={handleStop}
          />
        </aside>

        <section className="col col--center">
          <NetworkDiagram
            config={config}
            epochInternals={state.epochInternals}
            gradients={state.gradients}
            active={state.status === 'training'}
            memoryRetrieval={state.memoryRetrieval}
          />
          <LossCurve lossHistory={state.lossHistory} metrics={state.metrics} />
        </section>

        <aside className="col col--right">
          <GradientPanel gradients={state.gradients} />
          <WeightHeatmap epochInternals={state.epochInternals} />
          <ActivationHistogram epochInternals={state.epochInternals} />
        </aside>
      </main>
    </div>
  )
}
