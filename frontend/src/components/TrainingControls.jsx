import { useMemo } from 'react'

const STATUS_LABEL = {
  idle: 'Idle',
  training: 'Training',
  paused: 'Paused',
  stopped: 'Stopped',
  done: 'Done',
  error: 'Error',
}

export default function TrainingControls({
  state,
  epochs,
  stepsPerEpoch,
  onStart,
  onPause,
  onResume,
  onStop,
}) {
  const { status, epoch, step, stepTimes } = state
  const isRunning = status === 'training' || status === 'paused'

  // steps/sec from the global-step counter carried on each step event.
  const speed = useMemo(() => {
    if (stepTimes.length < 2) return 0
    const first = stepTimes[0]
    const last = stepTimes[stepTimes.length - 1]
    const dt = (last.t - first.t) / 1000
    const dg = last.g - first.g
    return dt > 0 ? dg / dt : 0
  }, [stepTimes])

  const totalSteps = epochs * stepsPerEpoch
  const doneSteps = (epoch > 0 ? (epoch - 1) * stepsPerEpoch : 0) + step
  const stepsLeft = Math.max(0, totalSteps - doneSteps)
  const eta = speed > 0 && isRunning ? stepsLeft / speed : null

  const epochPct = epochs > 0 ? Math.min(100, (epoch / epochs) * 100) : 0
  const stepPct = stepsPerEpoch > 0 ? Math.min(100, (step / stepsPerEpoch) * 100) : 0

  return (
    <div className="card">
      <h2 className="card__title">Training</h2>

      <div className="controls">
        <button
          type="button"
          className="btn btn--primary"
          onClick={onStart}
          disabled={isRunning}
        >
          {status === 'done' || status === 'stopped' || status === 'error' ? 'Restart' : 'Start'}
        </button>
        {status === 'paused' ? (
          <button type="button" className="btn" onClick={onResume}>Resume</button>
        ) : (
          <button type="button" className="btn" onClick={onPause} disabled={status !== 'training'}>
            Pause
          </button>
        )}
        <button type="button" className="btn btn--ghost" onClick={onStop} disabled={!isRunning}>
          Stop
        </button>
      </div>

      <div className="status-row">
        <span className={`status-badge status-badge--${status}`}>
          {STATUS_LABEL[status] || status}
        </span>
        {speed > 0 && <span className="muted">{speed.toFixed(1)} steps/s</span>}
        {eta != null && <span className="muted">~{formatEta(eta)} left</span>}
      </div>

      <Progress label={`Epoch ${epoch} / ${epochs}`} pct={epochPct} />
      <Progress label={`Step ${step} / ${stepsPerEpoch}`} pct={stepPct} small />

      {state.error && <div className="error-card">⚠ {state.error}</div>}
    </div>
  )
}

function Progress({ label, pct, small }) {
  return (
    <div className={`progress${small ? ' progress--sm' : ''}`}>
      <div className="progress__head">
        <span>{label}</span>
        <span className="muted">{Math.round(pct)}%</span>
      </div>
      <div className="progress__track">
        <div className="progress__fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function formatEta(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return `${m}m ${s}s`
}
