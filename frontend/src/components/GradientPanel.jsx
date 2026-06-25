const EXPLODING = '#EF4444'
const VANISHING = '#4A5A6A'

function barColor(norm, layerIndex) {
  if (norm < 0.01) return VANISHING
  if (norm > 10) return EXPLODING
  return `var(--layer-${layerIndex % 5})`
}

// Map a gradient norm to a 0–100% width on a log scale (1e-3 … 1e2).
function logWidth(norm) {
  if (norm <= 0) return 2
  const pct = ((Math.log10(norm) + 3) / 5) * 100
  return Math.max(2, Math.min(100, pct))
}

export default function GradientPanel({ gradients }) {
  const norms = gradients?.gradient_norms
  const dead = gradients?.dead_neuron_pct || []
  const labels = gradients?.layer_labels || []

  return (
    <div className="card">
      <h2 className="card__title">Gradients</h2>

      {!norms || norms.length === 0 ? (
        <p className="placeholder">Gradient norms appear once training starts</p>
      ) : (
        <div className="grad-list">
          {norms.map((norm, i) => (
            <div className="grad-row" key={i}>
              <div className="grad-row__head">
                <span className="grad-row__label">{labels[i] || `Layer ${i + 1}`}</span>
                <span className="grad-row__value">{norm == null ? '—' : norm.toExponential(2)}</span>
              </div>
              <div className="grad-row__track">
                <div
                  className="grad-row__bar"
                  style={{ width: `${logWidth(norm)}%`, background: barColor(norm, i) }}
                />
              </div>
              <div className="grad-row__dead">
                <span className="grad-row__dead-label">dead {Math.round(dead[i] || 0)}%</span>
                <div className="grad-row__dead-track">
                  <div
                    className="grad-row__dead-bar"
                    style={{ width: `${Math.min(100, dead[i] || 0)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="grad-legend">
        <span><i className="swatch" style={{ background: VANISHING }} /> Vanishing</span>
        <span><i className="swatch swatch--layer" /> Healthy</span>
        <span><i className="swatch" style={{ background: EXPLODING }} /> Exploding</span>
      </div>
    </div>
  )
}
