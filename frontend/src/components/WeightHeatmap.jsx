import { useEffect, useRef, useState } from 'react'

// Diverging colour: blue (negative) → white (zero) → red (positive).
function diverging(value, scale) {
  const t = scale > 0 ? Math.max(-1, Math.min(1, value / scale)) : 0
  if (t >= 0) {
    // white → red
    const g = Math.round(255 * (1 - t))
    return `rgb(255,${g},${g})`
  }
  // white → blue
  const r = Math.round(255 * (1 + t))
  return `rgb(${r},${r},255)`
}

export default function WeightHeatmap({ epochInternals }) {
  const matrices = epochInternals?.weight_matrices || []
  const extremes = epochInternals?.weight_extremes || []
  const labels = epochInternals?.layer_labels || []
  const [layer, setLayer] = useState(0)
  const canvasRef = useRef(null)

  const idx = Math.min(layer, Math.max(0, matrices.length - 1))
  const matrix = matrices[idx]
  const ext = extremes[idx]

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !matrix || matrix.length === 0) return
    const rows = matrix.length
    const cols = matrix[0].length
    const cssSize = 220
    const dpr = window.devicePixelRatio || 1
    canvas.width = cssSize * dpr
    canvas.height = cssSize * dpr
    canvas.style.width = `${cssSize}px`
    canvas.style.height = `${cssSize}px`
    const ctx = canvas.getContext('2d')
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    ctx.clearRect(0, 0, cssSize, cssSize)

    const scale = Math.max(1e-6, Math.abs(ext?.min || 0), Math.abs(ext?.max || 0))
    const cw = cssSize / cols
    const ch = cssSize / rows
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        ctx.fillStyle = diverging(matrix[r][c], scale)
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch))
      }
    }
  }, [matrix, ext])

  return (
    <div className="card">
      <h2 className="card__title">Weights</h2>

      {matrices.length === 0 ? (
        <p className="placeholder">Weight heatmap appears after the first epoch</p>
      ) : (
        <>
          <select className="input input--sm" value={idx} onChange={(e) => setLayer(Number(e.target.value))}>
            {matrices.map((_, i) => (
              <option key={i} value={i}>{labels[i] || `Layer ${i + 1}`}</option>
            ))}
          </select>
          <div className="heatmap">
            <canvas ref={canvasRef} className="heatmap__canvas" />
          </div>
          <div className="heatmap__scale">
            <span className="muted">min {Number(ext?.min ?? 0).toFixed(3)}</span>
            <span className="heatmap__gradient" />
            <span className="muted">max {Number(ext?.max ?? 0).toFixed(3)}</span>
          </div>
          {matrix && (
            <p className="hint">{matrix.length}×{matrix[0]?.length} slice · blue −, red +</p>
          )}
        </>
      )}
    </div>
  )
}
