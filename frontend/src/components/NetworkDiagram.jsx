import { useMemo } from 'react'

const MAX_NODES = 8
const HEALTHY = '#3B82F6'
const EXPLODING = '#EF4444'
const VANISHING = '#4A5A6A'

function layerColor(i) {
  return `var(--layer-${i % 5})`
}

function gradColor(norm) {
  if (norm == null) return VANISHING
  if (norm < 0.01) return VANISHING
  if (norm > 10) return EXPLODING
  return HEALTHY
}

// Mean |value| of a 20-bin histogram → a proxy for layer activation magnitude.
function histActivity(hist) {
  if (!hist || !hist.counts || hist.counts.length === 0) return 0
  let sum = 0
  let total = 0
  for (let i = 0; i < hist.counts.length; i += 1) {
    const center = (hist.edges[i] + hist.edges[i + 1]) / 2
    sum += Math.abs(center) * hist.counts[i]
    total += hist.counts[i]
  }
  return total > 0 ? sum / total : 0
}

export default function NetworkDiagram({
  config,
  epochInternals,
  gradients,
  active,
  memoryRetrieval,
}) {
  const nc = config.network_config
  const memDim = config.use_memory
    ? config.memory_backend === 'ollama'
      ? 4096
      : 384
    : 0

  const layers = useMemo(() => {
    const sizes = [
      (nc.input_size || 0) + memDim,
      ...nc.hidden_layers,
      nc.output_size || 0,
    ]
    return sizes.map((size, i) => {
      const visible = Math.max(1, Math.min(MAX_NODES, size || 1))
      let label = `Hidden ${i}`
      if (i === 0) label = memDim ? 'Input + M' : 'Input'
      else if (i === sizes.length - 1) label = 'Output'
      else label = `Hidden ${i}`
      return { size, visible, label }
    })
  }, [nc.input_size, nc.output_size, nc.hidden_layers, memDim])

  const W = 80 + (layers.length - 1) * 150
  const H = 220
  const cx = (i) => 40 + (i * (W - 80)) / Math.max(1, layers.length - 1)
  const nodeY = (count, idx) => {
    const span = Math.min(H - 60, count * 24)
    const top = (H - span) / 2 + 20
    return count === 1 ? H / 2 + 10 : top + (idx * span) / (count - 1)
  }

  const weights = epochInternals?.weight_matrices || []
  const histos = epochInternals?.activation_histograms || []
  const deadPct = gradients?.dead_neuron_pct || epochInternals?.dead_neuron_pct || []
  const gradNorms = gradients?.gradient_norms || epochInternals?.gradient_norms || []

  // Normalise activity across layers for circle brightness.
  const activities = histos.map(histActivity)
  const maxAct = Math.max(1e-6, ...activities)

  return (
    <div className="card diagram">
      <h2 className="card__title">Network</h2>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className={`diagram__svg${active ? ' diagram--active' : ''}`}
        preserveAspectRatio="xMidYMid meet"
      >
        {/* edges (drawn first, under nodes) */}
        {layers.slice(0, -1).map((layer, li) => {
          const next = layers[li + 1]
          const w = weights[li]
          return Array.from({ length: layer.visible }).flatMap((_, a) =>
            Array.from({ length: next.visible }).map((__, b) => {
              let mag = 0.15
              if (w && w[b] && typeof w[b][a] === 'number') {
                mag = Math.min(3, Math.abs(w[b][a]))
              }
              return (
                <line
                  key={`e-${li}-${a}-${b}`}
                  x1={cx(li)}
                  y1={nodeY(layer.visible, a)}
                  x2={cx(li + 1)}
                  y2={nodeY(next.visible, b)}
                  className="diagram__edge"
                  strokeWidth={0.4 + mag * 0.9}
                />
              )
            }),
          )
        })}

        {/* gradient-flow pulses per layer gap: two staggered dots streaming
            continuously while training, coloured by that layer's gradient health.
            (Discrete per-step pulses are imperceptible — XOR emits ~250 step
            events/sec — so the flow is driven by the training-active state and
            recoloured on every step event instead.) */}
        {layers.slice(0, -1).flatMap((layer, li) =>
          [0, 1].map((phase) => (
            <circle
              key={`pulse-${li}-${phase}`}
              r={4}
              cx={cx(li)}
              cy={H / 2 + 10}
              className="diagram__pulse"
              fill={gradColor(gradNorms[li])}
              style={{
                '--dist': `${cx(li + 1) - cx(li)}px`,
                '--delay': `${li * 90 + phase * 500}ms`,
              }}
            />
          )),
        )}

        {/* nodes */}
        {layers.map((layer, li) => {
          const act = activities[li] != null ? activities[li] / maxAct : 0
          const dead = (deadPct[li] || 0) > 50
          return Array.from({ length: layer.visible }).map((_, n) => (
            <circle
              key={`n-${li}-${n}`}
              cx={cx(li)}
              cy={nodeY(layer.visible, n)}
              r={9}
              className="diagram__node"
              style={{
                fill: dead ? VANISHING : layerColor(li),
                fillOpacity: dead ? 0.25 : 0.35 + act * 0.6,
                stroke: layerColor(li),
              }}
            />
          ))
        })}

        {/* memory injection badge on the input layer */}
        {memDim > 0 && (
          <g className={`diagram__mem${memoryRetrieval ? ' diagram__mem--glow' : ''}`} key={memoryRetrieval?.ts}>
            <circle cx={cx(0)} cy={nodeY(layers[0].visible, 0) - 26} r={11} className="diagram__mem-badge" />
            <text x={cx(0)} y={nodeY(layers[0].visible, 0) - 22} textAnchor="middle" className="diagram__mem-text">M</text>
          </g>
        )}

        {/* labels + sizes */}
        {layers.map((layer, li) => (
          <g key={`lbl-${li}`}>
            <text x={cx(li)} y={H - 14} textAnchor="middle" className="diagram__label">{layer.label}</text>
            <text x={cx(li)} y={H - 2} textAnchor="middle" className="diagram__size">{layer.size}</text>
          </g>
        ))}
      </svg>

      <div className="diagram__legend">
        <span><i className="swatch" style={{ background: HEALTHY }} /> healthy grad</span>
        <span><i className="swatch" style={{ background: EXPLODING }} /> exploding &gt;10</span>
        <span><i className="swatch" style={{ background: VANISHING }} /> vanishing &lt;0.01 / dead</span>
      </div>
    </div>
  )
}
