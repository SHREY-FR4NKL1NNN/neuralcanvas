import { useState } from 'react'
import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

const LAYER_HEX = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981']

export default function ActivationHistogram({ epochInternals }) {
  const histos = epochInternals?.activation_histograms || []
  const labels = epochInternals?.layer_labels || []
  const [layer, setLayer] = useState(0)

  const idx = Math.min(layer, Math.max(0, histos.length - 1))
  const hist = histos[idx]
  const color = LAYER_HEX[idx % LAYER_HEX.length]

  const data =
    hist && hist.counts
      ? hist.counts.map((count, i) => ({
          bin: ((hist.edges[i] + hist.edges[i + 1]) / 2).toFixed(2),
          count,
        }))
      : []

  return (
    <div className="card">
      <h2 className="card__title">Activations</h2>

      {histos.length === 0 ? (
        <p className="placeholder">Activation distribution appears after the first epoch</p>
      ) : (
        <>
          <select className="input input--sm" value={idx} onChange={(e) => setLayer(Number(e.target.value))}>
            {histos.map((_, i) => (
              <option key={i} value={i}>{labels[i] || `Layer ${i + 1}`}</option>
            ))}
          </select>
          <div className="histogram">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -16 }}>
                <XAxis dataKey="bin" stroke="#4A5A6A" tick={{ fill: '#7A8A9A', fontSize: 9 }} interval={4} tickLine={false} />
                <YAxis stroke="#4A5A6A" tick={{ fill: '#7A8A9A', fontSize: 9 }} tickLine={false} width={40} />
                <Tooltip
                  contentStyle={{
                    background: '#151C24',
                    border: '1px solid rgba(255,255,255,0.14)',
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                  labelStyle={{ color: '#7A8A9A' }}
                />
                <Bar dataKey="count" fill={color} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="hint">
            Distribution of pre-activation values — dead zones appear as a spike near 0 with ReLU
          </p>
        </>
      )}
    </div>
  )
}
