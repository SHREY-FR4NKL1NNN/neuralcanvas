import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

const TRAIN = '#3B82F6'
const VAL = '#EC4899'

function fmt(v, digits = 3) {
  return v == null ? '—' : Number(v).toFixed(digits)
}

function MetricCard({ label, value, color }) {
  return (
    <div className="metric-card">
      <span className="metric-card__label">{label}</span>
      <span className="metric-card__value" style={color ? { color } : undefined}>
        {value}
      </span>
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="chart-tip">
      <div className="chart-tip__title">Epoch {label}</div>
      {payload.map((p) => (
        <div key={p.dataKey} style={{ color: p.color }}>
          {p.dataKey === 'train_loss' ? 'train' : 'val'}: {fmt(p.value, 4)}
        </div>
      ))}
    </div>
  )
}

export default function LossCurve({ lossHistory, metrics }) {
  return (
    <div className="card loss">
      <h2 className="card__title">Loss</h2>
      <div className="loss__chart">
        {lossHistory.length === 0 ? (
          <p className="placeholder placeholder--center">Loss curve appears once training starts</p>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={lossHistory} margin={{ top: 8, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis
                dataKey="epoch"
                stroke="#4A5A6A"
                tick={{ fill: '#7A8A9A', fontSize: 11 }}
                tickLine={false}
              />
              <YAxis
                stroke="#4A5A6A"
                tick={{ fill: '#7A8A9A', fontSize: 11 }}
                tickLine={false}
                width={48}
                domain={['auto', 'auto']}
              />
              <Tooltip content={<ChartTooltip />} />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke={TRAIN}
                strokeWidth={2}
                dot={{ r: 2, fill: TRAIN }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke={VAL}
                strokeWidth={2}
                dot={{ r: 2, fill: VAL }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="metric-grid">
        <MetricCard label="train loss" value={fmt(metrics.train_loss, 4)} color={TRAIN} />
        <MetricCard label="val loss" value={fmt(metrics.val_loss, 4)} color={VAL} />
        <MetricCard label="train acc" value={metrics.train_acc == null ? '—' : `${(metrics.train_acc * 100).toFixed(1)}%`} />
        <MetricCard label="val acc" value={metrics.val_acc == null ? '—' : `${(metrics.val_acc * 100).toFixed(1)}%`} />
      </div>
    </div>
  )
}
