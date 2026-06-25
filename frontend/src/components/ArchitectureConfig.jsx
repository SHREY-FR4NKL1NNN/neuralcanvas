import { useState } from 'react'

const POWERS = [8, 16, 32, 64, 128, 256, 512]
const ACTIVATIONS = [
  ['relu', 'ReLU'],
  ['tanh', 'Tanh'],
  ['sigmoid', 'Sigmoid'],
  ['leaky_relu', 'Leaky ReLU'],
]
const OPTIMIZERS = [
  ['adam', 'Adam'],
  ['sgd', 'SGD'],
  ['rmsprop', 'RMSProp'],
]
const BATCH_SIZES = [16, 32, 64, 128, 256]

// Parse an uploaded CSV client-side to preview its shape (last column = label).
function inspectCsv(text) {
  const rows = text.split(/\r?\n/).filter((r) => r.trim().length > 0)
  if (rows.length === 0) return null
  let dataRows = rows
  const first = rows[0].split(',')
  if (first.some((v) => Number.isNaN(Number(v)))) dataRows = rows.slice(1)
  if (dataRows.length === 0) return null
  const cols = dataRows[0].split(',').length
  const labels = new Set(dataRows.map((r) => r.split(',').pop().trim()))
  return { input_size: cols - 1, output_size: labels.size, n_samples: dataRows.length }
}

export default function ArchitectureConfig({ config, onChange, disabled }) {
  const [csvError, setCsvError] = useState('')
  const nc = config.network_config

  const update = (patch) => onChange({ ...config, ...patch })
  const updateNet = (patch) =>
    onChange({ ...config, network_config: { ...nc, ...patch } })
  const updateParams = (patch) =>
    onChange({ ...config, dataset_params: { ...config.dataset_params, ...patch } })

  function selectDataset(dataset) {
    if (dataset === 'xor') {
      onChange({
        ...config,
        dataset,
        dataset_params: { n_samples: 1000, noise: 0.1 },
        network_config: { ...nc, input_size: 2, output_size: 1 },
      })
    } else if (dataset === 'mnist') {
      onChange({
        ...config,
        dataset,
        dataset_params: { n_samples: 5000 },
        network_config: { ...nc, input_size: 784, output_size: 10 },
      })
    } else {
      onChange({
        ...config,
        dataset,
        dataset_params: {},
        network_config: { ...nc, input_size: 0, output_size: 0 },
      })
    }
  }

  function onCsvFile(file) {
    setCsvError('')
    const reader = new FileReader()
    reader.onload = () => {
      const text = String(reader.result)
      const info = inspectCsv(text)
      if (!info || info.output_size < 2) {
        setCsvError('Could not detect features/labels (need ≥2 classes).')
        return
      }
      const b64 = btoa(unescape(encodeURIComponent(text)))
      onChange({
        ...config,
        dataset: 'csv',
        dataset_params: { csv_b64: b64, n_samples: info.n_samples },
        network_config: { ...nc, input_size: info.input_size, output_size: info.output_size },
      })
    }
    reader.onerror = () => setCsvError('Could not read file.')
    reader.readAsText(file)
  }

  function setLayer(i, value) {
    const layers = [...nc.hidden_layers]
    layers[i] = value
    updateNet({ hidden_layers: layers })
  }
  const addLayer = () => updateNet({ hidden_layers: [...nc.hidden_layers, 16] })
  const removeLayer = (i) =>
    updateNet({ hidden_layers: nc.hidden_layers.filter((_, idx) => idx !== i) })

  const memLabel = config.use_memory
    ? config.memory_backend === 'ollama'
      ? 4096
      : 384
    : 0
  const shape = [
    `${nc.input_size || '?'}${memLabel ? ` + ${memLabel}mem` : ''}`,
    ...nc.hidden_layers,
    nc.output_size || '?',
  ].join(' → ')

  return (
    <div className="card">
      <h2 className="card__title">Architecture</h2>

      <div className="field">
        <label className="field__label">Dataset</label>
        <div className="seg">
          {['xor', 'mnist', 'csv'].map((d) => (
            <button
              key={d}
              type="button"
              className={`seg__btn${config.dataset === d ? ' seg__btn--on' : ''}`}
              onClick={() => selectDataset(d)}
              disabled={disabled}
            >
              {d.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      {config.dataset === 'xor' && (
        <>
          <Slider
            label={`Samples: ${config.dataset_params.n_samples}`}
            min={100}
            max={5000}
            step={100}
            value={config.dataset_params.n_samples}
            onChange={(v) => updateParams({ n_samples: v })}
            disabled={disabled}
          />
          <Slider
            label={`Noise: ${Number(config.dataset_params.noise).toFixed(2)}`}
            min={0}
            max={0.3}
            step={0.01}
            value={config.dataset_params.noise}
            onChange={(v) => updateParams({ noise: v })}
            disabled={disabled}
          />
        </>
      )}

      {config.dataset === 'mnist' && (
        <>
          <div className="field">
            <label className="field__label">Samples</label>
            <select
              className="input"
              value={config.dataset_params.n_samples}
              onChange={(e) => updateParams({ n_samples: Number(e.target.value) })}
              disabled={disabled}
            >
              {[1000, 2500, 5000, 10000].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <label className="toggle">
            <input
              type="checkbox"
              checked={Boolean(config.dataset_params.classes)}
              onChange={(e) =>
                e.target.checked
                  ? onChange({
                      ...config,
                      dataset_params: { ...config.dataset_params, classes: [0, 1] },
                      network_config: { ...nc, output_size: 2 },
                    })
                  : onChange({
                      ...config,
                      dataset_params: { n_samples: config.dataset_params.n_samples },
                      network_config: { ...nc, output_size: 10 },
                    })
              }
              disabled={disabled}
            />
            Binary mode (0 vs 1)
          </label>
        </>
      )}

      {config.dataset === 'csv' && (
        <div className="field">
          <label className="field__label">Upload CSV (last column = label)</label>
          <input
            type="file"
            accept=".csv,text/csv"
            className="input"
            onChange={(e) => e.target.files[0] && onCsvFile(e.target.files[0])}
            disabled={disabled}
          />
          {csvError && <span className="error-text">{csvError}</span>}
          {nc.input_size > 0 && (
            <span className="hint">
              Detected: {nc.input_size} features → {nc.output_size} classes
            </span>
          )}
        </div>
      )}

      <div className="field">
        <label className="field__label">Hidden layers</label>
        {nc.hidden_layers.map((n, i) => (
          <div key={i} className="layer-row">
            <select
              className="input input--sm"
              value={n}
              onChange={(e) => setLayer(i, Number(e.target.value))}
              disabled={disabled}
            >
              {POWERS.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
            <button
              type="button"
              className="btn btn--ghost btn--sm"
              onClick={() => removeLayer(i)}
              disabled={disabled || nc.hidden_layers.length <= 1}
            >
              ×
            </button>
          </div>
        ))}
        <button type="button" className="btn btn--ghost btn--sm" onClick={addLayer} disabled={disabled}>
          + Add layer
        </button>
      </div>

      <div className="field">
        <label className="field__label">Activation</label>
        <select
          className="input"
          value={nc.activation}
          onChange={(e) => updateNet({ activation: e.target.value })}
          disabled={disabled}
        >
          {ACTIVATIONS.map(([v, l]) => (
            <option key={v} value={v}>{l}</option>
          ))}
        </select>
      </div>

      <Slider
        label={`Dropout: ${Number(nc.dropout).toFixed(2)}`}
        min={0}
        max={0.5}
        step={0.05}
        value={nc.dropout}
        onChange={(v) => updateNet({ dropout: v })}
        disabled={disabled}
      />

      <div className="field field--row">
        <div>
          <label className="field__label">Optimizer</label>
          <select
            className="input"
            value={config.optimizer}
            onChange={(e) => update({ optimizer: e.target.value })}
            disabled={disabled}
          >
            {OPTIMIZERS.map(([v, l]) => (
              <option key={v} value={v}>{l}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="field__label">Learning rate</label>
          <div className="lr">
            <input
              type="number"
              className="input input--sm"
              step="0.001"
              min="0.0001"
              value={config.learning_rate}
              onChange={(e) => update({ learning_rate: Number(e.target.value) })}
              disabled={disabled}
            />
            <div className="lr__presets">
              {[0.001, 0.01, 0.1].map((p) => (
                <button
                  key={p}
                  type="button"
                  className={`chip${config.learning_rate === p ? ' chip--on' : ''}`}
                  onClick={() => update({ learning_rate: p })}
                  disabled={disabled}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      <Slider
        label={`Epochs: ${config.epochs}`}
        min={1}
        max={50}
        step={1}
        value={config.epochs}
        onChange={(v) => update({ epochs: v })}
        disabled={disabled}
      />

      <div className="field">
        <label className="field__label">Batch size</label>
        <div className="seg">
          {BATCH_SIZES.map((b) => (
            <button
              key={b}
              type="button"
              className={`seg__btn${config.batch_size === b ? ' seg__btn--on' : ''}`}
              onClick={() => update({ batch_size: b })}
              disabled={disabled}
            >
              {b}
            </button>
          ))}
        </div>
      </div>

      <label className="toggle toggle--feature">
        <input
          type="checkbox"
          checked={config.use_memory}
          onChange={(e) =>
            update({
              use_memory: e.target.checked,
              network_config: {
                ...nc,
                memory_dim: e.target.checked ? (config.memory_backend === 'ollama' ? 4096 : 384) : 0,
              },
            })
          }
          disabled={disabled}
        />
        Memory augmentation
        <span className="hint hint--inline">backend chosen in the Memory panel ↓</span>
      </label>

      <div className="shape">
        <span className="shape__label">Network shape</span>
        <code className="shape__value">{shape}</code>
      </div>
    </div>
  )
}

function Slider({ label, min, max, step, value, onChange, disabled }) {
  return (
    <div className="field">
      <label className="field__label">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
      />
    </div>
  )
}
