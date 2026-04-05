import React from 'react'
import { StatusPayload, TrainingVisual } from '../types'
import { controlAction } from '../services/api'

interface Props {
  status: StatusPayload
}

const colors = {
  bg: '#0d1726',
  panel: 'rgba(13,23,38,0.92)',
  text: '#eef5ff',
  muted: '#97a9c6',
  cyan: '#5ad7ff',
  green: '#39d98a',
  amber: '#f3bb4a',
  red: '#ff7b8f',
}

const panelStyle: React.CSSProperties = {
  background: colors.panel,
  borderRadius: 10,
  padding: 16,
  marginBottom: 16,
  border: '1px solid rgba(90,215,255,0.10)',
}

const cardStyle: React.CSSProperties = {
  background: colors.bg,
  borderRadius: 8,
  padding: 14,
  flex: 1,
  minWidth: 180,
  border: '1px solid rgba(90,215,255,0.08)',
}

const btnBase: React.CSSProperties = {
  padding: '8px 18px',
  borderRadius: 6,
  border: 'none',
  cursor: 'pointer',
  fontWeight: 600,
  fontSize: 13,
  color: colors.bg,
  transition: 'opacity 0.15s',
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '8px 10px',
  borderBottom: `1px solid rgba(90,215,255,0.15)`,
  color: colors.muted,
  fontSize: 12,
  fontWeight: 600,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
}

const tdStyle: React.CSSProperties = {
  padding: '7px 10px',
  fontSize: 13,
  color: colors.text,
  borderBottom: '1px solid rgba(90,215,255,0.05)',
}

function stateColor(state?: string): string {
  if (!state) return colors.muted
  const s = state.toLowerCase()
  if (s === 'running' || s === 'training' || s === 'active') return colors.green
  if (s === 'failed' || s === 'error') return colors.red
  if (s === 'queued' || s === 'pending' || s === 'waiting') return colors.amber
  if (s === 'done' || s === 'complete' || s === 'idle') return colors.cyan
  return colors.muted
}

function PipelineCard({ label, visual }: { label: string; visual?: TrainingVisual }) {
  const pct = visual?.progress_pct ?? 0
  return (
    <div style={cardStyle}>
      <div style={{ fontSize: 11, color: colors.muted, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{
          display: 'inline-block',
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: stateColor(visual?.state),
        }} />
        <span style={{ fontSize: 14, fontWeight: 600, color: colors.text }}>
          {visual?.state ?? 'idle'}
        </span>
      </div>
      {visual?.current_symbol && (
        <div style={{ fontSize: 12, color: colors.cyan, marginBottom: 6 }}>
          Symbol: <span style={{ fontWeight: 600 }}>{visual.current_symbol}</span>
        </div>
      )}
      {/* Progress bar */}
      <div style={{
        background: 'rgba(90,215,255,0.08)',
        borderRadius: 4,
        height: 6,
        overflow: 'hidden',
        marginBottom: 4,
      }}>
        <div style={{
          width: `${Math.min(100, Math.max(0, pct))}%`,
          height: '100%',
          background: stateColor(visual?.state),
          borderRadius: 4,
          transition: 'width 0.4s ease',
        }} />
      </div>
      <div style={{ fontSize: 11, color: colors.muted, textAlign: 'right' }}>
        {pct.toFixed(0)}%
      </div>
      {visual?.fail_reason && (
        <div style={{
          marginTop: 6,
          padding: '6px 8px',
          background: 'rgba(255,123,143,0.10)',
          borderRadius: 4,
          fontSize: 12,
          color: colors.red,
        }}>
          {visual.fail_reason}
        </div>
      )}
    </div>
  )
}

const TrainingPanel: React.FC<Props> = ({ status }) => {
  const [loading, setLoading] = React.useState<string | null>(null)
  const [toast, setToast] = React.useState<{ msg: string; ok: boolean } | null>(null)

  const training = status.training

  const handleAction = async (action: string) => {
    setLoading(action)
    setToast(null)
    try {
      const res = await controlAction(action)
      setToast({ msg: res?.message ?? res?.status ?? 'OK', ok: true })
    } catch (err: any) {
      setToast({ msg: err?.message ?? 'Action failed', ok: false })
    } finally {
      setLoading(null)
    }
  }

  return (
    <section style={{ background: colors.bg, color: colors.text, borderRadius: 12, padding: 20, marginBottom: 20 }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 18, color: colors.cyan, fontWeight: 700 }}>
        Training Pipeline
      </h2>

      {/* Pipeline Overview — 3 cards */}
      <div style={{ ...panelStyle }}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Pipeline Overview</h3>
        <div style={{ display: 'flex', gap: 14 }}>
          <PipelineCard label="LSTM" visual={training?.visual?.lstm} />
          <PipelineCard label="PPO" visual={training?.visual?.ppo} />
          <PipelineCard label="Dreamer" visual={training?.visual?.dreamer} />
        </div>
      </div>

      {/* Symbol Queue Table */}
      <div style={{ ...panelStyle }}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Symbol Queue</h3>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Symbol</th>
                <th style={thStyle}>LSTM</th>
                <th style={thStyle}>PPO</th>
                <th style={thStyle}>Dreamer</th>
                <th style={thStyle}>Detail</th>
              </tr>
            </thead>
            <tbody>
              {(training?.symbol_stage_rows ?? []).length === 0 ? (
                <tr>
                  <td colSpan={5} style={{ ...tdStyle, color: colors.muted, textAlign: 'center', padding: 20 }}>
                    No symbols in queue
                  </td>
                </tr>
              ) : (
                (training?.symbol_stage_rows ?? []).map((row: any, idx: number) => (
                  <tr key={idx} style={{ background: idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.03)' }}>
                    <td style={{ ...tdStyle, fontWeight: 600, color: colors.cyan }}>{row.symbol ?? '-'}</td>
                    <td style={{ ...tdStyle, color: stateColor(row.lstm) }}>{row.lstm ?? '-'}</td>
                    <td style={{ ...tdStyle, color: stateColor(row.ppo) }}>{row.ppo ?? '-'}</td>
                    <td style={{ ...tdStyle, color: stateColor(row.dreamer) }}>{row.dreamer ?? '-'}</td>
                    <td style={{ ...tdStyle, color: colors.muted, fontSize: 12 }}>{row.detail ?? '-'}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Controls */}
      <div style={{ ...panelStyle }}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Controls</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            style={{ ...btnBase, background: colors.green, opacity: loading === 'start_training_cycle' ? 0.6 : 1 }}
            disabled={loading !== null}
            onClick={() => handleAction('start_training_cycle')}
          >
            {loading === 'start_training_cycle' ? 'Starting...' : 'Start Training Cycle'}
          </button>
          <button
            style={{ ...btnBase, background: colors.red, opacity: loading === 'stop_training_cycle' ? 0.6 : 1 }}
            disabled={loading !== null}
            onClick={() => handleAction('stop_training_cycle')}
          >
            {loading === 'stop_training_cycle' ? 'Stopping...' : 'Stop Training Cycle'}
          </button>
          <button
            style={{ ...btnBase, background: colors.amber, opacity: loading === 'force_ingest' ? 0.6 : 1 }}
            disabled={loading !== null}
            onClick={() => handleAction('force_ingest')}
          >
            {loading === 'force_ingest' ? 'Ingesting...' : 'Force Ingest'}
          </button>
        </div>
        {toast && (
          <div style={{
            marginTop: 10,
            padding: '8px 12px',
            borderRadius: 6,
            fontSize: 13,
            background: toast.ok ? 'rgba(57,217,138,0.12)' : 'rgba(255,123,143,0.12)',
            color: toast.ok ? colors.green : colors.red,
            border: `1px solid ${toast.ok ? 'rgba(57,217,138,0.25)' : 'rgba(255,123,143,0.25)'}`,
          }}>
            {toast.msg}
          </div>
        )}
      </div>

      {/* Cycle Heartbeat */}
      {training?.cycle_heartbeat && (
        <div style={{ ...panelStyle, display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: colors.muted, fontSize: 12 }}>Cycle Heartbeat:</span>
          <span style={{ fontFamily: 'monospace', fontSize: 13, color: colors.cyan }}>
            {typeof training.cycle_heartbeat === 'string'
              ? training.cycle_heartbeat
              : new Date(training.cycle_heartbeat).toLocaleString()}
          </span>
        </div>
      )}
    </section>
  )
}

export default TrainingPanel
