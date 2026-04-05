import React from 'react'
import { StatusPayload } from '../types'
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

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  color: colors.muted,
  textTransform: 'uppercase',
  letterSpacing: 0.5,
  marginBottom: 4,
  fontWeight: 600,
}

const valueStyle: React.CSSProperties = {
  fontSize: 13,
  color: colors.text,
  fontFamily: 'monospace',
  wordBreak: 'break-all',
}

const ModelsPanel: React.FC<Props> = ({ status }) => {
  const [loading, setLoading] = React.useState<string | null>(null)
  const [toast, setToast] = React.useState<{ msg: string; ok: boolean } | null>(null)

  const activeModels = status.active_models
  const canaryGate = status.canary_gate
  const registry = status.registry_summary

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
        Model Registry
      </h2>

      {/* Active Models */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Active Models</h3>
        {activeModels ? (
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <div style={{ flex: 1, minWidth: 200, background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)' }}>
              <div style={labelStyle}>Champion</div>
              <div style={valueStyle}>
                {activeModels.champion_path ?? activeModels.champion ?? 'None'}
              </div>
            </div>
            <div style={{ flex: 1, minWidth: 200, background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)' }}>
              <div style={labelStyle}>Canary</div>
              <div style={valueStyle}>
                {activeModels.canary_path ?? activeModels.canary ?? 'None'}
              </div>
            </div>
          </div>
        ) : (
          <div style={{ color: colors.muted, fontSize: 13 }}>No active model data available</div>
        )}
      </div>

      {/* Canary Gate */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Canary Gate</h3>
        {canaryGate ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 14px',
              borderRadius: 20,
              fontSize: 13,
              fontWeight: 700,
              background: canaryGate.ready
                ? 'rgba(57,217,138,0.15)'
                : 'rgba(243,187,74,0.15)',
              color: canaryGate.ready ? colors.green : colors.amber,
              border: `1px solid ${canaryGate.ready ? 'rgba(57,217,138,0.3)' : 'rgba(243,187,74,0.3)'}`,
            }}>
              <span style={{
                display: 'inline-block',
                width: 8,
                height: 8,
                borderRadius: '50%',
                background: canaryGate.ready ? colors.green : colors.amber,
              }} />
              {canaryGate.ready ? 'Ready' : 'Hold'}
            </span>
            {canaryGate.reason && (
              <span style={{ fontSize: 13, color: colors.muted }}>
                {canaryGate.reason}
              </span>
            )}
          </div>
        ) : (
          <div style={{ color: colors.muted, fontSize: 13 }}>No canary gate data available</div>
        )}
      </div>

      {/* Registry Summary */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Registry Summary</h3>
        {registry ? (
          <div>
            <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', marginBottom: 12 }}>
              <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', minWidth: 120 }}>
                <div style={labelStyle}>Symbol Count</div>
                <div style={{ fontSize: 22, fontWeight: 700, color: colors.cyan }}>
                  {registry.symbol_count ?? 0}
                </div>
              </div>
            </div>
            {registry.champion_history && Array.isArray(registry.champion_history) && registry.champion_history.length > 0 && (
              <div>
                <div style={{ ...labelStyle, marginBottom: 8 }}>Champion History</div>
                <div style={{
                  background: colors.bg,
                  borderRadius: 8,
                  padding: 10,
                  maxHeight: 200,
                  overflowY: 'auto',
                  border: '1px solid rgba(90,215,255,0.08)',
                }}>
                  {registry.champion_history.map((entry: any, idx: number) => (
                    <div
                      key={idx}
                      style={{
                        padding: '6px 8px',
                        fontSize: 12,
                        fontFamily: 'monospace',
                        color: colors.text,
                        borderBottom: '1px solid rgba(90,215,255,0.05)',
                        background: idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.02)',
                      }}
                    >
                      {typeof entry === 'string' ? entry : JSON.stringify(entry)}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: colors.muted, fontSize: 13 }}>No registry data available</div>
        )}
      </div>

      {/* Controls */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Controls</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <button
            style={{ ...btnBase, background: colors.green, opacity: loading === 'promote_canary' ? 0.6 : 1 }}
            disabled={loading !== null}
            onClick={() => handleAction('promote_canary')}
          >
            {loading === 'promote_canary' ? 'Promoting...' : 'Promote Canary'}
          </button>
          <button
            style={{ ...btnBase, background: colors.red, opacity: loading === 'rollback_champion' ? 0.6 : 1 }}
            disabled={loading !== null}
            onClick={() => handleAction('rollback_champion')}
          >
            {loading === 'rollback_champion' ? 'Rolling Back...' : 'Rollback Champion'}
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
    </section>
  )
}

export default ModelsPanel
