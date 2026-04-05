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

interface ActionButtonProps {
  action: string
  label: string
  color: string
  loading: string | null
  onAction: (action: string) => void
}

const ActionButton: React.FC<ActionButtonProps> = ({ action, label, color, loading, onAction }) => (
  <button
    style={{ ...btnBase, background: color, opacity: loading === action ? 0.6 : 1 }}
    disabled={loading !== null}
    onClick={() => onAction(action)}
  >
    {loading === action ? 'Processing...' : label}
  </button>
)

const SettingsPanel: React.FC<Props> = ({ status }) => {
  const [loading, setLoading] = React.useState<string | null>(null)
  const [toast, setToast] = React.useState<{ msg: string; ok: boolean } | null>(null)

  const telegram = status.telegram

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
        Settings &amp; Controls
      </h2>

      {/* Telegram Status */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Telegram Status</h3>
        {telegram ? (
          <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
            <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', minWidth: 120 }}>
              <div style={labelStyle}>Configured</div>
              <div style={{
                fontSize: 15,
                fontWeight: 700,
                color: telegram.configured ? colors.green : colors.red,
              }}>
                <span style={{
                  display: 'inline-block',
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: telegram.configured ? colors.green : colors.red,
                  marginRight: 6,
                  verticalAlign: 'middle',
                }} />
                {telegram.configured ? 'Yes' : 'No'}
              </div>
            </div>
            <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', minWidth: 120 }}>
              <div style={labelStyle}>Card Count</div>
              <div style={{ fontSize: 22, fontWeight: 700, color: colors.cyan }}>
                {telegram.card_count ?? 0}
              </div>
            </div>
            {(telegram.delivered !== undefined || telegram.failed !== undefined || telegram.delivery_stats) && (
              <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', minWidth: 200 }}>
                <div style={labelStyle}>Delivery Stats</div>
                {telegram.delivery_stats ? (
                  <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                    {Object.entries(telegram.delivery_stats).map(([key, val]) => (
                      <div key={key}>
                        <span style={{ color: colors.muted }}>{key}:</span>{' '}
                        <span style={{ color: colors.text, fontFamily: 'monospace' }}>{String(val)}</span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: 13, lineHeight: 1.8 }}>
                    <div>
                      <span style={{ color: colors.muted }}>Delivered:</span>{' '}
                      <span style={{ color: colors.green, fontFamily: 'monospace' }}>{telegram.delivered ?? 0}</span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>Failed:</span>{' '}
                      <span style={{ color: colors.red, fontFamily: 'monospace' }}>{telegram.failed ?? 0}</span>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ) : (
          <div style={{ color: colors.muted, fontSize: 13 }}>No Telegram data available</div>
        )}
      </div>

      {/* Server Controls */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>Server Controls</h3>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <ActionButton
            action="restart_server"
            label="Restart Server"
            color={colors.amber}
            loading={loading}
            onAction={handleAction}
          />
          <ActionButton
            action="hft_start"
            label="HFT Start"
            color={colors.green}
            loading={loading}
            onAction={handleAction}
          />
          <ActionButton
            action="hft_stop"
            label="HFT Stop"
            color={colors.red}
            loading={loading}
            onAction={handleAction}
          />
          <ActionButton
            action="rebuild_trade_memory"
            label="Rebuild Trade Memory"
            color={colors.cyan}
            loading={loading}
            onAction={handleAction}
          />
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

      {/* System Info */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px', fontSize: 14, color: colors.muted, fontWeight: 600 }}>System Info</h3>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', flex: 1, minWidth: 200 }}>
            <div style={labelStyle}>Repo Root</div>
            <div style={{
              fontSize: 13,
              color: colors.text,
              fontFamily: 'monospace',
              wordBreak: 'break-all',
            }}>
              {status.repo_root ?? '--'}
            </div>
          </div>
          <div style={{ background: colors.bg, borderRadius: 8, padding: 14, border: '1px solid rgba(90,215,255,0.08)', minWidth: 120 }}>
            <div style={labelStyle}>State</div>
            <div style={{
              fontSize: 15,
              fontWeight: 700,
              color: status.state === 'running' || status.state === 'active'
                ? colors.green
                : status.state === 'error' || status.state === 'failed'
                  ? colors.red
                  : colors.amber,
            }}>
              {status.state ?? '--'}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

export default SettingsPanel
