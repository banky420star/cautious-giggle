import React from 'react'
import { StatusPayload } from '../types'

interface Props {
  status: StatusPayload
}

const colors = {
  bg: '#0d1726',
  panelBg: 'rgba(13,23,38,0.92)',
  border: 'rgba(255,255,255,0.08)',
  text: '#eef5ff',
  muted: '#97a9c6',
  green: '#39d98a',
  red: '#ff7b8f',
  cyan: '#5ad7ff',
  amber: '#f3bb4a',
}

const panelStyle: React.CSSProperties = {
  background: colors.panelBg,
  border: `1px solid ${colors.border}`,
  borderRadius: 10,
  padding: 16,
  marginBottom: 16,
}

const cardStyle: React.CSSProperties = {
  background: 'rgba(20,32,52,0.85)',
  border: `1px solid ${colors.border}`,
  borderRadius: 8,
  padding: 16,
}

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  color: colors.muted,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  marginBottom: 2,
}

const valueStyle: React.CSSProperties = {
  fontSize: 13,
  color: colors.text,
  fontFamily: 'monospace',
  wordBreak: 'break-all',
}

interface PPODiagEntry {
  symbol?: string
  status?: string
  model_path?: string
  obs_shape?: any
  raw_action?: any
  decoded_target?: any
  threshold?: number
  skip_reason?: string
}

function extractPPODiags(status: StatusPayload): PPODiagEntry[] {
  const incidents = status.incidents ?? []
  const diags: PPODiagEntry[] = []
  for (const inc of incidents) {
    if (inc?.ppo_diag) {
      // ppo_diag could be an object or array
      if (Array.isArray(inc.ppo_diag)) {
        diags.push(...inc.ppo_diag)
      } else {
        diags.push(inc.ppo_diag)
      }
    }
  }
  return diags
}

interface ActiveModelEntry {
  symbol: string
  champion?: string
  canary?: string
}

function extractActiveModels(status: StatusPayload): ActiveModelEntry[] {
  const models = status.active_models
  if (!models) return []

  // active_models might be keyed by symbol, or have a per_symbol map
  if (models.per_symbol && typeof models.per_symbol === 'object') {
    return Object.entries(models.per_symbol).map(([symbol, data]: [string, any]) => ({
      symbol,
      champion: data?.champion ?? data?.champion_path ?? undefined,
      canary: data?.canary ?? data?.canary_path ?? undefined,
    }))
  }

  // If registry_summary has symbol_rows, use those
  const rows = status.registry_summary?.symbol_rows
  if (Array.isArray(rows)) {
    return rows.map((row: any) => ({
      symbol: row.symbol ?? '--',
      champion: row.champion ?? undefined,
      canary: row.canary ?? undefined,
    }))
  }

  // Fallback: single champion/canary at top level
  if (models.champion || models.canary) {
    return [{
      symbol: 'Global',
      champion: models.champion_path ?? models.champion ?? undefined,
      canary: models.canary_path ?? models.canary ?? undefined,
    }]
  }

  return []
}

function statusBadge(statusVal: string | undefined): { label: string; bgColor: string; textColor: string } {
  const s = (statusVal ?? '').toUpperCase()
  if (s === 'ACTIVE') {
    return { label: 'ACTIVE', bgColor: 'rgba(57,217,138,0.15)', textColor: colors.green }
  }
  if (s === 'SKIPPED') {
    return { label: 'SKIPPED', bgColor: 'rgba(255,123,143,0.15)', textColor: colors.red }
  }
  return { label: s || '--', bgColor: 'rgba(151,169,198,0.1)', textColor: colors.muted }
}

function formatValue(v: any): string {
  if (v == null) return '--'
  if (typeof v === 'number') return v.toFixed(4)
  if (typeof v === 'object') return JSON.stringify(v)
  return String(v)
}

const PPODiagPanel: React.FC<Props> = ({ status }) => {
  const diags = extractPPODiags(status)
  const modelEntries = extractActiveModels(status)

  return (
    <div style={{ background: colors.bg, color: colors.text, padding: 20 }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 18, color: colors.cyan, fontWeight: 700 }}>
        PPO Brain Diagnostics
      </h2>

      {/* PPO Symbol Diagnostics */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: colors.cyan }}>
          Per-Symbol Inference Status
        </h3>
        {diags.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13, padding: 12 }}>
            No PPO diagnostic data available. Diagnostics appear in status.incidents with ppo_diag payload.
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
              gap: 14,
            }}
          >
            {diags.map((diag, idx) => {
              const badge = statusBadge(diag.status)
              return (
                <div key={diag.symbol ?? idx} style={cardStyle}>
                  {/* Card header */}
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: 12,
                  }}>
                    <span style={{ fontSize: 16, fontWeight: 700, color: colors.cyan }}>
                      {diag.symbol ?? 'Unknown'}
                    </span>
                    <span style={{
                      padding: '3px 10px',
                      borderRadius: 12,
                      fontSize: 11,
                      fontWeight: 700,
                      background: badge.bgColor,
                      color: badge.textColor,
                      border: `1px solid ${badge.textColor}30`,
                    }}>
                      {badge.label}
                    </span>
                  </div>

                  {/* Skip reason - prominent if present */}
                  {diag.skip_reason && (
                    <div style={{
                      background: 'rgba(255,123,143,0.08)',
                      border: `1px solid rgba(255,123,143,0.2)`,
                      borderRadius: 6,
                      padding: '8px 12px',
                      marginBottom: 12,
                      fontSize: 13,
                      color: colors.red,
                      fontWeight: 600,
                    }}>
                      Skip: {diag.skip_reason}
                    </div>
                  )}

                  {/* Diagnostic fields */}
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                    <div>
                      <div style={labelStyle}>Model Path</div>
                      <div style={valueStyle}>{diag.model_path ?? '--'}</div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <div style={labelStyle}>Obs Shape</div>
                        <div style={valueStyle}>{formatValue(diag.obs_shape)}</div>
                      </div>
                      <div>
                        <div style={labelStyle}>Threshold</div>
                        <div style={valueStyle}>{diag.threshold != null ? diag.threshold.toFixed(4) : '--'}</div>
                      </div>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                      <div>
                        <div style={labelStyle}>Raw Action</div>
                        <div style={valueStyle}>{formatValue(diag.raw_action)}</div>
                      </div>
                      <div>
                        <div style={labelStyle}>Decoded Target</div>
                        <div style={valueStyle}>{formatValue(diag.decoded_target)}</div>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Model Registry */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: colors.cyan }}>
          Model Registry
        </h3>
        {modelEntries.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13, padding: 12 }}>
            No active model registry data available.
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))',
              gap: 14,
            }}
          >
            {modelEntries.map((entry, idx) => (
              <div key={entry.symbol ?? idx} style={cardStyle}>
                <div style={{
                  fontSize: 14,
                  fontWeight: 700,
                  color: colors.cyan,
                  marginBottom: 10,
                  paddingBottom: 8,
                  borderBottom: `1px solid ${colors.border}`,
                }}>
                  {entry.symbol}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  <div>
                    <div style={labelStyle}>Champion</div>
                    <div style={{
                      ...valueStyle,
                      color: entry.champion ? colors.green : colors.muted,
                    }}>
                      {entry.champion ?? 'None'}
                    </div>
                  </div>
                  <div>
                    <div style={labelStyle}>Canary</div>
                    <div style={{
                      ...valueStyle,
                      color: entry.canary ? colors.amber : colors.muted,
                    }}>
                      {entry.canary ?? 'None'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

export default PPODiagPanel
