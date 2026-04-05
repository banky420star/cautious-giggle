import React from 'react'
import { StatusPayload } from '../types'

interface Props {
  status: StatusPayload
}

const panelBg = 'rgba(13,23,38,0.92)'
const innerBg = '#0a111a'
const altRowBg = '#080e16'
const borderColor = '#334'
const textColor = '#eef5ff'
const mutedColor = '#889'
const accentBlue = '#4fd6ff'
const profitGreen = '#22d68a'
const profitRed = '#f5475b'

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: 8,
  borderBottom: '1px solid #444',
  fontWeight: 600,
  fontSize: 13,
  color: mutedColor,
  whiteSpace: 'nowrap',
}

const tdStyle: React.CSSProperties = {
  padding: 8,
  fontSize: 13,
  fontFamily: 'monospace',
}

const sectionHeading: React.CSSProperties = {
  fontSize: 16,
  fontWeight: 600,
  marginBottom: 10,
  color: textColor,
}

const cardOuter: React.CSSProperties = {
  background: panelBg,
  border: `1px solid ${borderColor}`,
  borderRadius: 8,
  padding: 16,
  marginBottom: 16,
}

const formatNum = (v: number | undefined | null, decimals = 2): string => {
  if (v == null || isNaN(v)) return '-'
  return v.toFixed(decimals)
}

const blendColors: Record<string, string> = {
  raw_target: '#6e8efb',
  ppo_target: '#a855f7',
  dreamer_target: '#f59e0b',
  agi_bias: '#ec4899',
}

const TradingPanel: React.FC<Props> = ({ status }) => {
  const positions = status?.account?.positions ?? []
  const laneRows = status?.training?.symbol_lane_rows ?? []
  const account = status?.account

  const balance = account?.balance ?? 0
  const equity = account?.equity ?? 0
  const freeMargin = account?.free_margin ?? 0
  const floatingPnl = equity - balance

  return (
    <section style={{ color: textColor }}>
      {/* Account Summary */}
      <div style={cardOuter}>
        <h3 style={sectionHeading}>Account Summary</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
          {[
            { label: 'Balance', value: formatNum(balance) },
            { label: 'Equity', value: formatNum(equity) },
            { label: 'Free Margin', value: formatNum(freeMargin) },
            {
              label: 'Floating P&L',
              value: formatNum(floatingPnl),
              color: floatingPnl > 0 ? profitGreen : floatingPnl < 0 ? profitRed : textColor,
            },
          ].map((item, i) => (
            <div
              key={i}
              style={{
                background: innerBg,
                borderRadius: 6,
                padding: 12,
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: 11, color: mutedColor, marginBottom: 4 }}>{item.label}</div>
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 700,
                  fontFamily: 'monospace',
                  color: item.color ?? accentBlue,
                }}
              >
                {item.value}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Active Positions */}
      <div style={cardOuter}>
        <h3 style={sectionHeading}>Active Positions</h3>
        {positions.length === 0 ? (
          <div style={{ color: mutedColor, textAlign: 'center', padding: 20 }}>
            No open positions
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {['Symbol', 'Side', 'Volume', 'Entry', 'SL', 'TP', 'Profit'].map((h) => (
                    <th key={h} style={thStyle}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {positions.map((pos: any, idx: number) => {
                  const profit = pos.profit ?? 0
                  const profitColor =
                    profit > 0 ? profitGreen : profit < 0 ? profitRed : textColor
                  return (
                    <tr
                      key={pos.ticket ?? idx}
                      style={{ background: idx % 2 === 0 ? innerBg : altRowBg }}
                    >
                      <td style={tdStyle}>{pos.symbol ?? '-'}</td>
                      <td
                        style={{
                          ...tdStyle,
                          color: (pos.type ?? '')
                            .toString()
                            .toLowerCase()
                            .includes('buy')
                            ? profitGreen
                            : profitRed,
                        }}
                      >
                        {pos.type ?? '-'}
                      </td>
                      <td style={tdStyle}>{formatNum(pos.volume, 2)}</td>
                      <td style={tdStyle}>{formatNum(pos.open_price, 5)}</td>
                      <td style={tdStyle}>{formatNum(pos.sl, 5)}</td>
                      <td style={tdStyle}>{formatNum(pos.tp, 5)}</td>
                      <td style={{ ...tdStyle, color: profitColor, fontWeight: 600 }}>
                        {formatNum(profit)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Per-Symbol Decision Blend */}
      <div style={cardOuter}>
        <h3 style={sectionHeading}>Per-Symbol Decision Blend</h3>
        {laneRows.length === 0 ? (
          <div style={{ color: mutedColor, textAlign: 'center', padding: 20 }}>
            No symbol lane data
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(340px, 1fr))',
              gap: 12,
            }}
          >
            {laneRows.map((row: any, idx: number) => {
              const d = row.decision ?? {}
              const confidence = d.confidence ?? 0

              const components = [
                { key: 'raw_target', value: d.raw_target ?? 0 },
                { key: 'ppo_target', value: d.ppo_target ?? 0 },
                { key: 'dreamer_target', value: d.dreamer_target ?? 0 },
                { key: 'agi_bias', value: d.agi_bias ?? 0 },
              ]
              const absSum =
                components.reduce((s, c) => s + Math.abs(c.value), 0) || 1

              return (
                <div
                  key={row.symbol ?? idx}
                  style={{
                    background: innerBg,
                    border: `1px solid ${borderColor}`,
                    borderRadius: 6,
                    padding: 12,
                  }}
                >
                  {/* Card header */}
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      marginBottom: 8,
                    }}
                  >
                    <span style={{ fontWeight: 700, fontSize: 14 }}>
                      {row.symbol ?? '-'}
                    </span>
                    <span
                      style={{
                        fontSize: 11,
                        padding: '2px 8px',
                        borderRadius: 4,
                        background: 'rgba(79,214,255,0.12)',
                        color: accentBlue,
                      }}
                    >
                      {d.regime ?? '-'}
                    </span>
                  </div>

                  {/* Metrics grid */}
                  <div
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: 4,
                      fontSize: 12,
                      marginBottom: 8,
                    }}
                  >
                    <div>
                      <span style={{ color: mutedColor }}>Final: </span>
                      <span style={{ fontFamily: 'monospace', fontWeight: 600 }}>
                        {formatNum(d.final_target, 4)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: mutedColor }}>Confidence: </span>
                      <span
                        style={{
                          fontFamily: 'monospace',
                          fontWeight: 600,
                          color:
                            confidence > 0.7
                              ? profitGreen
                              : confidence > 0.4
                                ? '#f59e0b'
                                : profitRed,
                        }}
                      >
                        {formatNum(confidence, 3)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: mutedColor }}>Raw: </span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {formatNum(d.raw_target, 4)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: mutedColor }}>PPO: </span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {formatNum(d.ppo_target, 4)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: mutedColor }}>Dreamer: </span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {formatNum(d.dreamer_target, 4)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: mutedColor }}>AGI Bias: </span>
                      <span style={{ fontFamily: 'monospace' }}>
                        {formatNum(d.agi_bias, 4)}
                      </span>
                    </div>
                  </div>

                  {/* Horizontal blend bar */}
                  <div
                    style={{
                      height: 8,
                      borderRadius: 4,
                      overflow: 'hidden',
                      display: 'flex',
                      background: '#1a2234',
                    }}
                  >
                    {components.map((c) => {
                      const pct = (Math.abs(c.value) / absSum) * 100
                      return (
                        <div
                          key={c.key}
                          title={`${c.key}: ${c.value.toFixed(4)}`}
                          style={{
                            width: `${pct}%`,
                            background: blendColors[c.key],
                            transition: 'width 0.3s ease',
                          }}
                        />
                      )
                    })}
                  </div>

                  {/* Blend legend */}
                  <div
                    style={{
                      display: 'flex',
                      gap: 10,
                      marginTop: 6,
                      flexWrap: 'wrap',
                    }}
                  >
                    {components.map((c) => (
                      <div
                        key={c.key}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 4,
                          fontSize: 10,
                          color: mutedColor,
                        }}
                      >
                        <span
                          style={{
                            width: 8,
                            height: 8,
                            borderRadius: 2,
                            background: blendColors[c.key],
                            display: 'inline-block',
                          }}
                        />
                        {c.key.replace('_target', '').replace('_', ' ')}
                      </div>
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </section>
  )
}

export default TradingPanel
