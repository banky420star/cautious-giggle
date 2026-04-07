import React from 'react'
import { StatusPayload } from '../types'
import { Trade, TradeSummary, fetchTrades, fetchTradesSummary } from '../services/api'

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
  padding: 14,
}

const kpiStyle: React.CSSProperties = {
  background: colors.bg,
  borderRadius: 8,
  padding: '12px 14px',
  border: `1px solid ${colors.border}`,
  textAlign: 'center',
}

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  color: colors.muted,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
  marginBottom: 4,
}

const thStyle: React.CSSProperties = {
  textAlign: 'left',
  padding: '8px 10px',
  borderBottom: `1px solid ${colors.border}`,
  fontWeight: 600,
  fontSize: 12,
  color: colors.muted,
  whiteSpace: 'nowrap',
  position: 'sticky',
  top: 0,
  background: '#0a111a',
  zIndex: 1,
}

const tdStyle: React.CSSProperties = {
  padding: '7px 10px',
  fontSize: 13,
  fontFamily: 'monospace',
  whiteSpace: 'nowrap',
}

function fmtNum(v: number | undefined | null, decimals = 2): string {
  if (v == null || isNaN(v)) return '--'
  return v.toFixed(decimals)
}

function fmtPnl(v: number): string {
  const s = v.toFixed(2)
  return v >= 0 ? `+${s}` : s
}

function pnlColor(v: number): string {
  if (v > 0) return colors.green
  if (v < 0) return colors.red
  return colors.text
}

function outcomeColor(outcome: string): string {
  if (outcome === 'win') return colors.green
  if (outcome === 'loss') return colors.red
  return colors.amber
}

const MAGIC_RANGES = [
  { label: 'Standard BTC', range: '51000 - 51999', color: colors.cyan },
  { label: 'Standard XAU', range: '52000 - 52999', color: colors.amber },
  { label: 'HFT BTC', range: '61000 - 61999', color: colors.green },
  { label: 'HFT XAU', range: '62000 - 62999', color: colors.red },
]

const HFT_CONFIG = [
  { label: 'Timeframe', value: 'M1' },
  { label: 'Risk Limits', value: 'Separate per-lane' },
  { label: 'Bot Lane', value: 'hft' },
  { label: 'Execution', value: 'Low-latency path' },
]

const HFTHealthPanel: React.FC<Props> = ({ status }) => {
  const [hftSummary, setHftSummary] = React.useState<TradeSummary | null>(null)
  const [recentTrades, setRecentTrades] = React.useState<Trade[]>([])

  const loadHFTData = React.useCallback(async () => {
    const [summaryRes, tradesRes] = await Promise.all([
      fetchTradesSummary({ bot_lane: 'hft' }).catch((): TradeSummary => ({ overall: {} as any, by_symbol: {} })),
      fetchTrades({ bot_lane: 'hft', limit: '20' }).catch(() => ({ trades: [] as Trade[], total: 0, limit: 20, offset: 0 })),
    ])
    setHftSummary(summaryRes)
    setRecentTrades(tradesRes.trades)
  }, [])

  React.useEffect(() => {
    loadHFTData()
    const interval = setInterval(loadHFTData, 15_000)
    return () => clearInterval(interval)
  }, [loadHFTData])

  const overall = hftSummary?.overall

  return (
    <div style={{ background: colors.bg, color: colors.text, padding: 20 }}>
      <h2 style={{ margin: '0 0 16px', fontSize: 18, color: colors.cyan, fontWeight: 700 }}>
        HFT Bot Health
      </h2>

      {/* Magic Range Display */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: colors.cyan }}>
          Magic Number Ranges
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
            gap: 12,
          }}
        >
          {MAGIC_RANGES.map(mr => (
            <div key={mr.label} style={cardStyle}>
              <div style={{ ...labelStyle, color: mr.color }}>{mr.label}</div>
              <div style={{ fontSize: 18, fontWeight: 700, fontFamily: 'monospace', color: colors.text }}>
                {mr.range}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* HFT Trade Stats */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
          gap: 12,
          marginBottom: 16,
        }}
      >
        <div style={kpiStyle}>
          <div style={labelStyle}>Total HFT Trades</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: colors.cyan }}>
            {overall?.total_trades ?? '--'}
          </div>
        </div>
        <div style={kpiStyle}>
          <div style={labelStyle}>Win Rate</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: colors.green }}>
            {overall?.win_rate != null ? `${(overall.win_rate * 100).toFixed(1)}%` : '--'}
          </div>
        </div>
        <div style={kpiStyle}>
          <div style={labelStyle}>Total PnL</div>
          <div style={{
            fontSize: 20,
            fontWeight: 700,
            color: overall?.total_pnl != null ? pnlColor(overall.total_pnl) : colors.text,
          }}>
            {overall?.total_pnl != null ? fmtPnl(overall.total_pnl) : '--'}
          </div>
        </div>
        <div style={kpiStyle}>
          <div style={labelStyle}>Profit Factor</div>
          <div style={{ fontSize: 20, fontWeight: 700, color: colors.amber }}>
            {overall?.profit_factor != null
              ? String(typeof overall.profit_factor === 'number' ? overall.profit_factor.toFixed(2) : overall.profit_factor)
              : '--'}
          </div>
        </div>
      </div>

      {/* Recent HFT Trades */}
      <div style={{ ...panelStyle, padding: 0 }}>
        <div style={{ padding: '14px 16px 0 16px' }}>
          <h3 style={{ margin: '0 0 10px', fontSize: 15, fontWeight: 600, color: colors.cyan }}>
            Recent HFT Trades
          </h3>
        </div>
        <div style={{ maxHeight: 420, overflowY: 'auto', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', minWidth: 780 }}>
            <thead>
              <tr>
                {['Close Time', 'Symbol', 'Side', 'Volume', 'Open Price', 'Close Price', 'PnL', 'Hold (min)', 'Outcome'].map(h => (
                  <th key={h} style={thStyle}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {recentTrades.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ ...tdStyle, textAlign: 'center', color: colors.muted, padding: 32 }}>
                    No HFT trades found
                  </td>
                </tr>
              ) : (
                recentTrades.map((t, idx) => (
                  <tr
                    key={t.ticket}
                    style={{
                      background: idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.02)',
                      transition: 'background 0.1s',
                    }}
                    onMouseEnter={e => { (e.currentTarget as HTMLElement).style.background = 'rgba(90,215,255,0.06)' }}
                    onMouseLeave={e => { (e.currentTarget as HTMLElement).style.background = idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.02)' }}
                  >
                    <td style={tdStyle}>{t.close_time ?? '--'}</td>
                    <td style={{ ...tdStyle, color: colors.cyan, fontWeight: 600 }}>{t.symbol}</td>
                    <td style={{
                      ...tdStyle,
                      color: t.side === 'BUY' ? colors.green : colors.red,
                      fontWeight: 600,
                    }}>
                      {t.side}
                    </td>
                    <td style={tdStyle}>{fmtNum(t.volume, 2)}</td>
                    <td style={tdStyle}>{fmtNum(t.open_price, 5)}</td>
                    <td style={tdStyle}>{fmtNum(t.close_price, 5)}</td>
                    <td style={{ ...tdStyle, color: pnlColor(t.profit), fontWeight: 700 }}>
                      {fmtPnl(t.profit)}
                    </td>
                    <td style={tdStyle}>{t.hold_minutes != null ? fmtNum(t.hold_minutes, 1) : '--'}</td>
                    <td style={tdStyle}>
                      <span style={{
                        padding: '2px 8px',
                        borderRadius: 4,
                        fontSize: 11,
                        fontWeight: 700,
                        textTransform: 'uppercase',
                        background: `${outcomeColor(t.outcome)}20`,
                        color: outcomeColor(t.outcome),
                      }}>
                        {t.outcome}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Config Info */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 14px', fontSize: 15, fontWeight: 600, color: colors.cyan }}>
          HFT Configuration
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 12,
          }}
        >
          {HFT_CONFIG.map(cfg => (
            <div key={cfg.label} style={cardStyle}>
              <div style={labelStyle}>{cfg.label}</div>
              <div style={{ fontSize: 14, fontWeight: 600, color: colors.text }}>
                {cfg.value}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default HFTHealthPanel
