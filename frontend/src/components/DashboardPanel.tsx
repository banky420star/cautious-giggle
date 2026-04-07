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
  amber: '#f3bb4a',
  red: '#ff7b8f',
  cyan: '#5ad7ff',
}

const panelStyle: React.CSSProperties = {
  background: colors.panelBg,
  border: `1px solid ${colors.border}`,
  borderRadius: 10,
  padding: 16,
}

const kpiStyle: React.CSSProperties = {
  ...panelStyle,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 4,
}

const labelStyle: React.CSSProperties = {
  fontSize: 11,
  fontWeight: 500,
  color: colors.muted,
  textTransform: 'uppercase',
  letterSpacing: '0.06em',
}

const valueStyle: React.CSSProperties = {
  fontSize: 22,
  fontWeight: 700,
  color: colors.text,
}

function fmtMoney(v: number | undefined | null): string {
  if (v == null || isNaN(v)) return '--'
  return v.toFixed(2)
}

function fmtSignal(v: number | undefined | null): string {
  if (v == null || isNaN(v)) return '--'
  return v.toFixed(4)
}

function statusBadge(
  active: boolean | undefined | null,
  onLabel: string,
  offLabel: string,
): { label: string; color: string } {
  if (active == null) return { label: '--', color: colors.muted }
  return active
    ? { label: onLabel, color: colors.green }
    : { label: offLabel, color: colors.red }
}

const DashboardPanel: React.FC<Props> = ({ status }) => {
  const [tradeSummary, setTradeSummary] = React.useState<TradeSummary | null>(null)
  const [recentTrades, setRecentTrades] = React.useState<Trade[]>([])

  React.useEffect(() => {
    const loadTradeData = async () => {
      const [summaryRes, tradesRes] = await Promise.all([
        fetchTradesSummary().catch((): TradeSummary => ({ overall: {} as any, by_symbol: {} })),
        fetchTrades({ limit: '5' }).catch(() => ({ trades: [] as Trade[], total: 0, limit: 5, offset: 0 })),
      ])
      setTradeSummary(summaryRes)
      setRecentTrades(tradesRes.trades)
    }
    loadTradeData()
    const interval = setInterval(loadTradeData, 15_000)
    return () => clearInterval(interval)
  }, [])

  const account = status.account
  const server = status.server
  const training = status.training
  const canary = status.canary_gate
  const lanes = training?.symbol_lane_rows ?? []
  const incidents = (status.incidents ?? []).slice(0, 8)
  const tradeOverall = tradeSummary?.overall

  const serverBadge = statusBadge(server?.running, 'running', 'stopped')
  const trainingBadge = statusBadge(training?.cycle_running, 'active', 'idle')
  const canaryBadge = statusBadge(canary?.ready, 'ready', 'hold')
  const pipelineSummary = training?.pipeline_summary
  const laneSummary = training?.lane_summary
  const activeStageLabel = training?.visual?.active_label ?? 'Idle'

  return (
    <div style={{ background: colors.bg, color: colors.text, padding: 20 }}>
      {/* KPI Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 14,
          marginBottom: 24,
        }}
      >
        {/* Row 1 */}
        <div style={kpiStyle}>
          <span style={labelStyle}>Account Balance</span>
          <span style={valueStyle}>${fmtMoney(account?.balance)}</span>
        </div>
        <div style={kpiStyle}>
          <span style={labelStyle}>Account Equity</span>
          <span style={valueStyle}>${fmtMoney(account?.equity)}</span>
        </div>
        <div style={kpiStyle}>
          <span style={labelStyle}>Open Positions</span>
          <span style={valueStyle}>{account?.open_positions ?? '--'}</span>
        </div>

        {/* Row 2 */}
        <div style={kpiStyle}>
          <span style={labelStyle}>Server Status</span>
          <span style={{ ...valueStyle, color: serverBadge.color }}>
            {serverBadge.label}
          </span>
        </div>
        <div style={kpiStyle}>
          <span style={labelStyle}>Training Cycle</span>
          <span style={{ ...valueStyle, color: trainingBadge.color }}>
            {trainingBadge.label}
          </span>
        </div>
        <div style={kpiStyle}>
          <span style={labelStyle}>Canary Gate</span>
          <span style={{ ...valueStyle, color: canaryBadge.color }}>
            {canaryBadge.label}
          </span>
        </div>
      </div>

      <div style={{ ...panelStyle, marginBottom: 24 }}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 14, color: colors.cyan, fontWeight: 600 }}>
          Pipeline Snapshot
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
            gap: 12,
          }}
        >
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Active training symbols</div>
            <div style={valueStyle}>{pipelineSummary?.training_active_symbols ?? '--'}</div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Champion live</div>
            <div style={valueStyle}>{pipelineSummary?.champion_live_symbols ?? '--'}</div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Trading ready symbols</div>
            <div style={valueStyle}>{pipelineSummary?.trading_ready_symbols ?? '--'}</div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Blocked symbols</div>
            <div style={valueStyle}>{laneSummary?.blocked_symbols ?? '--'}</div>
          </div>
        </div>
        <div style={{ marginTop: 12, fontSize: 12, color: colors.muted }}>
          Active stage: <span style={{ color: colors.cyan }}>{activeStageLabel}</span>
        </div>
      </div>

      {/* Signal Lanes */}
      <div style={{ ...panelStyle, marginBottom: 24 }}>
        <h3
          style={{
            margin: '0 0 14px 0',
            fontSize: 15,
            fontWeight: 600,
            color: colors.cyan,
          }}
        >
          Signal Lanes
        </h3>
        {lanes.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13 }}>
            No symbol lanes available.
          </div>
        ) : (
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: 12,
            }}
          >
            {lanes.map((lane: any, idx: number) => {
              const decision = lane.decision ?? {}
              const pipeline = lane.pipeline ?? {}
              const lstmState = pipeline.lstm?.state ?? '--'
              return (
                <div
                  key={lane.symbol ?? idx}
                  style={{
                    ...panelStyle,
                    background: 'rgba(20,32,52,0.85)',
                    padding: 14,
                  }}
                >
                  <div
                    style={{
                      fontSize: 14,
                      fontWeight: 700,
                      color: colors.cyan,
                      marginBottom: 8,
                    }}
                  >
                    {lane.symbol ?? 'UNKNOWN'}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 13 }}>
                    <div>
                      <span style={{ color: colors.muted }}>Regime: </span>
                      <span style={{ color: colors.text }}>{decision.regime ?? '--'}</span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>Final Target: </span>
                      <span style={{ color: colors.amber }}>
                        {fmtSignal(decision.final_target)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>PPO Target: </span>
                      <span style={{ color: colors.text }}>
                        {fmtSignal(decision.ppo_target)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>Dreamer Target: </span>
                      <span style={{ color: colors.text }}>
                        {fmtSignal(decision.dreamer_target)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>Confidence: </span>
                      <span style={{ color: colors.green }}>
                        {fmtSignal(decision.confidence)}
                      </span>
                    </div>
                    <div>
                      <span style={{ color: colors.muted }}>LSTM State: </span>
                      <span style={{ color: colors.text }}>{lstmState}</span>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Recent Incidents */}
      <div style={panelStyle}>
        <h3
          style={{
            margin: '0 0 14px 0',
            fontSize: 15,
            fontWeight: 600,
            color: colors.amber,
          }}
        >
          Recent Incidents
        </h3>
        {incidents.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13 }}>No recent incidents.</div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {incidents.map((inc: any, idx: number) => {
              const sevColor =
                inc.severity === 'critical'
                  ? colors.red
                  : inc.severity === 'warning'
                    ? colors.amber
                    : colors.muted
              return (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                    padding: '8px 10px',
                    background: 'rgba(20,32,52,0.6)',
                    borderRadius: 6,
                    borderLeft: `3px solid ${sevColor}`,
                    fontSize: 13,
                  }}
                >
                  <div style={{ flex: 1 }}>
                    <div style={{ fontWeight: 600, color: colors.text }}>
                      {inc.event ?? '--'}
                    </div>
                    <div style={{ color: colors.muted, marginTop: 2 }}>
                      {inc.summary ?? ''}
                    </div>
                  </div>
                  <div
                    style={{
                      fontSize: 11,
                      color: colors.muted,
                      whiteSpace: 'nowrap',
                      flexShrink: 0,
                    }}
                  >
                    {inc.ts ?? ''}
                  </div>
                </div>
              )
            })}
          </div>
        )}
      </div>

      {/* Trade Performance */}
      <div style={{ ...panelStyle, marginTop: 24 }}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 14, color: colors.cyan, fontWeight: 600 }}>
          Trade Performance
        </h3>
        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
            gap: 12,
          }}
        >
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Total Trades</div>
            <div style={valueStyle}>{tradeOverall?.total_trades ?? '--'}</div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Win Rate</div>
            <div style={{ ...valueStyle, color: colors.green }}>
              {tradeOverall?.win_rate != null ? `${(tradeOverall.win_rate * 100).toFixed(1)}%` : '--'}
            </div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Total PnL</div>
            <div style={{
              ...valueStyle,
              color: tradeOverall?.total_pnl != null
                ? (tradeOverall.total_pnl > 0 ? colors.green : tradeOverall.total_pnl < 0 ? colors.red : colors.text)
                : colors.text,
            }}>
              {tradeOverall?.total_pnl != null ? fmtMoney(tradeOverall.total_pnl) : '--'}
            </div>
          </div>
          <div style={{ ...panelStyle, padding: '10px 12px' }}>
            <div style={labelStyle}>Profit Factor</div>
            <div style={{ ...valueStyle, color: colors.amber }}>
              {tradeOverall?.profit_factor != null
                ? String(typeof tradeOverall.profit_factor === 'number' ? tradeOverall.profit_factor.toFixed(2) : tradeOverall.profit_factor)
                : '--'}
            </div>
          </div>
        </div>
      </div>

      {/* Recent Trades */}
      <div style={panelStyle}>
        <h3 style={{ margin: '0 0 12px 0', fontSize: 14, color: colors.cyan, fontWeight: 600 }}>
          Recent Trades
        </h3>
        {recentTrades.length === 0 ? (
          <div style={{ color: colors.muted, fontSize: 13 }}>No recent trades.</div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr>
                  {['Time', 'Symbol', 'Side', 'PnL', 'Outcome'].map(h => (
                    <th
                      key={h}
                      style={{
                        textAlign: 'left',
                        padding: '6px 10px',
                        borderBottom: `1px solid ${colors.border}`,
                        color: colors.muted,
                        fontWeight: 600,
                        fontSize: 11,
                        whiteSpace: 'nowrap',
                      }}
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {recentTrades.map((t, idx) => {
                  const profitColor = t.profit > 0 ? colors.green : t.profit < 0 ? colors.red : colors.text
                  const outcomeCol = t.outcome === 'win' ? colors.green : t.outcome === 'loss' ? colors.red : colors.amber
                  return (
                    <tr
                      key={t.ticket}
                      style={{ background: idx % 2 === 0 ? 'transparent' : 'rgba(90,215,255,0.02)' }}
                    >
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>
                        {t.close_time ?? '--'}
                      </td>
                      <td style={{ padding: '6px 10px', color: colors.cyan, fontWeight: 600 }}>
                        {t.symbol}
                      </td>
                      <td style={{
                        padding: '6px 10px',
                        color: t.side === 'BUY' ? colors.green : colors.red,
                        fontWeight: 600,
                      }}>
                        {t.side}
                      </td>
                      <td style={{ padding: '6px 10px', fontFamily: 'monospace', color: profitColor, fontWeight: 700 }}>
                        {t.profit >= 0 ? `+${fmtMoney(t.profit)}` : fmtMoney(t.profit)}
                      </td>
                      <td style={{ padding: '6px 10px' }}>
                        <span style={{
                          padding: '2px 8px',
                          borderRadius: 4,
                          fontSize: 11,
                          fontWeight: 700,
                          textTransform: 'uppercase',
                          background: `${outcomeCol}20`,
                          color: outcomeCol,
                        }}>
                          {t.outcome}
                        </span>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default DashboardPanel
