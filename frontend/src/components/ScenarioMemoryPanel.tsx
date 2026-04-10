import React from 'react'

const ScenarioMemoryPanel: React.FC = () => {
  const [scenarios, setScenarios] = React.useState<any[]>([])
  const [loading, setLoading] = React.useState(true)

  React.useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch('/api/scenarios')
        if (res.ok) setScenarios(await res.json())
      } catch { /* API not yet wired */ }
      setLoading(false)
    }
    load()
    const interval = setInterval(load, 15_000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div>
      <h2 style={{ fontSize: 20, fontWeight: 600, marginBottom: 16 }}>Scenario Memory</h2>
      {loading ? (
        <p style={{ color: '#8899aa' }}>Loading scenarios...</p>
      ) : scenarios.length === 0 ? (
        <p style={{ color: '#8899aa' }}>No scenario data available yet. The learning pipeline will populate this as trades are analyzed.</p>
      ) : (
        <div style={{ display: 'grid', gap: 12 }}>
          {scenarios.map((s, i) => (
            <div key={i} style={{ background: '#0f1d2e', borderRadius: 8, padding: 16, border: '1px solid #1a2a3e' }}>
              <div style={{ fontWeight: 600, color: '#5ad7ff', marginBottom: 4 }}>{s.scenario || s.label || `Scenario ${i + 1}`}</div>
              <div style={{ fontSize: 13, color: '#8899aa' }}>
                {s.win_rate != null && <span>Win Rate: {(s.win_rate * 100).toFixed(1)}% | </span>}
                {s.avg_pnl != null && <span>Avg PnL: ${s.avg_pnl.toFixed(2)} | </span>}
                {s.trade_count != null && <span>Trades: {s.trade_count}</span>}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ScenarioMemoryPanel
