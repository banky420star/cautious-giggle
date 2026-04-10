import React from 'react'
import { StatusPayload, PatternRecord } from './types'
import { fetchStatus, fetchPatterns, fetchPerf, createStatusWS } from './services/api'
import DashboardPanel from './components/DashboardPanel'
import TradingPanel from './components/TradingPanel'
import TrainingPanel from './components/TrainingPanel'
import ModelsPanel from './components/ModelsPanel'
import PatternLibraryPanel from './components/PatternLibraryPanel'
import SettingsPanel from './components/SettingsPanel'
import TradeHistoryPanel from './components/TradeHistoryPanel'
import PPODiagPanel from './components/PPODiagPanel'
import HFTHealthPanel from './components/HFTHealthPanel'
import PerpetualPanel from './components/PerpetualPanel'
import LRTimeline from './components/LRTimeline'
import ScenarioMemoryPanel from './components/ScenarioMemoryPanel'

type TabId = 'home' | 'trading' | 'training' | 'models' | 'patterns' | 'settings' | 'trades' | 'ppo' | 'hft' | 'perpetual' | 'lr-timeline' | 'scenarios'

const TABS: { id: TabId; label: string }[] = [
  { id: 'home', label: 'Home' },
  { id: 'trading', label: 'Trading' },
  { id: 'trades', label: 'Trade History' },
  { id: 'training', label: 'Training' },
  { id: 'models', label: 'Models' },
  { id: 'ppo', label: 'PPO Brain' },
  { id: 'hft', label: 'HFT Health' },
  { id: 'scenarios', label: 'Scenarios' },
  { id: 'perpetual', label: 'Perpetual' },
  { id: 'lr-timeline', label: 'LR Timeline' },
  { id: 'patterns', label: 'Patterns' },
  { id: 'settings', label: 'Settings' },
]

const App: React.FC = () => {
  const [status, setStatus] = React.useState<StatusPayload | null>(null)
  const [patterns, setPatterns] = React.useState<PatternRecord[]>([])
  const [perf, setPerf] = React.useState<any>(null)
  const [activeTab, setActiveTab] = React.useState<TabId>('home')

  React.useEffect(() => {
    const loadData = async () => {
      try { setStatus(await fetchStatus()) } catch { /* ignore */ }
      try { setPatterns(await fetchPatterns()) } catch { /* ignore */ }
      try { setPerf(await fetchPerf()) } catch { /* ignore */ }
    }

    // Initial fetch
    loadData()

    // Refresh every 10 seconds
    const interval = setInterval(loadData, 10_000)

    // Real-time status via WebSocket
    const ws = createStatusWS((data) => {
      setStatus(data)
    })

    return () => {
      clearInterval(interval)
      if (ws) ws.close()
    }
  }, [])

  React.useEffect(() => {
    const lib = status?.training?.pattern_library
    if (!lib) {
      return
    }
    const records: PatternRecord[] = Object.entries(lib)
      .map(([pattern_name, payload]) => ({
        pattern_name,
        ...(payload || {}),
      }))
      .sort((a, b) => new Date(b.discovered_at || 0).getTime() - new Date(a.discovered_at || 0).getTime())
      .slice(0, 2)
    setPatterns(records)
  }, [status])

  // Intro / loading screen while waiting for first API response
  if (!status) {
    return (
      <div style={{
        minHeight: '100vh',
        background: '#07111d',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
      }}>
        <div style={{
          width: 64, height: 64, borderRadius: '50%',
          border: '3px solid #162638', borderTopColor: '#5ad7ff',
          animation: 'spin 1s linear infinite',
          marginBottom: 28,
        }} />
        <h1 style={{ color: '#5ad7ff', fontSize: 28, fontWeight: 700, margin: '0 0 8px', letterSpacing: '-0.5px' }}>
          AGI Trading System
        </h1>
        <p style={{ color: '#8899aa', fontSize: 15, margin: 0 }}>
          Connecting to backend...
        </p>
        <div style={{ marginTop: 32, display: 'flex', gap: 16, fontSize: 12, color: '#556677' }}>
          <span>PPO + LSTM Hybrid</span>
          <span>|</span>
          <span>MetaTrader 5 Live</span>
          <span>|</span>
          <span>3 Symbols</span>
        </div>
        <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
      </div>
    )
  }

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return <DashboardPanel status={status} />
      case 'trading':
        return <TradingPanel status={status} />
      case 'trades':
        return <TradeHistoryPanel />
      case 'training':
        return <TrainingPanel status={status} />
      case 'models':
        return <ModelsPanel status={status} />
      case 'ppo':
        return <PPODiagPanel status={status} />
      case 'hft':
        return <HFTHealthPanel status={status} />
      case 'scenarios':
        return <ScenarioMemoryPanel />
      case 'perpetual':
        return <PerpetualPanel perf={perf} />
      case 'lr-timeline':
        return <LRTimeline data={perf?.adaptation_history ?? null} height={200} />
      case 'patterns':
        return <PatternLibraryPanel patterns={patterns} status={status} />
      case 'settings':
        return <SettingsPanel status={status} />
      default:
        return null
    }
  }

  return (
    <div
      style={{
        minHeight: '100vh',
        background: '#07111d',
        color: '#eef5ff',
        fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
      }}
    >
      {/* Tab Navigation Bar */}
      <nav
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '14px 24px',
          borderBottom: '1px solid #162638',
          background: '#0a1628',
        }}
      >
        <span
          style={{
            fontWeight: 700,
            fontSize: 18,
            color: '#5ad7ff',
            marginRight: 24,
            letterSpacing: '-0.3px',
          }}
        >
          Trading Dashboard
        </span>

        {TABS.map((tab) => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: '7px 18px',
                borderRadius: 12,
                border: 'none',
                cursor: 'pointer',
                fontSize: 14,
                fontWeight: isActive ? 600 : 400,
                background: isActive ? '#5ad7ff' : 'transparent',
                color: isActive ? '#07111d' : '#8899aa',
                transition: 'background 0.15s, color 0.15s',
              }}
            >
              {tab.label}
            </button>
          )
        })}
      </nav>

      {/* Tab Content */}
      <main style={{ padding: '24px 28px', maxWidth: 1400, margin: '0 auto' }}>
        {renderContent()}
      </main>
    </div>
  )
}

export default App
