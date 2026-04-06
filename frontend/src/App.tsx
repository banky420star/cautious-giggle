import React from 'react'
import { StatusPayload, PatternRecord } from './types'
import { fetchStatus, fetchPatterns, fetchPerf, createStatusWS } from './services/api'
import DashboardPanel from './components/DashboardPanel'
import TradingPanel from './components/TradingPanel'
import TrainingPanel from './components/TrainingPanel'
import ModelsPanel from './components/ModelsPanel'
import PatternLibraryPanel from './components/PatternLibraryPanel'
import SettingsPanel from './components/SettingsPanel'

type TabId = 'home' | 'trading' | 'training' | 'models' | 'patterns' | 'settings'

const TABS: { id: TabId; label: string }[] = [
  { id: 'home', label: 'Home' },
  { id: 'trading', label: 'Trading' },
  { id: 'training', label: 'Training' },
  { id: 'models', label: 'Models' },
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

  const renderContent = () => {
    switch (activeTab) {
      case 'home':
        return status ? <DashboardPanel status={status} /> : null
      case 'trading':
        return status ? <TradingPanel status={status} /> : null
      case 'training':
        return status ? <TrainingPanel status={status} /> : null
      case 'models':
        return status ? <ModelsPanel status={status} /> : null
      case 'patterns':
        return <PatternLibraryPanel patterns={patterns} status={status} />
      case 'settings':
        return status ? <SettingsPanel status={status} /> : null
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
