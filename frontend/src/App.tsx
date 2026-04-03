import React from 'react'
import PatternLibraryPanel from './components/PatternLibraryPanel'
import PerpetualPanel from './components/PerpetualPanel'
import TrainingPanel from './components/TrainingPanel'
import { PatternRecord, PerpetualImprovementSnapshot } from './types'
import { fetchStatus, fetchPatterns, fetchPerf } from './services/api'

const App: React.FC = () => {
  const [status, setStatus] = React.useState<any>(null)
  const [patterns, setPatterns] = React.useState<PatternRecord[]>([])
  const [perf, setPerf] = React.useState<PerpetualImprovementSnapshot | null>(null)
  const [route, setRoute] = React.useState<string>(window.location.hash.replace('#','')?.replace('/', '') || 'overview')

  React.useEffect(() => {
    const load = async () => {
      try { setStatus(await fetchStatus()) } catch {}
      try { setPatterns(await fetchPatterns()) } catch {}
      try { setPerf(await fetchPerf()) } catch {}
    }
    load()
    const t = setInterval(load, 15000)
    window.addEventListener('hashchange', () => setRoute(window.location.hash.replace('#','').replace('/', '') || 'overview'))
    return () => { clearInterval(t); window.removeEventListener('hashchange', () => {}) }
  }, [])

  const renderOverview = () => (
    <>
      <PatternLibraryPanel patterns={patterns} status={status} />
      <PerpetualPanel perf={perf} />
      <TrainingPanel />
    </>
  )

  const renderPatterns = () => (
    <PatternLibraryPanel patterns={patterns} status={status} />
  )

  const renderPerp = () => (
    <PerpetualPanel perf={perf} />
  )

  const renderTraining = () => (
    <TrainingPanel />
  )

  return (
    <div style={{ padding: 20 }}>
      <h1>Phase 2 SPA — Pattern & Perpetual Improvement</h1>
      <nav style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        <a href="#/overview">Overview</a>
        <a href="#/patterns">Patterns</a>
        <a href="#/perp">Perp</a>
        <a href="#/training">Training</a>
      </nav>
      {route === 'patterns' && renderPatterns()}
      {route === 'perp' && renderPerp()}
      {route === 'training' && renderTraining()}
      {(route === 'overview' || route === '') && renderOverview()}
    </div>
  )
}

export default App
