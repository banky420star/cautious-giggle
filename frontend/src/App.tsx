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

  React.useEffect(() => {
    fetchStatus().then(setStatus).catch(() => setStatus(null))
    fetchPatterns().then(setPatterns).catch(() => setPatterns([]))
    fetchPerf().then(setPerf).catch(() => setPerf(null))
  }, [])

  return (
    <div style={{ padding: 20 }}>
      <h1>Phase 2 SPA — Pattern & Perpetual Improvement</h1>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <PatternLibraryPanel patterns={patterns} status={status} />
        <PerpetualPanel perf={perf} />
      </div>
      <TrainingPanel />
    </div>
  )
}

export default App
