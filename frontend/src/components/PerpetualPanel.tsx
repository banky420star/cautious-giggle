import React from 'react'
import { PerpetualImprovementSnapshot } from '../types'

interface Props { perf: PerpetualImprovementSnapshot | null }
const PerpetualPanel: React.FC<Props> = ({ perf }) => {
  return (
    <section className="card" style={{ padding: 16 }}>
      <h2>Perpetual Improvement</h2>
      <pre style={{ maxHeight: 200, overflow: 'auto' }}>{perf ? JSON.stringify(perf, null, 2) : '{}'}</pre>
    </section>
  )
}
export default PerpetualPanel
