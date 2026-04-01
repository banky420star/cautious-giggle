import React from 'react'
import { PatternRecord } from '../types'

interface Props {
  patterns: PatternRecord[]
  status: any
}
const PatternLibraryPanel: React.FC<Props> = ({ patterns, status }) => {
  return (
    <section className="card" style={{ padding: 16 }}>
      <h2>Pattern Library</h2>
      <div style={{ maxHeight: 240, overflow: 'auto', border: '1px solid #334', borderRadius: 8, padding: 8 }}>
        {patterns.length === 0 ? (
          <div>No patterns yet</div>
        ) : (
          <ul>
            {patterns.map((p, idx) => (
              <li key={idx}>{p.pattern_name || p.symbol || 'pattern'}</li>
            ))}
          </ul>
        )}
      </div>
    </section>
  )
}
export default PatternLibraryPanel
