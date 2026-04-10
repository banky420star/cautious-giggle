import React from 'react'
import { PatternRecord } from '../types'

interface Props {
  patterns: PatternRecord[]
  status: any
}
const PatternLibraryPanel: React.FC<Props> = ({ patterns, status }) => {
  const [query, setQuery] = React.useState('')
  const [selected, setSelected] = React.useState<PatternRecord | null>(null)
  const filtered = patterns.filter((p) => {
    const s = (p.pattern_name || '') + ' ' + (p.symbol || '')
    return s.toLowerCase().includes(query.toLowerCase())
  })
  return (
    <section style={{ padding: 16, marginBottom: 16 }}>
      <h2>Pattern Library ({patterns.length} patterns)</h2>
      <input
        type="text"
        placeholder="Search patterns..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        style={{ padding: '8px 10px', borderRadius: 6, border: '1px solid #334', width: '100%', marginBottom: 8 }}
      />
      <div style={{ 
        border: '1px solid #334', 
        borderRadius: 8, 
        padding: 12, 
        maxHeight: 300, 
        overflowY: 'auto',
        background: '#0a111a'
      }}>
        {filtered.length === 0 ? (
          <div style={{ color: '#888', textAlign: 'center', padding: 20 }}>
            No patterns discovered yet
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #444' }}>Symbol</th>
                <th style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #444' }}>Pattern</th>
                <th style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #444' }}>Regime</th>
                <th style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #444' }}>Discovered</th>
                <th style={{ textAlign: 'left', padding: 8, borderBottom: '1px solid #444' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p, idx) => (
                <tr key={idx} style={{ background: idx % 2 === 0 ? '#0a111a' : '#080e16', cursor: 'pointer' }} onClick={() => setSelected(p)}>
                  <td style={{ padding: 8 }}>{p.symbol ?? '-'}</td>
                  <td style={{ padding: 8 }}>{p.pattern_name ?? '-'}</td>
                  <td style={{ padding: 8 }}>{p.regime ?? '-'}</td>
                  <td style={{ padding: 8 }}>{new Date(p.discovered_at || 0).toLocaleString()}</td>
                  <td style={{ padding: 8 }}>{p.count ?? 0}</td>
                </tr>
              ))}
            </tbody>
          </table>
            
            {/* Right column: detail of selected pattern */}
            <div style={{ padding: 6, borderLeft: '1px solid #334' }}>
              <h4>Pattern Details</h4>
              {selected ? (
                <pre style={{ whiteSpace: 'pre-wrap', maxHeight: 240, overflow: 'auto' }}>{JSON.stringify(selected, null, 2)}</pre>
              ) : (
                <div style={{ color: '#888' }}>Select a pattern to view details</div>
              )}
            </div>
          </div>
        )}
      </div>
    </section>
  )
}
export default PatternLibraryPanel
