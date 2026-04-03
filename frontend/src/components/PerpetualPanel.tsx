import React from 'react'
import { PerpetualImprovementSnapshot } from '../types'
import LRTimeline from './LRTimeline'

interface Props { perf: PerpetualImprovementSnapshot | null }
const PerpetualPanel: React.FC<Props> = ({ perf }) => {
  if (!perf) {
    return (
      <section className="card" style={{ padding: 16 }}>
        <h2>Perpetual Improvement</h2>
        <div>No data available</div>
      </section>
    )
  }

  const lastAction = perf.last_improvement_action || {}
  const adaptationHistory = perf.adaptation_history || []
  const learningRates = perf.learning_rates || {}

  return (
    <section className="card" style={{ padding: 16 }}>
      <h2>Perpetual Improvement</h2>
      
      {/* Last Improvement Action */}
      <div style={{ marginBottom: 16 }}>
        <h3>Last Improvement Action</h3>
        {Object.keys(lastAction).length === 0 ? (
          <div>No actions recorded</div>
        ) : (
          <div style={{ background: '#0a111a', padding: 12, borderRadius: 6 }}>
            <div><strong>Model:</strong> {lastAction.model_type ?? 'N/A'} {lastAction.symbol ?? ''}</div>
            <div><strong>Timestamp:</strong> {new Date(lastAction.timestamp || 0).toLocaleString()}</div>
            <div><strong>Performance Trend:</strong> {(lastAction.performance_trend ?? 0).toFixed(3)}</div>
            <div><strong>Pattern Adjustment:</strong> {(lastAction.pattern_adjustment ?? 0).toFixed(3)}</div>
            <div><strong>Total Adjustment:</strong> {(lastAction.total_adjustment ?? 0).toFixed(3)}</div>
          </div>
        )}
      </div>

      {/* Learning Rates */}
      <div style={{ marginBottom: 16 }}>
        <h3>Current Learning Rates</h3>
        {Object.keys(learningRates).length === 0 ? (
          <div>No learning rates configured</div>
        ) : (
          <div style={{ background: '#0a111a', padding: 12, borderRadius: 6 }}>
            {Object.entries(learningRates).map(([model, params], idx) => (
              <div key={idx} style={{ marginBottom: 8 }}>
                <div><strong>{model.toUpperCase()}:</strong></div>
                <div style={{ paddingLeft: 16 }}>
                  {Object.entries(params).map(([param, value], paramIdx) => (
                    <div key={paramIdx}>
                      <span style={{ fontFamily: 'monospace' }}>{param}:</span> 
                      <span style={{ color: '#4fd6ff', fontWeight: '600' }}>{value.toExponential(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Adaptation History + LR Timeline */}
      <div>
        <h3>Adaptation History</h3>
        {adaptationHistory.length === 0 ? (
          <div>No history recorded</div>
        ) : (
          <div>
            <div style={{ background: '#0a111a', padding: 12, borderRadius: 6, maxHeight: 200, overflowY: 'auto' }}>
              {adaptationHistory.slice(-10).map((record, idx) => (
                <div key={idx} style={{ padding: 8, marginBottom: 4, borderBottom: '1px solid #222', background: idx % 2 === 0 ? '#080e16' : 'transparent' }}>
                  <div style={{ fontSize: 12, color: '#888' }}>
                    {new Date(record.timestamp || 0).toLocaleTimeString()}
                  </div>
                  <div style={{ fontSize: 12 }}>
                    {record.action?.model_type ?? 'N/A'} {record.action?.symbol ?? ''}: 
                    Trend: {(record.action?.performance_trend ?? 0).toFixed(3)}, 
                    Adjust: {(record.action?.total_adjustment ?? 0).toFixed(3)}
                  </div>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 8 }}>
              <LRTimeline data={adaptationHistory} height={60} />
            </div>
          </div>
        )}
      </div>
    </section>
  )
}

export default PerpetualPanel
