import React from 'react'

interface Props { data: Array<any> | null; height?: number }
const defaultHeight = 60
const LRTimeline: React.FC<Props> = ({ data, height = defaultHeight }) => {
  if (!data || data.length === 0) {
    return <div style={{ height }}>{'No LR history'}</div>
  }
  // Extract numeric values from adaptation history: try action.total_adjustment or 0
  const pts = data.slice(-40).map((r) => {
    const a = r?.action ?? {}
    const v = typeof a.total_adjustment === 'number' ? a.total_adjustment : 0
    const t = r?.timestamp ? new Date(r.timestamp).getTime() : Date.now()
    return { t, v }
  })
  const values = pts.map((p) => p.v)
  const min = Math.min(0, ...values)
  const max = Math.max(1, ...values)
  const width = 600
  const heightPx = height
  const step = values.length > 1 ? width / (values.length - 1) : width
  const ys = values.map((v) => {
    const yNorm = (v - min) / ((max - min) || 1)
    return heightPx - yNorm * heightPx
  })
  const points = ys.map((y, i) => `${Math.round(i * step)},${Math.round(y)}`).join(' ')

  return (
    <svg width="100%" height={heightPx} viewBox={`0 0 ${width} ${heightPx}`} preserveAspectRatio="none">
      <polyline fill="none" stroke="#4fd6ff" strokeWidth={2} points={points} />
    </svg>
  )
}

export default LRTimeline
