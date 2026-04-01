import { PatternRecord } from '../types'

export async function fetchStatus(): Promise<any> {
  const r = await fetch('/api/status')
  return r.ok ? r.json() : {}
}

export async function fetchPatterns(): Promise<PatternRecord[]> {
  const r = await fetch('/api/patterns')
  if (!r.ok) return []
  const data = await r.json()
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.patterns)) return data.patterns
  return []
}

export async function fetchPerf(): Promise<any> {
  const r = await fetch('/api/perf')
  return r.ok ? r.json() : null
}
