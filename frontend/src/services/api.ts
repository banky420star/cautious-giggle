import { PatternRecord, StatusPayload } from '../types'

const BASE = ''  // relative — Vite proxy or same-origin in production

export async function fetchStatus(): Promise<StatusPayload> {
  const r = await fetch(`${BASE}/api/status`, { cache: 'no-store' })
  return r.ok ? r.json() : {}
}

export async function fetchPatterns(): Promise<PatternRecord[]> {
  const r = await fetch(`${BASE}/api/patterns`)
  if (!r.ok) return []
  const data = await r.json()
  if (Array.isArray(data)) return data
  if (Array.isArray(data?.patterns)) return data.patterns
  return []
}

export async function fetchPerf(): Promise<any> {
  const r = await fetch(`${BASE}/api/perf`)
  return r.ok ? r.json() : null
}

export async function controlAction(action: string, payload?: Record<string, any>): Promise<any> {
  const r = await fetch(`${BASE}/api/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action, ...payload }),
  })
  return r.json()
}

export function createStatusWS(onMessage: (data: StatusPayload) => void): WebSocket | null {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const ws = new WebSocket(`${proto}//${location.host}/ws`)
  ws.onmessage = (ev) => {
    try { onMessage(JSON.parse(ev.data)) } catch {}
  }
  return ws
}
