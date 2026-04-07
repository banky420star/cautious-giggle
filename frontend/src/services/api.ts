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

/* ─── Trade History API ─── */

export interface Trade {
  ticket: number
  symbol: string
  side: string
  volume: number
  open_time: string | null
  close_time: string | null
  open_price: number
  close_price: number
  profit: number
  comment: string
  hold_minutes: number | null
  magic: number | null
  bot_lane: string
  model: string
  action_type: string
  outcome: string  // "win" | "loss" | "breakeven"
}

export interface TradesResponse {
  trades: Trade[]
  total: number
  limit: number
  offset: number
}

export interface TradeSummary {
  overall: {
    total_trades: number
    wins: number
    losses: number
    win_rate: number
    total_pnl: number
    avg_profit: number
    avg_loss: number
    profit_factor: number | string
    avg_hold_minutes: number
    max_loss_streak: number
  }
  by_symbol: Record<string, any>
}

export async function fetchTrades(params: Record<string, string> = {}): Promise<TradesResponse> {
  const qs = new URLSearchParams(params).toString()
  const r = await fetch(`${BASE}/api/trades?${qs}`, { cache: 'no-store' })
  return r.ok ? r.json() : { trades: [], total: 0, limit: 50, offset: 0 }
}

export async function fetchTradesSummary(params: Record<string, string> = {}): Promise<TradeSummary> {
  const qs = new URLSearchParams(params).toString()
  const r = await fetch(`${BASE}/api/trades/summary?${qs}`, { cache: 'no-store' })
  return r.ok ? r.json() : { overall: {} as any, by_symbol: {} }
}
