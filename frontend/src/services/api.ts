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

/* ─── PPO Diagnostics API ─── */

export interface PPODiagnostics {
  ppo_loaded: boolean
  obs_shape: number[] | null
  action_shape: number[] | null
  is_canary: boolean
  device: string
  champion_path: string
  canary_path: string
  model_version: string
  last_actions: Record<string, any>
}

export async function fetchPPODiagnostics(): Promise<PPODiagnostics | null> {
  const r = await fetch(`${BASE}/api/ppo_diagnostics`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

/* ─── LSTM Explanations API ─── */

export interface LSTMExplanation {
  regime: string
  confidence: number
  top_indicators: Array<{ indicator: string; importance: number }>
  cached_at: number | null
}

export async function fetchLSTMExplanations(): Promise<Record<string, LSTMExplanation>> {
  const r = await fetch(`${BASE}/api/lstm_explanations`, { cache: 'no-store' })
  if (!r.ok) return {}
  const data = await r.json()
  return data?.symbols ?? {}
}

/* ─── Learning Pipeline API ─── */

export interface LearningStatus {
  canary: {
    active: boolean
    path: string | null
    version: string | null
    scorecard: Record<string, any>
  }
  champion: {
    path: string | null
    version: string | null
    scorecard: Record<string, any>
  }
  candidates: Array<{
    version: string
    path: string
    win_rate: number | null
    loss: number | null
    saved_at: string | null
    type: string | null
  }>
  training_schedule: {
    enabled: boolean
    interval_sec: number
    auto_canary: boolean
  }
  learning_log: any
}

export async function fetchLearning(): Promise<LearningStatus | null> {
  const r = await fetch(`${BASE}/api/learning`, { cache: 'no-store' })
  return r.ok ? r.json() : null
}

/* ─── Scenarios / Regime API ─── */

export async function fetchScenarios(): Promise<any> {
  const r = await fetch(`${BASE}/api/scenarios`, { cache: 'no-store' })
  return r.ok ? r.json() : { regimes: {} }
}

/* ─── Lanes API ─── */

export interface LaneStatus {
  symbol: string
  champion: string
  canary: string | null
  action: string
  exposure: number
  confidence: number
  volatility: string
  reason: string
  can_trade: boolean
  is_canary: boolean
  last_decision_at: number | null
  recent_decisions: number
}

export async function fetchLanes(): Promise<{ lanes: LaneStatus[] }> {
  const r = await fetch(`${BASE}/api/lanes`, { cache: 'no-store' })
  return r.ok ? r.json() : { lanes: [] }
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
