export interface PatternRecord {
  symbol?: string
  pattern_name?: string
  regime?: string
  discovered_at?: string
  count?: number
  details?: any
}

export interface PerpetualImprovementSnapshot {
  last_improvement_action?: any
  adaptation_history?: any[]
  learning_rates?: { [model: string]: { [param: string]: number } }
}

export interface AccountInfo {
  balance?: number
  equity?: number
  free_margin?: number
  open_positions?: number
  positions?: any[]
}

export interface TrainingVisual {
  state?: string
  current_symbol?: string
  progress_pct?: number
  queue?: any[]
  fail_reason?: string
}

export interface TrainingPipelineSummary {
  symbols_total?: number
  training_active_symbols?: number
  canary_review_symbols?: number
  champion_live_symbols?: number
  trading_ready_symbols?: number
  trading_active_symbols?: number
}

export interface TrainingLaneSummary {
  actionable_symbols?: number
  executed_symbols?: number
  blocked_symbols?: number
  neutral_symbols?: number
  open_positions?: number
}

export interface TrainingState {
  cycle_running?: boolean
  lstm_running?: boolean
  drl_running?: boolean
  dreamer_running?: boolean
  configured_symbols?: string[]
  visual?: {
    lstm?: TrainingVisual
    ppo?: TrainingVisual
    dreamer?: TrainingVisual
    active_label?: string
  }
  pattern_library?: { [pattern_name: string]: PatternRecord }
  symbol_stage_rows?: any[]
  symbol_lane_rows?: any[]
  pipeline_summary?: TrainingPipelineSummary
  cycle_heartbeat?: any
  lane_summary?: TrainingLaneSummary
}

export interface ServerInfo {
  running?: boolean
  pids?: number[]
}

export interface CanaryGate {
  ready?: boolean
  reason?: string
}

export interface StatusPayload {
  state?: string
  server?: ServerInfo
  account?: AccountInfo
  training?: TrainingState
  canary_gate?: CanaryGate
  active_models?: any
  incidents?: any[]
  logs?: { [key: string]: string }
  registry_summary?: any
  telegram?: any
  repo_root?: string
}
