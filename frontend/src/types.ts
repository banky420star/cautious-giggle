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

export interface TrainingState {
  cycle_running?: boolean
  configured_symbols?: string[]
  visual?: {
    lstm?: TrainingVisual
    ppo?: TrainingVisual
    dreamer?: TrainingVisual
  }
  symbol_stage_rows?: any[]
  symbol_lane_rows?: any[]
  pipeline_summary?: any
  cycle_heartbeat?: any
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
