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
