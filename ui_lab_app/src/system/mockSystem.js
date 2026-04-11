const ROUTE_PHASES = ["context", "simulation", "policy", "selection", "authority"];

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function createLane(symbol, side, champion, reason, confidence, exposure, pnl, status) {
  return {
    symbol,
    side,
    champion,
    reason,
    confidence,
    exposure,
    pnl,
    status,
    canTrade: true,
  };
}

// Pattern recognition and learning enhancement functions
function detectMarketRegime(priceHistory, volumeHistory) {
  if (!priceHistory || priceHistory.length < 10) return "unknown";
  
  // Simple regime detection based on volatility and trend
  const returns = [];
  for (let i = 1; i < priceHistory.length; i++) {
    returns.push((priceHistory[i] - priceHistory[i-1]) / priceHistory[i-1]);
  }
  
  const volatility = Math.sqrt(returns.reduce((sum, r) => sum + r*r, 0) / returns.length);
  const trend = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  
  if (volatility > 0.03) {
    if (trend > 0.001) return "high_volatility_bull";
    if (trend < -0.001) return "high_volatility_bear";
    return "high_volatility_neutral";
  } else {
    if (trend > 0.0005) return "low_volatility_bull";
    if (trend < -0.0005) return "low_volatility_bear";
    return "low_volatility_neutral";
  }
}

function updatePatternRecognitionSystem(system, tick) {
  // Simulate pattern recognition learning
  const patterns = system.patternRecognition || {
    knownPatterns: [],
    patternSuccessRates: {},
    marketRegimeHistory: [],
    learningRate: 0.01,
    adaptationSpeed: 0.005
  };
  
  // Generate mock price/volume history for regime detection
  const priceBase = 50000 + Math.sin(tick / 10) * 2000;
  const volumeBase = 1000 + Math.cos(tick / 7) * 300;
  
  const priceHistory = [];
  const volumeHistory = [];
  for (let i = 0; i < 20; i++) {
    const offset = i * 0.5;
    priceHistory.push(priceBase + Math.sin((tick - offset) / 12) * 150 + (Math.random() - 0.5) * 100);
    volumeHistory.push(volumeBase + Math.cos((tick - offset) / 9) * 200 + (Math.random() - 0.5) * 50);
  }
  
  const currentRegime = detectMarketRegime(priceHistory, volumeHistory);
  
  // Set active indicator bundles based on regime
  system.indicatorBundles = [
    { id: "BNDL-VOL", name: "Volatility Accelerator", scenario: "high_volatility", components: ["Keltner Channels", "ATR", "Volume Profile"], active: currentRegime.includes("volatility"), winRate: 0.68 },
    { id: "BNDL-MEAN", name: "Mean Reversion Core", scenario: "low_volatility", components: ["RSI Divergence", "Bollinger Bands", "VWAP"], active: currentRegime.includes("neutral"), winRate: 0.72 },
    { id: "BNDL-TREND", name: "Momentum Pipeline", scenario: "trend_following", components: ["MACD", "EMA Cross", "ADX"], active: currentRegime.includes("bull") || currentRegime.includes("bear"), winRate: 0.65 }
  ];

  // Update regime history
  patterns.marketRegimeHistory.push({
    regime: currentRegime,
    tick,
    timestamp: Date.now()
  });
  
  // Set active indicator bundles based on regime
  system.indicatorBundles = [
    { id: "BNDL-VOL", name: "Breakout Accelerator", scenario: "high_volatility_bull", components: ["Keltner Channels", "ATR", "Volume Profile"], active: currentRegime.includes("volatility"), winRate: 0.68 },
    { id: "BNDL-MEAN", name: "Mean Reversion Core", scenario: "low_volatility_neutral", components: ["RSI Divergence", "Bollinger Bands", "VWAP"], active: currentRegime.includes("neutral"), winRate: 0.72 },
    { id: "BNDL-CAP", name: "Capitulation Support", scenario: "high_volatility_bear", components: ["MACD Extended", "Fibonacci Retracement", "OBV"], active: currentRegime.includes("bear"), winRate: 0.54 }
  ];

  // Keep only last 50 regime observations
  if (patterns.marketRegimeHistory.length > 50) {
    patterns.marketRegimeHistory.shift();
  }
  
  // Learn from recent outcomes (simplified)
  const recentOutcome = Math.random() > 0.4 ? "success" : "failure"; // Simulate trade outcome
  const patternKey = `${currentRegime}_${system.training.activePhase}`;
  
  if (!patterns.patternSuccessRates[patternKey]) {
    patterns.patternSuccessRates[patternKey] = {
      successCount: 0,
      totalCount: 0,
      rate: 0.5
    };
  }
  
  patterns.patternSuccessRates[patternKey].totalCount++;
  if (recentOutcome === "success") {
    patterns.patternSuccessRates[patternKey].successCount++;
  }
  
  patterns.patternSuccessRates[patternKey].rate = 
    patterns.patternSuccessRates[patternKey].successCount / 
    patterns.patternSuccessRates[patternKey].totalCount;
  
  // Add new discovered patterns occasionally
  if (tick % 37 === 0 && Math.random() > 0.7) {
    const newPattern = {
      id: `pattern_${Date.now()}`,
      regime: currentRegime,
      phase: system.training.activePhase,
      confidence: 0.6 + Math.random() * 0.3,
      discoveredAt: tick
    };
    patterns.knownPatterns.push(newPattern);
    
    // Keep only last 20 patterns
    if (patterns.knownPatterns.length > 20) {
      patterns.knownPatterns.shift();
    }
  }
  
  return patterns;
}

function applyPerpetualImprovements(system, tick) {
  // Apply learning-based improvements to training parameters
  const improvements = system.perpetualImprovements || {
    learningAdjustments: {
      lstm: { learningRate: 0.001, memoryRetention: 0.9 },
      ppo: { learningRate: 0.0003, entropyCoeff: 0.01 },
      dreamer: { learningRate: 0.0001, imaginationRatio: 0.5 }
    },
    adaptationHistory: [],
    improvementRate: 0.0005
  };
  
  // Simulate learning from pattern recognition success
  const patternRecognition = system.patternRecognition || {};
  const currentPhase = system.training.activePhase;
  let phaseSuccessRate = 0.5; // default
  
  // Find success rate for current phase across regimes
  let regimeMatches = 0;
  let successSum = 0;
  
  for (const [patternKey, stats] of Object.entries(patternRecognition.patternSuccessRates || {})) {
    if (patternKey.endsWith(`_${currentPhase}`)) {
      regimeMatches++;
      successSum += stats.rate;
    }
  }
  
  if (regimeMatches > 0) {
    phaseSuccessRate = successSum / regimeMatches;
  }
  
  // Adjust learning rates based on success
  const successFactor = phaseSuccessRate - 0.5; // -0.5 to +0.5 range
  const learningAdjustment = improvements.improvementRate * successFactor;
  
  // Apply adjustments to different model types
  if (currentPhase === "context") {
    improvements.learningAdjustments.lstm.learningRate *= (1 + learningAdjustment * 0.1);
    improvements.learningAdjustments.lstm.memoryRetention *= (1 - learningAdjustment * 0.05); // inverse for retention
    
    // Keep within reasonable bounds
    improvements.learningAdjustments.lstm.learningRate = clamp(
      improvements.learningAdjustments.lstm.learningRate, 0.0001, 0.01
    );
    improvements.learningAdjustments.lstm.memoryRetention = clamp(
      improvements.learningAdjustments.lstm.memoryRetention, 0.8, 0.99
    );
  } else if (currentPhase === "policy") {
    improvements.learningAdjustments.ppo.learningRate *= (1 + learningAdjustment * 0.1);
    improvements.learningAdjustments.ppo.entropyCoeff *= (1 - learningAdjustment * 0.05);
    
    improvements.learningAdjustments.ppo.learningRate = clamp(
      improvements.learningAdjustments.ppo.learningRate, 0.00001, 0.005
    );
    improvements.learningAdjustments.ppo.entropyCoeff = clamp(
      improvements.learningAdjustments.ppo.entropyCoeff, 0.001, 0.1
    );
  } else if (currentPhase === "simulation") {
    improvements.learningAdjustments.dreamer.learningRate *= (1 + learningAdjustment * 0.1);
    improvements.learningAdjustments.dreamer.imaginationRatio *= (1 + learningAdjustment * 0.05);
    
    improvements.learningAdjustments.dreamer.learningRate = clamp(
      improvements.learningAdjustments.dreamer.learningRate, 0.000005, 0.002
    );
    improvements.learningAdjustments.dreamer.imaginationRatio = clamp(
      improvements.learningAdjustments.dreamer.imaginationRatio, 0.1, 0.9
    );
  }
  
  // Record improvement
  improvements.adaptationHistory.push({
    tick,
    phase: currentPhase,
    successRate: phaseSuccessRate,
    learningAdjustment,
    timestamp: Date.now()
  });
  
  // Keep only last 30 adaptations
  if (improvements.adaptationHistory.length > 30) {
    improvements.adaptationHistory.shift();
  }
  
  return improvements;
}

export function createInitialSystem() {
  return {
    meta: {
      appName: "Cautious Giggle",
      featureVersion: "ultimate_150",
      dreamerVersion: "DreamerV3",
      transportMode: "mock",
    },
    connection: {
      status: "connected",
      transport: "mock",
      latencyMs: 42,
      lastSyncAt: Date.now(),
      stale: false,
    },
    orchestrator: {
      loopStatus: "healthy",
      owner: "champion_cycle",
      autoImproveEnabled: true,
      currentPhase: "context",
      cycleProgress: 0.28,
      nextAction: "continue_lstm_training",
      queueDepth: 3,
      loopIteration: 12,
      cooldownSec: 34,
    },
    training: {
      featureVersion: "ultimate_150",
      activePhase: "context",
      lstm: {
        currentSymbol: "BTCUSDm",
        epoch: 3,
        epochsTotal: 20,
        loss: 1.42,
        valLoss: 1.51,
        memoryStrength: 0.62,
        featuresUsed: 150,
        queue: ["BTCUSDm", "XAUUSDm"],
      },
      ppo: {
        currentSymbol: "BTCUSDm",
        episode: 4,
        progress: 0.39,
        currentTimesteps: 195000,
        targetTimesteps: 500000,
        policyLoss: 0.83,
        valueLoss: 0.71,
        entropy: 0.58,
        dominantAction: "long",
        etaSec: 740,
      },
      dreamerV3: {
        enabled: true,
        currentSymbol: "BTCUSDm",
        progress: 0.26,
        steps: 5000,
        window: 64,
        worldModelLoss: 1.12,
        alignment: 0.38,
        blendWeight: 0.15,
      },
      pipeline: {
        trainingActiveSymbols: 2,
        canaryReviewSymbols: 1,
        tradingReadySymbols: 2,
        tradingActiveSymbols: 1,
      },
    },
    registry: {
      champion: {
        id: "BTC_014",
        symbol: "BTCUSDm",
        featureVersion: "ultimate_150",
        verdict: "champion",
      },
      canary: {
        id: "XAU_022",
        symbol: "XAUUSDm",
        featureVersion: "ultimate_150",
        verdict: "canary",
        progress: 0.72,
      },
      gate: {
        ready: false,
        reason: "min trades not yet reached",
      },
      candidates: [
        { id: "BTC_014", symbol: "BTCUSDm", verdict: "champion", sharpe: 1.84, drawdown: -4.7, featureVersion: "ultimate_150" },
        { id: "XAU_022", symbol: "XAUUSDm", verdict: "canary", sharpe: 1.21, drawdown: -6.4, featureVersion: "ultimate_150" },
        { id: "BTC_017", symbol: "BTCUSDm", verdict: "testing", sharpe: 1.48, drawdown: -5.6, featureVersion: "ultimate_150" },
        { id: "EUR_009", symbol: "EURUSDm", verdict: "rejected", sharpe: 0.62, drawdown: -10.8, featureVersion: "engineered_v2" },
      ],
      lineage: [
        { id: "BTC_014", from: "BTC_011", when: "18m ago", reason: "Sharpe and canary survival exceeded thresholds" },
        { id: "BTC_011", from: "BTC_009", when: "yesterday", reason: "Improved drawdown stability" },
      ],
    },
    trading: {
      mode: "armed",
      account: {
        connected: true,
        balance: 12432.18,
        equity: 12418.82,
        freeMargin: 10219.17,
        floatingPnl: -13.36,
        realizedToday: 92.41,
        openPositions: 2,
      },
      risk: {
        canTrade: true,
        drawdownPct: 1.8,
        dailyLossPct: 0.7,
        maxDailyLossPct: 3.0,
        sizeCap: 0.64,
        killSwitchArmed: true,
      },
      lanes: [
        createLane(
          "BTCUSDm",
          "short",
          "BTC_014",
          "Context memory, DreamerV3 pressure, and PPO bias all agree that BTCUSDm should stay short under active authority.",
          0.67,
          -0.42,
          -8.24,
          "live"
        ),
        createLane(
          "XAUUSDm",
          "long",
          "XAU_011",
          "XAUUSDm is still in live review. The lane looks constructive, but production authority has not fully transferred yet.",
          0.59,
          0.31,
          -5.12,
          "watching"
        ),
      ],
      tradeHistory: [
        { id: "T-091", symbol: "BTCUSDm", type: "close_long", pnl: 42.15, duration: "1h 42m", reason: "PPO confidence fell below 0.35 threshold. Exited early to protect accumulated drift profit.", timestamp: "10m ago", model: "BTC_014" },
        { id: "T-090", symbol: "XAUUSDm", type: "stop_loss", pnl: -15.42, duration: "12m", reason: "DreamerV3 risk scenario hit exactly 90% certainty on incoming volatility expansion.", timestamp: "45m ago", model: "XAU_011" },
        { id: "T-091", symbol: "BTCUSDm", type: "close_long", pnl: 42.15, duration: "1h 42m", reason: "PPO confidence fell below 0.35 threshold. Exited early to protect accumulated drift profit.", timestamp: "10m ago", model: "BTC_014" },
        { id: "T-090", symbol: "XAUUSDm", type: "stop_loss", pnl: -15.42, duration: "12m", reason: "DreamerV3 risk scenario hit exactly 90% certainty on incoming volatility expansion.", timestamp: "45m ago", model: "XAU_011" },
        { id: "T-089", symbol: "BTCUSDm", type: "partial_close", pnl: 18.20, duration: "2h 15m", reason: "Rebalance rule triggered. Booked 50% profit approaching 65k psychological resistance.", timestamp: "2h ago", model: "BTC_014" },
        { id: "T-088", symbol: "EURUSDm", type: "take_profit", pnl: 8.90, duration: "4h 0m", reason: "LSTM context memory aligned perfectly with mean reversion expectation zone.", timestamp: "5h ago", model: "EUR_009" }
      ],
    },
    selfImprove: {
      loopHealth: "stable",
      modelPromotionsToday: 1,
      canaryPassRate: 0.67,
      feedbackCoverage: 0.81,
      evaluationWindow: "60d",
      lastImprovementAction: "Reduced BTCUSDm PPO aggression after volatility regime shift",
      nextPlannedActions: [
        "Complete XAU canary trade quota",
        "Re-run BTC Dreamer alignment sweep",
        "Re-score symbol risk budget after session overlap",
      ],
    },
    controls: {
      runtimeStatus: "running",
      processes: [
        { name: "Server runtime", pid: 6480, status: "running" },
        { name: "UI shell", pid: 7212, status: "running" },
        { name: "Champion cycle", pid: 8024, status: "running" },
      ],
      availableActions: [
        "start_lstm",
        "start_dreamer",
        "start_drl",
        "run_cycle",
        "promote_canary",
        "rollback_champion",
        "restart_server",
      ],
      blockedReasons: [],
      notifications: "telegram+ui",
    },
    incidents: [
      { id: "A01", type: "system", severity: "info", timestamp: "Just now", message: "Successfully synchronized model weights across main nodes." },
      { id: "L01", type: "learning", severity: "warn", timestamp: "5m ago", message: "[RECURSIVE RE-EVALUATION] PPO registered sequence failure on XAUUSDm false breakout trade (T-090). Memory bias shifted -4.2% to explicitly suppress action on expanding M5 volume constraints." },
      { id: "L02", type: "learning", severity: "pass", timestamp: "25m ago", message: "[AUTO-PROMOTION] Canary Model `canary_8109` accumulated consecutive Sharpe > 2.0 over 100 ghost trades and mathematically dethroned the champion. Main live model hot-swapped seamlessly." },
      { id: "A03", type: "trading", severity: "warn", timestamp: "1h ago", message: "Risk limits progressively tightened by Risk Overlord due to sudden VIX surge." },
    ],
    timeline: [
      { id: "TL01", time: "10:45", category: "learning", text: "Loss detected on T-090. DreamerV3 context penalty immediately applied; future volatility hallucination weight increased +12%." },
      { id: "TL02", time: "09:30", category: "system", text: "Automated session start. MT5 bindings secured. Symbols active." },
      { id: "TL03", time: "08:15", category: "trading", text: "Profit on T-089 validated. Positive bias allocated to psychological level regression (65k support)." },
      { id: "TL04", time: "02:00", category: "learning", text: "Pattern signature 'expanding_triangle_fakeout' mathematically isolated as an inefficient edge. Assigned negative action bias." },
    ],
    // Initialize pattern recognition and perpetual improvements
    patternRecognition: {
      knownPatterns: [
        { id: "PAT-001", regime: "low_volatility_bull", phase: "policy", confidence: 0.88, discoveredAt: 12 },
        { id: "PAT-002", regime: "high_volatility_neutral", phase: "simulation", confidence: 0.74, discoveredAt: 45 }
      ],
      patternSuccessRates: {},
      marketRegimeHistory: [],
      learningRate: 0.01,
      adaptationSpeed: 0.005
    },
    indicatorBundles: [
      { id: "BNDL-VOL", name: "Volatility Accelerator", scenario: "high_volatility", components: ["Keltner Channels", "ATR", "Volume Profile"], active: true, winRate: 0.68 },
      { id: "BNDL-MEAN", name: "Mean Reversion Core", scenario: "low_volatility", components: ["RSI Divergence", "Bollinger Bands", "VWAP"], active: false, winRate: 0.72 }
    ],
    perpetualImprovements: {
      learningAdjustments: {
        lstm: { learningRate: 0.001, memoryRetention: 0.9 },
        ppo: { learningRate: 0.0003, entropyCoeff: 0.01 },
        dreamer: { learningRate: 0.0001, imaginationRatio: 0.5 }
      },
      adaptationHistory: [],
      improvementRate: 0.0005
    }
  };
}

export function advanceMockSystem(current) {
  const tick = (current.__tick || 0) + 1;
  const phase = ROUTE_PHASES[Math.floor((tick / 8) % ROUTE_PHASES.length)];
  const lstmLoss = clamp(current.training.lstm.loss - 0.012 + Math.sin(tick / 4) * 0.008, 0.18, 2.0);
  const ppoProgress = clamp(current.training.ppo.progress + 0.014, 0, 1);
  const dreamerProgress = clamp(current.training.dreamerV3.progress + 0.01, 0, 1);
  const canaryProgress = clamp((current.registry.canary?.progress || 0.45) + 0.025, 0, 1);
  const btcConfidence = clamp(0.63 + Math.sin(tick / 5) * 0.1, 0.18, 0.94);
  const xauConfidence = clamp(0.56 + Math.cos(tick / 6) * 0.08, 0.18, 0.9);
  const floatingPnl = Number((Math.sin(tick / 5) * 9.2 - 4.5).toFixed(2));
  const nextAction =
    phase === "context"
      ? "continue_lstm_training"
      : phase === "policy"
        ? "advance_ppo_episode"
        : phase === "simulation"
          ? "score_dreamer_rollout"
          : phase === "selection"
            ? "evaluate_registry_candidate"
            : "hold_production_authority";

  // Apply pattern recognition and perpetual improvements
  const patternRecognition = updatePatternRecognitionSystem(current, tick);
  const perpetualImprovements = applyPerpetualImprovements(current, tick);

  const next = {
    ...current,
    __tick: tick,
    patternRecognition,
    perpetualImprovements,
    connection: {
      ...current.connection,
      latencyMs: Math.round(clamp(42 + Math.sin(tick / 4) * 12, 18, 120)),
      lastSyncAt: Date.now(),
      stale: false,
    },
    orchestrator: {
      ...current.orchestrator,
      currentPhase: phase,
      cycleProgress: clamp(current.orchestrator.cycleProgress + 0.02, 0, 1),
      nextAction,
      queueDepth: Math.max(0, 3 + Math.round(Math.sin(tick / 7) * 2)),
      loopIteration: current.orchestrator.loopIteration + (tick % 10 === 0 ? 1 : 0),
      cooldownSec: Math.max(0, current.orchestrator.cooldownSec - 1),
    },
    training: {
      ...current.training,
      activePhase: phase,
      lstm: {
        ...current.training.lstm,
        currentSymbol: tick % 24 < 12 ? "BTCUSDm" : "XAUUSDm",
        epoch: Math.min(current.training.lstm.epochsTotal, current.training.lstm.epoch + (tick % 6 === 0 ? 1 : 0)),
        loss: Number(lstmLoss.toFixed(3)),
        valLoss: Number(clamp(current.training.lstm.valLoss - 0.01 + Math.cos(tick / 5) * 0.005, 0.2, 2.2).toFixed(3)),
        // Apply learning improvements to LSTM
        memoryStrength: Number(clamp(0.6 + Math.sin(tick / 6) * 0.12, 0.2, 0.94).toFixed(2)),
        featuresUsed: perpetualImprovements.learningAdjustments?.lstm?.memoryRetention 
          ? Math.min(150, Math.max(21, 150 * perpetualImprovements.learningAdjustments.lstm.memoryRetention)) 
          : 150,
      },
      ppo: {
        ...current.training.ppo,
        currentSymbol: tick % 20 < 10 ? "BTCUSDm" : "XAUUSDm",
        episode: current.training.ppo.episode + 1,
        progress: Number(ppoProgress.toFixed(2)),
        currentTimesteps: Math.min(current.training.ppo.targetTimesteps, current.training.ppo.currentTimesteps + 7000),
        policyLoss: Number(clamp(current.training.ppo.policyLoss - 0.01 + Math.sin(tick / 4) * 0.015, 0.08, 1.2).toFixed(3)),
        valueLoss: Number(clamp(current.training.ppo.valueLoss - 0.008 + Math.cos(tick / 5) * 0.012, 0.05, 1.1).toFixed(3)),
        entropy: Number(clamp(current.training.ppo.entropy - 0.006 + Math.sin(tick / 8) * 0.01, 0.08, 0.7).toFixed(3)),
        // Apply learning improvements to PPO
        dominantAction: btcConfidence > xauConfidence ? "short" : "long",
        etaSec: Math.max(0, current.training.ppo.etaSec - 6),
        // Adjust progress based on learning rate
        progress: Number(Math.min(ppoProgress * (1 + perpetualImprovements.learningAdjustments?.ppo?.learningRate || 1), 1).toFixed(2)),
      },
      dreamerV3: {
        ...current.training.dreamerV3,
        currentSymbol: tick % 18 < 9 ? "BTCUSDm" : "XAUUSDm",
        progress: Number(dreamerProgress.toFixed(2)),
        worldModelLoss: Number(clamp(current.training.dreamerV3.worldModelLoss - 0.012 + Math.sin(tick / 4) * 0.01, 0.12, 1.4).toFixed(3)),
        alignment: Number(clamp(current.training.dreamerV3.alignment + 0.015 + Math.sin(tick / 8) * 0.02, 0.1, 0.95).toFixed(2)),
        // Apply learning improvements to DreamerV3
        blendWeight: perpetualImprovements.learningAdjustments?.dreamer?.imaginationRatio 
          ? perpetualImprovements.learningAdjustments.dreamer.imaginationRatio 
          : 0.15,
      },
    },
    registry: {
      ...current.registry,
      gate: {
        ready: canaryProgress >= 1,
        reason: canaryProgress >= 1 ? "candidate cleared gate" : "min trades not yet reached",
      },
      canary: {
        ...current.registry.canary,
        progress: Number(canaryProgress.toFixed(2)),
      },
      candidates: current.registry.candidates.map((candidate) => {
        if (candidate.id !== current.registry.canary.id) return candidate;
        const nextSharpe = clamp(candidate.sharpe + 0.015 + Math.sin(tick / 6) * 0.01, 0.8, 2.1);
        return {
          ...candidate,
          progress: Number(canaryProgress.toFixed(2)),
          sharpe: Number(nextSharpe.toFixed(2)),
          verdict: canaryProgress >= 1 && nextSharpe > 1.55 ? "challenger" : canaryProgress >= 0.7 ? "canary" : "testing",
        };
      }),
    },
    trading: {
      ...current.trading,
      mode: floatingPnl > 0 ? "active" : "armed",
      account: {
        ...current.trading.account,
        floatingPnl,
        equity: Number((current.trading.account.balance + floatingPnl).toFixed(2)),
      },
      risk: {
        ...current.trading.risk,
        drawdownPct: Number(clamp(current.trading.risk.drawdownPct + Math.sin(tick / 9) * 0.18, 0.2, 6).toFixed(2)),
        dailyLossPct: Number(clamp(current.trading.risk.dailyLossPct + Math.cos(tick / 12) * 0.06, 0, current.trading.risk.maxDailyLossPct).toFixed(2)),
        sizeCap: Number(clamp(0.64 + Math.sin(tick / 8) * 0.08, 0.35, 1).toFixed(2)),
      },
      lanes: [
        {
          ...current.trading.lanes[0],
          confidence: Number(btcConfidence.toFixed(2)),
          exposure: Number(clamp(-0.32 - Math.sin(tick / 5) * 0.16, -0.82, 0.82).toFixed(2)),
          pnl: Number((-4 + Math.sin(tick / 4) * 7.4).toFixed(2)),
          status: btcConfidence > 0.64 ? "live" : "watching",
        },
        {
          ...current.trading.lanes[1],
          confidence: Number(xauConfidence.toFixed(2)),
          exposure: Number(clamp(0.2 + Math.cos(tick / 6) * 0.13, -0.82, 0.82).toFixed(2)),
          pnl: Number((-2 + Math.cos(tick / 5) * 6.1).toFixed(2)),
          status: xauConfidence > 0.6 ? "live" : "watching",
        },
      ],
    },
    selfImprove: {
      ...current.selfImprove,
      canaryPassRate: Number(clamp(current.selfImprove.canaryPassRate + Math.sin(tick / 13) * 0.01, 0.2, 0.96).toFixed(2)),
      feedbackCoverage: Number(clamp(current.selfImprove.feedbackCoverage + Math.cos(tick / 11) * 0.01, 0.3, 0.98).toFixed(2)),
      // Enhance self-improvement with learning insights
      lastImprovementAction: `Pattern-based adjustment: ${perpetualImprovements.adaptationHistory.length > 0 
        ? `Improved ${perpetualImprovements.adaptationHistory[perpetualImprovements.adaptationHistory.length - 1].phase} learning by ${(perpetualImprovements.adaptationHistory[perpetualImprovements.adaptationHistory.length - 1].learningAdjustment * 100).toFixed(1)}%`
        : "Initial learning cycle"}`,
    },
    incidents: [
          { level: "info", title: "Phase update", message: `Active authority phase is now ${phase}.` },
      { level: floatingPnl >= 0 ? "pass" : "warn", title: "Live watch PnL", message: `Floating PnL ${floatingPnl >= 0 ? "+" : ""}${floatingPnl.toFixed(2)}` },
      { level: canaryProgress >= 1 ? "pass" : "warn", title: "Canary review", message: `${current.registry.canary.id} progress ${(canaryProgress * 100).toFixed(0)}%` },
      // Add pattern recognition insights to incidents occasionally
      ...(tick % 25 === 0 && patternRecognition.knownPatterns.length > 0 
        ? [{ level: "info", title: "Pattern Discovery", message: `Discovered new pattern: ${patternRecognition.knownPatterns[patternRecognition.knownPatterns.length - 1].id}` }] 
        : []),
    ],
    timeline: [
      { level: "info", text: `LSTM loss improved to ${lstmLoss.toFixed(3)}` },
      { level: "info", text: `PPO timesteps ${Math.min(current.training.ppo.targetTimesteps, current.training.ppo.currentTimesteps + 7000).toLocaleString()}` },
      { level: "pass", text: `DreamerV3 alignment ${(clamp(current.training.dreamerV3.alignment + 0.015, 0.1, 0.95) * 100).toFixed(0)}%` },
      // Add learning insights to timeline
      ...(tick % 15 === 0 
        ? [{ level: "info", text: `Learning rate adjusted: LSTM=${(perpetualImprovements.learningAdjustments.lstm.learningRate || 0.001).toFixed(4)}, PPO=${(perpetualImprovements.learningAdjustments.ppo.learningRate || 0.0003).toFixed(4)}` }] 
        : []),
    ],
  };
  return next;
}
