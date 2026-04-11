import { advanceMockSystem, createInitialSystem } from "./mockSystem";

const HISTORY_MAX = 120;
const _historyBuffer = { equity: [], pnl: [], confidence: [] };

function deepMerge(base, patch) {
  if (!base || typeof base !== "object" || Array.isArray(base)) return patch;
  if (!patch || typeof patch !== "object" || Array.isArray(patch)) return patch;
  const next = { ...base };
  Object.keys(patch).forEach((key) => {
    const baseValue = base[key];
    const patchValue = patch[key];
    if (
      baseValue &&
      typeof baseValue === "object" &&
      !Array.isArray(baseValue) &&
      patchValue &&
      typeof patchValue === "object" &&
      !Array.isArray(patchValue)
    ) {
      next[key] = deepMerge(baseValue, patchValue);
    } else {
      next[key] = patchValue;
    }
  });
  return next;
}

function normalizeLevel(raw) {
  const value = String(raw || "info").toLowerCase();
  if (value === "warning" || value === "warn") return "warn";
  if (value === "critical" || value === "error" || value === "fail") return "fail";
  if (value === "activity" || value === "success" || value === "pass") return "pass";
  return "info";
}

function mapProcesses(status) {
  const rows = [];
  if (status?.server?.running) {
    rows.push({ name: "Server runtime", pid: (status.server.pids || [])[0] || "-", status: "running" });
  }
  if (status?.training?.lstm_running) {
    rows.push({ name: "LSTM trainer", pid: (status.training.lstm_pids || [])[0] || "-", status: "running" });
  }
  if (status?.training?.dreamer_running) {
    rows.push({ name: "DreamerV3 trainer", pid: (status.training.dreamer_pids || [])[0] || "-", status: "running" });
  }
  if (status?.training?.drl_running) {
    rows.push({ name: "PPO trainer", pid: (status.training.drl_pids || [])[0] || "-", status: "running" });
  }
  if (status?.training?.cycle_running) {
    rows.push({ name: "Champion cycle", pid: (status.training.cycle_pids || [])[0] || "-", status: "running" });
  }
  return rows;
}

function mapLanes(status, baseFeatureVersion) {
  // Backend sends symbol_lane_rows as an array inside training
  const laneRows = status?.training?.symbol_lane_rows || [];
  if (laneRows.length) {
    return laneRows.map((row) => ({
      symbol: row.symbol,
      side: String(row.side || row.position_side || "flat").toLowerCase(),
      champion: row.champion || "unknown",
      reason: row.reason || row.decision?.regime || "runtime lane",
      confidence: Number(row.confidence || row.decision?.confidence || 0),
      exposure: Number(row.exposure || 0),
      pnl: Number(row.pnl || 0),
      status: row.status || "watching",
      canTrade: row.canTrade !== false,
      featureVersion: baseFeatureVersion,
    }));
  }

  // Fallback: symbol_cards object (legacy format)
  const cards = status?.symbol_cards || {};
  const cardRows = Object.entries(cards).map(([symbol, card]) => ({
    symbol,
    side: String(card.position_side || card.signal || "flat").toLowerCase(),
    champion: card.champion || card.model || "unknown",
    reason: card.reason || card.signal || "runtime lane summary",
    confidence: Number(card.confidence || 0),
    exposure: Number(card.exposure || card.blend_exposure || 0),
    pnl: Number(card.position_profit || card.pnl || 0),
    status: card.status || (card.can_trade === false ? "blocked" : "watching"),
    canTrade: card.can_trade !== false,
    featureVersion: card.feature_version || baseFeatureVersion,
  }));
  if (cardRows.length) return cardRows;

  // Fallback: derive from positions
  return (status?.account?.positions || []).map((position) => ({
    symbol: position.symbol,
    side: String(position.type || "flat").toLowerCase(),
    champion: status?.active_models?.champion || "unknown",
    reason: "derived from live position snapshot",
    confidence: 0,
    exposure: 0,
    pnl: Number(position.profit || 0),
    status: "live",
    canTrade: true,
    featureVersion: baseFeatureVersion,
  }));
}

function deriveFeatureVersion(status) {
  const history = status?.registry?.champion_history || [];
  const newest = history[0]?.metadata || {};
  return (
    newest.feature_set_version ||
    newest.feature_version ||
    status?.training?.visual?.ppo?.feature_version ||
    status?.training?.visual?.dreamer?.feature_version ||
    "ultimate_150"
  );
}

export function mapStatusToProductShell(status, current = createInitialSystem()) {
  const featureVersion = deriveFeatureVersion(status);
  const activeModels = status?.active_models || {};
  const training = status?.training || {};
  const visual = training.visual || {};
  const dreamer = visual.dreamer || {};
  const ppo = visual.ppo || {};
  const lstm = visual.lstm || {};
  const incidents = (status?.incidents || []).slice(0, 10).map((incident) => ({
    level: normalizeLevel(incident.severity),
    title: incident.symbol || incident.type || incident.event || "incident",
    message: incident.message || incident.summary || incident.reason || JSON.stringify(incident.payload || {}),
  }));
  const timeline = [
    ...(status?.logs?.audit || []).slice(-3).map((line) => ({ level: "info", text: line })),
    ...(status?.logs?.ppo || []).slice(-2).map((line) => ({ level: "info", text: line })),
  ].slice(0, 8);

  const equity = Number(status?.account?.equity || 0);
  const floatingPnl = Number(status?.account?.profit || 0);
  const lanes = mapLanes(status, featureVersion);
  const topLaneConfidence = lanes.length ? Math.max(...lanes.map((l) => Number(l.confidence || 0))) : 0;

  _historyBuffer.equity.push(equity);
  _historyBuffer.pnl.push(floatingPnl);
  _historyBuffer.confidence.push(topLaneConfidence);
  if (_historyBuffer.equity.length > HISTORY_MAX) _historyBuffer.equity.shift();
  if (_historyBuffer.pnl.length > HISTORY_MAX) _historyBuffer.pnl.shift();
  if (_historyBuffer.confidence.length > HISTORY_MAX) _historyBuffer.confidence.shift();

  const merged = deepMerge(current, {
    meta: {
      appName: "Cautious Giggle",
      featureVersion,
      dreamerVersion: "DreamerV3",
      transportMode: "live",
    },
    connection: {
      status: "connected",
      transport: "backend",
      latencyMs: current.connection.latencyMs,
      lastSyncAt: Date.now(),
      stale: false,
    },
    orchestrator: {
      loopStatus: status?.state || "live",
      owner: training?.cycle_running ? "champion_cycle" : "runtime",
      currentPhase: visual.active_key || visual.active_label || current.orchestrator.currentPhase,
      cycleProgress: Number(((ppo.progress_pct || 0) / 100).toFixed(2)),
      nextAction: training?.cycle_running ? "run_cycle" : "runtime_watch",
      queueDepth: Number(training?.pipeline_summary?.training_active_symbols || 0),
    },
    training: {
      featureVersion,
      activePhase: visual.active_key || visual.active_label || current.training.activePhase,
      lstm: {
        currentSymbol: training?.lstm_symbol || lstm.current_symbol || current.training.lstm.currentSymbol,
        epoch: Number(training?.lstm_epoch || 0),
        epochsTotal: Number(training?.lstm_epochs_total || 0),
        loss: Number(lstm.loss || current.training.lstm.loss),
        valLoss: Number(lstm.val_loss || current.training.lstm.valLoss),
        memoryStrength: Number(lstm.memory_strength || current.training.lstm.memoryStrength),
        featuresUsed: featureVersion === "ultimate_150" ? 150 : 21,
        queue: (lstm.queue || []).map((row) => row.symbol),
      },
      ppo: {
        currentSymbol: ppo.current_symbol || training?.drl_symbol || current.training.ppo.currentSymbol,
        progress: Number(((ppo.progress_pct || 0) / 100).toFixed(2)),
        currentTimesteps: Number(ppo.current_timesteps || 0),
        targetTimesteps: Number(training?.drl_timesteps || ppo.target_timesteps || current.training.ppo.targetTimesteps),
        policyLoss: Number(ppo.policy_loss || current.training.ppo.policyLoss),
        valueLoss: Number(ppo.value_loss || current.training.ppo.valueLoss),
        entropy: Number(ppo.entropy || current.training.ppo.entropy),
        dominantAction: ppo.dominant_action || current.training.ppo.dominantAction,
        etaSec: Number(ppo.eta_seconds || 0),
      },
      dreamerV3: {
        enabled: training?.dreamer_running || dreamer.last_saved_symbol || current.training.dreamerV3.enabled,
        currentSymbol: dreamer.current_symbol || current.training.dreamerV3.currentSymbol,
        progress: Number(((dreamer.progress_pct || 0) / 100).toFixed(2)),
        steps: Number(dreamer.steps || current.training.dreamerV3.steps),
        window: Number(dreamer.window || current.training.dreamerV3.window),
        worldModelLoss: Number(dreamer.world_model_loss || current.training.dreamerV3.worldModelLoss),
        alignment: Number(dreamer.alignment || current.training.dreamerV3.alignment),
        blendWeight: Number(status?.profiles?.default?.dreamer_weight || current.training.dreamerV3.blendWeight),
      },
      pipeline: {
        trainingActiveSymbols: Number(training?.pipeline_summary?.training_active_symbols || 0),
        canaryReviewSymbols: Number(training?.pipeline_summary?.canary_review_symbols || 0),
        tradingReadySymbols: Number(training?.pipeline_summary?.trading_ready_symbols || 0),
        tradingActiveSymbols: Number(training?.pipeline_summary?.trading_active_symbols || 0),
      },
    },
    registry: {
      champion: {
        id: activeModels.champion || current.registry.champion.id,
        symbol: current.registry.champion.symbol,
        featureVersion,
        verdict: "champion",
      },
      canary: {
        id: activeModels.canary || current.registry.canary.id,
        symbol: current.registry.canary.symbol,
        featureVersion,
        verdict: activeModels.canary ? "canary" : "none",
        progress: Number(((status?.canary_gate?.ready ? 1 : 0.4) || 0).toFixed(2)),
      },
      gate: {
        ready: Boolean(status?.canary_gate?.ready),
        reason: status?.canary_gate?.reason || current.registry.gate.reason,
      },
      candidates: Object.entries(activeModels?.symbols || {}).map(([sym, info]) => ({
        id: info.champion || info.canary || sym,
        symbol: sym,
        verdict: info.canary ? "canary" : info.champion ? "champion" : "testing",
        sharpe: 0,
        drawdown: 0,
        featureVersion: featureVersion,
      })),
      lineage: (status?.registry?.champion_history || activeModels?.champion_history || []).slice(0, 5).map((row) => ({
        id: row.path || "unknown",
        from: row.previous || "unknown",
        when: row.recorded_at || "unknown",
        reason: row.metadata?.feature_set_version || row.metadata?.feature_version || "metadata recorded",
      })),
    },
    trading: {
      mode: (status?.mode || "DRY-RUN").toLowerCase() === "live" ? "active" : "armed",
      account: {
        connected: Boolean(status?.account?.connected ?? (status?.account?.balance > 0)),
        balance: Number(status?.account?.balance || 0),
        equity,
        freeMargin: Number(status?.account?.free_margin || 0),
        floatingPnl,
        realizedToday: Number(status?.account?.realized_today || 0),
        openPositions: Number(status?.account?.open_positions || 0),
      },
      lanes,
      tradeHistory: (status?._trades || []).slice(0, 20).map((t, i) => ({
        id: String(t.ticket || `T-${i}`),
        symbol: t.symbol || "UNKNOWN",
        type: t.action_type || t.side?.toLowerCase() || "trade",
        pnl: Number(t.profit || 0),
        duration: t.hold_minutes ? `${Math.floor(t.hold_minutes / 60)}h ${t.hold_minutes % 60}m` : "unknown",
        reason: t.comment || "executed by champion model",
        timestamp: t.close_time || t.open_time || "recent",
        model: t.model || "champion",
      })),
      risk: {
        canTrade: status?.risk?.can_trade !== false && !Boolean(status?.risk?.halt),
        drawdownPct: Number(status?.risk?.current_dd || status?.account?.drawdown_pct || current.trading.risk.drawdownPct),
        dailyLossPct: Number(status?.risk?.daily_loss_pct || current.trading.risk.dailyLossPct),
        maxDailyLossPct: Number(status?.risk?.max_daily_loss || status?.risk?.max_daily_loss_pct || current.trading.risk.maxDailyLossPct),
        sizeCap: Number(status?.risk?.size_cap || current.trading.risk.sizeCap),
        killSwitchArmed: true,
      },
    },
    selfImprove: {
      loopHealth: status?.state === "online" ? "stable" : "degraded",
      lastImprovementAction: `Runtime: ${status?.mode || "DRY-RUN"} mode, ${Number(status?.risk?.daily_trades || 0)} trades today`,
    },
    controls: {
      runtimeStatus: status?.server?.running ? "running" : "stopped",
      processes: mapProcesses(status),
      availableActions: current.controls.availableActions,
      notifications: "telegram+ui",
    },
    incidents,
    timeline,
  });

  merged._history = {
    equity: [..._historyBuffer.equity],
    pnl: [..._historyBuffer.pnl],
    confidence: [..._historyBuffer.confidence],
  };
  merged.__tick = (current.__tick || 0) + 1;

  return merged;
}

export function createSystemAdapter({
  transport = "mock",
  statusUrl = "/api/status",
  controlUrl = "/api/control",
  wsUrl = null,
  pollMs = 2000,
} = {}) {
  return {
    subscribe(onPatch, onError) {
      if (transport === "mock") {
        let state = createInitialSystem();
        onPatch(state);
        const intervalId = window.setInterval(() => {
          state = advanceMockSystem(state);
          onPatch(state);
        }, 1000);
        return () => window.clearInterval(intervalId);
      }

      if (transport === "poll") {
        let cancelled = false;
        async function tick() {
          try {
            const [statusRes, tradesRes] = await Promise.all([
              fetch(statusUrl, { cache: "no-store" }),
              fetch("/api/trades?limit=20", { cache: "no-store" }).catch(() => null),
            ]);
            if (!statusRes.ok) throw new Error(`status fetch failed: ${statusRes.status}`);
            const statusJson = await statusRes.json();
            if (tradesRes && tradesRes.ok) {
              statusJson._trades = await tradesRes.json().catch(() => []);
            }
            if (!cancelled) onPatch(statusJson);
          } catch (error) {
            if (!cancelled && onError) onError(error instanceof Error ? error : new Error("poll failed"));
          }
        }
        tick();
        const intervalId = window.setInterval(tick, pollMs);
        return () => {
          cancelled = true;
          window.clearInterval(intervalId);
        };
      }

      if (transport === "ws") {
        const target = wsUrl || `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}/ws`;
        const socket = new WebSocket(target);
        socket.onmessage = (event) => {
          try {
            onPatch(JSON.parse(event.data));
          } catch (error) {
            if (onError) onError(error instanceof Error ? error : new Error("invalid websocket payload"));
          }
        };
        socket.onerror = () => {
          if (onError) onError(new Error("websocket failed"));
        };
        return () => socket.close();
      }

      if (onError) onError(new Error(`unknown transport: ${transport}`));
      return undefined;
    },

    async dispatch(action, payload = {}) {
      if (transport === "mock") {
        return {
          ok: true,
          action,
          payload,
          message: `mock action executed: ${action}`,
        };
      }

      const response = await fetch(controlUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...payload }),
      });

      if (!response.ok) {
        return {
          ok: false,
          action,
          payload,
          message: `control request failed: ${response.status}`,
        };
      }

      const result = await response.json().catch(() => ({}));
      return {
        ok: result?.ok !== false,
        action,
        payload,
        message: result?.message || "control request complete",
      };
    },
  };
}
