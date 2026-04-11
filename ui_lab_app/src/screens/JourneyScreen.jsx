import React from "react";
import { Brain, GitBranch, Sparkles, TrendingUp, Workflow } from "lucide-react";
import { JOURNEY_STEPS, MetricTile, Panel, PipelineStageBoard, ProgressBar, shortDuration } from "../components/common";

function selectedLane(system, selectedSymbol) {
  return (system.trading.lanes || []).find((entry) => entry.symbol === selectedSymbol) || system.trading.lanes[0];
}

function selectedCandidate(system, selectedSymbol) {
  return (system.registry.candidates || []).find((entry) => entry.symbol === selectedSymbol && entry.verdict !== "rejected");
}

function stageData(system, selectedSymbol) {
  const lane = selectedLane(system, selectedSymbol);
  const candidate = selectedCandidate(system, selectedSymbol);
  return [
    {
      key: "context",
      title: "Context Engine",
      technical: "LSTM + Feature Engineering 150",
      purpose: "Builds memory from 150 engineered features so the system understands regime, momentum, volatility, structure, and pressure before it acts.",
      metricLabel: "Memory quality",
      metricValue: `${Math.round(system.training.lstm.memoryStrength * 100)}%`,
      status: `${system.training.featureVersion} · epoch ${system.training.lstm.epoch}/${system.training.lstm.epochsTotal}`,
      progress: system.training.lstm.memoryStrength,
      icon: Brain,
      tone: "pass",
    },
    {
      key: "simulation",
      title: "Scenario Engine",
      technical: "DreamerV3",
      purpose: "DreamerV3 tests imagined futures before the symbol gets real capital. It pressures the model in simulated market paths to expose weak assumptions early.",
      metricLabel: "Alignment score",
      metricValue: `${Math.round(system.training.dreamerV3.alignment * 100)}%`,
      status: `${system.meta.dreamerVersion} · ${system.training.dreamerV3.steps.toLocaleString()} steps`,
      progress: system.training.dreamerV3.alignment,
      icon: Sparkles,
      tone: "info",
    },
    {
      key: "policy",
      title: "Execution Policy",
      technical: "PPO",
      purpose: "PPO converts context and scenario pressure into action preference: long, short, hold, reduce, or stand down.",
      metricLabel: "Dominant action",
      metricValue: String(system.training.ppo.dominantAction || "hold").toUpperCase(),
      status: `${system.training.ppo.currentTimesteps.toLocaleString()} / ${system.training.ppo.targetTimesteps.toLocaleString()} · ETA ${shortDuration(system.training.ppo.etaSec)}`,
      progress: system.training.ppo.progress,
      icon: TrendingUp,
      tone: "info",
    },
    {
      key: "selection",
      title: "Live Review",
      technical: "Candidate review",
      purpose: "The symbol enters live review with limited authority. It must survive thresholds, risk pressure, and real trade evidence before promotion.",
      metricLabel: "Review status",
      metricValue: String(candidate?.verdict || "watching").toUpperCase(),
      status: candidate ? `Sharpe ${candidate.sharpe} · DD ${candidate.drawdown}` : system.registry.gate.reason,
      progress: system.registry.canary.progress || 0,
      icon: GitBranch,
      tone: system.registry.gate.ready ? "pass" : "warn",
    },
    {
      key: "authority",
      title: "Production Authority",
      technical: "Champion owner",
      purpose: "The symbol’s current production owner. Champions hold the lane until a better candidate proves it deserves control.",
      metricLabel: "Live owner",
      metricValue: lane?.champion || system.registry.champion.id,
      status: lane ? `Lane ${lane.status} · PnL ${lane.pnl >= 0 ? "+" : ""}${lane.pnl.toFixed(2)}` : "No live lane data",
      progress: Math.min(1, 0.55 + Math.max(0, lane?.confidence || 0) * 0.45),
      icon: Workflow,
      tone: "pass",
    },
  ];
}

function AuthorityStage({ stage, index, activeIndex }) {
  const Icon = stage.icon;
  const stageState = index < activeIndex ? "pass" : index === activeIndex ? stage.tone : "info";
  return (
    <div className={`authority-stage ${stageState}`} style={{ animationDelay: `${index * 120}ms` }}>
      <div className="authority-marker">
        <div className="authority-number">0{index + 1}</div>
        {index < JOURNEY_STEPS.length - 1 ? <div className="authority-line" /> : null}
      </div>
      <div className="authority-card">
        <div className="authority-card-top">
          <div>
            <div className="eyebrow-row">
              <Icon size={14} />
              <span>{stage.technical}</span>
            </div>
            <h3>{stage.title}</h3>
          </div>
          <div className={`candidate-chip ${stageState}`}>{stage.metricValue}</div>
        </div>
        <p className="authority-purpose">{stage.purpose}</p>
        <div className="authority-meta-row">
          <div className="authority-metric-label">{stage.metricLabel}</div>
          <div className="authority-status">{stage.status}</div>
        </div>
        <ProgressBar label={stage.metricLabel} value={stage.progress} tone={stageState} />
      </div>
    </div>
  );
}

export default function JourneyScreen({ system, selectedSymbol, activeIndex, onNext, onPrev }) {
  const lane = selectedLane(system, selectedSymbol);
  const candidate = selectedCandidate(system, selectedSymbol);
  const stages = stageData(system, selectedSymbol);

  return (
    <div className="stack">
      <Panel title="How a symbol earns live authority" subtitle={`${selectedSymbol} moves through a five-stage intelligence pipeline before it is trusted with real capital.`} icon={Workflow}>
        <div className="authority-hero">
          <div className="authority-hero-copy">
            <div className="authority-kicker">{selectedSymbol}</div>
            <h3>Every symbol trains for the right to trade.</h3>
            <p>
              Signal memory becomes scenario pressure. Scenario pressure becomes execution behavior. Execution behavior faces live review. Only then does a symbol earn production authority.
            </p>
          </div>
          <div className="authority-hero-stats">
            <MetricTile label="Current owner" value={lane?.champion || system.registry.champion.id} tone="pass" />
            <MetricTile label="Current confidence" value={`${Math.round((lane?.confidence || 0) * 100)}%`} tone="pass" />
            <MetricTile label="Review blocker" value={system.registry.gate.ready ? "cleared" : "active"} tone={system.registry.gate.ready ? "pass" : "warn"} />
            <MetricTile label="Feature contract" value={system.meta.featureVersion} />
          </div>
        </div>
        <PipelineStageBoard system={system} activeIndex={activeIndex} />
        <div className="button-row" style={{ justifyContent: "flex-end", marginTop: "24px" }}>
          <button className="button" onClick={onPrev} disabled={activeIndex === 0}>Previous Stage</button>
          <button className="button button-primary" onClick={onNext}>{activeIndex >= 4 ? "Enter Trading Watch" : "Advance Stage"}</button>
        </div>
      </Panel>

      <div className="screen-grid screen-grid-authority">
        <Panel title="Authority journey" subtitle={`${selectedSymbol} should read like a live lane under evaluation, not a pile of models.`} icon={GitBranch}>
          <div className="authority-timeline">
            {stages.map((stage, index) => (
              <AuthorityStage key={stage.key} stage={stage} index={index} activeIndex={activeIndex} />
            ))}
          </div>
        </Panel>

        <div className="stack">
          <Panel title="Selected symbol" subtitle="This is the one symbol the user should care about right now." icon={TrendingUp}>
            <div className="card-grid two-up">
              <MetricTile label="Lane" value={selectedSymbol} tone="pass" />
              <MetricTile label="Live status" value={(lane?.status || "watching").toUpperCase()} tone={lane?.status === "live" ? "pass" : "warn"} />
              <MetricTile label="Exposure" value={lane ? lane.exposure.toFixed(2) : "0.00"} />
              <MetricTile label="PnL" value={lane ? `${lane.pnl >= 0 ? "+" : ""}${lane.pnl.toFixed(2)}` : "+0.00"} tone={lane?.pnl >= 0 ? "pass" : "warn"} />
            </div>
            <div className="focus-card">
              <div className="metric-label">Why it matters</div>
              <div className="focus-value focus-copy">{lane?.reason || "This symbol is waiting for a stronger live rationale."}</div>
            </div>
          </Panel>

          <Panel title="Proof and gate" subtitle="A serious product shows evidence, blockers, and next authority move." icon={Sparkles}>
            <div className="card-grid two-up">
              <MetricTile label="Candidate" value={candidate?.id || "none"} tone="warn" />
              <MetricTile label="Gate" value={system.registry.gate.ready ? "READY" : "HOLD"} tone={system.registry.gate.ready ? "pass" : "warn"} />
              <MetricTile label="Sharpe" value={candidate ? String(candidate.sharpe) : "-"} />
              <MetricTile label="Drawdown" value={candidate ? String(candidate.drawdown) : "-"} />
            </div>
            <div className="rail-note">
              <strong>Promotion reason or blocker:</strong> {system.registry.gate.reason}
              <br />
              <strong>Last improvement event:</strong> {system.selfImprove.lastImprovementAction}
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
