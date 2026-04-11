import React, { useState } from "react";
import { Play, RefreshCcw, Shield, Wrench } from "lucide-react";
import { Button, MetricTile, Panel } from "../components/common";

function ActionButton({ action, busyAction, onAction }) {
  return (
    <button className={`action-button ${busyAction === action ? "busy" : ""}`} type="button" onClick={() => onAction(action)}>
      <span>{action}</span>
      <span>{busyAction === action ? "running" : "ready"}</span>
    </button>
  );
}

export default function ControlPlaneScreen({ system, busyAction, onAction, onBackToTrading }) {
  const [symbol, setSymbol] = useState("BTCUSDm");

  return (
    <div className="stack">
      <div className="screen-actions">
        <Button icon={Play} tone="primary" onClick={onBackToTrading}>
          Back to trading watch
        </Button>
      </div>

      <div className="card-grid two-up">
        <Panel title="Runtime processes" subtitle="This must stay separate from the narrative screens." icon={Wrench}>
          <div className="process-list">
            {system.controls.processes.map((process) => (
              <div className="process-row" key={`${process.name}-${process.pid}`}>
                <div>
                  <div className="process-title">{process.name}</div>
                  <div className="process-meta">PID {process.pid}</div>
                </div>
                <div className={`process-chip ${process.status}`}>{process.status}</div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Control actions" subtitle="Operator-only commands with explicit boundaries." icon={RefreshCcw}>
          <div className="control-input-row">
            <label htmlFor="symbol">Target symbol</label>
            <input id="symbol" value={symbol} onChange={(event) => setSymbol(event.target.value)} />
          </div>
          <div className="action-grid">
            {system.controls.availableActions.map((action) => (
              <ActionButton key={action} action={action} busyAction={busyAction} onAction={(name) => onAction(name, { symbol })} />
            ))}
          </div>
        </Panel>
      </div>

      <div className="card-grid two-up">
        <Panel title="Registry authority" subtitle="Promotion evidence must be visible before action." icon={Shield}>
          <div className="card-grid two-up">
            <MetricTile label="Champion" value={system.registry.champion.id} tone="pass" />
            <MetricTile label="Canary" value={system.registry.canary.id} tone="warn" />
            <MetricTile label="Gate ready" value={system.registry.gate.ready ? "yes" : "no"} tone={system.registry.gate.ready ? "pass" : "warn"} />
            <MetricTile label="Reason" value={system.registry.gate.reason} />
          </div>
          <div className="lineage-list">
            {system.registry.lineage.map((entry) => (
              <div className="lineage-row" key={`${entry.id}-${entry.when}`}>
                <div>
                  <div className="lineage-title">{entry.id}</div>
                  <div className="lineage-meta">from {entry.from}</div>
                </div>
                <div className="lineage-side">
                  <div>{entry.when}</div>
                  <div className="lineage-reason">{entry.reason}</div>
                </div>
              </div>
            ))}
          </div>
        </Panel>

        <Panel title="Guardrails" subtitle="What should block unsafe control actions." icon={Shield}>
          <div className="card-grid two-up">
            <MetricTile label="Runtime status" value={system.controls.runtimeStatus} tone={system.controls.runtimeStatus === "running" ? "pass" : "warn"} />
            <MetricTile label="Notifications" value={system.controls.notifications} />
            <MetricTile label="Feature version" value={system.meta.featureVersion} />
            <MetricTile label="Dreamer stack" value={system.meta.dreamerVersion} />
          </div>
          <div className="guardrail-text">
            Keep this screen isolated from the guided journey. It is the operator surface for process control, promotion actions, and guarded mutation paths.
          </div>
        </Panel>
      </div>
    </div>
  );
}
