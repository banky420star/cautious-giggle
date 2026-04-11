import React from "react";
import { Brain, CandlestickChart, Activity, Shield, TrendingUp, Zap, Target, Layout } from "lucide-react";
import { Button, LargeSparkline, MetricTile, Panel, ProgressBar, money, pct, shortDuration } from "../components/common";

export default function LandingScreen({ system, selectedSymbol, onStartJourney, onGoTrading, onGoControl }) {
  const lane = (system.trading.lanes || []).find((e) => e.symbol === selectedSymbol) || system.trading.lanes[0];
  const equityHistory = system._history?.equity || [];

  return (
    <div className="stack animate-in">
      
      {/* ─── ATMOSPHERIC HERO MONITOR ─── */}
      <section className="hero-monitor">
        <div className="hero-monitor-background">
           <LargeSparkline data={equityHistory} height={200} />
        </div>
        
        <div className="hero-monitor-content">
          <div className="hero-monitor-label">Total Account Equity</div>
          <div className="hero-monitor-value">{money(system.trading.account.equity)}</div>
          <div className="status-lamp-grid">
            <div className="status-lamp active-lamp"><div className="lamp-dot"/><span>Neural Core Live</span></div>
            <div className="status-lamp pass-lamp"><div className="lamp-dot"/><span>Data Stream Nominal</span></div>
            <div className="status-lamp"><div className="lamp-dot"/><span>Risk Shields Active</span></div>
          </div>
        </div>

        <div className="hero-monitor-side">
          <div className="secondary-monitor">
            <div className="secondary-monitor-label">Balance</div>
            <div className="secondary-monitor-value">{money(system.trading.account.balance)}</div>
          </div>
          <div className="secondary-monitor">
            <div className="secondary-monitor-label">Realized Today</div>
            <div className="secondary-monitor-value" style={{ color: "var(--green)" }}>{money(system.trading.account.realizedToday)}</div>
          </div>
          <div className="secondary-monitor">
            <div className="secondary-monitor-label">Floating PnL</div>
            <div className="secondary-monitor-value" style={{ color: system.trading.account.floatingPnl >= 0 ? "var(--green)" : "var(--red)" }}>
              {money(system.trading.account.floatingPnl)}
            </div>
          </div>
          <div className="secondary-monitor">
            <div className="secondary-monitor-label">Positions</div>
            <div className="secondary-monitor-value">{system.trading.account.openPositions}</div>
          </div>
        </div>
      </section>

      {/* ─── PRIMARY ACTIONS ─── */}
      <div className="card-grid three-up" style={{ marginTop: "-20px", position: "relative", zIndex: 10, padding: "0 40px" }}>
        <Button icon={TrendingUp} tone="primary" onClick={onGoTrading} style={{ height: "60px", fontSize: "1rem" }}>Open Trading Watch</Button>
        <Button icon={Target} onClick={onStartJourney} style={{ height: "60px", fontSize: "1rem" }}>Launch Guided Training</Button>
        <Button icon={Layout} onClick={onGoControl} style={{ height: "60px", fontSize: "1rem" }}>System Control Plane</Button>
      </div>

      <div className="screen-grid screen-grid-wide" style={{ marginTop: "24px" }}>
        {/* ─── TECHNICAL TELEMETRY ─── */}
        <div className="stack">
          <Panel title="Canary Pipeline Status" subtitle="AI model evolution and validation gate" icon={Brain}>
            <div className="authority-summary" style={{ gridTemplateColumns: "repeat(4, 1fr)" }}>
              <MetricTile label="Champion" value={system.registry.champion.id} tone="pass" />
              <MetricTile label="Canary" value={system.registry.canary.id} tone="warn" />
              <MetricTile label="Wait Pool" value={String(system.registry.candidates?.length || 0)} />
              <MetricTile label="Promotion" value={system.registry.gate.ready ? "READY" : "HOLD"} tone={system.registry.gate.ready ? "pass" : "warn"} />
            </div>
          </Panel>

          <Panel title="Training Engine Load" subtitle="Real-time resource allocation and neural loss" icon={Activity}>
             <div className="card-grid three-up compact-grid">
               <div className="metric-tile">
                  <div className="metric-label">LSTM Context</div>
                  <ProgressBar label="Memory" value={system.training.lstm.memoryStrength} tone="pass" meta={pct(system.training.lstm.memoryStrength)} />
               </div>
               <div className="metric-tile">
                  <div className="metric-label">PPO Execution</div>
                  <ProgressBar label="Progress" value={system.training.ppo.progress} tone="info" meta={pct(system.training.ppo.progress)} />
               </div>
               <div className="metric-tile">
                  <div className="metric-label">DreamerV3</div>
                  <ProgressBar label="Alignment" value={system.training.dreamerV3.alignment} tone="info" meta={pct(system.training.dreamerV3.alignment)} />
               </div>
             </div>
          </Panel>
        </div>

        {/* ─── RIGHT SIDE: RISKS & LOGS ─── */}
        <div className="stack">
          <Panel title="Safety & Guardrails" subtitle="Active risk mitigation status" icon={Shield}>
             <div style={{ display: "flex", justifyContent: "space-between", gap: "10px" }}>
                <MetricTile label="Drawdown" value={`${system.trading.risk.drawdownPct.toFixed(1)}%`} tone={system.trading.risk.drawdownPct > 5 ? "warn" : "pass"} />
                <MetricTile label="Daily Cap" value={`${system.trading.risk.maxDailyLossPct}%`} />
                <MetricTile label="Kill Switch" value={system.trading.risk.killSwitchArmed ? "ARMED" : "Safe"} tone={system.trading.risk.killSwitchArmed ? "warn" : "pass"} />
             </div>
          </Panel>

          <Panel title="Live Intelligence Feed" subtitle="Telemetry stream" icon={Zap}>
             <div className="narrative-list" style={{ maxHeight: "180px", overflowY: "auto" }}>
                {(system.timeline || []).slice(-5).map((t, i) => (
                  <div key={i} className="narrative-item" style={{ border: 0, padding: "8px 0", background: "transparent" }}>
                    <div style={{ width: 6, height: 6, borderRadius: "50%", background: t.level === "warn" ? "var(--amber)" : "var(--cyan)" }} />
                    <span style={{ fontSize: "0.8rem", color: "var(--muted)" }}>{t.text}</span>
                  </div>
                ))}
             </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
