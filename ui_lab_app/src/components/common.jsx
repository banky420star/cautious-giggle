import React from "react";
import {
  Activity, AlertTriangle, ArrowDown, ArrowRight, ArrowUp, CheckCircle2,
  Clock3, GitBranch, Minus, Radio, Shield, Sparkles, Workflow,
} from "lucide-react";

// ─── ROUTES & JOURNEY STEPS ───
export const ROUTES = {
  loading: "loading",
  landing: "landing",
  journey: "journey",
  trading: "trading",
  control: "control",
  about: "about",
  raw: "raw",
  strategies: "strategies"
};

export const JOURNEY_STEPS = [
  { key: "context", label: "Context", title: "Context Engine", subtitle: "150 engineered features build market memory before any symbol earns authority." },
  { key: "simulation", label: "Scenario", title: "Scenario Engine", subtitle: "DreamerV3 pressures the symbol through imagined futures before capital is trusted." },
  { key: "policy", label: "Execution", title: "Execution Policy", subtitle: "PPO converts memory and simulation pressure into actionable trading behavior." },
  { key: "selection", label: "Review", title: "Live Review", subtitle: "Candidates face reduced-risk live validation before they are trusted with a lane." },
  { key: "authority", label: "Authority", title: "Production Authority", subtitle: "Only proven models become live lane owners and keep control until displaced." },
];

// ─── FORMATTERS ───
export function pct(value, digits = 0) { return `${(Number(value || 0) * 100).toFixed(digits)}%`; }
export function money(value) { const n = Number(value || 0); return `${n >= 0 ? "+" : ""}${n.toFixed(2)}`; }
export function shortDuration(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds || 0)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function statusTone(level) {
  const v = String(level || "info").toLowerCase();
  if (v === "pass" || v === "good" || v === "ready" || v === "running") return "tone-pass";
  if (v === "warn" || v === "warning" || v === "holding") return "tone-warn";
  if (v === "fail" || v === "error" || v === "critical" || v === "stopped") return "tone-fail";
  return "tone-info";
}

// ─── SPARKLINE ───
export function LargeSparkline({ data = [], height = 160 }) {
  if (!data || !data.length) return <div className="empty-state">Not enough equity data to graph.</div>;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = (max - min) || 1;
  const width = 800; // internal viewbox coordinate width
  const points = data.map((v, i) => {
    const x = (i / Math.max(1, data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 30) - 15;
    return `${x},${y}`;
  });
  const areaPoints = `0,${height} ${points.join(" ")} ${width},${height}`;
  const last = data[data.length - 1];
  const first = data[0];
  const color = last >= first ? "var(--green)" : "var(--red)";
  return (
    <div style={{ position: "relative", width: "100%", height, marginTop: 12 }}>
      <svg width="100%" height="100%" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" style={{ overflow: "visible" }}>
        <defs>
          <linearGradient id="largeGlow" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0.0" />
          </linearGradient>
        </defs>
        <polygon points={areaPoints} fill="url(#largeGlow)" />
        <polyline points={points.join(" ")} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={width} cy={height - ((last - min) / range) * (height - 30) - 15} r="4" fill={color} className="pulse" />
      </svg>
    </div>
  );
}


export function Sparkline({ data = [], width = 64, height = 24, positive = true, color }) {
  if (!data.length) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data.map((v, i) => {
    const x = (i / Math.max(1, data.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const last = data[data.length - 1];
  const lastX = width;
  const lastY = height - ((last - min) / range) * (height - 4) - 2;
  const areaPoints = `0,${height} ${points.join(" ")} ${width},${height}`;
  const toneClass = color || (last >= data[0] ? "sparkline-positive" : "sparkline-negative");
  return (
    <div className="sparkline-container">
      <svg className={`sparkline-svg ${toneClass}`} width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
        <polygon className="sparkline-area" points={areaPoints} />
        <polyline className="sparkline-line" points={points.join(" ")} />
        <circle className="sparkline-dot" cx={lastX} cy={lastY} />
      </svg>
    </div>
  );
}

// ─── GAUGE ───
export function Gauge({ value = 0, max = 100, size = 80, strokeWidth = 6, label, color }) {
  const normalized = Math.min(1, Math.max(0, value / max));
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference * (1 - normalized);
  const displayValue = typeof value === "number" ? (value % 1 === 0 ? value : value.toFixed(1)) : value;
  const strokeColor = color || (normalized > 0.7 ? "var(--green)" : normalized > 0.4 ? "var(--amber)" : "var(--red)");
  return (
    <div className="gauge-container">
      <svg className="gauge-svg" width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle className="gauge-bg" cx={size / 2} cy={size / 2} r={radius} strokeWidth={strokeWidth} />
        <circle className="gauge-fill" cx={size / 2} cy={size / 2} r={radius} strokeWidth={strokeWidth}
          stroke={strokeColor} strokeDasharray={circumference} strokeDashoffset={offset} />
      </svg>
      <div className="gauge-center">
        <div className="gauge-value">{displayValue}</div>
        {label && <div className="gauge-label">{label}</div>}
      </div>
    </div>
  );
}

// ─── LOG FEED ───
export function LogFeed({ items = [], maxItems = 12 }) {
  const rows = items.slice(-maxItems);
  if (!rows.length) return <div className="empty-state">No log entries</div>;
  return (
    <div className="log-feed">
      {rows.map((item, i) => (
        <div className="log-line" key={i}>
          {item.time && <span className="log-time">{item.time}</span>}
          <span className={`log-level ${item.level || "info"}`}>{item.level || "info"}</span>
          <span className="log-msg">{item.text || item.message}</span>
        </div>
      ))}
    </div>
  );
}

// ─── KPI CARD ───
export function KpiCard({ label, value, sub, tone }) {
  return (
    <div className={`kpi-card ${statusTone(tone)}`}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

// ─── STAT ROW ───
export function StatRow({ icon: Icon, label, value, sub, tone }) {
  return (
    <div className={`stat-row ${statusTone(tone)}`}>
      <div className="stat-row-label">{Icon && <Icon size={13} />}<span>{label}</span></div>
      <div><span className="stat-row-value">{value}</span>{sub && <span className="stat-row-sub">{sub}</span>}</div>
    </div>
  );
}

// ─── TREND INDICATOR ───
function TrendArrow({ current, previous }) {
  if (previous == null || current == null) return null;
  const diff = current - previous;
  const pctDiff = previous !== 0 ? ((diff / Math.abs(previous)) * 100).toFixed(1) : "0.0";
  if (Math.abs(diff) < 0.001) return <div className="metric-trend neutral"><Minus size={10} />{pctDiff}%</div>;
  if (diff > 0) return <div className="metric-trend up"><ArrowUp size={10} />+{pctDiff}%</div>;
  return <div className="metric-trend down"><ArrowDown size={10} />{pctDiff}%</div>;
}

// ─── METRIC TILE ───
export function MetricTile({ label, value, meta, tone = "info", sparkData, trend }) {
  const hasSpark = sparkData && sparkData.length > 0;
  const hasTrend = trend != null;
  return (
    <div className={`metric-tile ${hasSpark ? "metric-tile-with-trend" : ""} ${statusTone(tone)}`}>
      <div className="metric-content">
        <div className="metric-label">{label}</div>
        <div className="metric-value">{value}</div>
        {hasTrend && <TrendArrow current={sparkData?.[sparkData.length - 1]} previous={sparkData?.[sparkData.length - 2]} />}
        {meta && <div className="metric-meta">{meta}</div>}
      </div>
      {hasSpark && <Sparkline data={sparkData} width={56} height={22} />}
    </div>
  );
}

// ─── BUTTON ───
export function Button({ icon: Icon, children, tone = "default", ...props }) {
  return (
    <button className={`button button-${tone}`} type="button" {...props}>
      {Icon ? <Icon size={14} /> : null}
      <span>{children}</span>
    </button>
  );
}

// ─── PANEL ───
export function Panel({ title, subtitle, children, icon: Icon, right }) {
  return (
    <section className="panel">
      <div className="panel-head">
        <div>
          <div className="eyebrow-row">{Icon ? <Icon size={13} /> : <Sparkles size={13} />}<span>{title}</span></div>
          {subtitle && <h2 className="panel-title">{subtitle}</h2>}
        </div>
        {right || null}
      </div>
      {children}
    </section>
  );
}

// ─── PROGRESS BAR ───
export function ProgressBar({ label, value, tone = "info", meta }) {
  const normalized = Math.max(0, Math.min(1, Number(value || 0)));
  return (
    <div className="progress-block">
      <div className="progress-row"><span>{label}</span><span>{meta || pct(normalized)}</span></div>
      <div className="progress-rail"><div className={`progress-fill ${statusTone(tone)}`} style={{ width: `${normalized * 100}%` }} /></div>
    </div>
  );
}

// ─── EVENT LIST ───
export function EventList({ items, empty = "Nothing to show." }) {
  if (!items.length) return <div className="empty-state">{empty}</div>;
  return (
    <div className="event-list">
      {items.map((item, index) => (
        <div className={`event-card ${statusTone(item.level)}`} key={`${item.title || item.text}-${index}`}>
          <div className="event-top">
            <span className="event-title">{item.title || item.level || "event"}</span>
            <span className="event-chip">{String(item.level || "info").toUpperCase()}</span>
          </div>
          <div className="event-body">{item.message || item.text}</div>
        </div>
      ))}
    </div>
  );
}

// ─── PIPELINE STAGE BOARD ───
export function PipelineStageBoard({ system, activeIndex }) {
  return (
    <div className="pipeline-board">
      {JOURNEY_STEPS.map((step, index) => {
        const state = index < activeIndex ? "pass" : index === activeIndex ? "info" : "default";
        return (
          <div className={`stage-card ${statusTone(state)}`} key={step.key}>
            <div className="stage-top">
              <span className="stage-label">{step.label}</span>
              <span className="stage-index">0{index + 1}</span>
            </div>
            <h3>{step.title}</h3>
            <p>{step.subtitle}</p>
            {step.key === "context" && <ProgressBar label="Memory" value={system.training.lstm.memoryStrength} tone="pass" meta={`${system.training.lstm.featuresUsed} feat`} />}
            {step.key === "simulation" && <ProgressBar label="Alignment" value={system.training.dreamerV3.alignment} tone="info" meta={`${system.training.dreamerV3.steps.toLocaleString()} steps`} />}
            {step.key === "policy" && <ProgressBar label="Progress" value={system.training.ppo.progress} tone="info" meta={`${system.training.ppo.currentTimesteps.toLocaleString()} ts`} />}
            {step.key === "selection" && <ProgressBar label="Review" value={system.registry.canary.progress || 0} tone={system.registry.gate.ready ? "pass" : "warn"} meta={system.registry.gate.reason} />}
            {step.key === "authority" && <ProgressBar label="Authority" value={Math.min(1, 0.68 + Math.max(0, system.trading.account.realizedToday) / 200)} tone="pass" meta={system.registry.champion.id} />}
          </div>
        );
      })}
    </div>
  );
}

// ─── HEADER SHELL ───
export function HeaderShell({ system, view, journeyTitle, transport, transportState, lastActionResult, selectedSymbol }) {
  const currentTitle = view === ROUTES.journey ? journeyTitle : view === ROUTES.trading ? "Trading Watch" : view === ROUTES.control ? "Control Plane" : "Overview";
  const uptime = system.__tick ? `${Math.floor(system.__tick / 60)}m ${system.__tick % 60}s` : "0s";
  return (
    <header className="hero-shell">
      <div className="hero-topline">
        <div>
          <div className="eyebrow">CAUTIOUS GIGGLE · {selectedSymbol} · {system.meta.featureVersion}</div>
          <h1>{currentTitle}</h1>
          <p>Monitoring {(system.trading.lanes || []).length} symbol lanes · {system.training.activePhase} phase · Champion {system.registry.champion.id}</p>
        </div>
        <div className="hero-stamp">
          <div className="stamp-line">{selectedSymbol}</div>
          <div className="stamp-line">Uptime {uptime}</div>
          <div className="stamp-line">Loop #{system.orchestrator.loopIteration}</div>
        </div>
      </div>
      <div className="hero-metrics">
        <MetricTile label="Transport" value={transport} />
        <MetricTile label="Feed" value={transportState.mode} tone={transportState.mode === "degraded" ? "warn" : "pass"} />
        <MetricTile label="Loop" value={system.orchestrator.loopStatus} tone="pass" />
        <MetricTile label="Phase" value={system.training.activePhase} />
        <MetricTile label="Latency" value={`${system.connection.latencyMs}ms`} tone={system.connection.latencyMs > 80 ? "warn" : "pass"} />
        <MetricTile label="Queue" value={system.orchestrator.queueDepth} />
        <MetricTile label="Cooldown" value={shortDuration(system.orchestrator.cooldownSec)} />
        <MetricTile label="Canary" value={system.registry.canary.id} tone={system.registry.gate.ready ? "pass" : "warn"} />
      </div>
      {(transportState.error || lastActionResult) && (
        <div className="hero-alerts">
          {transportState.error && <div className="banner banner-warn"><AlertTriangle size={14} /><span>{transportState.error}</span></div>}
          {lastActionResult && <div className={`banner ${lastActionResult.ok === false ? "banner-fail" : ""}`}><CheckCircle2 size={14} /><span>{lastActionResult.action}: {lastActionResult.message || (lastActionResult.ok ? "ok" : "failed")}</span></div>}
        </div>
      )}
    </header>
  );
}

// ─── SIDE NAV ───
export function SideNav({ system, view, selectedSymbol, availableSymbols, onChange, onSelectSymbol, onStartJourney, onGoTrading, onGoControl, onReset }) {
  return (
    <aside className="side-nav">
      <div className="side-brand">
        <div className="eyebrow">System Shell</div>
        <h2>Cautious Giggle</h2>
        <p>Autonomous trading supervision and narrative shell.</p>
      </div>
      <div className="side-block">
        <div className="side-label">Master Navigation</div>
        <RouteTabs view={view} onChange={onChange} />
        <div className="side-bottom-actions" style={{ marginTop: "24px" }}>
          <Button icon={Clock3} tone="fail" onClick={onReset} style={{ width: "100%" }}>System Day Zero Reset</Button>
        </div>
      </div>
      <div className="side-block">
        <div className="side-label">Symbol Lanes</div>
        <div className="symbol-list">
          {availableSymbols.map((symbol) => (
            <button key={symbol} type="button" className={`symbol-pill ${selectedSymbol === symbol ? "symbol-pill-active" : ""}`} onClick={() => onSelectSymbol(symbol)}>
              <Radio size={11} /><span>{symbol}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="side-block">
        <div className="side-label">Quick State</div>
        <div className="side-metrics">
          <MetricTile label="Champion" value={system.registry.champion.id} tone="pass" />
          <MetricTile label="Equity" value={money(system.trading.account.equity)} tone={system.trading.account.floatingPnl >= 0 ? "pass" : "warn"} />
          <MetricTile label="Positions" value={String(system.trading.account.openPositions)} />
          <MetricTile label="Risk Cap" value={pct(system.trading.risk.sizeCap)} />
        </div>
      </div>
    </aside>
  );
}

// ─── SUPPORT RAIL ───
export function SupportRail({ system, selectedSymbol }) {
  const gateTone = system.registry.gate.ready ? "pass" : "warn";
  const lane = (system.trading.lanes || []).find((e) => e.symbol === selectedSymbol) || system.trading.lanes[0];
  const candidate = (system.registry.candidates || []).find((e) => e.symbol === selectedSymbol && e.verdict !== "rejected");
  const pnlHistory = system._history?.pnl || [];
  const equityHistory = system._history?.equity || [];
  return (
    <aside className="support-rail">
      <Panel title="Live Intelligence" subtitle={`${selectedSymbol} authority status`} icon={GitBranch}
        right={<span className={`rail-chip ${statusTone(gateTone)}`}>{system.registry.gate.ready ? "ready" : "hold"}</span>}>
        <div className="side-metrics">
          <MetricTile label="Lane Owner" value={lane?.champion || system.registry.champion.id} tone="pass" />
          <MetricTile label="Confidence" value={pct(lane?.confidence || 0)} sparkData={system._history?.confidence || []} tone={lane?.confidence > 0.6 ? "pass" : "warn"} />
          <MetricTile label="PnL" value={money(lane?.pnl || 0)} sparkData={pnlHistory} tone={lane?.pnl >= 0 ? "pass" : "warn"} />
          <MetricTile label="Equity" value={money(system.trading.account.equity)} sparkData={equityHistory} tone={system.trading.account.floatingPnl >= 0 ? "pass" : "warn"} />
        </div>
        <div className="rail-note">
          <strong>Gate:</strong> {system.registry.gate.reason}<br />
          <strong>Next:</strong> {system.selfImprove.lastImprovementAction}
        </div>
      </Panel>

      <Panel title="Training Pulse" subtitle="Live training metrics" icon={Activity}>
        <StatRow icon={Sparkles} label="LSTM Loss" value={system.training.lstm.loss.toFixed(3)} sub={`val ${system.training.lstm.valLoss.toFixed(3)}`} tone={system.training.lstm.loss < 0.5 ? "pass" : "info"} />
        <StatRow icon={Activity} label="PPO Progress" value={pct(system.training.ppo.progress)} sub={`${system.training.ppo.currentTimesteps.toLocaleString()} ts`} />
        <StatRow icon={Workflow} label="DreamerV3" value={pct(system.training.dreamerV3.alignment)} sub={`${system.training.dreamerV3.steps.toLocaleString()} steps`} />
        <StatRow icon={Shield} label="Memory" value={pct(system.training.lstm.memoryStrength)} sub={`${system.training.lstm.featuresUsed} feat`} tone="pass" />
      </Panel>

      <Panel title="Incidents" subtitle="Warnings and activity" icon={AlertTriangle}>
        <EventList items={system.incidents} empty="No active incidents." />
      </Panel>
    </aside>
  );
}

// ─── ROUTE TABS ───
export function RouteTabs({ view, onChange }) {
  return (
    <div className="route-tabs">
      {[
        [ROUTES.landing, "Landing"],
        [ROUTES.journey, "Journey Logs"],
        [ROUTES.trading, "Trading Watch"],
        [ROUTES.control, "Control Plane"],
        [ROUTES.strategies, "Strategies Engine"],
        [ROUTES.raw, "Raw Bot Data"],
        [ROUTES.about, "System About"],
      ].map(([key, label]) => (
        <button key={key} className={`tab ${view === key ? "tab-active" : ""}`} type="button" onClick={() => onChange(key)}>
          {label}
        </button>
      ))}
    </div>
  );
}
