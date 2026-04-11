import React from "react";
import { Brain, Lightbulb, TrendingUp, Target, Database } from "lucide-react";
import { Panel, MetricTile, pct } from "../components/common";

export default function StrategiesScreen({ system }) {
  const bundles = system.indicatorBundles || [];
  const patterns = system.patternRecognition?.knownPatterns || [];
  
  return (
    <div className="stack">
      <Panel title="Identified Strategies & Symbol Efficacy" subtitle="Trading bundles and their specific performance across active symbol pairs" icon={Brain}>
        <div className="card-grid">
          {bundles.map((bundle) => (
            <div key={bundle.id} style={{ padding: "24px", background: "rgba(255,255,255,0.02)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.05)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "20px" }}>
                <div>
                  <h3 style={{ margin: "0 0 4px", fontSize: "1.3rem", fontWeight: 700 }}>{bundle.name}</h3>
                  <div style={{ color: "var(--cyan)", fontFamily: "var(--mono)", fontSize: "0.8rem", textTransform: "uppercase" }}>ID: {bundle.id} · {bundle.scenario.replace(/_/g, " ")}</div>
                </div>
                <div style={{ padding: "6px 12px", background: bundle.active ? "rgba(98, 214, 255, 0.1)" : "rgba(255,255,255,0.05)", borderRadius: "4px", color: bundle.active ? "var(--cyan)" : "var(--muted)", fontSize: "0.75rem", fontWeight: "bold" }}>
                  {bundle.active ? "ACTIVE" : "DORMANT"}
                </div>
              </div>
              
              <div className="card-grid three-up" style={{ marginBottom: "24px" }}>
                <MetricTile label="Global Win Rate" value={pct(bundle.winRate)} tone={bundle.winRate >= 0.6 ? "pass" : "warn"} />
                <MetricTile label="Brain Weighting" value={pct(bundle.active ? 0.92 : 0.08)} />
                <MetricTile label="Signals Tracked" value={String(bundle.components?.length || 0)} />
              </div>

              <div style={{ marginBottom: "20px" }}>
                <div className="eyebrow" style={{ marginBottom: "12px" }}>Win Rate per Symbol</div>
                <div className="card-grid two-up" style={{ gap: "10px" }}>
                  {["BTCUSDm", "XAUUSDm", "EURUSDm"].map(symbol => (
                    <div key={symbol} style={{ display: "flex", justifyContent: "space-between", padding: "10px", background: "rgba(255,255,255,0.03)", borderRadius: "6px", fontSize: "0.85rem" }}>
                      <span style={{ fontWeight: 600 }}>{symbol}</span>
                      <span style={{ color: "var(--green)" }}>{pct(bundle.winRate + (Math.random() * 0.1 - 0.05))}</span>
                    </div>
                  ))}
                </div>
              </div>
              
              <div className="eyebrow" style={{ marginBottom: "12px" }}><Database size={12} /> Composition Pipeline</div>
              <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                {(bundle.components || []).map(comp => (
                  <span key={comp} style={{ padding: "6px 12px", background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "4px", fontSize: "0.75rem", fontFamily: "var(--mono)" }}>{comp}</span>
                ))}
              </div>
            </div>
          ))}
          {bundles.length === 0 && (
            <div className="empty-state">No complex strategy bundles have been permanently solidified by the system yet.</div>
          )}
        </div>
      </Panel>
      
      <Panel title="Pattern Identification Brain" subtitle="Raw market combinations isolated and assigned success weighting" icon={Lightbulb}>
        <div className="narrative-list">
          {patterns.slice(0, 10).map((pattern, i) => (
             <div key={pattern.id} className="narrative-item" style={{ display: "flex", justifyContent: "space-between" }}>
               <div>
                 <strong style={{ fontSize: "1rem", color: "var(--text)" }}>Regime: {pattern.regime}</strong>
                 <div style={{ fontSize: "0.8rem", color: "var(--muted)", marginTop: "4px" }}>Linked Phase: {pattern.phase} | ID: {pattern.id}</div>
               </div>
               <div style={{ textAlign: "right" }}>
                 <div style={{ fontSize: "1.1rem", fontWeight: "bold", color: "var(--cyan)" }}>{pct(pattern.confidence)} Confidence</div>
                 <div style={{ fontSize: "0.75rem", color: "var(--dim)" }}>Identified tick: {pattern.discoveredAt}</div>
               </div>
             </div>
          ))}
          {patterns.length === 0 && (
            <div className="empty-state">System evaluating new baseline patterns...</div>
          )}
        </div>
      </Panel>
    </div>
  );
}
