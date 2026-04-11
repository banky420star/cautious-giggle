import React from "react";
import { BrainCircuit, GitMerge, ShieldAlert, Database, Rocket, Workflow } from "lucide-react";
import { Panel } from "../components/common";

function ArchitectureStage({ index, total, title, technical, purpose, icon: Icon, tone, description, detailGrid }) {
  return (
    <div className={`authority-stage ${tone} animate-in`} style={{ animationDelay: `${index * 150}ms`, width: "100%", maxWidth: "900px", margin: "0 auto" }}>
      <div className="authority-marker">
        <div className="authority-number">0{index + 1}</div>
        {index < total - 1 ? <div className="authority-line" /> : null}
      </div>
      <div className="authority-card">
        <div className="authority-card-top">
          <div>
            <div className="eyebrow-row">
              <Icon size={14} />
              <span>{technical}</span>
            </div>
            <h3>{title}</h3>
          </div>
        </div>
        <p className="authority-purpose">{purpose}</p>
        
        {description && (
          <div style={{ marginTop: "12px", borderTop: "1px solid var(--line)", paddingTop: "12px", color: "var(--muted)", fontSize: "0.95rem", lineHeight: 1.6 }}>
            {description}
          </div>
        )}

        {detailGrid && (
          <div className="card-grid two-up" style={{ marginTop: "16px" }}>
            {detailGrid.map((detail, i) => (
              <div key={i} style={{ padding: "14px", background: "rgba(0,0,0,0.25)", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.04)" }}>
                <strong style={{ color: detail.color || "var(--cyan)", display: "block", marginBottom: "6px" }}>{detail.title}</strong>
                <p style={{ color: "var(--muted)", fontSize: "0.85rem", lineHeight: 1.5, margin: 0 }}>{detail.desc}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default function AboutScreen() {
  const steps = [
    {
      title: "Granular Data & Feature Engineering", technical: "Data Pipeline", icon: Database, tone: "info",
      purpose: "The entire system feeds on live price, volume, and depth data before mathematically expanding it into hundreds of features.",
      description: "Data is retrieved natively in pure Python via a highly specialized MetaTrader5 HFT pipeline running at granular minute resolutions. Once collected, raw ticks are aggressively engineered into over 150 advanced mathematical features (e.g. Volume profiles, RSI divergencies, fractional differentiation, and volatility scaling bounds) before it touches any neural network."
    },
    {
      title: "Latent Context Engine", technical: "LSTM Time-Series", icon: BrainCircuit, tone: "pass",
      purpose: "Long Short-Term Memory parses complex arrays of past price action to intuitively 'feel' the market regime.",
      description: "It doesn't make trading decisions immediately; it summarizes a sequence of up to 60-depth inputs into a highly compressed 'Latent Context' representing the immediate physical scenario (e.g., chopping, trending, expanding mathematically)."
    },
    {
      title: "Execution & Imagination", technical: "DreamerV3 + PPO Policy", icon: Rocket, tone: "warn",
      purpose: "Two entirely separate neural networks evaluate the situation simultaneously; one simulates disaster while the other hunts for profit.",
      detailGrid: [
        {title: "DreamerV3 (Imagination)", color: "var(--cyan)", desc: "Simulates thousands of potential future ticks 'in its head' before execution. It issues direct alerts if a trajectory leads to a highly probable disaster."},
        {title: "PPO (The Pilot)", color: "var(--green)", desc: "The tactical pilot. It precisely merges the LSTM context with Dreamer's warnings to map out optimal pathways and assigns an exact confidence score for every symbol lane."}
      ]
    },
    {
      title: "Risk Overlord Interception", technical: "Hard Cutoff Hierarchy", icon: ShieldAlert, tone: "fail",
      purpose: "Absolute mathematical preservation of capital supersedes any AI model confidence.",
      description: "Risk calculation is completely decoupled from the autonomous AI brains. The `risk_supervisor` module strictly audits incoming PPO execution decisions. Rigid cutoff constraints physically intercept and kill trade pipelines if daily limits or maximum drawdowns are threatened."
    },
    {
      title: "The Evolutionary Loop", technical: "Continuous Learning", icon: GitMerge, tone: "pass",
      purpose: "A continuous survival-of-the-fittest architecture constantly replaces underperforming neural models with hyper-tuned mutations.",
      detailGrid: [
        {title: "Recursive Corrections", color: "var(--amber)", desc: "When trades result in stop-losses, the setup boundaries are mathematically isolated and specific neural action-weights are physically shifted downward in the feed."},
        {title: "Ghost Promotion", color: "var(--cyan)", desc: "Mutated 'canary' models trade continuously in the background. Once they mathematically outperform the live 'champion', they instantly take control."}
      ]
    }
  ];

  return (
    <div className="stack animate-in about-workflow">
      <div style={{ textAlign: "center", marginBottom: "30px", marginTop: "10px" }}>
        <h1 style={{ fontSize: "2.4rem", letterSpacing: "-0.05em", color: "var(--cyan)", margin: "0 0 10px 0", textTransform: "uppercase" }}>Pipeline Architecture</h1>
        <p style={{ color: "var(--muted)", maxWidth: "600px", margin: "0 auto", lineHeight: 1.6, fontSize: "1.1rem" }}>
          The autonomous trading pipeline isn't a simple conditional bot. It is a massive, self-improving sequence of 
          AI operations. The explicit workflow below details the pipeline from raw data to live execution.
        </p>
      </div>

      <div style={{ maxWidth: "900px", margin: "0 auto", width: "100%" }}>
        <Panel title="The Machine Workflow" subtitle="Detailed breakdown of the continuous intelligence pipeline" icon={Workflow}>
          <div className="authority-timeline" style={{ padding: "20px 10px 0" }}>
            {steps.map((step, idx) => (
              <ArchitectureStage 
                key={idx} 
                index={idx} 
                total={steps.length}
                {...step} 
              />
            ))}
          </div>
        </Panel>
      </div>
    </div>
  );
}
