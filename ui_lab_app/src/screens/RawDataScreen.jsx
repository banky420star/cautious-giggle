import React from "react";
import { Database, Binary, MemoryStick, Activity, Network } from "lucide-react";
import { Panel } from "../components/common";

export default function RawDataScreen({ system }) {
  return (
    <div className="stack">
      <Panel title="Raw Data Stream" subtitle="Unfiltered core engine telemetry and system variables" icon={Database}>
        <div className="card-grid two-up">
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.05)", maxHeight: "400px", overflowY: "auto" }}>
            <div className="eyebrow" style={{ marginBottom: "12px" }}><Binary size={12} /> Incoming Data</div>
            <pre style={{ margin: 0, color: "var(--cyan)", fontSize: "0.75rem", fontFamily: "var(--mono)", whiteSpace: "pre-wrap" }}>
{JSON.stringify({ connection: system.connection, account: system.trading.account }, null, 2)}
            </pre>
          </div>
          
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.05)", maxHeight: "400px", overflowY: "auto" }}>
            <div className="eyebrow" style={{ marginBottom: "12px" }}><MemoryStick size={12} /> LSTM & PPO Audit</div>
            <pre style={{ margin: 0, color: "var(--green)", fontSize: "0.75rem", fontFamily: "var(--mono)", whiteSpace: "pre-wrap" }}>
{JSON.stringify({ 
  lstm: system.training.lstm, 
  ppo: system.training.ppo,
  dreamer: system.training.dreamerV3 
}, null, 2)}
            </pre>
          </div>
          
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.05)", maxHeight: "400px", overflowY: "auto" }}>
            <div className="eyebrow" style={{ marginBottom: "12px" }}><Activity size={12} /> Added Features / Candidates</div>
            <pre style={{ margin: 0, color: "var(--amber)", fontSize: "0.75rem", fontFamily: "var(--mono)", whiteSpace: "pre-wrap" }}>
{JSON.stringify({ 
  candidates: system.registry.candidates,
  meta: system.meta,
  patternRecognition: system.patternRecognition
}, null, 2)}
            </pre>
          </div>
          
          <div style={{ background: "rgba(0,0,0,0.3)", padding: "16px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.05)", maxHeight: "400px", overflowY: "auto" }}>
            <div className="eyebrow" style={{ marginBottom: "12px" }}><Network size={12} /> All Running Processes</div>
            <pre style={{ margin: 0, color: "var(--purple)", fontSize: "0.75rem", fontFamily: "var(--mono)", whiteSpace: "pre-wrap" }}>
{JSON.stringify(system.controls.processes, null, 2)}
            </pre>
          </div>
        </div>
      </Panel>
    </div>
  );
}
