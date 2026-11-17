import React, { useEffect, useRef, useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, YAxis } from "recharts";
import { ArrowDownRight, ArrowUpRight, Clock, Cpu, Crosshair, GraduationCap, Newspaper, Pause, Radar, Zap } from "lucide-react";

const clamp = (x) => Math.max(-1, Math.min(1, Number(x) || 0));
const fmtPx = (v) => (v == null ? "—" : v.toLocaleString(undefined, {
  minimumFractionDigits: v > 1000 ? 2 : 3, maximumFractionDigits: v > 1000 ? 2 : 3 }));
const money = (v) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}$${Math.abs(v).toFixed(2)}`);
const until = (t0) => {
  const s = Math.round(t0 - Date.now() / 1000);
  if (s <= 0) return "now";
  return s >= 3600 ? `in ${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : `in ${Math.max(1, Math.round(s / 60))}m`;
};
const PLAY = { fade: "fade the sweep", breakout: "ride the breakout" };

/* ---------------------------------------------------------------- needle gauge */
const CX = 100, CY = 96, R = 78;
const ang = (v) => (clamp(v) * 90 - 90) * (Math.PI / 180);          // -1 -> left, +1 -> right
const pt = (v, r = R) => [CX + r * Math.cos(ang(v)), CY + r * Math.sin(ang(v))];
function arc(v1, v2, r = R) {
  const [x1, y1] = pt(v1, r), [x2, y2] = pt(v2, r);
  return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`;
}

export function Gauge({ value, threshold, marks }) {
  const v = clamp(value);
  const strong = Math.abs(v) >= threshold;
  const deg = v * 90;
  const ticks = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1];
  return (
    <div className="gauge">
      <svg viewBox="0 0 200 112" role="img" aria-label={`signal ${v.toFixed(2)}`}>
        <path d={arc(-1, 1)} className="g-track" />
        <path d={arc(-1, -threshold)} className="g-zone sell" />
        <path d={arc(threshold, 1)} className="g-zone buy" />
        {ticks.map((t) => {
          const [x1, y1] = pt(t, R - 13), [x2, y2] = pt(t, R - (t === 0 ? 22 : 18));
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} className="g-tick" />;
        })}
        {[-threshold, threshold].map((t) => {
          const [x1, y1] = pt(t, R + 7), [x2, y2] = pt(t, R - 7);
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} className="g-thr" />;
        })}
        {(marks || []).map((m) => {
          const [x, y] = pt(m.v, R + 1);
          return <circle key={m.k} cx={x} cy={y} r="3.6" className={`g-mark ${m.k}`}><title>{m.label}: {m.v.toFixed(2)}</title></circle>;
        })}
        <text x="14" y="108" className="g-lbl neg">SELL</text>
        <text x="186" y="108" textAnchor="end" className="g-lbl pos">BUY</text>
        <g className={`g-needle ${strong ? (v > 0 ? "buy" : "sell") : ""}`} style={{ transform: `rotate(${deg}deg)` }}>
          <polygon points={`${CX - 3.2},${CY} ${CX + 3.2},${CY} ${CX},${CY - R + 6}`} />
        </g>
        <circle cx={CX} cy={CY} r="7" className="g-hub" />
      </svg>
      <div className={`g-read ${v >= 0 ? "pos" : "neg"}`}>
        {v >= 0 ? "+" : ""}{v.toFixed(2)}
        <span className="muted"> / trades past ±{threshold.toFixed(2)}</span>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- decision pipeline */
function Pipe({ p }) {
  if (!p) return <div className="pipe muted small">waiting for its first check…</div>;
  const rows = [
    ["LSTM", p.lstm], ["PPO", p.ppo], ["Council", p.council], ["Blend", p.blend], ["Order", p.final],
  ];
  return (
    <div className="pipe">
      {rows.map(([k, v], i) => (
        <div className={`pipe-row ${i >= 3 ? "out" : ""} ${!v && i < 3 ? "off" : ""}`} key={k}>
          <span className="pipe-k">{i >= 3 ? "→ " : ""}{k}</span>
          <span className="pipe-bar">
            <span className={`pipe-fill ${v >= 0 ? "pos" : "neg"}`}
                  style={{ left: v >= 0 ? "50%" : `${50 + clamp(v) * 50}%`, width: `${Math.abs(clamp(v)) * 50}%` }} />
            <span className="pipe-mid" />
          </span>
          <span className={`pipe-v ${v > 0 ? "pos" : v < 0 ? "neg" : "muted"}`}>{v ? (v > 0 ? "+" : "") + v.toFixed(2) : "·"}</span>
        </div>
      ))}
    </div>
  );
}

/* ---------------------------------------------------------------- stacked trades */
function Slots({ trades, max }) {
  const n = Math.max(max || 1, trades.length);
  return (
    <div className="slots" title="each market can hold several trades; a new one is added only while the open ones are in profit">
      {Array.from({ length: n }).map((_, i) => {
        const t = trades[i];
        return (
          <div key={i} className={`slot ${t ? (t.side === "BUY" ? "long" : "short") : "empty"}`}>
            {t ? <><b>{t.side === "BUY" ? "▲" : "▼"} {t.volume}</b><span className={t.pnl >= 0 ? "pos" : "neg"}>{money(t.pnl)}</span></>
               : <span className="muted">slot {i + 1}</span>}
          </div>
        );
      })}
    </div>
  );
}

function useFlash(price) {
  const prev = useRef(price);
  const [f, setF] = useState("");
  useEffect(() => {
    if (price != null && prev.current != null && price !== prev.current) {
      setF(price > prev.current ? "flash-up" : "flash-down");
      const id = setTimeout(() => setF(""), 900);
      prev.current = price;
      return () => clearTimeout(id);
    }
    prev.current = price;
  }, [price]);
  return f;
}

/* ---------------------------------------------------------------- market card */
export function WarCard({ s }) {
  const flash = useFlash(s.bid);
  const bars = (s.bars || []).map((v, i) => ({ i, v }));
  if (s.bid != null) bars.push({ i: bars.length, v: s.bid });
  const p = s.position;
  const trades = s.trades || [];
  const up = bars.length > 1 && bars[bars.length - 1].v >= bars[0].v;
  const vals = bars.map((b) => b.v).concat(p ? [p.entry, p.sl, p.tp].filter(Boolean) : []);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pl = s.pipeline;
  const marks = pl ? [
    pl.lstm ? { k: "lstm", v: pl.lstm, label: "LSTM" } : null,
    pl.ppo ? { k: "ppo", v: pl.ppo, label: "PPO" } : null,
    pl.council ? { k: "council", v: pl.council, label: "Indicator council" } : null,
  ].filter(Boolean) : [];
  const total = trades.reduce((a, t) => a + t.pnl, 0);
  const live = s.news && ["prep", "armed", "in_trade"].includes(s.news.phase);
  return (
    <div className={`wcard ${p ? (p.side === "BUY" ? "in-long" : "in-short") : ""} ${live ? "news-live" : ""}`}>
      <div className="w-head">
        <div>
          <div className="w-name">{s.name}<span className="w-sym">{s.symbol}</span></div>
          <div className={`w-price ${flash}`}>{fmtPx(s.bid)}</div>
          {s.chg_pct != null && <div className={`w-chg ${s.chg_pct >= 0 ? "pos" : "neg"}`}>{s.chg_pct >= 0 ? "▲ +" : "▼ "}{s.chg_pct.toFixed(2)}% · 24h</div>}
        </div>
        <div className="w-state">
          {trades.length ? (
            <span className={`w-badge ${p.side === "BUY" ? "long" : "short"}`}>
              {p.side === "BUY" ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
              {trades.length > 1 ? `${trades.length} ${p.side === "BUY" ? "buys" : "sells"}` : p.side === "BUY" ? "Buying" : "Selling"}
              <b className={total >= 0 ? "pos" : "neg"}>{money(total)}</b>
            </span>
          ) : s.pending ? (
            <span className="w-badge pending"><Clock size={13} /> {s.pending.side === "BUY" ? "Buy" : "Sell"} on dip · {Math.floor(s.pending.expires_in / 60)}:{String(s.pending.expires_in % 60).padStart(2, "0")}</span>
          ) : (
            <span className="w-badge idle"><Pause size={13} /> Watching</span>
          )}
          {s.brain && <span className={`w-brain ${s.brain.startsWith("AI") ? "ai" : ""}`}>{s.brain}</span>}
        </div>
      </div>

      <div className="w-mid">
        <Gauge value={s.signal} threshold={s.threshold} marks={marks} />
        <Pipe p={pl} />
      </div>

      <Slots trades={trades} max={s.stack_max} />

      <p className="w-why">{s.why}</p>

      {s.news && (
        <div className={`w-news ${live ? "live" : ""}`}>
          <Newspaper size={13} /> {s.news.title} · {until(s.news.t0)} · {s.news.plan ? PLAY[s.news.plan] : "stand aside"}
          {live && <b> · {s.news.phase.replace("_", " ").toUpperCase()}</b>}
        </div>
      )}
      {s.grad && s.grad.status !== "live" && s.grad.status !== "graduated" && (
        <div className={`w-grad ${s.grad.status}`}><GraduationCap size={13} /> AI: {s.grad.label}</div>
      )}

      <div className="w-spark">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={bars} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
            <YAxis hide domain={[lo, hi]} />
            {p && <ReferenceLine y={p.entry} stroke="#facc15" strokeDasharray="3 3" />}
            {p && p.sl > 0 && <ReferenceLine y={p.sl} stroke="#f87171" strokeDasharray="3 3" />}
            {p && p.tp > 0 && <ReferenceLine y={p.tp} stroke="#34d399" strokeDasharray="3 3" />}
            <Line type="monotone" dataKey="v" stroke={up ? "#34d399" : "#f87171"} strokeWidth={1.6} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ---------------------------------------------------------------- training mission control */
const STATUS = {
  live: "trading · model kept", graduated: "graduated", training: "training", queued: "queued",
  exam_failed: "failed exam · next scenario queued", retrain_queued: "next scenario queued", needs_work: "needs work",
};

export function MissionControl({ rows, training }) {
  if (!rows || !rows.length) return null;
  return (
    <div className="card mission">
      <div className="card-title"><Radar size={16} /> Training mission control
        <span className="muted small right">each failed exam retrains with the next reward scenario · {training?.running ? "cycle running" : "between cycles"}</span>
      </div>
      <div className="m-grid">
        {rows.map((r) => {
          const tried = {};
          (r.history || []).forEach((h) => { tried[h.scenario] = h; });
          const pr = r.progress;
          const pct = pr && pr.total ? Math.min(100, (pr.step / pr.total) * 100) : null;
          const last = (r.history || [])[r.history.length - 1];
          return (
            <div key={r.symbol} className={`m-row ${r.status}`}>
              <div className="m-name"><b>{r.name}</b>
                <span className={`m-status ${r.status}`}>{STATUS[r.status] || r.status}</span>
              </div>
              <div className="m-scen">
                {(r.order || []).map((sc) => {
                  const h = tried[sc];
                  const cur = r.status === "training" && r.scenario === sc;
                  const cls = cur ? "cur" : h ? (h.passed ? "pass" : "fail") : r.best === sc ? "pass" : "";
                  return (
                    <span key={sc} className={`chip ${cls}`}
                          title={h ? `${sc}: return ${h.ret != null ? (h.ret * 100).toFixed(2) + "%" : "?"}, Sharpe ${h.sharpe != null ? h.sharpe.toFixed(2) : "?"}` : sc}>
                      {sc.replace("_", " ")}
                    </span>
                  );
                })}
              </div>
              <div className="m-prog">
                {pct != null ? (
                  <>
                    <div className="m-bar"><span style={{ width: `${pct}%` }} /></div>
                    <span className="small"><Cpu size={12} /> {pr.lane} · {Math.round(pct)}% · {pr.fps ? `${Math.round(pr.fps)}/s` : ""}
                      {pr.eta_s ? ` · ${Math.max(1, Math.round(pr.eta_s / 60))} min left` : ""}</span>
                  </>
                ) : r.status === "training" ? (
                  <span className="small muted"><Zap size={12} /> next in line for a training lane</span>
                ) : last ? (
                  <span className="small">last exam ({last.scenario}): <b className={last.ret >= 0 ? "pos" : "neg"}>
                    {last.ret != null ? `${(last.ret * 100).toFixed(2)}%` : "—"}</b>
                    {last.sharpe != null && <span className="muted"> · Sharpe {last.sharpe.toFixed(1)}</span>}</span>
                ) : (
                  <span className="small muted">no exam yet</span>
                )}
              </div>
              <div className="m-why small muted">{r.status === "training" && r.about ? `${r.scenario}: ${r.about}` : r.why}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export function RiskStrip({ d }) {
  const trades = d.symbols.reduce((a, s) => a + (s.trades || []).length, 0);
  const maxOpen = 14;
  const live = d.symbols.filter((s) => (s.trades || []).length).length;
  const a = d.account;
  const used = a.daily_loss_limit ? Math.max(0, Math.min(1, -a.today / a.daily_loss_limit)) : 0;
  return (
    <div className="risk-strip">
      <span><Crosshair size={13} /> {trades} open trade{trades === 1 ? "" : "s"} across {live} of {d.symbols.length} markets
        <span className="mini-bar"><span style={{ width: `${Math.min(100, (trades / maxOpen) * 100)}%` }} /></span> limit {maxOpen}</span>
      <span>daily loss stop ${a.daily_loss_limit?.toFixed(0)} · used {Math.round(used * 100)}%
        <span className={`mini-bar ${used > 0.6 ? "hot" : ""}`}><span style={{ width: `${used * 100}%` }} /></span></span>
    </div>
  );
}
