import React, { useEffect, useRef, useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, YAxis } from "recharts";
import {
  ArrowDownRight, ArrowUpRight, BookOpen, Brain, Clock, Cpu, Crosshair, Database, FlaskConical, GraduationCap,
  HelpCircle, Moon, Newspaper, Pause, Radar, Receipt, Scissors, ShieldAlert, Sparkles, Target, Trophy, X, Zap,
  ArrowLeftRight, Layers, Ruler, Wifi,
} from "lucide-react";

export const clamp = (x) => Math.max(-1, Math.min(1, Number(x) || 0));
export const fmtPx = (v) => (v == null ? "—" : v.toLocaleString(undefined, {
  minimumFractionDigits: v > 1000 ? 2 : 3, maximumFractionDigits: v > 1000 ? 2 : v < 10 ? 4 : 3 }));
export const money = (v, sign = true) => (v == null ? "—"
  : `${sign ? (v >= 0 ? "+" : "−") : ""}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
export const until = (t0) => {
  const s = Math.round(t0 - Date.now() / 1000);
  if (s <= 0) return `${Math.round(-s / 60)} min ago`;
  return s >= 3600 ? `in ${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : `in ${Math.max(1, Math.round(s / 60))} min`;
};
export const ago = (t) => {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)}h ago`;
};
const hhmm = (t) => (t ? new Date(t * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "--:--");
const PLAY = { fade: "fade the sweep", breakout: "ride the breakout" };
const isLive = (n) => n && ["prep", "armed", "in_trade"].includes(n.phase);

/* ================================================================ section frame */
export function Section({ id, no, title, sub, right, children }) {
  return (
    <section id={id} className="sec">
      <div className="sec-head">
        <span className="sec-no">{no}</span>
        <h2>{title}</h2>
        {sub && <span className="ex sec-sub">{sub}</span>}
        {right && <span className="sec-right">{right}</span>}
      </div>
      {children}
    </section>
  );
}

/* ================================================================ header pieces */
const SESSIONS = [
  ["Sydney", "Australia/Sydney", 22, 7], ["Tokyo", "Asia/Tokyo", 0, 9],
  ["London", "Europe/London", 8, 17], ["New York", "America/New_York", 13, 22],
];
function sessionOpen(fromUtc, toUtc) {
  const d = new Date();
  if (d.getUTCDay() === 6 || (d.getUTCDay() === 0 && d.getUTCHours() < 22)) return false;
  const h = d.getUTCHours() + d.getUTCMinutes() / 60;
  return fromUtc < toUtc ? h >= fromUtc && h < toUtc : h >= fromUtc || h < toUtc;
}
export function Clocks() {
  const [, tick] = useState(0);
  useEffect(() => { const id = setInterval(() => tick((x) => x + 1), 15000); return () => clearInterval(id); }, []);
  return (
    <div className="clocks" title="Forex trading sessions. Markets move most when London and New York overlap. Crypto trades 24/7.">
      {SESSIONS.map(([n, tz, a, b]) => {
        const open = sessionOpen(a, b);
        return (
          <span key={n} className={`clock ${open ? "open" : ""}`}>
            <i />{n} <b>{new Date().toLocaleTimeString([], { timeZone: tz, hour: "2-digit", minute: "2-digit" })}</b>
          </span>
        );
      })}
    </div>
  );
}

export function condition(d) {
  const a = d.account || {};
  const used = a.daily_loss_limit ? Math.max(0, -a.today / a.daily_loss_limit) : 0;
  const live = (d.symbols || []).some((s) => isLive(s.news));
  if (!d.bot?.running) return { level: "red", label: "CONDITION RED", why: "The bot is stopped: open trades only have their broker stops." };
  if (d.bot?.broker_ok === false) return { level: "red", label: "CONDITION RED", why: "Lost the connection to Exness." };
  if (used >= 0.66) return { level: "red", label: "CONDITION RED", why: `${Math.round(used * 100)}% of today's loss limit used; new trades stop at 100%.` };
  if (used >= 0.33 || live) return { level: "amber", label: "CONDITION AMBER",
    why: live ? "A high-impact news release is in play: normal entries paused, the news plan is active." : `${Math.round(used * 100)}% of today's loss limit used.` };
  return { level: "green", label: "CONDITION GREEN", why: "Bot running, broker connected, well inside today's loss limit." };
}

export function NewsAlert({ d }) {
  const s = (d.symbols || []).filter((x) => isLive(x.news));
  if (!s.length) return null;
  const n = s[0].news;
  return (
    <div className="alert">
      <ShieldAlert size={16} /> <b>NEWS IN PLAY · {n.title}</b>
      <span>{n.phase === "prep" ? "range marked, normal entries paused on " : n.phase === "armed" ? "watching for the stop sweep on " : "news trade open on "}
        {s.map((x) => x.name).join(", ")} · plan: {s.map((x) => `${x.name} ${x.news.plan ? PLAY[x.news.plan] : "stand aside"}`).join(" · ")}</span>
    </div>
  );
}

export function Sitrep({ d }) {
  const syms = d.symbols || [];
  const open = syms.reduce((n, s) => n + (s.trades || []).length, 0);
  const waiting = syms.filter((s) => s.pending).length;
  const fl = d.account?.floating || 0;
  const tr = (d.mission || []).find((r) => r.progress) || (d.mission || []).find((r) => r.status === "training");
  const nxt = (d.news?.plan || []).filter((e) => e.t0 > Date.now() / 1000).sort((a, b) => a.t0 - b.t0)[0];
  return (
    <div className="sitrep">
      <span className="sit-k">SITREP</span>
      <span><b>{syms.length}</b> markets watched every 5 min</span>
      <span><b>{open}</b> trade{open === 1 ? "" : "s"} open{open ? <> (<b className={fl >= 0 ? "pos" : "neg"}>{money(fl)}</b>)</> : ""}</span>
      {waiting > 0 && <span><b>{waiting}</b> waiting for a better entry price</span>}
      {tr && <span>training <b>{tr.name}</b>{tr.scenario ? ` · ${tr.scenario} reward` : ""}{tr.progress?.total ? ` · ${Math.round((tr.progress.step / tr.progress.total) * 100)}%` : ""}</span>}
      {nxt && <span>next news <b>{nxt.title.split(" + ")[0]}</b> {until(nxt.t0)}</span>}
    </div>
  );
}

/* ================================================================ command tiles */
function Tile({ label, value, sub, tone, hint, children }) {
  return (
    <div className="tile hud">
      <div className="tile-label">{label}</div>
      <div className={`tile-value ${tone || ""}`}>{value}</div>
      {sub && <div className="tile-sub">{sub}</div>}
      {children}
      {hint && <div className="ex tile-hint">{hint}</div>}
    </div>
  );
}

export function CommandBar({ d, Animated }) {
  const a = d.account, st = d.stats;
  const used = a.daily_loss_limit ? Math.max(0, Math.min(1, -a.today / a.daily_loss_limit)) : 0;
  const open = d.symbols.reduce((n, s) => n + (s.trades || []).length, 0);
  const inMk = d.symbols.filter((s) => (s.trades || []).length).length;
  const win = st.closed ? Math.round((st.wins / st.closed) * 100) : null;
  return (
    <div className="tiles">
      <Tile label="EQUITY" value={<Animated value={a.equity} format={(v) => money(v, false)} />}
            sub={`balance ${money(a.balance, false)} · open trades ${money(a.floating)}`}
            hint="What the account is worth right now, including trades that are still open. Balance counts closed trades only." />
      <Tile label="TODAY" tone={a.today >= 0 ? "pos" : "neg"} value={<Animated value={a.today} format={(v) => money(v)} />}
            sub={`${a.today_pct >= 0 ? "+" : ""}${a.today_pct.toFixed(2)}% since midnight`}
            hint="Profit or loss since the start of the day (broker time), closed and open trades together." />
      <Tile label="LOSS STOP" tone={used > 0.66 ? "neg" : ""} value={`${Math.round(used * 100)}%`}
            sub={`of the $${(a.daily_loss_limit || 0).toFixed(0)} daily limit used`}
            hint="Safety brake: if today's loss reaches the limit, the bot stops opening trades until tomorrow.">
        <div className={`gbar ${used > 0.66 ? "hot" : used > 0.33 ? "warm" : ""}`}><span style={{ width: `${used * 100}%` }} /></div>
      </Tile>
      <Tile label="WIN RATE" value={win == null ? "—" : `${win}%`} sub={`${st.wins} won · ${st.losses} lost · ${money(st.pnl_closed)}`}
            hint="Share of today's closed trades that made money. Under 50% can still profit if winners are bigger than losers." />
      <Tile label="EXPOSURE" value={`${open}`} sub={`trade${open === 1 ? "" : "s"} in ${inMk} of ${d.symbols.length} markets · max 14`}
            hint="Each market can hold up to 3 trades; extra ones are only added while the open ones are winning.">
        <div className="gbar cyan"><span style={{ width: `${Math.min(100, (open / 14) * 100)}%` }} /></div>
      </Tile>
    </div>
  );
}

/* ================================================================ needle gauge */
const CX = 100, CY = 96, R = 78;
const ang = (v) => (clamp(v) * 90 - 90) * (Math.PI / 180);
const pt = (v, r = R) => [CX + r * Math.cos(ang(v)), CY + r * Math.sin(ang(v))];
const arc = (v1, v2, r = R) => { const [x1, y1] = pt(v1, r), [x2, y2] = pt(v2, r); return `M ${x1} ${y1} A ${r} ${r} 0 0 1 ${x2} ${y2}`; };

export function Gauge({ value, threshold, marks }) {
  const v = clamp(value);
  const strong = Math.abs(v) >= threshold;
  return (
    <div className="gauge" title="The needle is the final order signal. It trades when it passes a gold line. Dots: where each model points.">
      <svg viewBox="0 0 200 112" role="img" aria-label={`signal ${v.toFixed(2)}`}>
        <path d={arc(-1, 1)} className="g-track" />
        <path d={arc(-1, -threshold)} className="g-zone sell" />
        <path d={arc(threshold, 1)} className="g-zone buy" />
        {[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1].map((t) => {
          const [x1, y1] = pt(t, R - 13), [x2, y2] = pt(t, R - (t === 0 ? 22 : 18));
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} className="g-tick" />;
        })}
        {[-threshold, threshold].map((t) => {
          const [x1, y1] = pt(t, R + 7), [x2, y2] = pt(t, R - 7);
          return <line key={t} x1={x1} y1={y1} x2={x2} y2={y2} className="g-thr" />;
        })}
        {(marks || []).map((m) => {
          const [x, y] = pt(m.v, R + 1);
          return <circle key={m.k} cx={x} cy={y} r="3.8" className={`g-mark ${m.k}`}><title>{m.label}: {m.v.toFixed(2)}</title></circle>;
        })}
        <text x="14" y="108" className="g-lbl neg">SELL</text>
        <text x="186" y="108" textAnchor="end" className="g-lbl pos">BUY</text>
        <g className={`g-needle ${strong ? (v > 0 ? "buy" : "sell") : ""}`} style={{ transform: `rotate(${v * 90}deg)` }}>
          <polygon points={`${CX - 3.2},${CY} ${CX + 3.2},${CY} ${CX},${CY - R + 6}`} />
        </g>
        <circle cx={CX} cy={CY} r="7" className="g-hub" />
      </svg>
      <div className={`g-read ${v >= 0 ? "pos" : "neg"}`}>{v >= 0 ? "+" : ""}{v.toFixed(2)}<span className="muted"> · line ±{threshold.toFixed(2)}</span></div>
    </div>
  );
}

/* ================================================================ decision briefing (bars + plain English) */
const STEP_ICON = { LSTM: Brain, PPO: Cpu, Council: Layers, Blend: ArrowLeftRight, Action: Crosshair };
function Briefing({ s }) {
  const rows = (s.briefing || []).filter((r) => r.k !== "status");
  if (!rows.length) return <div className="muted small">waiting for the first 5-minute check…</div>;
  return (
    <div className="brief">
      {rows.filter((r) => r.k !== "Action").map((r) => (
        <div key={r.k} className={`b-row ${!r.v && ["LSTM", "PPO"].includes(r.k) ? "off" : ""} ${r.k === "Blend" ? "out" : ""}`}>
          <span className="b-k">{r.k === "Blend" ? "→ Order" : r.k}</span>
          <span className="b-bar">
            <span className={`b-fill ${r.v >= 0 ? "pos" : "neg"}`}
                  style={{ left: r.v >= 0 ? "50%" : `${50 + clamp(r.v) * 50}%`, width: `${Math.abs(clamp(r.v)) * 50}%` }} />
            <span className="b-mid" />
          </span>
          <span className={`b-v ${r.v > 0 ? "pos" : r.v < 0 ? "neg" : "muted"}`}>{r.v ? (r.v > 0 ? "+" : "") + r.v.toFixed(2) : "off"}</span>
        </div>
      ))}
    </div>
  );
}

function Explain({ s }) {
  const rows = (s.briefing || []).filter((r) => r.k !== "Action" && r.k !== "status");
  if (!rows.length) return null;
  return (
    <details className="ex how">
      <summary>How it decided <span className="muted">· {rows.length} steps</span></summary>
    <ol className="explain">
      {rows.map((r) => {
        const Icon = STEP_ICON[r.k] || Sparkles;
        return (
          <li key={r.k}><Icon size={12} /> <b>{r.k}:</b> {r.text}
            {r.k === "Council" && (s.trusts?.length > 0 || s.ignores?.length > 0) && (
              <span className="muted"> Trusts {s.trusts.join(", ") || "—"}; ignores {s.ignores.join(", ") || "—"}.</span>
            )}
          </li>
        );
      })}
    </ol>
    </details>
  );
}

function Slots({ trades, max }) {
  const n = Math.max(max || 1, trades.length);
  return (
    <div className="slots" title="Up to 3 trades per market. A new one is added only while the open ones are in profit and the signal agrees.">
      {Array.from({ length: n }).map((_, i) => {
        const t = trades[i];
        return (
          <div key={i} className={`slot ${t ? (t.side === "BUY" ? "long" : "short") : "empty"}`}>
            {t ? <><b>{t.side === "BUY" ? "▲" : "▼"} {t.volume}</b><span className={t.pnl >= 0 ? "pos" : "neg"}>{money(t.pnl)}</span></>
               : <span className="muted">{i === 0 ? "free" : `stack ${i + 1}`}</span>}
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

/* ================================================================ market card */
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
  const live = isLive(s.news);
  const action = (s.briefing || []).find((r) => r.k === "Action" || r.k === "status");
  return (
    <div className={`wcard hud ${p ? (p.side === "BUY" ? "in-long" : "in-short") : ""} ${live ? "news-live" : ""}`}>
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
              {trades.length > 1 ? `${trades.length} ${p.side === "BUY" ? "BUYS" : "SELLS"}` : p.side === "BUY" ? "LONG" : "SHORT"}
              <b className={total >= 0 ? "pos" : "neg"}>{money(total)}</b>
            </span>
          ) : s.pending ? (
            <span className="w-badge pending"><Clock size={13} /> {s.pending.side} ON DIP · {Math.floor(s.pending.expires_in / 60)}:{String(s.pending.expires_in % 60).padStart(2, "0")}</span>
          ) : (
            <span className="w-badge idle"><Pause size={13} /> WATCHING</span>
          )}
          {s.brain && <span className={`w-brain ${s.brain.startsWith("AI") ? "ai" : ""}`}
                            title={s.brain.startsWith("AI") ? "Its trained AI models vote together with the indicators."
                              : "No AI model has passed its exam for this market yet, so the indicator council decides alone."}>
            {s.brain.startsWith("AI") ? "AI + indicators" : "indicators only"}</span>}
        </div>
      </div>

      <div className="w-mid">
        <Gauge value={s.signal} threshold={s.threshold} marks={marks} />
        <Briefing s={s} />
      </div>

      {action && <div className="w-action"><Crosshair size={13} /> {action.text}</div>}
      {s.pending && (
        <div className="w-pend">target entry <b>{fmtPx(s.pending.limit)}</b> · price at signal {fmtPx(s.pending.ref)} · now {fmtPx(s.pending.side === "BUY" ? s.ask : s.bid)}</div>
      )}
      <Explain s={s} />

      <Slots trades={trades} max={s.stack_max} />

      {s.news && (
        <div className={`w-news ${live ? "live" : ""}`}>
          <Newspaper size={13} /> {s.news.title.split(" + ")[0]} {until(s.news.t0)} · plan: {s.news.plan ? PLAY[s.news.plan] : "stand aside"}
          {live && <b> · {s.news.phase.replace("_", " ").toUpperCase()}</b>}
        </div>
      )}
      {s.grad && !["live", "graduated"].includes(s.grad.status) && (
        <div className="w-grad"><GraduationCap size={13} /> AI {s.grad.label}</div>
      )}

      <div className="w-candles" title="Last 7 hours of 5-minute candles. Dashed: entry (gold), stop (red), target (green), price now (white).">
        <Candles candles={s.candles} position={p} bid={s.bid} />
      </div>
    </div>
  );
}

/* ================================================================ ops log */
const OPS = {
  plan: ["PLAN", "amber", Clock], entry: ["ENTRY", "green", Crosshair], stack: ["STACK", "cyan", Layers],
  close: ["CLOSE", "grey", Target], reduce: ["TRIM", "grey", Scissors], manage: ["STOP", "violet", ShieldAlert],
  size: ["SIZE", "grey", Ruler], warn: ["ALERT", "red", Wifi], news: ["NEWS", "amber", Newspaper],
  signal: ["SIGNAL", "cyan", ArrowLeftRight],
};
const LEARN_ICON = { ppo: Cpu, eval: FlaskConical, lstm: Brain, dreamer: Moon, memory: Database };

export function OpsLog({ ops, feed, trades }) {
  const [tab, setTab] = useState("actions");
  const actions = [...(ops || []), ...(feed || []).filter((f) => f.kind === "signal").map((f) => ({ ...f, kind: "signal" }))]
    .sort((a, b) => (b.t || 0) - (a.t || 0)).slice(0, 60);
  const learn = (feed || []).filter((f) => LEARN_ICON[f.kind]);
  const results = (trades || []);
  return (
    <div className="card ops hud">
      <div className="card-head">
        <div className="tabs">
          {[["actions", `Actions ${actions.length ? `(${actions.length})` : ""}`], ["results", "Results"], ["learning", "Learning"]].map(([k, l]) => (
            <button key={k} className={k === tab ? "active" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
        <span className="ex muted small">
          {tab === "actions" ? "every move the bot makes, and why" : tab === "results" ? "today's fills and closed trades" : "what training and the memory changed, and by how much"}
        </span>
      </div>
      <div className="ops-list">
        {tab === "actions" && (actions.length ? actions.map((e, i) => {
          const [tag, tone, Icon] = OPS[e.kind] || ["INFO", "grey", Sparkles];
          return (
            <div key={`${e.t}-${i}`} className="op">
              <span className="op-t">{hhmm(e.t)}</span>
              <span className={`op-tag ${tone}`}><Icon size={11} /> {tag}</span>
              <div className="op-body">
                <div className="op-title">{e.title}</div>
                <div className="ex op-text">{e.text}</div>
              </div>
            </div>
          );
        }) : <div className="muted small">Actions appear here the moment the bot plans, enters, stacks, trims or closes a trade.</div>)}

        {tab === "results" && (results.length ? results.map((t, i) => (
          <div key={i} className="op">
            <span className="op-t">{new Date(t.t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
            <span className={`op-tag ${t.closing ? (t.pnl > 0 ? "green" : "red") : "grey"}`}>
              {t.closing ? (t.pnl > 0 ? <Trophy size={11} /> : <Target size={11} />) : <Receipt size={11} />} {t.closing ? (t.pnl > 0 ? "WIN" : "LOSS") : "FILL"}
            </span>
            <div className="op-body">
              <div className="op-title">{t.closing ? `Closed ${t.symbol}` : `${t.side === "BUY" ? "Bought" : "Sold"} ${t.volume.toFixed(2)} ${t.symbol}`} @ {fmtPx(t.price)}</div>
            </div>
            {t.closing && <b className={t.pnl > 0 ? "pos" : "neg"}>{money(t.pnl)}</b>}
          </div>
        )) : <div className="muted small">No fills yet today.</div>)}

        {tab === "learning" && (learn.length ? learn.map((it, i) => {
          const Icon = LEARN_ICON[it.kind] || Sparkles;
          return (
            <div key={`${it.t}-${i}`} className="op">
              <span className="op-t">{hhmm(it.t)}</span>
              <span className="op-tag violet"><Icon size={11} /> {it.kind.toUpperCase()}</span>
              <div className="op-body">
                <div className="op-title">{it.title}</div>
                <div className="ex op-text">{it.text}</div>
                {it.numbers?.filter((n) => n.from != null).map((n, j) => (
                  <span key={j} className={`delta ${n.to >= n.from ? "pos" : "neg"}`}>{n.label} {n.from} → {n.to}</span>
                ))}
              </div>
            </div>
          );
        }) : <div className="muted small">Training and memory updates land here as they happen.</div>)}
      </div>
    </div>
  );
}

/* ================================================================ training mission control */
const STATUS = {
  live: "trading · keeps its model", graduated: "graduated · AI trading", training: "in training", queued: "queued",
  exam_failed: "failed exam · next scenario queued", retrain_queued: "next scenario queued", needs_work: "needs work · retry in 24h",
};

export function MissionControl({ rows, training }) {
  if (!rows || !rows.length) return null;
  const t = training || {};
  const cur = t.current;
  return (
    <div className="mc">
      <div className="card hud mission">
        <div className="m-legend ex">
          <span className="chip pass">passed</span><span className="chip fail">failed</span><span className="chip cur">training now</span>
          <span className="muted small">Each market gets its own AI. A model only trades after it passes an exam on 2 months of data it never saw. A fail retrains it with the next reward scenario.</span>
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
                <div className="m-name"><b>{r.name}</b><span className={`m-status ${r.status}`}>{STATUS[r.status] || r.status}</span></div>
                <div className="m-scen">
                  {(r.order || []).map((sc) => {
                    const h = tried[sc];
                    const isCur = r.status === "training" && r.scenario === sc;
                    const cls = isCur ? "cur" : h ? (h.passed ? "pass" : "fail") : r.best === sc ? "pass" : "";
                    return <span key={sc} className={`chip ${cls}`} title={h ? `${sc}: exam return ${h.ret != null ? (h.ret * 100).toFixed(2) + "%" : "?"}, Sharpe ${h.sharpe != null ? h.sharpe.toFixed(2) : "?"}` : sc}>{sc.replace("_", " ")}</span>;
                  })}
                </div>
                <div className="m-prog">
                  {pct != null ? (
                    <>
                      <div className="m-bar"><span style={{ width: `${pct}%` }} /></div>
                      <span className="small"><Cpu size={12} /> {pr.lane} · {Math.round(pct)}%{pr.eta_s ? ` · ${Math.max(1, Math.round(pr.eta_s / 60))} min left` : ""}</span>
                    </>
                  ) : r.status === "training" ? <span className="small muted"><Zap size={12} /> next in line for a training lane</span>
                    : last ? <span className="small">last exam ({last.scenario}) <b className={last.ret >= 0 ? "pos" : "neg"}>{last.ret != null ? `${(last.ret * 100).toFixed(2)}%` : "—"}</b>{last.sharpe != null && <span className="muted"> · Sharpe {last.sharpe.toFixed(1)}</span>}</span>
                    : <span className="small muted">no exam yet</span>}
                </div>
                <div className="m-why small muted ex">{r.status === "training" && r.about ? `${r.scenario}: ${r.about}` : r.why}</div>
              </div>
            );
          })}
        </div>
      </div>
      <div className="card hud now">
        <div className="card-title"><Radar size={15} /> {t.running ? "Running now" : "Training paused"}</div>
        {cur ? (
          <>
            <div className="now-name">{cur.name} · {cur.symbol}</div>
            {t.stage?.stage_progress != null && <div className="m-bar big"><span style={{ width: `${t.stage.stage_progress * 100}%` }} /></div>}
            <dl className="now-dl">
              <dt>Learns</dt><dd>{cur.learn}</dd>
              <dt>Aims for</dt><dd>{cur.goal}</dd>
              <dt>Tested by</dt><dd>{cur.test}</dd>
            </dl>
          </>
        ) : <div className="muted small">{t.note || "Waiting for the next training cycle."}</div>}
        <div className="stages">{(t.stages || []).map((st, i) => <span key={i} className={`stage ${st.status}`}>{st.label}</span>)}</div>
        <div className="ex muted small">Stages of this cycle in order. “kept” = a model that already exists is reused, so only the trader model retrains.</div>
      </div>
    </div>
  );
}

/* ================================================================ onboarding tour + glossary */
const TOUR = [
  { id: null, title: "Welcome to the war room",
    text: "This dashboard runs a trading bot on your Exness demo account. Every 5 minutes it checks 7 markets, decides whether to buy, sell or wait, and explains each move. It's practice money: the risk settings are deliberately aggressive so the AI learns faster." },
  { id: "sec-command", title: "01 · Command",
    text: "The account at a glance. Equity is what the account is worth now. LOSS STOP is the safety brake: when today's loss hits the limit, no new trades until tomorrow. The colour badge at the top (GREEN / AMBER / RED) sums it up." },
  { id: "sec-positions", title: "02 · Positions",
    text: "Every open trade in one blotter. The rail shows where the price sits between the stop-loss (red, left) and the target (green, right); the gold tick is the entry. R is profit in units of the risk taken: −1R means stopped out." },
  { id: "sec-markets", title: "03 · Markets: reading a card",
    text: "The needle is the bot's final verdict for that market, from SELL (left) to BUY (right). It only trades when the needle passes a gold line. The dots on the dial and the bars beside it show each voter: the LSTM pattern reader, the PPO trader and the indicator council. The line under the dial says what it will do next." },
  { id: "sec-markets", title: "Trade slots",
    text: "Each market can hold up to 3 trades. The 2nd and 3rd are only added while the open ones are already in profit and the signal still agrees, each with its own stop-loss." },
  { id: "sec-ops", title: "04 · Ops log",
    text: "Every action as it happens, in plain English: planned entries (it waits for a small pullback to get a better price), fills, stacks, stop moves, trims and closes, and why each one happened." },
  { id: "sec-training", title: "05 · Training",
    text: "Each market gets its own AI. A new model only trades after passing an exam on data it has never seen. If it fails, it retrains with a different reward scenario (patient, hunter, steady…), so it tries a different style, not just more of the same." },
  { id: "sec-news", title: "06 · News",
    text: "Before big US releases the bot marks the price range, pauses normal entries, and trades the reaction with the play that has worked before for that market." },
  { id: null, title: "You're set",
    text: "Turn SOUND on to hear a chime on every fill, stack and close. Turn EXPLAIN off in the top bar for a compact view once you know your way around. The ? button replays this tour, and GLOSSARY explains every term." },
];

export function Tour({ open, onClose }) {
  const [i, setI] = useState(0);
  useEffect(() => { if (open) setI(0); }, [open]);
  useEffect(() => {
    document.querySelectorAll(".tour-focus").forEach((el) => el.classList.remove("tour-focus"));
    if (!open) return;
    const el = TOUR[i].id && document.getElementById(TOUR[i].id);
    if (el) { el.classList.add("tour-focus"); el.scrollIntoView({ behavior: "smooth", block: "start" }); }
    else window.scrollTo({ top: 0, behavior: "smooth" });
  }, [open, i]);
  if (!open) return null;
  const st = TOUR[i];
  const last = i === TOUR.length - 1;
  return (
    <div className="tour">
      <div className="tour-card hud">
        <div className="tour-top"><span className="sec-no">{i + 1}/{TOUR.length}</span><button className="icon-btn" onClick={onClose} aria-label="close"><X size={16} /></button></div>
        <h3>{st.title}</h3>
        <p>{st.text}</p>
        <div className="tour-dots">{TOUR.map((_, k) => <i key={k} className={k === i ? "on" : ""} onClick={() => setI(k)} />)}</div>
        <div className="tour-btns">
          <button className="btn ghost" onClick={onClose}>Skip</button>
          {i > 0 && <button className="btn ghost" onClick={() => setI(i - 1)}>Back</button>}
          <button className="btn" onClick={() => (last ? onClose() : setI(i + 1))}>{last ? "Start" : "Next"}</button>
        </div>
      </div>
    </div>
  );
}

const EXTRA_TERMS = [
  ["ATR", "Average True Range: how far price typically moves in one bar. Stops, targets and entry pullbacks are measured in ATRs so they fit each market's speed."],
  ["Pullback entry", "Instead of chasing the price when a signal fires, the bot waits up to 15 min for a small move back (0.25–0.5 ATR) and enters there."],
  ["Stacking", "Adding a 2nd or 3rd trade on the same market while the open ones are in profit and the signal still agrees. Each has its own stop."],
  ["Reward scenario", "What the AI is rewarded for while it practises: balanced, patient (fewer trades), hunter (must act), steady (smooth equity), trend rider."],
  ["Exam", "A new model replays recent months it never trained on. It must make money after costs with an acceptable Sharpe and drawdown to start trading."],
  ["Sharpe", "Return divided by how bumpy it was. Above 1 is good; below 0 means it lost money."],
  ["Drawdown", "The drop from the account's highest point. The bot halts at 40% on this demo."],
  ["Lot", "Trade size unit. 0.01 lots is the smallest on most markets here."],
  ["Basis point", "0.01%. Entry improvements are measured in basis points."],
  ["Liquidity sweep", "Around news, price often spikes just past the recent range to trigger stops, then snaps back. The bot fades that or rides a true breakout, whichever has worked for that market."],
];

export function Glossary({ open, onClose, terms }) {
  const [q, setQ] = useState("");
  if (!open) return null;
  const all = [...(terms || []).map((t) => [t.term.replace(" ┃", ""), t.text]), ...EXTRA_TERMS]
    .sort((a, b) => a[0].localeCompare(b[0]))
    .filter(([t, x]) => !q || (t + x).toLowerCase().includes(q.toLowerCase()));
  return (
    <div className="drawer-bg" onClick={onClose}>
      <aside className="drawer hud" onClick={(e) => e.stopPropagation()}>
        <div className="tour-top"><h3><BookOpen size={16} /> Glossary</h3><button className="icon-btn" onClick={onClose} aria-label="close"><X size={16} /></button></div>
        <input className="search" placeholder="Search a term…" value={q} onChange={(e) => setQ(e.target.value)} autoFocus />
        <dl className="gloss">{all.map(([t, x]) => <React.Fragment key={t}><dt>{t}</dt><dd>{x}</dd></React.Fragment>)}</dl>
      </aside>
    </div>
  );
}

export function HelpButtons({ explain, setExplain, onTour, onGloss }) {
  return (
    <div className="hbtns">
      <button className={`toggle ${explain ? "on" : ""}`} onClick={() => setExplain(!explain)} title="Show or hide the plain-English explanations">
        <i /> EXPLAIN
      </button>
      <button className="icon-btn" onClick={onGloss} title="Glossary"><BookOpen size={16} /></button>
      <button className="icon-btn" onClick={onTour} title="Replay the tour"><HelpCircle size={16} /></button>
    </div>
  );
}

/* ================================================================ ticker tape */
export function Tape({ symbols }) {
  const items = (symbols || []).map((s) => {
    const pnl = (s.trades || []).reduce((a, t) => a + t.pnl, 0);
    const sig = clamp(s.signal);
    return (
      <span key={s.symbol} className="tp-item">
        <b>{s.name.toUpperCase()}</b>
        <span className="tp-px">{fmtPx(s.bid)}</span>
        {s.chg_pct != null && <span className={s.chg_pct >= 0 ? "pos" : "neg"}>{s.chg_pct >= 0 ? "▲" : "▼"}{Math.abs(s.chg_pct).toFixed(2)}%</span>}
        <span className={`tp-sig ${Math.abs(sig) >= s.threshold ? (sig > 0 ? "pos" : "neg") : "muted"}`}>
          SIG {sig >= 0 ? "+" : ""}{sig.toFixed(2)}
        </span>
        {(s.trades || []).length > 0 && <span className={`tp-pos ${pnl >= 0 ? "pos" : "neg"}`}>{s.trades.length}× {s.position?.side} {money(pnl)}</span>}
      </span>
    );
  });
  return (
    <div className="tape" title="Live prices · 24h change · signal (coloured once it is past the trade line) · open trades">
      <div className="tape-track">{items}{items}</div>
    </div>
  );
}

/* ================================================================ candlesticks */
export function Candles({ candles, position, bid, height = 70 }) {
  const cs = candles || [];
  if (cs.length < 2) return <div className="w-spark" />;
  // scale to the candles (+ entry); a stop/target far outside would squash the chart flat,
  // so those are pinned to the edge instead
  let lo = Math.min(...cs.map((c) => c[3]), bid ?? Infinity, position?.entry || Infinity);
  let hi = Math.max(...cs.map((c) => c[2]), bid ?? -Infinity, position?.entry || -Infinity);
  const span = hi - lo || Math.abs(hi) * 1e-4 || 1;
  for (const v of position ? [position.sl, position.tp] : []) {
    if (v > 0 && v < hi + span * 0.35 && v > lo - span * 0.35) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  }
  const pin = (v) => Math.min(hi, Math.max(lo, v));
  const W = cs.length * 6, H = height, pad = 3;
  const y = (v) => pad + (hi === lo ? H / 2 : ((hi - v) / (hi - lo)) * (H - 2 * pad));
  const line = (v, cls) => <line x1="0" x2={W} y1={y(v)} y2={y(v)} className={cls} vectorEffect="non-scaling-stroke" />;
  return (
    <svg className="candles" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" style={{ height }}
         aria-label="5-minute candles, last 7 hours">
      {cs.map((c, i) => {
        const [, o, h, l, cl] = c;
        const up = cl >= o;
        const x = i * 6 + 3;
        return (
          <g key={c[0]} className={up ? "cu" : "cd"}>
            <line x1={x} x2={x} y1={y(h)} y2={y(l)} vectorEffect="non-scaling-stroke" />
            <rect x={x - 2} width="4" y={Math.min(y(o), y(cl))} height={Math.max(0.8, Math.abs(y(o) - y(cl)))} />
          </g>
        );
      })}
      {position && line(position.entry, "lv entry")}
      {position && position.sl > 0 && line(pin(position.sl), `lv sl ${pin(position.sl) !== position.sl ? "off" : ""}`)}
      {position && position.tp > 0 && line(pin(position.tp), `lv tp ${pin(position.tp) !== position.tp ? "off" : ""}`)}
      {bid != null && line(bid, "lv now")}
    </svg>
  );
}

/* ================================================================ positions blotter */
const held = (t) => {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 3600 ? `${Math.floor(s / 60)}m` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};
export function Blotter({ rows }) {
  const tot = (rows || []).reduce((a, r) => a + r.pnl, 0);
  return (
    <div className="card hud blotter">
      {!rows?.length ? (
        <div className="muted small">No open trades. When the bot enters, each trade appears here with its stop, target and live distance to both.</div>
      ) : (
        <table className="blt">
          <thead>
            <tr><th>Market</th><th>Side</th><th className="r">Lots</th><th className="r">Entry</th><th className="r">Now</th>
              <th className="rail-h"><span className="neg">STOP</span><span className="muted">where price is</span><span className="pos">TARGET</span></th>
              <th className="r">R</th><th className="r">P&amp;L</th><th className="r">Held</th></tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.ticket} className={r.pnl >= 0 ? "win" : "lose"}>
                <td><b>{r.name}</b></td>
                <td><span className={`side ${r.side === "BUY" ? "long" : "short"}`}>{r.side}</span></td>
                <td className="r">{r.volume}</td>
                <td className="r">{fmtPx(r.entry)}</td>
                <td className="r">{fmtPx(r.now)}</td>
                <td>
                  {r.progress != null ? (
                    <div className="rail" title={`stop ${fmtPx(r.sl)} (${r.dist_sl_pct?.toFixed(2)}% away) · target ${fmtPx(r.tp)} (${r.dist_tp_pct?.toFixed(2)}% away)`}>
                      <span className="rail-fill" style={{ width: `${r.progress * 100}%` }} />
                      {r.entry_pos != null && <i className="rail-entry" style={{ left: `${r.entry_pos * 100}%` }} />}
                      <b className="rail-now" style={{ left: `${r.progress * 100}%` }} />
                      <span className="rail-l">{fmtPx(r.sl)}</span><span className="rail-r">{fmtPx(r.tp)}</span>
                    </div>
                  ) : <span className="muted small">no stop/target</span>}
                </td>
                <td className={`r ${r.locked ? "cyan" : r.r >= 0 ? "pos" : "neg"}`} title={r.locked ? "Stop moved to breakeven or better: this trade can no longer lose" : "Profit in units of the risk taken"}>
                  {r.locked ? "LOCKED" : r.r == null ? "—" : `${r.r >= 0 ? "+" : ""}${r.r.toFixed(2)}R`}</td>
                <td className={`r ${r.pnl >= 0 ? "pos" : "neg"}`}><b>{money(r.pnl)}</b></td>
                <td className="r muted">{held(r.opened)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot><tr><td colSpan={7} className="muted small ex">R = profit in units of the risk taken (−1R = stopped out, about +1.6R = target). LOCKED = the stop has moved to breakeven or better, so the trade can no longer lose. Gold tick on the rail = entry.</td>
            <td className={`r ${tot >= 0 ? "pos" : "neg"}`}><b>{money(tot)}</b></td><td /></tr></tfoot>
        </table>
      )}
    </div>
  );
}

/* ================================================================ fill alerts (toast + chime) */
let audioCtx = null;
function chime(kind) {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const notes = kind === "close" ? [660, 440] : kind === "stack" ? [520, 660, 880] : [440, 660];
    notes.forEach((f, i) => {
      const o = audioCtx.createOscillator(), g = audioCtx.createGain();
      o.type = "triangle"; o.frequency.value = f;
      const t = audioCtx.currentTime + i * 0.11;
      g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.08, t + 0.02); g.gain.exponentialRampToValueAtTime(0.0001, t + 0.18);
      o.connect(g).connect(audioCtx.destination); o.start(t); o.stop(t + 0.2);
    });
  } catch { /* audio unavailable */ }
}

export function FillAlerts({ ops, sound }) {
  const seen = useRef(null);
  const [toasts, setToasts] = useState([]);
  useEffect(() => {
    const evs = (ops || []).filter((e) => ["entry", "close", "stack"].includes(e.kind));
    const key = (e) => `${e.t}|${e.title}`;
    if (seen.current === null) { seen.current = new Set(evs.map(key)); return; }   // don't replay history on load
    const fresh = evs.filter((e) => !seen.current.has(key(e)));
    if (!fresh.length) return;
    fresh.forEach((e) => seen.current.add(key(e)));
    if (sound) chime(fresh[0].kind);
    const add = fresh.slice(0, 3).map((e) => ({ id: key(e) + Math.random(), ...e }));
    setToasts((t) => [...add, ...t].slice(0, 4));
    add.forEach((a) => setTimeout(() => setToasts((t) => t.filter((x) => x.id !== a.id)), 9000));
  }, [ops, sound]);
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.kind}`} title="click to dismiss" onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}>
          <span className="toast-k">{t.kind === "entry" ? "FILLED" : t.kind === "close" ? "CLOSED" : "STACKED"}</span>
          <b>{t.title}</b>
          <span className="muted small">{t.text}</span>
        </div>
      ))}
    </div>
  );
}

export function SoundButton({ on, set }) {
  return (
    <button className={`toggle ${on ? "on" : ""}`} onClick={() => { set(!on); if (!on) chime("entry"); }} title="Chime when a trade is filled, stacked or closed">
      <i /> SOUND
    </button>
  );
}
