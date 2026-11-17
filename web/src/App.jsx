import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  Activity, ArrowDownRight, ArrowUpRight, Brain, CircleDollarSign, Clock, GraduationCap, Lightbulb, Pause,
  Receipt, ShieldCheck, Target, TrendingUp, Trophy, Wallet, Zap, Sparkles, Cpu, FlaskConical, Database,
  ArrowLeftRight, Moon, Newspaper, CalendarClock,
} from "lucide-react";
import { WarCard, MissionControl, RiskStrip } from "./WarRoom.jsx";

const fmtMoney = (v, sign = false) => {
  if (v == null || Number.isNaN(v)) return "—";
  const s = Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  if (!sign) return `$${s}`;
  return `${v >= 0 ? "+" : "−"}$${s}`;
};
const fmtPx = (v) => (v == null ? "—" : v.toLocaleString(undefined, {
  minimumFractionDigits: v > 1000 ? 2 : 3, maximumFractionDigits: v > 1000 ? 2 : 3 }));
const fmtDur = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}h ${Math.round((s % 3600) / 60)}m` : `${Math.max(1, Math.round(s / 60))} min`);
const fmtTime = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

function useApi(intervalMs = 2000) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const r = await fetch("/api/state", { cache: "no-store" });
        const j = await r.json();
        if (!alive) return;
        if (j.error) setErr(j.error); else { setData(j); setErr(null); }
      } catch (e) { if (alive) setErr("Can't reach the dashboard server"); }
    };
    tick();
    const id = setInterval(tick, intervalMs);
    return () => { alive = false; clearInterval(id); };
  }, [intervalMs]);
  return [data, err];
}

function AnimatedNumber({ value, format }) {
  const [shown, setShown] = useState(value ?? 0);
  const from = useRef(value ?? 0);
  useEffect(() => {
    if (value == null) return;
    const start = performance.now();
    const a = from.current;
    let raf;
    const step = (t) => {
      const k = Math.min(1, (t - start) / 700);
      const e = 1 - Math.pow(1 - k, 3);
      const v = a + (value - a) * e;
      setShown(v);
      if (k < 1) raf = requestAnimationFrame(step); else from.current = value;
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);
  return <>{format(shown)}</>;
}

function Countdown({ seconds }) {
  const [s, setS] = useState(seconds);
  useEffect(() => setS(seconds), [seconds]);
  useEffect(() => { const id = setInterval(() => setS((x) => (x > 0 ? x - 1 : 299)), 1000); return () => clearInterval(id); }, []);
  const frac = 1 - s / 300;
  const r = 9, c = 2 * Math.PI * r;
  return (
    <div className="countdown" title="The bot re-checks the market every 5 minutes">
      <svg width="24" height="24" viewBox="0 0 24 24">
        <circle cx="12" cy="12" r={r} className="ring-bg" />
        <circle cx="12" cy="12" r={r} className="ring-fg" strokeDasharray={c} strokeDashoffset={c * (1 - frac)} />
      </svg>
      <span>Next check <b>{Math.floor(s / 60)}:{String(s % 60).padStart(2, "0")}</b></span>
    </div>
  );
}

function Header({ d }) {
  const bot = d?.bot || {};
  return (
    <header className="header">
      <div className="brand">
        <img src="/icon.svg" alt="" className="logo" />
        <div>
          <div className="brand-name">Cautious Giggle <span className="wr-tag">WAR ROOM</span></div>
          <div className="brand-sub">Exness demo · {d?.symbols?.length || 2} markets trading · 5-minute bars</div>
        </div>
      </div>
      <div className="header-right">
        {bot.learning_mode && (
          <span className="pill violet" title="Demo only: trades more often so the models learn faster">
            <GraduationCap size={14} /> Learning mode
          </span>
        )}
        <span className={`pill ${bot.running ? "green" : "red"}`}>
          <span className={`dot ${bot.running ? "pulse" : ""}`} /> {bot.running ? "Bot running" : "Bot stopped"}
        </span>
        {d && <Countdown seconds={bot.next_check_s} />}
      </div>
    </header>
  );
}

function Kpi({ icon: Icon, label, children, sub, tone }) {
  return (
    <div className="card kpi">
      <div className="kpi-label"><Icon size={15} /> {label}</div>
      <div className={`kpi-value ${tone || ""}`}>{children}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  );
}

const RANGES = [["1H", 3600e3], ["6H", 6 * 3600e3], ["1D", 86400e3], ["1W", 7 * 86400e3], ["All", Infinity]];

function EquityChart({ series, equity }) {
  const [range, setRange] = useState("1D");
  const data = useMemo(() => {
    const ms = RANGES.find((r) => r[0] === range)[1];
    const now = Date.now();
    const pts = (series || []).filter((p) => now - p.t <= ms);
    return pts.length >= 2 ? pts : (series || []).slice(-2);
  }, [series, range]);
  const first = data[0]?.equity, last = data[data.length - 1]?.equity;
  const chg = first != null && last != null ? last - first : 0;
  const up = chg >= 0;
  const vals = data.map((p) => p.equity);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.15, 0.5);
  return (
    <div className="card chart-card">
      <div className="card-head">
        <div>
          <div className="card-title"><TrendingUp size={16} /> Equity</div>
          <div className="chart-big">
            <AnimatedNumber value={equity} format={(v) => fmtMoney(v)} />
            <span className={`chg ${up ? "pos" : "neg"}`}>
              {up ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
              {fmtMoney(chg, true)} ({first ? ((chg / first) * 100).toFixed(2) : "0.00"}%) · {range}
            </span>
          </div>
        </div>
        <div className="tabs">
          {RANGES.map(([k]) => (
            <button key={k} className={k === range ? "active" : ""} onClick={() => setRange(k)}>{k}</button>
          ))}
        </div>
      </div>
      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="eqfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={up ? "#34d399" : "#f87171"} stopOpacity={0.35} />
                <stop offset="100%" stopColor={up ? "#34d399" : "#f87171"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={fmtTime}
                   stroke="#5b6475" fontSize={11} tickLine={false} axisLine={false} minTickGap={50} />
            <YAxis domain={[lo - pad, hi + pad]} tickFormatter={(v) => `$${v.toFixed(0)}`} stroke="#5b6475"
                   fontSize={11} tickLine={false} axisLine={false} width={52} />
            <Tooltip content={<EqTip />} cursor={{ stroke: "rgba(255,255,255,0.2)" }} />
            <Area type="monotone" dataKey="balance" stroke="#64748b" strokeDasharray="4 4" fill="none"
                  strokeWidth={1.2} isAnimationActive={false} dot={false} />
            <Area type="monotone" dataKey="equity" stroke={up ? "#34d399" : "#f87171"} fill="url(#eqfill)"
                  strokeWidth={2.2} isAnimationActive animationDuration={600} dot={false} />
          </AreaChart>
        </ResponsiveContainer>
      </div>
      <div className="legend"><span className="lg eq" /> equity (incl. open trades) <span className="lg bal" /> balance (closed trades)</div>
    </div>
  );
}

function EqTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tip">
      <div className="tip-t">{new Date(p.t).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</div>
      <div>Equity <b>{fmtMoney(p.equity)}</b></div>
      <div className="muted">Balance {fmtMoney(p.balance)}</div>
    </div>
  );
}

function SignalMeter({ value, threshold }) {
  const clamp = (x) => Math.max(-1, Math.min(1, x));
  const pos = (x) => `${((clamp(x) + 1) / 2) * 100}%`;
  const strong = Math.abs(value) >= threshold;
  const left = value >= 0 ? 50 : ((clamp(value) + 1) / 2) * 100;
  const width = Math.abs(clamp(value)) * 50;
  return (
    <div className="meter">
      <div className="meter-labels"><span className="neg">SELL</span><span className="muted">signal strength</span><span className="pos">BUY</span></div>
      <div className="meter-track">
        <div className="zone sell" style={{ width: pos(-threshold) }} />
        <div className="zone buy" style={{ left: pos(threshold), right: 0 }} />
        <div className={`meter-fill ${value >= 0 ? "pos" : "neg"} ${strong ? "strong" : ""}`}
             style={{ left: `${left}%`, width: `${width}%` }} />
        <div className="thr" style={{ left: pos(-threshold) }} />
        <div className="thr" style={{ left: pos(threshold) }} />
        <div className="mid" />
        <div className={`needle ${strong ? "glow" : ""}`} style={{ left: pos(value) }} />
      </div>
      <div className="meter-foot">
        <span>now <b className={value >= 0 ? "pos" : "neg"}>{value >= 0 ? "+" : ""}{value.toFixed(2)}</b></span>
        <span className="muted">trades past the gold lines (±{threshold.toFixed(2)})</span>
      </div>
    </div>
  );
}

function usePriceFlash(price) {
  const prev = useRef(price);
  const [flash, setFlash] = useState("");
  useEffect(() => {
    if (price != null && prev.current != null && price !== prev.current) {
      setFlash(price > prev.current ? "flash-up" : "flash-down");
      const id = setTimeout(() => setFlash(""), 900);
      prev.current = price;
      return () => clearTimeout(id);
    }
    prev.current = price;
  }, [price]);
  return flash;
}

function SymbolCard({ s }) {
  const flash = usePriceFlash(s.bid);
  const bars = (s.bars || []).map((v, i) => ({ i, v }));
  if (s.bid != null) bars.push({ i: bars.length, v: s.bid });
  const p = s.position;
  const up = bars.length > 1 && bars[bars.length - 1].v >= bars[0].v;
  const vals = bars.map((b) => b.v).concat(p ? [p.entry, p.sl, p.tp].filter(Boolean) : []);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  return (
    <div className={`card symbol ${p ? (p.side === "BUY" ? "in-long" : "in-short") : ""}`}>
      <div className="sym-head">
        <div>
          <div className="sym-name">{s.name}</div>
          <div className={`sym-price ${flash}`}>{fmtPx(s.bid)}</div>
          {s.chg_pct != null && <div className={`sym-chg ${s.chg_pct >= 0 ? "pos" : "neg"}`}>{s.chg_pct >= 0 ? "▲ +" : "▼ "}{s.chg_pct.toFixed(2)}% in 24h</div>}
          {s.mode === "news only" && s.grad && (
            <div className={`mk-grad ${s.grad.status}`}>news only · {s.grad.label}{["training", "exam_failed"].includes(s.grad.status) ? ` · attempt ${s.grad.attempts}/5` : ""}</div>
          )}
        </div>
        {p ? (
          <div className={`status ${p.side === "BUY" ? "long" : "short"}`}>
            {p.side === "BUY" ? <ArrowUpRight size={16} /> : <ArrowDownRight size={16} />}
            In a {p.side === "BUY" ? "buy" : "sell"} trade
            <span className={`status-pnl ${p.pnl >= 0 ? "pos" : "neg"}`}>{fmtMoney(p.pnl, true)}</span>
          </div>
        ) : s.pending ? (
          <div className="status pending"><Clock size={14} /> {s.pending.side === "BUY" ? "Buying" : "Selling"} on a dip · {Math.floor(s.pending.expires_in / 60)}:{String(s.pending.expires_in % 60).padStart(2, "0")}</div>
        ) : (
          <div className="status idle"><Pause size={14} /> Waiting</div>
        )}
      </div>
      <p className="why">{s.why}</p>
      {s.pending && (
        <div className="pending-box">
          <span>Target entry <b>${fmtPx(s.pending.limit)}</b></span>
          <span className="muted">price at signal ${fmtPx(s.pending.ref)} · now ${fmtPx(s.pending.side === "BUY" ? s.ask : s.bid)}</span>
        </div>
      )}
      {s.mode !== "news only" && <SignalMeter value={s.signal} threshold={s.threshold} />}
      {s.news && (
        <div className={`sym-news ${["prep", "armed", "in_trade"].includes(s.news.phase) ? "live" : ""}`}>
          <Newspaper size={13} /> news {until(s.news.t0)} · {s.news.plan ? PLAY[s.news.plan] : "stand aside"}
          {["prep", "armed", "in_trade"].includes(s.news.phase) && <b> · {s.news.phase.toUpperCase()}</b>}
        </div>
      )}
      <div className="spark">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={bars} margin={{ top: 4, right: 4, left: 4, bottom: 0 }}>
            <YAxis hide domain={[lo, hi]} />
            {p && <ReferenceLine y={p.entry} stroke="#facc15" strokeDasharray="3 3" label={{ value: "entry", fill: "#facc15", fontSize: 10, position: "insideLeft" }} />}
            {p && p.sl > 0 && <ReferenceLine y={p.sl} stroke="#f87171" strokeDasharray="3 3" label={{ value: "stop", fill: "#f87171", fontSize: 10, position: "insideBottomRight" }} />}
            {p && p.tp > 0 && <ReferenceLine y={p.tp} stroke="#34d399" strokeDasharray="3 3" label={{ value: "target", fill: "#34d399", fontSize: 10, position: "insideTopRight" }} />}
            <Line type="monotone" dataKey="v" stroke={up ? "#34d399" : "#f87171"} strokeWidth={1.8} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {s.mode !== "news only" && <div className="sym-foot">
        <span title="indicator votes across 5m, 15m, 1h and 4h"><ArrowUpRight size={13} className="pos" /> {s.up} up</span>
        <span><ArrowDownRight size={13} className="neg" /> {s.down} down</span>
        <span title="indicators with a poor recent record are ignored"><ShieldCheck size={13} /> {s.voters - s.ignored} of {s.voters} trusted</span>
        <span className="muted">last 12h</span>
      </div>}
    </div>
  );
}

function Learning({ d }) {
  const t = d.training || {};
  const pct = Math.round((t.progress || 0) * 100);
  return (
    <div className="card learning">
      <div className="card-title"><Brain size={16} /> Is it getting smarter?</div>
      <div className="learned">
        {d.symbols.map((s) => (
          <div key={s.symbol} className="learned-row">
            <div className="learned-sym">{s.name}</div>
            <div>
              {s.trusts.length > 0 && <div><span className="tag pos">trusts</span> {s.trusts.join(" · ")}</div>}
              {s.ignores.length > 0 && <div><span className="tag neg">ignores</span> {s.ignores.join(" · ")}</div>}
              {!s.trusts.length && !s.ignores.length && <div className="muted">{s.mode === "news only"
                ? "trades news only until it passes its exam; its indicator memory starts then" : "building its track record…"}</div>}
            </div>
          </div>
        ))}
        <div className="muted small">Every 5-minute bar, each indicator is scored on whether it was right. Proven ones get more say; wrong ones get ignored.</div>
      </div>
      <div className="divider" />
      <div className="train-head">
        <span><GraduationCap size={15} /> Brain training</span>
        <span className="muted">{t.running ? `whole cycle ${pct}%${t.stage?.cycle_eta_s ? ` · ~${fmtDur(t.stage.cycle_eta_s)} left` : ""}` : "paused"}</span>
      </div>
      <div className="progress thin"><div className="progress-fill" style={{ width: `${pct}%` }} /></div>
      {!t.running && t.note && <div className="train-note">{t.note}</div>}
      {t.current && t.stage?.stage_progress != null && (
        <div className="stage-now">
          <div className="stage-now-head">
            <b>{t.current.name} · {t.current.symbol}</b>
            <span className="stage-pct">{Math.round(t.stage.stage_progress * 100)}%</span>
          </div>
          <div className="progress big"><div className="progress-fill" style={{ width: `${t.stage.stage_progress * 100}%` }} /></div>
          <div className="muted small">
            {t.stage.step?.toLocaleString()} / {t.stage.total?.toLocaleString()} {t.stage.unit}
            {t.stage.stage_eta_s ? ` · ~${fmtDur(t.stage.stage_eta_s)} left` : ""}
            {t.stage.phase === "evaluating" ? " · running a quick test on validation data…" : t.stage.phase === "finishing" ? " · target reached, saving and testing the final model…" : ""}
          </div>
        </div>
      )}
      <div className="stages">
        {(t.stages || []).map((st, i) => <span key={i} className={`stage ${st.status}`}>{st.label}</span>)}
      </div>
      {t.current ? (
        <div className="mission">
          <div className="mission-title">Now: {t.current.name} · {t.current.symbol}</div>
          <div className="mrow"><span className="mlabel">Learning</span>{t.current.learn}</div>
          <div className="mrow"><span className="mlabel">Goal</span>{t.current.goal}</div>
          <div className="mrow"><span className="mlabel">Tested by</span>{t.current.test}</div>
          {t.next && <div className="muted small">Next up: {t.next}. New models trade only after passing an exam on data they've never seen.</div>}
        </div>
      ) : (
        !t.note && <div className="muted small">No training running. Run <code>exness train</code> to start a learning cycle.</div>
      )}
    </div>
  );
}

const FEED_ICON = { ppo: Cpu, eval: FlaskConical, lstm: Brain, dreamer: Moon, memory: Database,
  signal: ArrowLeftRight, trade: Receipt };
const ago = (t) => {
  const s = Math.max(0, Date.now() / 1000 - t);
  return s < 60 ? "just now" : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${Math.floor(s / 3600)}h ago`;
};

function Feed({ items }) {
  const [filter, setFilter] = useState("all");
  const groups = { all: null, training: ["ppo", "eval", "lstm", "dreamer"], learning: ["memory"], trading: ["signal", "trade"] };
  const shown = (items || []).filter((it) => !groups[filter] || groups[filter].includes(it.kind));
  return (
    <div className="card feed">
      <div className="card-head">
        <div className="card-title"><Sparkles size={16} /> What changed, and why</div>
        <div className="tabs">
          {Object.keys(groups).map((k) => (
            <button key={k} className={k === filter ? "active" : ""} onClick={() => setFilter(k)}>{k}</button>
          ))}
        </div>
      </div>
      <div className="feed-list">
        {!shown.length && <div className="muted small">Updates appear here as training rounds finish, the memory re-scores indicators, signals change and trades close.</div>}
        {shown.map((it) => {
          const Icon = FEED_ICON[it.kind] || Sparkles;
          return (
            <div key={`${it.t}-${it.title}`} className={`feed-item ${it.tone}`}>
              <div className="feed-ic"><Icon size={15} /></div>
              <div className="feed-body">
                <div className="feed-head"><b>{it.title}</b><span className="muted small">{ago(it.t)}</span></div>
                <div className="feed-text">{it.text}</div>
                {it.numbers?.filter((n) => n.from != null).map((n, i) => (
                  <span key={i} className={`delta ${n.to >= n.from ? "pos" : "neg"}`}>
                    {n.label} {n.from} → {n.to} ({n.to - n.from >= 0 ? "+" : ""}{(n.to - n.from).toFixed(2)})
                  </span>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const NAME = { BTCUSDm: "Bitcoin", XAUUSDm: "Gold", EURUSDm: "EUR/USD", USDJPYm: "USD/JPY", XAGUSDm: "Silver",
  ETHUSDm: "Ether", USOILm: "Oil" };
const PHASE = { scheduled: "scheduled", prep: "PREP · range marked, normal trades paused", armed: "ARMED · watching for the sweep",
  in_trade: "IN TRADE", review: "reviewing result", done: "done" };
const PLAY = { fade: "Fade the sweep", breakout: "Ride the breakout" };
const until = (t) => {
  const s = t - Date.now() / 1000;
  if (s <= 0) return `${Math.round(-s / 60)} min ago`;
  return s < 3600 ? `in ${Math.round(s / 60)} min` : `in ${(s / 3600).toFixed(1)} h`;
};

function Markets({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="card markets">
      <div className="card-title"><TrendingUp size={16} /> All markets</div>
      <div className="mk-grid">
        {rows.map((r) => {
          const live = r.news && ["prep", "armed", "in_trade"].includes(r.news.phase);
          return (
            <div key={r.symbol} className={`mk ${live ? "live" : ""}`}>
              <div className="mk-top"><b>{r.name}</b><span className="muted small">{r.mode}</span></div>
              {r.grad && r.mode === "news only" && (
                <div className={`mk-grad ${r.grad.status}`} title={r.grad.why}>
                  {r.grad.label}{["training", "exam_failed"].includes(r.grad.status) ? ` · attempt ${r.grad.attempts}/5` : ""}
                </div>
              )}
              <div className="mk-px">{r.bid == null ? "—" : r.bid.toLocaleString(undefined, { maximumFractionDigits: r.bid < 10 ? 5 : r.bid < 1000 ? 3 : 2 })}</div>
              {r.chg_pct != null && <div className={r.chg_pct >= 0 ? "pos small" : "neg small"}>{r.chg_pct >= 0 ? "▲" : "▼"} {r.chg_pct.toFixed(2)}% 24h</div>}
              <div className="small">{r.lots ? <span className="mk-pos">{r.lots > 0 ? "LONG" : "SHORT"} {Math.abs(r.lots).toFixed(2)}
                {r.pnl != null && <span className={r.pnl >= 0 ? "pos" : "neg"}> {r.pnl >= 0 ? "+" : "−"}${Math.abs(r.pnl).toFixed(2)}</span>}</span>
                : <span className="muted">no position</span>}</div>
              {r.news && <div className="mk-news small">news {until(r.news.t0)} · {r.news.plan ? PLAY[r.news.plan] : "stand aside"}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function News({ news }) {
  const [tab, setTab] = useState("plan");
  if (!news) return null;
  const plan = news.plan || [];
  return (
    <div className="card news">
      <div className="card-head">
        <div className="card-title"><Newspaper size={16} /> News &amp; markets {news.enabled ? <span className="pill on">news trading on</span> : <span className="pill">off</span>}</div>
        <div className="tabs">
          {[["plan", "Next releases"], ["moved", "What moved markets"], ["record", "Track record"]].map(([k, l]) => (
            <button key={k} className={k === tab ? "active" : ""} onClick={() => setTab(k)}>{l}</button>
          ))}
        </div>
      </div>
      {tab === "plan" && (
        <div className="news-list">
          <div className="muted small">Before each big US release the bot marks the 30-min range (stops pile up just beyond it),
            pauses its normal trades, then after the release trades the play that has worked before for that market.</div>
          {!plan.length && <div className="muted small">No high-impact US release in the next 36 h (or the calendar is still loading).</div>}
          {plan.map((e, i) => (
            <div key={i} className={`news-item ${e.phase}`}>
              <CalendarClock size={15} />
              <div className="feed-body">
                <div className="feed-head"><b>{NAME[e.sym] || e.sym} · {e.title}</b><span className="muted small">{until(e.t0)}</span></div>
                <div className="feed-text"><span className="tag">{e.topic}</span> <b>{e.plan ? PLAY[e.plan] : "Stand aside"}</b> — {e.why}</div>
                <div className="muted small">{PHASE[e.phase] || e.phase}
                  {e.range && ` · range ${e.range[1].toFixed(2)}–${e.range[0].toFixed(2)}`}
                  {e.result && ` · paper result: ${Object.entries(e.result).map(([p, r]) => `${p} ${r == null ? "no trigger" : (r >= 0 ? "+" : "") + r + "R"}`).join(", ")}`}</div>
              </div>
            </div>
          ))}
          {(news.upcoming || []).filter((u) => !plan.some((p) => p.title === u.title)).slice(0, 6).map((u, i) => (
            <div key={`u${i}`} className="news-item dim">
              <CalendarClock size={15} />
              <div className="feed-body">
                <div className="feed-head"><span>{u.title} <span className="muted small">({u.impact})</span></span><span className="muted small">{until(u.t)}</span></div>
                <div className="muted small">Watch: {u.watch.join(", ") || "—"} · {u.mechanism}</div>
              </div>
            </div>
          ))}
        </div>
      )}
      {tab === "moved" && (
        <div className="news-list">
          {!(news.recent || []).length && <div className="muted small">Collecting headlines and measuring reactions (each item is checked 15 min after it lands)…</div>}
          {(news.recent || []).map((it) => (
            <div key={it.id} className={`news-item ${it.max_size >= 2 ? "hot" : ""}`}>
              <Newspaper size={15} />
              <div className="feed-body">
                <div className="feed-head"><b>{it.title}</b><span className="muted small">{it.source} · {ago(it.t)}</span></div>
                <div className="muted small"><span className="tag">{it.topic}</span> {it.mechanism}</div>
                <div className="chips">
                  {(it.chips || []).map((c) => (
                    <span key={c.symbol} className={`delta ${c.m15 >= 0 ? "pos" : "neg"} ${c.size >= 2 ? "big" : ""}`}>
                      {c.label} {c.m15 >= 0 ? "+" : ""}{c.m15.toFixed(2)}% · {c.size}×
                    </span>
                  ))}
                </div>
                {it.explain && <div className="feed-text">{it.explain}</div>}
              </div>
            </div>
          ))}
        </div>
      )}
      {tab === "record" && (
        <div className="news-list">
          <div className="muted small">Paper results of both plays at every release (R = multiples of the amount risked, after costs).
            A play is traded once it has 8+ results averaging at least +0.10R.</div>
          <table className="tbl">
            <thead><tr><th>Market</th><th>News type</th><th>Play</th><th>Trades</th><th>Win</th><th>Avg</th><th>Total</th></tr></thead>
            <tbody>
              {(news.playbook || []).map((r, i) => (
                <tr key={i}>
                  <td>{NAME[r.symbol] || r.symbol}</td><td>{r.topic === "us_release" ? "all US releases" : r.topic}</td>
                  <td>{PLAY[r.play] || r.play}</td><td>{r.n}/{r.events}</td><td>{Math.round(r.win * 100)}%</td>
                  <td className={r.avg_r >= 0 ? "pos" : "neg"}>{r.avg_r >= 0 ? "+" : ""}{r.avg_r.toFixed(2)}R</td>
                  <td className={r.total_r >= 0 ? "pos" : "neg"}>{r.total_r >= 0 ? "+" : ""}{r.total_r.toFixed(1)}R</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Trades({ trades }) {
  return (
    <div className="card trades">
      <div className="card-title"><Receipt size={16} /> Today's trades</div>
      {!trades.length && <div className="muted small">No trades yet today.</div>}
      <div className="trade-list">
        {trades.map((t, i) => (
          <div key={i} className="trade">
            <div className={`trade-ic ${t.closing ? (t.pnl > 0 ? "win" : "loss") : t.side === "BUY" ? "buy" : "sell"}`}>
              {t.closing ? (t.pnl > 0 ? <Trophy size={14} /> : <Target size={14} />) : t.side === "BUY" ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
            </div>
            <div className="trade-body">
              <div>{t.closing ? `Closed ${t.symbol}` : `${t.side === "BUY" ? "Bought" : "Sold"} ${t.volume.toFixed(2)} ${t.symbol}`}</div>
              <div className="muted small">{fmtTime(t.t)} · ${fmtPx(t.price)}</div>
            </div>
            {t.closing && <div className={`trade-pnl ${t.pnl > 0 ? "pos" : "neg"}`}>{fmtMoney(t.pnl, true)}</div>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [d, err] = useApi(2000);
  if (!d) {
    return (
      <div className="app"><Header d={null} />
        <div className="loading"><Activity className="spin" /> {err || "Connecting to the bot…"}</div>
      </div>
    );
  }
  const a = d.account, st = d.stats;
  const winRate = st.closed ? Math.round((st.wins / st.closed) * 100) : null;
  const lossUsed = a.daily_loss_limit ? Math.max(0, Math.min(1, -a.today / a.daily_loss_limit)) : 0;
  const open = d.symbols.reduce((n, s) => n + (s.trades || (s.position ? [1] : [])).length, 0);
  return (
    <div className="app">
      <Header d={d} />
      {err && <div className="banner">{err} · showing last known data</div>}
      <section className="kpis">
        <Kpi icon={Wallet} label="Equity" sub={`balance ${fmtMoney(a.balance)}`}>
          <AnimatedNumber value={a.equity} format={(v) => fmtMoney(v)} />
        </Kpi>
        <Kpi icon={CircleDollarSign} label="Today" tone={a.today >= 0 ? "pos" : "neg"}
             sub={<span>{a.today_pct >= 0 ? "+" : ""}{a.today_pct.toFixed(2)}% · loss limit used {Math.round(lossUsed * 100)}%
               <span className="mini-bar"><span style={{ width: `${lossUsed * 100}%` }} /></span></span>}>
          <AnimatedNumber value={a.today} format={(v) => fmtMoney(v, true)} />
        </Kpi>
        <Kpi icon={Trophy} label="Win rate today" sub={`${st.wins} won · ${st.losses} lost · ${fmtMoney(st.pnl_closed, true)}`}>
          {winRate == null ? "—" : `${winRate}%`}
        </Kpi>
        <Kpi icon={Zap} label="Open trades" sub={`floating ${fmtMoney(a.floating, true)}`}>{open}</Kpi>
      </section>
      <RiskStrip d={d} />
      <section className="war-grid">{d.symbols.map((s) => <WarCard key={s.symbol} s={s} />)}</section>
      <MissionControl rows={d.mission} training={d.training} />
      <section className="bottom">
        <EquityChart series={d.equity} equity={a.equity} />
        <Feed items={d.feed} />
      </section>
      <section className="bottom2">
        <News news={d.news} />
        <Trades trades={d.trades} />
      </section>
      <section className="main-full">
        <Learning d={d} />
      </section>
      <footer className="footer">
        <Lightbulb size={14} /> <b>{d.tip.term}:</b> {d.tip.text}
        <span className="muted right"><Clock size={12} /> updated {new Date(d.ts * 1000).toLocaleTimeString()}</span>
      </footer>
    </div>
  );
}
