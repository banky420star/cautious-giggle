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
const folded = {
  get: (id) => { try { return localStorage.getItem(`cg_fold_${id}`) === "1"; } catch { return false; } },
  set: (id, v) => { try { localStorage.setItem(`cg_fold_${id}`, v ? "1" : "0"); } catch { /* private mode */ } },
};

/* slides in the first time it scrolls into view */
function useReveal() {
  const ref = useRef(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return undefined;
    if (typeof IntersectionObserver === "undefined") { setSeen(true); return undefined; }
    const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { setSeen(true); io.disconnect(); } }, { rootMargin: "0px 0px -8% 0px" });
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);
  return [ref, seen];
}

export function Section({ id, no, title, sub, right, children }) {
  const [ref, seen] = useReveal();
  const [closed, setClosed] = useState(() => folded.get(id));
  const toggle = () => { setClosed((c) => { folded.set(id, !c); return !c; }); };
  return (
    <section id={id} ref={ref} className={`sec reveal ${seen ? "in" : ""} ${closed ? "closed" : ""}`}>
      <button type="button" className="sec-head" onClick={toggle} aria-expanded={!closed} aria-controls={`${id}-body`}
              title={closed ? "Show this section" : "Hide this section"}>
        <span className="sec-no">{no}</span>
        <h2>{title}</h2>
        {sub && <span className="ex sec-sub">{sub}</span>}
        {right && <span className="sec-right">{right}</span>}
        <span className="sec-chev" aria-hidden>▾</span>
      </button>
      <div className="sec-body" id={`${id}-body`}><div className="sec-inner">{children}</div></div>
    </section>
  );
}

/* sticky menu: jump to a section; the one on screen lights up */
export function SectionNav({ items }) {
  const [active, setActive] = useState(items[0]?.id);
  const ids = items.map((i) => i.id).join();
  useEffect(() => {
    // the section on screen = the last one whose top has passed the bottom of the sticky menu
    let raf = 0;
    const pick = () => {
      raf = 0;
      const nav = document.querySelector(".secnav");
      const line = (nav ? nav.getBoundingClientRect().bottom : 130) + 8;
      let cur = items[0]?.id;
      for (const { id } of items) {
        const el = document.getElementById(id);
        if (el && el.getBoundingClientRect().top <= line) cur = id;
      }
      if (window.innerHeight + window.scrollY >= document.body.scrollHeight - 4) cur = items[items.length - 1]?.id;
      setActive(cur);
    };
    const onScroll = () => { if (!raf) raf = requestAnimationFrame(pick); };
    pick();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => { window.removeEventListener("scroll", onScroll); window.removeEventListener("resize", onScroll); cancelAnimationFrame(raf); };
  }, [ids]);
  const go = (id) => {
    const el = document.getElementById(id);
    if (!el) return;
    if (el.classList.contains("closed")) el.querySelector(".sec-head")?.click();
    el.scrollIntoView({ behavior: "smooth", block: "start" });
    setActive(id);
  };
  return (
    <nav className="secnav" aria-label="Sections">
      {items.map(({ id, label, Icon, badge, tone }) => (
        <button key={id} type="button" className={`secnav-i ${active === id ? "on" : ""}`} onClick={() => go(id)}>
          {Icon && <Icon size={14} />}<span>{label}</span>
          {badge != null && badge !== "" && <b key={String(badge)} className={`secnav-b ${tone || ""}`}>{badge}</b>}
        </button>
      ))}
    </nav>
  );
}

export function BackToTop() {
  const [show, setShow] = useState(false);
  useEffect(() => {
    const f = () => setShow(window.scrollY > 600);
    f();
    window.addEventListener("scroll", f, { passive: true });
    return () => window.removeEventListener("scroll", f);
  }, []);
  return (
    <button type="button" className={`totop ${show ? "show" : ""}`} onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
            aria-label="Back to top" title="Back to top">↑</button>
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
  const a = d.account;
  return (
    <div className="tiles tiles-2">
      <Tile label="EQUITY" value={<Animated value={a.equity} format={(v) => money(v, false)} />}
            sub={`balance ${money(a.balance, false)} · open trades ${money(a.floating)}`}
            hint="What the account is worth right now, including trades that are still open. Balance counts closed trades only." />
      <Tile label="TODAY" tone={a.today >= 0 ? "pos" : "neg"} value={<Animated value={a.today} format={(v) => money(v)} />}
            sub={`${a.today_pct >= 0 ? "+" : ""}${a.today_pct.toFixed(2)}% since midnight`}
            hint="Profit or loss since the start of the day (broker time), closed and open trades together." />
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
/* ================================================================ Kronos: the next 30 minutes, candle by candle */
const pxf = (v) => (v == null ? "—" : v >= 1000 ? v.toFixed(2) : v >= 10 ? v.toFixed(3) : v.toFixed(5));
const sbps = (v) => (v == null ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(1)} bps`);

function ForecastChart({ k }) {
  const real = k.recent || [], fut = k.candles || [];
  const all = real.concat(fut);
  if (!all.length) return null;
  const W = 320, H = 130, padL = 4, padR = 54, top = 8, bot = 14;
  const lo = Math.min(...all.map((c) => c.l)), hi = Math.max(...all.map((c) => c.h));
  const y = (v) => top + (hi === lo ? 0.5 : (hi - v) / (hi - lo)) * (H - top - bot);
  const step = (W - padL - padR) / all.length, bw = Math.max(2, step * 0.62);
  const x = (i) => padL + step * i + step / 2;
  const fx0 = padL + step * real.length;
  const line = (v, cls, label) => (
    <g className={`fc-line ${cls}`}>
      <line x1={padL} x2={W - padR} y1={y(v)} y2={y(v)} />
      <text x={W - padR + 4} y={y(v) + 3}>{label}</text>
    </g>
  );
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="fc-chart" role="img" aria-label="Recent candles and Kronos's forecast of the next ones">
      <rect x={fx0} y={0} width={W - padR - fx0} height={H - bot + 2} className="fc-zone" />
      <text x={fx0 + 3} y={H - 3} className="fc-lab">Kronos forecast →</text>
      <text x={padL} y={H - 3} className="fc-lab">last 2 h</text>
      {all.map((c, i) => {
        const f = i >= real.length, up = c.c >= c.o;
        return (
          <g key={i} className={`fc-c ${up ? "up" : "dn"} ${f ? "fut" : ""}`} style={f ? { animationDelay: `${(i - real.length) * 0.12}s` } : undefined}>
            <line x1={x(i)} x2={x(i)} y1={y(c.h)} y2={y(c.l)} />
            <rect x={x(i) - bw / 2} y={y(Math.max(c.o, c.c))} width={bw} height={Math.max(1, Math.abs(y(c.o) - y(c.c)))} rx="1" />
          </g>
        );
      })}
      {line(k.last, "now", `now ${pxf(k.last)}`)}
      {line(k.buy.best_entry, "buy", `buy ${pxf(k.buy.best_entry)}`)}
      {line(k.sell.best_entry, "sell", `sell ${pxf(k.sell.best_entry)}`)}
    </svg>
  );
}

function KronosForecast({ k, record }) {
  if (!k) return <div className="fc card-lite muted small">Kronos is warming up: its first forecast for this market comes at the next 5-minute candle.</div>;
  const lean = k.move_bps > 0 ? "up" : k.move_bps < 0 ? "down" : "flat";
  const plan = (side, p) => (
    <div className={`fc-plan ${side.toLowerCase()}`}>
      <div className="fc-plan-h">If you {side} now</div>
      <div><span className="muted">first goes against you</span> <b className="neg">{sbps(p.drawdown_bps)}</b></div>
      <div><span className="muted">best entry</span> <b>{pxf(p.best_entry)}</b> <span className="muted">in ~{p.best_entry_in_min} min</span></div>
      <div><span className="muted">likely best exit</span> <b>{pxf(p.target)}</b> <b className={p.upside_bps >= 0 ? "pos" : "neg"}>{sbps(p.upside_bps)}</b></div>
    </div>
  );
  return (
    <div className="fc card-lite">
      <div className="fc-head">
        <b>Kronos · next {k.horizon_min} min</b>
        <span className={`fc-lean ${lean}`}>{lean === "up" ? "▲" : lean === "down" ? "▼" : "•"} {sbps(k.move_bps)}</span>
        <span className="muted small">normal move ±{k.typical_bps} bps</span>
      </div>
      <ForecastChart k={k} />
      <div className="fc-plans">{plan("BUY", k.buy)}{plan("SELL", k.sell)}</div>
      <div className="ex small muted">
        Kronos draws one likely path for the next {k.candles?.length || 6} candles. "Goes against you" is the worst dip (buy) or spike (sell)
        on that path before it turns; "best entry" is where that dip or spike bottoms out.
        {record && <> Its live record after costs here: <b className={(record.net_bps || 0) > 0 ? "pos" : "neg"}>{sbps(record.net_bps)}</b> per call over {record.n} calls ({record.verdict}). Treat the levels as a guess, not a promise.</>}
      </div>
    </div>
  );
}

/* ================================================================ who voted, and why */
const REC = { works: ["works", "green"], "doesn't work": ["ignored", "red"], neutral: ["neutral", "grey"], learning: ["learning", "amber"] };
const TFS = ["M5", "M15", "H1", "H4"];
const TFNAME = { M5: "5 min", M15: "15 min", H1: "1 hour", H4: "4 hours" };

/* a ruler for one reading: red = where it votes sell, green = where it votes buy, the pin = where it is now */
function Meter({ m }) {
  if (!m) return null;
  const W = 260, H = 34, L = 6, R = W - 6;
  const sx = (v) => L + ((Math.max(m.min, Math.min(m.max, v)) - m.min) / (m.max - m.min)) * (R - L);
  const zone = (z, cls) => z && <rect x={sx(z[0])} width={Math.max(1, sx(z[1]) - sx(z[0]))} y={10} height={10} className={`mz ${cls}`} />;
  const f = (v) => Number(v).toFixed(m.fmt ?? 2);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="meter" role="img" aria-label={`reading ${m.short}`}>
      <rect x={L} width={R - L} y={10} height={10} rx="3" className="mz track" />
      {zone(m.sell, "sell")}{zone(m.buy, "buy")}
      <text x={L} y={32} className="m-tick">{f(m.min)}</text>
      <text x={R} y={32} className="m-tick" textAnchor="end">{f(m.max)}</text>
      <g className="m-pin" style={{ transform: `translateX(${sx(m.v)}px)` }}>
        <line x1={0} x2={0} y1={4} y2={24} /><circle cx={0} cy={4} r={3} />
      </g>
      <text x={sx(m.v)} y={32} className="m-val" textAnchor="middle">{m.short}</text>
    </svg>
  );
}

function VoterDetail({ r }) {
  if (!r) return null;
  const [rl, rc] = REC[r.record?.verdict] || ["—", "grey"];
  return (
    <div className={`vdetail ${r.ignored ? "ign" : ""}`} key={r.key}>
      <div className="vd-top">
        <span className={`bal-side ${r.side === "BUY" ? "buy" : r.side === "SELL" ? "sell" : "none"}`}>{r.side}</span>
        <b>{r.name}</b>
        <span className="bal-w" title={`counts ${r.base}× an indicator, × ${r.mult} for its record`}>{r.ignored ? "ignored" : `weight ×${r.weight}`}</span>
        <span className={`op-tag ${rc}`}>{rl}</span>
      </div>
      <Meter m={r.meter} />
      <div className="vd-why">
        {r.ignored && <b className="neg">Not counted: its record here loses {sbps(r.record?.net_bps)} per call after the spread. </b>}
        {r.reason}
        {r.levels && <> Entry {pxf(r.levels.entry)} · stop {pxf(r.levels.stop)} · target {pxf(r.levels.target)} ({r.levels.state}).</>}
      </div>
      {r.headlines?.length > 0 && <ul className="bal-heads">{r.headlines.map((h, i) => <li key={i}>{h}</li>)}</ul>}
      <div className="muted small">Record here after costs: {sbps(r.record?.net_bps)} per call over {r.record?.n ?? 0} calls
        {r.record?.hit != null && <> · right {Math.round(r.record.hit * 100)}% of the time</>}.</div>
    </div>
  );
}

function cellStyle(r) {
  if (!r || r.ignored || r.side === "no view") return undefined;
  const a = 0.12 + 0.55 * Math.min(1, Math.abs(r.vote));
  return { background: r.vote > 0 ? `rgba(52,211,153,${a})` : `rgba(248,113,113,${a})` };
}

function Ballot({ rows }) {
  const [sel, setSel] = useState(null);
  if (!rows?.length) return null;
  const models = rows.filter((r) => r.model);
  const ind = rows.filter((r) => !r.model);
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]));
  const strategies = [...new Set(ind.map((r) => r.strategy))];
  const label = (st) => (ind.find((r) => r.strategy === st)?.name || st).replace(/ \((M5|M15|H1|H4)\)$/, "");
  const up = rows.filter((r) => r.side === "BUY" && !r.ignored).length;
  const dn = rows.filter((r) => r.side === "SELL" && !r.ignored).length;
  const cur = byKey[sel] || rows[0];
  return (
    <div className="ballot card-lite">
      <div className="ballot-head">
        <b>Who voted and why</b>
        <span className="muted small"><span className="pos">{up} buy</span> · <span className="neg">{dn} sell</span> · {rows.length - up - dn} no view or ignored. Click any voter to see its reading and reason.</span>
      </div>
      {models.length > 0 && (
        <div className="vmodels">
          {models.map((r) => (
            <button type="button" key={r.key} className={`vmodel ${cur?.key === r.key ? "on" : ""} ${r.ignored ? "ign" : ""}`} onClick={() => setSel(r.key)} style={cellStyle(r)}>
              <span className="vm-tag">YOUR MODEL · ×{r.base}</span>
              <b>{r.name.split(" (")[0]}</b>
              <span className="vm-side">{r.side} {r.meter?.short}</span>
            </button>
          ))}
        </div>
      )}
      <div className="vmatrix" role="grid">
        <div className="vm-row vm-headrow"><span />{TFS.map((t) => <span key={t} className="vm-tf">{TFNAME[t]}</span>)}</div>
        {strategies.map((st) => (
          <div className="vm-row" key={st}>
            <span className="vm-name" title={label(st)}>{label(st)}</span>
            {TFS.map((tf) => {
              const r = byKey[tf === "M5" ? st : `${st}@${tf}`];
              return r ? (
                <button type="button" key={tf} className={`vm-cell ${r.ignored ? "ign" : ""} ${cur?.key === r.key ? "on" : ""}`}
                        style={cellStyle(r)} onClick={() => setSel(r.key)} title={r.reason}>
                  <span className="vm-arrow">{r.side === "BUY" ? "▲" : r.side === "SELL" ? "▼" : "·"}</span>{r.meter?.short}
                </button>
              ) : <span key={tf} className="vm-cell empty">—</span>;
            })}
          </div>
        ))}
      </div>
      <div className="vm-legend muted small"><i className="lg-buy" /> votes buy <i className="lg-sell" /> votes sell (stronger colour = stronger vote) <i className="lg-ign" /> ignored for a losing record</div>
      <VoterDetail r={cur} />
    </div>
  );
}

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

      <KronosForecast k={s.kronos} record={(s.ballot || []).find((r) => r.key === "kronos")?.record} />
      <Ballot rows={s.ballot} />

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
  blocked: ["BLOCKED", "red", ShieldAlert], close: ["CLOSE", "grey", Target], reduce: ["TRIM", "grey", Scissors], manage: ["STOP", "violet", ShieldAlert],
  size: ["SIZE", "grey", Ruler], warn: ["ALERT", "red", Wifi], news: ["NEWS", "amber", Newspaper],
  signal: ["SIGNAL", "cyan", ArrowLeftRight], gate: ["HELD", "red", ShieldAlert], skip: ["SKIP", "grey", Clock],
};

const VERDICT = {
  full: ["PROVEN", "green", "normal size"], probe: ["LEARNING", "amber", "smallest lot"],
  retest: ["RE-TEST", "violet", "one smallest-lot try"], block: ["BLOCKED", "red", "no trades"],
};

const VERDICT_TAG = { works: ["WORKS", "green"], "doesn't work": ["IGNORED", "red"], neutral: ["NEUTRAL", "grey"], learning: ["LEARNING", "amber"] };
const pct = (x) => (x == null ? "—" : `${Math.round(x * 100)}%`);
const bps = (x) => (x == null ? "—" : `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(2)}`);

function VoteBar({ v }) {
  if (v == null) return <span className="muted small">waiting</span>;
  const w = Math.min(50, Math.abs(v) * 50);
  return (
    <span className="vbar" title={`${v >= 0 ? "+" : ""}${v.toFixed(2)} (−1 strongly down … +1 strongly up)`}>
      <i className={v >= 0 ? "up" : "dn"} style={v >= 0 ? { left: "50%", width: `${w}%` } : { left: `${50 - w}%`, width: `${w}%` }} />
      <b>{v >= 0 ? "+" : ""}{v.toFixed(2)}</b>
    </span>
  );
}

function testCell(t, spread) {
  if (!t || !t.active) return <span className="muted small">{t ? "no calls" : "—"}</span>;
  const good = t.edge_bps > 0;
  return (
    <span title={`${t.active} calls replayed · each trade on it averaged ${bps(t.follow_bps)} bps after the spread${spread != null ? ` (spread ${spread.toFixed(2)} bps)` : ""}`}>
      <b className={good ? "pos" : "neg"}>{pct(t.hit)}</b> <span className="muted small">edge {bps(t.edge_bps)}</span>
    </span>
  );
}

export function Models({ data }) {
  if (!data || !data.voters?.length) return <div className="card hud"><div className="muted small">{data?.error || "Your models are not loaded."}</div></div>;
  const tst = data.test || {};
  const cmp = Object.entries(tst.compare || {});
  return (
    <div className="models">
      <div className="card hud models-head">
        <span className={`dot ${data.live ? "on" : "off"}`} /> <b>{data.live ? "Voting live" : data.updated ? `Not voting: last vote ${ago(data.updated)} (bot stopped or broker connection lost)` : "Not voting (bot not running)"}</b>
        <span className="muted small">
          Each model gets a weight (an indicator counts 1). Its live record decides how much it counts: models that keep being right count up to 1.5× more, wrong ones are ignored.
          {" "}History test: {tst.running ? "running…" : tst.done ? `last ${tst.days} days, scored after costs over the next ${tst.horizon_min} min` : "not run yet"}. All figures are after the spread: above 0 means it paid for itself.
        </span>
      </div>
      <div className="models-grid">
        {data.voters.map((m) => (
          <div key={m.key} className="card hud model-card">
            <div className="card-head"><b>{m.name}</b><span className="op-tag cyan">weight {m.weight}×</span></div>
            <table className="blt small-t">
              <thead><tr><th>Market</th><th>Now</th><th className="r">Live hit</th><th className="r">Live edge (net)</th><th>Verdict</th><th>History test</th></tr></thead>
              <tbody>
                {m.rows.map((r) => {
                  const [lab, col] = VERDICT_TAG[r.live?.verdict] || ["—", "grey"];
                  const spread = tst.compare?.[r.symbol]?.spread_bps;
                  return (
                    <tr key={r.symbol} title={r.error || ""}>
                      <td><b>{r.name}</b>{r.error && <span className="neg small"> ⚠</span>}</td>
                      <td><VoteBar v={r.vote} /></td>
                      <td className="r">{pct(r.live?.hit)}<span className="muted small"> /{Math.round(r.live?.n || 0)}</span></td>
                      <td className={`r ${(r.live?.edge_bps || 0) >= 0 ? "pos" : "neg"}`}>{bps(r.live?.edge_bps)}</td>
                      <td><span className={`op-tag ${col}`}>{lab}</span></td>
                      <td>{testCell(r.test, spread)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      {cmp.length > 0 && (
        <div className="card hud">
          <div className="card-head"><b>Does adding your models help?</b><span className="muted small">same bars, indicators alone vs indicators + your models</span></div>
          <table className="blt">
            <thead><tr><th>Market</th><th className="r">Indicators only</th><th className="r">+ your models</th><th className="r">Spread (cost)</th><th>Result</th></tr></thead>
            <tbody>
              {cmp.map(([s, c]) => {
                const b = c.base?.edge_bps, w = c.with?.edge_bps;
                const res = b == null || w == null ? ["—", "grey"] : w > b + 0.2 ? ["HELPS", "green"] : w < b - 0.2 ? ["HURTS", "red"] : ["NO CHANGE", "grey"];
                return (
                  <tr key={s}>
                    <td><b>{c.name}</b></td>
                    <td className="r">{pct(c.base?.hit)} <span className="muted small">edge {bps(b)}</span></td>
                    <td className="r">{pct(c.with?.hit)} <span className="muted small">edge {bps(w)}</span></td>
                    <td className="r muted">{c.spread_bps == null ? "—" : `${c.spread_bps.toFixed(2)} bps`}</td>
                    <td><span className={`op-tag ${res[1]}`}>{res[0]}</span></td>
                  </tr>
                );
              })}
            </tbody>
            <tfoot><tr><td colSpan={5} className="muted small ex">Hit = share of calls that made money after the spread. Edge = average of (vote × next move − spread paid) in basis points (1 bp = 0.01%), so above 0 means profit after costs.</td></tr></tfoot>
          </table>
        </div>
      )}
    </div>
  );
}

export function Setups({ data }) {
  const [all, setAll] = useState(false);
  const rows = data?.rows || [];
  const c = data?.counts || {};
  if (!data?.enabled) return <div className="card hud"><div className="muted small">Setup gate is off.</div></div>;
  const shown = all ? rows : rows.slice(0, 12);
  return (
    <div className="card hud setups">
      <div className="setup-sum">
        {Object.entries(VERDICT).map(([k, [lab, col, what]]) => (
          <span key={k} className={`op-tag ${col}`} title={what}>{lab} {c[k] || 0}</span>
        ))}
        <span className="muted small">A setup = market · side · session · confidence · what the AI said. Proven after {data.min_full} profitable trades; blocked after {data.min_block} if it loses.</span>
      </div>
      {!rows.length ? <div className="muted small">No closed trades scored yet.</div> : (
        <table className="blt">
          <thead><tr><th>Setup</th><th className="r">Trades</th><th className="r">Won</th><th className="r">Net</th><th>Status</th><th>Why</th></tr></thead>
          <tbody>
            {shown.map((r) => {
              const [lab, col, what] = VERDICT[r.verdict] || [r.verdict, "grey", ""];
              return (
                <tr key={r.key}>
                  <td><b>{r.label}</b></td>
                  <td className="r">{r.n}</td>
                  <td className="r">{r.win}%</td>
                  <td className={`r ${r.net >= 0 ? "pos" : "neg"}`}>{r.net >= 0 ? "+" : "−"}${Math.abs(r.net).toFixed(2)}</td>
                  <td><span className={`op-tag ${col}`} title={what}>{lab}</span></td>
                  <td className="muted small">{r.why}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {rows.length > 12 && <button className="linkish" onClick={() => setAll(!all)}>{all ? "show fewer" : `show all ${rows.length}`}</button>}
    </div>
  );
}
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
