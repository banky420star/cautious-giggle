import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Area, AreaChart, CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  Activity, ArrowDownRight, ArrowUpRight, Brain, CircleDollarSign, Clock, GraduationCap, Lightbulb, Pause,
  Receipt, ShieldCheck, Target, TrendingUp, Trophy, Wallet, Zap, Sparkles, Cpu, FlaskConical, Database,
  ArrowLeftRight, Moon, Newspaper, CalendarClock, LogOut, LayoutGrid, ListChecks, Gauge,
} from "lucide-react";
import {
  WarCard, MissionControl, OpsLog, CommandBar, Section, Clocks, Sitrep, NewsAlert, Tour, Glossary, HelpButtons, condition, ago,
  Tape, Blotter, FillAlerts, SoundButton,
  Setups, Models, SectionNav, BackToTop,
} from "./WarRoom.jsx";
import { Boot, AccountSwitch, Logo } from "./Boot.jsx";

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
  const [flash, setFlash] = useState(null);
  const from = useRef(value ?? 0);
  useEffect(() => {
    if (value == null) return;
    if (Math.abs(value - from.current) > 1e-9) {
      setFlash(null);
      requestAnimationFrame(() => setFlash(value > from.current ? "up" : "down"));
    }
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
  return <span className={`num ${flash ? `flash-${flash}` : ""}`} onAnimationEnd={() => setFlash(null)}>{format(shown)}</span>;
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

const store = {
  get: (k, d) => { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set: (k, v) => { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
};

function Header({ d, explain, setExplain, onTour, onGloss, onAccount, sound, setSound, onLogout }) {
  const bot = d?.bot || {};
  const c = d ? condition(d) : null;
  return (
    <header className="header hud">
      <div className="brand">
        <Logo size={40} />
        <div>
          <div className="brand-name">CAUTIOUS GIGGLE <span className="wr-tag">WAR ROOM</span></div>
          <div className="brand-sub">
            <button className={`acct ${d?.account_mode === "live" ? "live" : ""}`} onClick={onAccount} title="Switch between demo and live">
              {d?.account_mode === "live" ? "LIVE · REAL MONEY" : "DEMO ACCOUNT"}
            </button>
            {" "}· {d?.symbols?.length || 0} markets · 5-minute bars
          </div>
        </div>
      </div>
      <Clocks />
      <div className="header-right">
        {c && <span className={`cond ${c.level}`} title={c.why}><i />{c.label}</span>}
        <span className={`pill ${bot.running ? "green" : "red"}`} title={bot.running ? "The trading bot process is running" : "The bot is stopped"}>
          <span className={`dot ${bot.running ? "pulse" : ""}`} /> {bot.running ? "BOT ONLINE" : "BOT OFFLINE"}
        </span>
        {d && <Countdown seconds={bot.next_check_s} />}
        <SoundButton on={sound} set={setSound} />
        <HelpButtons explain={explain} setExplain={setExplain} onTour={onTour} onGloss={onGloss} />
        <button className="logout" onClick={onLogout} title="Back to the login screen. The bot keeps trading; this only closes the war room view.">
          <LogOut size={14} /> LOG OUT
        </button>
      </div>
    </header>
  );
}

function navItems(d) {
  const blot = (d.blotter?.rows || d.blotter || []);
  const nPos = Array.isArray(blot) ? blot.length : 0;
  const blocked = (d.setups?.rows || []).filter((r) => r.verdict === "block").length;
  const nextNews = (d.news?.plan || []).filter((e) => e.t0 > Date.now() / 1000).length;
  return [
    { id: "sec-command", label: "Overview", Icon: Gauge },
    { id: "sec-positions", label: "Positions", Icon: Wallet, badge: nPos || null, tone: "cyan" },
    { id: "sec-setups", label: "Setups", Icon: ListChecks, badge: blocked ? `${blocked} blocked` : null, tone: "red" },
    { id: "sec-models", label: "Your models", Icon: Brain, badge: d.models ? (d.models.live ? "live" : "off") : null, tone: d.models?.live ? "green" : "red" },
    { id: "sec-markets", label: "Markets", Icon: LayoutGrid, badge: d.symbols?.length || null },
    { id: "sec-ops", label: "Log & equity", Icon: Activity },
    { id: "sec-training", label: "Training", Icon: GraduationCap },
    { id: "sec-news", label: "News", Icon: Newspaper, badge: nextNews || null, tone: "amber" },
  ];
}

export default function App() {
  const [d, err] = useApi(2000);
  const [booted, setBooted] = useState(() => { try { return window.location.hash === "#warroom" || sessionStorage.getItem("cg_booted") === "1"; } catch { return false; } });
  const [explain, setExplainS] = useState(() => store.get("cg_explain", true));
  const setExplain = (v) => { setExplainS(v); store.set("cg_explain", v); };
  const [sound, setSoundS] = useState(() => store.get("cg_sound", false));
  const setSound = (v) => { setSoundS(v); store.set("cg_sound", v); };
  const [tour, setTour] = useState(false);
  const [gloss, setGloss] = useState(false);
  const [acct, setAcct] = useState(false);
  const logout = () => {
    try { sessionStorage.removeItem("cg_booted"); } catch { /* ignore */ }
    if (window.location.hash) window.history.replaceState(null, "", window.location.pathname);
    window.scrollTo(0, 0);
    setBooted(false);
  };
  const enter = () => {
    setBooted(true);
    try { sessionStorage.setItem("cg_booted", "1"); } catch { /* ignore */ }
    if (!store.get("cg_toured", false)) { setTour(true); store.set("cg_toured", true); }
  };
  if (!booted || !d) return <Boot d={d} err={err} onEnter={enter} onAccount={() => setAcct(true)} acctOpen={acct} closeAcct={() => setAcct(false)} />;
  return (
    <div className={`app ${explain ? "" : "explain-off"}`}>
      <div className="scan" aria-hidden />
      <Header d={d} explain={explain} setExplain={setExplain} onTour={() => setTour(true)} onGloss={() => setGloss(true)} onAccount={() => setAcct(true)} sound={sound} setSound={setSound} onLogout={logout} />
      <Tape symbols={d.symbols} />
      <SectionNav items={navItems(d)} />
      {err && <div className="banner">{err} · showing last known data</div>}
      <NewsAlert d={d} />
      <Sitrep d={d} />

      <Section id="sec-command" no="01" title="COMMAND" sub="the account at a glance and the safety limits">
        <CommandBar d={d} Animated={AnimatedNumber} />
      </Section>

      <Section id="sec-positions" no="02" title="POSITIONS"
               sub="every open trade: where price is between its stop and its target, profit in R and dollars, time held">
        <Blotter rows={d.blotter} />
      </Section>

      <Section id="sec-setups" no="03" title="SETUPS"
               sub="which kinds of trade have earned normal size, which are still learning on the smallest lot, and which are blocked for losing">
        <Setups data={d.setups} />
      </Section>

      <Section id="sec-models" no="04" title="YOUR MODELS"
               sub="your own models voting in every decision: what each says now, its live record, and how it did when replayed on history">
        <Models data={d.models} />
      </Section>

      <Section id="sec-markets" no="05" title="MARKETS"
               sub="one card per market: the needle is the final verdict, the bars are the voters, the line underneath is the next move">
        <div className="war-grid">{d.symbols.map((s) => <WarCard key={s.symbol} s={s} />)}</div>
      </Section>

      <Section id="sec-ops" no="06" title="OPS LOG & EQUITY" sub="every action with its reason, and how the account has moved">
        <div className="bottom">
          <OpsLog ops={d.ops} feed={d.feed} trades={d.trades} />
          <EquityChart series={d.equity} equity={d.account.equity} />
        </div>
      </Section>

      <Section id="sec-training" no="07" title="TRAINING" sub="each market's AI: which reward style it is practising, and its exam record">
        <MissionControl rows={d.mission} training={d.training} />
      </Section>

      <Section id="sec-news" no="08" title="NEWS" sub="scheduled releases, what moved markets, and the news playbook's track record">
        <News news={d.news} />
      </Section>

      <footer className="footer">
        <Lightbulb size={14} /> <b>{d.tip.term.replace(" ┃", "")}:</b> {d.tip.text}
        <span className="muted right"><Clock size={12} /> updated {new Date(d.ts * 1000).toLocaleTimeString()}</span>
      </footer>
      <BackToTop />
      <FillAlerts ops={d.ops} sound={sound} />
      <Tour open={tour} onClose={() => setTour(false)} />
      <Glossary open={gloss} onClose={() => setGloss(false)} terms={d.glossary} />
      <AccountSwitch open={acct} onClose={() => setAcct(false)} />
    </div>
  );
}
