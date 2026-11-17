import React, { useMemo } from "react";
import {
  Area, CartesianGrid, Cell, ComposedChart, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart,
  Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";

const usd = (v, sign = false) => (v == null || Number.isNaN(v) ? "—"
  : `${sign ? (v >= 0 ? "+" : "−") : ""}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
const hhmm = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
const clamp01 = (x) => Math.max(0, Math.min(1, Number(x) || 0));
const MAX_OPEN = 14;

/* ============================================================ risk reactor: three rings, one verdict */
function Ring({ r, frac, color, width = 12 }) {
  const c = 2 * Math.PI * r;
  return (
    <>
      <circle cx="110" cy="110" r={r} className="rr-track" strokeWidth={width} />
      <circle cx="110" cy="110" r={r} className="rr-arc" stroke={color} strokeWidth={width}
              strokeDasharray={c} strokeDashoffset={c * (1 - clamp01(frac))} transform="rotate(-90 110 110)" />
    </>
  );
}

export function RiskReactor({ d }) {
  const a = d.account || {};
  const st = d.stats || {};
  const lossUsed = a.daily_loss_limit ? clamp01(-(a.today || 0) / a.daily_loss_limit) : 0;
  const open = (d.blotter || []).length;
  const win = st.closed ? st.wins / st.closed : null;
  const level = lossUsed >= 0.8 ? "danger" : lossUsed >= 0.5 || (win != null && win < 0.45) ? "careful" : "safe";
  const lossCol = lossUsed >= 0.8 ? "var(--neg)" : lossUsed >= 0.5 ? "var(--amber)" : "var(--pos)";
  const words = {
    safe: "Losses are small and the safety limits are far away.",
    careful: "Losing more than winning. The limits are still holding, but watch it.",
    danger: "Close to today's loss limit. One or two more losers and the bot stops opening trades until tomorrow.",
  };
  return (
    <div className="card hud reactor">
      <div className="card-head"><b>Risk reactor</b><span className={`rr-level ${level}`}>{level.toUpperCase()}</span></div>
      <div className="rr-body">
        <svg viewBox="0 0 220 220" className={`rr-svg ${level}`} role="img"
             aria-label={`Daily loss limit ${Math.round(lossUsed * 100)}% used, ${open} trades open, win rate ${win == null ? "unknown" : Math.round(win * 100) + "%"}`}>
          <Ring r={96} frac={lossUsed} color={lossCol} />
          <Ring r={78} frac={open / MAX_OPEN} color="var(--cyan)" />
          <Ring r={60} frac={win ?? 0} color={win != null && win >= 0.5 ? "var(--pos)" : "var(--violet)"} />
          <text x="110" y="104" className="rr-big">{Math.round(lossUsed * 100)}%</text>
          <text x="110" y="124" className="rr-small">loss stop used</text>
        </svg>
        <ul className="rr-legend">
          <li><i style={{ background: lossCol }} /><b>Loss stop {Math.round(lossUsed * 100)}%</b>
            <span>{usd(-(a.today || 0) > 0 ? -(a.today || 0) : 0)} lost of the {usd(a.daily_loss_limit)} allowed today. At 100% new trades stop until tomorrow.</span></li>
          <li><i style={{ background: "var(--cyan)" }} /><b>{open} of {MAX_OPEN} trades open</b>
            <span>How much is at work in the market right now.</span></li>
          <li><i style={{ background: "var(--violet)" }} /><b>Win rate {win == null ? "—" : `${Math.round(win * 100)}%`}</b>
            <span>{st.wins ?? 0} won, {st.losses ?? 0} lost ({usd(st.pnl_closed, true)}). Below 50% only works if winners are bigger than losers.</span></li>
        </ul>
      </div>
      <div className={`rr-verdict ${level}`}>{words[level]}</div>
    </div>
  );
}

/* ============================================================ today's story: equity with every trade on it */
function StoryTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="tip">
      <div className="tip-t">{hhmm(p.t)}</div>
      {p.trade ? <div><b className={p.trade.side === "BUY" ? "pos" : "neg"}>{p.trade.side}</b> {p.trade.symbol} · {p.trade.volume} lots @ {p.trade.price}</div> : null}
      <div>Account worth <b>{usd(p.equity)}</b></div>
    </div>
  );
}

export function DayStory({ d }) {
  const a = d.account || {};
  const start = (a.equity ?? 0) - (a.today ?? 0);
  const brake = start - (a.daily_loss_limit ?? 0);
  const { line, marks } = useMemo(() => {
    const mid = new Date(); mid.setHours(0, 0, 0, 0);
    const pts = (d.equity || []).filter((p) => p.t >= mid.getTime());
    const series = pts.length >= 2 ? pts : (d.equity || []).slice(-60);
    const near = (t) => {
      let best = series[0];
      for (const p of series) { if (Math.abs(p.t - t) < Math.abs(best.t - t)) best = p; }
      return best;
    };
    const mk = series.length ? (d.trades || []).filter((t) => t.t >= series[0].t).map((t) => ({ t: t.t, equity: near(t.t).equity, trade: t })) : [];
    return { line: series, marks: mk };
  }, [d.equity, d.trades]);
  if (line.length < 2) return <div className="card hud story"><div className="muted small">The story of the day starts after a few minutes of data.</div></div>;
  const vals = line.map((p) => p.equity).concat([start, brake]);
  const lo = Math.min(...vals), hi = Math.max(...vals), pad = Math.max((hi - lo) * 0.08, 1);
  const now = line[line.length - 1].equity;
  const up = now >= start;
  return (
    <div className="card hud story">
      <div className="card-head">
        <b>Today's story</b>
        <span className="muted small">started {usd(start)} → now <b className={up ? "pos" : "neg"}>{usd(now)}</b> ({usd(now - start, true)})</span>
      </div>
      <div className="story-chart">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={line} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="storyfill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={up ? "#34d399" : "#f87171"} stopOpacity={0.35} />
                <stop offset="100%" stopColor={up ? "#34d399" : "#f87171"} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" vertical={false} />
            <ReferenceArea y1={lo - pad} y2={brake} fill="rgba(248,113,113,.08)" ifOverflow="hidden" />
            <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={hhmm} stroke="#5b6475"
                   fontSize={11} tickLine={false} axisLine={false} minTickGap={60} />
            <YAxis type="number" domain={[lo - pad, hi + pad]} allowDataOverflow tickFormatter={(v) => `$${v.toFixed(0)}`} stroke="#5b6475" fontSize={11}
                   tickLine={false} axisLine={false} width={50} />
            <Tooltip content={<StoryTip />} cursor={{ stroke: "rgba(255,255,255,.25)" }} />
            <ReferenceLine y={start} stroke="#94a3b8" strokeDasharray="5 5"
                           label={{ value: "start of day", position: "insideTopLeft", fill: "#94a3b8", fontSize: 11 }} />
            <ReferenceLine y={brake} stroke="#f87171" strokeWidth={1.5}
                           label={{ value: "safety brake: no new trades below here", position: "insideBottomRight", fill: "#f87171", fontSize: 11 }} />
            <Area type="monotone" dataKey="equity" stroke={up ? "#34d399" : "#f87171"} strokeWidth={2.2} fill="url(#storyfill)"
                  isAnimationActive animationDuration={1200} dot={false} />
            <Scatter data={marks} dataKey="equity" isAnimationActive animationDuration={900}
                     shape={(p) => <path className="story-mark" transform={`translate(${p.cx},${p.cy})`}
                                         d={p.payload.trade.side === "BUY" ? "M0,-7 L6,4 L-6,4 Z" : "M0,7 L6,-4 L-6,-4 Z"}
                                         fill={p.payload.trade.side === "BUY" ? "#34d399" : "#f87171"} />} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <div className="ex small muted">Line = what the account is worth through the day. ▲ = a buy, ▼ = a sell (hover for details).
        The shaded red zone is where the daily safety brake kicks in.</div>
    </div>
  );
}

/* ============================================================ market radar: how close each market is to a trade */
function Orb({ s }) {
  const ready = s.threshold ? clamp01(Math.abs(s.signal || 0) / s.threshold) : 0;
  const dir = (s.signal || 0) > 0 ? "up" : (s.signal || 0) < 0 ? "down" : "flat";
  const r = 30, c = 2 * Math.PI * r;
  return (
    <svg viewBox="0 0 80 80" className={`orb ${dir} ${ready >= 1 ? "ready" : ""}`} aria-hidden>
      <circle cx="40" cy="40" r={r} className="orb-track" />
      <circle cx="40" cy="40" r={r} className="orb-fill" strokeDasharray={c} strokeDashoffset={c * (1 - ready)} transform="rotate(-90 40 40)" />
      <circle cx="40" cy="40" r={10 + 10 * ready} className="orb-core" />
      <text x="40" y="45" className="orb-arrow">{dir === "up" ? "▲" : dir === "down" ? "▼" : "•"}</text>
    </svg>
  );
}

export function MarketRadar({ d }) {
  const chg = Object.fromEntries((d.markets || []).map((m) => [m.symbol, m.chg_pct]));
  return (
    <div className="card hud radar">
      <div className="card-head"><b>Market radar</b>
        <span className="muted small">The ring fills as the voters agree. A full, glowing ring means the signal is strong enough to pay the spread.</span>
      </div>
      <div className="radar-grid">
        {(d.symbols || []).map((s) => {
          const ready = s.threshold ? Math.min(999, Math.abs(s.signal || 0) / s.threshold) : 0;
          const tot = (s.up || 0) + (s.down || 0);
          const upShare = tot ? (s.up || 0) / tot : 0.5;
          const pos = s.position;
          const ch = chg[s.symbol];
          return (
            <div key={s.symbol} className={`radar-tile ${pos ? "has-pos" : ""}`} title={s.why}>
              <div className="rt-top">
                <b>{s.name}</b>
                {ch != null && <span className={`rt-chg ${ch >= 0 ? "pos" : "neg"}`}>{ch >= 0 ? "+" : ""}{ch.toFixed(2)}%</span>}
              </div>
              <Orb s={s} />
              <div className="rt-ready">{ready >= 1 ? "strong enough to trade" : `${Math.round(ready * 100)}% of the way to a trade`}</div>
              <div className="tug" aria-label={`${s.up} voters say up, ${s.down} say down`}>
                <span className="tug-up" style={{ width: `${upShare * 100}%` }} />
                <span className="tug-knot" style={{ left: `${upShare * 100}%` }} />
              </div>
              <div className="tug-lab"><span className="pos">{s.up} up</span><span className="neg">{s.down} down</span></div>
              {pos ? (
                <div className={`rt-pos ${pos.pnl >= 0 ? "pos" : "neg"}`}>{pos.side} {pos.volume} · {usd(pos.pnl, true)}</div>
              ) : <div className="rt-pos muted">no trade open</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ============================================================ setup map: every kind of trade on one chart */
const VCOL = { full: "#34d399", probe: "#fbbf24", retest: "#a78bfa", block: "#f87171" };
const VLAB = { full: "proven", probe: "learning", retest: "re-test", block: "blocked" };

function SetupTip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const r = payload[0].payload;
  return (
    <div className="tip">
      <div className="tip-t">{r.label}</div>
      <div>{r.n} trades · won {r.win}% · <b className={r.net >= 0 ? "pos" : "neg"}>{usd(r.net, true)}</b></div>
      <div className="muted">{VLAB[r.verdict] || r.verdict}: {r.why}</div>
    </div>
  );
}

export function SetupMap({ data }) {
  const rows = data?.rows || [];  // win is already a percentage
  if (!rows.length) return null;
  const ys = rows.map((r) => r.net);
  const yMax = Math.max(5, ...ys.map(Math.abs)) * 1.15;
  return (
    <div className="card hud setmap">
      <div className="card-head"><b>Setup map</b>
        <span className="muted small">Each bubble is one kind of trade. Bigger = more trades. Top right made money and wins often; bottom left loses.</span>
      </div>
      <div className="setmap-chart">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 12, right: 16, bottom: 18, left: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" />
            <ReferenceArea x1={50} x2={100} y1={0} y2={yMax} fill="rgba(52,211,153,.06)" />
            <ReferenceArea x1={0} x2={50} y1={-yMax} y2={0} fill="rgba(248,113,113,.06)" />
            <XAxis type="number" dataKey="win" domain={[0, 100]} unit="%" stroke="#5b6475" fontSize={11} tickLine={false}
                   label={{ value: "how often it wins", position: "insideBottom", offset: -8, fill: "#7d8799", fontSize: 11 }} />
            <YAxis type="number" dataKey="net" domain={[-yMax, yMax]} tickFormatter={(v) => `$${v.toFixed(0)}`} stroke="#5b6475"
                   fontSize={11} tickLine={false} width={50} />
            <ZAxis type="number" dataKey="n" range={[40, 520]} />
            <ReferenceLine x={50} stroke="rgba(148,163,184,.35)" strokeDasharray="4 4" />
            <ReferenceLine y={0} stroke="rgba(148,163,184,.5)" />
            <Tooltip content={<SetupTip />} cursor={{ strokeDasharray: "3 3" }} />
            <Scatter data={rows} isAnimationActive animationDuration={1000}>
              {rows.map((r) => <Cell key={r.key} fill={VCOL[r.verdict] || "#94a3b8"} fillOpacity={0.55} stroke={VCOL[r.verdict] || "#94a3b8"} />)}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
        <span className="q q-tr">winners</span><span className="q q-tl">few wins, but big ones</span>
        <span className="q q-br">wins often, loses more</span><span className="q q-bl">losers</span>
      </div>
      <div className="setmap-leg">{Object.entries(VLAB).map(([k, l]) => <span key={k}><i style={{ background: VCOL[k] }} />{l}</span>)}</div>
    </div>
  );
}
