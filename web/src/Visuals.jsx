import React from "react";
import {
  CartesianGrid, Cell, ReferenceArea, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from "recharts";

const usd = (v, sign = false) => (v == null || Number.isNaN(v) ? "—"
  : `${sign ? (v >= 0 ? "+" : "−") : ""}$${Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`);
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
