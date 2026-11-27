import React, { useMemo, useState } from "react";
import { Area, AreaChart, Brush, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { money, moneyShort, pct, stamp, tone, usePoll, useStored, Segmented, Empty } from "./ui.jsx";

const RANGES = [["1h", "1H"], ["1d", "1D"], ["1w", "1W"], ["1m", "1M"], ["3m", "3M"], ["all", "All"]];
const MODES = [["equity", "Equity"], ["pnl", "P&L"]];

// "nice" axis bounds for whatever is visible, so a $15 move on a $1,700 account is readable
function niceDomain(lo, hi, includeZero) {
  if (includeZero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  if (!(hi > lo)) { const p = Math.abs(hi) * 0.01 || 1; lo -= p; hi += p; }
  const raw = (hi - lo) / 4;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  return [Math.floor(lo / step) * step, Math.ceil(hi / step) * step];
}

function tickTime(range) {
  return (ms) => {
    const d = new Date(ms);
    if (range === "1h" || range === "1d") return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  };
}

function Readout({ active, payload, mode }) {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="readout">
      <div className="ro-v">{mode === "pnl" ? money(p.pnl, true) : money(p.equity)}</div>
      {mode === "equity" && <div className="ro-row"><i className="key bal" />Balance <b>{money(p.balance)}</b></div>}
      <div className="ro-row"><i className="key dd" />Drawdown <b>{money(p.dd, true)}</b></div>
      <div className="ro-t">{stamp(p.t)}</div>
    </div>
  );
}

export default function EquityPanel({ account }) {
  const [range, setRange] = useStored("cg-eq-range", "1d");
  const [mode, setMode] = useStored("cg-eq-mode", "equity");
  const [view, err, loading] = usePoll(`/api/equity?range=${range}`, 15000);
  const [zoom, setZoom] = useState(null);           // [startIndex, endIndex] from the brush
  const pts = view?.range === range ? view.points : [];
  const s = view?.summary || {};
  const key = mode === "pnl" ? "pnl" : "equity";

  const vis = zoom ? pts.slice(zoom[0], zoom[1] + 1) : pts;
  const domain = useMemo(() => {
    if (!vis.length) return [0, 1];
    let lo = Infinity, hi = -Infinity;
    for (const p of vis) {
      lo = Math.min(lo, p[key]); hi = Math.max(hi, p[key]);
      if (mode === "equity") { lo = Math.min(lo, p.balance); hi = Math.max(hi, p.balance); }
    }
    return niceDomain(lo, hi, mode === "pnl");
  }, [vis, key, mode]);
  const ddDomain = useMemo(() => [Math.min(-1, ...vis.map((p) => p.dd)) * 1.1, 0], [vis]);
  // mark meaningful flows; label only the three largest so a burst of small ones doesn't pile up
  const flows = (view?.flows || []).filter((f) => Math.abs(f.amount) >= Math.max(25, 0.02 * (s.end_equity || 0)));
  const span = (pts.length ? pts[pts.length - 1].t - pts[0].t : 0) || 1;
  const labelled = new Set();
  for (const f of [...flows].sort((a, b) => Math.abs(b.amount) - Math.abs(a.amount))) {
    if (labelled.size < 3 && [...labelled].every((t) => Math.abs(t - f.t) > span * 0.2)) labelled.add(f.t);
  }
  const xFmt = tickTime(range);
  const up = (s.pnl ?? 0) >= 0;

  return (
    <section className="panel equity">
      <div className="eq-head">
        <div className="eq-hero">
          <span className="lbl">Net equity</span>
          <span className="big num">{money(account.equity)}</span>
          <span className={`chg num ${tone(account.today)}`}>{money(account.today, true)} ({pct(account.today_pct)}) today</span>
        </div>
        <dl className="eq-facts">
          <div><dt>Balance</dt><dd className="num">{money(account.balance)}</dd></div>
          <div><dt>Floating</dt><dd className={`num ${tone(account.floating)}`}>{money(account.floating, true)}</dd></div>
          <div><dt>Free margin</dt><dd className="num" title={account.free_margin >= account.equity ? "This account has unlimited leverage: open trades use no margin" : undefined}>{money(account.free_margin)}</dd></div>
          <div><dt>P&amp;L · {RANGES.find((r) => r[0] === range)[1]}</dt><dd className={`num ${tone(s.pnl)}`}>{money(s.pnl, true)}</dd></div>
          <div><dt>Max drawdown</dt><dd className="num down">{money(s.max_dd, true)}</dd></div>
        </dl>
      </div>
      <div className="eq-controls">
        <Segmented label="Range" value={range} options={RANGES} onChange={(v) => { setZoom(null); setRange(v); }} />
        <Segmented label="Measure" value={mode} options={MODES} onChange={setMode} />
        <span className="eq-legend">
          <span><i className={`key ${mode === "pnl" ? (up ? "pnl" : "pnl neg") : "eq"}`} />{mode === "pnl" ? "Trading P&L (deposits removed)" : "Equity"}</span>
          {mode === "equity" && <span><i className="key bal" />Balance</span>}
          {zoom && <button className="linkbtn" onClick={() => setZoom(null)}>Reset zoom</button>}
        </span>
      </div>
      <div className={`eq-charts ${loading && pts.length ? "stale" : ""}`}>
        {err && !pts.length ? <Empty>{err}</Empty> : pts.length < 2 ? <Empty>{loading ? "Loading…" : "Not enough history for this range yet."}</Empty> : (
          <>
            <ResponsiveContainer width="100%" height={280}>
              <ComposedChart data={pts} syncId="eq" syncMethod="value" margin={{ top: 8, right: 12, left: 4, bottom: 0 }}>
                <defs>
                  <linearGradient id="eqfill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="var(--series)" stopOpacity={0.16} />
                    <stop offset="100%" stopColor="var(--series)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid vertical={false} stroke="var(--grid)" />
                <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} tickFormatter={xFmt}
                  stroke="var(--axis)" tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} minTickGap={64} />
                <YAxis domain={domain} allowDataOverflow tickFormatter={moneyShort} orientation="right"
                  stroke="transparent" tick={{ fill: "var(--muted)", fontSize: 11 }} tickLine={false} width={64} tickCount={5} />
                <Tooltip content={<Readout mode={mode} />} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} isAnimationActive={false} />
                {mode === "pnl" && <ReferenceLine y={0} stroke="var(--axis)" />}
                {flows.map((f) => (
                  <ReferenceLine key={f.t} x={f.t} stroke="var(--muted)" strokeDasharray="2 3"
                    label={labelled.has(f.t) ? { value: `${f.kind === "account switch" ? "Account switch" : f.amount > 0 ? "Deposit" : "Withdrawal"} ${moneyShort(Math.abs(f.amount))}`,
                      position: "insideTopLeft", fill: "var(--muted)", fontSize: 10 } : undefined} />
                ))}
                {mode === "equity" && <Line type="stepAfter" dataKey="balance" stroke="var(--muted)" strokeWidth={1} strokeDasharray="3 3" dot={false} isAnimationActive={false} />}
                <Area type="monotone" dataKey={key} stroke={mode === "pnl" && !up ? "var(--down)" : "var(--series)"} strokeWidth={2}
                  fill={mode === "pnl" ? "none" : "url(#eqfill)"} dot={false} activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--panel)" }} isAnimationActive={false} />
                <Brush dataKey="t" height={22} travellerWidth={8} stroke="var(--axis)" fill="var(--panel)" tickFormatter={xFmt}
                  startIndex={zoom?.[0]} endIndex={zoom?.[1]}
                  onChange={(r) => setZoom(r && (r.startIndex > 0 || r.endIndex < pts.length - 1) ? [r.startIndex, r.endIndex] : null)} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="dd-label">Drawdown from peak</div>
            <ResponsiveContainer width="100%" height={72}>
              <AreaChart data={vis} syncId="eq" syncMethod="value" margin={{ top: 2, right: 12, left: 4, bottom: 0 }}>
                <XAxis dataKey="t" type="number" scale="time" domain={["dataMin", "dataMax"]} hide />
                <YAxis domain={ddDomain} orientation="right" tickFormatter={moneyShort} stroke="transparent"
                  tick={{ fill: "var(--muted)", fontSize: 10 }} width={64} tickCount={2} />
                <Tooltip content={() => null} cursor={{ stroke: "var(--axis)", strokeWidth: 1 }} />
                <ReferenceLine y={0} stroke="var(--axis)" />
                <Area type="monotone" dataKey="dd" stroke="var(--down)" strokeWidth={1} fill="var(--down-soft)" dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </>
        )}
      </div>
      {s.net_flows ? <p className="foot">Deposits, withdrawals and account switches in this range ({money(s.net_flows, true)}) are not counted as P&amp;L.</p> : null}
    </section>
  );
}
