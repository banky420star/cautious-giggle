import React, { useState } from "react";
import { Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Badge, Empty, Panel, Segmented, Side, money, num, px, stamp, tone, usePoll, useStored } from "./ui.jsx";

// Every trader ("brain") side by side: settings, record, P&L curve, open trades, latest calls, and each closed
// trade with the brain's own reason. Data: /api/brains (tools/web_dashboard.py brains_view).

const PERIODS = [["today", "Today"], ["week", "7 days"], ["all", "All time"]];
const NAMES = { BTCUSDm: "Bitcoin", XAUUSDm: "Gold", EURUSDm: "EUR/USD", USDJPYm: "USD/JPY", XAGUSDm: "Silver", ETHUSDm: "Ether", USOILm: "Oil" };
const pctOf = (v) => (v == null ? "—" : `${Math.round(v * 100)}%`);

function Curve({ curve }) {
  if (!curve || curve.length < 2) return <Empty>Not enough closed trades for a curve yet.</Empty>;
  const data = curve.map(([t, v]) => ({ t, v }));
  const last = data[data.length - 1].v;
  const oneDay = data[data.length - 1].t - data[0].t < 36 * 3600 * 1000;   // within a day: show times, not dates
  const tick = (t) => (oneDay ? new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : stamp(t).split(",")[0]);
  return (
    <div style={{ height: 150 }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tickFormatter={tick} tick={{ fontSize: 11 }} minTickGap={40} />
          <YAxis width={56} tick={{ fontSize: 11 }} tickFormatter={(v) => money(v)} />
          <ReferenceLine y={0} strokeDasharray="3 3" />
          <Tooltip formatter={(v) => [money(v, true), "Total P&L"]} labelFormatter={(t) => stamp(t)} />
          <Line type="stepAfter" dataKey="v" dot={false} strokeWidth={2} stroke={last >= 0 ? "var(--up)" : "var(--down)"} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function Record({ r }) {
  return (
    <div className="strip">
      <div className="cell"><span className="lbl">Trades</span><span className="val num">{r.trades}</span><span className="sub">{r.wins}W {r.trades - r.wins}L</span></div>
      <div className="cell"><span className="lbl">Win rate</span><span className="val num">{pctOf(r.win_rate)}</span></div>
      <div className="cell"><span className="lbl">Net</span><span className={`val num ${tone(r.pnl)}`}>{money(r.pnl, true)}</span></div>
      <div className="cell"><span className="lbl">Avg win / loss</span><span className="val num">{r.avg_win == null ? "—" : money(r.avg_win, true)}</span>
        <span className="sub down">{r.avg_loss == null ? "" : money(r.avg_loss, true)}</span></div>
      <div className="cell"><span className="lbl">Best / worst</span><span className={`val num ${tone(r.best)}`}>{r.best == null ? "—" : money(r.best, true)}</span>
        <span className={`sub ${tone(r.worst)}`}>{r.worst == null ? "" : money(r.worst, true)}</span></div>
    </div>
  );
}

function Calls({ calls, minc }) {
  const rows = Object.entries(calls || {});
  if (!rows.length) return null;
  return (
    <>
      <div className="subhead">Latest calls (every 15 minutes)</div>
      <table className="tbl">
        <thead><tr><th>Market</th><th>Call</th><th className="r">Sure</th><th className="hide-s">Why</th></tr></thead>
        <tbody>{rows.map(([sym, c]) => (
          <tr key={sym}><td className="strong">{NAMES[sym] || sym}</td>
            <td>{c.error ? <span className="down">no answer</span> : c.decision === "hold" ? <span className="dim">Hold</span>
              : <Side side={c.decision === "buy" ? "BUY" : "SELL"} />}</td>
            <td className={`r num ${c.decision !== "hold" && c.confidence >= (minc || 0.6) ? "strong" : "dim"}`}>{c.error ? "—" : pctOf(c.confidence)}</td>
            <td className="hide-s why" title={c.reason}>{(c.reason || "").slice(0, 120)}</td></tr>))}</tbody>
      </table>
    </>
  );
}

function OpenTrades({ open }) {
  if (!open?.length) return <p className="help">No open trades.</p>;
  return (
    <table className="tbl">
      <thead><tr><th>Market</th><th>Side</th><th className="r">Size</th><th className="r">Entry</th><th className="r hide-s">Stop</th><th className="r hide-s">Target</th><th className="r">P&amp;L</th></tr></thead>
      <tbody>{open.map((p) => (
        <tr key={p.ticket}><td className="strong">{p.name}</td><td><Side side={p.side} /></td><td className="r num">{num(p.volume)}</td>
          <td className="r num">{px(p.entry)}</td><td className="r num hide-s">{px(p.sl)}</td><td className="r num hide-s">{px(p.tp)}</td>
          <td className={`r num strong ${tone(p.pnl)}`}>{money(p.pnl, true)}</td></tr>))}</tbody>
    </table>
  );
}

function Recent({ rows }) {
  if (!rows?.length) return <p className="help">No closed trades yet.</p>;
  return (
    <table className="tbl">
      <thead><tr><th>Closed</th><th>Market</th><th>Side</th><th className="r hide-s">Entry</th><th className="r">Exit</th><th className="r">P&amp;L</th><th className="hide-s">Why it traded</th></tr></thead>
      <tbody>{rows.map((t) => (
        <tr key={`${t.ticket}-${t.t}`}><td className="num dim">{stamp(t.t)}</td><td className="strong">{t.name}</td>
          <td>{t.side ? <Side side={t.side} /> : <span className="dim">—</span>}</td>
          <td className="r num hide-s">{t.entry ? px(t.entry) : "—"}</td><td className="r num">{px(t.exit)}</td>
          <td className={`r num strong ${tone(t.pnl)}`}>{money(t.pnl, true)}</td>
          <td className="hide-s why" title={t.why}>{t.why ? t.why.slice(0, 140) : <span className="dim">council signal</span>}</td></tr>))}</tbody>
    </table>
  );
}

function BrainCard({ b }) {
  const [period, setPeriod] = useState("all");
  const r = b.record[period];
  const meta = [b.model, b.risk_pct != null ? `risk ${b.risk_pct}% per trade` : null,
    b.first_trade ? `trading since ${stamp(b.first_trade)}` : null].filter(Boolean).join(" · ");
  return (
    <Panel className="span-all"
      title={<span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>{b.who}
        <Badge kind={b.enabled ? "up" : ""}>{b.enabled ? "On" : "Off"}</Badge>
        {b.open.length > 0 && <Badge kind="info">{b.open.length} open</Badge>}</span>}
      meta={meta}
      actions={<Segmented value={period} options={PERIODS} onChange={setPeriod} label={`${b.who} record period`} />}>
      <div className="stack">
        <p className="help">{b.about}</p>
        <Record r={r} />
        <Curve curve={b.curve} />
        <div className="subhead">Open now</div>
        <OpenTrades open={b.open} />
        <Calls calls={b.calls} minc={b.min_confidence} />
        <div className="subhead">Recent closed trades</div>
        <Recent rows={b.recent} />
      </div>
    </Panel>
  );
}

export default function Brains() {
  const [d, err] = usePoll("/api/brains", 20000);
  const [pick, setPick] = useStored("cg.brains.pick", "all");
  if (err) return <Empty>{err}</Empty>;
  if (!d) return <Empty>Loading the brains…</Empty>;
  const brains = d.brains || [];
  const shown = pick === "all" ? brains : brains.filter((b) => b.who === pick);
  return (
    <div className="grid">
      <Panel title="Every brain side by side" meta="Tap one to see only it" className="span-all" flush>
        <table className="tbl hover">
          <thead><tr><th>Brain</th><th>Status</th><th className="r">Open</th><th className="r">Floating</th>
            <th className="r">Today</th><th className="r hide-s">7 days</th><th className="r">All time</th><th className="r hide-s">Win rate</th><th className="r hide-s">Trades</th></tr></thead>
          <tbody>{brains.map((b) => {
            const fl = b.open.reduce((a, p) => a + (p.pnl || 0), 0);
            return (
              <tr key={b.who} onClick={() => setPick(pick === b.who ? "all" : b.who)} className={pick === b.who ? "sel" : ""} style={{ cursor: "pointer" }}>
                <td className="strong">{b.who}</td><td>{b.enabled ? "On" : <span className="dim">Off</span>}</td>
                <td className="r num">{b.open.length}</td><td className={`r num ${tone(fl)}`}>{b.open.length ? money(fl, true) : "—"}</td>
                <td className={`r num ${tone(b.record.today.pnl)}`}>{money(b.record.today.pnl, true)}</td>
                <td className={`r num hide-s ${tone(b.record.week.pnl)}`}>{money(b.record.week.pnl, true)}</td>
                <td className={`r num strong ${tone(b.record.all.pnl)}`}>{money(b.record.all.pnl, true)}</td>
                <td className="r num hide-s">{pctOf(b.record.all.win_rate)}</td><td className="r num hide-s">{b.record.all.trades}</td></tr>);
          })}</tbody>
        </table>
      </Panel>
      {pick !== "all" && <div className="span-all"><button className="linkbtn" onClick={() => setPick("all")}>Show every brain</button></div>}
      {shown.map((b) => <BrainCard key={b.who} b={b} />)}
    </div>
  );
}
