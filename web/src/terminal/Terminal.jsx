import React, { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, Cell, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Activity, Brain, CandlestickChart, KeyRound, LayoutGrid, Layers, ListOrdered, Moon, Smartphone, Sun, Workflow, X } from "lucide-react";
import QRCode from "qrcode";
import EquityPanel from "./Equity.jsx";
import Account from "./Account.jsx";
import Pipeline from "./Pipeline.jsx";
import Brains from "./Brains.jsx";
import { Badge, Empty, Panel, Segmented, SignalMeter, Side, Spark, clock, money, num, pct, px, signed, stamp, tone, usePoll, useStored } from "./ui.jsx";
import "./terminal.css";

const NAV = [
  ["overview", "Overview", LayoutGrid],
  ["pipeline", "Pipeline", Workflow],
  ["markets", "Markets", CandlestickChart],
  ["brains", "Brains", Brain],
  ["strategies", "Strategies", Layers],
  ["trades", "Trades", ListOrdered],
  ["activity", "Activity", Activity],
  ["account", "Account", KeyRound],
];

// ------------------------------------------------------------------ shared tables
function Positions({ positions, symbols }) {
  const pend = (symbols || []).filter((s) => s.pending && !s.position);
  const rows = positions || [];
  if (!rows.length && !pend.length) return <Empty>No open positions.</Empty>;
  return (
    <table className="tbl">
      <thead><tr><th>Market</th><th>Side</th><th className="hide-s">By</th><th className="r">Size</th><th className="r">Entry</th><th className="r">Last</th>
        <th className="r hide-s">Stop</th><th className="r hide-s">Target</th><th className="r">P&amp;L</th></tr></thead>
      <tbody>
        {rows.map((p) => (
          <tr key={p.ticket}><td className="strong">{p.name}</td><td><Side side={p.side} /></td>
            <td className="dim hide-s">{p.who}</td>
            <td className="r num">{num(p.volume)}</td><td className="r num">{px(p.entry)}</td><td className="r num">{p.now ? px(p.now) : "—"}</td>
            <td className="r num hide-s">{p.sl ? px(p.sl) : "—"}</td><td className="r num hide-s">{p.tp ? px(p.tp) : "—"}</td>
            <td className={`r num strong ${tone(p.pnl)}`}>{money(p.pnl, true)}</td></tr>))}
        {pend.map((s) => (
          <tr key={`p-${s.symbol}`}><td className="strong">{s.name}</td><td><Badge kind="info">Pending {s.pending.side}</Badge></td>
            <td className="dim hide-s">Council bot</td><td className="r">—</td>
            <td className="r num">{px(s.pending.limit)}</td><td className="r num">{px(s.bid)}</td><td className="r hide-s">—</td><td className="r hide-s">—</td><td className="r">—</td></tr>))}
      </tbody>
    </table>
  );
}

function Watchlist({ symbols, full }) {
  return (
    <table className="tbl hover">
      <thead><tr><th>Market</th><th className="r">Last</th><th className="r">Chg</th><th className="hide-s">Session</th>
        <th>Signal</th>{full && <th className="hide-m">Model</th>}{full && <th className="hide-m">Why</th>}</tr></thead>
      <tbody>{(symbols || []).map((s) => (
        <tr key={s.symbol}>
          <td><span className="strong">{s.name}</span>{s.position && <span className={`dot-pos ${s.position.side === "BUY" ? "up" : "down"}`} title="open position" />}</td>
          <td className="r num">{px(s.bid)}</td>
          <td className={`r num ${tone(s.chg_pct)}`}>{pct(s.chg_pct)}</td>
          <td className="hide-s"><Spark candles={s.candles} /></td>
          <td><SignalMeter value={s.signal} threshold={s.threshold} /></td>
          {full && <td className="hide-m dim small">{s.brain || s.mode}</td>}
          {full && <td className="hide-m why">{s.why}</td>}
        </tr>))}</tbody>
    </table>
  );
}

function TradesTable({ trades, limit }) {
  const rows = (trades || []).slice(0, limit || 999);
  if (!rows.length) return <Empty>No trades yet.</Empty>;
  return (
    <table className="tbl">
      <thead><tr><th>Time</th><th>Market</th><th>Action</th><th className="r">Size</th><th className="r">Price</th><th className="r">Realized</th></tr></thead>
      <tbody>{rows.map((t, i) => (
        <tr key={i}><td className="num dim">{stamp(t.t)}</td><td className="strong">{t.symbol.replace(/m$/, "")}</td>
          <td>{t.closing ? <span className="dim">Close {t.side === "BUY" ? "short" : "long"}</span> : <Side side={t.side} />}</td>
          <td className="r num">{num(t.volume)}</td><td className="r num">{px(t.price)}</td>
          <td className={`r num ${tone(t.pnl)}`}>{t.pnl == null ? "" : money(t.pnl, true)}</td></tr>))}</tbody>
    </table>
  );
}

function Feed({ items, limit = 12 }) {
  const rows = (items || []).slice(0, limit);
  if (!rows.length) return <Empty>Nothing yet.</Empty>;
  return (
    <ol className="feed">{rows.map((f, i) => (
      <li key={i} className={f.tone || ""}><time className="num">{clock((f.t || 0) * 1000)}</time>
        <div><div className="f-t">{f.title}</div>{f.text && <div className="f-x">{f.text}</div>}</div></li>))}</ol>
  );
}

// ------------------------------------------------------------------ risk strip
function RiskStrip({ d }) {
  const a = d.account || {}, s = d.stats || {};
  const used = a.daily_loss_limit ? Math.max(0, -Math.min(0, a.today)) / a.daily_loss_limit : 0;
  const open = d.blotter || [];
  const g = d.strategies?.gold_breakout?.state || {};
  const gStat = !d.strategies?.gold_breakout?.enabled ? "Off" : g.ticket && !g.closed ? `In ${g.side > 0 ? "long" : "short"}`
    : g.closed ? "Done today" : g.done ? "No trade today" : g.hi ? "Waiting for break" : "Building range";
  return (
    <div className="strip">
      <div className="cell">
        <span className="lbl">Daily loss used</span>
        <span className="val num">{Math.round(used * 100)}%</span>
        <div className="bar"><span className={used > 0.7 ? "down" : used > 0.4 ? "warn" : "ok"} style={{ width: `${Math.min(100, used * 100)}%` }} /></div>
        <span className="sub">Halts at {money(-a.daily_loss_limit)}</span>
      </div>
      <div className="cell">
        <span className="lbl">Open positions</span>
        <span className="val num">{open.length}</span>
        <span className={`sub ${tone(a.floating)}`}>{money(a.floating, true)} floating</span>
      </div>
      <div className="cell">
        <span className="lbl">Closed today</span>
        <span className="val num">{s.wins ?? 0}<span className="dim">W</span> {s.losses ?? 0}<span className="dim">L</span></span>
        <span className={`sub ${tone(s.pnl_closed)}`}>{money(s.pnl_closed, true)} realized</span>
      </div>
      <div className="cell">
        <span className="lbl">Gold Breakout</span>
        <span className="val">{gStat}</span>
        <span className="sub">{g.hi ? `Range ${px(g.lo)} – ${px(g.hi)}` : "00:00–07:00 GMT+3"}</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------ strategies
function GoldBreakout({ g }) {
  if (!g) return <Empty>Gold Breakout data not available.</Empty>;
  const st = g.state || {}, bt = (g.backtest || {}).all || {};
  const months = Object.entries(g.by_month || {}).map(([m, r]) => ({ m, r }));
  return (
    <div className="stack">
      <dl className="facts">
        <div><dt>Range high</dt><dd className="num">{st.hi ? px(st.hi) : "—"}</dd></div>
        <div><dt>Range low</dt><dd className="num">{st.lo ? px(st.lo) : "—"}</dd></div>
        <div><dt>Entry</dt><dd className="num">{st.entry ? px(st.entry) : "—"}</dd></div>
        <div><dt>Stop</dt><dd className="num">{st.stop ? px(st.stop) : "—"}</dd></div>
        <div><dt>Risk / trade</dt><dd className="num">{g.risk_pct}%</dd></div>
        <div><dt>Stop distance</dt><dd className="num">{g.stop_pct}%</dd></div>
      </dl>
      {st.done && st.done !== "traded" && <p className="help">{st.done}</p>}
      <div className="subhead">Backtest · 17 months of Exness gold, spread paid</div>
      <dl className="facts">
        <div><dt>Trades</dt><dd className="num">{bt.n ?? "—"}</dd></div>
        <div><dt>Win rate</dt><dd className="num">{bt.win != null ? `${Math.round(bt.win * 100)}%` : "—"}</dd></div>
        <div><dt>Avg / trade</dt><dd className={`num ${tone(bt.avgR)}`}>{signed(bt.avgR)}R</dd></div>
        <div><dt>Profit factor</dt><dd className="num">{num(bt.pf)}</dd></div>
        <div><dt>Worst drawdown</dt><dd className="num">{bt.maxDD_R ?? "—"}R</dd></div>
        <div><dt>t-stat</dt><dd className="num">{num(bt.t)}</dd></div>
      </dl>
      {months.length > 0 && (
        <ResponsiveContainer width="100%" height={140}>
          <BarChart data={months} margin={{ top: 6, right: 0, left: 0, bottom: 0 }} barCategoryGap={2}>
            <XAxis dataKey="m" stroke="var(--axis)" tick={{ fill: "var(--muted)", fontSize: 10 }} tickLine={false} tickFormatter={(m) => m.slice(2)} interval="preserveStartEnd" />
            <YAxis orientation="right" stroke="transparent" tick={{ fill: "var(--muted)", fontSize: 10 }} width={32} />
            <ReferenceLine y={0} stroke="var(--axis)" />
            <Tooltip cursor={{ fill: "var(--hover)" }} contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", borderRadius: 6, fontSize: 12 }}
              formatter={(v) => [`${signed(v)}R`, "Month"]} />
            <Bar dataKey="r" radius={[3, 3, 0, 0]} isAnimationActive={false}>{months.map((x) => <Cell key={x.m} fill={x.r >= 0 ? "var(--up)" : "var(--down)"} />)}</Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      {(g.events || []).length > 0 && (
        <>
          <div className="subhead">Recent events</div>
          <ol className="feed tight">{[...g.events].reverse().slice(0, 8).map((e, i) => (
            <li key={i}><time className="num">{stamp(e.t * 1000)}</time>
              <div><span className="f-t">{e.event}</span> <span className="f-x inline">{Object.entries(e).filter(([k, v]) => !["t", "event", "day"].includes(k) && v != null).map(([k, v]) => `${k} ${v}`).join(" · ")}</span></div></li>))}</ol>
        </>
      )}
    </div>
  );
}

function GemmaDesk({ gt, symbols, who = "Gemma" }) {
  if (!gt) return <Empty>{who} trader data not available.</Empty>;
  const last = gt.state?.last || {}, open = gt.state?.open || {};
  return (
    <div className="stack">
      <p className="help">{who} ({gt.model}) reads every council vote, your models, the regime model (only where it has proven an edge), the meta-brain's prediction
        and the SMC structure, then makes its own call each 15 minutes. It trades its own positions, separate from the council,
        when it is at least {Math.round(gt.min_confidence * 100)}% sure. ATR stop and target (the stop manager then moves the stop into profit), 8-hour limit, risk {gt.risk_pct}% per trade. It also sees the account, every open position and whose it is, the session, upcoming news and its own track record.</p>
      <div>
        <table className="tbl">
          <thead><tr><th>Market</th><th>Call</th><th className="r">Sure</th><th className="hide-s">Why</th><th>Position</th></tr></thead>
          <tbody>{(symbols || []).map((s) => {
            const d = last[s.symbol], p = open[s.symbol];
            return (
              <tr key={s.symbol}><td className="strong">{s.name}</td>
                <td>{!d ? <span className="dim">waiting</span> : d.decision === "hold" ? <span className="dim">Hold</span>
                  : <Side side={d.decision === "buy" ? "BUY" : "SELL"} />}</td>
                <td className={`r num ${d && d.confidence >= gt.min_confidence && d.decision !== "hold" ? "strong" : "dim"}`}>{d ? `${Math.round(d.confidence * 100)}%` : "—"}</td>
                <td className="hide-s why" title={d?.reason || ""}>{d ? d.reason.slice(0, 140) : ""}</td>
                <td>{p ? <><Side side={p.side > 0 ? "BUY" : "SELL"} /> <span className="dim num">{num(p.lots)} @ {px(p.entry)}</span></> : <span className="dim">flat</span>}</td></tr>);
          })}</tbody>
        </table>
      </div>
    </div>
  );
}

function MetaBrain({ mt, pred, symbols }) {
  if (!mt) return <Empty>Meta-brain trader data not available.</Empty>;
  const open = mt.state?.open || {};
  const names = Object.fromEntries((symbols || []).map((s) => [s.symbol, s.name]));
  return (
    <div className="stack">
      <p className="help">One model trained on every strategy: the 48-indicator council, bankbot, ICT FVG→iFVG, Sweep + FVG,
        LuxAlgo SMC, SMC rules and Gold Breakout's range break. It opens when the better side is predicted to make more than
        its margin, with an ATR stop and target, and closes after 8 hours. Risk {mt.risk_pct}% per trade.</p>
      <div>
        <table className="tbl">
          <thead><tr><th>Market</th><th className="r">Predicted long</th><th className="r">Predicted short</th><th>Position</th></tr></thead>
          <tbody>{(symbols || []).map((s) => {
            const w = pred?.why?.[s.symbol] || "";
            const m = w.match(/long ([+\-]?[\d.]+)R, short ([+\-]?[\d.]+)R/);
            const p = open[s.symbol];
            return (
              <tr key={s.symbol}><td className="strong">{s.name}</td>
                <td className={`r num ${m && +m[1] > 0 ? "up" : ""}`}>{m ? `${m[1]}R` : "—"}</td>
                <td className={`r num ${m && +m[2] > 0 ? "up" : ""}`}>{m ? `${m[2]}R` : "—"}</td>
                <td>{p ? <><Side side={p.side > 0 ? "BUY" : "SELL"} /> <span className="dim num">{num(p.lots)} @ {px(p.entry)}</span></> : <span className="dim">flat</span>}</td></tr>);
          })}</tbody>
        </table>
      </div>
      {(mt.events || []).length > 0 && (
        <>
          <div className="subhead">Recent events</div>
          <ol className="feed tight">{[...mt.events].reverse().slice(0, 10).map((e, i) => (
            <li key={i}><time className="num">{stamp(e.t * 1000)}</time>
              <div><span className="f-t">{e.event} {names[e.symbol] || e.symbol || ""}</span> <span className="f-x inline">{Object.entries(e).filter(([k, v]) => !["t", "event", "symbol"].includes(k) && v != null).map(([k, v]) => `${k} ${v}`).join(" · ")}</span></div></li>))}</ol>
        </>
      )}
    </div>
  );
}

function Shadow({ shadow, symbols }) {
  const keys = [["smc_gemma", "Gemma SMC"], ["smc_rules", "SMC rules"]];
  return (
    <>
      <table className="tbl">
        <thead><tr><th>Market</th>{keys.map(([k, l]) => <th key={k} className="r">{l}</th>)}</tr></thead>
        <tbody>{(symbols || []).map((s) => (
          <tr key={s.symbol}><td className="strong">{s.name}</td>{keys.map(([k]) => {
            const v = shadow?.[k]?.votes?.[s.symbol];
            return <td key={k} className={`r num ${tone(v)}`} title={shadow?.[k]?.why?.[s.symbol] || ""}>{v == null ? "—" : signed(v)}</td>;
          })}</tr>))}</tbody>
      </table>
      <p className="help pad">Gemma votes in the council (weight 4). SMC rules is inside the meta-brain; its live read is shown here.</p>
    </>
  );
}

function WalkForward({ wf }) {
  const rows = Object.entries(wf?.strategies || {});
  if (!rows.length) return <Empty>Run tools/meta_brain_train.py to fill this table.</Empty>;
  return (
    <>
      <table className="tbl">
        <thead><tr><th>Strategy</th><th className="r">Trades</th><th className="r">Win</th><th className="r">Avg</th><th className="r">Total</th><th className="r hide-s">PF</th><th className="r hide-s">Worst DD</th></tr></thead>
        <tbody>{rows.map(([k, v]) => (
          <tr key={k} className={k.includes("+0.30R") ? "hl" : ""}><td>{k}</td><td className="r num">{v.n}</td>
            <td className="r num">{v.win != null ? `${Math.round(v.win * 100)}%` : "—"}</td>
            <td className={`r num ${tone(v.avgR)}`}>{signed(v.avgR, 3)}R</td><td className={`r num ${tone(v.totalR)}`}>{signed(v.totalR, 0)}R</td>
            <td className="r num hide-s">{num(v.pf)}</td><td className="r num hide-s">{v.maxDD_R}R</td></tr>))}</tbody>
      </table>
      <p className="help pad">Each month traded by a model trained only on earlier months; live stops, spread paid; 1R = the stop.
        {wf.period && ` ${wf.period[0].slice(0, 10)} → ${wf.period[1].slice(0, 10)}.`}
        {wf.top_features?.length > 0 && ` Strongest inputs: ${wf.top_features.slice(0, 8).join(", ")}.`}</p>
    </>
  );
}

function Scorecard({ board }) {
  const rows = (board || []).filter((r) => r.calls >= 20).sort((a, b) => b.hit - a.hit);
  if (!rows.length) return <Empty>Not enough live calls yet.</Empty>;
  return (
    <table className="tbl">
      <thead><tr><th>Indicator</th><th className="r">Calls</th><th className="r">Right</th><th className="hide-s" /></tr></thead>
      <tbody>{rows.map((r) => (
        <tr key={r.key}><td>{r.name}</td><td className="r num dim">{r.calls}</td><td className="r num">{Math.round(r.hit * 100)}%</td>
          <td className="hide-s"><div className="hit"><span style={{ width: `${r.hit * 100}%` }} className={r.hit >= 0.55 ? "ok" : r.hit < 0.45 ? "down" : "mid"} /><i /></div></td></tr>))}</tbody>
    </table>
  );
}

// ------------------------------------------------------------------ pages
function Overview({ d }) {
  return (
    <div className="grid g-ov">
      <div className="span-all"><EquityPanel account={d.account || {}} /></div>
      <div className="span-all"><RiskStrip d={d} /></div>
      <Panel title="Open positions" className="span-2" flush><Positions positions={d.blotter} symbols={d.symbols} /></Panel>
      <Panel title="Latest" className="span-1" flush><Feed items={d.feed} limit={7} /></Panel>
      <Panel title="Watchlist" className="span-2" flush><Watchlist symbols={d.symbols} /></Panel>
      <Panel title="Recent fills" className="span-1" flush><TradesTable trades={d.trades} limit={8} /></Panel>
    </div>
  );
}

function Strategies({ d }) {
  const s = d.strategies || {};
  const sf = s.sweep_fvg || {};
  return (
    <div className="grid g-ov">
      <Panel title="Gold Breakout" meta="Own strategy · live on demo" className="span-2"><GoldBreakout g={s.gold_breakout} /></Panel>
      <Panel title="SMC votes" className="span-1" flush><Shadow shadow={s.shadow} symbols={d.symbols} /></Panel>
      <Panel title="Gemma desk" meta="Gemma's own calls · live on demo" className="span-all"><GemmaDesk gt={s.gemma_trader} symbols={d.symbols} /></Panel>
      {s.jev_trader?.enabled && <Panel title="Jev desk" meta="Jev's own calls (probabilities) · separate positions and record" className="span-all"><GemmaDesk gt={s.jev_trader} symbols={d.symbols} who="Jev" /></Panel>}
      {s.qwen_trader?.enabled && <Panel title="Qwen desk" meta="Qwen's own calls · separate positions and record" className="span-all"><GemmaDesk gt={s.qwen_trader} symbols={d.symbols} who="Qwen" /></Panel>}
      <Panel title="Meta-brain" meta="All strategies combined · live on demo" className="span-all"><MetaBrain mt={s.meta_trader} pred={s.shadow?.meta_brain} symbols={d.symbols} /></Panel>
      <Panel title="Walk-forward test" meta="Meta-brain vs every strategy" className="span-all" flush><WalkForward wf={s.walk_forward} /></Panel>
      <Panel title="Sweep + FVG reversal" meta="Own rules, not traded" className="span-1">
        <dl className="facts one">{["M5", "M15"].map((tf) => (
          <div key={tf}><dt>{tf}</dt><dd className={`num ${tone(sf[tf]?.avgR)}`}>{sf[tf] ? `${signed(sf[tf].avgR)}R / trade` : "—"}</dd>
            <span className="sub">{sf[tf] ? `${sf[tf].n} trades · ${Math.round(sf[tf].win * 100)}% win · PF ${num(sf[tf].pf)}` : ""}</span></div>))}</dl>
      </Panel>
      <Panel title="Indicator hit rate" meta="Live calls, 20+ each" className="span-2 scroll" flush><Scorecard board={d.scorecard?.board} /></Panel>
    </div>
  );
}

const TRADE_RANGES = [["1d", "1D"], ["1w", "1W"], ["1m", "1M"], ["3m", "3M"], ["all", "All"]];

function Scout() {
  const [s, err] = usePoll("/api/scout", 60000);
  if (err) return <Panel title="New markets" flush><Empty>{err}</Empty></Panel>;
  if (!s) return <Panel title="New markets" flush><Empty>Loading…</Empty></Panel>;
  const res = { passed: "up", failed: "down", deferred: "dim", unavailable: "dim" };
  return (
    <Panel title="New markets" meta={`${s.markets.length} of ${s.max} markets · tests one a day (${s.schedule})`} flush>
      {s.added.length > 0 && (
        <table className="tbl">
          <thead><tr><th>Added</th><th>Market</th><th>Status</th><th className="r">Test avg</th><th className="r hide-s">Trades tested</th></tr></thead>
          <tbody>{s.added.map((a) => (
            <tr key={a.symbol}><td className="num dim">{(a.at || "").slice(0, 10)}</td><td className="strong">{a.symbol}</td>
              <td>{a.stage}</td><td className={`r num ${tone(a.avg_r)}`}>{a.avg_r != null ? `${a.avg_r >= 0 ? "+" : ""}${a.avg_r.toFixed(3)}R` : "—"}</td>
              <td className="r num hide-s">{a.trades ?? "—"}</td></tr>))}</tbody>
        </table>)}
      {s.tested.length ? (
        <table className="tbl">
          <thead><tr><th>Tested</th><th>Market</th><th>Result</th><th className="hide-s">Why</th><th className="r">Avg</th></tr></thead>
          <tbody>{s.tested.slice(0, 20).map((t) => (
            <tr key={t.symbol}><td className="num dim">{(t.at || "").slice(0, 10)}</td><td className="strong">{t.symbol}</td>
              <td className={res[t.result] || ""}>{t.result}</td><td className="dim hide-s">{t.why || "—"}</td>
              <td className={`r num ${tone(t.avg_r)}`}>{t.avg_r != null ? `${t.avg_r >= 0 ? "+" : ""}${t.avg_r.toFixed(3)}R` : "—"}</td></tr>))}</tbody>
        </table>) : <Empty>No market tested yet.</Empty>}
      {s.queue.length > 0 && <div className="dim" style={{ padding: "8px 12px" }}>Next up: {s.queue.join(", ")}</div>}
    </Panel>
  );
}

function Trades({ d }) {
  const [range, setRange] = useStored("cg.trades.range", "1w");
  const [who, setWho] = useState("all");
  const [h, err] = usePoll(`/api/trades?range=${range}`, 10000);
  const all = h?.trades || [];
  const rows = who === "all" ? all : all.filter((t) => t.who === who);
  const net = rows.reduce((a, t) => a + t.pnl, 0);
  const wins = rows.filter((t) => t.pnl > 0).length;
  const open = d.blotter || [];
  return (
    <div className="grid">
      <div className="strip">
        <div className="cell"><span className="lbl">Open now</span><span className="val num">{open.length}</span>
          <span className={`sub ${tone(d.account?.floating)}`}>{money(d.account?.floating, true)} floating</span></div>
        <div className="cell"><span className="lbl">Closed</span><span className="val num">{rows.length}</span><span className="sub">in range</span></div>
        <div className="cell"><span className="lbl">Win rate</span><span className="val num">{rows.length ? `${Math.round((wins / rows.length) * 100)}%` : "—"}</span>
          <span className="sub">{wins}W {rows.length - wins}L</span></div>
        <div className="cell"><span className="lbl">Net realized</span><span className={`val num ${tone(net)}`}>{money(net, true)}</span></div>
      </div>
      <Panel title="Open positions" meta={`${open.length} open`} className="span-2" flush><Positions positions={open} symbols={d.symbols} /></Panel>
      <Panel title="By strategy" meta="Tap one to filter" className="span-2" flush>
        <table className="tbl hover">
          <thead><tr><th>Strategy</th><th className="r">Trades</th><th className="r">Win rate</th><th className="r">Net</th></tr></thead>
          <tbody>
            {(h?.by_strategy || []).map((b) => (
              <tr key={b.who} onClick={() => setWho(who === b.who ? "all" : b.who)} className={who === b.who ? "sel" : ""} style={{ cursor: "pointer" }}>
                <td className="strong">{b.who}</td><td className="r num">{b.trades}</td>
                <td className="r num">{b.trades ? `${Math.round((b.wins / b.trades) * 100)}%` : "—"}</td>
                <td className={`r num ${tone(b.pnl)}`}>{money(b.pnl, true)}</td></tr>))}
          </tbody>
        </table>
      </Panel>
      <Panel title={who === "all" ? "Closed trades" : `Closed trades: ${who}`} className="span-2 scroll trades-hist" flush
        meta={<span style={{ display: "inline-flex", gap: 8, alignItems: "center" }}>
          {who !== "all" && <button className="link" onClick={() => setWho("all")}>Show all</button>}
          <Segmented value={range} options={TRADE_RANGES} onChange={setRange} label="Trade history range" /></span>}>
        {err ? <Empty>{err}</Empty> : !h ? <Empty>Loading…</Empty> : !rows.length ? <Empty>No closed trades in this range.</Empty> : (
          <table className="tbl">
            <thead><tr><th>Closed</th><th>Market</th><th>Side</th><th className="hide-s">By</th><th className="r">Size</th>
              <th className="r hide-s">Entry</th><th className="r">Exit</th><th className="r">P&amp;L</th></tr></thead>
            <tbody>{rows.map((t) => (
              <tr key={`${t.ticket}-${t.t}`}><td className="num dim">{stamp(t.t)}</td><td className="strong">{t.name}</td>
                <td>{t.side ? <Side side={t.side} /> : <span className="dim">—</span>}</td><td className="dim hide-s">{t.who}</td>
                <td className="r num">{num(t.volume)}</td><td className="r num hide-s">{t.entry ? px(t.entry) : "—"}</td>
                <td className="r num">{px(t.exit)}</td><td className={`r num strong ${tone(t.pnl)}`}>{money(t.pnl, true)}</td></tr>))}</tbody>
          </table>)}
      </Panel>
    </div>
  );
}

// ------------------------------------------------------------------ shell
function useTheme() {
  const [theme, setTheme] = useStored("cg-theme", "dark");
  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  return [theme, () => setTheme(theme === "dark" ? "light" : "dark")];
}

function useNow() {
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const id = setInterval(() => setNow(Date.now()), 1000); return () => clearInterval(id); }, []);
  return now;
}

// The full snapshot is heavy (logs, models); prices and open P&L come every second from /api/prices.
function withLive(d, live) {
  if (!d || !live) return d;
  const px = live.prices || {}, pos = live.positions || {};
  return {
    ...d,
    symbols: (d.symbols || []).map((s) => (px[s.symbol] ? { ...s, ...px[s.symbol] } : s)),
    blotter: (d.blotter || []).map((p) => (pos[p.ticket] ? { ...p, now: pos[p.ticket].now ?? p.now, pnl: pos[p.ticket].pnl } : p)),
    account: d.account ? { ...d.account, floating: live.floating ?? d.account.floating } : d.account,
  };
}

// ------------------------------------------------------------------ phone QR
function usePhoneQR(size) {
  const [link, setLink] = useState(null);
  const [svg, setSvg] = useState("");
  useEffect(() => {
    let alive = true;
    fetch("/api/phone", { cache: "no-store" }).then((r) => r.json()).then(async (j) => {
      if (!alive) return;
      setLink(j);
      if (j.url) setSvg(await QRCode.toString(j.url, { type: "svg", margin: size === "corner" ? 0 : 1, errorCorrectionLevel: size === "corner" ? "L" : "M", color: { dark: "#111317", light: "#ffffff" } }));
    }).catch(() => alive && setLink({ url: null, why: "Can't reach the dashboard server" }));
    return () => { alive = false; };
  }, [size]);
  return [link, svg];
}

// Always on screen (computer only): scan it to open this dashboard on the phone; click for the big version.
function PhoneCorner({ onOpen }) {
  const [link, svg] = usePhoneQR("corner");
  if (!link?.url || !svg) return null;
  return (
    <button className="qr-corner" onClick={onOpen} title={`Scan to open on your phone: ${link.url}`} aria-label="Phone QR code, click to enlarge">
      <span className="qr-corner-code" dangerouslySetInnerHTML={{ __html: svg }} />
      <span className="qr-corner-l">Phone</span>
    </button>
  );
}

function PhoneQR({ onClose }) {
  const [link, svg] = usePhoneQR("big");
  useEffect(() => {
    const esc = (e) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="modal-bg" onClick={onClose}>
      <div className="modal" role="dialog" aria-label="Open on your phone" onClick={(e) => e.stopPropagation()}>
        <div className="modal-h"><h2>Open on your phone</h2><button className="icon-btn" onClick={onClose} aria-label="Close"><X size={16} /></button></div>
        {!link ? <Empty>Loading…</Empty> : !link.url ? <div className="alert bad">{link.why}</div> : (<>
          <div className="qr" dangerouslySetInnerHTML={{ __html: svg }} />
          <ol className="qr-steps">
            <li>Open the <b>Tailscale</b> app on your phone and make sure it's connected (same account as this Mac).</li>
            <li>Point the phone camera at the code and tap the link.</li>
          </ol>
          <div className="qr-url num">{link.url}</div>
          <p className="dim small">Private: only your own Tailscale devices can open it.</p>
        </>)}
      </div>
    </div>
  );
}

export default function Terminal() {
  const [raw, err] = usePoll("/api/state", 3000);
  const [live] = usePoll("/api/prices", 1000);
  const d = useMemo(() => withLive(raw, live), [raw, live]);
  const [phoneOpen, setPhoneOpen] = useState(false);
  const onPhone = location.hostname.endsWith(".ts.net");
  const [theme, toggle] = useTheme();
  const now = useNow();
  const valid = NAV.map((n) => n[0]);
  const fromHash = () => { const h = (location.hash || "").slice(1); return valid.includes(h) ? h : "overview"; };
  const [page, setPage] = useState(fromHash);
  useEffect(() => {
    const on = () => setPage(fromHash());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  const go = (p) => { if (p !== page) { location.hash = p; window.scrollTo(0, 0); } };
  const b = d?.bot || {};
  const next = b.next_check_s != null && d ? Math.max(0, b.next_check_s - Math.floor((now - d.ts * 1000) / 1000)) : null;
  const title = NAV.find((n) => n[0] === page)[1];

  let body = <Empty>Connecting…</Empty>;
  if (page === "account") body = <Account />;
  else if (page === "pipeline") body = <Pipeline />;
  else if (page === "brains") body = <Brains />;
  else if (d) {
    body = page === "markets" ? (
        <div className="grid">
          <Panel title="Markets" meta="Council signal vs. trade threshold" className="span-2" flush><Watchlist symbols={d.symbols} full /></Panel>
          <div className="span-2"><Scout /></div>
        </div>)
      : page === "strategies" ? <Strategies d={d} />
      : page === "trades" ? <Trades d={d} />
      : page === "activity" ? (
        <div className="grid g-ov">
          <Panel title="Bot activity" className="span-2" flush><Feed items={d.feed} limit={80} /></Panel>
          <Panel title="Automatic changes" className="span-1" flush><Feed items={(d.changes || []).map((c) => ({ t: c.t, title: c.kind, text: c.text }))} limit={40} /></Panel>
        </div>)
      : <Overview d={d} />;
  }

  return (
    <div className="app">
      <nav className="rail" aria-label="Sections">
        <div className="mark" aria-hidden="true"><span /></div>
        {NAV.map(([id, label, Icon]) => (
          <button key={id} className={page === id ? "on" : ""} onClick={() => go(id)} aria-current={page === id ? "page" : undefined}>
            <Icon size={18} strokeWidth={1.75} /><span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="main">
        <header className="top">
          <div className="top-l">
            <h1>{title}</h1>
            {d && <Badge kind={d.account_mode === "live" ? "warn" : "info"}>{d.account_mode === "live" ? "Live" : "Demo"}</Badge>}
          </div>
          <div className="top-r">
            <span className={`st ${b.running ? "ok" : "bad"}`}><i />Bot {b.running ? "running" : "stopped"}</span>
            <span className={`st ${b.broker_ok ? "ok" : "bad"}`} title={b.broker_err || ""}><i />Broker {b.broker_ok ? "connected" : "offline"}</span>
            {next != null && <span className="st plain hide-s">Next check <b className="num">{Math.floor(next / 60)}:{String(next % 60).padStart(2, "0")}</b></span>}
            <span className="st plain hide-s num">{new Date(now).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span>
            {!onPhone && <button className="theme-btn hide-s" onClick={() => setPhoneOpen(true)} title="Open this dashboard on your phone" aria-label="Show phone QR code">
              <Smartphone size={15} strokeWidth={1.75} /><span>Phone</span>
            </button>}
            <button className="theme-btn" onClick={toggle} title={theme === "dark" ? "Switch to light" : "Switch to dark"} aria-label="Switch light or dark theme">
              {theme === "dark" ? <Sun size={15} strokeWidth={1.75} /> : <Moon size={15} strokeWidth={1.75} />}
              <span>{theme === "dark" ? "Light" : "Dark"}</span>
            </button>
          </div>
        </header>
        {phoneOpen && <PhoneQR onClose={() => setPhoneOpen(false)} />}
        {!onPhone && <PhoneCorner onOpen={() => setPhoneOpen(true)} />}
        {err && <div className="alert bad banner">{err}{d ? " · showing the last data received" : ""}</div>}
        <main className="content">{body}</main>
      </div>
    </div>
  );
}
