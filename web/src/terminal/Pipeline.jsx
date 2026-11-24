import React from "react";
import { RefreshCw } from "lucide-react";
import { Empty, money, tone, usePoll } from "./ui.jsx";

const ago = (s) => (s == null ? "—" : s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`);
const NAME = { XAUUSDm: "Gold", BTCUSDm: "Bitcoin", EURUSDm: "EUR/USD", USDJPYm: "USD/JPY", XAGUSDm: "Silver", ETHUSDm: "Ether", USOILm: "Oil" };
const nm = (s) => NAME[s] || s;

function Stat({ label, value, cls = "" }) {
  return <div className="pl-stat"><span className="lbl">{label}</span><span className={`num ${cls}`}>{value}</span></div>;
}

function Stage({ n, title, status, what, children }) {
  const st = { ok: ["ok", "Running"], warn: ["warn", "Check"], off: ["off", "Waiting"], bad: ["bad", "Problem"] }[status] || ["off", status];
  return (
    <li className="pl-stage">
      <div className="pl-rail" aria-hidden="true"><span className={`pl-dot ${st[0]}`}>{n}</span></div>
      <section className="pl-card">
        <header><h3>{title}</h3><span className={`pl-pill ${st[0]}`}>{st[1]}</span></header>
        <p className="pl-what">{what}</p>
        {children && <div className="pl-stats">{children}</div>}
      </section>
    </li>
  );
}

export default function Pipeline() {
  const [p, err] = usePoll("/api/pipeline", 15000);
  if (err) return <Empty>{err}</Empty>;
  if (!p) return <Empty>Loading the pipeline…</Empty>;
  const L = p.loop, S = p.signals, F = p.filters, P = p.protect, B = p.broker, R = p.learn;
  const loopStatus = L.oldest_decision_s == null ? "off" : L.oldest_decision_s > 900 ? "bad" : L.oldest_decision_s > 420 ? "warn" : "ok";
  const blocked = F.trend_block + F.council_off + F.setup_block + F.size_block;
  const llmBad = Object.values(S.llm || {}).some((x) => x.last_s == null || x.last_s > 1800);
  return (
    <div className="pipeline">
      <div className={`pl-health ${loopStatus}`}>
        <RefreshCw size={15} aria-hidden="true" />
        <span>
          {loopStatus === "ok" ? "The loop is running" : loopStatus === "warn" ? "The loop is slow" : loopStatus === "bad" ? "The loop looks stuck" : "Waiting for the bot"}
          {" · "}every market is re-checked each 5-minute candle · last decision {ago(L.newest_decision_s)}
          {" · "}{L.errors_1h} errors, {L.warnings_1h} warnings in the last hour
        </span>
      </div>
      <ol className="pl-list">
        <Stage n={1} title="Prices in" status={loopStatus}
          what="Each 5-minute candle the bot pulls fresh prices and candles for every market from Exness. The broker allows 2 requests a second, shared with this dashboard, so prices are cached and reused.">
          <Stat label="Markets" value={L.markets} />
          <Stat label="Decisions, last hour" value={L.decisions_1h} />
          <Stat label="Slowest market" value={ago(L.oldest_decision_s)} cls={L.oldest_decision_s > 420 ? "down" : ""} />
          <Stat label="Bot restarted" value={ago(L.restarted_s)} />
        </Stage>

        <Stage n={2} title="Signals" status={S.ppo_markets.length ? (llmBad ? "warn" : "ok") : "warn"}
          what="Every market is scored by several brains. The indicator council: 48 indicators on 4 timeframes plus your models (gold order blocks, Kronos, LightGBM, the news reader, Gemma's SMC read), each weighted by how often it has been right lately. PPO: a reinforcement-learning model per market. The meta-brain: one model trained on every strategy. The LSTM only counts on markets where it has proven an edge on unseen data. Gemma (local) and Jev (TypeSafe, via OpenRouter) each answer every market once per 15-minute bar.">
          <Stat label="PPO models" value={`${S.ppo_markets.length} of ${L.markets || S.lstm_markets}`} />
          <Stat label="LSTM proven" value={S.lstm_proven.length ? S.lstm_proven.map(nm).join(", ") : `none of ${S.lstm_markets} (ignored)`} cls={S.lstm_proven.length ? "up" : "dim"} />
          <Stat label="LSTM checked" value={S.lstm_checked ? S.lstm_checked.slice(0, 16).replace("T", " ") + " UTC" : "—"} />
          {Object.entries(S.llm || {}).map(([k, v]) => (
            <Stat key={k} label={`${k} answered`} value={`${v.answered} of 7 · ${ago(v.last_s)}`} cls={v.last_s == null || v.last_s > 1800 ? "down" : ""} />))}
        </Stage>

        <Stage n={3} title="Decision and filters" status="ok"
          what="The signals are blended into one number per market. Before the council bot may open a trade it must pass: the 4-hour 200-EMA trend has to agree; markets where no setup made money after the spread are off (EUR/USD, USD/JPY, oil); setups with a losing record under the current rules are blocked; the smallest lot must not risk more than the cap; and the entry waits up to 15 minutes for a small pullback. Gemma, Jev and the meta-brain open nothing from 30 minutes before to 30 minutes after high-impact news (10 for medium) that moves the market, or from 15 minutes before to 30 minutes after the London and New York opens.">
          <Stat label="Against the trend" value={F.trend_block} />
          <Stat label="Market off" value={F.council_off} />
          <Stat label="Losing setup" value={F.setup_block} />
          <Stat label="Too big for the cap" value={F.size_block} />
          <Stat label="Waiting for pullback" value={F.entry_plans} />
          <Stat label="News / market open" value={F.news_open_block ?? 0} />
          <Stat label="Entered" value={F.entries} cls="up" />
        </Stage>

        <Stage n={4} title="Traders" status="ok"
          what="Each trader acts on its own, with its own positions and record, so you can see which one earns its keep. The council bot trades the blended signal (8 x ATR stop, 12.8 x ATR target, 24 h). Gold Breakout trades the Asian-range break on gold. The meta-brain trades its own predictions. Gemma (a local AI model) and Jev (TypeSafe's decision model) read the whole desk (every signal, the account, all positions, market opens, news, their own record) and make their own call every 15 minutes; Jev answers with probabilities and trades only when one side is at least 60% likely.">
          <table className="tbl pl-tbl">
            <thead><tr><th>Trader</th><th className="r">Open</th><th className="r">Floating</th><th className="r">Closed today</th><th className="r">Today</th></tr></thead>
            <tbody>{p.traders.map((t) => (
              <tr key={t.who}><td className="strong">{t.who}</td><td className="r num">{t.open}</td>
                <td className={`r num ${tone(t.floating)}`}>{t.open ? money(t.floating, true) : "—"}</td>
                <td className="r num">{t.closed_today ? `${t.closed_today} (${t.wins_today}W)` : "0"}</td>
                <td className={`r num ${tone(t.pnl_today)}`}>{money(t.pnl_today, true)}</td></tr>))}</tbody>
          </table>
        </Stage>

        <Stage n={5} title="Protection" status={P.halted ? "bad" : P.used_pct > 70 ? "warn" : "ok"}
          what="Every 10 seconds the stop manager checks every bot trade and moves its stop into profit using that market's tested rule; it never touches trades you opened by hand. Council trades still open after 24 hours are closed. The risk guard stops new trades for the day once the daily loss stop or the drawdown limit is reached. A watchdog restarts the bot if its loop stops for 10 minutes.">
          <Stat label="Stops moved, last hour" value={P.stop_moves_1h} />
          <Stat label="24 h time exits" value={P.time_exits_1h} />
          <Stat label="Daily loss used" value={P.used_pct == null ? "loading…" : `${P.used_pct}% of ${money(P.daily_limit)}`} cls={P.used_pct > 70 ? "down" : ""} />
          <Stat label="Today" value={money(P.today, true)} cls={tone(P.today)} />
          <Stat label="Risk halt" value={P.halted ? "ON: no new trades" : "off"} cls={P.halted ? "down" : ""} />
          {P.watchdog_1h > 0 && <Stat label="Watchdog restarts, last hour" value={P.watchdog_1h} cls="down" />}
        </Stage>

        <Stage n={6} title="Broker" status="ok"
          what="Orders go to Exness with the stop and target attached straight away. Each order is logged with which trader sent it, so every position on the Trades page shows its owner.">
          <Stat label="Orders, last hour" value={B.orders_1h} />
          <Stat label="Open positions" value={B.open} />
          <Stat label="Equity" value={money(B.equity)} />
          <Stat label="Floating" value={money(B.floating, true)} cls={tone(B.floating)} />
        </Stage>

        <Stage n={7} title="Learning" status="ok"
          what="Results flow back into step 2. Model memory re-scores every indicator after seeing what price did. The setup gate scores each setup. Gemma and Jev read their own record. Graduation retrains a market's PPO and LSTM when its results slip. Each day the LSTM check decides which LSTMs may count, and the market scout tests a new market, adding it if it passes (up to 12).">
          <Stat label="Graduation" value={Object.entries(R.graduation || {}).map(([k, v]) => `${nm(k)} ${v.replace("_", " ")}`).join(" · ") || "—"} />
          <Stat label="Markets" value={`${R.scout_markets || L.markets} of ${R.scout_max}`} />
          <Stat label="Next market to test" value={R.scout_next || "—"} />
          <Stat label="LSTM retrain" value={R.lstm_retrain || "none running"} />
        </Stage>
      </ol>
      <div className="pl-loop"><RefreshCw size={14} aria-hidden="true" /> Back to step 1 on the next 5-minute candle, now with what was just learned.</div>
      {(L.errors.length > 0 || L.warnings.length > 0) && (
        <div className="pl-issues">
          <h3>Last problems logged</h3>
          <ul>{[...L.errors, ...L.warnings].map((x, i) => <li key={i} className="num">{x}</li>)}</ul>
        </div>)}
    </div>
  );
}
