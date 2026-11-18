import React, { useEffect, useMemo, useState } from "react";

const STATUS_TONE = { FAIL: "red", WARN: "amber", PASS: "green" };
const SECTION_NAME = {
  feed: "Price feed", live: "Live prices", council: "Indicator council", kronos: "Kronos", ob_gold: "Order-block gold",
  catboost: "LightGBM voter", news: "News", signal: "Decisions", filter: "Big-move filter", learning: "Learning",
  lstm: "LSTM", ppo: "PPO",
};
const MODEL_NAME = { lstm: "LSTM", ppo: "PPO", council: "Council", blend: "Final vote", trade: "Bot's trades" };
const ago = (iso) => {
  if (!iso) return "—";
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  return s < 90 ? `${Math.round(s)}s ago` : s < 5400 ? `${Math.round(s / 60)} min ago` : `${Math.round(s / 3600)} h ago`;
};
const bps = (g) => (g == null ? <span className="muted">—</span> :
  <span className={g.bps >= 0 ? "pos" : "neg"} title={`${g.n} calls${g.hit != null ? `, right ${Math.round(g.hit * 100)}%` : ""}`}>
    {g.bps >= 0 ? "+" : ""}{g.bps.toFixed(2)}</span>);

/* ------------------------------------------------------------------ learning */
export function Learning({ data }) {
  const [open, setOpen] = useState(null);
  if (!data) return <div className="card hud lrn"><div className="muted small">Waiting for the first learning snapshot…</div></div>;
  if (data.error) return <div className="card hud lrn"><div className="neg small">Learning view failed: {data.error}</div></div>;
  const secs = data.audit?.sections || [];
  const tot = secs.reduce((a, s) => ({ f: a.f + s.fail, w: a.w + s.warn, p: a.p + s.pass }), { f: 0, w: 0, p: 0 });
  return (
    <div className="lrn-wrap">
      <div className="card hud lrn">
        <div className="card-head"><b>Where each lesson goes</b>
          <span className="muted small">what every learner learns from, and the part of the bot that uses it</span></div>
        <div className="lrn-layers">
          {data.layers.map((l) => (
            <div key={l.learner} className="lrn-layer">
              <span className={`op-tag ${l.live ? "green" : "amber"}`}>{l.live ? "IN USE" : "WATCHING"}</span>
              <b>{l.learner}</b>
              <span className="muted small">{l.learns_from}</span>
              <span className="lrn-arrow">→</span>
              <span className="small">{l.feeds}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="card hud lrn">
        <div className="card-head"><b>Per market</b>
          <span className="muted small">real trade record and the size it allows · big-move filter now · what each model has
            earned per call after costs (bps), in all hours and split by GO / STOP · the PPO trial</span></div>
        <div className="tw">
          <table className="lrn-table">
            <thead><tr>
              <th>Market</th><th>Real record</th><th>Size</th><th>Filter</th>
              {Object.keys(MODEL_NAME).map((m) => <th key={m} className="n">{MODEL_NAME[m]}<div className="muted">all · GO · STOP</div></th>)}
              <th>PPO trial</th>
            </tr></thead>
            <tbody>
              {data.markets.map((m) => (
                <tr key={m.symbol}>
                  <td><b>{m.name}</b></td>
                  <td className="small">{m.record.trades} trades · {m.record.win_rate.toFixed(0)}% won ·{" "}
                    <span className={m.record.pnl >= 0 ? "pos" : "neg"}>{m.record.pnl >= 0 ? "+" : "−"}${Math.abs(m.record.pnl).toFixed(2)}</span>
                    <div className="muted">profit factor {m.record.profit_factor.toFixed(2)}</div></td>
                  <td><span className={`op-tag ${m.size.scale < 1 ? "amber" : "green"}`}>{m.size.scale < 1 ? `×${m.size.scale}` : "NORMAL"}</span>
                    <div className="muted small">{m.size.why}</div></td>
                  <td>{m.filter ? <><span className={`op-tag ${m.filter.go ? "cyan" : "grey"}`}>{m.filter.go ? "GO" : "STOP"}</span>
                    <div className="muted small">{Math.round(m.filter.p * 100)}% · {ago(m.filter.ts)}</div></> : <span className="muted small">no reading</span>}</td>
                  {Object.keys(MODEL_NAME).map((k) => (
                    <td key={k} className="n mono small">{bps(m.grades[k].all)} · {bps(m.grades[k].go)} · {bps(m.grades[k].stop)}</td>
                  ))}
                  <td className="small">{m.trial.canary ? <>
                    <span className="mono">{m.trial.trades ?? 0}/{m.trial.min_trades ?? "?"}</span> trades ·{" "}
                    <span className={(m.trial.pnl || 0) >= 0 ? "pos" : "neg"}>{(m.trial.pnl || 0) >= 0 ? "+" : "−"}${Math.abs(m.trial.pnl || 0).toFixed(2)}</span>
                    <div className="muted">{m.trial.champion ? `champion ${m.trial.champion}` : "no champion yet"} · checked {ago(m.trial.checked)}</div>
                  </> : <span className="muted">no trial running</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card hud lrn">
        <div className="card-head"><b>Audit by section</b>
          <span className="muted small">{tot.f} fail · {tot.w} warn · {tot.p} pass · rerun every 15 min · last {data.audit?.t ? ago(new Date(data.audit.t * 1000).toISOString()) : "never"}</span></div>
        <div className="lrn-secs">
          {secs.map((s) => (
            <div key={s.section} className="lrn-sec">
              <button type="button" className="lrn-sec-head" onClick={() => setOpen(open === s.section ? null : s.section)} aria-expanded={open === s.section}>
                <span className={`lrn-dot ${s.fail ? "red" : s.warn ? "amber" : "green"}`} />
                <b>{SECTION_NAME[s.section] || s.section}</b>
                <span className="muted small">{s.fail ? `${s.fail} fail · ` : ""}{s.warn ? `${s.warn} warn · ` : ""}{s.pass} pass</span>
              </button>
              {open === s.section && s.rows.map((r, i) => (
                <div key={i} className="lrn-row">
                  <span className={`op-tag ${STATUS_TONE[r.status]}`}>{r.status}</span>
                  <span className="mono small">{r.market === "-" ? "all" : r.market.replace(/m$/, "")}</span>
                  <span className="small">{r.what}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ indicator map */
function colour(v, mode) {
  if (mode === "big") {
    const t = Math.max(0, Math.min(1, v / 0.2));
    return `rgba(167,139,250,${0.06 + 0.9 * t})`;
  }
  const t = Math.max(-1, Math.min(1, v / 0.08));
  return t >= 0 ? `rgba(52,211,153,${0.06 + 0.9 * t})` : `rgba(248,113,113,${0.06 - 0.9 * t})`;
}

function Timeline({ m }) {
  const [k, setK] = useState(null);
  const W = 760, H = 240, L = 56, R = 10, T = 10, B = 26;
  const p = m.p, n = p.length;
  if (!n) return null;
  const lo = Math.min(...p), hi = Math.max(...p), pad = (hi - lo) * 0.06 || 1;
  const y = (v) => T + (H - T - B) * (1 - (v - lo + pad) / (hi - lo + 2 * pad));
  const x = (i) => L + (W - L - R) * (i / Math.max(1, n - 1));
  const idx = {};
  m.t.forEach((t, i) => { idx[t] = i; });
  const at = (t) => {
    const q = t.slice(0, 11) + String(Math.floor(+t.slice(14, 16) / 15) * 15).padStart(2, "0");
    if (q in idx) return idx[q];
    const j = m.t.findIndex((u) => u >= q);
    return j < 0 ? n - 1 : j;
  };
  const dp = hi > 100 ? 0 : hi > 10 ? 2 : 4;
  const ticks = [0, 0.5, 1].map((q) => lo - pad + (hi - lo + 2 * pad) * q);
  const days = m.t.map((t, i) => [t, i]).filter(([t]) => t.endsWith("00:00"));
  const sel = k == null ? null : m.events[k];
  const mx = m.flagcount.length ? m.flagcount[0][1] : 1;
  return (
    <div className="imap-tl">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${m.name} price, last 10 days, big moves marked`}>
        {ticks.map((v, i) => <g key={i}><line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="rgba(255,255,255,.07)" />
          <text x={L - 6} y={y(v) + 3} textAnchor="end" className="imap-ax">{v.toFixed(dp)}</text></g>)}
        {days.map(([t, i]) => <text key={t} x={x(i)} y={H - 8} textAnchor="middle" className="imap-ax">{t.slice(0, 5)}</text>)}
        <path d={p.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("")} fill="none" stroke="var(--cyan)" strokeWidth="1.4" />
        {m.events.map((e, i) => {
          const j = at(e.t);
          return <circle key={i} cx={x(j)} cy={y(p[j])} r={k === i ? 6.5 : 4.5} fill={e.dir === "up" ? "var(--pos)" : "var(--neg)"}
            stroke="var(--bg)" strokeWidth="1.5" style={{ cursor: "pointer" }} tabIndex={0} role="button"
            aria-label={`${e.dir} ${e.mv} bps at ${e.t}`} onClick={() => setK(i)}
            onKeyDown={(ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); setK(i); } }} />;
        })}
      </svg>
      <div className="imap-insp">
        {sel ? <>
          <div><span className={`op-tag ${sel.dir === "up" ? "green" : "red"}`}>{sel.dir.toUpperCase()}</span>{" "}
            <b>{sel.mv > 0 ? "+" : ""}{sel.mv} bps</b> <span className="muted small">next hour · {sel.t} UTC</span></div>
          <div className="muted small">indicators furthest from their normal level just before (z-score)</div>
          {sel.top.map((t) => <div key={t.f} className="imap-bar"><span className="mono small">{t.f}</span>
            <i style={{ width: `${Math.min(100, Math.abs(t.z) * 20)}%` }} /><span className="mono small">{t.z > 0 ? "+" : ""}{t.z}</span></div>)}
          <button type="button" className="imap-back" onClick={() => setK(null)}>Show the tally for all moves</button>
        </> : <>
          <div><b>Most often unusual before big moves</b></div>
          <div className="muted small">{m.events.length} moves in 10 days. Click a dot for one move. Slow daily indicators stay
            unusual for days, so they top this list without timing anything.</div>
          {m.flagcount.map(([f, c]) => <div key={f} className="imap-bar"><span className="mono small">{f}</span>
            <i style={{ width: `${(100 * c) / mx}%` }} /><span className="mono small">{c}×</span></div>)}
        </>}
      </div>
    </div>
  );
}

export function IndicatorMap() {
  const [d, setD] = useState(null);
  const [err, setErr] = useState(null);
  const [mode, setMode] = useState("dir");
  const [mk, setMk] = useState(null);
  useEffect(() => {
    fetch("/api/indicator_map").then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then((j) => { setD(j); setMk(Object.keys(j.markets)[0]); }).catch((e) => setErr(String(e.message || e)));
  }, []);
  const MK = d ? Object.keys(d.markets) : [];
  const feats = useMemo(() => (d ? d.markets[MK[0]].features.map((f) => f.f) : []), [d]);
  const byM = useMemo(() => (d ? Object.fromEntries(MK.map((s) => [s, Object.fromEntries(d.markets[s].features.map((f) => [f.f, f]))])) : {}), [d]);
  if (err) return <div className="card hud lrn"><div className="neg small">Indicator map not available ({err}). Run tools/indicator_map.py.</div></div>;
  if (!d) return <div className="card hud lrn"><div className="muted small">Loading the indicator map…</div></div>;
  const holds = MK.reduce((a, s) => a + d.markets[s].features.filter((f) => f.holds).length, 0);
  const combos = MK.reduce((a, s) => a + d.markets[s].combos.hold_test, 0);
  const tested = MK.reduce((a, s) => a + d.markets[s].combos.tested, 0);
  const val = (f) => (mode === "big" ? f.big_test : mode === "tr" ? f.dir_train : f.dir_test);
  let lastG = "";
  return (
    <div className="lrn-wrap">
      <div className="imap-verdict">
        <div className="card hud"><b className="neg mono">{holds} of {feats.length * MK.length}</b>
          <span className="muted small">indicator–market pairs that called the next hour's direction on both halves of the year</span></div>
        <div className="card hud"><b className="neg mono">{combos} of {tested.toLocaleString()}</b>
          <span className="muted small">two-indicator combinations that held on unseen data</span></div>
        <div className="card hud"><b className="mono" style={{ color: "var(--violet)" }}>size, not direction</b>
          <span className="muted small">volume, time of day and distance from the averages warn of big moves; the big-move filter is built on them</span></div>
      </div>

      <div className="card hud lrn">
        <div className="card-head"><b>Every indicator on every market</b>
          <div className="tabs" role="group" aria-label="Colour by">
            {[["dir", "Direction · unseen 30%"], ["big", "Big-move warning · unseen"], ["tr", "Direction · learning 70%"]].map(([k, l]) =>
              <button key={k} type="button" className={mode === k ? "active" : ""} aria-pressed={mode === k} onClick={() => setMode(k)}>{l}</button>)}
          </div></div>
        <div className="muted small">{mode === "big" ? "Purple = the indicator was unusual before large moves in either direction (AUC − 0.5)."
          : mode === "tr" ? "The same measure on the learning 70%. Colour here that fades on the unseen 30% was chance."
          : `Green = higher value, market rose next hour; red = it fell (rank correlation, scale ±0.08). An outlined box would mark a signal past the strict bar (z ≥ ${d.z_bar}) on both halves; none qualifies.`}</div>
        <div className="imap-heat">
          <div className="imap-grid" style={{ gridTemplateColumns: `minmax(140px,1.4fr) repeat(${MK.length}, minmax(48px,1fr))` }}>
            <div className="imap-hd">Indicator</div>
            {MK.map((s) => <div key={s} className="imap-hd">{d.markets[s].name}</div>)}
            {feats.map((f) => {
              const g = byM[MK[0]][f].g;
              const first = g !== lastG;
              lastG = g;
              return (
                <React.Fragment key={f}>
                  <div className={`imap-fn${first ? " grp" : ""}`} title={f}>{f}</div>
                  {MK.map((s) => {
                    const x = byM[s][f];
                    return <div key={s} className={`imap-c${x && x.holds && mode !== "big" ? " hold" : ""}`}
                      style={{ background: x ? colour(val(x), mode) : "transparent" }}
                      title={x ? `${d.markets[s].name} · ${f}\ndirection: learning ${x.dir_train.toFixed(3)} → unseen ${x.dir_test.toFixed(3)}\nbig-move warning: ${x.big_test.toFixed(3)}` : ""} />;
                  })}
                </React.Fragment>
              );
            })}
          </div>
        </div>
      </div>

      <div className="card hud lrn">
        <div className="card-head"><b>Big moves in the last 10 days, and what was unusual just before</b>
          <div className="tabs" role="group" aria-label="Market">
            {MK.map((s) => <button key={s} type="button" className={mk === s ? "active" : ""} aria-pressed={mk === s} onClick={() => setMk(s)}>{d.markets[s].name}</button>)}
          </div></div>
        {mk && <Timeline key={mk} m={d.markets[mk]} />}
        <div className="muted small">Big move = a 1-hour move more than twice the usual hour ({d.markets[mk]?.up_pct}% of hours up, {d.markets[mk]?.dn_pct}% down) ·
          unseen data from {d.markets[mk]?.test_from}</div>
      </div>
    </div>
  );
}
