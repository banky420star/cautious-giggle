import React, { useEffect, useState } from "react";

// ------------------------------------------------------------------ formatting
const nf = (d) => new Intl.NumberFormat(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const N2 = nf(2);
export const money = (v, sign = false) => {
  if (v == null || Number.isNaN(+v)) return "—";
  const s = N2.format(Math.abs(v));
  if (sign) return `${v > 0 ? "+" : v < 0 ? "−" : ""}$${s}`;
  return `${v < 0 ? "−" : ""}$${s}`;
};
export const moneyShort = (v) => {
  if (v == null) return "—";
  const a = Math.abs(v);
  const s = a >= 1e6 ? `${(a / 1e6).toFixed(2)}M` : a >= 1e4 ? `${(a / 1e3).toFixed(1)}k`
    : a >= 100 || Number.isInteger(a) ? Math.round(a).toLocaleString() : a.toFixed(2);
  return `${v < 0 ? "−" : ""}$${s}`;
};
export const pct = (v, d = 2, sign = true) =>
  v == null || Number.isNaN(+v) ? "—" : `${sign ? (v > 0 ? "+" : v < 0 ? "−" : "") : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}%`;
export const px = (v) => {
  if (v == null) return "—";
  const d = v >= 1000 ? 2 : v >= 10 ? 3 : 5;
  return nf(d).format(v);
};
export const num = (v, d = 2) => (v == null || Number.isNaN(+v) ? "—" : (+v).toFixed(d));
export const signed = (v, d = 2) => (v == null || Number.isNaN(+v) ? "—" : `${v > 0 ? "+" : v < 0 ? "−" : ""}${Math.abs(v).toFixed(d)}`);
export const tone = (v) => (v == null || Math.abs(v) < 1e-9 ? "" : v > 0 ? "up" : "down");
export const clock = (ms) => new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
export const stamp = (ms) => new Date(ms).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });

// ------------------------------------------------------------------ data
export const api = (path, body) => fetch(path, {
  method: body ? "POST" : "GET", cache: "no-store",
  headers: body ? { "Content-Type": "application/json", "X-CG-Client": "dashboard" } : {},
  body: body ? JSON.stringify(body) : undefined,
}).then((r) => r.json());

export function usePoll(url, ms) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    const tick = async () => {
      try {
        const j = await api(url);
        if (!alive) return;
        if (j.error) setErr(j.error); else { setData(j); setErr(null); }
      } catch { if (alive) setErr("Can't reach the dashboard server"); }
      if (alive) setLoading(false);
    };
    tick();
    const id = setInterval(tick, ms);
    return () => { alive = false; clearInterval(id); };
  }, [url, ms]);
  return [data, err, loading];
}

export function useStored(key, init) {
  const [v, setV] = useState(() => { try { return localStorage.getItem(key) ?? init; } catch { return init; } });
  useEffect(() => { try { localStorage.setItem(key, v); } catch { /* private mode */ } }, [key, v]);
  return [v, setV];
}

// ------------------------------------------------------------------ parts
export const Panel = ({ title, meta, actions, children, className = "", flush = false }) => (
  <section className={`panel ${className}`}>
    {(title || actions || meta) && (
      <header className="panel-h">
        <h2>{title}</h2>
        {meta && <span className="panel-meta">{meta}</span>}
        {actions && <div className="panel-act">{actions}</div>}
      </header>
    )}
    <div className={flush ? "panel-b flush" : "panel-b"}>{children}</div>
  </section>
);

export const Badge = ({ kind = "", children, title }) => <span className={`badge ${kind}`} title={title}>{children}</span>;
export const Empty = ({ children }) => <div className="empty">{children}</div>;
export const Side = ({ side }) => <span className={`side ${side === "BUY" ? "up" : "down"}`}>{side === "BUY" ? "Long" : "Short"}</span>;

export function Segmented({ value, options, onChange, label }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([v, l]) => (
        <button key={v} className={v === value ? "on" : ""} onClick={() => onChange(v)} aria-pressed={v === value}>{l}</button>
      ))}
    </div>
  );
}

export function Spark({ candles, w = 88, h = 24 }) {
  if (!candles || candles.length < 2) return <svg width={w} height={h} />;
  const c = candles.map((k) => k[4]);
  const lo = Math.min(...c), hi = Math.max(...c), r = hi - lo || 1;
  const d = c.map((v, i) => `${i ? "L" : "M"}${((i / (c.length - 1)) * w).toFixed(1)},${(h - 2 - ((v - lo) / r) * (h - 4)).toFixed(1)}`).join(" ");
  return <svg width={w} height={h} className={`spark ${c[c.length - 1] >= c[0] ? "up" : "down"}`} aria-hidden="true"><path d={d} /></svg>;
}

export function SignalMeter({ value, threshold }) {
  const v = Math.max(-1, Math.min(1, value || 0));
  const t = Math.min(1, Math.abs(threshold || 0));
  const hit = Math.abs(v) >= t && t > 0;
  return (
    <div className="meter-sig" title={`Signal ${signed(value)} · trades at ±${num(threshold)}`}>
      <div className="track">
        <span className="mid" />
        <span className="th" style={{ left: `${50 - t * 50}%` }} />
        <span className="th" style={{ left: `${50 + t * 50}%` }} />
        <span className={`fill ${v >= 0 ? "up" : "down"} ${hit ? "hit" : ""}`}
          style={v >= 0 ? { left: "50%", width: `${v * 50}%` } : { left: `${50 + v * 50}%`, width: `${-v * 50}%` }} />
      </div>
      <span className={`num ${hit ? tone(v) : "dim"}`}>{signed(v)}</span>
    </div>
  );
}
