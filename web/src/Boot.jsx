import React, { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, KeyRound, Loader2, ShieldCheck, X } from "lucide-react";

/* ================================================================ logo */
export function Logo({ size = 40, spin = true }) {
  return (
    <svg width={size} height={size} viewBox="0 0 64 64" className={`logo-svg ${spin ? "spin" : ""}`} aria-label="Cautious Giggle">
      <defs>
        <linearGradient id="lg-bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0f1a2b" /><stop offset="1" stopColor="#060a12" />
        </linearGradient>
        <radialGradient id="lg-sweep" cx="32" cy="32" r="26" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#34d399" stopOpacity=".0" /><stop offset="1" stopColor="#34d399" stopOpacity=".55" />
        </radialGradient>
      </defs>
      <rect x="2" y="2" width="60" height="60" rx="15" fill="url(#lg-bg)" stroke="#1f2b3e" />
      <circle cx="32" cy="32" r="24" fill="none" stroke="#38bdf8" strokeOpacity=".35" />
      <circle cx="32" cy="32" r="15" fill="none" stroke="#38bdf8" strokeOpacity=".25" />
      <path d="M32 8v48M8 32h48" stroke="#38bdf8" strokeOpacity=".18" />
      <g className="lg-sweep"><path d="M32 32 L32 8 A24 24 0 0 1 52.8 20 Z" fill="url(#lg-sweep)" /></g>
      {/* rising candles */}
      <g strokeLinecap="round">
        <line x1="21" y1="34" x2="21" y2="46" stroke="#f87171" strokeWidth="1.4" /><rect x="18.5" y="37" width="5" height="6" rx="1" fill="#f87171" />
        <line x1="31" y1="26" x2="31" y2="42" stroke="#34d399" strokeWidth="1.4" /><rect x="28.5" y="29" width="5" height="9" rx="1" fill="#34d399" />
        <line x1="41" y1="16" x2="41" y2="34" stroke="#34d399" strokeWidth="1.4" /><rect x="38.5" y="19" width="5" height="11" rx="1" fill="#34d399" />
      </g>
      <circle cx="47" cy="15" r="2.6" fill="#facc15" className="lg-blip" />
    </svg>
  );
}

const api = (path, body) => fetch(path, {
  method: body ? "POST" : "GET", cache: "no-store",
  headers: body ? { "Content-Type": "application/json", "X-CG-Client": "dashboard" } : {},
  body: body ? JSON.stringify(body) : undefined,
}).then((r) => r.json());

/* ================================================================ boot / login screen */
function bootLines(d, err) {
  if (!d) return [[err ? "FAIL" : "WAIT", err ? `Command server unreachable: ${err}` : "Linking to command server…"]];
  const ai = d.symbols.filter((s) => (s.brain || "").startsWith("AI")).length;
  const tr = (d.mission || []).find((r) => r.progress) || (d.mission || []).find((r) => r.status === "training");
  const nxt = (d.news?.plan || []).filter((e) => e.t0 > Date.now() / 1000).sort((a, b) => a.t0 - b.t0)[0];
  const mins = nxt ? Math.round((nxt.t0 - Date.now() / 1000) / 60) : null;
  return [
    ["OK", "Command server linked"],
    [d.bot.broker_ok ? "OK" : "FAIL", `Broker link · Exness ${d.account_mode === "live" ? "LIVE" : "DEMO"} · $${d.account.equity.toFixed(2)}`],
    [d.bot.running ? "OK" : "FAIL", `Trading bot ${d.bot.running ? "online · next check in " + Math.floor(d.bot.next_check_s / 60) + ":" + String(d.bot.next_check_s % 60).padStart(2, "0") : "OFFLINE"}`],
    ["OK", `${d.symbols.length} markets armed · ${ai} on AI models · ${d.symbols.length - ai} on indicator council`],
    [tr ? "OK" : "IDLE", tr ? `Training lane · ${tr.name}${tr.scenario ? " · " + tr.scenario + " reward" : ""}${tr.progress?.total ? " · " + Math.round((tr.progress.step / tr.progress.total) * 100) + "%" : ""}` : "Training lanes idle"],
    ["OK", nxt ? `News desk · ${nxt.title.split(" + ")[0]} in ${mins >= 60 ? Math.floor(mins / 60) + "h " + (mins % 60) + "m" : mins + " min"}` : "News desk · no big US release in the next 36 h"],
    ["OK", `Risk limits armed · $${(d.account.daily_loss_limit || 0).toFixed(0)} daily stop · max 14 trades`],
  ];
}

export function Boot({ d, err, onEnter, onAccount, acctOpen, closeAcct }) {
  const lines = bootLines(d, err);
  const [shown, setShown] = useState(0);
  useEffect(() => {
    if (shown >= lines.length) return;
    const id = setTimeout(() => setShown((n) => n + 1), shown === 0 ? 500 : 280);
    return () => clearTimeout(id);
  }, [shown, lines.length]);
  const ready = d && shown >= lines.length;
  useEffect(() => {
    const k = (e) => { if (ready && e.key === "Enter") onEnter(); };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, [ready, onEnter]);
  const live = d?.account_mode === "live";
  return (
    <div className="boot">
      <div className="boot-grid" aria-hidden />
      <div className="boot-inner">
        <div className="boot-logo"><Logo size={128} /><div className="boot-ring" /></div>
        <h1 className="boot-title">CAUTIOUS GIGGLE</h1>
        <div className="boot-sub">AI TRADING WAR ROOM</div>

        <div className="boot-log">
          {lines.slice(0, shown).map(([st, txt], i) => (
            <div key={i} className="boot-line">
              <span className={`bl-st ${st.toLowerCase()}`}>[{st.padEnd(4, " ")}]</span> {txt}
            </div>
          ))}
          {shown < lines.length && <div className="boot-line cursor">▌</div>}
        </div>

        <div className={`boot-accounts ${ready ? "in" : ""}`}>
          <button className={`acct-card ${!live ? "active" : ""}`} onClick={live ? onAccount : onEnter}>
            <span className="ac-tag demo">DEMO</span>
            <b>Practice account</b>
            <span className="muted small">{!live && d ? `$${d.account.equity.toFixed(2)} · aggressive learning settings` : "switch back any time"}</span>
          </button>
          <button className={`acct-card live ${live ? "active" : ""}`} onClick={live ? onEnter : onAccount}>
            <span className="ac-tag live">LIVE</span>
            <b>Real money</b>
            <span className="muted small">{live && d ? `$${d.account.equity.toFixed(2)} · conservative risk` : "set up or switch · confirmation required"}</span>
          </button>
        </div>

        <button className="boot-enter" disabled={!ready} onClick={onEnter}>
          {ready ? <>ENTER WAR ROOM <span className="kbd">↵</span></> : <><Loader2 size={15} className="spin-i" /> BOOTING…</>}
        </button>
        <div className="boot-foot muted small">Exness {live ? "LIVE" : "demo"} · local only · 127.0.0.1</div>
      </div>
      <AccountSwitch open={acctOpen} onClose={closeAcct} />
    </div>
  );
}

/* ================================================================ account switch */
export function AccountSwitch({ open, onClose }) {
  const [info, setInfo] = useState(null);
  const [chk, setChk] = useState(null);
  const [busy, setBusy] = useState("");
  const [phrase, setPhrase] = useState("");
  const [msg, setMsg] = useState(null);
  const [editing, setEditing] = useState(false);
  useEffect(() => {
    if (!open) return;
    setChk(null); setMsg(null); setPhrase(""); setEditing(false);
    api("/api/account").then(setInfo).catch(() => setMsg({ bad: true, text: "Can't reach the dashboard server." }));
  }, [open]);
  if (!open) return null;
  const live = info?.live, active = info?.active;
  const check = async () => {
    setBusy("check"); setChk(null);
    try { setChk(await api("/api/account/check", { profile: "live" })); } catch { setChk({ ok: false, error: "request failed" }); }
    setBusy("");
  };
  const doSwitch = async (profile) => {
    setBusy("switch"); setMsg(null);
    try {
      const r = await api("/api/account/switch", { profile, confirm: phrase });
      if (r.ok) {
        setMsg({ text: `Switched to ${profile.toUpperCase()}. Restarting the bot and dashboard on the new account…` });
        try { sessionStorage.removeItem("cg_booted"); } catch { /* ignore */ }
        setTimeout(() => window.location.reload(), 9000);
        return;
      }
      setMsg({ bad: true, text: r.error || "switch failed" });
    } catch { setMsg({ bad: true, text: "request failed" }); }
    setBusy("");
  };
  return (
    <div className="drawer-bg center" onClick={busy ? undefined : onClose}>
      <div className="acct-modal hud" onClick={(e) => e.stopPropagation()}>
        <div className="tour-top"><h3><KeyRound size={16} /> Account</h3>
          <button className="icon-btn" onClick={onClose} disabled={!!busy} aria-label="close"><X size={16} /></button></div>
        {!info ? <div className="muted"><Loader2 size={14} className="spin-i" /> loading…</div> : (
          <>
            <div className="acct-now">Active: <b className={active === "live" ? "neg" : "pos"}>{active === "live" ? "LIVE · real money" : "DEMO · practice"}</b></div>

            {active === "live" ? (
              <>
                <p className="small">Switching back to demo restarts the bot on the practice account. Open live trades keep their stop-loss and take-profit at Exness.</p>
                <button className="btn" disabled={!!busy} onClick={() => doSwitch("demo")}>{busy === "switch" ? "Switching…" : "Switch to DEMO"}</button>
              </>
            ) : !live?.ready || editing ? (
              <LiveForm form={info.form} onSaved={(r) => { setEditing(false); api("/api/account").then(setInfo); setChk(null); if (r.live?.ready) check(); }}
                        onCancel={live?.ready ? () => setEditing(false) : null} />
            ) : (
              <>
                <div className="saved-row small"><CheckCircle2 size={14} className="pos" /> Live keys saved on this Mac
                  <button className="linkish" disabled={!!busy} onClick={() => setEditing(true)}>change</button></div>
                <div className="rules">
                  <div className="rules-h"><ShieldCheck size={14} /> On a real-money account the bot automatically switches to:</div>
                  <ul>{(info.live_rules || []).map((r) => <li key={r}>{r}</li>)}</ul>
                </div>
                {!chk ? (
                  <button className="btn ghost" disabled={!!busy} onClick={check}>{busy === "check" ? <><Loader2 size={14} className="spin-i" /> Checking…</> : "Run connection check"}</button>
                ) : chk.ok && !chk.is_demo ? (
                  <div className="chk ok"><CheckCircle2 size={15} /> Connected to a REAL account · balance {chk.currency} {chk.balance?.toFixed(2)}</div>
                ) : (
                  <div className="chk bad"><AlertTriangle size={15} /> {chk.ok && chk.is_demo ? "Those keys belong to a DEMO account. live.env must point at your real account." : chk.error}</div>
                )}
                {chk?.ok && !chk.is_demo && (
                  <div className="golive">
                    <label className="small">Real money is at risk. Type <b>{info.confirm}</b> to confirm.</label>
                    <input className="search" value={phrase} onChange={(e) => setPhrase(e.target.value)} placeholder={info.confirm} />
                    <button className="btn danger" disabled={phrase.trim().toUpperCase() !== info.confirm || !!busy} onClick={() => doSwitch("live")}>
                      {busy === "switch" ? "Switching…" : "Switch to LIVE"}
                    </button>
                  </div>
                )}
              </>
            )}
            {msg && <div className={`chk ${msg.bad ? "bad" : "ok"}`}>{msg.text}</div>}
          </>
        )}
      </div>
    </div>
  );
}

/* ================================================================ live key form (no files, no code) */
function Field({ label, hint, err, saved, children }) {
  return (
    <label className={`fld ${err ? "bad" : ""}`}>
      <span className="fld-l">{label}{saved && <em className="fld-saved">saved · leave blank to keep</em>}</span>
      {children}
      {err ? <span className="fld-err">{err}</span> : hint && <span className="fld-h">{hint}</span>}
    </label>
  );
}

function LiveForm({ form, onSaved, onCancel }) {
  const saved = form?.saved || {};
  const [v, setV] = useState({ api_key: "", account_id: "", gateway: form?.gateway_hint || "", private_key: "" });
  const [errs, setErrs] = useState({});
  const [busy, setBusy] = useState(false);
  const [fail, setFail] = useState("");
  const set = (k) => (e) => { setV({ ...v, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }); };
  const need = (k) => !saved[k] && !String(v[k]).trim();
  const ready = !["api_key", "account_id", "gateway", "private_key"].some(need);
  const submit = async (e) => {
    e.preventDefault();
    setBusy(true); setFail("");
    try {
      const r = await api("/api/account/live_setup", v);
      if (r.ok) { setV({ ...v, private_key: "" }); onSaved(r); }
      else if (r.errors) setErrs(r.errors);
      else setFail(r.error || "could not save");
    } catch { setFail("Can't reach the dashboard server."); }
    setBusy(false);
  };
  return (
    <form className="live-form" onSubmit={submit} autoComplete="off">
      <p className="small"><b>Connect your real-money account.</b> Fill in the four boxes from Exness Personal Area → <b>Key Management</b>
        (create an API key for your <b>real</b> account). They are saved only on this Mac, readable only by your user,
        and never shown again or sent anywhere except Exness.</p>
      <Field label="API key ID" hint="The key's ID in Key Management" err={errs.api_key} saved={saved.api_key}>
        <input className="search" value={v.api_key} onChange={set("api_key")} spellCheck={false} placeholder={saved.api_key ? "••••••••" : "e.g. 3f9c…"} />
      </Field>
      <Field label="Account number" hint="Your REAL trading account (not the demo)" err={errs.account_id} saved={saved.account_id}>
        <input className="search" value={v.account_id} onChange={set("account_id")} spellCheck={false} inputMode="numeric" placeholder={saved.account_id ? "••••••••" : "e.g. 12345678"} />
      </Field>
      <Field label="Gateway" hint="Filled in from your demo account; change it only if Key Management shows a different host" err={errs.gateway}>
        <input className="search" value={v.gateway} onChange={set("gateway")} spellCheck={false} />
      </Field>
      <Field label="Private key" hint="Open the key file you downloaded, copy everything, paste it here" err={errs.private_key} saved={saved.private_key}>
        <textarea className="search secret" rows={3} value={v.private_key} onChange={set("private_key")} spellCheck={false}
                  placeholder={saved.private_key ? "saved · paste a new one to replace it" : "paste the private key"} />
      </Field>
      {fail && <div className="chk bad"><AlertTriangle size={15} /> {fail}</div>}
      <div className="live-form-btns">
        {onCancel && <button type="button" className="btn ghost" onClick={onCancel} disabled={busy}>Cancel</button>}
        <button type="submit" className="btn" disabled={!ready || busy}>
          {busy ? <><Loader2 size={14} className="spin-i" /> Saving…</> : "Save & test connection"}
        </button>
      </div>
      <div className="muted small">Saving does not switch to live. Next you'll see a read-only connection test (no orders), then the GO LIVE confirmation.</div>
    </form>
  );
}
