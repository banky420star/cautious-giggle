import React, { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, KeyRound, Loader2, ShieldCheck, X } from "lucide-react";

/* ================================================================ logo */
export function Logo({ size = 40 }) {
  // Gorilla Trader: gorilla face, pirate tricorn, eyepatch, gold earring, blunt with a live ember
  return (
    <svg width={size} height={size} viewBox="0 0 128 128" className="logo-svg gorilla"
         role="img" aria-label="Gorilla Trader">
      <defs>
        <radialGradient id="gt-bg" cx="50%" cy="38%" r="70%">
          <stop offset="0" stopColor="#1d2a1f" /><stop offset="1" stopColor="#070a08" />
        </radialGradient>
        <radialGradient id="gt-fur" cx="50%" cy="35%" r="65%">
          <stop offset="0" stopColor="#3a3f47" /><stop offset="1" stopColor="#16191e" />
        </radialGradient>
        <radialGradient id="gt-skin" cx="50%" cy="40%" r="60%">
          <stop offset="0" stopColor="#5c616b" /><stop offset="1" stopColor="#30343b" />
        </radialGradient>
        <radialGradient id="gt-ember" cx="50%" cy="50%" r="50%">
          <stop offset="0" stopColor="#fff3b0" /><stop offset=".45" stopColor="#ff8a1f" /><stop offset="1" stopColor="#ff3d00" stopOpacity="0" />
        </radialGradient>
        <linearGradient id="gt-gold" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#fde68a" /><stop offset=".5" stopColor="#f5b83d" /><stop offset="1" stopColor="#b7791f" />
        </linearGradient>
      </defs>
      <rect x="3" y="3" width="122" height="122" rx="28" fill="url(#gt-bg)" stroke="url(#gt-gold)" strokeWidth="2.5" />

      {/* head and ears */}
      <circle cx="23" cy="78" r="11" fill="url(#gt-fur)" /><circle cx="23" cy="78" r="5.5" fill="#3b2f2a" />
      <circle cx="105" cy="78" r="11" fill="url(#gt-fur)" /><circle cx="105" cy="78" r="5.5" fill="#3b2f2a" />
      <circle cx="19" cy="90" r="3.6" fill="none" stroke="url(#gt-gold)" strokeWidth="2" />
      <path d="M64 36 C 94 36 108 56 106 80 C 104 104 88 118 64 118 C 40 118 24 104 22 80 C 20 56 34 36 64 36 Z" fill="url(#gt-fur)" />

      {/* face */}
      <path d="M64 58 C 84 56 94 68 92 84 C 90 102 80 112 64 112 C 48 112 38 102 36 84 C 34 68 44 56 64 58 Z" fill="url(#gt-skin)" />
      <path d="M36 70 C 46 58 82 58 92 70 C 82 66 46 66 36 70 Z" fill="#101317" />
      <ellipse cx="51" cy="75" rx="5.5" ry="4.3" fill="#f4e4c1" />
      <circle cx="52" cy="75.5" r="2.6" fill="#2a1606" /><circle cx="53" cy="74.4" r=".9" fill="#fff" className="gt-glint" />
      {/* eyepatch */}
      <path d="M34 60 L98 92" stroke="#0a0b0d" strokeWidth="2.4" strokeLinecap="round" />
      <ellipse cx="77" cy="75" rx="9" ry="7.5" fill="#0a0b0d" stroke="#2c3138" strokeWidth="1" />
      {/* muzzle, nose, mouth */}
      <ellipse cx="64" cy="96" rx="20" ry="14" fill="#4a4f58" />
      <path d="M58 82 C 60 78 68 78 70 82 C 72 88 56 88 58 82 Z" fill="#23262b" />
      <ellipse cx="60" cy="88.5" rx="3.4" ry="2.5" fill="#121417" /><ellipse cx="68" cy="88.5" rx="3.4" ry="2.5" fill="#121417" />
      <path d="M50 100 Q 62 107 78 100" stroke="#121417" strokeWidth="2.6" fill="none" strokeLinecap="round" />

      {/* blunt + ember + smoke */}
      <g transform="rotate(-14 77 100)">
        <rect x="76" y="97" width="30" height="6.5" rx="3.2" fill="#8a5a2b" stroke="#5a3814" strokeWidth=".8" />
        <path d="M84 97.4 v5.8 M92 97.4 v5.8 M100 97.4 v5.8" stroke="#6b4420" strokeWidth=".8" />
        <rect x="104" y="97" width="4" height="6.5" rx="1.6" fill="#ff7a1a" />
        <circle cx="107" cy="100.2" r="7" fill="url(#gt-ember)" className="gt-ember" />
      </g>
      <g className="gt-smoke" fill="none" strokeLinecap="round">
        <path d="M108 90 C 103 82 114 78 109 70 C 105 63 114 58 110 50" stroke="#d1d9e0" strokeWidth="2.6" strokeOpacity=".55" />
        <path d="M112 88 C 118 80 110 74 116 66" stroke="#d1d9e0" strokeWidth="1.8" strokeOpacity=".35" />
      </g>

      {/* pirate tricorn */}
      <path d="M30 46 C 32 20 96 20 98 46 Z" fill="#121418" />
      <path d="M8 50 C 34 28 94 28 120 50 C 104 42 24 42 8 50 Z" fill="#0b0c0f" stroke="url(#gt-gold)" strokeWidth="2.2" strokeLinejoin="round" />
      <path d="M31 44 C 50 38 78 38 97 44" stroke="url(#gt-gold)" strokeWidth="1.4" fill="none" opacity=".7" />
      <g className="gt-skull">
        <circle cx="64" cy="30" r="6" fill="#f3f4f6" />
        <rect x="60.5" y="33.5" width="7" height="4" rx="1.2" fill="#f3f4f6" />
        <circle cx="61.6" cy="29.6" r="1.5" fill="#0b0c0f" /><circle cx="66.4" cy="29.6" r="1.5" fill="#0b0c0f" />
        <path d="M55 37 L73 43 M73 37 L55 43" stroke="#f3f4f6" strokeWidth="2.4" strokeLinecap="round" />
      </g>
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
        <h1 className="boot-title">GORILLA TRADER</h1>
        <div className="boot-sub">AI TRADING WAR ROOM · BIG APE ENERGY</div>

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
            <span className="muted small">{live && d ? `$${d.account.equity.toFixed(2)} · same settings as demo` : "set up or switch · confirmation required"}</span>
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
                  <div className="rules-h"><ShieldCheck size={14} /> On your real-money account the bot will run with:</div>
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
