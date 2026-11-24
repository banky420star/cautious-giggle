import React, { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, ShieldCheck } from "lucide-react";
import { api, Panel, Badge, Empty } from "./ui.jsx";

const FIELDS = [
  ["api_key", "API key ID", "The key's ID in Key Management", "e.g. 3f9c…"],
  ["account_id", "Account number", "Your REAL trading account (not the demo)", "e.g. 12345678"],
  ["gateway", "Gateway", "Filled in from your demo account; change only if Key Management shows a different host", ""],
  ["private_key", "Private key", "Open the key file you downloaded, copy everything, paste it here", "paste the private key"],
];

function LiveForm({ form, onSaved, onCancel }) {
  const saved = form?.saved || {};
  const [v, setV] = useState({ api_key: "", account_id: "", gateway: form?.gateway_hint || "", private_key: "" });
  const [errs, setErrs] = useState({});
  const [busy, setBusy] = useState(false);
  const [fail, setFail] = useState("");
  const need = (k) => !saved[k] && !String(v[k]).trim();
  const ready = !FIELDS.some(([k]) => need(k));
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
    <form className="form" onSubmit={submit} autoComplete="off">
      <p className="help">Connect your real-money account with the four values from Exness Personal Area → Key Management
        (create an API key for your <b>real</b> account). They are stored only on this Mac and are never shown again.</p>
      {FIELDS.map(([k, label, hint, ph]) => (
        <label key={k} className={`field ${errs[k] ? "bad" : ""}`}>
          <span className="f-l">{label}{saved[k] && <Badge kind="up">saved</Badge>}</span>
          {k === "private_key"
            ? <textarea rows={3} value={v[k]} spellCheck={false} onChange={(e) => { setV({ ...v, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }); }}
                placeholder={saved[k] ? "saved · paste a new one to replace it" : ph} />
            : <input value={v[k]} spellCheck={false} inputMode={k === "account_id" ? "numeric" : undefined}
                onChange={(e) => { setV({ ...v, [k]: e.target.value }); setErrs({ ...errs, [k]: undefined }); }}
                placeholder={saved[k] ? "••••••••" : ph} />}
          <span className="f-h">{errs[k] || hint}</span>
        </label>
      ))}
      {fail && <div className="alert bad"><AlertTriangle size={15} /> {fail}</div>}
      <div className="row-btns">
        {onCancel && <button type="button" className="btn" onClick={onCancel} disabled={busy}>Cancel</button>}
        <button type="submit" className="btn primary" disabled={!ready || busy}>{busy ? "Saving…" : "Save and test connection"}</button>
      </div>
      <p className="help">Saving does not switch to live. A read-only connection test comes next, then the confirmation.</p>
    </form>
  );
}

export default function Account() {
  const [info, setInfo] = useState(null);
  const [chk, setChk] = useState(null);
  const [busy, setBusy] = useState("");
  const [phrase, setPhrase] = useState("");
  const [msg, setMsg] = useState(null);
  const [editing, setEditing] = useState(false);
  const load = () => api("/api/account").then(setInfo).catch(() => setMsg({ bad: true, text: "Can't reach the dashboard server." }));
  useEffect(() => { load(); }, []);
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
        setMsg({ text: `Switched to ${profile.toUpperCase()}. The bot and dashboard are restarting on the new account…` });
        setTimeout(() => window.location.reload(), 9000);
        return;
      }
      setMsg({ bad: true, text: r.error || "switch failed" });
    } catch { setMsg({ bad: true, text: "request failed" }); }
    setBusy("");
  };
  if (!info) return <Panel title="Account">{msg ? <div className="alert bad">{msg.text}</div> : <Empty>Loading…</Empty>}</Panel>;
  const live = info.live, active = info.active;
  return (
    <div className="grid g-acct">
      <Panel title="Trading account" actions={<Badge kind={active === "live" ? "warn" : "info"}>{active === "live" ? "LIVE · real money" : "DEMO · practice"}</Badge>}>
        {active === "live" ? (
          <div className="stack">
            <p className="help">Switching back to demo restarts the bot on the practice account. Open live trades keep their stop-loss and take-profit at Exness.</p>
            <div className="row-btns"><button className="btn" disabled={!!busy} onClick={() => doSwitch("demo")}>{busy === "switch" ? "Switching…" : "Switch to demo"}</button></div>
          </div>
        ) : !live?.ready || editing ? (
          <LiveForm form={info.form} onSaved={(r) => { setEditing(false); load(); setChk(null); if (r.live?.ready) check(); }}
            onCancel={live?.ready ? () => setEditing(false) : null} />
        ) : (
          <div className="stack">
            <div className="line-item"><CheckCircle2 size={15} className="up" /> Live keys saved on this Mac
              <button className="linkbtn" disabled={!!busy} onClick={() => setEditing(true)}>Change</button></div>
            {!chk ? (
              <div className="row-btns"><button className="btn" disabled={!!busy} onClick={check}>{busy === "check" ? <><Loader2 size={14} className="spin" /> Checking…</> : "Run connection check"}</button></div>
            ) : chk.ok && !chk.is_demo ? (
              <div className="alert ok"><CheckCircle2 size={15} /> Connected to a real account · balance {chk.currency} {chk.balance?.toFixed(2)}</div>
            ) : (
              <div className="alert bad"><AlertTriangle size={15} /> {chk.ok && chk.is_demo ? "Those keys belong to a DEMO account. Use the keys of your real account." : chk.error}</div>
            )}
            {chk?.ok && !chk.is_demo && (
              <div className="confirm">
                <label className="field"><span className="f-l">Real money is at risk. Type <b>{info.confirm}</b> to confirm.</span>
                  <input value={phrase} onChange={(e) => setPhrase(e.target.value)} placeholder={info.confirm} /></label>
                <div className="row-btns">
                  <button className="btn danger" disabled={phrase.trim().toUpperCase() !== info.confirm || !!busy} onClick={() => doSwitch("live")}>
                    {busy === "switch" ? "Switching…" : "Switch to live"}</button>
                </div>
              </div>
            )}
          </div>
        )}
        {msg && <div className={`alert ${msg.bad ? "bad" : "ok"}`}>{msg.text}</div>}
      </Panel>
      <Panel title="Rules on the real-money account">
        <ul className="rules">{(info.live_rules || []).map((r) => <li key={r}><ShieldCheck size={14} />{r}</li>)}</ul>
      </Panel>
    </div>
  );
}
