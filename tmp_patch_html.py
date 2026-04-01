with open('C:/Users/Administrator/work/cautious-giggle-clone-20260320161357/tools/ui_assets/project_status_ui.html', encoding='utf-8') as f:
    content = f.read()

# 1. Add pipe9 CSS after stageBarFill.failed rule
OLD_CSS = '.stageBarFill.failed,.stageBarFill.paused{background:linear-gradient(90deg,var(--red),var(--red))}'
NEW_CSS = OLD_CSS + (
    '\n.pipe9Wrap{overflow-x:auto;padding-bottom:8px}'
    '\n.pipe9Header{display:grid;grid-template-columns:110px repeat(9,minmax(88px,1fr));gap:5px;padding:0 2px;color:var(--muted);font-size:10px;letter-spacing:.16em;text-transform:uppercase;margin-bottom:4px;min-width:920px}'
    '\n.pipe9Row{display:grid;grid-template-columns:110px repeat(9,minmax(88px,1fr));gap:5px;margin-bottom:5px;align-items:stretch;min-width:920px}'
    '\n.pipe9Symbol{padding:10px 12px;border-radius:12px;border:1px solid var(--line);background:var(--white-a03);display:flex;flex-direction:column;justify-content:center}'
    '\n.pipe9Node{padding:8px 10px;border-radius:12px;border:1px solid var(--line);background:var(--white-a03);font-size:12px}'
    '\n.pipe9Node.active{border-color:var(--cyan-a28);background:var(--cyan-a08)}'
    '\n.pipe9Node.done,.pipe9Node.live,.pipe9Node.ready,.pipe9Node.armed{border-color:var(--green-a24);background:var(--green-a08)}'
    '\n.pipe9Node.failed,.pipe9Node.paused{border-color:rgba(255,80,80,.28);background:rgba(255,80,80,.06)}'
    '\n.pipe9Node.testing,.pipe9Node.partial{border-color:var(--amber-a28);background:var(--amber-a08)}'
    '\n.pipe9NodeTop{display:flex;align-items:center;justify-content:space-between;gap:4px;margin-bottom:3px}'
    '\n.pipe9NodeName{font-size:9px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}'
    '\n.pipe9NodeDetail{font-size:10px;color:var(--text);line-height:1.3;margin-top:1px;opacity:.85;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}'
    '\n.pipe9Bar{height:3px;border-radius:999px;background:var(--white-a08);overflow:hidden;margin-top:5px}'
    '\n.pipe9BarFill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--cyan),var(--violet));transition:width .8s ease}'
    '\n.pipe9BarFill.done,.pipe9BarFill.live,.pipe9BarFill.ready,.pipe9BarFill.armed{background:linear-gradient(90deg,var(--green-ramp),var(--green))}'
    '\n.pipe9BarFill.testing,.pipe9BarFill.partial{background:linear-gradient(90deg,var(--amber-ramp),var(--amber))}'
    '\n.pipe9BarFill.failed,.pipe9BarFill.paused{background:linear-gradient(90deg,var(--red),var(--red))}'
)
assert OLD_CSS in content, 'CSS marker not found'
content = content.replace(OLD_CSS, NEW_CSS, 1)

# 2. Add fullPipeline9 div as first panel in training section
OLD_TRAIN_LAYOUT = 'data-route="training"><div class="layout">'
NEW_TRAIN_LAYOUT = (
    'data-route="training"><div class="layout">'
    '<div class="panel">'
    '<div class="panelHead"><div>'
    '<div class="eyebrow">Full autonomous pipeline</div>'
    '<div class="panelTitle">9-Stage symbol lifecycle</div>'
    '</div><span class="tag">Live</span></div>'
    '<div class="meta">MT5 Data Ingestion \u2192 150 Features \u2192 LSTM \u2192 DreamerV3 \u2192 PPO \u2192 Candidate \u2192 Backtest \u2192 Champion \u2192 Trade \u2014 repeats autonomously until a better champion arrives.</div>'
    '<div id="fullPipeline9"></div>'
    '</div>'
)
assert OLD_TRAIN_LAYOUT in content, 'Training layout marker not found'
content = content.replace(OLD_TRAIN_LAYOUT, NEW_TRAIN_LAYOUT, 1)

# 3. Update pipelineBoard panel meta text
OLD_PIPE_META = 'Tracks each configured symbol from LSTM through Dreamer, PPO, Canary, Champion, and into live trading.'
NEW_PIPE_META = 'Full 9-stage lifecycle per symbol: MT5 Data \u2192 Features \u2192 LSTM \u2192 Dreamer \u2192 PPO \u2192 Candidate \u2192 Backtest \u2192 Champion \u2192 Trade.'
content = content.replace(OLD_PIPE_META, NEW_PIPE_META, 1)

# 4. Insert renderFullPipeline JS function before render(d)
RENDER_MARKER = 'function render(d){'
FULL_PIPELINE_JS = (
    "function renderFullPipeline(d){"
    "const el=byId('fullPipeline9');"
    "if(!el)return;"
    "const t=d.training||{};"
    "const rows=(t.symbol_stage_rows||[]).filter(r=>symbolMatch(r.symbol));"
    "if(!rows.length){el.innerHTML='<div class=\"empty\">Pipeline data not yet available \u2014 waiting for first training cycle.</div>';return;}"
    "const stages=[['data_ingest','MT5 Data'],['features','Features'],['lstm','LSTM'],['dreamer','Dreamer'],['ppo','PPO'],['candidate','Candidate'],['backtest','Backtest'],['champion','Champion'],['trading','Trade']];"
    "const header='<div class=\"pipe9Header\"><div>Symbol</div>'+stages.map(([,lbl])=>'<div>'+esc(lbl)+'</div>').join('')+'</div>';"
    "const body=rows.map(row=>{"
    "const sym=row.symbol||'';"
    "const symCell='<div class=\"pipe9Symbol\"><div style=\"font-size:10px;color:var(--muted)\">Symbol</div><div style=\"font-weight:600;font-size:13px;margin-top:2px\">'+esc(sym)+'</div></div>';"
    "const cells=stages.map(([key,lbl])=>{"
    "const stage=row[key]||{};"
    "const state=String(stage.state||'queued');"
    "const detail=(key==='champion'||key==='canary')?tailLabel(stage.detail):(stage.detail||'\u2014');"
    "const pctV=clamp(Number(stage.progress_pct)||0,0,100);"
    "const badge='<span class=\"stageBadge '+esc(state)+'\">'+esc(state)+'</span>';"
    "const bar='<div class=\"pipe9Bar\"><div class=\"pipe9BarFill '+esc(state)+'\" style=\"width:'+pctV+'%\"></div></div>';"
    "return '<div class=\"pipe9Node '+esc(state)+'\"><div class=\"pipe9NodeTop\"><span class=\"pipe9NodeName\">'+esc(lbl)+'</span>'+badge+'</div><div class=\"pipe9NodeDetail\" title=\"'+esc(detail)+'\">'+esc(detail)+'</div>'+bar+'</div>';"
    "}).join('');"
    "return '<div class=\"pipe9Row\">'+symCell+cells+'</div>';"
    "}).join('');"
    "el.innerHTML='<div class=\"pipe9Wrap\">'+header+body+'</div>';}"
    "\n"
)
assert RENDER_MARKER in content, 'render function marker not found'
content = content.replace(RENDER_MARKER, FULL_PIPELINE_JS + RENDER_MARKER, 1)

# 5. Add renderFullPipeline(d) call in render(d)
OLD_RENDER_CALL = 'function render(d){window.__status=d; renderHeader(d); renderHero(d); renderOverview(d); renderTraining(d); renderPipelineBoards(d);'
NEW_RENDER_CALL = 'function render(d){window.__status=d; renderHeader(d); renderHero(d); renderOverview(d); renderTraining(d); renderPipelineBoards(d); renderFullPipeline(d);'
assert OLD_RENDER_CALL in content, 'render call marker not found'
content = content.replace(OLD_RENDER_CALL, NEW_RENDER_CALL, 1)

# 6. Update renderPipelineBoards columns to show 9 stages
OLD_PIPE_COLS = "const columns=[['lstm','LSTM'],['dreamer','Dreamer'],['ppo','PPO'],['canary','Canary'],['champion','Champion'],['trading','Trading']]"
NEW_PIPE_COLS = "const columns=[['data_ingest','MT5 Data'],['features','Features'],['lstm','LSTM'],['dreamer','Dreamer'],['ppo','PPO'],['candidate','Candidate'],['backtest','Backtest'],['champion','Champion'],['trading','Trade']]"
assert OLD_PIPE_COLS in content, 'pipeline cols marker not found: ' + repr(content[content.find("const columns="):content.find("const columns=")+200])
content = content.replace(OLD_PIPE_COLS, NEW_PIPE_COLS, 1)

# 7. Update pipelineRow grid from 6 to 9 columns
OLD_GRID = '.pipelineRow{display:grid;grid-template-columns:140px repeat(6,minmax(0,1fr))'
NEW_GRID = '.pipelineRow{display:grid;grid-template-columns:120px repeat(9,minmax(0,1fr))'
assert OLD_GRID in content, 'pipelineRow grid marker not found'
content = content.replace(OLD_GRID, NEW_GRID, 1)

with open('C:/Users/Administrator/work/cautious-giggle-clone-20260320161357/tools/ui_assets/project_status_ui.html', 'w', encoding='utf-8') as f:
    f.write(content)
print('Done. File length:', len(content), 'chars')
