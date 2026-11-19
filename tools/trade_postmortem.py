"""Trade post-mortem: replay every real demo trade on M1 bars (exit type, how far it went for/against, what holding would have done). Writes logs/trade_postmortem.csv. .venv/bin/python tools/trade_postmortem.py"""
import json, numpy as np, pandas as pd, datetime as dt, collections
R='/Volumes/AI_DRIVE/cautious-giggle/'
op,cl={},{}
for l in open(R+'logs/trade_events.jsonl'):
    e=json.loads(l); p=e['payload']
    if e['event']=='trade_open': op[p['ticket']]=(e['ts'],p)
    if e['event']=='trade_closed': cl[p['ticket']]=(e['ts'],p)
import re
sends=collections.defaultdict(list)
for l in open(R+'logs/server.log',errors='ignore'):
    if 'action=open' in l and 'ORDER_SEND' in l and 'retcode=10009' in l:
        m=re.search(r'^(\S+ \S+) \+02:00.*ORDER_SEND (\S+) \| action=open side=(\w+) lots=(\S+) price=(\S+) sl=(\S+) tp=(\S+)',l)
        if m and m.group(6)!='None':
            ts=dt.datetime.fromisoformat(m.group(1)+'+02:00').timestamp()
            sends[m.group(2)].append((ts,float(m.group(5)),float(m.group(6)),float(m.group(7)),float(m.group(4))))
bars={}
def m1(s):
    b=s.rstrip('m')
    if b not in bars:
        a=np.load(R+f'data/exness_cache/{b}_M1.npy',allow_pickle=True); bars[b]=pd.DataFrame(a).drop_duplicates('time').sort_values('time').set_index('time')
    return bars[b]
rows=[]
for t,(cts,c) in cl.items():
    if t not in op: continue
    ots,o=op[t]; s=o['symbol']; side=1 if o['side']=='BUY' else -1
    t0=dt.datetime.fromisoformat(ots).timestamp(); t1=dt.datetime.fromisoformat(cts).timestamp()
    po,pc=o['open_price'],c['price']; sl,tp=o.get('sl'),o.get('tp')
    cand=[x for x in sends[s] if abs(x[0]-t0)<180 and abs(x[1]-po)/po<0.003]
    orig = bool(cand)
    if cand: x=min(cand,key=lambda x:abs(x[0]-t0)); sl,tp=x[2],x[3]
    if not sl or not tp or side*(po-sl)<=0: continue
    if not sl or not tp: continue
    risk=abs(po-sl); 
    b=m1(s); w=b[(b.index>=t0//60*60)&(b.index<=t1)]; after=b[(b.index>t1)&(b.index<=t1+4*3600)]
    if len(w)==0: continue
    fav=((w.high.max()-po) if side>0 else (po-w.low.min()))/risk
    adv=((po-w.low.min()) if side>0 else (w.high.max()-po))/risk
    res=side*(pc-po)/risk
    if abs(pc-sl)/risk<0.08: ex='stop hit'
    elif abs(pc-tp)/risk<0.08: ex='target hit'
    else: ex='bot closed'
    # what if held: would tp or sl come first in next 4h
    fut='neither'
    for _,r in after.iterrows():
        hi_tp = (r.high>=tp) if side>0 else (r.low<=tp); hi_sl=(r.low<=sl) if side>0 else (r.high>=sl)
        if hi_sl: fut='stop'; break
        if hi_tp: fut='target'; break
    rows.append(dict(t0=t0,spr=float(w.spread.iloc[0]),sym=s,side=o['side'],profit=c['profit'],R=res,fav=fav,adv=adv,exit=ex,hold=(t1-t0)/60,
        hour=dt.datetime.fromtimestamp(t0,dt.UTC).hour,rr=abs(tp-po)/risk,orig=orig,lots=o['volume'],stopbps=risk/po*1e4,after=fut,day=ots[:10]))
d=pd.DataFrame(rows); d.to_csv('/Volumes/AI_DRIVE/cautious-giggle/logs/trade_postmortem.csv',index=False)
pd.set_option('display.width',200)
print(len(d),'trades, total $',round(d.profit.sum(),2),'| target:stop distance median', round(d.rr.median(),2))
g=lambda k: d.groupby(k).agg(n=('profit','size'),usd=('profit','sum'),win=('profit',lambda x:(x>0).mean()*100),avgR=('R','mean'),bestR=('fav','mean')).round(2)
print(g('exit')); print(g('sym')); print(g('side'))
bc=d[d.exit=='bot closed']
print('\nBOT-CLOSED trades: how far they had gone for us before closing (in units of the stop):')
print(pd.cut(bc.fav,[-1,0.25,0.5,1,2,9]).value_counts().sort_index())
print('bot-closed, what the price did next 4h if left alone:',bc.after.value_counts().to_dict())
w,l=d[d.profit>0],d[d.profit<=0]
print('\nwinners: avg $%.2f, avg R %.2f, hold %.0f min | losers: avg $%.2f, avg R %.2f, hold %.0f min'%(w.profit.mean(),w.R.mean(),w.hold.median(),l.profit.mean(),l.R.mean(),l.hold.median()))
print('losers that were ahead >= 0.5 stop first:',int((l.fav>=0.5).sum()),'of',len(l),'| ahead >=1 stop:',int((l.fav>=1).sum()))
print('winners that went >= 0.8 stop against first:',int((w.adv>=0.8).sum()),'of',len(w))
d['hb']=pd.cut(d.hour,[-1,6,12,16,20,24],labels=['00-06 Asia','07-12 London','13-16 London+NY','17-20 NY','21-23 late'])
print(g('hb')); d['holdb']=pd.cut(d.hold,[-1,2,5,15,60,1e5],labels=['<2m','2-5m','5-15m','15-60m','>1h']); print(g('holdb'))
print(g('day'))
