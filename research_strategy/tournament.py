"""Offline strategy tournament. No broker or order APIs."""
import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

def signals(rows):
    """Signals at candle close; execution is on the following open."""
    c = [r["close"] for r in rows]
    h = [r["high"] for r in rows]
    l = [r["low"] for r in rows]
    v = [r["volume"] for r in rows]
    names = ["donchian", "ema_cross", "sma_cross", "momentum", "bollinger_reversion",
             "rsi_reversion", "vwap_reversion", "keltner_breakout", "volume_breakout",
             "macd", "stochastic_reversion", "atr_breakout"]
    out = {n: [0]*len(c) for n in names}
    def ema(period):
        a = 2/(period+1)
        result = [c[0]]
        for x in c[1:]:
            result.append(a*x+(1-a)*result[-1])
        return result
    e12,e26,e9,e21 = (ema(p) for p in (12,26,9,21))
    mac = [a-b for a,b in zip(e12,e26)]
    ms = [mac[0]]
    for x in mac[1:]:
        ms.append(.2*x+.8*ms[-1])
    tr = [h[0]-l[0]]+[max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1])) for i in range(1,len(c))]
    for i in range(50,len(c)):
        x=c[i]; prev=c[i-20:i]; avg=st.mean(prev); sd=st.pstdev(prev)
        hi=max(h[i-20:i]); lo=min(l[i-20:i]); atr=st.mean(tr[i-20:i])
        delta=[c[j]-c[j-1] for j in range(i-13,i+1)]
        gain=sum(max(d,0) for d in delta); loss=sum(max(-d,0) for d in delta)
        rsi=100*gain/(gain+loss) if gain+loss else 50
        vv=sum(v[i-19:i+1]); vw=sum(c[j]*v[j] for j in range(i-19,i+1))/vv if vv else x
        sk=100*(x-min(l[i-13:i+1]))/(max(h[i-13:i+1])-min(l[i-13:i+1])) if max(h[i-13:i+1])>min(l[i-13:i+1]) else 50
        def direction(a,b): return int(a>b)-int(a<b)
        def band(x,upper,lower): return 1 if x>upper else -1 if x<lower else 0
        values=[band(x,hi,lo),direction(e9[i],e21[i]),
                direction(st.mean(c[i-9:i+1]),st.mean(c[i-49:i+1])),
                direction(x,c[i-20]),-band(x,avg+2*sd,avg-2*sd),
                -band(rsi,70,30),-band(x,vw+atr,vw-atr),
                band(x,e21[i-1]+2*atr,e21[i-1]-2*atr),
                band(x,hi,lo) if v[i]>1.5*st.mean(v[i-20:i]) else 0,
                direction(mac[i],ms[i]),-band(sk,80,20),
                band(x,c[i-1]+atr,c[i-1]-atr)]
        for n,s in zip(names,values): out[n][i]=s
    return out

def evaluate(rows, signal, start, stop, cost_bps=10):
    """Unlevered signed exposure, next-open fills, explicit turnover costs."""
    if not 1<=start<stop<=len(rows)-1: raise ValueError("invalid interval")
    if not math.isfinite(cost_bps) or cost_bps<0: raise ValueError("invalid cost")
    equity=peak=1.; dd=0.; pos=0; entries=0; returns=[]; wins=losses=0.
    for i in range(start,stop):
        target=signal[i-1]
        if target not in (-1,0,1): raise ValueError("invalid signal")
        change=abs(target-pos)
        if target and target!=pos: entries+=1
        gross=target*(rows[i+1]["open"]/rows[i]["open"]-1)
        net=gross-change*cost_bps/10000
        if i==stop-1: net-=abs(target)*cost_bps/10000
        equity*=max(0,1+net); peak=max(peak,equity); dd=max(dd,1-equity/peak)
        returns.append(net); wins+=max(net,0); losses+=max(-net,0); pos=target
        if equity==0: break
    return {"net_return":equity-1,"max_drawdown":dd,"entries":entries,
            "bar_profit_factor":wins/losses if losses else None,
            "mean_bar_return":st.mean(returns)}

def load(path):
    rows=[]
    with open(path,newline="",encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            x={k:float(r[k]) for k in ("open","high","low","close","volume")}
            x["timestamp"]=r["timestamp"]
            if not all(math.isfinite(x[k]) for k in ("open","high","low","close","volume")) or min(x[k] for k in ("open","high","low","close"))<=0 or x["volume"]<0 or x["high"]<max(x["open"],x["close"]) or x["low"]>min(x["open"],x["close"]):
                raise ValueError("invalid OHLCV")
            if rows and x["timestamp"]<=rows[-1]["timestamp"]: raise ValueError("timestamps must be unique and ascending ISO UTC")
            rows.append(x)
    if len(rows)<300: raise ValueError("at least 300 bars required")
    return rows

def tournament(rows,cost_bps=10,sigs=None):
    a=int(len(rows)*.6); b=int(len(rows)*.8); end=len(rows)-1
    results=[]
    for name,s in (sigs or signals(rows)).items():
        results.append({"name":name,"development":evaluate(rows,s,51,a,cost_bps),
                        "validation":evaluate(rows,s,a,b,cost_bps),
                        "holdout":evaluate(rows,s,b,end,cost_bps)})
    results.sort(key=lambda r:r["validation"]["net_return"],reverse=True)
    return {"mode":"research_only","cost_bps_per_side":cost_bps,"bars":len(rows),
            "ranking":"validation net return; holdout not used for ranking",
            "strategies":results}

WARMUP=51

def walk_forward(rows,cost_bps=10,folds=5,sigs=None):
    """Expanding-window walk-forward: pick the best strategy on all bars before
    each test window (with costs), then score only that pick on the window.

    If no strategy made money in training the fold stays flat ("WAIT").
    Buy-and-hold over the same windows is reported as a benchmark."""
    if folds<1: raise ValueError("folds must be >= 1")
    end=len(rows)-1
    edges=[WARMUP+(end-WARMUP)*k//(folds+1) for k in range(folds+2)]
    if any(b-a<2 for a,b in zip(edges,edges[1:])): raise ValueError("too few bars for folds")
    sigs=sigs or signals(rows)
    hold=[1]*len(rows)
    out=[]; equity=bench=1.
    for k in range(1,folds+1):
        start,stop=edges[k],edges[k+1]
        train={n:evaluate(rows,s,WARMUP,start,cost_bps)["net_return"] for n,s in sigs.items()}
        best=max(train,key=train.get)
        pick=best if train[best]>0 else "WAIT"
        test=evaluate(rows,sigs[pick] if pick!="WAIT" else [0]*len(rows),start,stop,cost_bps)
        bh=evaluate(rows,hold,start,stop,cost_bps)
        equity*=1+test["net_return"]; bench*=1+bh["net_return"]
        out.append({"fold":k,"train_bars":[WARMUP,start],"test_bars":[start,stop],
                    "pick":pick,"train_net_return":train[best],
                    "test":test,"buy_and_hold_test_net_return":bh["net_return"]})
    return {"folds":out,"oos_net_return":equity-1,"buy_and_hold_oos_net_return":bench-1,
            "folds_positive":sum(f["test"]["net_return"]>0 for f in out),
            "worst_fold_drawdown":max(f["test"]["max_drawdown"] for f in out)}

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv"); p.add_argument("--cost-bps",type=float,default=10)
    p.add_argument("--folds",type=int,default=5)
    p.add_argument("--output",default="strategy_results.json")
    args=p.parse_args()
    rows=load(args.csv)
    report=tournament(rows,args.cost_bps)
    report["walk_forward"]=walk_forward(rows,args.cost_bps,args.folds)
    Path(args.output).write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report,indent=2,allow_nan=False))
