"""Advisory review only: no trade execution or promotion."""
import argparse
import json
import urllib.request

def review(report, model):
    candidates=report["strategies"][:5]
    evidence=[{"name":r["name"],"validation":r["validation"]} for r in candidates]
    schema={"type":"object","properties":{"strategy":{"type":"string","enum":["WAIT"]+[r["name"] for r in candidates]},"reason":{"type":"string"}},"required":["strategy","reason"]}
    body={"model":model,"stream":False,"format":schema,"options":{"temperature":0},"messages":[{"role":"system","content":"You are an offline research reviewer. Select a candidate for further paper testing or WAIT. Do not claim profitability. Prefer WAIT for weak evidence. Do not recommend live orders."},{"role":"user","content":json.dumps(evidence)}]}
    req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req,timeout=30) as response:
            result=json.loads(json.load(response)["message"]["content"])
        if result["strategy"] not in schema["properties"]["strategy"]["enum"] or not isinstance(result["reason"],str):
            raise ValueError("invalid review")
        return {"mode":"advisory_only",**result}
    except (OSError,ValueError,KeyError,TypeError) as exc:
        return {"mode":"advisory_only","strategy":"WAIT","reason":type(exc).__name__}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("report")
    p.add_argument("--model",default="qwen3:8b")
    args=p.parse_args()
    with open(args.report) as f:
        print(json.dumps(review(json.load(f),args.model),indent=2))
