import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_strategy.tournament import signals, evaluate, tournament, walk_forward, load
from research_strategy import exness_data, ollama_review


def fixture(n=400):
    return [{"open":100+i*.1,"close":100+i*.1+.02,"high":101+i*.1,"low":99+i*.1,"volume":100+i%7} for i in range(n)]


def wavy(n=900):
    import math
    rows=[]
    for i in range(n):
        p=100+5*math.sin(i/15)+i*.01
        rows.append({"open":p,"close":p+.05,"high":p+.5,"low":p-.5,"volume":100+i%11})
    return rows


class TournamentTests(unittest.TestCase):
    def test_prefix_invariance(self):
        rows=fixture()
        full=signals(rows)
        short=signals(rows[:250])
        for name in full:
            self.assertEqual(full[name][:250],short[name])
    def test_flat(self):
        rows=fixture()
        r=evaluate(rows,[0]*len(rows),51,100)
        self.assertEqual(r["net_return"],0)
    def test_cost(self):
        rows=fixture()
        s=[1]*len(rows)
        self.assertLess(evaluate(rows,s,51,100,20)["net_return"],evaluate(rows,s,51,100,0)["net_return"])
    def test_tournament(self):
        r=tournament(fixture())
        self.assertEqual(len(r["strategies"]),12)
        self.assertEqual(r["mode"],"research_only")
    def test_next_open(self):
        rows=fixture()
        s=[0]*len(rows)
        s[50]=1
        result=evaluate(rows,s,51,52,0)
        self.assertAlmostEqual(result["net_return"],rows[52]["open"]/rows[51]["open"]-1)


class WalkForwardTests(unittest.TestCase):
    def test_folds_cover_range_without_overlap(self):
        rows=wavy()
        wf=walk_forward(rows,5,folds=4)
        self.assertEqual(len(wf["folds"]),4)
        for a,b in zip(wf["folds"],wf["folds"][1:]):
            self.assertEqual(a["test_bars"][1],b["test_bars"][0])
        self.assertEqual(wf["folds"][-1]["test_bars"][1],len(rows)-1)
        for f in wf["folds"]:
            self.assertEqual(f["train_bars"][1],f["test_bars"][0])

    def test_pick_ignores_future_bars(self):
        rows=wavy()
        base=walk_forward(rows,5,folds=4)
        first_test=base["folds"][0]["test_bars"][0]
        changed=[dict(r) for r in rows]
        for r in changed[first_test+1:]:
            for k in ("open","high","low","close"):
                r[k]*=3
        moved=walk_forward(changed,5,folds=4)
        self.assertEqual(base["folds"][0]["pick"],moved["folds"][0]["pick"])
        self.assertEqual(base["folds"][0]["train_net_return"],moved["folds"][0]["train_net_return"])

    def test_waits_when_nothing_made_money_in_training(self):
        rows=wavy()
        wf=walk_forward(rows,10_000,folds=3)   # costs so high every rule loses
        self.assertTrue(all(f["pick"]=="WAIT" for f in wf["folds"]))
        self.assertEqual(wf["oos_net_return"],0)

    def test_rejects_too_few_bars(self):
        with self.assertRaises(ValueError):
            walk_forward(fixture(60),5,folds=5)


class LoadTests(unittest.TestCase):
    def write(self, rows):
        f=tempfile.NamedTemporaryFile("w",suffix=".csv",delete=False,newline="")
        f.write("timestamp,open,high,low,close,volume\n")
        for r in rows: f.write(",".join(str(x) for x in r)+"\n")
        f.close()
        return f.name

    def good(self,n=300):
        return [(f"2026-01-01T00:{i//60:02d}:{i%60:02d}+00:00",100,101,99,100.5,10) for i in range(n)]

    def test_accepts_valid_file(self):
        self.assertEqual(len(load(self.write(self.good()))),300)

    def test_rejects_high_below_close_and_unsorted(self):
        bad=self.good(); bad[10]=(bad[10][0],100,100.2,99,100.5,10)
        with self.assertRaises(ValueError): load(self.write(bad))
        swapped=self.good(); swapped[5],swapped[6]=swapped[6],swapped[5]
        with self.assertRaises(ValueError): load(self.write(swapped))

    def test_rejects_short_history(self):
        with self.assertRaises(ValueError): load(self.write(self.good(100)))


class ExnessDataTests(unittest.TestCase):
    def test_resample_to_hour_buckets(self):
        base=1_800_000_000-1_800_000_000%3600
        rows=[{"_t":base+i*300,"timestamp":"","open":10+i,"high":11+i,"low":9+i,"close":10.5+i,"volume":1}
              for i in range(24)]
        h=exness_data.resample(rows,60)
        self.assertEqual(len(h),2)
        self.assertEqual((h[0]["open"],h[0]["close"],h[0]["high"],h[0]["low"],h[0]["volume"]),(10,21.5,22,9,12))

    def test_cost_includes_commission_spread_and_slippage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"c.json"
            p.write_text(json.dumps({"X":{"commission_bps":1.0,"standard_spread_usd":0.2,
                                          "commission_usd_round_trip":0.1}}))
            self.assertEqual(exness_data.cost_per_side_bps("X",0.5,p),2.0)  # .5 + 1 + .5


class OllamaReviewTests(unittest.TestCase):
    report={"strategies":[{"name":f"s{i}","validation":{"net_return":.01*i},"holdout":{"net_return":9}}
                          for i in range(7)]}

    def reply(self,content):
        return io.BytesIO(json.dumps({"message":{"content":json.dumps(content)}}).encode())

    def test_valid_pick_and_holdout_never_sent(self):
        sent={}
        def fake(req,timeout):
            sent["body"]=json.loads(req.data)
            return self.reply({"strategy":"s2","reason":"best evidence"})
        with patch("urllib.request.urlopen",fake):
            r=ollama_review.review(self.report,"m")
        self.assertEqual(r,{"mode":"advisory_only","strategy":"s2","reason":"best evidence"})
        self.assertNotIn("holdout",sent["body"]["messages"][1]["content"])

    def test_unknown_strategy_falls_back_to_wait(self):
        with patch("urllib.request.urlopen",lambda req,timeout:self.reply({"strategy":"s6","reason":"x"})):
            self.assertEqual(ollama_review.review(self.report,"m")["strategy"],"WAIT")

    def test_offline_ollama_falls_back_to_wait(self):
        def down(req,timeout): raise OSError("connection refused")
        with patch("urllib.request.urlopen",down):
            r=ollama_review.review(self.report,"m")
        self.assertEqual((r["strategy"],r["reason"]),("WAIT","OSError"))


if __name__=="__main__":
    unittest.main()
