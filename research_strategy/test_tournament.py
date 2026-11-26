import unittest
from research_strategy.tournament import signals,evaluate,tournament

def fixture(n=400):
    return [{"open":100+i*.1,"close":100+i*.1+.02,"high":101+i*.1,"low":99+i*.1,"volume":100+i%7} for i in range(n)]

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

if __name__=="__main__":
    unittest.main()
