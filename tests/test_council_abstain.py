from Python.indicator_council import council_vote


def test_external_zero_abstains_but_indicator_zero_counts():
    w = {"ob_gold": 8.0, "news_llm": 4.0}
    votes = {"rsi": 0.5, "macd": 0.0, "ob_gold": 0.0, "news_llm": 0.1}
    r = council_vote(votes, {}, weights=w)
    assert r["abstained"] == ["ob_gold"]                       # no zone near price: sits out
    # rsi 0.5 + macd 0 (neutral indicator, counts) + news 0.1 x4 -> (0.5 + 0.4) / 6
    assert abs(r["raw"] - 0.9 / 6) < 1e-9
    assert abs(r["target_zeros_counted"] - __import__("math").tanh(2 * 0.9 / 14)) < 1e-9
    old = council_vote(votes, {}, weights=w, abstain_zero=False)          # the previous rule, still available
    assert old["abstained"] == [] and abs(old["target"] - r["target_zeros_counted"]) < 1e-12


def test_external_opinion_still_counts_full_weight():
    r = council_vote({"rsi": 0.2, "ob_gold": -1.0}, {}, weights={"ob_gold": 8.0})
    assert r["abstained"] == [] and abs(r["raw"] - (0.2 - 8.0) / 9) < 1e-9


def test_all_external_silent_falls_back_to_indicators():
    r = council_vote({"rsi": 0.3, "ob_gold": 0.0, "news_llm": 0.0}, {}, weights={"ob_gold": 8, "news_llm": 4})
    assert abs(r["raw"] - 0.3) < 1e-9
