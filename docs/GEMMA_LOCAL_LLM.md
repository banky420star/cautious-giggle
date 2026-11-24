# Quantized Gemma as a local LLM (research note)

Added 2026-11-24. Research only; nothing in the bot uses Gemma yet.

## Where it would fit here

The only LLM in the stack today is the **news_llm** external voter
(`Python/external_voters.py`, `_NewsLLM`). It scores the latest headlines per market through a
local Ollama server and casts a vote in [-1, 1] with weight 4 (`config.yaml`,
`external_voters.news_llm`, currently `model: "qwen3:4b"`, every 20 min, 6 h lookback).

A quantized Gemma 4 QAT checkpoint would be a drop-in candidate for that voter: pull it in Ollama
and change `news_llm.model`. Because each voter is tracked by model memory (`ind:news_llm`), a
worse model gets down-weighted on its own, but compare it against qwen3:4b on the same headlines
before switching.

## Background

Quantized Gemma models, especially Gemma 4 Quantization-Aware Training (QAT) checkpoints, run
locally on consumer hardware for research, sentiment analysis and agent workflows with no API fees.
There is no official "Gemma Trading" model; using Gemma for trading means pairing a quantized
Gemma 4 (or Gemma 3) checkpoint with your own prompts, agents and local runtime.
Sources: [Google blog on Gemma 4 QAT](https://blog.google/innovation-and-ai/technology/developers-tools/quantization-aware-training-gemma-4/),
[Unsloth Gemma 4 QAT docs](https://unsloth.ai/docs/models/gemma-4/qat),
[DeepMind Gemma](https://deepmind.google/models/gemma/),
[video overview](https://www.youtube.com/watch?v=_EMHgCPsy2M&t=186).

### How it fits trading work

- **Local, low cost:** QAT compresses weights (for example 4-bit `Q4_0`, or mobile-optimized
  schemes) and keeps accuracy better than post-training quantization. Bit-width breakdown:
  [Gemma 4 Quantization Guide](https://gemma4all.com/blog/gemma-4-quantization-guide).
- **Reasoning vs speed:** larger or reasoning-enabled Gemma 4 builds suit multi-step research or
  agents; smaller checkpoints give fast, direct answers when latency matters
  ([LM Studio gemma-4-12b-qat](https://lmstudio.ai/models/google/gemma-4-12b-qat)).
- **Runtimes:** Ollama, llama.cpp, LM Studio, MLX, vLLM and Unsloth all run the quantized
  checkpoints.

### Quantization tradeoffs

- **QAT vs PTQ:** QAT simulates low-bit rounding during training so the weights adapt; PTQ
  compresses already-trained weights afterwards. On Gemma 4 E2B, going below 4-bit (`Q3_K_M`,
  `Q2_K`) noticeably hurts accuracy or stops the model answering
  ([BF16 to Q2_K tradeoffs](https://dev.to/gde/accuracy-vs-speed-gemma-4-e2b-quantization-tradeoffs-bf16-to-q2k-aea)).
  Stay at 4-bit or above.
- **Memory:** the smallest edge variants need about 1 GB; larger quantized variants such as
  `26B-A4B` want roughly 16 GB class hardware.

### Risk note

Use the LLM to summarize news, parse filings and assist research, not to place live trades on
its own. Any Gemma voter goes through the same backtesting, cost/trend gates and human oversight
as every other voter.

## Built: Gemma as a Smart Money Concepts analyst (2026-11-24)

- `Python/smc.py` finds the SMC structure with plain code (no lookahead): H1 + M15 BOS/CHoCH,
  order blocks, unfilled FVGs, equal highs/lows, liquidity sweeps, premium/discount of the
  current leg. `rules_vote` scores it with fixed rules; `describe` turns it into text.
- Two new external voters in `Python/external_voters.py`:
  - `smc_rules`: the fixed-rule score. Enabled in `config.yaml` in **shadow** (weight 0).
  - `smc_gemma`: Gemma (`gemma4:e4b-it-qat` via Ollama) reads the same facts and returns
    score/setup/invalidation/reason. A long with its invalidation above price (or the mirror) is
    thrown out. Disabled until the model is pulled.
- Shadow means model memory scores the voter (`ind:smc_rules`, `ind:smc_gemma`) but the
  council gives it no say. Raise its weight in `external_voters.weights` only once memory and
  `tools/smc_eval.py` show it works.

### Backtest of the fixed rules (`tools/smc_eval.py`, ~1 month of M15 per market, |vote| > 0.2)

| Market | 2h hit rate | 2h edge (ATR) | t |
|---|---|---|---|
| XAUUSD | 51.2% | +0.066 | 1.4 |
| BTCUSD | 49.0% | +0.032 | 0.6 |
| ETHUSD | 48.2% | +0.035 | 0.7 |
| EURUSD | 47.0% | -0.122 | -2.4 |
| USDJPY | 48.0% | +0.058 | 1.1 |
| XAGUSD | 50.0% | +0.010 | 0.2 |
| USOIL | 47.8% | -0.024 | -0.5 |

No market shows a reliable edge, and EURUSD is significantly *wrong*. Textbook SMC rules alone
don't predict these markets, so Gemma has to beat this baseline before it gets weight:
`.venv/bin/python tools/smc_eval.py --gemma 60` compares both on the same bars.
