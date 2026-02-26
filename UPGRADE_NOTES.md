# 🚀 Next Upgrade Path: Continuous Position Management

> **Note:** The current `HybridBrain` (V1 Autonomy) uses PPO only to pick a **direction** (BUY / SELL / HOLD) from its continuous position action. 

That logic is valid and safe for getting started, but there is a slight mismatch in the engine matrix: 
- Your RL environment (`TradingEnv`) is **continuous-position based** (meaning it conceptually holds and scales a single position from -1.0 to 1.0).
- Your live executor (`MT5Executor`) currently uses **discrete market orders** (meaning it opens a brand new ticket every time it gets a signal).

It’s "close enough" for V1, but to achieve **higher fidelity** (where the live bot perfectly mirrors the RL environment), the next architectural upgrade must be implemented:

## The Translation Layer Upgrade

Instead of reading the PPO output and immediately opening a new trade ticket, the `HybridBrain` should keep the PPO output as a **target exposure level**, and translate it dynamically:

1. **Increase / Decrease Position**: Instead of always opening new trades, calculate the delta. If PPO says `0.5` (Buy) and we are currently at `0.2`, we only buy `0.3` more.
2. **Track and Modify**: Continuously monitor the open ticket volume and modify it to match the PPO tensor target.

---

### Action Required for V2

If you want to implement this higher-fidelity version next:
1. Paste your **MT5 position management logic** (if you have any written).
2. Or confirm you want to stick with **"one-shot market orders only"** for now while the Canary models prove profitability.

*For now, sticking to discrete, one-shot orders is the safest way to validate the pipeline before introducing complex netting/hedging calculations across live broker environments.*
