import os
import glob
import pandas as pd
from loguru import logger
import asyncio

# The user will use the OpenAI client natively formatted for xAI API
from openai import AsyncOpenAI

# 2026 XAI Python SDK integration
class GrokTradingAgent:
    """
    Phase C: Fully Autonomous Grok Self-Improver Swarm.
    Reads recent Walk-Forward test results and Gradient Flow metrics to auto-tune the PPO+LSTM brain.
    """
    def __init__(self):
        # We rely on standardized OpenAI compatible clients for Grok
        self.api_key = os.getenv("XAI_API_KEY", "dummy_key_until_configured")
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url="https://api.x.ai/v1"
        )
        self.system_prompt = """You are Grok, an elite 2026 quantitative trading AI architect managing the cautious-giggle hedge fund.
Your job is to read Walk-Forward testing reports and LSTM Gradient Flow patterns, and output EXACT Python adjustments (hyperparameters, reward function tweaks, or learning rate changes) to improve out-of-sample Sharpe Ratio and Calmar Ratio.
Only output Python code blocks or direct numerical suggestions. Respect the RiskEngine parameters and never suggest removing Max Drawdown limits!"""

    def _get_latest_walkforward_data(self) -> str:
        try:
            reports = glob.glob(os.path.join("logs", "walkforward_report_*.csv"))
            if not reports:
                return "No walk-forward reports available yet."
            latest_report = max(reports, key=os.path.getctime)
            df = pd.read_csv(latest_report)
            avg_sharpe = df['sharpe'].mean()
            avg_dd = df['max_drawdown'].mean()
            logger.info(f"Loaded Walk-Forward Report: {latest_report}")
            return f"Latest OOS Report averages - Sharpe: {avg_sharpe:.2f}, Max Drawdown: {avg_dd:.2%} \n" + df.to_string()
        except Exception as e:
            return f"Error reading walk-forward data: {e}"

    def _get_latest_gradient_data(self) -> str:
        try:
            grad_file = "logs/ppo_training.log"
            if not os.path.exists(grad_file):
                return "Gradient Flow logs not found."
            with open(grad_file, "r") as f:
                # Grab the last 50 lines of logs to see final gradient flow
                lines = f.readlines()[-50:]
            return "Recent Gradient Flow & Training Logs:\n" + "".join(lines)
        except Exception as e:
            return f"Error reading gradient data: {e}"

    async def analyze_and_improve(self) -> str:
        logger.info("🧠 Grok Swarm activated: Gathering system diagnostic data...")
        wf_data = self._get_latest_walkforward_data()
        grad_data = self._get_latest_gradient_data()

        prompt = f"""
        Here is the latest data from the Joint LSTM-PPO training execution:
        
        === WALK-FORWARD OUT-OF-SAMPLE TEST ===
        {wf_data}
        
        === GRADIENT FLOW DIAGNOSTICS ===
        {grad_data}
        
        Task: Analyze the gradients (vanishing? exploding? steady?) and the Out-of-Sample Sharpe/Returns.
        If the LSTM gradients are too small (<1e-6), suggest increasing the LSTM learning rate in `train_drl.py`.
        If the Sharpe is under 1.5, suggest specific modifications to the `TradingEnv` reward function.
        Provide a concise analysis followed by the EXACT lines of Python code to adjust in `train_drl.py` or `trading_env.py`.
        """
        
        logger.info("📡 Transmitting telemetry to Grok Matrix...")
        try:
            response = await self.client.chat.completions.create(
                model="grok-3", # Upgraded 2026 reasoning model natively provided
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            suggestion = response.choices[0].message.content
            logger.success("✅ Grok analysis received.")
            return suggestion
        except Exception as e:
            logger.error(f"XAI API Error (Ensure XAI_API_KEY is exported): {e}")
            return ""

# Execution block
if __name__ == "__main__":
    agent = GrokTradingAgent()
    analysis_patch = asyncio.run(agent.analyze_and_improve())
    print("\n" + "="*80)
    print("🧠 GROK SWARM SELF-IMPROVER PATCH PROPOSAL:")
    print("="*80)
    print(analysis_patch)
    print("="*80)
