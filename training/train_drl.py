import datetime
import json
import os
import shutil
import sys

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import polars as pl
import torch
import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize

from analysis.gradient_flow_analyzer import LSTMGradientDiagnostics
from drl.adaptive_feature_extractor import AdaptiveLSTMFeatureExtractor
from drl.lstm_feature_extractor import LSTMFeatureExtractor
from drl.trading_env import TradingEnv
from Python.config_utils import load_project_config
from Python.data_feed import fetch_training_data, get_combined_training_df
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, normalize_feature_version
from Python.trade_learning import load_trade_memory
from alerts.telegram_alerts import TelegramAlerter

LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "ppo_training.log"), rotation="10 MB", level="INFO")


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _build_alerter(project_root: str):
    cfg_path = os.path.join(project_root, "config.yaml")
    cfg = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))


class EvalCallbackSaveVec(EvalCallback):
    def __init__(self, *args, vec_env=None, vec_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_env = vec_env
        self.vec_save_path = vec_save_path

    def _on_step(self) -> bool:
        old_best = self.best_mean_reward if self.best_mean_reward is not None else -np.inf
        cont = super()._on_step()
        if self.best_mean_reward is not None and self.best_mean_reward > old_best:
            if self.vec_env is not None and self.vec_save_path:
                os.makedirs(os.path.dirname(self.vec_save_path), exist_ok=True)
                self.vec_env.save(self.vec_save_path)
                logger.success(f"Saved VecNormalize with new best model -> {self.vec_save_path}")
        return cont


def _normalize_interval(interval: str | None) -> str:
    if not interval:
        return "5m"
    m = str(interval).strip().lower()
    if m.startswith("m") and m[1:].isdigit():
        return f"{m[1:]}m"
    if m.startswith("h") and m[1:].isdigit():
        return f"{m[1:]}h"
    return m


def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def get_mt5_equity(default_balance: float = 10000.0, cfg: dict | None = None) -> float:
    cfg = cfg or {}
    mt5_cfg = cfg.get("mt5", {})
    login = int(os.environ.get("MT5_LOGIN", mt5_cfg.get("login", 0)) or 0)
    password = os.environ.get("MT5_PASSWORD", mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER", mt5_cfg.get("server", ""))

    try:
        import MetaTrader5 as mt5

        if login and password and server:
            connected = mt5.initialize(login=login, password=password, server=server)
        else:
            connected = mt5.initialize()

        if connected:
            info = mt5.account_info()
            if info and float(info.equity) > 0:
                logger.info(f"Using MT5 equity from account {info.login}: {float(info.equity):.2f}")
                return float(info.equity)
    except Exception as e:
        logger.warning(f"Failed to pull MT5 equity, using default balance: {e}")

    return float(default_balance)


def make_env(
    df,
    seed: int,
    initial_balance: float,
    reward_weights: dict,
    trade_memory: dict | None = None,
    feature_version: str = ULTIMATE_150,
):
    def _init():
        set_random_seed(seed)
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df.copy()
        if "time" in pdf.columns:
            pdf["time"] = pd.to_datetime(pdf["time"], utc=True)
            pdf = pdf.sort_values("time").set_index("time")
        env = TradingEnv(
            pdf,
            initial_balance=initial_balance,
            reward_weights=reward_weights,
            trade_memory=trade_memory,
            feature_version=feature_version,
        )
        return Monitor(env)

    return _init


def _prepare_df(symbols: list[str], period: str, interval: str, per_symbol_mode: bool, candles: int, data_source: str | None) -> pd.DataFrame:
    if per_symbol_mode and len(symbols) == 1:
        df = fetch_training_data(
            symbols[0],
            period=period,
            interval=interval,
            strict=False,
            bars=int(candles),
            min_bars=int(candles),
            source=data_source,
        )
    else:
        df = get_combined_training_df(
            symbols,
            period=period,
            interval=interval,
            bars=int(candles),
            min_bars=int(candles),
            source=data_source,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(x) for x in col if x is not None and str(x) != ""]) for col in df.columns.to_list()]

    df.columns = [str(c) for c in df.columns]

    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

    df = df.loc[~df.index.duplicated(keep="last")].sort_index()
    df = df.reset_index(drop=False) if "time" not in df.columns else df.reset_index(drop=True)

    if df.isna().any().any():
        logger.warning("NaNs detected in historical data. Cleaning via ffill/bfill.")
        df = df.ffill().bfill()

    return df


def _is_vecnorm_compatible(vec_path: str, feature_version: str) -> bool:
    try:
        dummy = DummyVecEnv([lambda: TradingEnv(feature_version=feature_version)])
        _ = VecNormalize.load(vec_path, dummy)
        return True
    except Exception:
        return False


def _default_ppo_params() -> dict:
    return {
        "learning_rate": 1e-4,
        "n_steps": 4096,
        "batch_size": 512,
        "n_epochs": 10,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": 0.01,
        "use_sde": True,
        "sde_sample_freq": 4,
    }


def _policy_kwargs_for(feature_version: str) -> dict:
    if feature_version == ULTIMATE_150:
        return dict(
            features_extractor_class=AdaptiveLSTMFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=256, window_size=100),
            net_arch=[512, 256],
            activation_fn=torch.nn.ReLU,
        )
    return dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[512, 256],
        activation_fn=torch.nn.ReLU,
    )


def _build_model(env, feature_version: str, ppo_params: dict):
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=_policy_kwargs_for(feature_version),
        learning_rate=linear_schedule(ppo_params["learning_rate"]),
        n_steps=ppo_params["n_steps"],
        batch_size=ppo_params["batch_size"],
        n_epochs=ppo_params["n_epochs"],
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params["gae_lambda"],
        clip_range=ppo_params["clip_range"],
        ent_coef=ppo_params["ent_coef"],
        vf_coef=ppo_params["vf_coef"],
        max_grad_norm=ppo_params["max_grad_norm"],
        target_kl=ppo_params["target_kl"],
        use_sde=ppo_params["use_sde"],
        sde_sample_freq=ppo_params["sde_sample_freq"],
        tensorboard_log=os.path.join(LOG_DIR, "drl_joint"),
        device="cuda"
        if torch.cuda.is_available()
        else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"),
        verbose=1,
    )


def _maybe_optimize_ppo_params(df_pd: pd.DataFrame, cfg: dict, initial_balance: float, reward_weights: dict, trade_memory: dict | None, feature_version: str) -> dict:
    drl_cfg = cfg.get("drl", {}) or {}
    trials = int(drl_cfg.get("optuna_trials", 0) or 0)
    if trials <= 0:
        return _default_ppo_params()

    try:
        import optuna
    except Exception as exc:
        logger.warning(f"Optuna disabled because the package is unavailable: {exc}")
        return _default_ppo_params()

    timesteps = int(drl_cfg.get("optuna_timesteps", min(25_000, max(5_000, int(drl_cfg.get("total_timesteps", 100_000)) // 5))) or 10_000)
    sample_rows = min(len(df_pd), max(2_000, int(drl_cfg.get("optuna_rows", 10_000) or 10_000)))
    sample_df = df_pd.tail(sample_rows).copy()
    df = pl.from_pandas(sample_df)

    def objective(trial):
        params = _default_ppo_params()
        params["learning_rate"] = trial.suggest_float("learning_rate", 3e-5, 5e-4, log=True)
        params["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        params["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 2e-2, log=True)
        params["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)

        env = DummyVecEnv([make_env(df, 11, initial_balance, reward_weights, trade_memory=trade_memory, feature_version=feature_version)])
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        eval_env = DummyVecEnv([make_env(df, 99, initial_balance, reward_weights, trade_memory=trade_memory, feature_version=feature_version)])
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        eval_env.obs_rms = env.obs_rms
        eval_env.training = False
        eval_env.norm_reward = False

        model = _build_model(env, feature_version, params)
        callback = EvalCallback(eval_env, best_model_save_path=None, log_path=None, eval_freq=max(1_000, timesteps // 4), deterministic=True, render=False)
        model.learn(total_timesteps=timesteps, callback=callback, progress_bar=False)
        score = float(callback.best_mean_reward) if callback.best_mean_reward is not None else -1e9
        env.close()
        eval_env.close()
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = _default_ppo_params()
    if study.best_trial:
        best.update(study.best_trial.params)
        logger.info(f"Optuna best params selected: {study.best_trial.params}")
    return best


def _stage_candidate(
    symbols,
    total_timesteps,
    period,
    interval,
    reward_cfg,
    df_rows,
    ppo_params,
    eval_windows,
    feature_version,
    data_source,
    src_model_path: str | None = None,
    src_vec_path: str | None = None,
):
    from Python.model_registry import ModelRegistry

    registry = ModelRegistry()
    best_dir = os.path.join("models", "best_eval_models")
    src_model = src_model_path or os.path.join(best_dir, "best_model.zip")
    src_vec = src_vec_path or os.path.join(best_dir, "vec_normalize.pkl")

    if not os.path.exists(src_model) or not os.path.exists(src_vec):
        raise RuntimeError("Missing best_model.zip or vec_normalize.pkl after training")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate_path = os.path.join(registry.candidates_dir, timestamp)
    os.makedirs(candidate_path, exist_ok=True)

    shutil.copy2(src_model, os.path.join(candidate_path, "ppo_trading.zip"))
    shutil.copy2(src_vec, os.path.join(candidate_path, "vec_normalize.pkl"))

    meta = {
        "type": "ppo",
        "symbol": symbols[0] if len(symbols) == 1 else None,
        "symbols": symbols,
        "timeframe": str(interval),
        "period": str(period),
        "candles": int(df_rows),
        "timesteps": int(total_timesteps),
        "data_source": str(data_source or "mt5"),
        "feature_set_version": str(feature_version),
        "normalization_version": "vecnorm_v1",
        "reward": reward_cfg,
        "reward_version": str(reward_cfg.get("version", "v2_risk_adjusted")),
        "ppo_params": ppo_params,
        "policy_extractor": "adaptive_lstm" if feature_version == ULTIMATE_150 else "agi_lstm",
        "window_size": 100,
        "windows": {
            "train": str(period),
            "validate": str(eval_windows.get("validate", "120d")),
            "forward": list(eval_windows.get("forward", [])),
        },
        "source": "EvalCallback best_model.zip + matching VecNormalize",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(os.path.join(candidate_path, "scorecard.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(candidate_path, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.success(f"Candidate staged to: {candidate_path}")
    return candidate_path


def _train_once(symbols: list[str], cfg: dict, total_timesteps: int, initial_balance: float, alerter=None):
    drl_cfg = cfg.get("drl", {})
    trading_cfg = cfg.get("trading", {})

    period = str(drl_cfg.get("period", "90d"))
    interval = _normalize_interval(drl_cfg.get("interval", trading_cfg.get("timeframe", "M5")))
    candles = int(drl_cfg.get("candles_per_symbol", 100000))
    reward_cfg = drl_cfg.get("reward", {}) if isinstance(drl_cfg.get("reward", {}), dict) else {}
    reward_weights = reward_cfg.get("weights", {}) if isinstance(reward_cfg.get("weights", {}), dict) else {}
    logs_root = os.path.join(os.getcwd(), "logs", "learning")
    symbol_hint = symbols[0] if len(symbols) == 1 else None
    trade_memory = load_trade_memory(logs_root, symbol=symbol_hint)
    feature_version = normalize_feature_version(
        os.environ.get("AGI_FEATURE_VERSION") or drl_cfg.get("feature_version", ULTIMATE_150),
        default=ULTIMATE_150,
    )
    data_source = drl_cfg.get("data_source")

    per_symbol_mode = len(symbols) == 1
    logger.info(
        f"DRL Training | symbols={symbols} | timesteps={total_timesteps:,} | period={period} | tf={interval} | candles={candles:,} | per_symbol={per_symbol_mode} | initial_balance={initial_balance:.2f} | features={feature_version} | source={data_source or 'mt5'}"
    )
    if alerter is not None:
        try:
            alerter.training(
                "PPO",
                f"Start {symbols} | timesteps={total_timesteps:,} | period={period} | tf={interval} | candles={candles:,} | features={feature_version}",
            )
        except Exception:
            pass

    df_pd = _prepare_df(symbols, period=period, interval=interval, per_symbol_mode=per_symbol_mode, candles=candles, data_source=data_source)
    if df_pd.empty:
        raise RuntimeError("No valid training data found")

    df = pl.from_pandas(df_pd)
    n_envs = 4

    env = DummyVecEnv(
        [make_env(df, i, initial_balance=initial_balance, reward_weights=reward_weights, trade_memory=trade_memory, feature_version=feature_version) for i in range(n_envs)]
    )
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = DummyVecEnv(
        [make_env(df, 99, initial_balance=initial_balance, reward_weights=reward_weights, trade_memory=trade_memory, feature_version=feature_version)]
    )
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    eval_env.obs_rms = env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

    ppo_params = _maybe_optimize_ppo_params(df_pd, cfg, initial_balance, reward_weights, trade_memory, feature_version)
    model = _build_model(env, feature_version, ppo_params)

    best_dir = os.path.join("models", "best_eval_models")
    os.makedirs(best_dir, exist_ok=True)
    best_vec_path = os.path.join(best_dir, "vec_normalize.pkl")

    eval_callback = EvalCallbackSaveVec(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        vec_env=env,
        vec_save_path=best_vec_path,
    )

    grad_callback = LSTMGradientDiagnostics()

    logger.info("Starting PPO training")
    model.learn(total_timesteps=total_timesteps, callback=[eval_callback, grad_callback], progress_bar=True)
    best_score = float(eval_callback.best_mean_reward) if eval_callback.best_mean_reward is not None else None

    latest_dir = os.path.join("models", "latest_run")
    os.makedirs(latest_dir, exist_ok=True)
    latest_model = os.path.join(latest_dir, "latest_model.zip")
    latest_vec = os.path.join(latest_dir, "latest_vec_normalize.pkl")
    model.save(latest_model)
    env.save(latest_vec)

    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg.get("evaluation", {}), dict) else {}
    eval_windows = {
        "validate": str(drl_cfg.get("eval_period", "120d")),
        "forward": eval_cfg.get("forward_windows", []),
    }

    stage_model = latest_model
    stage_vec = latest_vec
    if not _is_vecnorm_compatible(stage_vec, feature_version=feature_version):
        best_model = os.path.join(best_dir, "best_model.zip")
        best_vec = os.path.join(best_dir, "vec_normalize.pkl")
        if os.path.exists(best_model) and os.path.exists(best_vec) and _is_vecnorm_compatible(best_vec, feature_version=feature_version):
            stage_model = best_model
            stage_vec = best_vec

    candidate_path = _stage_candidate(
        symbols,
        total_timesteps,
        period,
        interval,
        reward_cfg,
        df_rows=len(df_pd),
        ppo_params=ppo_params,
        eval_windows=eval_windows,
        feature_version=feature_version,
        data_source=data_source,
        src_model_path=stage_model,
        src_vec_path=stage_vec,
    )
    if alerter is not None:
        try:
            alerter.training(
                "PPO",
                f"Complete {symbols} | best_score={best_score if best_score is not None else 'n/a'} | candidate={candidate_path}",
            )
        except Exception:
            pass


def train_drl():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg = load_project_config(project_root, live_mode=False)

    symbols = cfg.get("trading", {}).get("symbols", ["EURUSDm"])
    if not symbols:
        symbols = ["EURUSDm"]

    one_symbol = os.environ.get("AGI_DRL_SYMBOL")
    if one_symbol:
        symbols = [one_symbol]

    total_timesteps = int(os.environ.get("AGI_DRL_TIMESTEPS", cfg.get("drl", {}).get("total_timesteps", 100_000)))
    initial_balance = get_mt5_equity(default_balance=10000.0, cfg=cfg)

    per_symbol = bool(cfg.get("drl", {}).get("per_symbol", True))
    alerter = _build_alerter(project_root)
    if one_symbol:
        _train_once(symbols, cfg, total_timesteps, initial_balance, alerter=alerter)
        return

    if per_symbol:
        for symbol in symbols:
            _train_once([symbol], cfg, total_timesteps, initial_balance, alerter=alerter)
    else:
        _train_once(symbols, cfg, total_timesteps, initial_balance, alerter=alerter)


if __name__ == "__main__":
    train_drl()
