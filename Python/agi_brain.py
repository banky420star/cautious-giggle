import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "ret_10",
    "rsi_14",
    "atr_14",
    "ema_12",
    "ema_26",
    "macd_line",
    "macd_signal",
    "bb_width_20",
    "stoch_k_14",
    "vol_z_20",
]


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Feature engineering section (MT5 OHLCV -> LSTM feature engine)
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    close = _as_series(out, "close")
    high = _as_series(out, "high")
    low = _as_series(out, "low")
    volume = _as_series(out, "volume")

    out["ret_1"] = close.pct_change().fillna(0.0)
    out["ret_5"] = close.pct_change(5).fillna(0.0)
    out["ret_10"] = close.pct_change(10).fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    out["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().fillna(0.0)
    out["bb_width_20"] = ((bb_std * 4.0) / (bb_mid.abs() + 1e-12)).fillna(0.0)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    out["stoch_k_14"] = (((close - low_14) / ((high_14 - low_14) + 1e-12)) * 100.0).fillna(50.0)

    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().fillna(0.0)
    out["vol_z_20"] = ((volume - vol_mean_20) / (vol_std_20 + 1e-12)).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return out


class AGIModel(nn.Module):
    def __init__(self, input_dim: int = len(FEATURE_COLUMNS)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 128, 3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])


class SmartAGI:
    def __init__(self):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.prediction_count = 0
        self.symbol_models = {}
        self._warned_missing_symbol = set()

        from Python.model_registry import ModelRegistry

        self.registry = ModelRegistry()
        self.active_dir = self.registry.load_active_model(prefer_canary=True)

        if self.active_dir:
            model_path = os.path.join(self.active_dir, "lstm_model.pth")
            scaler_path = os.path.join(self.active_dir, "lstm_scaler.pkl")
            logger.info(f"registry active model dir: {self.active_dir}")
        else:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            model_path = os.path.join(model_dir, "lstm_agi_trained.pt")
            scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")

        self.default_bundle = self._load_bundle(model_path, scaler_path, "default")

        # Backward-compatible aliases used by PPO feature extractor.
        self.model = self.default_bundle["model"]
        self.scaler = self.default_bundle["scaler"]
        self.scaler_loaded = self.default_bundle["scaler_loaded"]

    def _load_bundle(self, model_path: str, scaler_path: str, label: str):
        model = AGIModel().to(self.device)
        scaler = MinMaxScaler()
        scaler_loaded = False

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.eval()
                logger.success(f"AGI Brain loaded {label} model on {self.device.upper()}")
            except Exception as exc:
                logger.warning(f"{label} model load failed ({exc}); using fresh weights")
        else:
            logger.warning(f"no trained {label} model found at {model_path}; using fresh weights")

        if os.path.exists(scaler_path):
            import joblib

            try:
                scaler = joblib.load(scaler_path)
                scaler_loaded = True
                logger.success(f"loaded {label} feature scaler")
            except Exception as exc:
                logger.warning(f"{label} scaler load failed: {exc}")

        return {"model": model, "scaler": scaler, "scaler_loaded": scaler_loaded}

    def _symbol_artifact_paths(self, symbol: str):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        safe = symbol.replace("/", "_")

        symbol_active = self.registry.load_active_model(prefer_canary=True, symbol=symbol)
        if symbol_active:
            reg_model = os.path.join(symbol_active, "per_symbol", f"lstm_{safe}.pt")
            reg_scaler = os.path.join(symbol_active, "per_symbol", f"lstm_scaler_{safe}.pkl")
            if os.path.exists(reg_model) and os.path.exists(reg_scaler):
                return reg_model, reg_scaler

        if self.active_dir:
            reg_model = os.path.join(self.active_dir, "per_symbol", f"lstm_{safe}.pt")
            reg_scaler = os.path.join(self.active_dir, "per_symbol", f"lstm_scaler_{safe}.pkl")
            if os.path.exists(reg_model) and os.path.exists(reg_scaler):
                return reg_model, reg_scaler

        model = os.path.join(root, "models", "per_symbol", f"lstm_{safe}.pt")
        scaler = os.path.join(root, "models", "per_symbol", f"lstm_scaler_{safe}.pkl")
        return model, scaler

    def _bundle_for_symbol(self, symbol: str):
        if symbol in self.symbol_models:
            return self.symbol_models[symbol]

        model_path, scaler_path = self._symbol_artifact_paths(symbol)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            bundle = self._load_bundle(model_path, scaler_path, f"symbol[{symbol}]")
            self.symbol_models[symbol] = bundle
            return bundle

        if symbol not in self._warned_missing_symbol:
            logger.warning(f"no per-symbol artifacts for {symbol}; using default model")
            self._warned_missing_symbol.add(symbol)
        return self.default_bundle

    def predict(self, df: pd.DataFrame, production: bool = False) -> dict:
        self.prediction_count += 1

        symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) else "UNKNOWN"
        bundle = self._bundle_for_symbol(symbol)

        feat_df = build_feature_frame(df)
        if len(feat_df) < 60:
            return {"signal": "LOW_VOLATILITY", "confidence": 0.0, "symbol": symbol}

        features = feat_df[FEATURE_COLUMNS].astype(float).values

        scaler = bundle["scaler"]
        if bundle["scaler_loaded"] and hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) == features.shape[1]:
            data = scaler.transform(features)
        else:
            data = scaler.fit_transform(features)

        seq = torch.tensor(data[-60:].reshape(1, 60, len(FEATURE_COLUMNS)), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = bundle["model"](seq)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
            pred = int(np.argmax(probs)) if production else int(np.random.choice(3, p=probs))

        signal = ["LOW_VOLATILITY", "MED_VOLATILITY", "HIGH_VOLATILITY"][pred]
        confidence = round(float(probs[pred]), 4)

        return {"signal": signal, "confidence": confidence, "symbol": symbol}

    def extract_features(self, seq: torch.Tensor) -> torch.Tensor:
        seq = seq.to(self.device).float()
        self.model.train()

        expected = int(self.model.lstm.input_size)
        got = int(seq.shape[-1])
        if got < expected:
            pad = torch.zeros(seq.shape[0], seq.shape[1], expected - got, device=seq.device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=-1)
        elif got > expected:
            seq = seq[:, :, :expected]

        x, _ = self.model.lstm(seq)
        return x[:, -1, :]
