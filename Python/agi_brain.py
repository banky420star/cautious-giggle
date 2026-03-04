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
    "rsi_14",
    "atr_14",
    "ema_12",
    "ema_26",
]


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated(keep="first")]

    close = _as_series(out, "close")
    high = _as_series(out, "high")
    low = _as_series(out, "low")

    out["ret_1"] = close.pct_change().fillna(0.0)

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
        self.model = AGIModel()
        self.scaler = MinMaxScaler()

        if torch.cuda.is_available():
            self.device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.model.to(self.device)
        self.prediction_count = 0

        from Python.model_registry import ModelRegistry

        registry = ModelRegistry()
        active_dir = registry.load_active_model(prefer_canary=True)

        if active_dir:
            model_path = os.path.join(active_dir, "lstm_model.pth")
            scaler_path = os.path.join(active_dir, "lstm_scaler.pkl")
            logger.info(f"registry active model dir: {active_dir}")
        else:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            model_path = os.path.join(model_dir, "lstm_agi_trained.pt")
            scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")

        if os.path.exists(model_path):
            try:
                state = torch.load(model_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state)
                self.model.eval()
                logger.success(f"AGI Brain loaded trained model on {self.device.upper()}")
            except Exception as exc:
                logger.warning(f"model load failed ({exc}); using fresh weights")
        else:
            logger.warning(f"no trained model found at {model_path}; using fresh weights")

        self.scaler_loaded = False
        if os.path.exists(scaler_path):
            import joblib

            try:
                self.scaler = joblib.load(scaler_path)
                self.scaler_loaded = True
                logger.success("loaded persistent feature scaler")
            except Exception as exc:
                logger.warning(f"scaler load failed: {exc}")

    def predict(self, df: pd.DataFrame, production: bool = False) -> dict:
        self.prediction_count += 1

        feat_df = build_feature_frame(df)
        if len(feat_df) < 60:
            return {"signal": "LOW_VOLATILITY", "confidence": 0.0, "symbol": df["symbol"].iloc[0]}

        features = feat_df[FEATURE_COLUMNS].astype(float).values

        if self.scaler_loaded and hasattr(self.scaler, "n_features_in_") and int(self.scaler.n_features_in_) == features.shape[1]:
            data = self.scaler.transform(features)
        else:
            data = self.scaler.fit_transform(features)

        seq = torch.tensor(data[-60:].reshape(1, 60, len(FEATURE_COLUMNS)), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(seq)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
            pred = int(np.argmax(probs)) if production else int(np.random.choice(3, p=probs))

        signal = ["LOW_VOLATILITY", "MED_VOLATILITY", "HIGH_VOLATILITY"][pred]
        confidence = round(float(probs[pred]), 4)

        return {"signal": signal, "confidence": confidence, "symbol": df["symbol"].iloc[0]}

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

