import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from Python.feature_pipeline import ENGINEERED_V2, ENGINEERED_LSTM_COLUMNS, build_lstm_feature_frame

FEATURE_COLUMNS = list(ENGINEERED_LSTM_COLUMNS)


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    features, _ = build_lstm_feature_frame(df, feature_version=ENGINEERED_V2)
    return features


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
        self._warned_incompatible_symbol = set()

        from Python.model_registry import ModelRegistry

        self.registry = ModelRegistry()
        self.active_dir = self._resolve_registry_default_dir()

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

    def _resolve_registry_default_dir(self) -> str | None:
        preferred = self.registry.load_active_model(prefer_canary=True)
        if self._has_default_bundle(preferred):
            return preferred

        champion = self.registry.load_active_model(prefer_canary=False)
        if self._has_default_bundle(champion):
            return champion
        return None

    @staticmethod
    def _has_default_bundle(candidate_dir: str | None) -> bool:
        if not candidate_dir:
            return False
        model_path = os.path.join(candidate_dir, "lstm_model.pth")
        scaler_path = os.path.join(candidate_dir, "lstm_scaler.pkl")
        return os.path.exists(model_path) and os.path.exists(scaler_path)

    def _load_bundle(self, model_path: str, scaler_path: str, label: str):
        feature_columns = list(FEATURE_COLUMNS)
        feature_version = ENGINEERED_V2
        metadata_path = os.path.splitext(model_path)[0] + ".meta.json"
        if os.path.exists(metadata_path):
            try:
                import json

                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f) or {}
                cols = metadata.get("feature_columns")
                if isinstance(cols, list) and cols:
                    feature_columns = [str(col) for col in cols]
                feature_version = str(metadata.get("feature_version", ENGINEERED_V2) or ENGINEERED_V2)
            except Exception as exc:
                logger.warning(f"{label} metadata load failed: {exc}")

        model = AGIModel(input_dim=len(feature_columns)).to(self.device)
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

        return {
            "model": model,
            "scaler": scaler,
            "scaler_loaded": scaler_loaded,
            "feature_columns": feature_columns,
            "feature_version": feature_version,
        }

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

    def _is_compatible_lstm_model(self, model_path: str) -> bool:
        if not os.path.exists(model_path):
            return False
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
            w = state.get("lstm.weight_ih_l0")
            if w is None or len(w.shape) != 2:
                return False
            metadata_path = os.path.splitext(model_path)[0] + ".meta.json"
            expected = len(FEATURE_COLUMNS)
            if os.path.exists(metadata_path):
                try:
                    import json

                    with open(metadata_path, "r", encoding="utf-8") as f:
                        metadata = json.load(f) or {}
                    cols = metadata.get("feature_columns")
                    if isinstance(cols, list) and cols:
                        expected = len(cols)
                except Exception:
                    pass
            got = int(w.shape[1])
            return got == expected
        except Exception:
            return False

    def _bundle_for_symbol(self, symbol: str):
        if symbol in self.symbol_models:
            return self.symbol_models[symbol]

        model_path, scaler_path = self._symbol_artifact_paths(symbol)
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            if not self._is_compatible_lstm_model(model_path):
                if symbol not in self._warned_incompatible_symbol:
                    try:
                        bad_model = model_path + ".incompatible"
                        if os.path.exists(model_path) and not os.path.exists(bad_model):
                            os.replace(model_path, bad_model)
                        bad_scaler = scaler_path + ".incompatible"
                        if os.path.exists(scaler_path) and not os.path.exists(bad_scaler):
                            os.replace(scaler_path, bad_scaler)
                    except Exception:
                        pass
                    logger.info(
                        f"incompatible per-symbol model for {symbol} (feature shape mismatch); using default model"
                    )
                    self._warned_incompatible_symbol.add(symbol)
                return self.default_bundle
            bundle = self._load_bundle(model_path, scaler_path, f"symbol[{symbol}]")
            self.symbol_models[symbol] = bundle
            return bundle

        return self.default_bundle

    def predict(self, df: pd.DataFrame, production: bool = False) -> dict:
        self.prediction_count += 1

        symbol = str(df["symbol"].iloc[0]) if "symbol" in df.columns and len(df) else "UNKNOWN"
        bundle = self._bundle_for_symbol(symbol)

        feature_version = str(bundle.get("feature_version", ENGINEERED_V2) or ENGINEERED_V2)
        feat_df, available_columns = build_lstm_feature_frame(df, feature_version=feature_version)
        if len(feat_df) < 60:
            return {"signal": "LOW_VOLATILITY", "confidence": 0.0, "symbol": symbol}

        bundle_columns = [str(col) for col in bundle.get("feature_columns", FEATURE_COLUMNS)]
        use_columns = bundle_columns if set(bundle_columns).issubset(set(available_columns)) else available_columns
        features = feat_df[use_columns].astype(float).values

        scaler = bundle["scaler"]
        if bundle["scaler_loaded"] and hasattr(scaler, "n_features_in_") and int(scaler.n_features_in_) == features.shape[1]:
            data = scaler.transform(features)
        else:
            data = scaler.fit_transform(features)

        seq = torch.tensor(data[-60:].reshape(1, 60, features.shape[1]), dtype=torch.float32).to(self.device)

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
