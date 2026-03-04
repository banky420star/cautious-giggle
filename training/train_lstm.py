import os
import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.agi_brain import AGIModel, FEATURE_COLUMNS, build_feature_frame
from Python.data_feed import fetch_training_data

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="INFO")


def create_sequences(data: np.ndarray, close_col: int, atr_col: int, rsi_col: int, seq_len: int = 60):
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len : i])

        prev_close = data[i - 1, close_col]
        next_close = data[i, close_col]
        future_ret = (next_close - prev_close) / (abs(prev_close) + 1e-8)

        atr_norm = abs(data[i, atr_col] / (abs(next_close) + 1e-8))
        rsi = data[i, rsi_col]

        up_thr = max(0.0007, atr_norm * 0.35)
        dn_thr = -up_thr

        if future_ret > up_thr and rsi > 52:
            y.append(1)
        elif future_ret < dn_thr and rsi < 48:
            y.append(2)
        else:
            y.append(0)

    return np.array(X), np.array(y)


def _train_one_symbol(symbol: str, epochs: int, seq_len: int, device: str, out_dir: str):
    df = fetch_training_data(symbol, period="60d")
    if df.empty or len(df) < seq_len + 50:
        logger.warning(f"insufficient data for {symbol}, skipping")
        return None

    fdf = build_feature_frame(df)
    if len(fdf) < seq_len + 50:
        logger.warning(f"insufficient engineered rows for {symbol}, skipping")
        return None

    feat = fdf[FEATURE_COLUMNS].values.astype(np.float32)
    scaler = MinMaxScaler()
    feat_scaled = scaler.fit_transform(feat)

    close_col = FEATURE_COLUMNS.index("close")
    atr_col = FEATURE_COLUMNS.index("atr_14")
    rsi_col = FEATURE_COLUMNS.index("rsi_14")

    X, y = create_sequences(feat_scaled, close_col, atr_col, rsi_col, seq_len=seq_len)
    if len(X) == 0:
        logger.warning(f"no sequences for {symbol}")
        return None

    model = AGIModel(input_dim=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    batch_size = 64
    n_batches = max(1, len(X_tensor) // batch_size)

    model.train()
    last_loss = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(X_tensor))
        X_epoch = X_tensor[perm]
        y_epoch = y_tensor[perm]

        correct = 0
        total = 0
        epoch_loss = 0.0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_epoch))
            xb = X_epoch[start:end]
            yb = y_epoch[start:end]
            if len(xb) == 0:
                continue

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(yb.size(0))

        last_loss = epoch_loss / max(1, n_batches)
        acc = (correct / max(1, total)) * 100.0
        logger.info(f"{symbol} | epoch {epoch + 1}/{epochs} | loss {last_loss:.4f} | acc {acc:.2f}%")

    os.makedirs(out_dir, exist_ok=True)
    safe = symbol.replace("/", "_")
    model_path = os.path.join(out_dir, f"lstm_{safe}.pt")
    scaler_path = os.path.join(out_dir, f"lstm_scaler_{safe}.pkl")

    torch.save(model.state_dict(), model_path)

    import joblib

    joblib.dump(scaler, scaler_path)

    return {
        "symbol": symbol,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "loss": last_loss,
        "samples": int(len(X)),
    }


def train_lstm(symbols=None, epochs=20, seq_len=60):
    if symbols is None:
        symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"]

    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    logger.info(f"LSTM per-symbol training on {device.upper()} | symbols={symbols} | epochs={epochs}")

    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    per_symbol_dir = os.path.join(model_dir, "per_symbol")

    results = []
    for symbol in symbols:
        res = _train_one_symbol(symbol, epochs=epochs, seq_len=seq_len, device=device, out_dir=per_symbol_dir)
        if res:
            results.append(res)

    if not results:
        logger.error("no symbol models trained")
        return

    best = sorted(results, key=lambda x: x["loss"])[0]
    # Keep backward-compatible default artifacts by linking to best per-symbol model.
    import shutil

    shutil.copy2(best["model_path"], os.path.join(model_dir, "lstm_agi_trained.pt"))
    shutil.copy2(best["scaler_path"], os.path.join(model_dir, "lstm_scaler.pkl"))
    logger.success(f"default lstm artifacts now point to best symbol model: {best['symbol']}")


if __name__ == "__main__":
    import yaml

    cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    symbols = None
    epochs = 20

    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        symbols = cfg.get("trading", {}).get("symbols")
        epochs = int(cfg.get("training", {}).get("lstm_epochs", 20))

    train_lstm(symbols=symbols, epochs=epochs)
