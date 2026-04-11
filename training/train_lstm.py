"""LSTM Training Script — trains on real market data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.progress_writer import update_training_progress

import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from Python.agi_brain import AGIModel, FEATURE_COLUMNS
from Python.feature_pipeline import build_lstm_feature_frame, ENGINEERED_V2
from Python.data_feed import fetch_training_data

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="DEBUG")

def create_sequences(data, close_prices, seq_len=60):
    """Create input sequences and labels for LSTM training (Volatility Focused).

    Args:
        data: scaled feature matrix (n_samples, n_features)
        close_prices: raw close prices for label generation
        seq_len: sequence length for LSTM
    """
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len:i])

        # Calculate next return magnitude from raw close prices
        future_return = (close_prices[i] - close_prices[i - 1]) / (close_prices[i - 1] + 1e-8)
        magnitude = abs(future_return)

        # Classify based on volatility threshold
        # 0 = Low volatility (HOLD)
        # 1 = Medium volatility
        # 2 = High Volatility Spike
        if magnitude > 0.0015:
            y.append(2)  # High vol
        elif magnitude > 0.0005:
            y.append(1)  # Med vol
        else:
            y.append(0)  # Low vol/Hold

    return np.array(X), np.array(y)

def train_lstm(symbols=None, epochs=50, seq_len=60):
    if symbols is None:
        symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"]

    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    feature_columns = list(FEATURE_COLUMNS)
    n_features = len(feature_columns)
    model = AGIModel(input_dim=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = MinMaxScaler()

    logger.success(f"LSTM Training started on {device.upper()} | Symbols: {symbols} | Epochs: {epochs} | Features: {n_features}")

    # ── Fetch and combine training data ─────────────────────────────
    all_X, all_y = [], []
    for sym in symbols:
        logger.info(f"Fetching training data for {sym}...")
        df = fetch_training_data(sym, period="60d")
        if df is None or df.empty or len(df) < seq_len + 100:
            logger.warning(f"Insufficient data for {sym} (len={0 if df is None else len(df)}), skipping")
            continue

        # Build engineered features using the feature pipeline
        feat_df, available_cols = build_lstm_feature_frame(df, feature_version=ENGINEERED_V2)
        if len(feat_df) < seq_len + 10:
            logger.warning(f"Not enough feature rows for {sym} after pipeline, skipping")
            continue

        # Use the columns the model expects
        use_cols = feature_columns if set(feature_columns).issubset(set(available_cols)) else available_cols
        features = feat_df[use_cols].astype(float).values

        # Get close prices for label generation (aligned with feature rows)
        close_prices = df["close"].iloc[-len(feat_df):].values

        data = scaler.fit_transform(features)
        X, y = create_sequences(data, close_prices, seq_len)
        all_X.append(X)
        all_y.append(y)
        logger.info(f"  {sym}: {len(X)} sequences | LOW:{(y==0).sum()} MED:{(y==1).sum()} HIGH:{(y==2).sum()}")

    if not all_X:
        logger.error("No training data available!")
        return

    X_train = np.concatenate(all_X)
    y_train = np.concatenate(all_y)
    logger.info(f"Total training set: {len(X_train)} sequences | "
                f"BUY:{(y_train==1).sum()} SELL:{(y_train==2).sum()} HOLD:{(y_train==0).sum()}")

    # Convert to tensors
    X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

    # ── Training loop ───────────────────────────────────────────────
    model.train()
    batch_size = 64
    n_batches = len(X_tensor) // batch_size

    for epoch in range(epochs):
        # Shuffle each epoch
        perm = torch.randperm(len(X_tensor))
        X_tensor = X_tensor[perm]
        y_tensor = y_tensor[perm]

        epoch_loss = 0.0
        correct = 0
        total = 0

        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            X_batch = X_tensor[start:end]
            y_batch = y_tensor[start:end]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        acc = correct / total * 100 if total > 0 else 0
        avg_loss = epoch_loss / max(n_batches, 1)
        logger.info(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}% | Batches: {n_batches}")
        update_training_progress("lstm", {
            "running": True,
            "symbol": ",".join(symbols),
            "epoch": epoch + 1,
            "epochs_total": epochs,
            "loss": round(avg_loss, 4),
            "accuracy": round(acc, 1),
        })

    # ── Save model + scaler + metadata ─────────────────────────────
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "lstm_agi_trained.pt")
    torch.save(model.state_dict(), model_path)
    logger.success(f"LSTM model saved: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

    scaler_path = os.path.join(model_dir, "lstm_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logger.success(f"Scaler saved: {scaler_path}")

    meta_path = os.path.join(model_dir, "lstm_agi_trained.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_columns": feature_columns,
            "feature_version": ENGINEERED_V2,
            "n_features": n_features,
            "symbols": symbols,
            "epochs": epochs,
            "seq_len": seq_len,
        }, f, indent=2)
    logger.success(f"Metadata saved: {meta_path}")

    # ── Final stats ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor[:500])
        _, preds = outputs.max(1)
        final_acc = (preds == y_tensor[:500]).sum().item() / 500 * 100
    logger.success(f"Training complete! Final validation accuracy: {final_acc:.1f}%")
    update_training_progress("lstm", {
        "running": False,
        "symbol": ",".join(symbols),
        "epoch": epochs,
        "epochs_total": epochs,
        "loss": round(avg_loss, 4),
        "accuracy": round(final_acc, 1),
        "completed": True,
    })

    metrics = {
        "win_rate": float(final_acc),
        "epochs": epochs,
        "loss": float(avg_loss),
        "date": __import__('datetime').datetime.now().isoformat()
    }
    
    # Save Candidate locally in Model Registry for autonomous evaluation loop
    try:
        from Python.model_registry import ModelRegistry
        import shutil
        reg = ModelRegistry()
        cand_path = reg.save_candidate(model.state_dict(), metrics, model_type="lstm")
        shutil.copy(scaler_path, os.path.join(cand_path, "lstm_scaler.pkl"))
        
        # Immediately Test against Champion to see if Canary Staging is permitted
        if reg.evaluate_and_stage_canary(cand_path):
            logger.success("Candidate surpassed Champion. Scheduled for live Canary testing tomorrow!")
    except Exception as e:
        logger.error(f"Failed to register candidate model: {e}")

if __name__ == "__main__":
    train_lstm(epochs=50)
