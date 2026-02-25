"""LSTM Training Script — trains on real market data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loguru import logger
from Python.agi_brain import AGIModel
from Python.data_feed import fetch_training_data

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "lstm_training.log"), rotation="10 MB", level="DEBUG")

def create_sequences(data, seq_len=60):
    """Create input sequences and labels for LSTM training (Volatility Focused)."""
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        X.append(data[i - seq_len:i])
        
        # Calculate next return magnitude
        future_return = (data[i, 3] - data[i - 1, 3]) / (data[i - 1, 3] + 1e-8)  
        magnitude = abs(future_return)
        
        # We classify based on volatility threshold instead of raw direction
        # 0 = Dead zone (Low volatility, HOLD)
        # 1 = Small trend (Medium confidence)
        # 2 = High Volatility Spike (Extreme confidence)
        if magnitude > 0.0015:
            y.append(2)  # High vol
        elif magnitude > 0.0005:
            y.append(1)  # Med vol
        else:
            y.append(0)  # Low vol/Hold
    
    return np.array(X), np.array(y)

def train_lstm(symbols=None, epochs=50, seq_len=60):
    if symbols is None:
        symbols = ["EURUSD", "GBPUSD", "XAUUSD"]

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = AGIModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    scaler = MinMaxScaler()

    logger.success(f"LSTM Training started on {device.upper()} | Symbols: {symbols} | Epochs: {epochs}")

    # ── Fetch and combine training data ─────────────────────────────
    all_X, all_y = [], []
    for sym in symbols:
        logger.info(f"Fetching training data for {sym}...")
        df = fetch_training_data(sym, period="60d")
        if df.empty or len(df) < seq_len + 10:
            logger.warning(f"Insufficient data for {sym}, skipping")
            continue

        data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].values)
        X, y = create_sequences(data, seq_len)
        all_X.append(X)
        all_y.append(y)
        logger.info(f"  {sym}: {len(X)} sequences created | BUY:{(y==1).sum()} SELL:{(y==2).sum()} HOLD:{(y==0).sum()}")

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

    # ── Save model ──────────────────────────────────────────────────
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_agi_trained.pt")
    torch.save(model.state_dict(), model_path)
    logger.success(f"LSTM model saved: {model_path} ({os.path.getsize(model_path)/1024:.1f} KB)")

    # ── Final stats ─────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor[:500])
        _, preds = outputs.max(1)
        final_acc = (preds == y_tensor[:500]).sum().item() / 500 * 100
    logger.success(f"Training complete! Final validation accuracy: {final_acc:.1f}%")

if __name__ == "__main__":
    train_lstm(epochs=50)
