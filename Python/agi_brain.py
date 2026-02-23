import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

class AGIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(5, 128, 3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[:, -1, :])

class SmartAGI:
    def __init__(self):
        self.model = AGIModel()
        self.scaler = MinMaxScaler()
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.model.to(self.device)
        self.prediction_count = 0

        # Load trained model if available
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "lstm_agi_trained.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model.eval()
            size_kb = os.path.getsize(model_path) / 1024
            logger.success(f"AGI Brain loaded TRAINED model ({size_kb:.1f} KB) on {self.device.upper()}")
        else:
            logger.warning(f"No trained model found at {model_path} — using random weights!")
            logger.success(f"AGI Brain loaded on {self.device.upper()} (untrained)")

    def predict(self, df: pd.DataFrame) -> dict:
        self.prediction_count += 1
        data = self.scaler.fit_transform(df[['open','high','low','close','volume']].values)
        seq = torch.tensor(data[-60:].reshape(1, 60, 5), dtype=torch.float32).to(self.device)

        with torch.no_grad():
            logits = self.model(seq)
            probs = F.softmax(logits, dim=-1).cpu().numpy().flatten()
            pred = np.random.choice(3, p=probs)  # sample from probability distribution

        signal = ["HOLD", "BUY", "SELL"][pred]
        confidence = round(float(probs[pred]), 4)

        logger.info(
            f"[#{self.prediction_count}] Signal: {signal} | "
            f"Confidence: {confidence:.2%} | "
            f"Probs: H={probs[0]:.3f} B={probs[1]:.3f} S={probs[2]:.3f} | "
            f"{df['symbol'].iloc[0]}"
        )
        return {"signal": signal, "confidence": confidence, "symbol": df['symbol'].iloc[0]}
