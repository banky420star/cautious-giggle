import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from loguru import logger
import sys, os

# Ensure we can import Python.agi_brain
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.agi_brain import SmartAGI

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Trainable LSTM feature extractor for joint LSTM + PPO training.
    Gradients flow through both LSTM and PPO policy.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim + 3)
        
        logger.info("Initializing LSTMFeatureExtractor for Joint PPO Hybrid...")
        
        # Load pre-trained SmartAGI LSTM
        self.lstm_brain = SmartAGI()
        
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.lstm_brain.model.to(device)
        self.lstm_brain.model.train()   # CRITICAL for joint training
        
        # Projection head (trainable)
        self.projection = nn.Linear(128, features_dim)
        self.projection.to(device)
        
        # Make LSTM parameters trainable
        for param in self.lstm_brain.model.parameters():
            param.requires_grad = True

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split sequence data and portfolio state
        seq_features = observations[:, :-3]
        portfolio_state = observations[:, -3:]
        
        # Reshape to (batch_size, 100, 5)
        seq = seq_features.view(batch_size, 100, 5)
        
        # Joint forward pass through LSTM (no no_grad!)
        lstm_embedding = self.lstm_brain.extract_features(seq)   # gradients flow
        
        # Project + combine with portfolio state
        projected = self.projection(lstm_embedding)
        combined = torch.cat([projected, portfolio_state], dim=1)
        
        return combined
