import os
import sys

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python.agi_brain import SmartAGI


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Trainable LSTM feature extractor for joint LSTM + PPO training.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim + 3)

        logger.info("initializing LSTMFeatureExtractor")
        self.lstm_brain = SmartAGI()

        self.projection = torch.nn.Linear(128, features_dim)

        # Keep all extractor params trainable.
        for param in self.lstm_brain.model.parameters():
            param.requires_grad = True

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        device = observations.device
        self.lstm_brain.model.to(device)
        self.projection = self.projection.to(device)

        batch_size = observations.shape[0]

        seq_features = observations[:, :-3]
        portfolio_state = observations[:, -3:]

        seq = seq_features.view(batch_size, 100, 5)
        lstm_embedding = self.lstm_brain.extract_features(seq)

        projected = self.projection(lstm_embedding)
        return torch.cat([projected, portfolio_state], dim=1)
