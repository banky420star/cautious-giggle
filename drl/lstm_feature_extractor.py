import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
from loguru import logger


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Lightweight LSTM feature extractor for PPO.
    Dynamically infers n_features from observation shape and window_size.
    No dependency on SmartAGI — trained jointly from scratch with PPO.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        window_size: int = 100,
        portfolio_feature_count: int = 3,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
    ):
        obs_dim = int(observation_space.shape[0])
        seq_dim = obs_dim - portfolio_feature_count
        n_features = seq_dim // window_size

        output_dim = features_dim + portfolio_feature_count
        super().__init__(observation_space, features_dim=output_dim)

        self.window_size = window_size
        self.n_features = n_features
        self.portfolio_feature_count = portfolio_feature_count

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.projection = nn.Linear(lstm_hidden, features_dim)

        logger.info(
            f"LSTMFeatureExtractor: obs_dim={obs_dim}, "
            f"n_features={n_features}, window={window_size}, "
            f"portfolio={portfolio_feature_count}, output={output_dim}"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        seq_features = observations[:, :-self.portfolio_feature_count]
        portfolio_state = observations[:, -self.portfolio_feature_count:]

        seq = seq_features.view(batch_size, self.window_size, self.n_features)

        lstm_out, _ = self.lstm(seq)
        last_hidden = lstm_out[:, -1, :]

        projected = self.projection(last_hidden)
        combined = torch.cat([projected, portfolio_state], dim=1)

        return combined
