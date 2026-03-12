import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class AdaptiveLSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    Trainable extractor for newer feature contracts where the live AGI LSTM
    bundle should not dictate PPO observation handling.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256, window_size: int = 100):
        total_obs = int(observation_space.shape[0])
        self.seq_window = int(window_size)
        self.portfolio_dim = total_obs % self.seq_window
        seq_flat = total_obs - self.portfolio_dim
        if seq_flat <= 0 or seq_flat % self.seq_window != 0:
            raise ValueError(
                f"Invalid observation shape for AdaptiveLSTMFeatureExtractor: total={total_obs}, "
                f"window={self.seq_window}, portfolio_dim={self.portfolio_dim}"
            )

        super().__init__(observation_space, features_dim=features_dim + self.portfolio_dim)
        self.seq_feature_dim = seq_flat // self.seq_window
        self.encoder = torch.nn.LSTM(
            input_size=self.seq_feature_dim,
            hidden_size=160,
            num_layers=2,
            dropout=0.1,
            batch_first=True,
        )
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(160, features_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(features_dim, features_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        seq_features = observations[:, :-self.portfolio_dim] if self.portfolio_dim else observations
        portfolio_state = observations[:, -self.portfolio_dim :] if self.portfolio_dim else observations.new_zeros((batch_size, 0))
        seq = seq_features.view(batch_size, self.seq_window, self.seq_feature_dim)
        encoded, _ = self.encoder(seq)
        projected = self.projection(encoded[:, -1, :])
        if self.portfolio_dim:
            return torch.cat([projected, portfolio_state], dim=1)
        return projected
