import torch
from torch import nn


class ClockResidualSelector(nn.Module):
    """Add a zero-initialized structural residual to frozen-LM clock scores."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64, dropout: float = 0.10):
        super().__init__()
        self.residual = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, candidate_features):
        return self.residual(candidate_features).squeeze(-1)
