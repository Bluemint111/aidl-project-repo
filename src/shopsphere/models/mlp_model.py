"""Shared PyTorch MLP definition for the ShopSphere propensity model."""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in: int, hidden: tuple[int, int, int] = (256, 128, 64),
                 dropout: tuple[float, float, float] = (0.35, 0.30, 0.20)):
        super().__init__()
        h1, h2, h3 = hidden
        d1, d2, d3 = dropout
        self.net = nn.Sequential(
            nn.Linear(n_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(d1),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(d2),
            nn.Linear(h2, h3), nn.BatchNorm1d(h3), nn.ReLU(), nn.Dropout(d3),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
