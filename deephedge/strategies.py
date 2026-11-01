"""Hedging strategies: map simulated paths to share positions.

The deep hedger is one small MLP shared across time steps. At each date it
sees the normalised log moneyness, the time left, and the position it already
holds. That last input is what lets it learn to *not* trade when trading is
expensive: without it the optimal policy under costs cannot be represented.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from .market import bs_call_delta


def time_grid(n_steps: int, T: float) -> torch.Tensor:
    return torch.linspace(0.0, T, n_steps + 1)


def no_hedge(paths: torch.Tensor) -> torch.Tensor:
    return torch.zeros(paths.shape[0], paths.shape[1] - 1)


def bs_delta_hedge(paths: torch.Tensor, K: float, sigma: float, T: float) -> torch.Tensor:
    tau = T - time_grid(paths.shape[1] - 1, T)[:-1]
    return bs_call_delta(paths[:, :-1], K, tau, sigma)


def banded_delta_hedge(paths: torch.Tensor, K: float, sigma: float, T: float, band: float) -> torch.Tensor:
    """Delta hedge that only rebalances when the position is more than `band` away from delta."""
    target = bs_delta_hedge(paths, K, sigma, T)
    pos = torch.zeros(paths.shape[0])
    out = []
    for k in range(target.shape[1]):
        lo, hi = target[:, k] - band, target[:, k] + band
        pos = torch.minimum(torch.maximum(pos, lo), hi)
        out.append(pos)
    return torch.stack(out, dim=1)


class DeepHedger(nn.Module):
    def __init__(self, K: float, sigma: float, T: float, hidden: int = 64, depth: int = 3):
        super().__init__()
        self.K, self.T = K, T
        self.scale = sigma * math.sqrt(T)
        layers, width = [], 3
        for _ in range(depth):
            layers += [nn.Linear(width, hidden), nn.SiLU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, paths: torch.Tensor) -> torch.Tensor:
        n_paths, n_steps = paths.shape[0], paths.shape[1] - 1
        t = time_grid(n_steps, self.T)
        pos = torch.zeros(n_paths, device=paths.device)
        out = []
        for k in range(n_steps):
            features = torch.stack([
                torch.log(paths[:, k] / self.K) / self.scale,
                torch.full_like(pos, (self.T - t[k].item()) / self.T),
                pos,
            ], dim=1)
            pos = self.net(features).squeeze(1)
            out.append(pos)
        return torch.stack(out, dim=1)
