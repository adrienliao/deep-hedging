"""Profit and loss of a hedged short call.

positions[:, k] is the number of shares held from t_k to t_{k+1}. The trader
sells the call for `premium`, trades to each new position paying proportional
costs `cost * |trade| * S`, and closes the stock position at maturity (also
paying costs). Rates are zero, which keeps the accounting readable.
"""
from __future__ import annotations

import torch


def hedge_pnl(paths: torch.Tensor, positions: torch.Tensor, K: float, cost: float = 0.0,
              premium: float | torch.Tensor = 0.0) -> torch.Tensor:
    n_steps = paths.shape[1] - 1
    if positions.shape[1] != n_steps:
        raise ValueError(f"expected {n_steps} positions per path, got {positions.shape[1]}")
    gains = (positions * paths.diff(dim=1)).sum(dim=1)

    zeros = torch.zeros_like(positions[:, :1])
    held = torch.cat([zeros, positions, zeros], dim=1)        # start flat, end flat
    trades = held.diff(dim=1).abs()                           # n_steps + 1 trades
    costs = cost * (trades * paths).sum(dim=1)

    payoff = (paths[:, -1] - K).clamp_min(0.0)
    return premium + gains - costs - payoff
