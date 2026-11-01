"""Risk measures on losses (positive number = money lost).

CVaR, or expected shortfall, is the average of the worst (1 - alpha) share of
outcomes. Rockafellar and Uryasev (2000) showed it can be written as
    CVaR(L) = min_w  w + E[(L - w)^+] / (1 - alpha)
which is differentiable in both w and L, so it can be minimised jointly with
the hedging network by gradient descent. The minimising w is the VaR.
"""
from __future__ import annotations

import torch
from torch import nn


def cvar(losses: torch.Tensor, alpha: float = 0.95) -> torch.Tensor:
    """Empirical CVaR: mean of the losses at or above the alpha quantile."""
    k = max(1, int(round((1 - alpha) * losses.numel())))
    return losses.flatten().topk(k).values.mean()


class CVaRLoss(nn.Module):
    def __init__(self, alpha: float = 0.95):
        super().__init__()
        self.alpha = alpha
        self.w = nn.Parameter(torch.zeros(()))

    def forward(self, losses: torch.Tensor) -> torch.Tensor:
        return self.w + (losses - self.w).clamp_min(0.0).mean() / (1 - self.alpha)
