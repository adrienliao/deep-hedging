"""Path simulation and Black-Scholes references, in torch so everything stays on one device."""
from __future__ import annotations

import math

import torch

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(x / _SQRT2))


def simulate_gbm(n_paths: int, n_steps: int, S0: float = 100.0, sigma: float = 0.2, T: float = 30 / 365,
                 mu: float = 0.0, generator: torch.Generator | None = None) -> torch.Tensor:
    """Exact GBM on a regular grid, shape (n_paths, n_steps + 1)."""
    dt = T / n_steps
    z = torch.randn(n_paths, n_steps, generator=generator)
    log_inc = (mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z
    log_s = math.log(S0) + torch.cumsum(log_inc, dim=1)
    return torch.cat([torch.full((n_paths, 1), math.log(S0)), log_s], dim=1).exp()


def bs_call(S, K, tau, sigma):
    tau = torch.as_tensor(tau, dtype=S.dtype).clamp_min(1e-12)
    vol = sigma * torch.sqrt(tau)
    d1 = (torch.log(S / K) + 0.5 * vol**2) / vol
    return S * norm_cdf(d1) - K * norm_cdf(d1 - vol)


def bs_call_delta(S, K, tau, sigma):
    tau = torch.as_tensor(tau, dtype=S.dtype).clamp_min(1e-12)
    vol = sigma * torch.sqrt(tau)
    return norm_cdf((torch.log(S / K) + 0.5 * vol**2) / vol)
