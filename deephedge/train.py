from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch

from .market import simulate_gbm
from .pnl import hedge_pnl
from .risk import CVaRLoss
from .strategies import DeepHedger


@dataclass
class Config:
    S0: float = 100.0
    K: float = 100.0
    sigma: float = 0.2
    T: float = 30 / 365
    n_steps: int = 30
    cost: float = 0.0
    alpha: float = 0.95
    batch: int = 8192
    iters: int = 1500
    lr: float = 3e-3
    seed: int = 0


def train(cfg: Config, verbose: bool = False):
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = DeepHedger(cfg.K, cfg.sigma, cfg.T)
    risk = CVaRLoss(cfg.alpha)
    opt = torch.optim.Adam(list(model.parameters()) + list(risk.parameters()), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.iters)

    history, t0 = [], time.time()
    for it in range(cfg.iters):
        # fresh paths at every step: there is no fixed training set to overfit
        paths = simulate_gbm(cfg.batch, cfg.n_steps, cfg.S0, cfg.sigma, cfg.T, generator=gen)
        loss = risk(-hedge_pnl(paths, model(paths), cfg.K, cfg.cost))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(loss.item())
        if verbose and it % 250 == 0:
            print(f"  iter {it:5d}  CVaR {loss.item():.4f}  ({time.time() - t0:.0f}s)")
    return model, history


def config_dict(cfg: Config) -> dict:
    return asdict(cfg)
