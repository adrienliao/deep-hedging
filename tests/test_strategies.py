import torch

from deephedge import market, pnl, risk, strategies
from deephedge.train import Config, train

T = 30 / 365


def _paths(n=50_000, steps=30, seed=0):
    return market.simulate_gbm(n, steps, T=T, generator=torch.Generator().manual_seed(seed))


def test_delta_hedge_replicates_the_premium():
    """Without costs, premium + delta hedge gains should match the payoff up to discretisation noise."""
    paths = _paths(steps=365)
    prem = market.bs_call(torch.tensor(100.0), 100.0, T, 0.2)
    err = pnl.hedge_pnl(paths, strategies.bs_delta_hedge(paths, 100.0, 0.2, T), 100.0, premium=prem)
    assert abs(err.mean().item()) < 0.02
    assert err.std().item() < 0.3


def test_band_zero_equals_plain_delta():
    p = _paths(n=1000)
    a = strategies.bs_delta_hedge(p, 100.0, 0.2, T)
    b = strategies.banded_delta_hedge(p, 100.0, 0.2, T, band=0.0)
    assert torch.allclose(a, b)


def test_band_reduces_turnover():
    p = _paths(n=1000)
    plain = strategies.banded_delta_hedge(p, 100.0, 0.2, T, band=0.0)
    banded = strategies.banded_delta_hedge(p, 100.0, 0.2, T, band=0.1)
    assert banded.diff(dim=1).abs().sum() < 0.5 * plain.diff(dim=1).abs().sum()


def test_cvar_objective_matches_empirical_cvar_at_optimum():
    losses = torch.randn(100_000, generator=torch.Generator().manual_seed(1))
    obj = risk.CVaRLoss(0.95)
    with torch.no_grad():
        obj.w.fill_(torch.quantile(losses, 0.95))
    assert abs(obj(losses).item() - risk.cvar(losses, 0.95).item()) < 0.01
    # standard normal: CVaR_95 = pdf(1.645) / 0.05 = 2.063
    assert abs(risk.cvar(losses, 0.95).item() - 2.063) < 0.03


def test_deep_hedger_learns_something_quickly():
    cfg = Config(iters=150, batch=2048, n_steps=10)
    model, hist = train(cfg)
    p = market.simulate_gbm(20_000, 10, T=T, generator=torch.Generator().manual_seed(9))
    with torch.no_grad():
        deep = risk.cvar(-pnl.hedge_pnl(p, model(p), 100.0))
    none = risk.cvar(-pnl.hedge_pnl(p, strategies.no_hedge(p), 100.0))
    assert deep < 0.5 * none
    assert hist[-1] < hist[0]
