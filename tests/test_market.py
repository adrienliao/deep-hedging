import math

import pytest
import torch

from deephedge import market


def test_gbm_is_a_martingale():
    g = torch.Generator().manual_seed(0)
    p = market.simulate_gbm(200_000, 10, S0=100.0, sigma=0.3, T=1.0, generator=g)
    se = p[:, -1].std() / math.sqrt(p.shape[0])
    assert abs(p[:, -1].mean().item() - 100.0) < 4 * se.item()


def test_bs_against_known_value():
    # S=K=100, sigma=20%, T=1, r=0: 100 * (2 N(0.1) - 1) = 7.9656
    assert market.bs_call(torch.tensor(100.0), 100.0, 1.0, 0.2).item() == pytest.approx(7.9656, abs=1e-3)


def test_delta_is_derivative_of_price():
    S = torch.tensor(103.0, dtype=torch.float64, requires_grad=True)
    price = market.bs_call(S, 100.0, 0.25, 0.2)
    price.backward()
    assert S.grad.item() == pytest.approx(market.bs_call_delta(S.detach(), 100.0, 0.25, 0.2).item(), rel=1e-8)
