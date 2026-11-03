import pytest
import torch

from deephedge.pnl import hedge_pnl


def test_hand_computed_path():
    paths = torch.tensor([[100.0, 110.0, 105.0]])
    pos = torch.tensor([[0.5, 1.0]])
    # gains 0.5*10 + 1*(-5) = 0, payoff 5, trades 0.5 @100, 0.5 @110, 1 @105
    pnl = hedge_pnl(paths, pos, K=100.0, cost=0.01, premium=6.0)
    expected = 6.0 + 0.0 - 0.01 * (50 + 55 + 105) - 5.0
    assert pnl.item() == pytest.approx(expected)


def test_shape_check():
    with pytest.raises(ValueError):
        hedge_pnl(torch.ones(2, 4), torch.ones(2, 2), K=1.0)
