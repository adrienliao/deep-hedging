"""Train one deep hedger per cost level, evaluate everything on the same test paths,
write results/results.json and the figures in docs/.

    python scripts/run_experiments.py            # about 10 minutes on a laptop CPU
    python scripts/run_experiments.py --quick    # a couple of minutes, noisier
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deephedge import market, pnl, risk, strategies  # noqa: E402
from deephedge.train import Config, config_dict, train  # noqa: E402

BLUE, ORANGE, AQUA, GREY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#6250d6"
plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#b5b4ae", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
    "axes.labelcolor": GREY, "xtick.color": GREY, "ytick.color": GREY, "legend.frameon": False,
})
COSTS = [0.0, 0.0005, 0.001, 0.002, 0.005]
BANDS = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]


def evaluate(paths, positions, cfg, cost):
    loss = -pnl.hedge_pnl(paths, positions, cfg.K, cost)
    return {"cvar95": risk.cvar(loss, 0.95).item(), "mean_loss": loss.mean().item(), "std": loss.std().item(),
            "turnover": positions.diff(dim=1).abs().sum(dim=1).mean().item()}


def best_band(cfg, cost, gen):
    """Pick the no-trade band on separate validation paths, so the baseline is tuned fairly too."""
    val = market.simulate_gbm(50_000, cfg.n_steps, cfg.S0, cfg.sigma, cfg.T, generator=gen)
    scores = {b: risk.cvar(-pnl.hedge_pnl(val, strategies.banded_delta_hedge(val, cfg.K, cfg.sigma, cfg.T, b),
                                         cfg.K, cost)).item() for b in BANDS}
    return min(scores, key=scores.get)


def main(quick=False):
    base = Config(iters=400 if quick else 1500)
    gen = torch.Generator().manual_seed(12345)
    test = market.simulate_gbm(200_000, base.n_steps, base.S0, base.sigma, base.T, generator=gen)
    premium = market.bs_call(torch.tensor(base.S0), base.K, base.T, base.sigma).item()

    results, models = {"config": config_dict(base), "bs_premium": premium, "by_cost": {}}, {}
    (ROOT / "results").mkdir(exist_ok=True)
    for cost in COSTS:
        t0 = time.time()
        cfg = Config(**{**config_dict(base), "cost": cost})
        model, hist = train(cfg)
        models[cost] = model
        torch.save(model.state_dict(), ROOT / "results" / f"deep_hedger_cost{cost:g}.pt")
        band = best_band(cfg, cost, gen)
        with torch.no_grad():
            row = {
                "no_hedge": evaluate(test, strategies.no_hedge(test), cfg, cost),
                "bs_delta": evaluate(test, strategies.bs_delta_hedge(test, cfg.K, cfg.sigma, cfg.T), cfg, cost),
                "banded_delta": {**evaluate(test, strategies.banded_delta_hedge(test, cfg.K, cfg.sigma, cfg.T, band), cfg, cost),
                                 "band": band},
                "deep_hedge": evaluate(test, model(test), cfg, cost),
            }
        row["training_curve"] = hist[::10]
        results["by_cost"][str(cost)] = row
        print(f"cost {cost:.4f}: " + "  ".join(f"{k} {v['cvar95']:.3f}" for k, v in row.items() if k != "training_curve")
              + f"  ({time.time() - t0:.0f}s)")

    (ROOT / "results" / "results.json").write_text(json.dumps(results, indent=1))
    figures(results, models, test, base)


def figures(results, models, test, cfg):
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    by = results["by_cost"]
    bps = [float(c) * 1e4 for c in by]

    # 1. risk vs cost
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for key, label, c, style in [("bs_delta", "Black-Scholes delta", GREY, "o--"),
                                 ("banded_delta", "delta with tuned no-trade band", AQUA, "s-"),
                                 ("deep_hedge", "deep hedge", ORANGE, "o-")]:
        ax.plot(bps, [by[k][key]["cvar95"] for k in by], style, color=c, lw=2, ms=5, label=label)
    ax.set_xlabel("proportional transaction cost (bp of notional traded)")
    ax.set_ylabel("95% CVaR of hedging loss")
    ax.set_title("Short 1 month ATM call, daily rebalancing", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "cvar_vs_cost.png", dpi=160)

    # 2. what the network learned: trade size against current holding, at the money, mid life
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    S = torch.linspace(85, 115, 200)
    tau = 0.5
    bs = market.bs_call_delta(S, cfg.K, tau * cfg.T, cfg.sigma)
    with torch.no_grad():
        for cost, c in [(0.0, BLUE), (0.005, ORANGE)]:
            net = models[cost].net
            feats = torch.stack([torch.log(S / cfg.K) / models[cost].scale, torch.full_like(S, tau), bs], dim=1)
            axes[0].plot(S, net(feats).squeeze(1), color=c, lw=2, label=f"deep hedge, cost {cost * 1e4:g} bp")
            prev = torch.linspace(0.0, 1.0, 200)
            feats = torch.stack([torch.zeros_like(prev), torch.full_like(prev, tau), prev], dim=1)
            axes[1].plot(prev, net(feats).squeeze(1) - prev, color=c, lw=2)
    axes[0].plot(S, bs, "--", color=GREY, lw=1.2, label="Black-Scholes delta")
    axes[0].set_xlabel("spot")
    axes[0].set_title("position chosen, half way to expiry", loc="left")
    axes[0].legend()
    bs_atm = market.bs_call_delta(torch.tensor(cfg.K), cfg.K, tau * cfg.T, cfg.sigma).item()
    axes[1].axvline(bs_atm, color=GREY, lw=1.2, ls="--")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_xlabel("shares currently held (spot at the strike)")
    axes[1].set_title("trade vs current holding: with costs it only\ncloses part of the gap (partial rebalancing)", loc="left")
    fig.tight_layout()
    fig.savefig(out / "learned_policy.png", dpi=160)

    # 3. one path, both strategies, with costs
    cost = 0.002
    idx = 7
    with torch.no_grad():
        deep = models[cost](test[idx:idx + 1])[0]
    bsd = strategies.bs_delta_hedge(test[idx:idx + 1], cfg.K, cfg.sigma, cfg.T)[0]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    ax0.plot(np.arange(cfg.n_steps + 1), test[idx], color=GREY, lw=1.5)
    ax0.axhline(cfg.K, color=GREY, lw=0.8, ls=":")
    ax0.set_ylabel("spot")
    days = np.arange(cfg.n_steps + 1)
    bsd, deep = torch.cat([bsd, bsd[-1:]]), torch.cat([deep, deep[-1:]])
    turn = lambda x: (x[:-1].diff().abs().sum() + x[0].abs() + x[-1].abs()).item()
    ax1.step(days, bsd, where="post", color=GREY, lw=1.5, label=f"Black-Scholes delta (traded {turn(bsd):.2f} shares)")
    ax1.step(days, deep, where="post", color=ORANGE, lw=2, label=f"deep hedge (traded {turn(deep):.2f} shares)")
    ax1.set_ylabel("shares held")
    ax1.set_xlabel("trading day")
    ax1.legend()
    ax0.set_title(f"Same path, {cost * 1e4:g} bp costs: the network trades less", loc="left")
    fig.tight_layout()
    fig.savefig(out / "sample_path.png", dpi=160)


def figures_from_saved():
    """Redraw the figures from results/ without retraining."""
    cfg = Config()
    models = {}
    for cost in COSTS:
        m = strategies.DeepHedger(cfg.K, cfg.sigma, cfg.T)
        m.load_state_dict(torch.load(ROOT / "results" / f"deep_hedger_cost{cost:g}.pt"))
        models[cost] = m.eval()
    test = market.simulate_gbm(200_000, cfg.n_steps, cfg.S0, cfg.sigma, cfg.T,
                               generator=torch.Generator().manual_seed(12345))
    figures(json.loads((ROOT / "results" / "results.json").read_text()), models, test, cfg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true", help="reuse the trained models in results/")
    a = ap.parse_args()
    figures_from_saved() if a.figures_only else main(a.quick)
