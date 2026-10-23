"""StockBench's performance measures for the agent and the buy-and-hold baseline.

Ported from upstream's evaluate() and extract_key_metrics_summary() in
backtest/metrics.py and build_per_symbol_bh_benchmark() in backtest/engine.py at the
pinned commit. Sortino is upstream's daily version: the mean daily return divided by
the sample standard deviation of the negative daily returns, with no annualization and
no risk-free rate.
"""

from __future__ import annotations

import statistics

from stockbench_inspect.data import MarketData
from stockbench_inspect.engine import INITIAL_CASH

# The baseline pays commission plus slippage once, on entry (upstream: 1 + 2 bps).
BASELINE_FEE_RATE = 1.0 / 1e4 + 2.0 / 1e4


def daily_returns(nav: list[float]) -> list[float]:
    """Upstream's nav.pct_change().fillna(0.0): the first day's return counts as zero."""
    return [0.0] + [nav[i] / nav[i - 1] - 1.0 for i in range(1, len(nav))]


def max_drawdown(nav: list[float]) -> float:
    peak = nav[0]
    worst = 0.0
    for value in nav:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst


def sortino(nav: list[float]) -> float:
    returns = daily_returns(nav)
    downside = [r for r in returns if r < 0]
    downside_vol = statistics.stdev(downside) if len(downside) > 1 else 0.0
    return (sum(returns) / len(returns)) / downside_vol if downside_vol > 0 else 0.0


def agent_metrics(nav: list[float]) -> dict[str, float]:
    """Return on starting cash (costs included), max drawdown and daily Sortino."""
    return {"final_return": nav[-1] - 1.0, "max_drawdown": max_drawdown(nav), "sortino": sortino(nav)}


def baseline_nav(data: MarketData, days: list[str]) -> list[float]:
    """Equal dollars in every stock at the first day's open, held, valued at each open."""
    first = days[0]
    cash_per_symbol = INITIAL_CASH / len(data.symbols)
    net_per_symbol = cash_per_symbol / (1.0 + BASELINE_FEE_RATE)
    shares = {s: net_per_symbol / data.open_price(s, first) for s in data.symbols}
    nav = []
    for day in days:
        values = [shares[s] * data.open_price(s, day) / cash_per_symbol for s in data.symbols]
        nav.append(sum(values) / len(values))
    return nav


def baseline_metrics(data: MarketData, days: list[str]) -> dict[str, float]:
    """As upstream reports it: the return is measured from the first day's NAV, so the entry fee drops out."""
    nav = baseline_nav(data, days)
    return {"final_return": nav[-1] / nav[0] - 1.0, "max_drawdown": max_drawdown(nav), "sortino": sortino(nav)}
