"""The StockBench backtest engine: daily loop, order checks, fills at the open, dividends, NAV.

Ported from upstream's BacktestEngine.run (backtest/engine.py) at the pinned commit for
daily bars. Arithmetic follows upstream step by step so cash and positions match it to
the cent. Differences from upstream:
- the calendar is the 83 days that have price data, so upstream's 3 extra holiday
  days (2025-04-18, 2025-05-26, 2025-06-19) are not simulated;
- stock splits are not handled, because none of the 20 stocks split in the window.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

from stockbench_inspect.agent import MAX_ATTEMPTS
from stockbench_inspect.data import MarketData

INITIAL_CASH = 100_000.0
COMMISSION_BPS = 1.0
SLIPPAGE_BPS = 2.0  # only moves the recorded average cost, never cash (as upstream)


class Agent(Protocol):
    async def on_bar(self, ctx: dict[str, Any]) -> list[dict[str, Any]]: ...

    def record_executed_decisions(self, executed_symbols: list[str], portfolio: Any) -> None: ...


@dataclass
class Position:
    shares: float = 0.0
    avg_price: float = 0.0
    holding_days: int = 0


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    def update_cash(self, amount: float) -> bool:
        new_cash = self.cash + amount
        if new_cash < 0:
            return False
        self.cash = new_cash
        return True


@dataclass
class Trade:
    day: str
    symbol: str
    side: str
    qty: float
    price: float
    net_cost: float


@dataclass
class BacktestResult:
    days: list[str]
    nav: list[float]
    trades: list[Trade]
    final_cash: float
    final_positions: dict[str, float]


def _fill_at_open(qty: float, open_price: float) -> tuple[float, float, float]:
    """Upstream's _fill_at_open: (filled qty, slippage-adjusted price, net cash cost)."""
    side = 1 if qty > 0 else -1
    if side > 0:
        px = float(open_price) * (1.0 + SLIPPAGE_BPS / 10_000.0)
    else:
        px = float(open_price) * (1.0 - SLIPPAGE_BPS / 10_000.0)
    filled_qty = round(abs(qty), 2) * side
    gross_open = open_price * abs(filled_qty)
    commission = gross_open * (COMMISSION_BPS / 10_000.0)
    net_cost = gross_open + commission if side > 0 else -(gross_open - commission)
    return filled_qty, px, net_cost


def _apply_dividends(portfolio: Portfolio, data: MarketData, symbol: str, day: str) -> None:
    pos = portfolio.positions.get(symbol)
    if pos is None or (pos.shares == 0 and pos.avg_price == 0.0):
        return
    if data.dividends[symbol] and pos.shares > 0:
        for cash in data.dividend_per_share(symbol, day):
            if cash and cash != 0.0:
                portfolio.update_cash(float(pos.shares) * cash)


async def run_backtest(
    data: MarketData,
    agent: Agent,
    days: list[str] | None = None,
    day_context: Callable[[str], AbstractAsyncContextManager[Any]] | None = None,
) -> BacktestResult:
    """Run the daily loop over `days` (default: every trading day) and return the NAV path.

    `day_context`, if given, wraps each day (the Inspect task uses it to open a span).
    """
    days = data.trading_days if days is None else days
    pf = Portfolio(cash=INITIAL_CASH)
    nav: list[float] = []
    trades: list[Trade] = []

    for day in days:
        async with day_context(day) if day_context else nullcontext():
            for symbol in data.symbols:
                _apply_dividends(pf, data, symbol, day)
            open_map = {symbol: data.open_price(symbol, day) for symbol in data.symbols}
            ctx: dict[str, Any] = {"date": day, "open_map": open_map, "portfolio": pf}

            # Up to 3 attempts: a buy that alone costs more than all cash sends the day back
            # to the agent with the rejected orders, which re-runs both model calls.
            retry_attempt = 0
            rejected_orders: list[dict[str, Any]] = []
            acceptable_orders: list[dict[str, Any]] = []
            while retry_attempt < MAX_ATTEMPTS:
                if rejected_orders:
                    ctx["rejected_orders"] = rejected_orders
                    agent_stats = getattr(agent, "stats", None)
                    if agent_stats is not None:
                        agent_stats.engine_retries += 1
                else:
                    ctx.pop("rejected_orders", None)
                orders = await agent.on_bar(ctx)
                if not isinstance(orders, list):
                    orders = []

                current_rejected: list[dict[str, Any]] = []
                acceptable_orders = []
                total_cash_required = 0.0
                order_costs: dict[str, tuple[float, float, float]] = {}
                for od in orders:
                    sym = od.get("symbol")
                    qty = round(float(od.get("qty", 0)), 2) if od.get("qty") else 0
                    if sym and qty != 0 and open_map.get(sym) is not None:
                        filled, exec_px, net_cost = _fill_at_open(qty, float(open_map[sym]))
                        order_costs[sym] = (filled, exec_px, net_cost)
                        if filled > 0 and net_cost > 0:
                            total_cash_required += net_cost
                for od in orders:
                    sym = od.get("symbol")
                    qty = round(float(od.get("qty", 0)), 2)
                    if not sym or qty == 0 or open_map.get(sym) is None:
                        continue
                    if sym in order_costs:
                        filled, exec_px, net_cost = order_costs[sym]
                    else:
                        filled, exec_px, net_cost = _fill_at_open(qty, float(open_map[sym]))
                    if filled > 0 and net_cost > 0 and pf.cash < net_cost:
                        shortfall = total_cash_required - pf.cash
                        current_rejected.append(
                            {
                                "symbol": sym,
                                "qty": qty,
                                "reason": "insufficient_cash",
                                "rejection_reason": "insufficient_cash",
                                "context": {
                                    "required_cash_this_order": net_cost,
                                    "available_cash": pf.cash,
                                    "total_cash_required_all_orders": total_cash_required,
                                    "cash_shortfall": shortfall,
                                    "retry_attempt": retry_attempt,
                                    "all_orders_count": len(orders),
                                    "portfolio_rebalance_needed": True,
                                    "suggestion": f"Total portfolio cash requirement ({total_cash_required:.2f}) exceeds available cash ({pf.cash:.2f}) by {shortfall:.2f}. Please reduce all order sizes proportionally or select fewer positions to fit within budget.",
                                },
                                "retry_count": retry_attempt,
                            }
                        )
                        continue
                    acceptable_orders.append(od)

                if current_rejected:
                    rejected_orders = current_rejected
                    retry_attempt += 1
                    if retry_attempt < MAX_ATTEMPTS:
                        continue
                break

            # Execute in universe order with a running cash check: a buy that no longer fits
            # is skipped silently, and sells later in the list do not fund it.
            executed_symbols: list[str] = []
            for od in acceptable_orders:
                sym = od.get("symbol")
                qty = round(float(od.get("qty", 0)), 2)
                if not sym or qty == 0 or open_map.get(sym) is None:
                    continue
                filled, exec_px, net_cost = _fill_at_open(qty, float(open_map[sym]))
                # Upstream creates the (empty) position before the cash check, so keep that order.
                pos = pf.positions.setdefault(sym, Position())
                if filled > 0 and net_cost > 0 and pf.cash < net_cost:
                    continue
                if (pos.shares >= 0 and filled > 0) or (pos.shares <= 0 and filled < 0) or pos.shares == 0:
                    new_shares = pos.shares + filled
                    if new_shares != 0:
                        pos.avg_price = (pos.avg_price * abs(pos.shares) + abs(filled) * exec_px) / abs(new_shares)
                    pos.shares = new_shares
                else:
                    pos.shares += filled
                    if pos.shares == 0:
                        pos.avg_price = 0.0
                if not pf.update_cash(-net_cost):
                    continue
                trades.append(
                    Trade(day, sym, "buy" if filled > 0 else "sell", abs(filled), float(open_map[sym]), net_cost)
                )
                if sym not in executed_symbols:
                    executed_symbols.append(sym)

            mark_to_market = 0.0
            for sym, pos in pf.positions.items():
                px = open_map.get(sym)
                if px is None:
                    px = pos.avg_price
                mark_to_market += pos.shares * px
                if pos.shares > 0:
                    pos.holding_days += 1
            nav.append((pf.cash + mark_to_market) / INITIAL_CASH)
            agent.record_executed_decisions(executed_symbols, pf)

    return BacktestResult(
        days=list(days),
        nav=nav,
        trades=trades,
        final_cash=pf.cash,
        final_positions={sym: pos.shares for sym, pos in pf.positions.items() if pos.shares != 0},
    )
