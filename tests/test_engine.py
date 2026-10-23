import asyncio
from typing import Any

import pytest
from conftest import DAYS, make_data

from stockbench_inspect.engine import INITIAL_CASH, _fill_at_open, run_backtest
from stockbench_inspect.metrics import agent_metrics


class ScriptedAgent:
    """Returns fixed orders per day and records what the engine reports back."""

    def __init__(self, orders_by_day: dict[str, list[dict[str, Any]]]) -> None:
        self.orders_by_day = orders_by_day
        self.executed: dict[str, list[str]] = {}
        self.contexts: list[dict[str, Any]] = []

    async def on_bar(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        self.contexts.append(dict(ctx))
        return self.orders_by_day.get(ctx["date"], [])

    def record_executed_decisions(self, executed_symbols: list[str], portfolio: Any) -> None:
        self.executed[self.contexts[-1]["date"]] = list(executed_symbols)


def test_fill_charges_one_bps_commission_on_cash_only() -> None:
    filled, px, net_cost = _fill_at_open(10.0, 100.0)
    assert filled == 10.0
    assert px == pytest.approx(100.02)  # slippage moves the recorded cost, not cash
    assert net_cost == pytest.approx(1000.1)
    filled, px, net_cost = _fill_at_open(-10.0, 100.0)
    assert filled == -10.0 and net_cost == pytest.approx(-999.9)


def test_always_hold_scores_zero(two_stock_data) -> None:
    result = asyncio.run(run_backtest(two_stock_data, ScriptedAgent({})))
    assert result.nav == [1.0] * len(DAYS)
    assert agent_metrics(result.nav) == {"final_return": 0.0, "max_drawdown": 0.0, "sortino": 0.0}


def test_last_buy_is_skipped_when_commission_exhausts_cash(two_stock_data) -> None:
    # Two buys of exactly half the cash each: commission makes the second one unaffordable.
    orders = {DAYS[0]: [{"symbol": "AAA", "qty": 500.0}, {"symbol": "BBB", "qty": 1000.0}]}
    agent = ScriptedAgent(orders)
    result = asyncio.run(run_backtest(two_stock_data, agent))
    assert [(t.symbol, t.qty) for t in result.trades] == [("AAA", 500.0)]
    assert agent.executed[DAYS[0]] == ["AAA"]
    assert result.final_cash == pytest.approx(INITIAL_CASH - 50_005.0)


def test_buy_bigger_than_cash_is_sent_back_to_the_agent(two_stock_data) -> None:
    orders = {DAYS[0]: [{"symbol": "AAA", "qty": 1001.0}]}
    agent = ScriptedAgent(orders)
    asyncio.run(run_backtest(two_stock_data, agent))
    first_day = [c for c in agent.contexts if c["date"] == DAYS[0]]
    assert len(first_day) == 3  # the first attempt and two retries
    rejection = first_day[1]["rejected_orders"][0]
    assert rejection["retry_count"] == 0 and rejection["context"]["portfolio_rebalance_needed"]
    assert first_day[2]["rejected_orders"][0]["retry_count"] == 1


def test_dividends_credit_cash_only_for_held_shares() -> None:
    data = make_data(
        {"AAA": [100.0, 100.0, 100.0, 100.0]},
        dividends={
            "AAA": [{"ex_dividend_date": DAYS[1], "cash_amount": 2.5}, {"ex_dividend_date": DAYS[0], "cash_amount": 9}]
        },
    )
    agent = ScriptedAgent({DAYS[0]: [{"symbol": "AAA", "qty": 10.0}]})
    result = asyncio.run(run_backtest(data, agent))
    # The day-one ex-date comes before the buy, so only the day-two dividend is paid.
    assert result.final_cash == pytest.approx(INITIAL_CASH - 1000.1 + 25.0)


def test_holding_days_count_days_held_after_trading(two_stock_data) -> None:
    agent = ScriptedAgent({DAYS[0]: [{"symbol": "AAA", "qty": 1.0}], DAYS[2]: [{"symbol": "AAA", "qty": -1.0}]})
    asyncio.run(run_backtest(two_stock_data, agent))
    portfolio = agent.contexts[-1]["portfolio"]
    assert portfolio.positions["AAA"].shares == 0.0
    assert portfolio.positions["AAA"].holding_days == 2  # never reset after closing, as upstream
