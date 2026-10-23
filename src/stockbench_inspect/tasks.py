"""StockBench as an Inspect task.

One sample is one backtest: the agent trades 20 Dow Jones stocks with $100,000 over the
83 trading days from 2025-03-03 to 2025-06-30, making two model calls per day (a
fundamental filter, then the trading decision). The paper reports the mean of 3 runs,
so the task runs 3 epochs by default. Pass --epochs 1 for a single run.

Paper: StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?
https://arxiv.org/abs/2510.02209
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import transcript
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.util import StoreModel, span
from pydantic import Field

from stockbench_inspect.agent import StockBenchAgent
from stockbench_inspect.data import END_DATE, START_DATE, MarketData, load_market_data
from stockbench_inspect.engine import INITIAL_CASH, run_backtest
from stockbench_inspect.metrics import agent_metrics, baseline_metrics


class BacktestState(StoreModel):
    days: list[str] = Field(default_factory=list)
    nav: list[float] = Field(default_factory=list)
    trades: list[dict[str, Any]] = Field(default_factory=list)
    final_cash: float = 0.0
    final_positions: dict[str, float] = Field(default_factory=dict)
    stats: dict[str, int] = Field(default_factory=dict)


@task
def stockbench(fundamentals: bool = True, days: int | None = None) -> Task:
    """StockBench backtest.

    Args:
        fundamentals: Give the decision agent the fundamentals snapshot for stocks the
            filter selects, as in the paper. False drops them: upstream's snapshots are
            not point-in-time (they hold later 52-week highs and lows).
        days: Trade only the first N trading days. For smoke tests, not for results.
    """
    trading_days = _trading_days(load_market_data(), days)
    sample = Sample(
        id=f"{trading_days[0]}_to_{trading_days[-1]}",
        input=(
            f"Trade the 20 StockBench stocks from {trading_days[0]} to {trading_days[-1]} "
            f"({len(trading_days)} trading days), starting with ${INITIAL_CASH:,.0f}."
        ),
        metadata={"start": trading_days[0], "end": trading_days[-1], "trading_days": len(trading_days)},
    )
    return Task(
        dataset=MemoryDataset([sample], name="stockbench"),
        solver=stockbench_agent(fundamentals=fundamentals, days=days),
        scorer=stockbench_scorer(),
        epochs=3,
        config=GenerateConfig(temperature=0.7, max_tokens=16000),
        headline_metric="stockbench_scorer.final_return",
        version="1.0.0",
        metadata={"paper_window": f"{START_DATE} to {END_DATE}", "fundamentals": fundamentals},
    )


@solver
def stockbench_agent(fundamentals: bool = True, days: int | None = None) -> Solver:
    """Run the daily backtest loop, calling the model under evaluation twice a day."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        data = load_market_data()
        trading_days = _trading_days(data, days)
        model = get_model()

        async def llm(role: str, system_prompt: str, user_prompt: str) -> str:
            output = await model.generate(
                [ChatMessageSystem(content=system_prompt), ChatMessageUser(content=user_prompt)]
            )
            if is_failed_call(output):
                raise RuntimeError(f"{role} call failed ({output.stop_reason}): {output.error or output.completion}")
            return output.completion

        agent = StockBenchAgent(data, llm, use_fundamentals=fundamentals)

        @asynccontextmanager
        async def trading_day(day: str) -> AsyncIterator[None]:
            async with span(day, type="trading_day"):
                yield
                decisions = {
                    symbol: {"action": d.get("action"), "target": d.get("target_cash_amount")}
                    for symbol, d in agent.last_decisions.items()
                    if d.get("action") != "hold"
                }
                transcript().info({"day": day, "non_hold_decisions": decisions}, source="stockbench")

        result = await run_backtest(data, agent, trading_days, day_context=trading_day)

        backtest = state.store_as(BacktestState)
        backtest.days = result.days
        backtest.nav = result.nav
        backtest.trades = [vars(trade) for trade in result.trades]
        backtest.final_cash = result.final_cash
        backtest.final_positions = result.final_positions
        backtest.stats = vars(agent.stats)

        summary = (
            f"Finished {len(result.days)} trading days with {len(result.trades)} trades. "
            f"Final NAV {result.nav[-1]:.4f} (start 1.0000), cash ${result.final_cash:,.2f}, "
            f"{len(result.final_positions)} open positions."
        )
        state.messages.append(ChatMessageAssistant(content=summary))
        state.output = ModelOutput.from_content(model=model.name, content=summary)
        return state

    return solve


@scorer(
    metrics={
        "final_return": [mean()],
        "max_drawdown": [mean()],
        "sortino": [mean()],
        "excess_return": [mean()],
        "baseline_return": [mean()],
        "trades": [mean()],
        "decision_unusable_rate": [mean()],
        "fallback_days": [mean()],
        "filter_unusable_rate": [mean()],
    }
)
def stockbench_scorer() -> Scorer:
    """Final return (on starting cash, after costs), max drawdown and daily Sortino.

    Also reports the equal-weight buy-and-hold baseline over the same days, and how
    often the model's replies could not be used.
    """

    async def score(state: TaskState, target: Target) -> Score:
        backtest = state.store_as(BacktestState)
        data = load_market_data()
        agent = agent_metrics(backtest.nav)
        baseline = baseline_metrics(data, backtest.days)
        stats = backtest.stats
        value = {
            "final_return": agent["final_return"],
            "max_drawdown": agent["max_drawdown"],
            "sortino": agent["sortino"],
            "excess_return": agent["final_return"] - baseline["final_return"],
            "baseline_return": baseline["final_return"],
            "trades": float(len(backtest.trades)),
            "decision_unusable_rate": _rate(stats.get("decision_unusable", 0), stats.get("decision_calls", 0)),
            "fallback_days": float(stats.get("fallback_days", 0)),
            "filter_unusable_rate": _rate(stats.get("filter_unusable", 0), stats.get("filter_calls", 0)),
        }
        return Score(
            value=value,
            answer=f"{agent['final_return']:.4%}",
            explanation=(
                f"Return {agent['final_return']:.2%}, max drawdown {agent['max_drawdown']:.2%}, "
                f"Sortino {agent['sortino']:.4f}. Buy-and-hold: {baseline['final_return']:.2%}, "
                f"{baseline['max_drawdown']:.2%}, {baseline['sortino']:.4f}."
            ),
            metadata={"days": backtest.days, "nav": backtest.nav, "baseline": baseline, "stats": stats},
        )

    return score


def is_failed_call(output: ModelOutput) -> bool:
    """True for an error that Inspect returned as a reply instead of raising.

    Inspect turns some rejected requests (a prompt longer than the context window, a
    blocked prompt) into a reply holding the error text, with no token usage. Upstream
    sees these as API errors, and so does this port.
    """
    if output.error:
        return True
    return output.usage is None and output.stop_reason in ("model_length", "content_filter")


def _trading_days(data: MarketData, days: int | None) -> list[str]:
    if days is not None and not 1 <= days <= len(data.trading_days):
        raise ValueError(f"days must be between 1 and {len(data.trading_days)}, got {days}")
    return data.trading_days if days is None else data.trading_days[:days]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0
