import asyncio
import json
from collections.abc import Callable

from conftest import DAYS, make_data

from stockbench_inspect.agent import StockBenchAgent
from stockbench_inspect.data import MarketData
from stockbench_inspect.engine import run_backtest

NO_FUNDAMENTALS = '<FILTER_RESULT>{"stocks_need_fundamental": [], "reasoning": {}}</FILTER_RESULT>'


class StubLLM:
    """Answers the filter and decision agents from scripted functions and records prompts."""

    def __init__(self, decision: Callable[[dict, int], str], filter_reply: str = NO_FUNDAMENTALS) -> None:
        self.decision = decision
        self.filter_reply = filter_reply
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, role: str, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((role, user_prompt))
        if role == "fundamental_filter":
            return self.filter_reply
        payload, _ = json.JSONDecoder().raw_decode(user_prompt)
        return self.decision(payload, sum(1 for r, _ in self.calls if r == "decision_agent"))


def run(data: MarketData, llm: StubLLM, days: int = 1, fundamentals: bool = True) -> StockBenchAgent:
    agent = StockBenchAgent(data, llm, use_fundamentals=fundamentals)
    asyncio.run(run_backtest(data, agent, data.trading_days[:days]))
    return agent


def decision_prompts(llm: StubLLM) -> list[str]:
    return [user for role, user in llm.calls if role == "decision_agent"]


def hold_everything(payload: dict, call: int) -> str:
    return '<DECISION>{"decisions": {}}</DECISION>'


def test_prompts_follow_upstream_layout(two_stock_data) -> None:
    llm = StubLLM(hold_everything)
    run(two_stock_data, llm)
    filter_prompt = json.loads(llm.calls[0][1])
    assert list(filter_prompt) == ["portfolio_info", "symbols", "history"]
    assert filter_prompt["portfolio_info"] == {
        "total_assets": 100000.0,
        "available_cash": 100000.0,
        "position_value": 0.0,
    }
    assert list(filter_prompt["symbols"]["AAA"]["features"]) == ["market_data", "news_events", "position_state"]
    assert filter_prompt["symbols"]["AAA"]["features"]["news_events"] == {"top_k_events": ["No news available"]}
    assert filter_prompt["history"] == {"AAA": [], "BBB": []}
    decision = json.loads(decision_prompts(llm)[0])["symbols"]["AAA"]["features"]
    assert list(decision) == ["market_data", "news_events", "position_state", "filter_reasoning"]
    assert decision["market_data"]["date"] == DAYS[0]
    assert decision["filter_reasoning"] == "Technical analysis sufficient (no specific reasoning provided)"


def test_overspending_retries_with_upstream_notes(two_stock_data) -> None:
    def overspend_then_hold(payload: dict, call: int) -> str:
        if call == 1:
            return '<DECISION>{"decisions": {"AAA": {"action": "increase", "target_cash_amount": 150000}}}</DECISION>'
        return hold_everything(payload, call)

    llm = StubLLM(overspend_then_hold)
    agent = run(two_stock_data, llm)
    first, second = decision_prompts(llm)
    notes = second[len(first) :]
    # Upstream joins its notes with the two characters backslash and n, not newlines.
    assert notes.startswith("\\n\\n\\n\\n💰 INSUFFICIENT FUNDS: Total predicted cash usage 150000.00")
    assert "\\n\\n⚖️ CASH RATIO VIOLATION" in notes
    assert agent.stats.decision_validation_failures == 1


def test_reply_without_json_holds_everything_without_retry(two_stock_data) -> None:
    llm = StubLLM(lambda payload, call: "I would rather wait.", filter_reply="Nothing needs fundamentals.")
    agent = run(two_stock_data, llm, days=2)
    assert len(decision_prompts(llm)) == 2
    assert {s: d["action"] for s, d in agent.last_decisions.items()} == {"AAA": "hold", "BBB": "hold"}
    stats = agent.stats
    assert (stats.decision_unusable, stats.filter_unusable, stats.fallback_days) == (2, 2, 2)


def test_unreadable_fields_count_once_per_stock_not_per_call(two_stock_data) -> None:
    reply = '{"decisions": {"AAA": {"action": "hold", "confidence": "high"}, "BBB": {"action": "hold", "confidence": "low"}}}'
    agent = run(two_stock_data, StubLLM(lambda payload, call: reply))
    stats = agent.stats
    assert (stats.decision_calls, stats.decision_unusable, stats.symbol_parse_errors) == (1, 0, 2)
    assert stats.fallback_days == 1  # neither stock's decision could be used


def test_three_invalid_replies_fall_back_to_holding(two_stock_data) -> None:
    llm = StubLLM(lambda payload, call: '{"decisions": {"AAA": {"action": "decrease", "target_cash_amount": 10}}}')
    agent = run(two_stock_data, llm)
    assert len(decision_prompts(llm)) == 3
    assert agent.stats.fallback_days == 1 and agent.stats.decision_validation_failures == 3


def test_history_keeps_holds_and_executed_trades_only(two_stock_data) -> None:
    def buy_both_halves(payload: dict, call: int) -> str:
        if call == 1:
            return (
                '<DECISION>{"decisions": {'
                '"AAA": {"action": "increase", "target_cash_amount": 50000, "confidence": 0.8},'
                '"BBB": {"action": "increase", "target_cash_amount": 50000, "confidence": 0.8}}}</DECISION>'
            )
        return hold_everything(payload, call)

    llm = StubLLM(buy_both_halves)
    run(two_stock_data, llm, days=2)
    history = json.loads(decision_prompts(llm)[1])["history"]
    # AAA bought 500 shares. BBB's buy no longer fit after commission, so it is not recorded.
    assert history["AAA"] == [
        {
            "date": DAYS[0],
            "action": "increase",
            "cash_change": 50000.0,
            "target_cash_amount": 50000.0,
            "shares": 500.0,
            "confidence": 0.8,
        }
    ]
    assert history["BBB"] == []


def test_filter_fallbacks_and_fundamentals_flag() -> None:
    data = make_data({"AAA": [100.0] * 4, "BBB": [50.0] * 4})
    data.indicators[("AAA", DAYS[0])] = {"market_cap": 1e9, "pe_ratio": -3, "week_52_high": 120.0}
    for reply, expected in [('["AAA"]', True), ("{}", True), ("No fundamentals needed.", False)]:
        llm = StubLLM(hold_everything, filter_reply=reply)
        agent = run(data, llm)
        features = json.loads(decision_prompts(llm)[0])["symbols"]["AAA"]["features"]
        assert ("fundamental_data" in features) is expected
        assert agent.stats.filter_unusable == 1
        if expected:
            assert features["fundamental_data"]["market_cap"] == 1e9
            assert features["fundamental_data"]["pe_ratio"] == 0.0  # only positive values are kept
    llm = StubLLM(hold_everything, filter_reply='["AAA"]')
    run(data, llm, fundamentals=False)
    assert "fundamental_data" not in json.loads(decision_prompts(llm)[0])["symbols"]["AAA"]["features"]


def test_unparseable_reply_is_requested_again_like_upstream(two_stock_data) -> None:
    # An integer over Python's 4,300-digit limit raises ValueError, not a JSON decode error.
    # Upstream treats that like an API failure and calls the model again.
    huge = '{"decisions": {"AAA": {"action": "increase", "target_cash_amount": 1' + "0" * 5000 + "}}}"

    def huge_then_hold(payload: dict, call: int) -> str:
        return huge if call == 1 else '{"decisions": {"AAA": {"action": "hold"}}}'

    llm = StubLLM(huge_then_hold)
    agent = run(two_stock_data, llm)
    assert len(decision_prompts(llm)) == 2
    assert decision_prompts(llm)[0] == decision_prompts(llm)[1]  # same prompt, not a business retry
    assert agent.stats.decision_calls == 1 and agent.model_decided == 1
