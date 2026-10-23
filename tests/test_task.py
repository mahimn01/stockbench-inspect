import asyncio
import sys
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model

from stockbench_inspect import stockbench, stockbench_agent, stockbench_scorer
from stockbench_inspect.agent import StockBenchAgent
from stockbench_inspect.engine import run_backtest
from stockbench_inspect.tasks import is_failed_call

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "parity"))
from replies import reply


def scores(log) -> dict[str, float]:
    return {score.name: score.metrics["mean"].value for score in log.results.scores}


@pytest.mark.data
def test_mock_model_runs_end_to_end_and_holds(tmp_path) -> None:
    [log] = eval(stockbench(days=2), model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    result = scores(log)
    assert result["final_return"] == 0.0 and result["trades"] == 0.0
    assert result["baseline_return"] != 0.0
    # mockllm answers in prose, so no reply is usable and every day is a fallback day.
    assert result["decision_unusable_rate"] == 1.0 and result["filter_unusable_rate"] == 1.0
    assert result["fallback_days"] == 2.0
    assert log.results.headline.score == "final_return"


@pytest.mark.data
def test_inspect_wrapper_matches_the_engine(tmp_path, market_data) -> None:
    def scripted(messages, tools, tool_choice, config) -> ModelOutput:
        return ModelOutput.from_content("mockllm/model", reply(messages[0].text, messages[1].text, "equal"))

    model = get_model("mockllm/model", custom_outputs=scripted)
    [log] = eval(stockbench(days=5), model=model, log_dir=str(tmp_path), display="none")
    assert log.status == "success"

    async def llm(role: str, system_prompt: str, user_prompt: str) -> str:
        return reply(system_prompt, user_prompt, "equal")

    direct = asyncio.run(run_backtest(market_data, StockBenchAgent(market_data, llm), market_data.trading_days[:5]))
    navs = [sample.scores["stockbench_scorer"].metadata["nav"] for sample in log.samples]
    assert navs == [direct.nav] * 3  # every epoch follows the engine's daily NAV exactly
    assert scores(log)["trades"] == len(direct.trades) > 0


@pytest.mark.data
def test_solver_and_scorer_work_on_their_own(tmp_path) -> None:
    task = Task(dataset=[Sample(input="trade")], solver=stockbench_agent(days=1), scorer=stockbench_scorer())
    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "success"
    sample = log.samples[0]
    assert sample.store["BacktestState:days"] == ["2025-03-03"]
    score = sample.scores["stockbench_scorer"]
    assert score.value["final_return"] == 0.0 and score.metadata["nav"] == [1.0]


@pytest.mark.data
@pytest.mark.parametrize("days", [0, -1, 84])
def test_days_outside_the_window_are_rejected(days: int, tmp_path) -> None:
    with pytest.raises(ValueError, match="days must be between 1 and 83"):
        stockbench(days=days)
    task = Task(dataset=[Sample(input="trade")], solver=stockbench_agent(days=days), scorer=stockbench_scorer())
    [log] = eval(task, model="mockllm/model", log_dir=str(tmp_path), display="none")
    assert log.status == "error" and "days must be between 1 and 83" in str(log.error)


@pytest.mark.data
def test_error_returned_as_a_reply_fails_the_sample(tmp_path) -> None:
    def failed(messages, tools, tool_choice, config) -> ModelOutput:
        return ModelOutput.from_content("mockllm/model", "", error="upstream provider error")

    model = get_model("mockllm/model", custom_outputs=failed)
    [log] = eval(stockbench(days=1), model=model, epochs=1, log_dir=str(tmp_path), display="none")
    assert log.status == "error"
    assert "fundamental_filter call failed" in str(log.error)


def test_rejected_requests_count_as_failed_calls() -> None:
    too_long = ModelOutput.from_content("m", "Input is too long", stop_reason="model_length")
    assert too_long.usage is None and is_failed_call(too_long)
    assert is_failed_call(ModelOutput.from_content("m", "blocked", stop_reason="content_filter"))
    assert not is_failed_call(ModelOutput.from_content("m", "<DECISION>{}</DECISION>"))
    assert not is_failed_call(ModelOutput.from_content("m", "partial reply", stop_reason="max_tokens"))
