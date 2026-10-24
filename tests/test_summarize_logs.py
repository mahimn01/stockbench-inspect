import statistics
import sys
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.model import ModelOutput, get_model

from stockbench_inspect import stockbench
from stockbench_inspect.metrics import sortino

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).parents[1] / "scripts" / "parity"))
from replies import reply
from summarize_logs import EpochRow, check_against_nav, collect, log_paths, paper_row, paper_sortino, render, spread


def test_paper_rows_are_found_for_the_two_local_models() -> None:
    assert paper_row("ollama/gpt-oss:20b")[1:] == (-2.8, -14.4, -0.0069)
    assert paper_row("ollama/qwen3:30b-a3b-thinking-2507-q4_K_M")[1:] == (2.1, -13.5, 0.0255)
    assert paper_row("ollama/gpt-oss:120b") is None
    assert paper_row("mockllm/model") is None


def test_paper_formula_sortino_is_the_root_mean_square_of_the_negative_returns() -> None:
    nav = [1.0, 1.02, 1.0, 1.01, 0.98]
    returns = [0.0] + [nav[i] / nav[i - 1] - 1 for i in range(1, len(nav))]
    negatives = [r for r in returns if r < 0]
    expected = statistics.fmean(returns) / (sum(r * r for r in negatives) / len(negatives)) ** 0.5
    assert paper_sortino(nav) == pytest.approx(expected)
    assert paper_sortino(nav) != pytest.approx(sortino(nav))  # the port keeps the released code's version
    assert paper_sortino([1.0, 1.01, 1.02]) == 0.0


def test_spread_of_one_value_is_not_a_number() -> None:
    assert spread([1.0]) != spread([1.0])
    assert spread([1.0, 3.0]) == pytest.approx(2**0.5)


def row(**changes) -> EpochRow:
    base = dict(
        log="a.eval",
        epoch=1,
        final_return=0.1,
        max_drawdown=-0.05,
        sortino=0.1,
        sortino_paper_formula=0.1,
        trades=3,
        decision_unusable_rate=0.0,
        filter_unusable_rate=0.0,
        fallback_days=0,
        baseline={},
    )
    return EpochRow(**{**base, **changes})


def test_check_against_nav_flags_a_log_that_disagrees_with_its_own_nav() -> None:
    nav = [1.0, 1.1, 0.99]
    raw = {"final_return": nav[-1] - 1.0, "max_drawdown": -0.1, "sortino": sortino(nav)}
    assert check_against_nav(row(trades=3), nav, raw, 3) == []
    problems = check_against_nav(row(trades=3), nav, {**raw, "final_return": 0.5}, 2)
    assert len(problems) == 2
    assert "final_return" in problems[0] and "trades" in problems[1]


def scripted_model(kind: str):
    def scripted(messages, tools, tool_choice, config) -> ModelOutput:
        return ModelOutput.from_content("mockllm/model", reply(messages[0].text, messages[1].text, kind))

    return get_model("mockllm/model", custom_outputs=scripted)


@pytest.mark.data
def test_summary_matches_the_mean_inspect_reports(tmp_path, market_data) -> None:
    [log] = eval(stockbench(days=5), model=scripted_model("equal"), log_dir=str(tmp_path), display="none")
    reported = {score.name: score.metrics["mean"].value for score in log.results.scores}

    # A run with -T days=N is a smoke run, so it is only counted when asked for.
    assert collect(log_paths([tmp_path]), None, include_smoke=False) == {}
    assert collect(log_paths([tmp_path]), "no-such-model", include_smoke=True) == {}
    summary = collect(log_paths([tmp_path]), "mockllm", include_smoke=True)["mockllm/model"]

    assert [r.epoch for r in summary.rows] == [1, 2, 3]
    assert not summary.notes  # every epoch agrees with its own NAV and trade list
    assert sum(r.trades for r in summary.rows) > 0
    for name, values in {
        "final_return": [r.final_return for r in summary.rows],
        "max_drawdown": [r.max_drawdown for r in summary.rows],
        "sortino": [r.sortino for r in summary.rows],
        "trades": [float(r.trades) for r in summary.rows],
        "decision_unusable_rate": [r.decision_unusable_rate for r in summary.rows],
        "filter_unusable_rate": [r.filter_unusable_rate for r in summary.rows],
        "fallback_days": [float(r.fallback_days) for r in summary.rows],
    }.items():
        assert statistics.fmean(values) == pytest.approx(reported[name]), name
    table = render(summary, paper_formula=True)
    assert "Mean of 3" in table and "Buy and hold (paper)" in table


@pytest.mark.data
def test_an_epoch_that_ended_in_an_error_is_left_out(tmp_path) -> None:
    def failed(messages, tools, tool_choice, config) -> ModelOutput:
        return ModelOutput.from_content("mockllm/model", "", error="upstream provider error")

    model = get_model("mockllm/model", custom_outputs=failed)
    eval(stockbench(days=1), model=model, epochs=1, log_dir=str(tmp_path), display="none")
    assert collect(log_paths([tmp_path]), None, include_smoke=True) == {}
