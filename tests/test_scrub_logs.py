import getpass
import json
import sys
from pathlib import Path

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSpec, read_eval_log, write_eval_log
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.scorer import includes
from inspect_ai.solver import generate

from stockbench_inspect import stockbench

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from scrub_logs import NEWS_PLACEHOLDER, leftovers, main, news_from_data, news_leftovers, redact_news, scrub


@pytest.mark.data
def test_scrub_removes_local_paths_and_keeps_the_scores(tmp_path, market_data) -> None:
    [log] = eval(stockbench(days=2), model="mockllm/model", log_dir=str(tmp_path / "in"), display="none")
    user = getpass.getuser()
    log.eval.metadata = {
        "home": f"{Path.home()}/work/logs",
        "other_user": f"/Users/{user}/x",
        "disk": "/Volumes/MacOffload/stockbench-work/logs",
        "disk_in_text": 'see "/Volumes/Backup Disk/a" or /Volumes/Other/b',
    }
    assert leftovers(log) >= 5

    scrubbed = scrub(log)
    assert leftovers(scrubbed) == 0
    assert scrubbed.eval.metadata["home"] == "~/work/logs"
    assert scrubbed.eval.metadata["disk"] == "/Volumes/volume/stockbench-work/logs"
    assert scrubbed.eval.metadata["disk_in_text"].startswith('see "/Volumes/volume')

    target = tmp_path / "out" / "log.eval"
    target.parent.mkdir()
    write_eval_log(scrubbed, str(target))
    reread = read_eval_log(str(target))
    assert reread.status == log.status
    assert [s.scores["stockbench_scorer"].value for s in reread.samples] == [
        s.scores["stockbench_scorer"].value for s in log.samples
    ]
    assert [s.scores["stockbench_scorer"].metadata["nav"] for s in reread.samples] == [
        s.scores["stockbench_scorer"].metadata["nav"] for s in log.samples
    ]


HEADLINES = [
    "Acme Corp beats estimates for the third quarter - Revenue rose 12% to $4.1 billion on strong cloud demand.",
    'Regulators open a probe into "Globex" buyback plans - The inquiry covers filings from 2024 and 2025.',
]
TITLE = "Acme Corp beats estimates for the third quarter"
PROMPT = (
    'Rules: hold unless sure.\n{"symbol": "ACME", "features": {"market_data": {"open": 101.5, "close_7d": [99.0, 100.0]}, '
    f'"news_events": {{"top_k_events": {json.dumps(HEADLINES)}}}, "position_state": {{"shares": 3.0}}}}}}\n'
    '{"symbol": "GLBX", "features": {"news_events": {"top_k_events": ["No news available"]}}}'
)
REPLY = f"I will hold ACME. {TITLE} is already priced in. Also: {HEADLINES[1]}"


def _mock_log(tmp_path, reply=REPLY):
    task = Task(
        dataset=[Sample(input=PROMPT, target="hold")],
        solver=generate(),
        scorer=includes(),
    )
    model = get_model("mockllm/model", custom_outputs=[ModelOutput.from_content("mockllm/model", reply)])
    [log] = eval(task, model=model, log_dir=str(tmp_path / "in"), display="none")
    return log


def test_redact_news_replaces_the_lists_and_any_quotes_of_them(tmp_path) -> None:
    log = _mock_log(tmp_path)
    known = {TITLE}
    assert news_leftovers(log, known) > 0

    redacted, stats = redact_news(log, known)

    assert news_leftovers(redacted, known | set(HEADLINES)) == 0
    assert stats.items >= 2
    assert stats.unparsed == 0
    text = redacted.model_dump_json()
    for headline in HEADLINES:
        assert headline not in text
        assert json.dumps(headline)[1:-1] not in text
    assert TITLE not in text
    prompt = redacted.samples[0].input
    assert prompt.count(NEWS_PLACEHOLDER) == 2
    assert '"top_k_events": ["No news available"]' in prompt
    assert prompt.startswith("Rules: hold unless sure.\n")
    assert '"open": 101.5, "close_7d": [99.0, 100.0]' in prompt
    assert f"I will hold ACME. {NEWS_PLACEHOLDER} is already priced in." in redacted.samples[0].output.completion
    assert redacted.samples[0].scores["includes"].value == log.samples[0].scores["includes"].value
    assert redacted.status == log.status


def test_redact_news_replaces_a_partial_quote_but_not_short_or_unrelated_text(tmp_path) -> None:
    reply = (
        'Quoting "the inquiry covers filings from 2024" but also "Revenue rose 12% to $4.1 billion on strong" here.\n'
        "Short phrases such as revenue rose 12% stay. The inquiry covers nothing else today."
    )
    log = _mock_log(tmp_path, reply)

    redacted, stats = redact_news(log, set())

    out = redacted.samples[0].output.completion
    assert out == (
        f'Quoting "{NEWS_PLACEHOLDER}" but also "{NEWS_PLACEHOLDER}" here.\n'
        "Short phrases such as revenue rose 12% stay. The inquiry covers nothing else today."
    )
    assert stats.partial >= 2
    assert news_leftovers(redacted, set(HEADLINES)) == 0
    # Switched off, the partial quotes stay and are counted as leftovers.
    kept, _ = redact_news(log, set(), min_words=0)
    assert "the inquiry covers filings from 2024" in kept.samples[0].output.completion
    assert news_leftovers(kept, set(HEADLINES), min_words=0) == 0
    assert news_leftovers(kept, set(HEADLINES)) > 0


def test_redact_news_does_not_change_the_original_or_touch_short_strings(tmp_path) -> None:
    log = _mock_log(tmp_path)
    before = log.model_dump_json()
    redacted, _ = redact_news(log, {"ACME", "hold"})
    assert log.model_dump_json() == before
    assert "I will hold ACME" in redacted.samples[0].output.completion


def test_redact_news_counts_a_list_it_cannot_parse() -> None:
    log = EvalLog(
        status="success",
        eval=EvalSpec(created="2026-01-01", task="t", dataset=EvalDataset(), model="m", config=EvalConfig()),
    )
    log.eval.metadata = {"broken": 'x "top_k_events": ["cut off headline that never ends'}
    redacted, stats = redact_news(log)
    assert stats.unparsed == 1
    assert news_leftovers(redacted, set()) >= 1


def test_cli_redacts_news_and_reports_it(tmp_path, monkeypatch, capsys) -> None:
    log = _mock_log(tmp_path)
    source = Path(log.location)
    monkeypatch.setattr(
        sys, "argv", ["scrub_logs.py", str(source), "--out", str(tmp_path / "out"), "--redact-news", "--no-dataset"]
    )
    main()
    out = capsys.readouterr().out
    assert "news: " in out and "list items, " in out and "partial quotes" in out
    assert out.rstrip().endswith("clean)")
    reread = read_eval_log(str(tmp_path / "out" / source.name))
    assert HEADLINES[0] not in reread.model_dump_json()
    assert reread.samples[0].scores["includes"].value == log.samples[0].scores["includes"].value


@pytest.mark.data
def test_redact_news_on_a_real_run_keeps_the_scores(tmp_path, market_data) -> None:
    [log] = eval(stockbench(days=3), model="mockllm/model", log_dir=str(tmp_path / "in"), display="none")
    known = news_from_data(market_data)
    assert len(known) > 1000
    assert news_leftovers(log, known) > 0

    redacted, stats = redact_news(log, known)

    assert stats.items > 0
    assert stats.unparsed == 0
    assert news_leftovers(redacted, known | stats.found) == 0
    target = tmp_path / "out" / "log.eval"
    target.parent.mkdir()
    write_eval_log(redacted, str(target))
    reread = read_eval_log(str(target))
    assert news_leftovers(reread, known | stats.found) == 0
    assert reread.status == log.status
    score = "stockbench_scorer"
    assert [s.scores[score].value for s in reread.samples] == [s.scores[score].value for s in log.samples]
    assert [s.scores[score].metadata["nav"] for s in reread.samples] == [
        s.scores[score].metadata["nav"] for s in log.samples
    ]
    assert [e.event for e in reread.samples[0].events] == [e.event for e in log.samples[0].events]
