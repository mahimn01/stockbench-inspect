import math

from stockbench_inspect.parsing import parse_reply, round_numbers, round_sig


def test_round_sig_follows_upstream_rules() -> None:
    assert round_sig(0) == 0.0 and isinstance(round_sig(0), float)
    assert round_sig(1234.5678) == 1234.57
    assert round_sig(0.012345) == 0.012
    assert round_sig(-0.5555) == -0.56
    assert round_sig(True) is True
    assert math.isnan(round_sig(float("nan")))
    assert round_sig(7) == 7 and isinstance(round_sig(7), int)


def test_round_numbers_recurses_into_dicts_and_lists() -> None:
    assert round_numbers({"a": [1.234, 0.0001234], "b": "text", "c": None}) == {
        "a": [1.23, 0.00012],
        "b": "text",
        "c": None,
    }


def test_plain_json_reply() -> None:
    assert parse_reply('{"decisions": {"AAPL": {"action": "hold"}}}') == {"decisions": {"AAPL": {"action": "hold"}}}


def test_decision_tag_wins_over_earlier_braces() -> None:
    reply = 'Thinking about {weights}.\n<DECISION>\n{"decisions": {"MSFT": {"action": "close", "target_cash_amount": 0}}}\n</DECISION>'
    assert parse_reply(reply) == {"decisions": {"MSFT": {"action": "close", "target_cash_amount": 0}}}


def test_fenced_json_and_trailing_commas_are_repaired() -> None:
    reply = 'Plan:\n```json\n{"decisions": {"JPM": {"action": "hold",},},}\n```'
    assert parse_reply(reply) == {"decisions": {"JPM": {"action": "hold"}}}


def test_prose_and_empty_replies_become_raw_content() -> None:
    assert parse_reply("Holding everything today.") == {"raw_content": "Holding everything today."}
    assert parse_reply("   ") == {"raw_content": ""}
    assert parse_reply(None) == {"raw_content": ""}


def test_nan_target_survives_parsing() -> None:
    parsed = parse_reply('<DECISION>{"decisions": {"V": {"action": "increase", "target_cash_amount": NaN}}}</DECISION>')
    assert math.isnan(parsed["decisions"]["V"]["target_cash_amount"])
