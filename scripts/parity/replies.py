"""Scripted model replies for the parity check.

Each reply depends only on the script name and the prompt text (it is seeded by the
prompt's SHA-256), so the original engine and this port get the same reply whenever
they send the same prompt. Three scripts:

- equal: day one buys every stock with an equal share of cash, then holds. The buys use
  almost all cash, so commission makes the last buy too expensive (upstream skips it).
- random: seeded holds, buys, trims and closes, with some overspending and invalid
  targets that trigger the agent's validation retries.
- adversarial: malformed and unusual replies for both agents (prose, fenced JSON,
  trailing commas, made-up tickers, text confidences, NaN targets, all cash in one stock
  to trigger the engine's retry, truncated JSON, and so on).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from typing import Any

SCRIPTS = ("equal", "random", "adversarial")


def reply(system_prompt: str, user_prompt: str, script: str) -> str:
    seed = hashlib.sha256(f"{script}\n{system_prompt}\n{user_prompt}".encode()).hexdigest()
    rng = random.Random(seed)
    payload, _ = json.JSONDecoder().raw_decode(user_prompt)
    if "<FILTER_RESULT>" in system_prompt:
        return _filter_reply(payload, rng, script)
    return _decision_reply(payload, rng, script)


def _filter_reply(payload: dict[str, Any], rng: random.Random, script: str) -> str:
    symbols = list(payload["symbols"])
    if script == "equal":
        return '<ANALYSIS>Routine day.</ANALYSIS>\n<FILTER_RESULT>\n{"stocks_need_fundamental": [], "reasoning": {}}\n</FILTER_RESULT>'
    chosen = rng.sample(symbols, k=rng.randint(0, 5))
    if rng.random() < 0.2:
        chosen.append("TSLA")
    reasoning = {s: f"Checked {s} on its 7-day closes." for s in symbols if rng.random() < 0.5}
    valid = json.dumps({"stocks_need_fundamental": chosen, "reasoning": reasoning}, indent=2)
    if script == "random":
        return f"<ANALYSIS>Scanning news and prices.</ANALYSIS>\n<FILTER_RESULT>\n{valid}\n</FILTER_RESULT>"
    variants = [
        f"<FILTER_RESULT>\n{valid}\n</FILTER_RESULT>",
        "No stock needs fundamentals today.",
        '["AAPL", "MSFT"]',
        '{"stocks_need_fundamental": "AAPL", "reasoning": {}}',
        '{"stocks_need_fundamental": [{"symbol": "AAPL"}], "reasoning": {}}',
        '{"stocks_need_fundamental": ["AAPL", "JPM"], "reasoning": ["not a dict"]}',
        "",
        "<ANALYSIS>Draft {notes} first.</ANALYSIS>\n<FILTER_RESULT>" + valid + "</FILTER_RESULT>",
        "{}",
        "```json\n" + valid + "\n```",
    ]
    return rng.choice(variants)


def _decision_reply(payload: dict[str, Any], rng: random.Random, script: str) -> str:
    symbols = list(payload["symbols"])
    cash = float(payload["portfolio_info"]["available_cash"])
    current = {s: float(payload["symbols"][s]["features"]["position_state"]["current_position_value"]) for s in symbols}

    if script == "equal":
        if all(v == 0 for v in current.values()):
            per_stock = math.floor(cash / len(symbols) * 100) / 100
            decisions = {
                s: {
                    "action": "increase",
                    "target_cash_amount": per_stock,
                    "reasons": ["Equal weight"],
                    "confidence": 0.6,
                }
                for s in symbols
            }
        else:
            decisions = {s: {"action": "hold", "reasons": ["Keep equal weight"], "confidence": 0.6} for s in symbols}
        return "<THINK>Equal weight.</THINK>\n<DECISION>\n" + json.dumps({"decisions": decisions}) + "\n</DECISION>"

    decisions = _random_decisions(symbols, cash, current, rng)
    body = json.dumps({"decisions": decisions}, indent=2)
    if script == "random":
        return f"<THINK>Rebalancing.</THINK>\n<DECISION>\n{body}\n</DECISION>"

    variant = rng.randrange(14)
    if variant == 0:
        return f"<DECISION>\n{body}\n</DECISION>"
    if variant == 1:
        return "Holding everything today, nothing to add."
    if variant == 2:
        return f"Here is my plan.\n```json\n{body}\n```"
    if variant == 3:
        return "<DECISION>" + body.replace("\n    }", ",\n    }").replace("\n  }", ",\n  }") + "</DECISION>"
    if variant == 4:
        decisions["TSLA"] = {"action": "increase", "target_cash_amount": 5000, "confidence": 0.9}
        decisions["NVDA"] = {"action": "close", "target_cash_amount": 0}
        return "<DECISION>" + json.dumps({"decisions": decisions}) + "</DECISION>"
    if variant == 5:
        for d in decisions.values():
            d["confidence"] = rng.choice(["high", "medium", 0.7])
        return "<DECISION>" + json.dumps({"decisions": decisions}) + "</DECISION>"
    if variant == 6:
        for d in decisions.values():
            if "target_cash_amount" in d:
                d["target_cash_amount"] = f"{d['target_cash_amount']:.2f}"
        return "<DECISION>" + json.dumps({"decisions": decisions}) + "</DECISION>"
    if variant == 7:
        target = rng.choice(symbols)
        all_in = {s: {"action": "hold", "confidence": 0.5} for s in symbols}
        all_in[target] = {"action": "increase", "target_cash_amount": current[target] + cash, "confidence": 0.9}
        return "<DECISION>" + json.dumps({"decisions": all_in}) + "</DECISION>"
    if variant == 8:
        return '{"decisions": [{"symbol": "AAPL", "action": "hold"}]}'
    if variant == 9:
        return "{}"
    if variant == 10:
        s = rng.choice(symbols)
        return (
            '<DECISION>{"decisions": {"'
            + s
            + '": {"action": "increase", "target_cash_amount": NaN, "confidence": 0.5}}}</DECISION>'
        )
    if variant == 11:
        odd = {}
        for s in symbols[:6]:
            odd[s] = {
                "action": rng.choice(["BUY", " increase ", "Increase", "DECREASE", "close"]),
                "target_cash_amount": rng.choice([-500, 0, current[s] + 100]),
                "confidence": 0.5,
            }
        return "<DECISION>" + json.dumps({"decisions": odd}) + "</DECISION>"
    if variant == 12:
        return "<DECISION>" + body[: len(body) * 2 // 3]
    return "<DECISION>\n```json\n" + body + "\n```\n</DECISION>"


def _random_decisions(
    symbols: list[str], cash: float, current: dict[str, float], rng: random.Random
) -> dict[str, dict[str, Any]]:
    actions = {}
    for s in symbols:
        r = rng.random()
        actions[s] = (
            "hold" if r < 0.4 else "increase" if r < 0.65 else "decrease" if r < 0.8 else "close" if r < 0.9 else None
        )
    buyers = [s for s in symbols if actions[s] == "increase"]
    budget = cash * (1.5 if rng.random() < 0.1 else rng.uniform(0.3, 0.95))
    weights = {s: rng.random() + 0.1 for s in buyers}
    total_weight = sum(weights.values()) or 1.0
    decisions: dict[str, dict[str, Any]] = {}
    for s in symbols:
        action = actions[s]
        if action is None:
            continue
        decision: dict[str, Any] = {
            "action": action,
            "reasons": [f"{action} {s}"],
            "confidence": rng.choice([0.55, 0.7, 0.85, 1.2]),
        }
        if action == "increase":
            target = current[s] + budget * weights[s] / total_weight
            if rng.random() < 0.05:
                target = current[s] * 0.5
            decision["target_cash_amount"] = round(target, 2)
        elif action == "decrease":
            decision["target_cash_amount"] = round(current[s] * rng.uniform(0.2, 0.9), 2)
        elif action == "close":
            decision["target_cash_amount"] = 0
        elif rng.random() < 0.3:
            decision["target_cash_amount"] = current[s]
        decisions[s] = decision
    return decisions
