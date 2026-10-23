"""Number rounding and JSON extraction for model replies, as upstream does them.

Ported from stockbench/utils/formatting.py and the OpenAI-provider branch of
stockbench/llm/llm_client.py (LLMClient.generate_json and its helpers) at the pinned
commit. The extraction order and repair steps are kept identical because they decide
which replies count as valid decisions.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from typing import Any

import demjson3
from json_repair import repair_json


def round_sig(value: Any, sig: int = 2) -> Any:
    """Upstream's _round_sig_number: 2 decimals at or above 1, else 2 significant digits."""
    try:
        if isinstance(value, bool):
            return value
        if value == 0 or value is None:
            return 0.0
        abs_val = abs(float(value))
        if abs_val == 0 or math.isinf(abs_val) or math.isnan(abs_val):
            return value
        if abs_val >= 1:
            return round(value, 2)
        digits = sig - 1 - int(math.floor(math.log10(abs_val)))
        return round(value, digits)
    except Exception:
        return value


def round_numbers(obj: Any, sig: int = 2) -> Any:
    """Upstream's round_numbers_in_obj: round every number inside dicts and lists."""
    if obj is None:
        return obj
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return round_sig(obj, sig)
    if isinstance(obj, dict):
        return {k: round_numbers(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_numbers(v, sig) for v in obj]
    return obj


def parse_reply(content: str | None) -> Any:
    """Turn a model reply into upstream's parsed result.

    Returns parsed JSON, or {"raw_content": text} when no JSON can be extracted.
    """
    if not content or not content.strip():
        return {"raw_content": ""}
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = extract_json(content)
        if extracted:
            return extracted
        return {"raw_content": content}


def extract_json(content: str) -> Any:
    """Upstream's _extract_json_with_improved_logic: try each strategy in order."""
    strategies: list[Callable[[str], str | None]] = [
        _from_decision_tag,
        _from_markdown_block,
        _first_complete_object,
        _largest_object,
        _from_truncated_response,
    ]
    for strategy in strategies:
        try:
            json_str = strategy(content)
            if json_str:
                json_str = json_str.strip()
                json_str = fix_common_json_issues(json_str)
                return json.loads(json_str)
        except (json.JSONDecodeError, AttributeError, TypeError):
            continue
    return None


def _match_braces(text: str, start: int) -> str | None:
    """Upstream's stack matcher: the object starting at `start`, or None if it never closes."""
    stack: list[str] = []
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        char = text[i]
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append(char)
        elif char == "}" and stack:
            stack.pop()
            if not stack:
                return text[start : i + 1]
    return None


def _from_decision_tag(content: str) -> str | None:
    match = re.search(r"<DECISION>\s*(.*?)\s*</DECISION>", content, re.DOTALL)
    if match:
        clean = match.group(1).strip()
        clean = re.sub(r"```json\s*", "", clean)
        clean = re.sub(r"```\s*$", "", clean)
        start = clean.find("{")
        if start >= 0:
            return _match_braces(clean, start)
    return None


def _from_markdown_block(content: str) -> str | None:
    match = re.search(r"```json\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        return match.group(1)
    match = re.search(r"```\s*(\{.*\})\s*```", content, re.DOTALL)
    if match:
        return match.group(1)
    return None


def _first_complete_object(content: str) -> str | None:
    start = content.find("{")
    if start == -1:
        return None
    return _match_braces(content, start)


def _largest_object(content: str) -> str | None:
    candidates = []
    for i, char in enumerate(content):
        if char == "{":
            found = _match_braces(content, i)
            if found is not None:
                candidates.append(found)
    if candidates:
        return max(candidates, key=len)
    return None


def _from_truncated_response(content: str) -> str | None:
    """Upstream's _handle_truncated_response: rebuild decisions from a cut-off reply."""
    try:
        if content.find('"decisions"') == -1:
            return None
        stock_decisions: dict[str, dict[str, Any]] = {}
        for symbol, action in re.findall(r'"([A-Z]{1,5})":\s*\{\s*"action":\s*"([^"]+)"', content):
            decision: dict[str, Any] = {"action": action}
            if content.find(f'"{symbol}":') == -1:
                continue
            target = re.search(rf'"{symbol}":[^}}]*?"target_cash_amount":\s*"([^"]+)"', content)
            if target:
                decision["target_cash_amount"] = target.group(1)
            elif action == "hold":
                decision["target_cash_amount"] = "0.0"
            else:
                continue
            conf = re.search(rf'"{symbol}":[^}}]*?"confidence":\s*"([^"]+)"', content)
            decision["confidence"] = conf.group(1) if conf else "0.5"
            reasons: list[str] = []
            reasons_match = re.search(rf'"{symbol}":[^}}]*?"reasons":\s*\[(.*?)\]', content, re.DOTALL)
            if reasons_match:
                items = re.findall(r'"([^"]*)"', reasons_match.group(1))
                if items:
                    reasons = items
            if not reasons:
                if action == "hold":
                    reasons = ["Hold decision extracted from truncated response"]
                else:
                    reasons = [
                        f"{action} decision extracted from truncated response",
                        "Based on partial available data",
                    ]
            decision["reasons"] = reasons
            stock_decisions[symbol] = decision
        if stock_decisions:
            result = {"decisions": stock_decisions}
            return json.dumps(result)
    except Exception:
        pass
    return None


def fix_common_json_issues(json_str: str) -> str:
    """Upstream's _fix_common_json_issues: json-repair, then demjson3, then regex fixes."""
    try:
        fixed = repair_json(json_str)
        json.loads(fixed)
        return fixed
    except Exception:
        pass
    try:
        return json.dumps(demjson3.decode(json_str, strict=False), ensure_ascii=False)
    except Exception:
        pass
    json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
    json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
    json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)
    json_str = re.sub(r"\\(\$)", r"\1", json_str)
    json_str = re.sub(r"\\(%)", r"\1", json_str)
    json_str = re.sub(r"\\(#)", r"\1", json_str)
    json_str = re.sub(r"\\(&)", r"\1", json_str)
    for phrase in [
        '"cash-heavy stock in doghouse"',
        '"pre-market"',
        '"high-risk"',
        '"post-earnings"',
        '"blue-chip"',
        '"high-yield"',
        '"market-cap"',
    ]:
        if phrase in json_str:
            json_str = json_str.replace(phrase, phrase.replace('"', '\\"'))
    return json_str
