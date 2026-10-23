"""The StockBench agent: prompt inputs, the two model calls, validation, retries and orders.

Ported from upstream's LLM decision strategy (backtest/strategies/llm_decision.py), the
dual-agent code (agents/dual_agent_llm.py, agents/fundamental_filter_agent.py) and the
feature builder (core/features.py) at the pinned commit. Control flow follows upstream
closely, quirks included, so the prompts and decisions match the original. Comments mark
the places where upstream behaviour is surprising but kept on purpose.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from importlib import resources
from typing import Any

from stockbench_inspect.data import (
    DECISION_PRICE_LOOKBACK_DAYS,
    FILTER_PRICE_LOOKBACK_DAYS,
    INDICATOR_FIELDS,
    NEWS_MAX_ITEMS,
    MarketData,
)
from stockbench_inspect.parsing import parse_reply, round_numbers

# llm(role, system_prompt, user_prompt) returns the reply text. role is
# "fundamental_filter" or "decision_agent".
LLM = Callable[[str, str, str], Awaitable[str | None]]

MAX_ATTEMPTS = 3  # upstream agents.retry.max_attempts, shared by the engine and the agent
API_ATTEMPTS = 4  # upstream LLM profile retry.max_retries (3) plus the first try
MIN_CASH_RATIO = 0.0  # upstream portfolio.min_cash_ratio
MAX_RECORDS_PER_SYMBOL = 7
MAX_HISTORY_DAYS = 30


def load_prompt(name: str) -> str:
    return resources.files("stockbench_inspect").joinpath("prompts", name).read_text(encoding="utf-8")


FILTER_PROMPT = load_prompt("fundamental_filter_v1.txt")
DECISION_PROMPT = load_prompt("decision_agent_v1.txt")


@dataclass
class AgentStats:
    """Counts of how often the model's replies could be used. Not part of upstream."""

    filter_calls: int = 0
    filter_unusable: int = 0  # filter replies with no usable JSON (upstream falls back silently)
    decision_calls: int = 0
    decision_unusable: int = 0  # decision replies with no decision for any of the 20 stocks
    decision_validation_failures: int = 0  # replies sent back for breaking the cash or action rules
    symbol_parse_errors: int = 0  # single-stock decisions that could not be read and became holds
    fallback_days: int = 0  # days on which no stock's final decision came from the model
    engine_retries: int = 0


@dataclass
class StockBenchAgent:
    """Upstream's Strategy plus decide_batch_dual_agent, driven by the engine each day."""

    data: MarketData
    llm: LLM
    use_fundamentals: bool = True
    stats: AgentStats = field(default_factory=AgentStats)
    decision_history: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pending_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_date: str | None = None
    last_decisions: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_decided: int = 0  # stocks whose latest decision came from the model's reply

    async def on_bar(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the day's features, run both agents, and turn decisions into orders."""
        open_map: dict[str, float] = ctx["open_map"]
        if not open_map:
            return []
        day: str = ctx["date"]
        features_list = self._build_features_for_day(ctx)
        self._cleanup_old_history(day)
        current_symbols = [fi["symbol"] for fi in features_list]
        history = self._history_for_prompt(current_symbols)
        decisions = await self._decide(features_list, history, ctx, ctx.get("rejected_orders"))

        orders: list[dict[str, Any]] = []
        portfolio = ctx["portfolio"]
        for fi in features_list:
            symbol = fi["symbol"]
            decision = decisions.get(symbol, {})
            action = decision.get("action", "hold")
            ref_px = open_map.get(symbol)
            if ref_px is None or ref_px <= 0:
                continue
            pos = portfolio.positions.get(symbol)
            current_value = (pos.shares * float(ref_px)) if pos else 0.0
            if action == "hold" and "target_cash_amount" not in decision:
                target_cash = current_value
            else:
                target_cash = float(decision.get("target_cash_amount", 0.0))
            target_value = max(0.0, target_cash)
            delta_value = target_value - current_value
            if (action == "increase" or action == "decrease") and abs(delta_value) < 0.01:
                action = "hold"
            if action == "increase" and delta_value > 0:
                qty = round(delta_value / float(ref_px), 2)
                if qty > 0:
                    orders.append({"symbol": symbol, "side": "buy", "qty": qty})
            elif action in ("decrease", "close") and delta_value < 0:
                qty = round(abs(delta_value) / float(ref_px), 2)
                if pos and pos.shares > 0:
                    qty = min(qty, pos.shares)
                if qty > 0:
                    orders.append({"symbol": symbol, "side": "sell", "qty": -qty})

        self.pending_decisions = dict(decisions)
        self.pending_date = day
        self.last_decisions = decisions
        return orders

    def record_executed_decisions(self, executed_symbols: list[str], portfolio: Any) -> None:
        """Holds are always recorded. Other actions only if an order for that symbol executed."""
        if not self.pending_decisions:
            return
        if self.model_decided == 0:
            self.stats.fallback_days += 1
        final: dict[str, dict[str, Any]] = {}
        for symbol, decision in self.pending_decisions.items():
            action = decision.get("action", "hold").lower()
            position = portfolio.positions.get(symbol)
            decision["shares"] = round(float(position.shares), 2) if position else 0.0
            if action == "hold" or symbol in executed_symbols:
                final[symbol] = decision
        if final and self.pending_date is not None:
            self._clear_decisions_for_date(self.pending_date)
            for symbol, decision in final.items():
                records = self.decision_history.setdefault(symbol, [])
                records.insert(0, {"date": self.pending_date, "decision": decision.copy()})
                if len(records) > MAX_RECORDS_PER_SYMBOL:
                    self.decision_history[symbol] = records[:MAX_RECORDS_PER_SYMBOL]
        self.pending_decisions = {}
        self.pending_date = None

    def _cleanup_old_history(self, day: str) -> None:
        cutoff = (date.fromisoformat(day) - timedelta(days=MAX_HISTORY_DAYS)).isoformat()
        for symbol in list(self.decision_history):
            valid = [entry for entry in self.decision_history[symbol] if entry["date"] >= cutoff]
            self.decision_history[symbol] = valid
            if not valid:
                del self.decision_history[symbol]

    def _clear_decisions_for_date(self, day: str) -> None:
        for symbol in list(self.decision_history):
            remaining = [entry for entry in self.decision_history[symbol] if entry.get("date") != day]
            if len(remaining) != len(self.decision_history[symbol]):
                self.decision_history[symbol] = remaining
                if not remaining:
                    del self.decision_history[symbol]

    def _history_for_prompt(self, symbols: list[str]) -> dict[str, list[dict[str, Any]]]:
        history: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            records = []
            for entry in self.decision_history.get(symbol, []):
                decision = entry["decision"]
                records.append(
                    {
                        "date": entry["date"],
                        "action": decision.get("action", "hold"),
                        "cash_change": decision.get("cash_change", 0.0),
                        "target_cash_amount": decision.get("target_cash_amount", 0.0),
                        "shares": decision.get("shares", 0.0),
                        "confidence": decision.get("confidence", 0.5),
                    }
                )
            history[symbol] = records
        return history

    def _build_features_for_day(self, ctx: dict[str, Any]) -> list[dict[str, Any]]:
        """Filter-agent features: price, last 7 closes, news and position for each symbol."""
        features_list = []
        open_map: dict[str, float] = ctx["open_map"]
        portfolio = ctx["portfolio"]
        day: str = ctx["date"]
        for symbol in open_map:
            ref_price = open_map.get(symbol, 0.0)
            position = portfolio.positions.get(symbol)
            current_position_value = 0.0
            if position and position.shares:
                current_position_value = position.shares * ref_price
            holding_days = int(position.holding_days or 0) if position else 0
            shares = round(float((position.shares if position else 0) or 0), 2)

            news_texts = []
            for item in self.data.news_items(symbol, day):
                title = item.get("title", "")
                description = item.get("description", "")
                if title:
                    if description and description.strip():
                        news_texts.append(f"{title} - {description}")
                    else:
                        news_texts.append(title)
            if not news_texts:
                news_texts = ["No news available"]

            features_list.append(
                {
                    "symbol": symbol,
                    "features": {
                        "market_data": {
                            "ticker": symbol,
                            "open": ref_price,
                            "close_7d": self.data.close_7d(symbol, day, FILTER_PRICE_LOOKBACK_DAYS),
                        },
                        "news_events": {"top_k_events": news_texts},
                        "position_state": {
                            "current_position_value": current_position_value,
                            "holding_days": holding_days,
                            "shares": shares,
                        },
                    },
                }
            )
        return features_list

    def _decision_features(self, item: dict[str, Any], day: str, with_fundamentals: bool) -> dict[str, Any]:
        """Upstream's build_features_for_prompt for the decision agent (include_price=False)."""
        symbol = item["symbol"]
        features = item["features"]
        ref_price = features["market_data"]["open"]
        # Upstream rebuilds news from the filter features' strings, keeping the first 5.
        top_events = []
        for event in features["news_events"]["top_k_events"][:NEWS_MAX_ITEMS]:
            if event:
                top_events.append(event.strip())
        sections: dict[str, Any] = {
            "market_data": {
                "ticker": symbol,
                "open": float(self.data.open_price(symbol, day)),
                "close_7d": self.data.close_7d(symbol, day, DECISION_PRICE_LOOKBACK_DAYS),
                "date": day,
            },
            "news_events": {"top_k_events": top_events if top_events else ["No news available"]},
            "position_state": {
                "current_position_value": float(features["position_state"]["current_position_value"]),
                "holding_days": int(features["position_state"]["holding_days"]),
                "shares": float(features["position_state"]["shares"]),
            },
        }
        if with_fundamentals:
            fundamentals = dict.fromkeys(INDICATOR_FIELDS, 0.0)
            if ref_price and ref_price > 0:
                indicators = self.data.indicators_for(symbol, day) or dict.fromkeys(INDICATOR_FIELDS, 0.0)
                try:
                    for key in INDICATOR_FIELDS:
                        value = indicators.get(key, 0)
                        if value > 0:
                            fundamentals[key] = float(value)
                except Exception:
                    pass  # upstream keeps whatever was filled in before the error
            sections["fundamental_data"] = fundamentals
        return sections

    async def _filter(
        self, features_list: list[dict[str, Any]], history: dict[str, Any], ctx: dict[str, Any]
    ) -> dict[str, Any]:
        """The fundamental-filter agent: which symbols get fundamentals in the decision prompt."""
        symbols: dict[str, Any] = {}
        total_current_position = 0.0
        for item in features_list:
            filtered = {k: v for k, v in item["features"].items() if k != "fundamental_data"}
            total_current_position += filtered.get("position_state", {}).get("current_position_value", 0.0)
            symbols[item["symbol"]] = {"features": filtered}
        current_cash = float(ctx["portfolio"].cash)
        filter_input = {
            "portfolio_info": {
                "total_assets": current_cash + total_current_position,
                "available_cash": current_cash,
                "position_value": total_current_position,
            },
            "symbols": symbols,
            "history": history,
        }
        user_prompt = json.dumps(round_numbers(filter_input, 2), ensure_ascii=False)
        self.stats.filter_calls += 1
        data = await self._generate_json("fundamental_filter", FILTER_PROMPT, user_prompt)
        all_symbols = [item["symbol"] for item in features_list]

        def fallback(reason: str) -> dict[str, Any]:
            self.stats.filter_unusable += 1
            return {"stocks_need_fundamental": all_symbols, "reasoning": dict.fromkeys(all_symbols, reason)}

        if not data or not isinstance(data, dict):
            return fallback("LLM returned invalid format, requiring fundamental analysis as fallback")
        if "raw_content" in data:
            # No JSON in the reply. Upstream reads this as "no stock needs fundamentals".
            self.stats.filter_unusable += 1
        stocks_need_fundamental = data.get("stocks_need_fundamental", [])
        reasoning = data.get("reasoning", {})
        if not isinstance(stocks_need_fundamental, list):
            return fallback("Invalid filter result format, requiring fundamental analysis as fallback")
        if not isinstance(reasoning, dict):
            reasoning = {}
        input_symbols = set(all_symbols)
        for symbol in input_symbols:
            if symbol not in reasoning:
                if symbol in stocks_need_fundamental:
                    reasoning[symbol] = "Requires fundamental analysis (no specific reasoning provided)"
                else:
                    reasoning[symbol] = "Technical analysis sufficient (no specific reasoning provided)"
        unhashable = next((s for s in stocks_need_fundamental if isinstance(s, (dict, list))), None)
        if unhashable is not None:
            # Upstream's set lookup fails on a JSON object or list, and its catch-all puts the
            # error text in the prompt. No other parsed reply can raise in this step. Use
            # Python 3.11's wording (upstream's version) so prompts match on any Python.
            error = f"unhashable type: '{type(unhashable).__name__}'"
            return fallback(f"Error during filtering ({error[:50]}), requiring fundamental analysis as fallback")
        selected = [s for s in stocks_need_fundamental if s in input_symbols]
        return {"stocks_need_fundamental": selected, "reasoning": reasoning}

    async def _generate_json(self, role: str, system_prompt: str, user_prompt: str) -> Any:
        """Call the model and parse the reply the way upstream's generate_json does.

        Upstream calls the API again when parsing fails with anything other than a JSON
        decode error (for example an integer longer than Python's 4,300-digit limit), up
        to 4 tries in all, and then returns None.
        """
        for _ in range(API_ATTEMPTS):
            reply = await self._call_model(role, system_prompt, user_prompt)
            try:
                return parse_reply(reply)
            except Exception:
                continue
        return None

    async def _call_model(self, role: str, system_prompt: str, user_prompt: str) -> str | None:
        try:
            return await self.llm(role, system_prompt, user_prompt)
        except Exception as e:
            raise _ModelCallFailed() from e

    async def _decide(
        self,
        features_list: list[dict[str, Any]],
        history: dict[str, Any],
        ctx: dict[str, Any],
        rejected_orders: list[dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Upstream's decide_batch_dual_agent: filter call, decision features, decision loop.

        As upstream, an unexpected error in this code makes every stock hold for the day.
        Unlike upstream, a failed model call is not swallowed: it fails the sample.
        """
        try:
            day: str = ctx["date"]
            filter_result = await self._filter(features_list, history, ctx)
            stocks_need_fundamental = filter_result.get("stocks_need_fundamental", [])
            reasoning = filter_result.get("reasoning", {})
            enhanced_list = []
            for item in features_list:
                symbol = item["symbol"]
                with_fundamentals = self.use_fundamentals and symbol in stocks_need_fundamental
                enhanced = self._decision_features(item, day, with_fundamentals)
                enhanced["filter_reasoning"] = reasoning.get(symbol, "No reasoning provided")
                enhanced_list.append({"symbol": symbol, "features": enhanced})
            return await self._decision_loop(enhanced_list, history, ctx, rejected_orders)
        except _ModelCallFailed as failure:
            assert failure.__cause__ is not None
            raise failure.__cause__ from None
        except Exception as e:
            self.model_decided = 0
            return {
                item["symbol"]: _hold(
                    float(item["features"]["position_state"].get("current_position_value", 0.0)),
                    f"Dual-agent error ({str(e)[:50]}), {item['symbol']} maintains current position",
                )
                for item in features_list
            }

    async def _decision_loop(
        self,
        features_list: list[dict[str, Any]],
        history: dict[str, Any],
        ctx: dict[str, Any],
        rejected_orders: list[dict[str, Any]] | None,
    ) -> dict[str, dict[str, Any]]:
        """Upstream's _decide_batch_portfolio_dual_agent, with its shared retry budget.

        Variables deliberately keep upstream's lifetimes: values left over from an earlier
        symbol or attempt can feed later checks, exactly as they do upstream.
        """
        results: dict[str, dict[str, Any]] = {}
        symbols: dict[str, Any] = {}
        total_current_position = 0.0
        for item in features_list:
            features = item["features"]
            total_current_position += features.get("position_state", {}).get("current_position_value", 0.0)
            symbols[item["symbol"]] = {"features": features}

        current_cash = float(ctx["portfolio"].cash)
        total_assets = current_cash + total_current_position
        available_cash = current_cash
        min_cash_ratio = MIN_CASH_RATIO
        portfolio_input = {
            "portfolio_info": {
                "total_assets": total_assets,
                "available_cash": available_cash,
                "position_value": total_current_position,
            },
            "symbols": symbols,
            "history": history,
        }
        base_user_prompt = json.dumps(round_numbers(portfolio_input, 2), ensure_ascii=False)

        # Rejected orders from the engine add one "rebalance" notice and set the retry budget.
        order_rejection_info: list[str] = []
        current_retry_attempt = 0
        engine_retry_count = 0
        if rejected_orders:
            rejected_by_symbol = {order.get("symbol"): order for order in rejected_orders}
            for symbol in symbols:
                rejection = rejected_by_symbol.get(symbol)
                if rejection:
                    rejection_reason = rejection.get("reason", "Order rejected")
                    rejection_context = rejection.get("context", {})
                    engine_retry_count = max(engine_retry_count, rejection.get("retry_count", 0))
                    if rejection_context.get("portfolio_rebalance_needed", False):
                        total_required = rejection_context.get("total_cash_required_all_orders", 0)
                        # Upstream overwrites available_cash here. The value is the same cash.
                        available_cash = rejection_context.get("available_cash", 0)
                        cash_shortfall = rejection_context.get("cash_shortfall", 0)
                        suggestion = rejection_context.get("suggestion", "")
                        # Upstream writes "\\n", so the model sees a literal backslash-n.
                        prompt = "\\n\\n🚨 CRITICAL: PORTFOLIO-WIDE CASH CONSTRAINT VIOLATION\\n"
                        prompt += "❌ Previous attempt failed due to insufficient total cash for all positions\\n"
                        prompt += "📊 Financial Summary:\\n"
                        prompt += f"   • Available Cash: ${available_cash:,.2f}\\n"
                        prompt += f"   • Total Required: ${total_required:,.2f}\\n"
                        prompt += f"   • Cash Shortfall: ${cash_shortfall:,.2f}\\n"
                        prompt += f"\\n💡 ACTION REQUIRED: {suggestion}\\n"
                        prompt += (
                            "\\n🔄 You MUST rebalance the ENTIRE portfolio to fit within the available cash budget.\\n"
                        )
                        prompt += "Consider reducing all position sizes proportionally or selecting fewer stocks.\\n"
                        if len(order_rejection_info) == 0:
                            order_rejection_info.append(prompt)
                    else:
                        prompt = f"\\n\\n❌ IMPORTANT: Previous order for {symbol} was rejected.\\n"
                        prompt += f"Rejection reason: {rejection_reason}\\n"
                        if rejection_context:
                            prompt += f"Additional context: {rejection_context}\\n"
                        prompt += f"Please provide a corrected decision for {symbol} that addresses this rejection.\\n"
                        order_rejection_info.append(prompt)
                    current_retry_attempt = engine_retry_count

        retry_count = current_retry_attempt
        retry_notes: str | None = None
        # Upstream reuses these across symbols and attempts (see the docstring).
        target_cash_amount: Any = None
        current_position_value: Any = None

        def hold_all() -> dict[str, dict[str, Any]]:
            self.model_decided = 0
            for sym in symbols:
                value = symbols[sym]["features"].get("position_state", {}).get("current_position_value", 0.0)
                results[sym] = _hold(value, "Dual-agent fallback: maintaining current position")
            return results

        while True:
            if engine_retry_count + retry_count >= MAX_ATTEMPTS:
                return hold_all()
            if order_rejection_info:
                user_prompt = base_user_prompt + "\\n\\n" + "\\n".join(order_rejection_info)
            else:
                user_prompt = base_user_prompt
            if retry_count > current_retry_attempt and retry_notes is not None:
                user_prompt += "\\n\\n" + retry_notes

            self.stats.decision_calls += 1
            data = await self._generate_json("decision_agent", DECISION_PROMPT, user_prompt)

            if not data or not isinstance(data, dict):
                self.stats.decision_unusable += 1
                if engine_retry_count + retry_count + 1 < MAX_ATTEMPTS:
                    retry_count += 1
                    continue
                return hold_all()

            decisions_data = data.get("decisions", data)
            if not isinstance(decisions_data, dict):
                self.stats.decision_unusable += 1
                if engine_retry_count + retry_count + 1 < MAX_ATTEMPTS:
                    retry_count += 1
                    continue
                return hold_all()

            # A reply with no JSON arrives as {"raw_content": ...}: no symbols survive this
            # filter, so every stock holds below without a retry, as upstream does.
            decisions_data = {s: d for s, d in decisions_data.items() if s in symbols}
            if not any(isinstance(d, dict) for d in decisions_data.values()):
                self.stats.decision_unusable += 1

            logic_validation_failed = False
            invalid_decisions: list[dict[str, Any]] = []
            predicted_cash_usage = 0.0
            for symbol in symbols:
                symbol_decision = decisions_data.get(symbol)
                if isinstance(symbol_decision, dict):
                    try:
                        action = str(symbol_decision.get("action", "hold")).lower().strip()
                        if action == "hold" and "target_cash_amount" not in symbol_decision:
                            target_cash_amount = (
                                symbols[symbol]["features"].get("position_state", {}).get("current_position_value", 0.0)
                            )
                        else:
                            target_cash_amount = float(symbol_decision.get("target_cash_amount", 0.0))
                        current_position_value = (
                            symbols[symbol]["features"].get("position_state", {}).get("current_position_value", 0.0)
                        )
                        if not _validate_decision_logic(action, target_cash_amount, current_position_value):
                            logic_validation_failed = True
                            invalid_decisions.append(
                                {
                                    "symbol": symbol,
                                    "action": action,
                                    "target_cash_amount": target_cash_amount,
                                    "current_position_value": current_position_value,
                                }
                            )
                    except (ValueError, TypeError) as e:
                        logic_validation_failed = True
                        invalid_decisions.append({"symbol": symbol, "error": str(e)})
                    try:
                        cash_change = target_cash_amount - current_position_value  # type: ignore[operator]
                        if cash_change > 0:
                            predicted_cash_usage += cash_change
                    except Exception:
                        pass

            available_cash_after = available_cash - predicted_cash_usage
            predicted_remaining_ratio = available_cash_after / total_assets if total_assets > 0 else 0.0
            cash_shortage_detected = available_cash_after < 0
            cash_ratio_violation = predicted_remaining_ratio < min_cash_ratio

            if not logic_validation_failed and not cash_shortage_detected and not cash_ratio_violation:
                self.model_decided = 0
                for symbol, symbol_data in symbols.items():
                    position_value = (
                        symbol_data["features"].get("position_state", {}).get("current_position_value", 0.0)
                    )
                    symbol_decision = decisions_data.get(symbol)
                    if isinstance(symbol_decision, dict):
                        try:
                            action = str(symbol_decision.get("action", "hold")).lower()
                            if action == "hold" and "target_cash_amount" not in symbol_decision:
                                target = position_value
                            else:
                                target = float(symbol_decision.get("target_cash_amount", position_value))
                            reasons = symbol_decision.get("reasons", ["No specific reason"])
                            if not isinstance(reasons, list):
                                reasons = [str(reasons)]
                            confidence = float(symbol_decision.get("confidence", 0.5))
                            target = max(0.0, target)
                            confidence = max(0.0, min(1.0, confidence))
                            results[symbol] = round_numbers(
                                {
                                    "action": action,
                                    "target_cash_amount": target,
                                    "cash_change": target - position_value,
                                    "reasons": reasons,
                                    "confidence": confidence,
                                },
                                2,
                            )
                            self.model_decided += 1
                        except Exception as e:
                            self.stats.symbol_parse_errors += 1
                            results[symbol] = _hold(position_value, f"Dual-agent decision parsing error: {str(e)[:50]}")
                    else:
                        results[symbol] = _hold(position_value, "No decision provided by dual-agent")
                return results

            self.stats.decision_validation_failures += 1
            if engine_retry_count + retry_count + 1 < MAX_ATTEMPTS:
                notes = []
                if logic_validation_failed:
                    details = []
                    for invalid in invalid_decisions:
                        if "error" not in invalid:
                            if invalid["action"] == "increase":
                                details.append(
                                    f"{invalid['symbol']}: increase operation requires target_cash_amount > current_position_value, but you set {invalid['target_cash_amount']:.0f} <= {invalid['current_position_value']:.0f}"
                                )
                            elif invalid["action"] == "decrease":
                                details.append(
                                    f"{invalid['symbol']}: decrease operation requires target_cash_amount < current_position_value, but you set {invalid['target_cash_amount']:.0f} >= {invalid['current_position_value']:.0f}"
                                )
                            elif invalid["action"] == "close":
                                details.append(
                                    f"{invalid['symbol']}: close operation requires target_cash_amount = 0, but you set {invalid['target_cash_amount']:.0f}"
                                )
                        else:
                            details.append(f"{invalid['symbol']}: parsing error - {invalid['error']}")
                    notes.append(
                        "❌ DECISION LOGIC ERRORS: The following decisions have unreasonable logic:\\n"
                        + "\\n".join(details)
                        + "\\nPlease correct these logical inconsistencies."
                    )
                if cash_shortage_detected:
                    notes.append(
                        f"💰 INSUFFICIENT FUNDS: Total predicted cash usage {predicted_cash_usage:.2f} exceeds available cash {available_cash:.2f}. Please reduce purchase amounts or choose different stocks."
                    )
                if cash_ratio_violation:
                    notes.append(
                        f"⚖️ CASH RATIO VIOLATION: Predicted remaining cash ratio {predicted_remaining_ratio:.3f} is below minimum requirement {min_cash_ratio:.3f}. Please maintain higher cash reserves."
                    )
                retry_notes = "\\n\\n" + "\\n\\n".join(notes)
                retry_count += 1
                continue
            return hold_all()


class _ModelCallFailed(Exception):
    """Carries a model-call error past the catch-all in StockBenchAgent._decide."""


def _hold(position_value: float, reason: str) -> dict[str, Any]:
    return round_numbers(
        {
            "action": "hold",
            "target_cash_amount": position_value,
            "cash_change": 0.0,
            "reasons": [reason],
            "confidence": 0.5,
        },
        2,
    )


def _validate_decision_logic(action: str, target_cash_amount: Any, current_position_value: Any) -> bool:
    try:
        action = str(action).lower().strip()
        target = float(target_cash_amount)
        current = float(current_position_value)
        # Upstream rejects on these comparisons, so a NaN target passes. Keep them as written.
        if action == "increase":
            return not target <= current
        if action == "decrease":
            return not target >= current
        if action == "close":
            return not target > 0.01
        return True
    except Exception:
        return False
