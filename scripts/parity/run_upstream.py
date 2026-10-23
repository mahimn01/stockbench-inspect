"""Run the original StockBench engine on scripted model replies, for the parity check.

Usage (from an extracted copy of upstream at the pinned commit, in its own venv with
upstream's requirements and json-repair==0.51.0, with no API keys set):

    python scripts/parity/run_upstream.py UPSTREAM_DIR SCRIPT OUT.json

The upstream code is not modified. In memory, this script:
- replaces the model call with scripts/parity/replies.py (no network);
- uses the data-defined trading calendar instead of upstream's holiday rules, which
  add 3 non-trading days (the port's calendar deviation);
- returns no news where upstream's offline fallback would widen the news window to
  T-4..T+1, and drops news published on or after the decision day (the port's two news
  deviations), counting how often each happens;
- counts hits on upstream's older per-query news cache, which the port also reads.
It writes every prompt, the daily NAV and the trades to OUT.json.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

upstream_dir = Path(sys.argv[1]).resolve()
script = sys.argv[2]
out_path = Path(sys.argv[3]).resolve()
sys.path.insert(0, str(upstream_dir))
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(upstream_dir)
for key in ("FINNHUB_API_KEY", "POLYGON_API_KEY", "LLM_API_KEY"):
    os.environ.pop(key, None)
os.environ["OPENAI_API_KEY"] = "parity-stub"

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from loguru import logger  # noqa: E402
from replies import reply  # noqa: E402

logger.remove()
logging.disable(logging.CRITICAL)

from stockbench.backtest.datasets import Datasets  # noqa: E402
from stockbench.backtest.engine import BacktestEngine  # noqa: E402
from stockbench.backtest.slippage import Slippage  # noqa: E402
from stockbench.backtest.strategies.llm_decision import Strategy  # noqa: E402
from stockbench.core import data_hub  # noqa: E402
from stockbench.llm import llm_client  # noqa: E402

data_hub.set_data_mode("offline_only")
config = yaml.safe_load((upstream_dir / "config.yaml").read_text(encoding="utf-8"))
symbols = list(config["symbols_universe"])

# Trading calendar: dates with a price file for every symbol.
dates_per_symbol = [{p.stem for p in (upstream_dir / "storage/parquet" / s / "day").glob("*.parquet")} for s in symbols]
calendar = sorted(set.intersection(*dates_per_symbol))
data_hub.is_trading_day = lambda d: pd.Timestamp(d).strftime("%Y-%m-%d") in calendar


def _next_trading_day(d: pd.Timestamp, max_search_days: int = 10) -> pd.Timestamp:
    later = [c for c in calendar if c > pd.Timestamp(d).strftime("%Y-%m-%d")]
    return pd.Timestamp(later[0]) if later else pd.Timestamp(d) + pd.Timedelta(days=1)


data_hub.get_next_trading_day = _next_trading_day

counts = {"extended_news_fallback": 0, "hash_news_cache_hits": 0, "same_day_news_removed": 0}
_read_range = data_hub._read_news_from_day_cache_range
_hash_cache = data_hub._get_news_from_cache


def _read_range_without_fallback(ticker: str, start: str, end: str, max_items: int) -> list:
    # get_news only asks for 2 x top_k (10) items on its extended offline fallback.
    if max_items == 10:
        counts["extended_news_fallback"] += 1
        return []
    return _read_range(ticker, start, end, max_items)


def _counted_hash_cache(cache_key: str, cfg: dict | None = None) -> list | None:
    result = _hash_cache(cache_key, cfg)
    if result is not None:
        counts["hash_news_cache_hits"] += 1
    return result


_get_news = data_hub.get_news


def _news_before_decision_day(ticker: str, gte: str, lte: str, *args, **kwargs) -> tuple:
    # The strategy passes the decision day as lte. Drop items published on or after it.
    items, token = _get_news(ticker, gte, lte, *args, **kwargs)
    cutoff = pd.Timestamp(lte).normalize()
    kept = []
    for item in items:
        published = pd.to_datetime(item.get("published_utc") or item.get("published_date"), utc=True, errors="coerce")
        if pd.isna(published) or published.tz_localize(None) < cutoff:
            kept.append(item)
    counts["same_day_news_removed"] += len(items) - len(kept)
    return kept, token


data_hub._read_news_from_day_cache_range = _read_range_without_fallback
data_hub._get_news_from_cache = _counted_hash_cache
data_hub.get_news = _news_before_decision_day

calls: list[dict] = []


def _scripted_call(self, cfg, system_prompt: str, user_prompt: str) -> dict:
    content = reply(system_prompt, user_prompt, script)
    role = "fundamental_filter" if "<FILTER_RESULT>" in system_prompt else "decision_agent"
    calls.append({"role": role, "system": system_prompt, "user": user_prompt, "reply": content})
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {"status_code": 200, "data": {"choices": [{"message": {"content": content}}], "usage": usage}}


llm_client.LLMClient._call_openai_official = _scripted_call

config["llm"] = dict(config["llm_profiles"]["openai"])
config["cache"]["mode"] = "off"
config["backtest"]["summary_llm"] = False
config["backtest"]["enable_detailed_logging"] = False
engine = BacktestEngine(config, Datasets(config), Slippage.from_cfg(config))
result = engine.run(strategy=Strategy(config), start="2025-03-03", end="2025-06-30", symbols=symbols, timespan="day")

trades = (
    [
        {
            "day": pd.Timestamp(row["ts"]).strftime("%Y-%m-%d"),
            "symbol": row["symbol"],
            "side": row["side"],
            "qty": float(row["qty"]),
        }
        for _, row in result["trades"].iterrows()
    ]
    if len(result["trades"])
    else []
)
out = {
    "engine": "upstream",
    "script": script,
    "calls": calls,
    "days": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in result["nav"].index],
    "nav": [float(v) for v in result["nav"].values],
    "trades": trades,
    "counts": counts,
}
out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"upstream {script}: {len(calls)} calls, {len(out['days'])} days, {len(trades)} trades, {counts}")
