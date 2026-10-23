"""Run this port's engine on scripted model replies, for the parity check.

Usage: uv run python scripts/parity/run_port.py SCRIPT OUT.json

Uses the same scripted replies as run_upstream.py and writes every prompt, the daily NAV
and the trades to OUT.json. Read-only apart from that file and the data cache.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from replies import reply

from stockbench_inspect.agent import StockBenchAgent
from stockbench_inspect.data import load_market_data
from stockbench_inspect.engine import run_backtest


async def run(script: str) -> dict:
    """Run the full backtest on the scripted replies and return prompts, NAV and trades."""
    calls: list[dict] = []

    async def llm(role: str, system_prompt: str, user_prompt: str) -> str:
        content = reply(system_prompt, user_prompt, script)
        calls.append({"role": role, "system": system_prompt, "user": user_prompt, "reply": content})
        return content

    data = load_market_data()
    agent = StockBenchAgent(data, llm)
    result = await run_backtest(data, agent)
    return {
        "engine": "port",
        "script": script,
        "calls": calls,
        "days": result.days,
        "nav": result.nav,
        "trades": [{"day": t.day, "symbol": t.symbol, "side": t.side, "qty": t.qty} for t in result.trades],
        "stats": vars(agent.stats),
    }


async def main(script: str, out_path: Path) -> None:
    out = await run(script)
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(
        f"port {script}: {len(out['calls'])} calls, {len(out['days'])} days, {len(out['trades'])} trades, {out['stats']}"
    )


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], Path(sys.argv[2])))
