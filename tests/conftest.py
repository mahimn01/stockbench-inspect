from __future__ import annotations

import pytest

from stockbench_inspect.data import Bar, MarketData

DAYS = ["2025-03-03", "2025-03-04", "2025-03-05", "2025-03-06"]


def make_data(opens: dict[str, list[float]], dividends: dict[str, list[dict]] | None = None) -> MarketData:
    """Small offline market: one bar per day per symbol, close = open + 1, no news."""
    bars = {
        symbol: [Bar(day=day, open=price, close=price + 1.0) for day, price in zip(DAYS, prices, strict=True)]
        for symbol, prices in opens.items()
    }
    return MarketData(
        symbols=list(opens),
        trading_days=list(DAYS),
        bars=bars,
        news={symbol: {} for symbol in opens},
        indicators={},
        dividends={symbol: (dividends or {}).get(symbol, []) for symbol in opens},
        query_news={},
    )


@pytest.fixture(scope="session")
def market_data() -> MarketData:
    """The real upstream data (downloaded and checked once, about 6 MB)."""
    from stockbench_inspect.data import load_market_data

    return load_market_data()


@pytest.fixture
def two_stock_data() -> MarketData:
    return make_data({"AAA": [100.0, 110.0, 90.0, 100.0], "BBB": [50.0, 50.0, 55.0, 60.0]})
