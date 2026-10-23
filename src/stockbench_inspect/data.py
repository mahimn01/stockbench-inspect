"""StockBench market data, loaded from the pinned upstream commit.

The eval downloads one tarball of ChenYXxxx/stockbench at UPSTREAM_COMMIT and keeps
only the files listed in data/SHA256SUMS, checking each file's hash. The data is
never re-hosted here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from filelock import FileLock
from platformdirs import user_cache_dir

UPSTREAM_REPO = "ChenYXxxx/stockbench"
UPSTREAM_COMMIT = "ce8b2b3483590646ad3b650ac8221f43f76fd091"
TARBALL_URL = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{UPSTREAM_COMMIT}"
MANIFEST_PATH = Path(__file__).parent / "data" / "SHA256SUMS"

# Universe order matters: it is the order of symbols in every prompt and of every order.
SYMBOLS = [
    "GS", "MSFT", "HD", "V", "SHW", "CAT", "MCD", "UNH", "AXP", "AMGN",
    "TRV", "CRM", "JPM", "IBM", "HON", "BA", "AMZN", "AAPL", "PG", "JNJ",
]  # fmt: skip
START_DATE = "2025-03-03"
END_DATE = "2025-06-30"

# The earliest bar any prompt can see: 20 calendar days before the first trading day.
FIRST_BAR_DATE = "2025-02-11"
FIRST_NEWS_DATE = "2025-03-01"

# Upstream's feature windows, in calendar days before the decision day.
FILTER_PRICE_LOOKBACK_DAYS = 20  # strategy warmup_days 15 + 5 buffer
DECISION_PRICE_LOOKBACK_DAYS = 15  # strategy warmup_days
NEWS_LOOKBACK_DAYS = 2
NEWS_MAX_ITEMS = 5
PRICE_SERIES_DAYS = 7

INDICATOR_FIELDS = [
    "market_cap", "pe_ratio", "dividend_yield",
    "week_52_high", "week_52_low", "quarterly_dividend",
]  # fmt: skip


@dataclass(frozen=True)
class Bar:
    day: str
    open: float
    close: float | None


@dataclass(frozen=True)
class MarketData:
    symbols: list[str]
    trading_days: list[str]
    bars: dict[str, list[Bar]]
    news: dict[str, dict[str, list[dict[str, Any]]]]
    indicators: dict[tuple[str, str], dict[str, Any]]
    dividends: dict[str, list[dict[str, Any]]]
    # Upstream's older news cache (cache/news/news_<md5>.json), keyed by (symbol, day).
    query_news: dict[tuple[str, str], list[dict[str, Any]] | None]

    def open_price(self, symbol: str, day: str) -> float:
        for bar in self.bars[symbol]:
            if bar.day == day:
                return bar.open
        raise KeyError(f"No bar for {symbol} on {day}")

    def close_7d(self, symbol: str, day: str, lookback_days: int) -> list[float]:
        """Closes of the 7 rows before `day`, among bars dated within the lookback window.

        Mirrors upstream: it reads bars from day - lookback_days to day, drops the last
        row (the decision day), keeps the last 7, and pads with zeros at the front.
        """
        start = _shift(day, -lookback_days)
        rows = [bar for bar in self.bars[symbol] if start <= bar.day <= day]
        closes: list[float] = []
        if len(rows) > 1:
            history = rows[:-1]
            for bar in history[-PRICE_SERIES_DAYS:]:
                closes.append(0.0 if bar.close is None else float(bar.close))
        if len(closes) < PRICE_SERIES_DAYS:
            closes = [0.0] * (PRICE_SERIES_DAYS - len(closes)) + closes
        return closes

    def news_items(self, symbol: str, day: str) -> list[dict[str, Any]]:
        """The news shown for `symbol` on decision day `day`.

        Mirrors upstream's offline get_news: up to 5 items from the per-day cache for the
        two days before `day`, or its older per-query cache when that has 5 or more
        items, then the strategy's time filter from day - 2 at 00:00.

        Two upstream paths can show news from the decision day itself, after the open
        where the trades fill. The port closes both:
        - when both caches are empty, upstream widens the window to day - 4 to day + 1.
          The port shows no news instead.
        - upstream's time filter runs to the end of `day`, and the older cache holds two
          such items (IBM on 2025-03-04). The port keeps only news published before `day`.
        """
        day_cached = self._news_from_day_files(symbol, day)
        if len(day_cached) >= NEWS_MAX_ITEMS:
            selected = day_cached
        else:
            query_cached = self.query_news.get((symbol, day))
            if query_cached is not None and len(query_cached) >= NEWS_MAX_ITEMS:
                selected = query_cached
            elif day_cached:
                selected = day_cached
            elif query_cached is not None:
                selected = query_cached
            else:
                selected = []
        window_start = _midnight(_shift(day, -NEWS_LOOKBACK_DAYS))
        window_end = _midnight(day)
        kept = []
        for item in selected:
            if not isinstance(item, dict):
                continue
            published = _parse_published(item.get("published_utc") or item.get("published_date"))
            if published is not None and window_start <= published < window_end:
                kept.append(item)
        return kept

    def _news_from_day_files(self, symbol: str, day: str) -> list[dict[str, Any]]:
        """Upstream's _read_news_from_day_cache_range for day - 2 to day - 1, newest first."""
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for offset in range(1, NEWS_LOOKBACK_DAYS + 1):
            items = self.news[symbol].get(_shift(day, -offset))
            if items is None:
                continue
            valid: list[dict[str, Any]] = []
            for item in items:
                title = str(item.get("title", "")).strip()
                desc = str(item.get("description", "")).strip()
                if title or desc:
                    key = _news_unique_key(item)
                    if key and key not in seen:
                        seen.add(key)
                        valid.append(item)
            valid.sort(key=lambda x: str(x.get("published_utc", "")), reverse=True)
            collected.extend(valid)
            if len(collected) >= NEWS_MAX_ITEMS:
                break
        return collected[:NEWS_MAX_ITEMS]

    def indicators_for(self, symbol: str, day: str) -> dict[str, Any] | None:
        return self.indicators.get((symbol, day))

    def dividend_per_share(self, symbol: str, day: str) -> list[float]:
        """Cash dividends per share with an ex-dividend date of `day`, one per record."""
        return [
            float(row["cash_amount"]) for row in self.dividends[symbol] if str(row.get("ex_dividend_date"))[:10] == day
        ]


def calendar_days(start: str, end: str) -> list[str]:
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    return [(first + timedelta(days=i)).isoformat() for i in range((last - first).days + 1)]


def _shift(day: str, days: int) -> str:
    return (date.fromisoformat(day) + timedelta(days=days)).isoformat()


def _midnight(day: str) -> datetime:
    return datetime.fromisoformat(day)


def _parse_published(value: Any) -> datetime | None:
    """Parse a news timestamp the way upstream's pd.to_datetime(utc=True) does, as naive UTC."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def news_query_key(symbol: str, day: str) -> str:
    """Upstream's _generate_news_cache_key for the strategy's query on decision day `day`.

    The strategy asks for news from day - 2 to day, with get_news defaults of 5 items and
    a 3-day lookback (it passes no config).
    """
    cache_string = f"{symbol}|{_shift(day, -NEWS_LOOKBACK_DAYS)}|{day}|{NEWS_MAX_ITEMS}|3"
    return f"news_{hashlib.md5(cache_string.encode('utf-8')).hexdigest()}"


def _news_unique_key(item: dict[str, Any]) -> str:
    item_id = str(item.get("id") or "").strip()
    if item_id:
        return f"id:{item_id}"
    url = str(item.get("url") or "").strip()
    if url:
        return f"url:{url}"
    title = str(item.get("title") or "").strip()
    ts = str(item.get("published_utc") or "").strip()
    return f"tt:{title}|{ts}"


def read_manifest() -> dict[str, str]:
    """Map each upstream path under storage/ to its expected SHA-256."""
    manifest: dict[str, str] = {}
    for line in MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.split(maxsplit=1)
            manifest[path] = digest
    return manifest


def cache_dir() -> Path:
    configured = os.environ.get("STOCKBENCH_INSPECT_CACHE_DIR")
    base = Path(configured).expanduser() if configured else Path(user_cache_dir("stockbench_inspect"))
    return base / UPSTREAM_COMMIT


def ensure_data() -> Path:
    """Download and verify the pinned upstream data once, returning its local root.

    A file lock makes concurrent first runs wait for one download instead of racing.
    """
    root = cache_dir()
    root.parent.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest()
    manifest_digest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    with FileLock(str(root.parent / f"{UPSTREAM_COMMIT}.lock")):
        if _is_complete(root, manifest, manifest_digest):
            return root
        staging = Path(tempfile.mkdtemp(prefix="staging-", dir=root.parent))
        try:
            with tempfile.TemporaryFile() as tarball:
                with urllib.request.urlopen(TARBALL_URL, timeout=120) as response:
                    shutil.copyfileobj(response, tarball)
                tarball.seek(0)
                _extract_manifest_files(tarball, manifest, staging)
            (staging / ".verified").write_text(manifest_digest, encoding="utf-8")
            if root.exists():
                shutil.rmtree(root)
            staging.rename(root)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return root


def _is_complete(root: Path, manifest: dict[str, str], manifest_digest: str) -> bool:
    """True if a previous download of this manifest finished and none of its files are missing."""
    marker = root / ".verified"
    if not marker.is_file() or marker.read_text(encoding="utf-8") != manifest_digest:
        return False
    return all((root / path).is_file() for path in manifest)


def _extract_manifest_files(tarball: Any, manifest: dict[str, str], dest: Path) -> None:
    prefix = f"stockbench-{UPSTREAM_COMMIT}/storage/"
    found: set[str] = set()
    with tarfile.open(fileobj=tarball, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(prefix):
                continue
            rel = member.name[len(prefix) :]
            expected = manifest.get(rel)
            if expected is None:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Could not read {rel} from the upstream tarball")
            content = extracted.read()
            actual = hashlib.sha256(content).hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA-256 mismatch for {rel}: expected {expected}, got {actual}")
            # Manifest paths are fixed relative paths, so writing them cannot escape dest.
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            found.add(rel)
    missing = sorted(set(manifest) - found)
    if missing:
        raise RuntimeError(f"{len(missing)} manifest files missing from the upstream tarball, e.g. {missing[:3]}")


@cache
def load_market_data() -> MarketData:
    root = ensure_data()
    manifest = read_manifest()
    bars: dict[str, list[Bar]] = {}
    news: dict[str, dict[str, list[dict[str, Any]]]] = {}
    indicators: dict[tuple[str, str], dict[str, Any]] = {}
    dividends: dict[str, list[dict[str, Any]]] = {}

    for symbol in SYMBOLS:
        bar_paths = sorted(p for p in manifest if p.startswith(f"parquet/{symbol}/day/"))
        bars[symbol] = [_read_bar(root / p) for p in bar_paths]
        news[symbol] = {}
        for path in manifest:
            if path.startswith(f"cache/news_by_day/{symbol}/"):
                day = path.rsplit("/", 1)[1].removesuffix(".json")
                payload = json.loads((root / path).read_text(encoding="utf-8"))
                news[symbol][day] = payload.get("items", []) if isinstance(payload, dict) else payload
        dividends[symbol] = json.loads(
            (root / f"cache/corporate_actions/{symbol}.dividends.json").read_text(encoding="utf-8")
        )
    query_files = {
        path.rsplit("/", 1)[1].removesuffix(".json"): path for path in manifest if path.startswith("cache/news/")
    }
    query_news: dict[tuple[str, str], list[dict[str, Any]] | None] = {}
    for symbol in SYMBOLS:
        for day in calendar_days(START_DATE, END_DATE):
            path = query_files.get(news_query_key(symbol, day))
            if path is not None:
                payload = json.loads((root / path).read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    query_news[(symbol, day)] = payload
                elif isinstance(payload, dict) and "items" in payload:
                    query_news[(symbol, day)] = payload["items"]
                else:
                    query_news[(symbol, day)] = None
    for path in manifest:
        if path.startswith("cache/stock_indicators/"):
            symbol, day = path.rsplit("/", 1)[1].removesuffix(".json").split("_", 1)
            indicators[(symbol, day)] = json.loads((root / path).read_text(encoding="utf-8"))

    trading_days = sorted(
        {bar.day for bar in bars[SYMBOLS[0]]}.intersection(*({b.day for b in bars[s]} for s in SYMBOLS[1:]))
    )
    trading_days = [d for d in trading_days if START_DATE <= d <= END_DATE]
    return MarketData(SYMBOLS, trading_days, bars, news, indicators, dividends, query_news)


def _read_bar(path: Path) -> Bar:
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 1:
        raise RuntimeError(f"Expected one row in {path}, found {len(rows)}")
    row = rows[0]
    day = path.name.removesuffix(".parquet")
    if str(row["date"])[:10] != day:
        raise RuntimeError(f"{path} holds a bar dated {row['date']}")
    close = row.get("close")
    return Bar(day=day, open=float(row["open"]), close=None if close is None else float(close))
