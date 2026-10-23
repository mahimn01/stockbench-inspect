import hashlib
import io
import tarfile
from datetime import datetime, timedelta

import pytest

from stockbench_inspect.data import UPSTREAM_COMMIT, _extract_manifest_files, _parse_published, news_query_key
from stockbench_inspect.metrics import baseline_metrics


def test_news_query_key_matches_upstream() -> None:
    # Value from upstream's _generate_news_cache_key("AAPL", "2025-03-02", "2025-03-04", 5, 3).
    assert news_query_key("AAPL", "2025-03-04") == "news_3c3e9cc8377724cb0a1286546dff78d0"


def test_tampered_upstream_file_is_rejected(tmp_path) -> None:
    content = b"real bytes"
    manifest = {"cache/x.json": hashlib.sha256(content).hexdigest()}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo(f"stockbench-{UPSTREAM_COMMIT}/storage/cache/x.json")
        tampered = b"fake bytes"
        info.size = len(tampered)
        archive.addfile(info, io.BytesIO(tampered))
    buffer.seek(0)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        _extract_manifest_files(buffer, manifest, tmp_path)


@pytest.mark.data
def test_calendar_is_the_83_days_with_prices(market_data) -> None:
    days = market_data.trading_days
    assert (len(days), days[0], days[-1]) == (83, "2025-03-03", "2025-06-30")
    assert not {"2025-04-18", "2025-05-26", "2025-06-19"} & set(days)


@pytest.mark.data
def test_baseline_reproduces_the_paper(market_data) -> None:
    # Paper Table 2, passive baseline: 0.4% return, -15.2% max drawdown, Sortino 0.0155.
    baseline = baseline_metrics(market_data, market_data.trading_days)
    assert round(baseline["final_return"] * 100, 1) == 0.4
    assert round(baseline["max_drawdown"] * 100, 1) == -15.2
    assert round(baseline["sortino"], 4) == 0.0155


@pytest.mark.data
def test_prompt_news_was_published_before_the_decision_day(market_data) -> None:
    for symbol in market_data.symbols:
        for day in market_data.trading_days:
            start = datetime.fromisoformat(day) - timedelta(days=2)
            for item in market_data.news_items(symbol, day):
                published = _parse_published(item.get("published_utc") or item.get("published_date"))
                assert published is not None and start <= published < datetime.fromisoformat(day)


def test_cache_with_a_missing_file_is_not_trusted(tmp_path) -> None:
    from stockbench_inspect.data import _is_complete

    manifest = {"a.json": "x", "b/c.json": "y"}
    (tmp_path / ".verified").write_text("digest", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    assert not _is_complete(tmp_path, manifest, "digest")
    (tmp_path / "b").mkdir()
    (tmp_path / "b/c.json").write_text("{}", encoding="utf-8")
    assert _is_complete(tmp_path, manifest, "digest")
    assert not _is_complete(tmp_path, manifest, "other-digest")


@pytest.mark.data
def test_stock_days_where_upstream_widens_the_news_window(market_data) -> None:
    # Upstream's offline fallback (day - 4 to day + 1) fires when both news caches are empty.
    empty = [
        (s, d)
        for s in market_data.symbols
        for d in market_data.trading_days
        if not market_data._news_from_day_files(s, d) and (s, d) not in market_data.query_news
    ]
    assert len(empty) == 62
