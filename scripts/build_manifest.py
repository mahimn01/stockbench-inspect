"""Write src/stockbench_inspect/data/SHA256SUMS from the pinned upstream tarball.

Lists every upstream file the eval reads (prices, day-level news, the older per-query
news cache, indicator snapshots and dividends for the 20 symbols) with its SHA-256.
Run it only when the pinned commit changes:

    uv run python scripts/build_manifest.py [path/to/tarball.tar.gz]

Without a path it downloads the tarball. It only writes the manifest file.
"""

from __future__ import annotations

import hashlib
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from stockbench_inspect.data import (
    END_DATE,
    FIRST_BAR_DATE,
    FIRST_NEWS_DATE,
    MANIFEST_PATH,
    START_DATE,
    SYMBOLS,
    TARBALL_URL,
    UPSTREAM_COMMIT,
    calendar_days,
    news_query_key,
)

QUERY_NEWS_FILES = {f"{news_query_key(s, d)}.json" for s in SYMBOLS for d in calendar_days(START_DATE, END_DATE)}


def wanted(rel: str) -> bool:
    parts = rel.split("/")
    if rel.startswith("parquet/") and len(parts) == 4 and parts[2] == "day":
        return parts[1] in SYMBOLS and rel.endswith(".parquet") and FIRST_BAR_DATE <= parts[3][:10] <= END_DATE
    if rel.startswith("cache/news_by_day/") and len(parts) == 4:
        return parts[2] in SYMBOLS and rel.endswith(".json") and FIRST_NEWS_DATE <= parts[3][:10] < END_DATE
    if rel.startswith("cache/stock_indicators/") and len(parts) == 3:
        symbol, _, day = parts[2].removesuffix(".json").partition("_")
        return symbol in SYMBOLS and START_DATE <= day <= END_DATE
    if rel.startswith("cache/news/") and len(parts) == 3:
        return parts[2] in QUERY_NEWS_FILES
    if rel.startswith("cache/corporate_actions/") and len(parts) == 3:
        return parts[2] in {f"{s}.dividends.json" for s in SYMBOLS}
    return False


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        if len(sys.argv) > 1:
            tarball = Path(sys.argv[1])
        else:
            tarball = Path(tmp) / "upstream.tar.gz"
            urllib.request.urlretrieve(TARBALL_URL, tarball)
        prefix = f"stockbench-{UPSTREAM_COMMIT}/storage/"
        lines = []
        with tarfile.open(tarball, mode="r:gz") as archive:
            for member in archive:
                if member.isfile() and member.name.startswith(prefix):
                    rel = member.name[len(prefix) :]
                    if wanted(rel):
                        content = archive.extractfile(member).read()  # type: ignore[union-attr]
                        lines.append(f"{hashlib.sha256(content).hexdigest()}  {rel}")
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text("\n".join(sorted(lines, key=lambda line: line.split("  ", 1)[1])) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
