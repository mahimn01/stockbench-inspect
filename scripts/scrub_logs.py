"""Remove local file paths and the user name from Inspect logs before sharing them.

Usage: uv run python scripts/scrub_logs.py LOG.eval [LOG.eval ...] --out OUT_DIR [--redact-news]

Writes a scrubbed copy of each log to OUT_DIR (same file name). The home directory
becomes "~", any other path segment equal to the user name becomes "user", and the name of
an external volume ("/Volumes/NAME") becomes "volume". It then counts what is left of the
user name and of /Users/ and /Volumes/ paths, so they can be checked by hand.

With --redact-news, the news headlines and summaries are also replaced by a placeholder.
The prompts of a run hold them in the "top_k_events" lists, and a model can quote them in its
replies and reasoning. The script replaces every item of every list, then replaces any other
exact occurrence of those items, and of the title and description of every news item in the
pinned StockBench data, anywhere in the log. Finally it replaces any run of --min-words (default 6)
words in a row that is found in a news item, which catches a model quoting part of a headline.
A model that paraphrases a headline is not caught.
Scores, portfolio values, decisions and the rest of each prompt are not changed. The script
prints how many known news strings are left in each copy, which should be 0. Pass
--no-dataset to skip the pinned data (no download) and use only the items found in the prompts.

The original logs are not changed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, read_eval_log, write_eval_log

NEWS_PLACEHOLDER = "[news text redacted]"
NO_NEWS = "No news available"
EVENTS_KEY = '"top_k_events": '
# Shorter strings are too likely to match ordinary text, so they are only redacted inside the lists.
MIN_NEWS_CHARS = 12
# A model often quotes only part of a headline, so a run of this many words in a row that appears in a
# news item is redacted too. Shorter runs are too likely to be ordinary text.
MIN_NEWS_WORDS = 6
WORD = re.compile(r"\S+")
WORD_EDGES = " \t\r\n\\\"'.,;:!?()[]{}<>*_`\u2018\u2019\u201c\u201d"


def scrub(log: EvalLog) -> EvalLog:
    text = log.model_dump_json()
    home = str(Path.home())
    user = getpass.getuser()
    text = text.replace(home, "~").replace(f"/{user}/", "/user/")
    text = re.sub(r"/Volumes/[^/\\\s\"]+", "/Volumes/volume", text)
    return EvalLog.model_validate_json(text)


@dataclass
class NewsRedaction:
    lists: int = 0  # top_k_events lists rewritten
    items: int = 0  # news items replaced inside those lists
    elsewhere: int = 0  # exact occurrences replaced outside the lists
    partial: int = 0  # pieces of headlines (runs of words) replaced outside the lists
    unparsed: int = 0  # lists that could not be parsed and so were left alone
    found: set[str] = field(default_factory=set)


def news_from_data(data: Any) -> set[str]:
    """Title and description of every news item in a MarketData (both caches)."""
    items = [i for days in data.news.values() for day_items in days.values() for i in day_items]
    items += [i for cached in data.query_news.values() for i in cached or []]
    texts: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            texts.update(str(item.get(key) or "").strip() for key in ("title", "description"))
    return {text for text in texts if len(text) >= MIN_NEWS_CHARS}


def _strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _map_strings(node: Any, fn: Any) -> Any:
    if isinstance(node, str):
        return fn(node)
    if isinstance(node, dict):
        return {key: _map_strings(value, fn) for key, value in node.items()}
    if isinstance(node, list):
        return [_map_strings(value, fn) for value in node]
    return node


def _words(text: str) -> list[tuple[str, int, int]]:
    """Each whitespace-separated word as (lowercase word without edge punctuation, start, end) in `text`."""
    words = []
    for match in WORD.finditer(text):
        raw = match.group()
        core = raw.strip(WORD_EDGES)
        if core:
            start = match.start() + raw.index(core[0])
            words.append((core.casefold(), start, start + len(core)))
        else:
            words.append((raw, match.start(), match.end()))
    return words


def _word_runs(known: Iterable[str], size: int) -> set[tuple[str, ...]]:
    """Every run of `size` words in a row found in any of the `known` news strings."""
    runs: set[tuple[str, ...]] = set()
    for news in known:
        words = [word for word, _, _ in _words(news)]
        runs.update(tuple(words[i : i + size]) for i in range(len(words) - size + 1))
    return runs


def _quoted_spans(text: str, runs: set[tuple[str, ...]], size: int) -> list[tuple[int, int]]:
    """Character spans of `text` made of words that sit in a run of `size` words from the news.

    A span never crosses a line break, so the quote marks and line breaks around a quote stay.
    """
    words = _words(text)
    covered = [False] * len(words)
    for i in range(len(words) - size + 1):
        if tuple(word for word, _, _ in words[i : i + size]) in runs:
            covered[i : i + size] = [True] * size
    spans: list[tuple[int, int]] = []
    for i, (_, start, end) in enumerate(words):
        if not covered[i]:
            continue
        if i > 0 and covered[i - 1] and "\n" not in text[words[i - 1][2] : start]:
            spans[-1] = (spans[-1][0], end)
        else:
            spans.append((start, end))
    return spans


def _event_lists(text: str) -> Iterable[tuple[int, int, list[Any] | None]]:
    """Position and parsed value of each top_k_events list in `text` (None if it does not parse)."""
    decoder = json.JSONDecoder()
    start = text.find(EVENTS_KEY)
    while start != -1:
        begin = start + len(EVENTS_KEY)
        try:
            value, end = decoder.raw_decode(text, begin)
        except ValueError:
            value, end = None, begin
        # A model may write the key with "[...]" as the value. That holds no news, so it is skipped.
        if isinstance(value, list) or text.startswith('["', begin):
            yield begin, end, value if isinstance(value, list) else None
        start = text.find(EVENTS_KEY, end)


def redact_news(
    log: EvalLog, known: Iterable[str] = (), min_words: int = MIN_NEWS_WORDS
) -> tuple[EvalLog, NewsRedaction]:
    """Copy of `log` with news text replaced by a placeholder, see the module docstring."""
    tree = json.loads(log.model_dump_json())
    stats = NewsRedaction()
    for text in _strings(tree):
        for _, _, value in _event_lists(text):
            stats.found.update(item for item in value or [] if isinstance(item, str) and item != NO_NEWS)
    needles = sorted(
        {n for n in set(known) | stats.found if len(n) >= MIN_NEWS_CHARS and n != NEWS_PLACEHOLDER},
        key=len,
        reverse=True,
    )
    runs = _word_runs(needles, min_words) if min_words > 0 else set()

    memo: dict[str, tuple[str, NewsRedaction]] = {}

    def redact(text: str) -> str:
        # Logs repeat many strings (for example the same prompt in several events), so each is done once
        # and its counts are added again for every repeat.
        if text not in memo:
            once = NewsRedaction()
            memo[text] = (redact_text(text, once), once)
        result, once = memo[text]
        stats.lists += once.lists
        stats.items += once.items
        stats.elsewhere += once.elsewhere
        stats.partial += once.partial
        stats.unparsed += once.unparsed
        return result

    def redact_text(text: str, stats: NewsRedaction) -> str:
        if EVENTS_KEY in text:
            pieces, last = [], 0
            for begin, end, value in _event_lists(text):
                if value is None:
                    stats.unparsed += 1
                    continue
                kept = [item if item == NO_NEWS else NEWS_PLACEHOLDER for item in value]
                stats.lists += 1
                stats.items += sum(item != NO_NEWS for item in value)
                pieces += [text[last:begin], json.dumps(kept, ensure_ascii=False)]
                last = end
            text = "".join(pieces) + text[last:]
        for needle in needles:
            if needle in text:
                stats.elsewhere += text.count(needle)
                text = text.replace(needle, NEWS_PLACEHOLDER)
        if runs:
            spans = _quoted_spans(text, runs, min_words)
            stats.partial += len(spans)
            for start, end in reversed(spans):
                text = text[:start] + NEWS_PLACEHOLDER + text[end:]
        return text

    return EvalLog.model_validate(_map_strings(tree, redact)), stats


def news_leftovers(log: EvalLog, known: Iterable[str], min_words: int = MIN_NEWS_WORDS) -> int:
    """Occurrences of the `known` news strings in the log, plus news-list items that are not placeholders,
    plus lists that cannot be parsed, plus runs of `min_words` words from the news (0 turns this off).

    A string counts when it appears as it is or JSON-escaped (for example inside a JSON string).
    """
    needles = {n for n in known if len(n) >= MIN_NEWS_CHARS}
    needles |= {json.dumps(n, ensure_ascii=ensure)[1:-1] for n in needles for ensure in (False, True)}
    runs = _word_runs(known, min_words) if min_words > 0 else set()
    count = 0
    for text, times in Counter(_strings(json.loads(log.model_dump_json()))).items():
        found = sum(text.count(n) for n in needles)
        found += len(_quoted_spans(text, runs, min_words)) if runs else 0
        for _, _, value in _event_lists(text):
            # A list that does not parse counts as one leftover, since its items cannot be checked.
            found += 1 if value is None else sum(item not in (NEWS_PLACEHOLDER, NO_NEWS) for item in value)
        count += found * times
    return count


def leftovers(log: EvalLog) -> int:
    """Mentions of the user name, or of /Users/ and /Volumes/ paths, still in the log."""
    text = log.model_dump_json()
    return text.count(getpass.getuser()) + len(re.findall(r"/Users/|/Volumes/(?!volume\b)", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--redact-news", action="store_true", help="also replace news headlines and summaries")
    parser.add_argument("--no-dataset", action="store_true", help="with --redact-news, do not load the pinned data")
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_NEWS_WORDS,
        help="with --redact-news, also replace any run of this many words in a row found in a news item, 0 to skip",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset: set[str] = set()
    if args.redact_news and not args.no_dataset:
        from stockbench_inspect.data import load_market_data

        dataset = news_from_data(load_market_data())
    for path in args.logs:
        scrubbed = scrub(read_eval_log(str(path)))
        news_note = ""
        if args.redact_news:
            scrubbed, stats = redact_news(scrubbed, dataset, args.min_words)
        target = args.out / path.name
        write_eval_log(scrubbed, str(target))
        written = read_eval_log(str(target))
        leftover = leftovers(written)
        if args.redact_news:
            news_left = news_leftovers(written, dataset | stats.found, args.min_words)
            news_note = (
                f", news: {stats.items} list items, {stats.elsewhere} other mentions and {stats.partial} partial quotes "
                "replaced, "
                f"{'clean' if news_left == 0 else f'{news_left} left'}"
            )
        print(
            f"{path.name} -> {target} ({'clean' if leftover == 0 else f'{leftover} local paths or user name mentions left'}{news_note})"
        )


if __name__ == "__main__":
    main()
