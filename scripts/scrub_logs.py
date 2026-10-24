"""Remove local file paths and the user name from Inspect logs before sharing them.

Usage: uv run python scripts/scrub_logs.py LOG.eval [LOG.eval ...] --out OUT_DIR

Writes a scrubbed copy of each log to OUT_DIR (same file name). The home directory
becomes "~", any other path segment equal to the user name becomes "user", and the name of
an external volume ("/Volumes/NAME") becomes "volume". It then counts what is left of the
user name and of /Users/ and /Volumes/ paths, so they can be checked by hand.
The original logs are not changed.
"""

from __future__ import annotations

import argparse
import getpass
import re
from pathlib import Path

from inspect_ai.log import EvalLog, read_eval_log, write_eval_log


def scrub(log: EvalLog) -> EvalLog:
    text = log.model_dump_json()
    home = str(Path.home())
    user = getpass.getuser()
    text = text.replace(home, "~").replace(f"/{user}/", "/user/")
    text = re.sub(r"/Volumes/[^/\\\s\"]+", "/Volumes/volume", text)
    return EvalLog.model_validate_json(text)


def leftovers(log: EvalLog) -> int:
    """Mentions of the user name, or of /Users/ and /Volumes/ paths, still in the log."""
    text = log.model_dump_json()
    return text.count(getpass.getuser()) + len(re.findall(r"/Users/|/Volumes/(?!volume\b)", text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for path in args.logs:
        scrubbed = scrub(read_eval_log(str(path)))
        target = args.out / path.name
        write_eval_log(scrubbed, str(target))
        leftover = leftovers(read_eval_log(str(target)))
        print(
            f"{path.name} -> {target} ({'clean' if leftover == 0 else f'{leftover} local paths or user name mentions left'})"
        )


if __name__ == "__main__":
    main()
