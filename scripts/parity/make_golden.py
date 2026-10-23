"""Record upstream parity runs as tests/golden/upstream_parity.json.

Usage: python scripts/parity/make_golden.py UPSTREAM-equal.json UPSTREAM-random.json UPSTREAM-adversarial.json

Stores, for each script, the number of model calls, a SHA-256 over every prompt, the
daily NAV and a SHA-256 over the trades, all taken from run_upstream.py output. The
test suite replays the same scripts through the port and checks them against this file,
so parity can be re-checked without installing upstream. Writes only that file.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

GOLDEN_PATH = Path(__file__).parents[2] / "tests" / "golden" / "upstream_parity.json"


def fingerprint(run: dict) -> dict:
    prompts = [[c["role"], c["system"], c["user"]] for c in run["calls"]]
    return {
        "calls": len(run["calls"]),
        "prompts_sha256": hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest(),
        "days": run["days"],
        "nav": run["nav"],
        "trades": len(run["trades"]),
        "trades_sha256": hashlib.sha256(json.dumps(run["trades"]).encode()).hexdigest(),
    }


def main(paths: list[str]) -> None:
    golden: dict = {
        "source": "scripts/parity/run_upstream.py at upstream ce8b2b3483590646ad3b650ac8221f43f76fd091",
        "scripts": {},
    }
    for path in paths:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        assert run["engine"] == "upstream", f"{path} is not an upstream run"
        golden["scripts"][run["script"]] = fingerprint(run) | {"upstream_counts": run["counts"]}
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(golden, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {GOLDEN_PATH} for {sorted(golden['scripts'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
