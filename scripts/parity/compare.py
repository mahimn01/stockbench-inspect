"""Compare a port run with an upstream run of the same parity script.

Usage: python scripts/parity/compare.py UPSTREAM.json PORT.json

Checks that both engines made the same model calls with byte-identical prompts, got the
same trades, and ended every day at the same NAV. Exits 1 on any difference. Read-only.
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path


def main(upstream_path: str, port_path: str) -> int:
    upstream = json.loads(Path(upstream_path).read_text(encoding="utf-8"))
    port = json.loads(Path(port_path).read_text(encoding="utf-8"))
    problems = 0

    up_calls, port_calls = upstream["calls"], port["calls"]
    if len(up_calls) != len(port_calls):
        print(f"call count differs: upstream {len(up_calls)}, port {len(port_calls)}")
        problems += 1
    for i, (a, b) in enumerate(zip(up_calls, port_calls, strict=False)):
        for field in ("role", "system", "user"):
            if a[field] != b[field]:
                print(f"call {i} ({a['role']}) differs in {field}:")
                diff = difflib.unified_diff(
                    a[field].replace(", ", ",\n").splitlines(),
                    b[field].replace(", ", ",\n").splitlines(),
                    "upstream",
                    "port",
                    n=1,
                    lineterm="",
                )
                print("\n".join(list(diff)[:40]))
                return 1

    if upstream["days"] != port["days"]:
        print(f"days differ: upstream {len(upstream['days'])}, port {len(port['days'])}")
        problems += 1
    worst = max((abs(a - b) for a, b in zip(upstream["nav"], port["nav"], strict=False)), default=0.0)
    if worst != 0.0:
        print(f"NAV differs: largest gap {worst * 100_000:.6f} dollars")
        problems += 1
    if upstream["trades"] != port["trades"]:
        print(f"trades differ: upstream {len(upstream['trades'])}, port {len(port['trades'])}")
        problems += 1

    print(
        f"{upstream['script']}: {len(up_calls)} calls, {len(upstream['days'])} days, "
        f"{len(upstream['trades'])} trades, final NAV {upstream['nav'][-1]:.6f} vs {port['nav'][-1]:.6f}, "
        f"upstream counts {upstream.get('counts')}: {'MATCH' if problems == 0 else 'DIFFERENT'}"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
