import asyncio
import json
import sys
from pathlib import Path

import pytest

PARITY_DIR = Path(__file__).parents[1] / "scripts" / "parity"
sys.path.insert(0, str(PARITY_DIR))
from make_golden import fingerprint  # noqa: E402
from replies import SCRIPTS  # noqa: E402
from run_port import run  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "golden" / "upstream_parity.json").read_text(encoding="utf-8"))


@pytest.mark.data
@pytest.mark.parametrize("script", SCRIPTS)
def test_port_matches_the_original_engine(script: str) -> None:
    """Same scripted replies, same prompts byte for byte, same trades, same NAV every day."""
    port = fingerprint(asyncio.run(run(script)))
    expected = {k: v for k, v in GOLDEN["scripts"][script].items() if k != "upstream_counts"}
    assert port == expected
