import getpass
import sys
from pathlib import Path

import pytest
from inspect_ai import eval
from inspect_ai.log import read_eval_log, write_eval_log

from stockbench_inspect import stockbench

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
from scrub_logs import leftovers, scrub


@pytest.mark.data
def test_scrub_removes_local_paths_and_keeps_the_scores(tmp_path, market_data) -> None:
    [log] = eval(stockbench(days=2), model="mockllm/model", log_dir=str(tmp_path / "in"), display="none")
    user = getpass.getuser()
    log.eval.metadata = {
        "home": f"{Path.home()}/work/logs",
        "other_user": f"/Users/{user}/x",
        "disk": "/Volumes/MacOffload/stockbench-work/logs",
        "disk_in_text": 'see "/Volumes/Backup Disk/a" or /Volumes/Other/b',
    }
    assert leftovers(log) >= 5

    scrubbed = scrub(log)
    assert leftovers(scrubbed) == 0
    assert scrubbed.eval.metadata["home"] == "~/work/logs"
    assert scrubbed.eval.metadata["disk"] == "/Volumes/volume/stockbench-work/logs"
    assert scrubbed.eval.metadata["disk_in_text"].startswith('see "/Volumes/volume')

    target = tmp_path / "out" / "log.eval"
    target.parent.mkdir()
    write_eval_log(scrubbed, str(target))
    reread = read_eval_log(str(target))
    assert reread.status == log.status
    assert [s.scores["stockbench_scorer"].value for s in reread.samples] == [
        s.scores["stockbench_scorer"].value for s in log.samples
    ]
    assert [s.scores["stockbench_scorer"].metadata["nav"] for s in reread.samples] == [
        s.scores["stockbench_scorer"].metadata["nav"] for s in log.samples
    ]
