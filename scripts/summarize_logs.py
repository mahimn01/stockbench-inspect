"""Summarize StockBench Inspect logs: one row per epoch, the mean, and the paper's numbers.

Usage: uv run python scripts/summarize_logs.py LOG_OR_DIR [LOG_OR_DIR ...] [--model SUBSTRING]
                                               [--paper-formula] [--json] [--include-smoke]

Why this exists: `inspect view` shows only the mean over epochs, but the README needs the
per-epoch spread, how often the replies were unusable, and the paper's numbers beside ours.

Every number comes from the stockbench_scorer score stored in the log, which is what the
eval itself reported. The script also recomputes return, max drawdown and Sortino from the
stored daily NAV with the port's own functions (stockbench_inspect.metrics) and warns if
the log disagrees, so a truncated or edited log does not pass silently.
The mean across epochs is a plain mean of the per-epoch values, which is what Inspect's
`mean()` metric reports and what the paper does with its 3 runs.

Skipped: logs of other tasks, smoke runs (`-T days=N`, unless --include-smoke), logs with
no finished sample, and samples that ended in an error (counted in the notes).
A log that is still being written may not open. Copy it somewhere first and read the copy.
The paper's Sortino is the one printed by the released code (sample standard deviation of
the negative daily returns). `--paper-formula` adds a column with the formula as printed in
the paper (root mean square of the negative daily returns), for comparison only.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from inspect_ai.log import EvalLog, read_eval_log

from stockbench_inspect.metrics import agent_metrics, daily_returns

TASK = "stockbench"
SCORER = "stockbench_scorer"
TOLERANCE = 1e-9

# Table 2 of the paper (arXiv 2510.02209): return %, max drawdown %, Sortino. Mean of 3 runs.
PAPER: dict[str, tuple[str, float, float, float]] = {
    "gpt-oss-20b": ("GPT-OSS-20B", -2.8, -14.4, -0.0069),
    "qwen3-30b-a3b-thinking": ("Qwen3-30B-Think", 2.1, -13.5, 0.0255),
}
PAPER_BASELINE = (0.4, -15.2, 0.0155)


@dataclass
class EpochRow:
    log: str
    epoch: int
    final_return: float
    max_drawdown: float
    sortino: float
    sortino_paper_formula: float
    trades: int
    decision_unusable_rate: float
    filter_unusable_rate: float
    fallback_days: int
    baseline: dict[str, float]


@dataclass
class ModelSummary:
    model: str
    settings: str
    label: str  # the model, plus its settings when the model ran with more than one
    rows: list[EpochRow] = field(default_factory=list)
    logs: set[str] = field(default_factory=set)
    commits: set[str] = field(default_factory=set)
    expected_epochs: int = 0
    notes: list[str] = field(default_factory=list)


def paper_sortino(nav: list[float]) -> float:
    """The formula as printed in the paper: mean daily return over the root mean square of the negative returns."""
    returns = daily_returns(nav)
    downside = [r for r in returns if r < 0]
    if not downside:
        return 0.0
    deviation = math.sqrt(sum(r * r for r in downside) / len(downside))
    return (sum(returns) / len(returns)) / deviation


def paper_row(model: str) -> tuple[str, float, float, float] | None:
    """Table 2's row for a model name such as ollama/gpt-oss:20b, or None if the paper has no such model."""
    key = model.lower().split("/", 1)[-1].replace(":", "-")
    for prefix, row in PAPER.items():
        if key.startswith(prefix):
            return row
    return None


def log_paths(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        paths.extend(sorted(item.glob("*.eval")) if item.is_dir() else [item])
    return paths


def check_against_nav(row: EpochRow, nav: list[float], raw: dict[str, Any], trades_in_store: int | None) -> list[str]:
    """Differences between the stored scores and what the stored NAV and counters imply."""
    problems = []
    recomputed = agent_metrics(nav)
    for name, value in recomputed.items():
        if abs(value - raw[name]) > TOLERANCE:
            problems.append(f"{name} is {raw[name]:.10f} in the log but {value:.10f} from its NAV")
    if trades_in_store is not None and trades_in_store != row.trades:
        problems.append(f"trades is {row.trades} in the log but the stored trade list has {trades_in_store}")
    return problems


def run_settings(log: EvalLog) -> str:
    """The settings that change results: task arguments other than days, then temperature and max tokens."""
    generate = log.eval.model_generate_config.merge(log.plan.config)  # the eval's config wins, as in Inspect
    args = [f"{name}={value}" for name, value in sorted(log.eval.task_args.items()) if name != "days"]
    return ", ".join([*args, f"temperature={generate.temperature}", f"max_tokens={generate.max_tokens}"])


def collect(paths: list[Path], model_filter: str | None, include_smoke: bool) -> dict[str, ModelSummary]:
    """One summary per model and settings, keyed by its label. Runs with different settings are never averaged."""
    groups: dict[tuple[str, str], ModelSummary] = {}
    for path in paths:
        try:
            log: EvalLog = read_eval_log(str(path))
        except Exception as error:  # a log still being written, or not a log at all
            print(f"skipped {path.name}: cannot read it ({type(error).__name__}: {error})", file=sys.stderr)
            continue
        if log.eval.task.rsplit("/", 1)[-1] != TASK or not log.samples:
            continue
        if log.eval.task_args.get("days") is not None and not include_smoke:
            continue
        if model_filter and model_filter.lower() not in log.eval.model.lower():
            continue
        settings = run_settings(log)
        summary = groups.setdefault(
            (log.eval.model, settings), ModelSummary(model=log.eval.model, settings=settings, label=log.eval.model)
        )
        summary.logs.add(path.name)
        summary.expected_epochs = max(summary.expected_epochs, log.eval.config.epochs or 1)
        if log.eval.revision:
            summary.commits.add(log.eval.revision.commit[:7] + ("+dirty" if log.eval.revision.dirty else ""))
        if log.status != "success":
            summary.notes.append(f"{path.name} has status '{log.status}', so it may still be running")
        for sample in log.samples:
            score = sample.scores.get(SCORER) if sample.scores else None
            if sample.error or score is None:
                summary.notes.append(f"{path.name} epoch {sample.epoch} ended in an error and was left out")
                continue
            raw = score.value
            nav = score.metadata["nav"]
            row = EpochRow(
                log=path.name,
                epoch=sample.epoch,
                final_return=raw["final_return"],
                max_drawdown=raw["max_drawdown"],
                sortino=raw["sortino"],
                sortino_paper_formula=paper_sortino(nav),
                trades=int(raw["trades"]),
                decision_unusable_rate=raw["decision_unusable_rate"],
                filter_unusable_rate=raw["filter_unusable_rate"],
                fallback_days=int(raw["fallback_days"]),
                baseline=score.metadata["baseline"],
            )
            store_trades = sample.store.get("BacktestState:trades")
            for problem in check_against_nav(row, nav, raw, None if store_trades is None else len(store_trades)):
                summary.notes.append(f"CHECK FAILED {path.name} epoch {sample.epoch}: {problem}")
            summary.rows.append(row)
    finished = [summary for summary in groups.values() if summary.rows]
    for summary in finished:
        variants = sum(other.model == summary.model for other in finished)
        if variants > 1:
            summary.label = f"{summary.model} ({summary.settings})"
            summary.notes.insert(
                0,
                f"the runs of {summary.model} used {variants} different settings. "
                "Each is summarized on its own, and they are not averaged together.",
            )
    return {summary.label: summary for summary in finished}


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def spread(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else float("nan")


def render(summary: ModelSummary, paper_formula: bool) -> str:
    rows = sorted(summary.rows, key=lambda r: (r.log, r.epoch))
    several_logs = len(summary.logs) > 1
    header = ["Epoch", "Return %", "Max drawdown %", "Sortino"]
    if paper_formula:
        header.append("Sortino (paper formula)")
    header += ["Trades", "Unusable decisions %", "Unusable filters %", "Fallback days"]

    def cells(
        label: str, ret: float, dd: float, so: float, sp: float, trades: float, dec: float, fil: float, fb: float
    ):
        out = [label, f"{ret * 100:.2f}", f"{dd * 100:.1f}", f"{so:.4f}"]
        if paper_formula:
            out.append(f"{sp:.4f}")
        return out + [
            f"{trades:g}" if trades == int(trades) else f"{trades:.1f}",
            f"{dec * 100:.1f}",
            f"{fil * 100:.1f}",
            f"{fb:g}",
        ]

    table = [header]
    for r in rows:
        label = f"{r.log[:19]} #{r.epoch}" if several_logs else str(r.epoch)
        table.append(
            cells(
                label,
                r.final_return,
                r.max_drawdown,
                r.sortino,
                r.sortino_paper_formula,
                r.trades,
                r.decision_unusable_rate,
                r.filter_unusable_rate,
                r.fallback_days,
            )
        )
    columns = [
        [r.final_return for r in rows],
        [r.max_drawdown for r in rows],
        [r.sortino for r in rows],
        [r.sortino_paper_formula for r in rows],
        [float(r.trades) for r in rows],
        [r.decision_unusable_rate for r in rows],
        [r.filter_unusable_rate for r in rows],
        [float(r.fallback_days) for r in rows],
    ]
    table.append(cells(f"**Mean of {len(rows)}**", *[mean(c) for c in columns]))
    if len(rows) > 1:
        table.append(cells("Std across epochs", *[spread(c) for c in columns]))
    base = {k: mean([r.baseline[k] for r in rows]) for k in ("final_return", "max_drawdown", "sortino")}
    paper = paper_row(summary.model)
    if paper:
        _, ret, dd, so = paper
        table.append(
            [f"Paper {paper[0]}", f"{ret:.1f}", f"{dd:.1f}", f"{so:.4f}"]
            + (["n/a"] if paper_formula else [])
            + ["n/a"] * 4
        )
        table.append(
            [
                "Ours minus paper",
                f"{mean(columns[0]) * 100 - ret:+.2f}",
                f"{mean(columns[1]) * 100 - dd:+.1f}",
                f"{mean(columns[2]) - so:+.4f}",
            ]
            + (["n/a"] if paper_formula else [])
            + ["n/a"] * 4
        )
    table.append(
        [
            "Buy and hold (ours)",
            f"{base['final_return'] * 100:.2f}",
            f"{base['max_drawdown'] * 100:.1f}",
            f"{base['sortino']:.4f}",
        ]
        + (["n/a"] if paper_formula else [])
        + ["n/a"] * 4
    )
    pr, pd_, ps = PAPER_BASELINE
    table.append(
        ["Buy and hold (paper)", f"{pr:.1f}", f"{pd_:.1f}", f"{ps:.4f}"]
        + (["n/a"] if paper_formula else [])
        + ["n/a"] * 4
    )

    lines = [f"### {summary.label}", ""]
    lines.append("| " + " | ".join(table[0]) + " |")
    lines.append("|" + "|".join("---" if i == 0 else "---:" for i in range(len(table[0]))) + "|")
    lines += ["| " + " | ".join(row) + " |" for row in table[1:]]
    lines.append("")
    status = (
        f"{len(rows)} of {summary.expected_epochs} epochs finished"
        if len(rows) < summary.expected_epochs
        else f"{len(rows)} epochs"
    )
    lines.append(
        f"{status}. Settings: {summary.settings}. Commit: {', '.join(sorted(summary.commits)) or 'unknown'}. "
        f"Logs: {', '.join(sorted(summary.logs))}."
    )
    if several_logs:
        lines.append("NOTE: more than one log contributes. Check that they are one run, not a restart.")
    if len(summary.commits) > 1:
        lines.append("NOTE: the logs come from different commits.")
    lines += [f"NOTE: {note}" for note in summary.notes]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("logs", nargs="+", type=Path, help="log files or directories of .eval files")
    parser.add_argument("--model", help="keep only models whose name contains this text, such as gpt-oss")
    parser.add_argument(
        "--paper-formula", action="store_true", help="add a Sortino column using the paper's printed formula"
    )
    parser.add_argument("--include-smoke", action="store_true", help="also count runs made with -T days=N")
    parser.add_argument("--json", action="store_true", help="print JSON instead of tables")
    args = parser.parse_args()

    summaries = collect(log_paths(args.logs), args.model, args.include_smoke)
    if not summaries:
        sys.exit("No finished StockBench epochs found.")
    if args.json:
        print(
            json.dumps(
                {
                    label: {
                        "model": s.model,
                        "settings": s.settings,
                        "rows": [asdict(r) for r in s.rows],
                        "notes": s.notes,
                    }
                    for label, s in summaries.items()
                },
                indent=2,
            )
        )
        return
    print("\n\n".join(render(s, args.paper_formula) for s in summaries.values()))


if __name__ == "__main__":
    main()
