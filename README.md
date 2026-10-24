# StockBench for Inspect

This is a port of [StockBench](https://github.com/ChenYXxxx/stockbench) to [Inspect](https://inspect.aisi.org.uk/).
StockBench tests whether LLM agents can trade stocks profitably.
The agent manages $100,000 across 20 Dow Jones stocks for the 83 trading days from March 3 to June 30, 2025.
Each day it sees prices, news and position data, and it decides how much to hold in each stock.

Paper: [StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?](https://arxiv.org/abs/2510.02209) (Chen et al.)

## Run it

```bash
git clone https://github.com/mahimn01/stockbench-inspect
cd stockbench-inspect
uv sync
uv run inspect eval src/stockbench_inspect/tasks.py@stockbench --model openai/gpt-5-mini
```

The paper reports the mean of 3 runs per model, so the task runs 3 epochs by default. Add `--epochs 1` for a single run.
`uv sync` installs the `openai` package, which OpenAI and Ollama models need. For other providers, add their package (for example `uv pip install anthropic`).

To run a local model with Ollama, give it at least 40k tokens of context, because the daily prompts are long:

```bash
OLLAMA_CONTEXT_LENGTH=40960 ollama serve
uv run inspect eval src/stockbench_inspect/tasks.py@stockbench --model ollama/gpt-oss:20b --max-connections 1
```

The first run downloads the market data (about 6 MB) from the StockBench repository at a pinned commit.
Every file is checked against `src/stockbench_inspect/data/SHA256SUMS`.

### Options

| Option | Default | Meaning |
|---|---|---|
| `-T fundamentals=false` | `true` | Leave out the fundamentals snapshot. Upstream's snapshot is not point-in-time (see below). |
| `-T days=5` | all 83 | Trade only the first N days. For smoke tests only. |

The task sends `temperature=0.7` and `max_tokens=16000`, as the released code does. Override them with `--temperature` and `--max-tokens`.

## How it works

Each trading day the agent makes two model calls, using the original prompts unchanged:

1. A filter agent sees prices, the last 7 closes, recent news and positions for every stock. It picks which stocks need fundamentals.
2. A decision agent sees the same data, plus fundamentals for the picked stocks. It returns a target dollar amount for each stock.

Orders fill at that day's open with a 1 basis point commission. Shares can be fractional and short selling is not allowed.
Invalid decisions are sent back to the model with the original error notes, up to 3 attempts. If no attempt is usable, every stock holds.

## Scores

| Score | Meaning |
|---|---|
| `final_return` | Return on the starting $100,000 after costs. This is the headline score. |
| `max_drawdown` | Largest fall from a previous peak in daily portfolio value. |
| `sortino` | Mean daily return divided by the standard deviation of the negative daily returns. Daily, not annualized, as in the paper. |
| `baseline_return` | Equal-weight buy and hold of the 20 stocks over the same days. |
| `excess_return` | `final_return` minus `baseline_return`. |
| `trades` | Number of executed orders. |
| `decision_unusable_rate` | Share of decision replies that gave no usable decision for any stock (for example prose with no JSON). |
| `fallback_days` | Days on which no stock's final decision came from the model, so every stock held. |
| `filter_unusable_rate` | Share of filter replies with no usable JSON. |

The last three show when a weak model looks safe only because its replies could not be used.

## How close it is to the original

**Parity with the original engine.** `scripts/parity/` runs the original StockBench code and this port on the same scripted model replies.
The scripts cover equal-weight buying, random rebalancing and malformed replies.
Across all three, every prompt is identical byte for byte, and the trades and daily portfolio value match exactly.
`tests/test_parity_golden.py` checks the port against recorded fingerprints of those original runs, so the check runs without installing the original.

**The paper's baseline.** The buy-and-hold baseline comes out at 0.4% return, -15.2% max drawdown and Sortino 0.0155. That matches Table 2 of the paper.

## Results versus the paper

Results from this port are in the first table. The paper's numbers are in the second. Each model ran 3 epochs on the full window, 2025-03-03 to 2025-06-30, and the numbers are the mean of the 3 epochs, as the paper averages 3 runs. Every number in this section was produced by `scripts/summarize_logs.py` from the logs at commit `TBD`.

| Model | Return | Max drawdown | Sortino | Trades | Unusable decisions | Fallback days |
|---|---:|---:|---:|---:|---:|---:|
| `gpt-oss:20b` | TBD | TBD | TBD | TBD | TBD | TBD |
| `qwen3:30b-a3b-thinking-2507` | TBD | TBD | TBD | TBD | TBD | TBD |
| Buy and hold | TBD | TBD | TBD | n/a | n/a | n/a |

| Paper, Table 2 | Return | Max drawdown | Sortino |
|---|---:|---:|---:|
| GPT-OSS-20B | -2.8% | -14.4% | -0.0069 |
| Qwen3-30B-Think | 2.1% | -13.5% | 0.0255 |
| Passive baseline | 0.4% | -15.2% | 0.0155 |

Per-epoch numbers are in `TBD`.

How to read the comparison:

- **Three epochs is a small sample.** Each epoch is one 83-day backtest of one model. Epochs of the same model can land several points apart, so a gap to the paper smaller than that spread is not evidence of a difference. The per-epoch table shows the spread.
- **The setup is not identical to the paper's.** The models here ran locally in Ollama with 4-bit quantized weights (`TBD`), a context of `TBD` tokens, temperature 0.7 and a maximum of 16000 output tokens. The paper does not say which quantization or serving stack it used, and it reports a 32,768 token context. A different quantization, context size or serving stack can change the replies and so the trades.
- **Costs.** This port charges a 1 basis point commission on every order, as the released code does. The limitations section of the paper's second version says trading costs are not modeled.
- **The window length differs by one day.** The paper's text says 82 trading days. This port uses the 83 days that have price data, which is the length that reproduces the paper's baseline Sortino.
- **Compare against buy and hold first.** In this window the equal-weight baseline returned about 0.4% with a drawdown of about -15.2%. A model that ends near zero but trades very little or falls back to holding on most days is behaving like the baseline, so check the fallback and unusable-reply columns before reading a return as skill.
- **The Sortino formula in the paper's text differs from the released code.** The text describes a root mean square of the negative returns. The released code, and this port, divide the mean daily return by the sample standard deviation of the negative daily returns. The baseline Sortino of 0.0155 in Table 2 reproduces only with the code's version. The text's version gives 0.0124 for the same baseline.

## Differences from the original code

Each one fixes a bug or a look-ahead in the released code.

1. **Calendar.** The original treats Good Friday, Memorial Day and Juneteenth as trading days, so it simulates 86 days. The port uses the 83 days that have price data. The paper's baseline Sortino (0.0155) matches 83 days. With 86 it would be 0.0149.
2. **No news from the decision day.** When a stock has no stored news for the previous two days, the original's offline mode widens the window to 4 days back and 1 day ahead. That can show news from the decision day itself. This affects 62 stock-days. The original's older news cache also adds two same-day headlines for IBM on 2025-03-04. The port shows only news published before the decision day.
3. **Failed model calls stop the run.** The original retries a failed API call a few times and then carries on: the filter gives every stock fundamentals, and the decision step retries or holds every stock. Here a call that still fails after Inspect's retries ends the sample with an error. That includes a prompt longer than the model's context window, which is why local models need at least 40k tokens of context.
4. **No end-of-run summary call.** The original asks a model to write a summary after the backtest. It is not scored, so the port leaves it out.

## Known issues kept from the original

These are part of the published setup, so the port keeps them:

- The fundamentals snapshots are not point-in-time. For 15 of the 20 stocks, the 52-week high or low shown on March 3 is a price from a later date, inside the test window for 14 of them. Use `-T fundamentals=false` to remove them.
- The stored dividend yield is 100 times too large (for example 41.2 for Apple).
- The decision prompt contains placeholders that are never filled in, such as `{total_assets}`.
- Error notes on retries contain the characters `\n` instead of line breaks.
- Dividends are paid to the agent but not to the buy-and-hold baseline.
- Orders execute in a fixed stock order with a running cash check. A buy that no longer fits after commission and earlier buys is skipped without telling the model, and sells later in the order do not pay for it.

## Data and license

The code is under the Apache License 2.0, like the original. See `LICENSE` and `NOTICE`.
The prompts in `src/stockbench_inspect/prompts/` are copied unchanged from StockBench.
The market data comes from Polygon and Finnhub through the StockBench repository. It is downloaded from there and not included here.

## Development

```bash
uv run pytest            # unit tests, plus parity and data tests (downloads the data once)
uv run ruff check .
```

To repeat the parity check against the original code, see the instructions at the top of `scripts/parity/run_upstream.py`.

## Citation

```bibtex
@misc{chen2026stockbenchllmagentstrade,
      title={StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets?},
      author={Yanxu Chen and Zijun Yao and Yantao Liu and Amy Xin and Jin Ye and Jianing Yu and Lei Hou and Juanzi Li},
      year={2026},
      eprint={2510.02209},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.02209},
}
```
