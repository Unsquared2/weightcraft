# weightcraft

Pure portfolio-construction numerics: sizing, risk, turnover, alignment.

Every function here is a function of arrays. No I/O, no configuration lookup, no
global state, no registry — hand it numpy, get numpy back. That is the whole
design constraint, and it is what makes this library safe to share between the
research pipeline that produces weights and the services that consume them.

## Conventions

- **A panel is `(dates, assets)`.** A stack of panels is `(sources, dates, assets)`.
  The aliases in `weightcraft.arrays` carry the rank in the type, so a signature
  says whether it wants a series, a panel, or a stack.
- **Missing is `NaN`, always, and never silently filled.** A gap is information:
  an asset outside the universe on that date is not an asset with a zero weight.
  **An infinity is missing too** — it is a broken input, not a large number, and
  every function here asks the question the same way, with no exceptions.
- **A sizing multiplier sums to the held count, not to one.** Constructors
  *scale* an existing book rather than replacing its gross, so they compose.
- **`WeightFrame` is immutable.** Its arrays are copied and marked read-only on
  construction, so neither the holder nor whoever handed the data over can
  change it afterwards.

## Install

```bash
uv add "weightcraft @ git+https://github.com/Unsquared2/weightcraft.git"
```

## Use

```python
from weightcraft import EqualRiskConfig, align, equal_risk_weights, nanmean_stack

stack = align([first, second])
combined = stack.with_values(nanmean_stack(stack.values))

sized = equal_risk_weights(returns, holdings, EqualRiskConfig(period=90, rebalance=5))
```

## What is here

| Module | What it holds |
| --- | --- |
| `frame` | `WeightFrame`, the immutable dates x assets container, plus polars interop |
| `align` | union-aligning several frames into one `(sources, dates, assets)` stack |
| `combine` | reductions over a stack: mean, median, weighted mean, share normalisation |
| `normalize` | gross, net, caps, lot sizes, tilts, quantile bins to a long/short book |
| `risk` | trailing volatility, inverse-vol, equal risk contribution, volatility targeting |
| `costs` | turnover, transaction costs, the lag between a decision and its return |
| `smoothing` | rolling and exponentially weighted means that wait for a full window |
| `cross_section` | row z-score, percentile rank, top-N selection, Gram-Schmidt neutralisation |
| `metrics` | compounded return, CAGR, Sharpe, beta, drawdown |

## Design notes

A few behaviours here look like bugs and are not. Each is load-bearing, and each
has a test naming why it is right so the next reader does not "fix" it.

- **The equal-risk update is the geometric mean** of the old and new weight. The
  undamped `w <- sigma^2 / (n * (Sigma w)_i)` oscillates forever and never lands.
- **The equal-risk marginal is floored.** A sample covariance over a few hundred
  bars is not positive semi-definite, and `(Sigma w)_i` goes negative for the
  calmest name on the first iteration.
- **`trailing_std` centres on the window mean** rather than differencing
  cumulative sums of squares. The differenced form cancels catastrophically once
  the level dwarfs the variation — for `1e10 + noise` it reports a confident
  zero, which `volatility_target` then reads as "no risk".
- **`rolling_sums` windows rather than differencing a cumulative sum**, so a gap
  voids the windows containing it instead of every window after it.
- **`turnover` treats a gap as a flat position.** A cell going from missing to
  held has been traded into. Since `align` pads every uncovered cell with NaN by
  construction, blanking those transitions would make the transaction-cost knob
  a no-op on exactly the books it is meant to price.
- **`weights_from_bins` maps from both endpoints**, not just the row maximum.
  Dividing by the maximum alone only lands on `[-1, 1]` when the lowest bin is
  exactly zero; a 1-based scheme comes out silently net long, and a negative one
  comes out inverted.
- **The smoothers count observations, not rows.** `min_periods` means what
  pandas means by it, so a column that started late does not answer early off a
  single point.
- **`WeightFrame` rejects duplicate dates.** `align` places rows by
  `searchsorted` against the union of every frame's dates, so a repeated date is
  not a duplicate row — it is one row silently overwriting another.
- **`sharpe` returns `-10`** for a series that never varies, so a sort puts it
  last instead of dropping it.

## Development

```bash
uv sync
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
```

The suite is roughly 690 tests at ~99.9% coverage, built around invariants
rather than stored arrays: scale and translation invariance, idempotence, cross
-checks against naive reference implementations (including an independent
coordinate-descent solver for equal risk contribution), and a sweep of every
public function over empty, single-row, constant, all-missing and overflowing
inputs. `filterwarnings = ["error"]` is deliberate — an all-missing row is
ordinary here, and a library that warns on it cannot be called in a loop.
