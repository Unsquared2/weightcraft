# AGENTS.md

## What this is

`weightcraft` is a library of pure numerical functions for portfolio
construction. It is meant to be consumed by more than one caller, which is the
reason for every rule below.

## Hard rules

- **No pandas.** numpy and polars only. `ruff` bans the import outright.
- **No I/O, no global state, no configuration lookup.** If a function needs a
  number, it takes it as an argument or as a field on a frozen config dataclass.
  A function that reads a file or an environment variable does not belong here.
- **No dependency on any consumer.** Nothing in this repo may import, or know
  the name of, anything that calls it.
- **Missing is NaN and stays NaN.** Never `nan_to_num` a caller's data on their
  behalf. NaN positions are compared exactly in tests, never through a fill.
- **Frozen dataclasses.** Anything with more than a couple of knobs is a
  `@dataclass(frozen=True, slots=True)`, not a pile of keyword arguments.
- **Shape in the type.** Use the aliases in `weightcraft.arrays` rather than
  `npt.NDArray`, so a signature says what rank it expects.

## Quality gates

All four must pass; CI runs them on every push.

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy          # strict, plus disallow_any_explicit and warn_unreachable
uv run pytest        # 95% coverage floor
```

## Tests

Assert **invariants**, not golden numbers. "Inverse-vol sizes the quietest name
largest", "equal risk contributions come out equal", "rescaling to a gross
target reaches it" — these survive a refactor and a numerical change of basis;
a stored array does not.

The suite is organised around four ideas, borrowed from how empyrical,
PyPortfolioOpt, ffn and cvxportfolio test the same kind of code:

| File | What it does |
| --- | --- |
| `canonical.py` | named series — flat line, positive line, sparse noise, all-missing — so a new metric can be swept over every shape at once, the way empyrical's fixtures work |
| `test_invariants.py` | scale, translation, idempotence, composition and order invariance. A change of units must not move a ratio; a change of origin must not move a dispersion |
| `test_against_reference.py` | every kernel against a second, deliberately naive implementation written in the test file. `equal_risk_row` is checked against cyclical coordinate descent — a different algorithm, so agreement is evidence rather than tautology |
| `test_degenerate_inputs.py` | every public function swept over empty, one-row, one-column, constant, all-missing, infinite and overflowing panels. Nothing may raise, warn, or return a number that looks settled when it is not |
| `test_structure.py` | positive semi-definiteness, correlation round trips, shrinkage endpoints, and the alignment invariant that each frame keeps its own values at its own labels |
| `test_properties.py` | Hypothesis, generating NaN and infinities deliberately |
| `test_regressions.py` | one test per defect found in review, each named after what it broke |

`filterwarnings = ["error"]` is load-bearing: a leaked numpy RuntimeWarning is
a test failure, because an all-missing row is the ordinary case here and a
library that warns on it is unusable inside a loop.

A behaviour that looks wrong needs a test naming why it is right, so the next
person does not "fix" it. See the design notes in the README.

## Comments

Prefer none. A comment has to earn its place by saying something the code
cannot: why a non-obvious choice was made, what breaks without it, a constraint
imposed from outside. Restating what the next line does is worse than silence.

When one is warranted, make it a single sentence.

## Docstrings

**Three lines maximum**, and to the point. Say what the function is — the
expression it reproduces, the shape it returns, the one thing a caller would get
wrong. Not how it works, not what was tried first.
