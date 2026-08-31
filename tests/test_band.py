from __future__ import annotations

import numpy as np
import pytest

from weightcraft.band import no_trade_band


def test_a_move_at_exactly_the_threshold_is_not_traded() -> None:
    # Binary-exact fractions, so the comparison lands on the boundary with no
    # floating-point slop: scale is mean(0.625, 0.375) = 0.5, band * scale is
    # exactly 0.125, and the move from 0.5 is exactly 0.125.
    target = np.asarray([[0.5, 0.5], [0.625, 0.375]])
    held = no_trade_band(target, 0.25)
    assert held[1].tolist() == [0.5, 0.5]


def test_a_move_a_hair_past_the_threshold_is_traded() -> None:
    target = np.asarray([[0.5, 0.5], [0.625001, 0.375]])
    held = no_trade_band(target, 0.25)
    assert held[1, 0] == pytest.approx(0.625001)


def test_the_scale_comes_from_the_target_row_not_the_held_row() -> None:
    # A tiny held level next to a huge target row must be judged against the
    # target's own mean, not against whatever is still sitting in `current`.
    target = np.asarray([[0.001, 0.001], [10.0, -10.0]])
    held = no_trade_band(target, 0.5)
    assert held[1].tolist() == [10.0, -10.0]


def test_a_dropped_name_always_exits_even_under_a_huge_band() -> None:
    target = np.asarray([[0.5, 0.5], [0.5, np.nan]])
    held = no_trade_band(target, 1000.0)
    assert np.isnan(held[1, 1])


def test_an_appearing_name_always_enters_even_under_a_huge_band() -> None:
    target = np.asarray([[0.5, np.nan], [0.5, 0.5]])
    held = no_trade_band(target, 1000.0)
    assert held[1, 1] == 0.5


def test_an_all_missing_row_flattens_the_book() -> None:
    target = np.asarray([[0.5, 0.5], [np.nan, np.nan]])
    held = no_trade_band(target, 0.1)
    assert np.isnan(held[1]).all()


def test_an_all_zero_row_passes_straight_through() -> None:
    # scale <= 0 disables the band for that row, so it behaves as a hold-open.
    target = np.asarray([[0.5, -0.5], [0.0, 0.0]])
    held = no_trade_band(target, 0.1)
    assert held[1].tolist() == [0.0, 0.0]


def test_a_zero_band_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        no_trade_band(np.zeros((1, 1)), 0.0)


def test_a_negative_band_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        no_trade_band(np.zeros((1, 1)), -0.1)


def test_the_first_row_always_enters_at_the_target() -> None:
    target = np.asarray([[0.2, -0.2, 0.6]])
    held = no_trade_band(target, 0.1)
    assert held[0].tolist() == [0.2, -0.2, 0.6]


def test_a_later_row_cannot_change_an_earlier_answer() -> None:
    # Each row's decision depends only on what was actually held into it, so
    # no amount of rewriting the future may move a row already decided.
    generator = np.random.default_rng(9)
    target = generator.normal(size=(20, 4)) * 0.1
    target[generator.random(target.shape) < 0.2] = np.nan
    rewritten = target.copy()
    rewritten[10:] = generator.normal(size=rewritten[10:].shape) * 0.1
    original = no_trade_band(target, 0.3)
    changed = no_trade_band(rewritten, 0.3)
    assert np.array_equal(original[:10], changed[:10], equal_nan=True)
