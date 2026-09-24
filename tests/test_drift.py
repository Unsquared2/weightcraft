from __future__ import annotations

import numpy as np
import pytest

from weightcraft import drifted_weights


def test_price_gains_change_exposure_and_nav_without_resetting_the_target() -> None:
    result = drifted_weights(
        np.array([[0.5, 0.0]]),
        np.array([[100.0, 100.0]]),
        np.array([[110.0, 100.0]]),
    )

    # A $50 holding becomes $55 while the $50 cash balance stays unchanged.
    assert result[0] == pytest.approx([55.0 / 105.0, 0.0])


@pytest.mark.parametrize("missing", [np.nan, np.inf, 0.0, -1.0])
def test_an_unheld_asset_needs_no_mark_but_a_held_asset_does(missing: float) -> None:
    prices = np.array([[110.0, missing]])

    valued = drifted_weights(
        np.array([[0.5, 0.0]]), np.array([[100.0, missing]]), prices
    )
    unvalued = drifted_weights(
        np.array([[0.5, 0.2]]), np.array([[100.0, 100.0]]), prices
    )

    assert valued[0] == pytest.approx([55.0 / 105.0, 0.0])
    assert np.isnan(unvalued).all()


@pytest.mark.parametrize(
    ("reference", "price"), [(100.0, 50.0), (100.0, 40.0), (1.0, 1e308)]
)
def test_insolvent_or_overflowing_rows_cannot_look_like_a_valid_portfolio(
    reference: float,
    price: float,
) -> None:
    result = drifted_weights(
        np.array([[2.0]]), np.array([[reference]]), np.array([[price]])
    )

    assert np.isnan(result).all()


def test_prices_in_different_units_produce_the_same_weights() -> None:
    targets = np.array([[0.5, -0.3]])
    before = np.array([[100.0, 40.0]])
    after = np.array([[110.0, 35.0]])
    units = np.array([[1000.0, 0.01]])

    assert drifted_weights(targets, before, after) == pytest.approx(
        drifted_weights(targets, before * units, after * units)
    )


def test_two_marks_without_a_target_reset_equal_one_direct_mark() -> None:
    targets = np.array([[0.5, -0.3], [2.0, 0.0]])
    before = np.array([[100.0, 40.0], [100.0, 40.0]])
    middle = np.array([[105.0, 45.0], [95.0, 40.0]])
    after = np.array([[110.0, 35.0], [90.0, 40.0]])

    assert drifted_weights(
        drifted_weights(targets, before, middle), middle, after
    ) == pytest.approx(drifted_weights(targets, before, after))


@pytest.mark.parametrize("shape", [(0, 3), (3, 0)])
def test_empty_arrays_preserve_their_shape(shape: tuple[int, int]) -> None:
    values = np.empty(shape)

    assert drifted_weights(values, values, values).shape == shape


@pytest.mark.parametrize("weight", [np.nan, np.inf, -np.inf])
def test_an_unknown_weight_invalidates_the_whole_row(weight: float) -> None:
    assert np.isnan(
        drifted_weights(np.array([[0.5, weight]]), np.ones((1, 2)), np.ones((1, 2)))
    ).all()


def test_price_rows_cannot_silently_broadcast_across_targets() -> None:
    with pytest.raises(ValueError, match="same shape"):
        drifted_weights(np.ones((2, 2)), np.ones((1, 2)), np.ones((2, 2)))
