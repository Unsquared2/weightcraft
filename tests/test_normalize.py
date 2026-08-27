from __future__ import annotations

import numpy as np
import pytest

from weightcraft.normalize import (
    center,
    clip_allocation,
    fill_missing,
    gross,
    held,
    normalised_share,
    quantize,
    rescaled_to_held_count,
    row_sums,
    tilt,
    to_gross,
    weights_from_bins,
)


def test_row_sums_skip_gaps_and_keep_the_column_shape() -> None:
    values = np.asarray([[1.0, np.nan, 2.0]])
    assert row_sums(values).shape == (1, 1)
    assert row_sums(values).tolist() == [[3.0]]


def test_a_zero_weight_is_not_a_holding() -> None:
    assert held(np.asarray([[0.0, 0.5, np.nan]])).tolist() == [[False, True, False]]


def test_fill_missing_prefers_the_first_argument() -> None:
    filled = fill_missing(np.asarray([[1.0, np.nan]]), np.asarray([[9.0, 9.0]]))
    assert filled.tolist() == [[1.0, 9.0]]


def test_a_sizing_multiplier_sums_to_the_held_count_not_to_one() -> None:
    measure = np.asarray([[1.0, 3.0]])
    holdings = np.asarray([[True, True]], dtype=np.bool_)
    sized = normalised_share(measure, holdings, power=1.0)
    assert sized.sum() == pytest.approx(2.0)


def test_a_negative_power_favours_the_smaller_measure() -> None:
    sized = normalised_share(
        np.asarray([[1.0, 3.0]]), np.asarray([[True, True]], dtype=np.bool_), power=-1.0
    )
    assert sized[0, 0] > sized[0, 1]


def test_inverse_variance_concentrates_harder_than_inverse_volatility() -> None:
    deviation = np.asarray([[1.0, 3.0]])
    holdings = np.asarray([[True, True]], dtype=np.bool_)
    by_vol = normalised_share(deviation, holdings, power=-1.0)
    by_variance = normalised_share(np.square(deviation), holdings, power=-1.0)
    assert by_variance[0, 0] / by_variance[0, 1] > by_vol[0, 0] / by_vol[0, 1]


def test_rescaling_a_side_makes_it_sum_to_how_many_it_holds() -> None:
    rescaled = rescaled_to_held_count(np.asarray([[1.0, 3.0, np.nan]]))
    assert np.nansum(rescaled) == pytest.approx(2.0)


def test_gross_is_the_sum_of_the_absolute_weights() -> None:
    assert gross(np.asarray([[0.5, -0.5]])).tolist() == [[1.0]]


def test_rescaling_to_a_gross_target_hits_it_exactly() -> None:
    scaled = to_gross(np.asarray([[0.1, -0.1]]), 2.0)
    assert gross(scaled).tolist() == [[2.0]]


def test_rescaling_leaves_a_row_that_holds_nothing_alone() -> None:
    scaled = to_gross(np.asarray([[0.0, 0.0]]), 1.0)
    assert scaled.tolist() == [[0.0, 0.0]]


def test_rescaling_preserves_the_ratio_between_positions() -> None:
    scaled = to_gross(np.asarray([[0.2, -0.6]]), 1.0)
    assert scaled[0, 1] / scaled[0, 0] == pytest.approx(-3.0)


def test_a_non_positive_gross_target_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        to_gross(np.asarray([[1.0]]), 0.0)


def test_centering_forces_zero_net_exposure() -> None:
    centred = center(np.asarray([[0.6, 0.2, 0.2]]))
    assert centred.sum() == pytest.approx(0.0)


def test_centering_ignores_a_gap_when_computing_the_mean() -> None:
    centred = center(np.asarray([[1.0, 3.0, np.nan]]))
    assert centred[0, 0] == pytest.approx(-1.0)
    assert np.isnan(centred[0, 2])


def test_a_cap_binds_on_both_sides() -> None:
    capped = clip_allocation(np.asarray([[0.9, -0.9, 0.05]]), 0.1)
    assert capped.tolist() == [[0.1, -0.1, 0.05]]


def test_a_non_positive_cap_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        clip_allocation(np.asarray([[1.0]]), 0.0)


def test_quantising_rounds_to_the_lot_size() -> None:
    quantized = quantize(np.asarray([[0.117, -0.123]]), 0.05)
    assert quantized[0].tolist() == pytest.approx([0.1, -0.1])


def test_a_non_positive_lot_size_is_refused() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        quantize(np.asarray([[1.0]]), 0.0)


def test_a_tilt_moves_net_exposure_by_its_own_size() -> None:
    tilted = tilt(np.asarray([[0.5, -0.5]]), 0.2)
    assert tilted.sum() == pytest.approx(0.2)


def test_a_tilt_leaves_a_name_that_is_not_held_alone() -> None:
    tilted = tilt(np.asarray([[0.5, 0.0]]), 0.2)
    assert tilted[0, 1] == 0.0


def test_a_tilt_on_an_empty_row_changes_nothing() -> None:
    assert tilt(np.asarray([[0.0, 0.0]]), 0.2).tolist() == [[0.0, 0.0]]


def test_bins_become_a_long_short_book_of_gross_one() -> None:
    weights = weights_from_bins(np.asarray([[0.0, 1.0, 2.0]]))
    assert np.nansum(np.abs(weights)) == pytest.approx(1.0)
    assert weights[0, 0] < 0.0 < weights[0, 2]


def test_the_middle_bin_carries_no_position() -> None:
    weights = weights_from_bins(np.asarray([[0.0, 1.0, 2.0]]))
    assert weights[0, 1] == pytest.approx(0.0)
