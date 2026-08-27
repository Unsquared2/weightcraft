"""Properties that must hold for any input, not just the ones we thought of."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from conftest import panel_frame
from weightcraft.align import align
from weightcraft.combine import nanmean_stack, normalised_shares
from weightcraft.frame import WeightFrame
from weightcraft.normalize import gross, to_gross, weights_from_bins

if TYPE_CHECKING:
    from weightcraft.arrays import Matrix, Vector

# NaN and inf are generated deliberately: a gap is the ordinary case here, and
# the reductions disagreed about infinities until a review caught it.
_CELLS = st.one_of(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, width=32),
    st.just(np.nan),
    st.sampled_from([np.inf, -np.inf]),
)
_PANEL = arrays(
    np.float64, array_shapes(min_dims=2, max_dims=2, max_side=5), elements=_CELLS
)
_FINITE_PANEL = arrays(
    np.float64,
    array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=5),
    elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, width=32),
)


@given(values=_PANEL)
@settings(max_examples=200)
def test_rescaling_to_a_gross_target_reaches_it_or_holds_nothing(
    values: Matrix,
) -> None:
    scaled = to_gross(values, 1.0)
    reached = gross(scaled)[:, 0]
    scalable = np.isfinite(gross(values)[:, 0]) & (gross(values)[:, 0] > 0.0)
    assert np.allclose(reached[scalable], 1.0)


@given(values=_PANEL)
@settings(max_examples=200)
def test_ensembling_one_source_returns_that_source_unchanged(values: Matrix) -> None:
    frame = panel_frame(values)
    combined = nanmean_stack(align([frame]).values)
    # An infinity is missing to every reduction, so it comes back as a gap.
    expected = np.where(np.isfinite(values), values, np.nan)
    assert np.array_equal(combined, expected, equal_nan=True)


@given(values=_PANEL, other=_PANEL)
@settings(max_examples=200)
def test_ensembling_does_not_depend_on_the_order_the_sources_arrive_in(
    values: Matrix, other: Matrix
) -> None:
    # Deliberately *not* truncated to a shared shape: the union logic in
    # `align` is the part worth testing, and equal shapes never exercise it.
    left = panel_frame(values, "2026-01-01")
    right = panel_frame(other, "2026-01-03")
    forward = nanmean_stack(align([left, right]).values)
    backward = nanmean_stack(align([right, left]).values)
    assert np.array_equal(forward, backward, equal_nan=True)


@given(values=_PANEL, other=_PANEL)
@settings(max_examples=200)
def test_the_union_covers_every_date_and_asset_either_source_carried(
    values: Matrix, other: Matrix
) -> None:
    left = panel_frame(values, "2026-01-01")
    right = panel_frame(other, "2026-02-01")
    stack = align([left, right])
    assert set(stack.assets) == set(left.assets) | set(right.assets)
    assert set(stack.dates.tolist()) == set(left.dates.tolist()) | set(
        right.dates.tolist()
    )


@given(values=_FINITE_PANEL)
@settings(max_examples=200)
def test_a_bin_row_is_of_gross_one_and_ranks_the_way_its_bins_do(
    values: Matrix,
) -> None:
    """Gross one, and monotone in the bin -- but not net-zero.

    Net exposure depends on how the names are spread across the bins, and
    balancing that is `center`'s job, not this function's.
    """
    weights = weights_from_bins(values)
    for source, row in zip(values, weights, strict=True):
        total = float(np.nansum(np.abs(row)))
        assert total == pytest.approx(0.0) or total == pytest.approx(1.0)
        order = np.argsort(source, kind="stable")
        ranked = row[order]
        assert np.all(np.diff(ranked) >= -1e-12)


@given(
    raw=arrays(
        np.float64,
        array_shapes(min_dims=1, max_dims=1, min_side=1, max_side=8),
        elements=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, width=32),
    )
)
@settings(max_examples=200)
def test_normalised_shares_always_sum_to_one(raw: Vector) -> None:
    assert float(normalised_shares(raw).sum()) == pytest.approx(1.0)


@given(values=_FINITE_PANEL)
@settings(max_examples=100)
def test_a_polars_round_trip_never_changes_a_frame(values: Matrix) -> None:
    original = panel_frame(values)
    assert WeightFrame.from_polars(original.to_polars()) == original
