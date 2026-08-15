"""
Tests for scripts/build_narratives.py's V6 RSP-based Narrative Strength/
Thrust ("Jeff-inspired Relative Strength gegen RSP" / "Jeff-inspired
Thrust candidate v1" — explicitly candidate formulas, NOT a reproduction
of Jeff Sun's exact unpublished formula). Worked examples taken directly
from the spec so the exact arithmetic is pinned down, not just directional
sign checks. All synthetic data — no network required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import (  # noqa: E402
    compute_narrative_equal_weight_return_series,
    build_synthetic_narrative_index,
    compute_relative_strength_line,
    percentile_rank_of_current,
    strength_percentile_at,
    compute_strength_windows_rsp,
    compute_narrative_thrust_rsp,
    build_benchmark_rsp_series,
)


def make_days(n):
    return [f"2026-01-{d:02d}" if d <= 31 else f"2026-02-{d - 31:02d}" for d in range(1, n + 1)]


# ── point 10: equal-weight MEAN daily return (not median, not cap-weighted) ──

def test_equal_weight_mean_not_median():
    # A=+4%, B=0% -> equal-weight MEAN = +2% (the median of two values would
    # ALSO be 2% here by coincidence for n=2, so use 3 members to actually
    # distinguish mean from median).
    days = make_days(3)
    daily_ret = pd.DataFrame(
        {"A": [None, 4.0, 4.0], "B": [None, 0.0, 0.0], "C": [None, 1.0, 1.0]}, index=days)
    series = compute_narrative_equal_weight_return_series(daily_ret, ["A", "B", "C"])
    # mean(4,0,1) = 1.667, median(4,0,1) = 1.0 -- must match the MEAN.
    assert series.iloc[1] == pytest.approx((4.0 + 0.0 + 1.0) / 3, abs=1e-9)


def test_two_member_worked_example_from_spec():
    # Spec worked example: A=+4%, B=0% -> narrative_return = +2%.
    days = make_days(2)
    daily_ret = pd.DataFrame({"A": [None, 4.0], "B": [None, 0.0]}, index=days)
    series = compute_narrative_equal_weight_return_series(daily_ret, ["A", "B"])
    assert series.iloc[1] == pytest.approx(2.0)


def test_missing_member_excluded_from_just_that_day_not_fabricated():
    # A member missing its return on a given day is excluded from that
    # day's mean only -- never treated as 0%.
    days = make_days(2)
    daily_ret = pd.DataFrame({"A": [None, 10.0], "B": [None, None]}, index=days)
    series = compute_narrative_equal_weight_return_series(daily_ret, ["A", "B"])
    assert series.iloc[1] == pytest.approx(10.0)  # NOT (10+0)/2=5.0


def test_day_with_no_valid_member_return_yields_nan_never_zero():
    days = make_days(2)
    daily_ret = pd.DataFrame({"A": [None, None], "B": [None, None]}, index=days)
    series = compute_narrative_equal_weight_return_series(daily_ret, ["A", "B"])
    assert pd.isna(series.iloc[1])


# ── point 10: synthetic index compounding, starting at 100 ──

def test_synthetic_index_worked_example_from_spec():
    # +2% then -1%, starting at 100 -> 102, then 100.98.
    days = make_days(3)
    returns = pd.Series([None, 2.0, -1.0], index=days)
    idx = build_synthetic_narrative_index(returns)
    assert idx.iloc[0] == pytest.approx(102.0)
    assert idx.iloc[1] == pytest.approx(100.98)


def test_synthetic_index_drops_nan_days_never_forward_fills():
    days = make_days(4)
    returns = pd.Series([None, 2.0, None, 2.0], index=days)  # day 2 has no valid narrative return
    idx = build_synthetic_narrative_index(returns)
    assert len(idx) == 2  # only the two valid-return days survive, the NaN day is dropped entirely
    assert idx.iloc[0] == pytest.approx(102.0)
    assert idx.iloc[1] == pytest.approx(102.0 * 1.02)  # compounds from the LAST VALID point, no gap-fill


def test_synthetic_index_empty_when_no_valid_returns():
    days = make_days(2)
    returns = pd.Series([None, None], index=days)
    idx = build_synthetic_narrative_index(returns)
    assert idx.empty


# ── point 11 step 2: relative-strength line = narrative_close / benchmark_close ──

def test_relative_strength_line_worked_examples_from_spec():
    days = make_days(2)
    narrative_index = pd.Series([105.0, 106.0], index=days)
    benchmark_close = pd.Series([100.0, 104.0], index=days)
    rel = compute_relative_strength_line(narrative_index, benchmark_close)
    assert rel.iloc[0] == pytest.approx(1.05)
    assert rel.iloc[1] == pytest.approx(106.0 / 104.0, abs=1e-5)  # ~1.01923


def test_relative_strength_line_uses_the_line_not_the_raw_return():
    # Explicit regression: Strength must be computed on the ratio LINE, not
    # directly on the narrative's own daily return -- a narrative that rose
    # LESS than the benchmark on day 2 (106/104=1.019 < 105/100=1.05) must
    # show a DECLINING relative-strength line even though its own price
    # still went up in absolute terms.
    days = make_days(2)
    narrative_index = pd.Series([105.0, 106.0], index=days)
    benchmark_close = pd.Series([100.0, 104.0], index=days)
    rel = compute_relative_strength_line(narrative_index, benchmark_close)
    assert rel.iloc[1] < rel.iloc[0]


def test_relative_strength_line_drops_dates_missing_on_either_side():
    days = make_days(3)
    narrative_index = pd.Series([105.0, 106.0, 107.0], index=days)
    benchmark_close = pd.Series([100.0, None, 104.0], index=days)
    rel = compute_relative_strength_line(narrative_index, benchmark_close)
    assert len(rel) == 2
    assert days[1] not in rel.index


# ── point 11 step 3: percentile rank of the CURRENT value in its own window,
# reusing the repo's canonical percentile_ranks() sort-position convention ──

def test_percentile_rank_top_of_window_worked_example_from_spec():
    window = [1.00, 1.01, 0.99, 1.02, 1.03]  # current = 1.03, the max
    assert percentile_rank_of_current(window) == 100.0


def test_percentile_rank_bottom_of_window_mirrored_example():
    window = [1.03, 1.02, 1.01, 1.00, 0.99]  # current = 0.99, the min of 5
    assert percentile_rank_of_current(window) == 20.0  # 1/5 * 100, same convention as percentile_ranks()


def test_percentile_rank_empty_window_is_none():
    assert percentile_rank_of_current([]) is None


# ── point 11: 1W/1M/3M/6M session counts, never calendar days ──

def test_strength_windows_use_exact_session_counts_not_calendar_days():
    days = make_days(130)
    # Build a monotonically increasing relative-strength line so the
    # CURRENT value is always the max of any trailing window -> Strength=100
    # regardless of window size, PROVING each window actually pulled exactly
    # its configured session count (not some other/calendar-based length) --
    # if a window secretly used the wrong count it would still return 100
    # here, so pin down window lengths on a NON-monotonic tail instead.
    rel = pd.Series([100.0 + i for i in range(126)] + [50.0], index=days[:127])
    # last value (50.0) is the lowest of the whole series -> lowest percentile
    # in EVERY window that includes it, for every configured window length.
    windows = {"1w": 5, "1m": 20, "3m": 63, "6m": 126}
    result = compute_strength_windows_rsp(rel, windows)
    # percentile_rank_of_current reuses percentile_ranks()'s round(..., 1)
    # convention, so compare against the same rounded expectation.
    assert result["1w"] == round(1 / 5 * 100, 1)
    assert result["1m"] == round(1 / 20 * 100, 1)
    assert result["3m"] == round(1 / 63 * 100, 1)
    assert result["6m"] == round(1 / 126 * 100, 1)


def test_strength_window_none_when_insufficient_history():
    days = make_days(10)
    rel = pd.Series([100.0 + i for i in range(10)], index=days)
    result = compute_strength_windows_rsp(rel, {"1w": 5, "1m": 20, "3m": 63, "6m": 126})
    assert result["1w"] is not None
    assert result["1m"] is None
    assert result["3m"] is None
    assert result["6m"] is None


def test_timeframes_are_fully_separate_no_averaging_no_composite():
    # Two different relative-strength lines that agree on the last 5
    # sessions but differ everywhere else must produce identical Strength_1W
    # but potentially different Strength_1M/3M/6M -- proving 1W's percentile
    # rank does not leak into or get blended with the other windows.
    days = make_days(30)
    tail = [10.0, 10.1, 9.9, 10.2, 10.3]  # identical last 5 sessions
    rel_a = pd.Series([100.0] * 25 + tail, index=days)
    rel_b = pd.Series([1.0] * 25 + tail, index=days)
    windows = {"1w": 5, "1m": 20}
    result_a = compute_strength_windows_rsp(rel_a, windows)
    result_b = compute_strength_windows_rsp(rel_b, windows)
    assert result_a["1w"] == result_b["1w"]  # same 5-session tail -> same Strength_1W
    assert result_a["1m"] != result_b["1m"]  # different 20-session history -> different Strength_1M


def test_strength_percentile_at_sessions_ago_uses_real_sessions_not_calendar_days():
    days = make_days(10)
    rel = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=days)
    # "Today" (sessions_ago=0): window = last 5 real sessions [6,7,8,9,10] -> current=10 is max -> 100.
    assert strength_percentile_at(rel, window=5, sessions_ago=0) == 100.0
    # 3 REAL sessions ago: window ends 3 positions back = [3,4,5,6,7] -> current(of that window)=7 is max -> 100.
    assert strength_percentile_at(rel, window=5, sessions_ago=3) == 100.0
    # 3 sessions ago vs today must reference genuinely different windows.
    rel2 = pd.Series([5.0, 4.0, 3.0, 2.0, 1.0, 10.0, 9.0, 8.0, 7.0, 6.0], index=days)
    today = strength_percentile_at(rel2, window=5, sessions_ago=0)   # window [10,9,8,7,6], current=6 -> min of 5 -> 20
    ago = strength_percentile_at(rel2, window=5, sessions_ago=3)     # window [3,2,1,10,9], current=9 -> 4th of 5 -> 80
    # today and ago land on genuinely different, correctly-offset windows
    # (proves sessions_ago actually shifts the reference point, not a no-op).
    assert today == 20.0
    assert ago == 80.0


# ── regression: no outperformance-day binary / win-share model ──

def test_relative_strength_reflects_magnitude_not_binary_win_day_share():
    # Two narratives both beat RSP EVERY single day (5/5 win days, i.e. an
    # identical "outperformance-day share" of 100%) but by very different
    # margins. A mean(outperform_t)/win-day-share style score would rank
    # them IDENTICALLY; the real relative-strength-line formula must not --
    # it must reflect the actual compounded magnitude of outperformance.
    days = make_days(6)
    rsp_ret = pd.Series([None] + [0.1] * 5, index=days)
    small_edge = pd.Series([None] + [0.11] * 5, index=days)   # barely beats RSP every day
    big_edge = pd.Series([None] + [1.0] * 5, index=days)      # crushes RSP every day

    def relative_strength_line(narrative_ret):
        idx = build_synthetic_narrative_index(narrative_ret)
        rsp_idx = build_synthetic_narrative_index(rsp_ret)
        return compute_relative_strength_line(idx, rsp_idx)

    rel_small = relative_strength_line(small_edge)
    rel_big = relative_strength_line(big_edge)
    # Both "win" every day, but the big-edge narrative's relative-strength
    # line must have moved MUCH further from 1.0 than the small-edge one --
    # proving magnitude (not a binary win/loss count) drives the number.
    small_move = abs(rel_small.iloc[-1] - 1.0)
    big_move = abs(rel_big.iloc[-1] - 1.0)
    assert big_move > small_move * 5


# ── point 12: Thrust — worked example, non-clamping, missing-input rules ──

def test_thrust_worked_example_from_spec():
    # Strength_1W=80, Strength_1M=70, Strength_1W(3 sessions ago)=60
    # -> 0.60*80 + 0.40*70 + 0.10*(80-60) = 48 + 28 + 2 = 78
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    thrust = compute_narrative_thrust_rsp(80, 70, 60, weights)
    assert thrust == pytest.approx(78.0)


def test_thrust_positive_acceleration_pushes_thrust_up():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    accelerating = compute_narrative_thrust_rsp(80, 70, 60, weights)   # +20 acceleration
    flat = compute_narrative_thrust_rsp(80, 70, 80, weights)           # 0 acceleration
    assert accelerating > flat


def test_thrust_negative_acceleration_pulls_thrust_down():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    decelerating = compute_narrative_thrust_rsp(80, 70, 95, weights)   # negative acceleration
    flat = compute_narrative_thrust_rsp(80, 70, 80, weights)
    assert decelerating < flat


def test_thrust_can_exceed_100_not_clamped():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    thrust = compute_narrative_thrust_rsp(100, 100, 0, weights)  # 60+40+10 = 110
    assert thrust == pytest.approx(110.0)
    assert thrust > 100


def test_thrust_can_go_negative_not_clamped():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    thrust = compute_narrative_thrust_rsp(0, 0, 100, weights)  # 0+0+0.10*(0-100) = -10
    assert thrust == pytest.approx(-10.0)
    assert thrust < 0


def test_thrust_none_when_strength_1m_missing():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    assert compute_narrative_thrust_rsp(80, None, 60, weights) is None


def test_thrust_none_when_strength_1w_today_missing():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    assert compute_narrative_thrust_rsp(None, 70, 60, weights) is None


def test_thrust_none_when_strength_1w_n_sessions_ago_missing():
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    assert compute_narrative_thrust_rsp(80, 70, None, weights) is None


# ── point 29B/29C: single canonical RSP close/date series ──

def test_benchmark_rsp_series_ticker_and_shape():
    days = make_days(5)
    prices = pd.DataFrame({"RSP": [170.0, 171.0, 172.5, None, 173.0], "AAPL": [1, 2, 3, 4, 5]}, index=days)
    result = build_benchmark_rsp_series(prices, "RSP")
    assert result["ticker"] == "RSP"
    assert len(result["dates"]) == 4  # the None entry is dropped, never passed to the renderer
    assert len(result["dates"]) == len(result["close"])
    assert all(v is not None for v in result["close"])


def test_benchmark_rsp_series_none_when_ticker_absent():
    days = make_days(3)
    prices = pd.DataFrame({"AAPL": [1, 2, 3]}, index=days)
    assert build_benchmark_rsp_series(prices, "RSP") is None


def test_benchmark_rsp_series_dates_are_sorted_ascending():
    days = make_days(5)
    prices = pd.DataFrame({"RSP": [170.0, 171.0, 172.0, 173.0, 174.0]}, index=days)
    result = build_benchmark_rsp_series(prices, "RSP")
    assert result["dates"] == sorted(result["dates"])
