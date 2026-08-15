"""
Tests for scripts/build_narratives.py's RSP-based Narrative Strength/Thrust
("Jeff-inspired Relative Strength gegen RSP" / "Jeff-inspired Thrust
candidate v1" — explicitly candidate formulas, NOT a reproduction of Jeff
Sun's exact unpublished formula). Worked examples taken directly from the
spec so the exact arithmetic is pinned down, not just directional sign
checks. All synthetic data — no network required.

V6.1 (Narrative Ranking & UI Bugfix Patch) REPLACED the V6 self-window
percentile methodology (percentile_rank_of_current/strength_percentile_at/
compute_strength_windows_rsp, all REMOVED) with cross-sectional ranking
(relative_performance_at/cross_sectional_percentile_ranks/compute_narrative_rs)
— see point 5's bug report: ranking a narrative only against its own trailing
1W/1M/3M/6M history collapsed onto a handful of discrete values (20/40/60/80/
100 for a 5-observation 1W window) and let multiple narratives tie at 100
simultaneously. The Thrust formula itself (compute_narrative_thrust_rsp) is
UNCHANGED — only its inputs changed from self-window Strength to
cross-sectional Narrative RS.
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
    relative_performance_at,
    cross_sectional_percentile_ranks,
    compute_narrative_rs,
    compute_narrative_thrust_rsp,
    build_benchmark_rsp_series,
    find_healthcare_biotech_leaks,
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


# ── point 11 step 2 / 6.3: relative-strength line = narrative_close / benchmark_close ──

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


# ── point 6.3/7: raw relative_performance_at, ratio-based, mathematically
# identical to (1+narrative_return)/(1+rsp_return)-1 ──

def test_relative_performance_at_worked_example_from_spec():
    # Spec section 22-B: Narrative 1W = +10%, RSP 1W = +5%
    # -> relative_performance = 1.10/1.05 - 1 ~= 4.7619%.
    days = make_days(2)
    narrative_index = pd.Series([100.0, 110.0], index=days)
    benchmark_close = pd.Series([100.0, 105.0], index=days)
    rel = compute_relative_strength_line(narrative_index, benchmark_close)
    perf = relative_performance_at(rel, window=1, sessions_ago=0)
    assert perf == pytest.approx(1.10 / 1.05 - 1, abs=1e-6)
    assert perf == pytest.approx(0.047619, abs=1e-5)


def test_relative_performance_at_none_when_insufficient_history():
    days = make_days(4)
    rel = pd.Series([1.0, 1.01, 1.02, 1.03], index=days)
    assert relative_performance_at(rel, window=5, sessions_ago=0) is None


def test_relative_performance_at_sessions_ago_shifts_the_reference_point():
    days = make_days(10)
    rel = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=days)
    # today (sessions_ago=0), window=5: current=10 (idx9), base=5 (idx4) -> 10/5-1 = 1.0
    today = relative_performance_at(rel, window=5, sessions_ago=0)
    assert today == pytest.approx(1.0)
    # 3 sessions ago, window=5: current=7 (idx6), base=2 (idx1) -> 7/2-1 = 2.5
    ago = relative_performance_at(rel, window=5, sessions_ago=3)
    assert ago == pytest.approx(2.5)
    assert today != ago  # proves sessions_ago actually shifts the window, not a no-op


# ── point 7: cross-sectional percentile ranking (replaces self-window) ──

def test_cross_sectional_percentile_ranks_worked_example():
    # Spec section 22-C: A=+8%, B=+4%, C=+1%, D=-2% -> A > B > C > D.
    values = {"A": 0.08, "B": 0.04, "C": 0.01, "D": -0.02}
    ranks = cross_sectional_percentile_ranks(values)
    assert ranks["A"] > ranks["B"] > ranks["C"] > ranks["D"]
    assert ranks["A"] == 100.0  # highest of 4 -> top percentile
    assert ranks["D"] == 25.0   # lowest of 4 -> 1/4 * 100


def test_cross_sectional_percentile_ranks_ties_get_identical_rank():
    # Spec section 22-E: exactly equal relative_performance -> exactly equal RS.
    values = {"A": 0.05, "B": 0.05, "C": 0.01}
    ranks = cross_sectional_percentile_ranks(values)
    assert ranks["A"] == ranks["B"]
    assert ranks["A"] > ranks["C"]


def test_cross_sectional_percentile_ranks_tie_is_order_independent():
    # Same tie, members supplied in a different dict insertion order -> same
    # ranks (average-rank method, not a stable-sort position).
    values_1 = {"A": 0.05, "B": 0.05, "C": 0.01}
    values_2 = {"C": 0.01, "B": 0.05, "A": 0.05}
    ranks_1 = cross_sectional_percentile_ranks(values_1)
    ranks_2 = cross_sectional_percentile_ranks(values_2)
    assert ranks_1["A"] == ranks_2["A"] == ranks_1["B"] == ranks_2["B"]


def test_cross_sectional_percentile_ranks_excludes_none_from_pool():
    values = {"A": 0.08, "B": None, "C": 0.01}
    ranks = cross_sectional_percentile_ranks(values)
    assert "B" not in ranks
    assert set(ranks.keys()) == {"A", "C"}
    assert ranks["A"] == 100.0
    assert ranks["C"] == 50.0  # 1 of 2 -> bottom half, avg-rank pct


def test_cross_sectional_percentile_ranks_empty_when_all_none():
    assert cross_sectional_percentile_ranks({"A": None, "B": None}) == {}


def test_no_20_40_60_80_100_bucket_ceiling_with_many_active_narratives():
    # Spec section 22-D: with e.g. 46 active narratives, 1W RS must not be
    # limited to five discrete values -- this was the actual production bug
    # (V6's self-window percentile_rank_of_current on a 5-observation window
    # could only ever produce one of 20/40/60/80/100). Cross-sectional
    # ranking over N narratives naturally produces up to N distinct values.
    n = 46
    values = {f"narrative_{i}": i * 0.001 for i in range(n)}  # all distinct
    ranks = cross_sectional_percentile_ranks(values)
    assert len(set(ranks.values())) == n  # every narrative gets its own distinct rank


# ── point 7-8: compute_narrative_rs — fully separate 1W/1M/3M/6M, no composite ──

def test_compute_narrative_rs_windows_are_fully_separate_no_averaging():
    days = make_days(30)
    # relative_performance_at is a RATIO (current/base - 1), so what must be
    # shared between A and B for an IDENTICAL 1W result is the pair (base,
    # current) at the 1W reference points -- not just "the same-looking
    # trailing values". window=5, sessions_ago=0 -> base=index 24, current=
    # index 29 (both series' last index). window=20 -> base=index 9, SAME
    # current=index 29. Sharing indices 24..29 verbatim but diverging at
    # index 9 gives identical 1W and different 1M, by construction.
    tail = [10.0, 10.0, 10.0, 10.0, 10.0, 11.0]  # indices 24..29, shared
    prefix_a = [1.0] * 9 + [5.0] + [1.0] * 14  # index 9 = 5.0
    prefix_b = [1.0] * 9 + [8.0] + [1.0] * 14  # index 9 = 8.0
    rel_a = pd.Series(prefix_a + tail, index=days)
    rel_b = pd.Series(prefix_b + tail, index=days)
    windows = {"1w": 5, "1m": 20}
    narrative_rs, relative_performance = compute_narrative_rs({"A": rel_a, "B": rel_b}, windows)
    # identical (base, current) pair at the 1W reference point -> identical raw relative_performance
    assert relative_performance["1w"]["A"] == pytest.approx(relative_performance["1w"]["B"])
    # cross-sectional over just {A, B} with equal 1W values -> tied RS
    assert narrative_rs["1w"]["A"] == narrative_rs["1w"]["B"]
    # different 20-session base -> different 1M relative_performance/RS
    assert relative_performance["1m"]["A"] != relative_performance["1m"]["B"]
    assert narrative_rs["1m"]["A"] != narrative_rs["1m"]["B"]


def test_compute_narrative_rs_none_when_window_insufficient():
    days = make_days(10)
    rel = pd.Series([100.0 + i for i in range(10)], index=days)
    narrative_rs, _ = compute_narrative_rs({"A": rel}, {"1w": 5, "1m": 20, "3m": 63, "6m": 126})
    assert narrative_rs["1w"].get("A") is not None
    assert narrative_rs["1m"].get("A") is None
    assert narrative_rs["3m"].get("A") is None
    assert narrative_rs["6m"].get("A") is None


def test_compute_narrative_rs_sessions_ago_reconstructs_historical_rs():
    # Point 9: RS1W as of 3 sessions ago must come from the SAME
    # cross-sectional method, evaluated at an earlier reference point.
    days = make_days(15)
    rel_a = pd.Series([1.0 + i * 0.1 for i in range(15)], index=days)   # steadily rising
    rel_b = pd.Series([1.0 - i * 0.05 for i in range(15)], index=days)  # steadily falling
    today_rs, _ = compute_narrative_rs({"A": rel_a, "B": rel_b}, {"1w": 5}, sessions_ago=0)
    ago_rs, _ = compute_narrative_rs({"A": rel_a, "B": rel_b}, {"1w": 5}, sessions_ago=3)
    # A always outperforms B at every reference point (rising vs falling lines).
    assert today_rs["1w"]["A"] > today_rs["1w"]["B"]
    assert ago_rs["1w"]["A"] > ago_rs["1w"]["B"]


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


# ── point 12: Thrust — worked example, non-clamping, missing-input rules
# (formula UNCHANGED in V6.1, only its inputs changed to cross-sectional RS) ──

def test_thrust_worked_example_from_spec():
    # RS_1W=80, RS_1M=70, RS_1W(3 sessions ago)=60
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


# ── point 18-19: Healthcare/Biotech Sanity Gate ──

# NOTE: "life sciences" deliberately excluded -- collides with the real,
# already-validated-not-Healthcare/Biotech narrative name "Life Sciences
# Tools & Consumables" (see config's own comment + the regression test below).
AUDIT_KEYWORDS = ["biotech", "biotechnology", "biopharma", "pharma", "pharmaceutical",
                   "healthcare", "medical device", "diagnostic", "therapeutic"]


def _members(symbols):
    return [{"symbol": s} for s in symbols]


def test_sanity_gate_flags_narrative_named_biotech_with_enough_members():
    # Point 18's real example: an active "Biotech" narrative with >= 5
    # eligible members must be caught by name alone, even if no individual
    # member's SIC/description happens to match.
    output_narratives = [{"id": "biotech", "name": "Biotech", "members": _members(["A", "B", "C", "D", "E"])}]
    violations = find_healthcare_biotech_leaks(output_narratives, {}, AUDIT_KEYWORDS, min_active_members=5)
    assert len(violations) == 1
    assert violations[0]["narrative_name"] == "Biotech"
    assert "biotech" in violations[0]["name_matched_keywords"]


def test_sanity_gate_below_minimum_active_members_is_not_flagged():
    # Same name, but under the active-size gate -> not flagged (point 19
    # explicitly ties the check to the '>= 5 eligible members' bar).
    output_narratives = [{"id": "biotech", "name": "Biotech", "members": _members(["A", "B", "C"])}]
    violations = find_healthcare_biotech_leaks(output_narratives, {}, AUDIT_KEYWORDS, min_active_members=5)
    assert violations == []


def test_sanity_gate_flags_content_concentrated_narrative_even_with_neutral_name():
    # A neutrally-named narrative whose eligible members are OVERWHELMINGLY
    # (majority) Healthcare/Biotech by sic_description must also be caught --
    # defense in depth beyond just the narrative's name.
    market_features = {
        "A": {"sic_description": "PHARMACEUTICAL PREPARATIONS"},
        "B": {"sic_description": "BIOLOGICAL PRODUCTS, NO DIAGNOSTIC SUBSTANCES"},
        "C": {"sic_description": "IN VITRO & IN VIVO DIAGNOSTIC SUBSTANCES"},
        "D": {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"},
        "E": {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"},
    }
    output_narratives = [{"id": "n1", "name": "Growth Innovators", "members": _members(["A", "B", "C", "D", "E"])}]
    violations = find_healthcare_biotech_leaks(output_narratives, market_features, AUDIT_KEYWORDS, min_active_members=5)
    assert len(violations) == 1
    assert violations[0]["flagged_member_count"] == 3  # A, B, C
    assert violations[0]["name_matched_keywords"] == []


def test_sanity_gate_ignores_company_description_customer_vertical_false_positive():
    # Regression for the real production false positive this gate caused:
    # laboratory-instrument/testing-lab companies (AVTR/BRKR/NEO-shaped)
    # whose OWN sic_description does NOT match, but whose free-text
    # company_description mentions serving healthcare/biopharma/life-
    # sciences CUSTOMERS, must NOT flag the narrative -- company_description
    # is deliberately excluded from this gate's signal (sic_description only).
    market_features = {
        "AVTR": {"sic_code": "3826", "sic_description": "LABORATORY ANALYTICAL INSTRUMENTS",
                  "company_description": "Provides products and services to customers in the biopharma, "
                                          "healthcare, education & government, and advanced technologies industries."},
        "BRKR": {"sic_code": "3826", "sic_description": "LABORATORY ANALYTICAL INSTRUMENTS",
                  "company_description": "Manufactures scientific instruments and diagnostic tests for customers "
                                          "in the life sciences, applied markets, pharmaceutical, and biotechnology industries."},
        "NEO": {"sic_code": "8734", "sic_description": "SERVICES-TESTING LABORATORIES",
                 "company_description": "Provides oncology diagnostic testing and consultative services."},
        "D": {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"},
        "E": {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"},
    }
    output_narratives = [{"id": "life_sciences_tools_consumables", "name": "Life Sciences Tools & Consumables",
                           "members": _members(["AVTR", "BRKR", "NEO", "D", "E"])}]
    violations = find_healthcare_biotech_leaks(output_narratives, market_features, AUDIT_KEYWORDS, min_active_members=5)
    assert violations == []


def test_sanity_gate_clean_narrative_is_not_flagged():
    market_features = {s: {"sic_description": "SEMICONDUCTORS & RELATED DEVICES"} for s in "ABCDE"}
    output_narratives = [{"id": "n1", "name": "Semiconductors", "members": _members(list("ABCDE"))}]
    violations = find_healthcare_biotech_leaks(output_narratives, market_features, AUDIT_KEYWORDS, min_active_members=5)
    assert violations == []


def test_sanity_gate_minority_healthcare_members_alone_do_not_trigger():
    # Only 1 of 5 members hits an audit keyword (20%, well under the
    # majority bar) and the narrative's own name is neutral -> not flagged.
    market_features = {
        "A": {"sic_description": "PHARMACEUTICAL PREPARATIONS"},
        "B": {"sic_description": "SEMICONDUCTORS"}, "C": {"sic_description": "SEMICONDUCTORS"},
        "D": {"sic_description": "SEMICONDUCTORS"}, "E": {"sic_description": "SEMICONDUCTORS"},
    }
    output_narratives = [{"id": "n1", "name": "Growth Innovators", "members": _members(["A", "B", "C", "D", "E"])}]
    violations = find_healthcare_biotech_leaks(output_narratives, market_features, AUDIT_KEYWORDS, min_active_members=5)
    assert violations == []
