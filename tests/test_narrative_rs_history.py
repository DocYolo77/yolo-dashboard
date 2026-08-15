"""
Tests for scripts/build_narratives.compute_narrative_rs_history (RVOL/
Screener/Benchmark/Futures Patch point 10): the multi-narrative-vs-RSP
relative-strength time series behind the restored old pill-comparison
Benchmark chart. Reuses the SAME relative_strength_by_id lines
(narrative_index_t / rsp_close_t) that drive the headline narrative_rs
percentiles above -- this test module only exercises the chart-windowing /
"line always starts at 0%" transform on top of that, never a second
calculation model. Benchmark stays RSP, never SPY (spec point 9's explicit
constraint). All tests use synthetic pandas data — no network / no
MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import compute_narrative_rs_history  # noqa: E402


def make_days(n):
    return [f"2026-01-{d:02d}" if d <= 31 else f"2026-02-{d-31:02d}" for d in range(1, n + 1)]


def test_no_trading_days_returns_none():
    assert compute_narrative_rs_history([{"id": "n1"}], {"n1": pd.Series([1.0], index=["2026-01-01"])}, []) is None


def test_no_narrative_has_relative_strength_data_returns_none():
    days = make_days(20)
    narratives = [{"id": "n1"}, {"id": "n2"}]
    assert compute_narrative_rs_history(narratives, {}, days) is None


def test_narrative_without_data_is_skipped_not_crashed():
    days = make_days(15)
    rel = pd.Series([1.0 + 0.01 * i for i in range(15)], index=days)
    narratives = [{"id": "has_data"}, {"id": "no_data"}]
    result = compute_narrative_rs_history(narratives, {"has_data": rel}, days, lookback_days=15)
    assert "has_data" in result["narratives"]
    assert "no_data" not in result["narratives"]


def test_narrative_with_fewer_than_10_valid_points_is_skipped():
    days = make_days(20)
    # Only 5 valid points in the 20-day window (rest NaN) -> below the
    # 10-point minimum this function has always required.
    values = [None] * 15 + [1.0, 1.01, 1.02, 1.03, 1.04]
    rel = pd.Series(values, index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=20)
    assert result is None


def test_chart_line_always_starts_at_zero_percent():
    days = make_days(20)
    rel = pd.Series([1.0 + 0.02 * i for i in range(20)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=20)
    series = result["narratives"]["n1"]
    assert series[0] == 0.0


def test_outperformance_yields_positive_chart_values():
    days = make_days(20)
    # Rising narrative_index/rsp_close ratio -> the narrative is
    # outperforming RSP since the chart's own visible start -> positive %.
    rel = pd.Series([1.0 + 0.02 * i for i in range(20)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=20)
    series = result["narratives"]["n1"]
    assert series[-1] > 0
    assert all(b >= a - 1e-9 for a, b in zip(series, series[1:]))


def test_underperformance_yields_negative_chart_values():
    days = make_days(20)
    rel = pd.Series([1.0 - 0.01 * i for i in range(20)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=20)
    series = result["narratives"]["n1"]
    assert series[-1] < 0


def test_flat_relative_strength_yields_zero_throughout():
    days = make_days(15)
    rel = pd.Series([1.234] * 15, index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=15)
    series = result["narratives"]["n1"]
    assert all(abs(v) < 1e-9 for v in series)


def test_lookback_trims_to_requested_window():
    days = make_days(40)
    rel = pd.Series([1.0 + 0.01 * i for i in range(40)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=10)
    assert result["lookback_trading_days"] == 10
    assert len(result["dates"]) == 10
    assert len(result["narratives"]["n1"]) == 10


def test_benchmark_label_defaults_to_rsp_not_spy():
    days = make_days(15)
    rel = pd.Series([1.0 + 0.01 * i for i in range(15)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=15)
    assert result["benchmark"] == "RSP"


def test_benchmark_label_is_configurable_but_never_defaults_to_spy():
    days = make_days(15)
    rel = pd.Series([1.0 + 0.01 * i for i in range(15)], index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=15, benchmark_ticker="QQQ")
    assert result["benchmark"] == "QQQ"
    assert result["benchmark"] != "SPY"


def test_gaps_within_window_preserved_as_null_not_fabricated():
    # A date inside the window where this narrative's relative_strength line
    # has no value (e.g. a day with no valid narrative-member return, see
    # build_synthetic_narrative_index's own gap-dropping contract) must
    # surface as null in the chart series, never interpolated/forward-filled.
    days = make_days(12)
    values = [1.0 + 0.01 * i for i in range(12)]
    values[5] = None
    rel = pd.Series(values, index=days)
    result = compute_narrative_rs_history([{"id": "n1"}], {"n1": rel}, days, lookback_days=12)
    series = result["narratives"]["n1"]
    assert series[5] is None
    assert series[0] == 0.0
