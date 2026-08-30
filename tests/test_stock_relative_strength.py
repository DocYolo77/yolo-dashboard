"""
Tests for scripts/build_market_features.py's compute_stock_relative_strength_pct
(Calibration-aware Opportunities UI v1, "Relative Strength"): generalizes
build_narratives.py's Narrative RS methodology (compute_relative_strength_line
+ compute_narrative_rs, REUSED unchanged) to individual stocks vs. RSP,
instead of inventing a second formula. All synthetic data — no network / no
file I/O required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_market_features import compute_stock_relative_strength_pct  # noqa: E402

WINDOWS = {"1w": 5, "1m": 20, "3m": 63, "6m": 126}


def make_close_df(n_days=140):
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B")
    rsp = pd.Series([100.0 + i * 0.05 for i in range(n_days)], index=idx)  # mild steady uptrend
    strong = pd.Series([100.0 + i * 0.5 for i in range(n_days)], index=idx)  # far outpaces RSP
    weak = pd.Series([100.0 - i * 0.2 for i in range(n_days)], index=idx)  # underperforms RSP
    return pd.DataFrame({"RSP": rsp, "STRONG": strong, "WEAK": weak})


def test_strong_outperformer_ranks_above_weak_underperformer_on_every_window():
    close_df = make_close_df()
    eligible = {"STRONG": True, "WEAK": True}
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    for label in WINDOWS:
        assert result[label]["STRONG"] > result[label]["WEAK"]


def test_all_four_windows_present_in_result():
    close_df = make_close_df()
    eligible = {"STRONG": True, "WEAK": True}
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    assert set(result.keys()) == set(WINDOWS.keys())


def test_ineligible_stock_excluded_from_ranking_and_from_result():
    close_df = make_close_df()
    eligible = {"STRONG": True, "WEAK": False}  # WEAK not eligible
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    for label in WINDOWS:
        assert "WEAK" not in result[label]
        assert "STRONG" in result[label]


def test_missing_benchmark_column_returns_empty_dicts_for_every_window_not_a_crash():
    close_df = make_close_df().drop(columns=["RSP"])
    eligible = {"STRONG": True, "WEAK": True}
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    assert result == {label: {} for label in WINDOWS}


def test_benchmark_ticker_itself_never_appears_in_its_own_ranking():
    close_df = make_close_df()
    eligible = {"STRONG": True, "WEAK": True, "RSP": True}  # even if mistakenly marked eligible
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    for label in WINDOWS:
        assert "RSP" not in result[label]


def test_insufficient_history_for_a_window_yields_no_entry_not_an_improvised_value():
    # Only 10 sessions total -- the 1m/3m/6m windows (20/63/126) can never
    # produce a value; 1w (5) can.
    idx = pd.date_range("2025-01-01", periods=10, freq="B")
    close_df = pd.DataFrame({
        "RSP": pd.Series([100.0 + i * 0.05 for i in range(10)], index=idx),
        "STRONG": pd.Series([100.0 + i * 0.5 for i in range(10)], index=idx),
    })
    eligible = {"STRONG": True}
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    assert "STRONG" in result["1w"]
    assert "STRONG" not in result["1m"]
    assert "STRONG" not in result["3m"]
    assert "STRONG" not in result["6m"]


def test_percentiles_are_0_to_100():
    close_df = make_close_df()
    eligible = {"STRONG": True, "WEAK": True}
    result = compute_stock_relative_strength_pct(close_df, eligible, "RSP", WINDOWS)
    for label in WINDOWS:
        for v in result[label].values():
            assert 0.0 <= v <= 100.0
