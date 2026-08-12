"""
Tests for scripts/build_market_features.py: Universe-Filter (Asset-Type/ETF-
Ausschluss), ADR-Filter, Market-Cap-Filter, ATR-Extension-Formel (SMA50-
basiert), RS-Perzentile. All tests use synthetic data — no network / no
MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_market_features import (  # noqa: E402
    calc_ticker_features, calc_true_range, compute_eligibility, compute_eligible_universe,
    eligible_percentile_ranks, type_eligible_universe,
)
from build_narratives import percentile_ranks  # noqa: E402


# ── Universe filter / ETF exclusion ─────────────────────────────

EXCLUDED_TYPES = {"ETF", "ETN", "ETS", "FUND", "PFD", "WARRANT", "RIGHT", "UNIT", "INDEX"}


def test_type_eligible_universe_excludes_etf_etn_fund():
    types_ref = {"tickers": {
        "AAPL": {"type": "CS"},
        "SMH": {"type": "ETF"},
        "UVXY": {"type": "ETN"},
        "PFF": {"type": "PFD"},
        "TSLA": {"type": "CS"},
    }}
    universe = type_eligible_universe(types_ref, EXCLUDED_TYPES)
    assert universe == {"AAPL", "TSLA"}


def test_type_eligible_universe_keeps_adr_common_stock():
    types_ref = {"tickers": {"BABA": {"type": "ADRC"}, "TSM": {"type": "ADRC"}, "SMH": {"type": "ETF"}}}
    universe = type_eligible_universe(types_ref, EXCLUDED_TYPES)
    assert universe == {"BABA", "TSM"}  # ADR common stock is regular equity, not excluded


def test_type_eligible_universe_handles_missing_type():
    types_ref = {"tickers": {"XYZ": {"type": None}}}
    universe = type_eligible_universe(types_ref, EXCLUDED_TYPES)
    assert universe == set()  # unknown type is conservatively excluded, not assumed eligible


# ── Market Cap / ADR eligibility ────────────────────────────────

@pytest.fixture
def universe_cfg():
    return {"adr_lookback_sessions": 20, "adr_minimum_pct": 4.0, "market_cap_minimum_usd": 1_000_000_000}


def test_eligible_requires_both_adr_and_market_cap(universe_cfg):
    assert compute_eligibility(adr20=5.0, market_cap=2_000_000_000, universe_cfg=universe_cfg) is True
    assert compute_eligibility(adr20=3.0, market_cap=2_000_000_000, universe_cfg=universe_cfg) is False  # ADR too low
    assert compute_eligibility(adr20=5.0, market_cap=500_000_000, universe_cfg=universe_cfg) is False  # cap too low
    assert compute_eligibility(adr20=None, market_cap=2_000_000_000, universe_cfg=universe_cfg) is False  # missing ADR
    assert compute_eligibility(adr20=5.0, market_cap=None, universe_cfg=universe_cfg) is False  # missing cap


def test_adr_exactly_at_threshold_is_not_eligible(universe_cfg):
    # Spec: "ADR20 > 4%" is strict-greater-than, not >=.
    assert compute_eligibility(adr20=4.0, market_cap=2_000_000_000, universe_cfg=universe_cfg) is False


def test_market_cap_exactly_at_threshold_is_eligible(universe_cfg):
    # Spec: "Market Cap >= 1B" is inclusive.
    assert compute_eligibility(adr20=5.0, market_cap=1_000_000_000, universe_cfg=universe_cfg) is True


# ── ADR20 / EMA / SMA50 / ATR / ATR-Extension formulas ──────────

def make_ohlc(n_days=65, base=100.0, daily_range_pct=6.0, drift_pct=0.0, seed=0):
    """Synthetic OHLC: Low = price, High = price*(1+daily_range_pct/100), so
    (High/Low - 1)*100 == daily_range_pct EXACTLY regardless of price level —
    matches the ADR20 formula (High/Low ratio based) bit-for-bit. Plus a
    linear drift so SMA50/EMA differ from the last close."""
    dates = pd.date_range("2026-01-01", periods=n_days, freq="B").strftime("%Y-%m-%d")
    closes, highs, lows = [], [], []
    price = base
    for i in range(n_days):
        price = price * (1 + drift_pct / 100.0)
        low = price
        high = price * (1 + daily_range_pct / 100.0)
        closes.append((low + high) / 2)
        highs.append(high)
        lows.append(low)
    close = pd.Series(closes, index=dates)
    high = pd.Series(highs, index=dates)
    low = pd.Series(lows, index=dates)
    return close, high, low


def test_adr20_matches_high_low_ratio_formula():
    close, high, low = make_ohlc(n_days=65, daily_range_pct=6.0, drift_pct=0.0)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)
    # (High/Low - 1) * 100 == 6.0 exactly by construction, independent of price level.
    assert out["TEST"]["adr20"] == pytest.approx(6.0, abs=0.01)


def test_atr_extension_matches_user_supplied_sma50_formula():
    # Strong uptrend -> close well above SMA50 -> positive ATR Extension.
    close, high, low = make_ohlc(n_days=65, base=100.0, daily_range_pct=4.0, drift_pct=0.8)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]

    last_close = close.iloc[-1]
    sma50 = close.iloc[-50:].mean()
    atr = out["atr"]
    # Manually recompute A = ATR% = ATR/Price, B = %Gain-50MA, Extension = B/A
    a_atr_pct = atr / last_close * 100.0
    b_gain_pct = (last_close - sma50) / sma50 * 100.0
    expected_extension = b_gain_pct / a_atr_pct

    assert out["sma50"] == pytest.approx(sma50, abs=0.02)  # rounded to 2 decimals by calc_ticker_features
    assert out["gain_from_sma50_pct"] == pytest.approx(b_gain_pct, rel=1e-3)
    assert out["atr_extension"] == pytest.approx(expected_extension, rel=1e-2)
    assert out["atr_extension"] > 0  # uptrend -> positive extension above the 50-MA


def test_atr_extension_negative_when_below_sma50():
    close, high, low = make_ohlc(n_days=65, base=100.0, daily_range_pct=4.0, drift_pct=-0.8)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]
    assert out["atr_extension"] < 0


def test_ema10_ema20_distance_signs():
    close, high, low = make_ohlc(n_days=65, drift_pct=0.5)  # uptrend
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]
    # In a steady uptrend, price sits above both fast EMAs.
    assert out["ema10_distance_pct"] > 0
    assert out["ema20_distance_pct"] > 0


def test_calc_true_range_matches_manual_formula():
    close = pd.Series([100.0, 102.0, 101.0])
    high = pd.Series([101.0, 103.0, 103.0])
    low = pd.Series([99.0, 100.0, 100.0])
    tr = calc_true_range(high, low, close)
    # day0: no prev close -> just H-L = 2.0 (prev_close NaN -> max ignores NaN comparisons oddly,
    # but day1/day2 are the meaningful ones)
    # day1: max(103-100, |103-100|, |100-100|) = max(3,3,0) = 3
    assert tr.iloc[1] == pytest.approx(3.0)
    # day2: max(103-100, |103-102|, |100-102|) = max(3,1,2) = 3
    assert tr.iloc[2] == pytest.approx(3.0)


def test_insufficient_history_is_skipped():
    close, high, low = make_ohlc(n_days=30)  # < 51 required for SMA50
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)
    assert "TEST" not in out


# ── RS Percentiles (reused across curated + full-market universe) ──

def test_percentile_ranks_basic_ordering():
    metrics = {"A": {"pct": 1.0}, "B": {"pct": 5.0}, "C": {"pct": -2.0}, "D": {"pct": 3.0}}
    ranks = percentile_ranks(metrics, "pct")
    assert ranks["C"] < ranks["A"] < ranks["D"] < ranks["B"]
    assert ranks["B"] == 100.0  # highest value -> 100th percentile
    assert ranks["C"] == 25.0   # lowest of 4 -> 1/4 * 100


def test_percentile_ranks_ignores_missing_values():
    metrics = {"A": {"pct": 1.0}, "B": {"pct": None}, "C": {"pct": 3.0}}
    ranks = percentile_ranks(metrics, "pct")
    assert "B" not in ranks
    assert set(ranks.keys()) == {"A", "C"}


def test_percentile_ranks_full_market_vs_curated_universe_differ():
    curated = {"NVDA": {"pct": 10.0}, "AMD": {"pct": 5.0}}
    full_market = {**curated, "SMALLCAP1": {"pct": 50.0}, "SMALLCAP2": {"pct": 60.0}, "SMALLCAP3": {"pct": 80.0}}
    curated_ranks = percentile_ranks(curated, "pct")
    full_ranks = percentile_ranks(full_market, "pct")
    # NVDA is top of the curated set (100th pct) but only mid-pack full-market
    # -> this is exactly the Leadership migration point 24 is about.
    assert curated_ranks["NVDA"] == 100.0
    assert full_ranks["NVDA"] < 100.0


# ── Full-Market RS fix: eligible-first percentile ordering (V1, point 7) ──

def test_eligible_universe_excludes_low_adr_and_low_market_cap():
    u_cfg = {"adr_minimum_pct": 4.0, "market_cap_minimum_usd": 1_000_000_000}
    features = {
        "BIG_HIGH_ADR": {"adr20": 6.0},   # eligible
        "BIG_LOW_ADR":  {"adr20": 2.0},   # ADR too low
        "SMALL_HIGH_ADR": {"adr20": 6.0},  # market cap too low
        "NO_CAP_DATA":  {"adr20": 6.0},   # market cap unknown -> not eligible-by-default
    }
    market_caps = {
        "BIG_HIGH_ADR": 5_000_000_000,
        "BIG_LOW_ADR": 5_000_000_000,
        "SMALL_HIGH_ADR": 500_000_000,
        "NO_CAP_DATA": None,
    }
    eligible = compute_eligible_universe(features, market_caps, u_cfg)
    assert eligible == {
        "BIG_HIGH_ADR": True,
        "BIG_LOW_ADR": False,
        "SMALL_HIGH_ADR": False,
        "NO_CAP_DATA": False,
    }


def test_non_eligible_stock_does_not_influence_rs_percentile_of_eligible_stocks():
    # A non-eligible micro-cap with an extreme return must NOT drag the
    # percentile scale for eligible stocks — the old bug ranked against
    # every ticker with a computable feature, this one included.
    features = {
        "ELIGIBLE_A": {"pct": 5.0},
        "ELIGIBLE_B": {"pct": 10.0},
        "NOT_ELIGIBLE_MOONSHOT": {"pct": 500.0},  # would dominate the top rank if included
    }
    eligible_by_symbol = {"ELIGIBLE_A": True, "ELIGIBLE_B": True, "NOT_ELIGIBLE_MOONSHOT": False}
    ranks = eligible_percentile_ranks(features, eligible_by_symbol, "pct")
    assert "NOT_ELIGIBLE_MOONSHOT" not in ranks  # excluded, not ranked at the bottom either
    assert ranks["ELIGIBLE_B"] == 100.0  # top of the ELIGIBLE-only pool
    assert ranks["ELIGIBLE_A"] < ranks["ELIGIBLE_B"]


def test_eligible_percentile_ranks_matches_plain_percentile_ranks_on_eligible_subset():
    features = {"A": {"pct": 1.0}, "B": {"pct": 5.0}, "C": {"pct": 9.0}}
    eligible_by_symbol = {"A": True, "B": True, "C": False}
    got = eligible_percentile_ranks(features, eligible_by_symbol, "pct")
    expected = percentile_ranks({"A": {"pct": 1.0}, "B": {"pct": 5.0}}, "pct")
    assert got == expected
