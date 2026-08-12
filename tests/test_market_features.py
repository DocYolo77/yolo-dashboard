"""
Tests for scripts/build_market_features.py: Universe-Filter (Asset-Type/ETF-
Ausschluss), ADR-Filter, Market-Cap-Filter, ATR-Extension-Formel (SMA50-
basiert), RS-Perzentile. All tests use synthetic data — no network / no
MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_market_features import (  # noqa: E402
    calc_ticker_features, calc_true_range, compute_eligibility, compute_eligible_universe,
    eligible_percentile_ranks, type_eligible_universe,
    calc_sma50_trend_fields, renormalized_weighted_sum, clamp_0_100,
    compute_recent_leader_bootstrap, load_price_cache, save_price_cache,
    PRICE_CACHE_SCHEMA_VERSION,
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
    # n_days=75: calc_ticker_features' V1.1 min_history gate is
    # 51 + sma50_slope_lookback (default 20) = 71, so 65 is no longer enough.
    close, high, low = make_ohlc(n_days=75, daily_range_pct=6.0, drift_pct=0.0)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)
    # (High/Low - 1) * 100 == 6.0 exactly by construction, independent of price level.
    assert out["TEST"]["adr20"] == pytest.approx(6.0, abs=0.01)


def test_atr_extension_matches_user_supplied_sma50_formula():
    # Strong uptrend -> close well above SMA50 -> positive ATR Extension.
    # n_days=75: see min_history note in test_adr20_matches_high_low_ratio_formula.
    close, high, low = make_ohlc(n_days=75, base=100.0, daily_range_pct=4.0, drift_pct=0.8)
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
    close, high, low = make_ohlc(n_days=75, base=100.0, daily_range_pct=4.0, drift_pct=-0.8)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]
    assert out["atr_extension"] < 0


def test_ema10_ema20_distance_signs():
    close, high, low = make_ohlc(n_days=75, drift_pct=0.5)  # uptrend
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


# ── V1.1: SMA50 trend-strength anchor (slope + persistence) ─────

def test_calc_sma50_trend_fields_insufficient_history_returns_none():
    close = pd.Series([100.0] * 40)  # too short for a 50-window SMA at all
    sma50_series = close.rolling(50).mean()
    slope, pct_above = calc_sma50_trend_fields(close, sma50_series, slope_lookback=20, persistence_lookback=20)
    assert slope is None and pct_above is None


def test_calc_sma50_trend_fields_matches_manual_slope_and_persistence():
    # Flat at 100 for 60 days (SMA50 stabilizes at 100), then a clean +1%/day
    # run for 20 more days -> SMA50 rises steadily and close stays above it
    # the whole time -> persistence should be 100%, slope clearly positive.
    flat = [100.0] * 60
    trend = [100.0 * (1.01 ** i) for i in range(1, 21)]
    close = pd.Series(flat + trend)
    sma50_series = close.rolling(50).mean()

    slope, pct_above = calc_sma50_trend_fields(close, sma50_series, slope_lookback=20, persistence_lookback=20)

    expected_slope = round(float((sma50_series.iloc[-1] / sma50_series.iloc[-21] - 1) * 100), 2)
    assert slope == pytest.approx(expected_slope, abs=0.01)
    assert slope > 0  # SMA50 rising
    assert pct_above == 100.0  # close never dips below SMA50 in this construction


def test_calc_sma50_trend_fields_persistence_below_100_when_price_dips_under_sma50():
    # Uptrend overall, but the LAST 20 sessions include a stretch where
    # close sits below the (still-lagging) SMA50 -> persistence < 100%.
    # 60 flat days (so SMA50 has fully rolled off the initial NaN warm-up by
    # the time the last-20 persistence window starts) + a 10-day dip.
    flat = [100.0] * 60
    dip = [95.0] * 10  # sudden drop, below the SMA50 that formed on the flat period
    close = pd.Series(flat + dip)
    sma50_series = close.rolling(50).mean()
    slope, pct_above = calc_sma50_trend_fields(close, sma50_series, slope_lookback=20, persistence_lookback=20)
    assert pct_above is not None
    assert pct_above < 100.0


# ── V1.1: calc_ticker_features new fields (multi-timeframe returns, SMA50 anchor) ──

def test_calc_ticker_features_v1_1_fields_present():
    close, high, low = make_ohlc(n_days=280, drift_pct=0.2)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]

    assert out["sma50_distance_pct"] == out["gain_from_sma50_pct"]  # V1.1 alias, same value
    assert out["return_3m"] is not None
    assert out["return_6m"] is not None
    assert out["return_12m"] is not None
    assert out["sma50_slope_20d_pct"] is not None
    assert out["pct_sessions_above_sma50_20d"] is not None
    # steady uptrend -> multi-timeframe returns should all be positive
    assert out["return_3m"] > 0 and out["return_6m"] > 0 and out["return_12m"] > 0


def test_calc_ticker_features_min_history_gate_includes_slope_buffer():
    # min_history = max(adr_lookback, 51 + max(slope_lookback, persistence_lookback))
    # = 51 + 20 = 71 with the default 20/20 lookbacks -> 70 days must be
    # skipped, 71 days must produce output (V1.1 point 5/6 dependency).
    close_short, high_short, low_short = make_ohlc(n_days=70)
    out_short = calc_ticker_features(
        pd.DataFrame({"TEST": close_short}), pd.DataFrame({"TEST": high_short}), pd.DataFrame({"TEST": low_short}),
        adr_lookback=20)
    assert "TEST" not in out_short

    close_ok, high_ok, low_ok = make_ohlc(n_days=71)
    out_ok = calc_ticker_features(
        pd.DataFrame({"TEST": close_ok}), pd.DataFrame({"TEST": high_ok}), pd.DataFrame({"TEST": low_ok}),
        adr_lookback=20)
    assert "TEST" in out_ok


# ── V1.1: renormalized_weighted_sum / clamp_0_100 (structural_rs / trend_strength primitives) ──

def test_renormalized_weighted_sum_full_data():
    result = renormalized_weighted_sum({"a": 80.0, "b": 40.0}, {"a": 0.5, "b": 0.5})
    assert result == pytest.approx(60.0)


def test_renormalized_weighted_sum_renormalizes_when_a_component_missing():
    # weights sum to 1 over {a, b, c}=0.2/0.3/0.5; with c missing, a/b should
    # be renormalized to sum to 1 (0.2/0.5, 0.3/0.5) rather than treating c as 0.
    result = renormalized_weighted_sum({"a": 100.0, "b": 0.0, "c": None}, {"a": 0.2, "b": 0.3, "c": 0.5})
    expected = (100.0 * 0.2 + 0.0 * 0.3) / 0.5
    assert result == pytest.approx(expected)


def test_renormalized_weighted_sum_all_missing_returns_none():
    assert renormalized_weighted_sum({"a": None, "b": None}, {"a": 0.5, "b": 0.5}) is None


def test_clamp_0_100_bounds_and_rounds():
    assert clamp_0_100(None) is None
    assert clamp_0_100(-5.0) == 0.0
    assert clamp_0_100(105.0) == 100.0
    assert clamp_0_100(42.345) == 42.3


# ── V1.1: Recent Leader bootstrap reconstruction ─────────────────

def test_compute_recent_leader_bootstrap_flags_structural_leader_not_laggard():
    n = 300
    dates = [f"d{i}" for i in range(n)]
    steady_up = pd.Series([100.0 * (1.003 ** i) for i in range(n)], index=dates)
    steady_down = pd.Series([100.0 * (0.999 ** i) for i in range(n)], index=dates)
    flat = pd.Series([100.0] * n, index=dates)
    close_df = pd.DataFrame({"LEAD": steady_up, "LAG": steady_down, "FLAT1": flat, "FLAT2": flat + 0.01})

    eligible = {sym: True for sym in close_df.columns}
    structural_weights = {"rs_1m": 0.20, "rs_3m": 0.35, "rs_6m": 0.30, "rs_12m": 0.15}
    trend_weights = {"slope_percentile": 0.60, "persistence": 0.40}
    leader_entry_cfg = {"structural_rs_min": 85, "trend_strength_min": 70}

    result = compute_recent_leader_bootstrap(
        close_df, eligible, structural_weights, trend_weights,
        slope_lookback=20, persistence_lookback=20, memory_sessions=15, leader_entry_cfg=leader_entry_cfg)

    assert result["LEAD"] is True
    assert result["LAG"] is False


def test_compute_recent_leader_bootstrap_ignores_non_eligible_columns():
    n = 300
    dates = [f"d{i}" for i in range(n)]
    steady_up = pd.Series([100.0 * (1.003 ** i) for i in range(n)], index=dates)
    close_df = pd.DataFrame({"LEAD": steady_up, "EXCLUDED": steady_up})
    eligible = {"LEAD": True, "EXCLUDED": False}
    structural_weights = {"rs_1m": 0.20, "rs_3m": 0.35, "rs_6m": 0.30, "rs_12m": 0.15}
    trend_weights = {"slope_percentile": 0.60, "persistence": 0.40}
    leader_entry_cfg = {"structural_rs_min": 85, "trend_strength_min": 70}

    result = compute_recent_leader_bootstrap(
        close_df, eligible, structural_weights, trend_weights,
        slope_lookback=20, persistence_lookback=20, memory_sessions=15, leader_entry_cfg=leader_entry_cfg)
    assert "EXCLUDED" not in result


# ── V1.1: persistent price-history cache round-trip ──────────────

def test_price_cache_round_trip(tmp_path):
    cache_path = tmp_path / "market_history.json"
    trading_days = ["2026-01-01", "2026-01-02"]
    per_ticker_close = {"AAA": {"2026-01-01": 10.0, "2026-01-02": 11.0}}
    per_ticker_high = {"AAA": {"2026-01-01": 10.5, "2026-01-02": 11.5}}
    per_ticker_low = {"AAA": {"2026-01-01": 9.5, "2026-01-02": 10.5}}

    assert load_price_cache(cache_path) is None  # nothing written yet

    save_price_cache(cache_path, trading_days, per_ticker_close, per_ticker_high, per_ticker_low, {"AAA"})
    loaded = load_price_cache(cache_path)

    assert loaded["schema_version"] == PRICE_CACHE_SCHEMA_VERSION
    assert loaded["dates"] == trading_days
    assert loaded["tickers"]["AAA"]["close"] == [10.0, 11.0]


def test_price_cache_skips_tickers_with_no_observed_data(tmp_path):
    cache_path = tmp_path / "market_history.json"
    trading_days = ["2026-01-01"]
    save_price_cache(cache_path, trading_days, {}, {}, {}, {"NEVER_SEEN"})
    loaded = load_price_cache(cache_path)
    assert "NEVER_SEEN" not in loaded["tickers"]  # all-null row -> not persisted


def test_price_cache_discards_mismatched_schema_version(tmp_path):
    cache_path = tmp_path / "market_history.json"
    cache_path.write_text(json.dumps({"schema_version": PRICE_CACHE_SCHEMA_VERSION + 1, "dates": [], "tickers": {}}))
    assert load_price_cache(cache_path) is None  # stale schema -> treated as cold cache
