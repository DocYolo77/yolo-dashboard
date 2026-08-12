"""
Tests for scripts/build_narratives.py's V1.1 Structural Narrative Engine
primitives: Trend Participation, Structural Leadership %, the Momentum
Modifier, and the shared price-history cache reader. All tests use
synthetic data — no network / no MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import (  # noqa: E402
    renormalized_weighted_sum, clamp_0_100,
    calc_trend_participation, calc_structural_leadership_pct, calc_momentum_modifier,
    load_shared_price_cache_frame, SHARED_CACHE_MIN_TRADING_DAYS,
)
from build_market_features import PRICE_CACHE_SCHEMA_VERSION  # noqa: E402


# ── renormalized_weighted_sum / clamp_0_100 (local copies, same contract) ──

def test_renormalized_weighted_sum_matches_full_weighted_average():
    assert renormalized_weighted_sum({"a": 80.0, "b": 40.0}, {"a": 0.5, "b": 0.5}) == pytest.approx(60.0)


def test_renormalized_weighted_sum_all_missing_returns_none():
    assert renormalized_weighted_sum({"a": None}, {"a": 1.0}) is None


def test_clamp_0_100_bounds():
    assert clamp_0_100(-1.0) == 0.0
    assert clamp_0_100(101.0) == 100.0
    assert clamp_0_100(None) is None


# ── Trend Participation ──────────────────────────────────────────

def test_trend_participation_all_above_and_rising():
    members = ["A", "B", "C"]
    market_features = {
        "A": {"sma50_distance_pct": 5.0, "sma50_slope_20d_pct": 1.0},
        "B": {"sma50_distance_pct": 2.0, "sma50_slope_20d_pct": 0.5},
        "C": {"sma50_distance_pct": 10.0, "sma50_slope_20d_pct": 3.0},
    }
    above, rising, both = calc_trend_participation(members, market_features)
    assert above == 100.0 and rising == 100.0 and both == 100.0


def test_trend_participation_mixed_signals():
    members = ["A", "B", "C", "D"]
    market_features = {
        "A": {"sma50_distance_pct": 5.0, "sma50_slope_20d_pct": 1.0},   # above + rising
        "B": {"sma50_distance_pct": -2.0, "sma50_slope_20d_pct": 1.0},  # below but rising
        "C": {"sma50_distance_pct": 5.0, "sma50_slope_20d_pct": -1.0},  # above but falling
        "D": {"sma50_distance_pct": -2.0, "sma50_slope_20d_pct": -1.0},  # neither
    }
    above, rising, both = calc_trend_participation(members, market_features)
    assert above == 50.0
    assert rising == 50.0
    assert both == 25.0  # only A qualifies for both simultaneously


def test_trend_participation_excludes_members_without_data():
    members = ["A", "NO_DATA"]
    market_features = {"A": {"sma50_distance_pct": 5.0, "sma50_slope_20d_pct": 1.0}}
    above, rising, both = calc_trend_participation(members, market_features)
    assert above == 100.0  # NO_DATA excluded from denominator, not counted as failing


def test_trend_participation_no_data_returns_all_none():
    assert calc_trend_participation(["X"], {}) == (None, None, None)


# ── Structural Leadership % ──────────────────────────────────────

def test_structural_leadership_pct_counts_at_or_above_threshold():
    members = ["A", "B", "C"]
    market_features = {
        "A": {"structural_rs": 90.0},
        "B": {"structural_rs": 80.0},  # inclusive at threshold
        "C": {"structural_rs": 79.9},
    }
    assert calc_structural_leadership_pct(members, market_features, threshold=80) == pytest.approx(66.7, abs=0.1)


def test_structural_leadership_pct_no_data_returns_none():
    assert calc_structural_leadership_pct(["X"], {}, threshold=80) is None


# ── Momentum Modifier ─────────────────────────────────────────────

MODIFIER_CFG = {
    "accelerating": {"thrust_1w_positive_required": True, "thrust_percentile_1w_min": 75},
    "cooling": {"structural_score_min": 65, "thrust_1w_negative_required": True},
}


def test_momentum_modifier_accelerating():
    result = calc_momentum_modifier(thrust_1w=1.5, thrust_percentile_1w=80, structural_score=50,
                                     modifier_cfg=MODIFIER_CFG)
    assert result == "ACCELERATING"


def test_momentum_modifier_cooling():
    result = calc_momentum_modifier(thrust_1w=-0.5, thrust_percentile_1w=20, structural_score=70,
                                     modifier_cfg=MODIFIER_CFG)
    assert result == "COOLING"


def test_momentum_modifier_none_when_neither_condition_met():
    result = calc_momentum_modifier(thrust_1w=0.1, thrust_percentile_1w=50, structural_score=50,
                                     modifier_cfg=MODIFIER_CFG)
    assert result is None


def test_momentum_modifier_cooling_requires_high_structural_score():
    # Negative thrust alone isn't COOLING -- a structurally weak narrative
    # fading further is just weak, not "cooling off a strong trend".
    result = calc_momentum_modifier(thrust_1w=-1.0, thrust_percentile_1w=10, structural_score=40,
                                     modifier_cfg=MODIFIER_CFG)
    assert result is None


# ── Shared price-history cache reader ─────────────────────────────

def _write_cache(path, n_days, tickers):
    dates = [f"d{i:04d}" for i in range(n_days)]
    data = {"schema_version": PRICE_CACHE_SCHEMA_VERSION, "dates": dates,
            "tickers": {sym: {"close": [100.0 + i for i in range(n_days)], "high": [], "low": []}
                        for sym in tickers}}
    path.write_text(json.dumps(data))
    return dates


def test_load_shared_price_cache_frame_missing_file_returns_none(tmp_path):
    close_df, dates = load_shared_price_cache_frame(tmp_path / "nope.json", {"AAPL"})
    assert close_df is None and dates is None


def test_load_shared_price_cache_frame_too_short_returns_none(tmp_path):
    cache_path = tmp_path / "cache.json"
    _write_cache(cache_path, SHARED_CACHE_MIN_TRADING_DAYS - 1, {"AAPL"})
    close_df, dates = load_shared_price_cache_frame(cache_path, {"AAPL"})
    assert close_df is None and dates is None


def test_load_shared_price_cache_frame_returns_restricted_frame(tmp_path):
    cache_path = tmp_path / "cache.json"
    expected_dates = _write_cache(cache_path, SHARED_CACHE_MIN_TRADING_DAYS, {"AAPL", "TSLA", "NOT_NEEDED"})
    close_df, dates = load_shared_price_cache_frame(cache_path, {"AAPL", "TSLA"})
    assert dates == expected_dates
    assert set(close_df.columns) == {"AAPL", "TSLA"}
    assert "NOT_NEEDED" not in close_df.columns
    assert len(close_df) == SHARED_CACHE_MIN_TRADING_DAYS
