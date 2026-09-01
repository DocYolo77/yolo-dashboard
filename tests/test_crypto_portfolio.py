"""
Tests for scripts/build_crypto_portfolio.py: the manually-maintained YOLO
Investing crypto portfolio card (positions/allocation/avg-entry/cash_pct
come from the user via chat, config/crypto_portfolio.json; only current
prices are fetched live, via yfinance). Synthetic price fetcher only — no
network required.
Run with: pytest tests/ -v
"""

import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_crypto_portfolio import (  # noqa: E402
    calc_delta_pct,
    build_positions,
    compute_weighted_pct,
    compute_days_since_entry,
)


def test_calc_delta_pct_basic():
    assert calc_delta_pct(110.0, 100.0) == 10.0
    assert calc_delta_pct(90.0, 100.0) == -10.0


def test_calc_delta_pct_none_when_price_missing():
    assert calc_delta_pct(None, 100.0) is None


def test_calc_delta_pct_none_when_avg_entry_zero_or_missing():
    assert calc_delta_pct(100.0, 0) is None
    assert calc_delta_pct(100.0, None) is None


# ── Reproduces the user's own reference screenshot exactly, as a
# regression guard on the weighted-performance formula (verified by hand:
# 0.50*-2.08 + 0.20*-0.26 + 0.15*5.06 + 0.15*4.79 = 0.3855 ≈ 0.38/0.39,
# then *0.33 invested_pct ≈ 0.127 ≈ 0.13) ──

SCREENSHOT_CFG = {
    "entry_date": "2026-08-25",
    "cash_pct": 67,
    "positions": [
        {"symbol": "BTC-USD", "label": "BTC", "allocation_pct": 50, "avg_entry": 79679},
        {"symbol": "ETH-USD", "label": "ETH", "allocation_pct": 20, "avg_entry": 2460},
        {"symbol": "SOL-USD", "label": "SOL", "allocation_pct": 15, "avg_entry": 99.60},
        {"symbol": "HYPE32196-USD", "label": "HYPE", "allocation_pct": 15, "avg_entry": 79.60},
    ],
}
SCREENSHOT_PRICES = {
    "BTC-USD": 78021.0, "ETH-USD": 2454.0, "SOL-USD": 104.64, "HYPE32196-USD": 83.41,
}


def test_build_positions_matches_reference_deltas():
    positions = build_positions(SCREENSHOT_CFG, price_fetcher=lambda sym: SCREENSHOT_PRICES[sym])
    by_symbol = {p["symbol"]: p for p in positions}
    assert by_symbol["BTC-USD"]["delta_pct"] == pytest.approx(-2.08, abs=0.01)
    assert by_symbol["ETH-USD"]["delta_pct"] == pytest.approx(-0.24, abs=0.02)
    assert by_symbol["SOL-USD"]["delta_pct"] == pytest.approx(5.06, abs=0.01)
    assert by_symbol["HYPE32196-USD"]["delta_pct"] == pytest.approx(4.79, abs=0.01)


def test_weighted_pct_matches_reference_screenshot():
    positions = build_positions(SCREENSHOT_CFG, price_fetcher=lambda sym: SCREENSHOT_PRICES[sym])
    weighted = compute_weighted_pct(positions)
    assert weighted == pytest.approx(0.38, abs=0.03)  # reference shows "+0,38 %"


def test_weighted_pct_none_when_any_position_missing_a_price():
    positions = build_positions(SCREENSHOT_CFG, price_fetcher=lambda sym: None if sym == "ETH-USD" else 100.0)
    assert compute_weighted_pct(positions) is None


def test_weighted_pct_never_renormalizes_over_a_partial_subset():
    # Even with 3/4 positions priced, the sum still uses ALL allocation_pct
    # weights (which sum to 100 across all 4) -- a None delta must not be
    # silently dropped from the weighting, which would misrepresent the
    # true weighted return.
    cfg = {"positions": [
        {"allocation_pct": 50, "avg_entry": 100}, {"allocation_pct": 50, "avg_entry": 100},
    ]}
    positions = build_positions(
        {"positions": [{"symbol": "A", "label": "A", "allocation_pct": 50, "avg_entry": 100},
                        {"symbol": "B", "label": "B", "allocation_pct": 50, "avg_entry": 100}]},
        price_fetcher=lambda sym: 110.0 if sym == "A" else None)
    assert compute_weighted_pct(positions) is None


def test_days_since_entry_is_inclusive_matching_the_user_reference():
    # Reference: entry 25.8.2026, "Stand 01.09.2026" -> "8 Tage".
    as_of = datetime(2026, 9, 1, 19, 33, tzinfo=timezone.utc)
    assert compute_days_since_entry("2026-08-25", as_of) == 8


def test_days_since_entry_same_day_is_day_one():
    as_of = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    assert compute_days_since_entry("2026-08-25", as_of) == 1


def test_build_positions_current_price_none_on_fetch_failure():
    positions = build_positions(SCREENSHOT_CFG, price_fetcher=lambda sym: None)
    assert all(p["current_price"] is None and p["delta_pct"] is None for p in positions)
