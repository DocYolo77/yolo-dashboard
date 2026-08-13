"""
Tests for scripts/build_ticker_charts.py — the hover-mini-chart data
builder (per-ticker OHLC + EMA10/EMA20/SMA50/SMA200 windows, computed
entirely from the existing rolling price-history cache, zero extra API
calls). All synthetic data — no network required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import build_ticker_charts as bc  # noqa: E402


def make_dates(n):
    return [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(n)]


def make_flat_series(n, value):
    return [value] * n


def test_compute_eligible_set_filters_on_eligible_flag():
    mf = {"tickers": {"AAA": {"eligible": True}, "BBB": {"eligible": False}, "CCC": {"eligible": True}}}
    assert bc.compute_eligible_set(mf) == {"AAA", "CCC"}


def test_compute_ticker_chart_returns_none_below_min_bars():
    dates = make_dates(3)
    close = [10.0, 10.1, None]
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=60)
    assert result is None


def test_compute_ticker_chart_full_history_populates_sma200_across_whole_window():
    # 260 days of clean data -> the last 60 (WINDOW_DAYS) must have SMA200
    # populated for EVERY point, not just the tail -- this is the entire
    # reason WINDOW_DAYS=60 was chosen (260 - 200 = 60).
    n = 260
    dates = make_dates(n)
    close = [100.0 + 0.01 * i for i in range(n)]  # smooth, no gaps
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=60)
    assert result is not None
    assert len(result["dates"]) == 60
    assert all(v is not None for v in result["sma200"])
    assert all(v is not None for v in result["sma50"])
    assert all(v is not None for v in result["ema10"])
    assert all(v is not None for v in result["ema20"])


def test_compute_ticker_chart_short_history_leaves_sma200_null_not_fabricated():
    n = 40  # well under 200 -> SMA200 can never be valid anywhere
    dates = make_dates(n)
    close = [50.0 + 0.1 * i for i in range(n)]
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=60)
    assert result is not None
    assert len(result["dates"]) == 40  # can't have a 60-day window with only 40 days of history
    assert all(v is None for v in result["sma200"])
    assert any(v is not None for v in result["ema10"])  # EMA10 needs far less warm-up


def test_compute_ticker_chart_drops_leading_none_before_computing_indicators():
    # A ticker with 250 days of leading gaps (not yet trading) followed by
    # 20 real days must NOT let those leading Nones corrupt the EMA/SMA
    # warm-up -- dropna() first, same convention as calc_ticker_features.
    n = 270
    dates = make_dates(n)
    close = [None] * 250 + [20.0 + 0.05 * i for i in range(20)]
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=60)
    assert result is not None
    assert len(result["dates"]) == 20
    assert result["dates"][0] == dates[250]
    assert result["c"][0] == 20.0


def test_compute_ticker_chart_window_uses_tickers_own_trading_dates_not_market_dates():
    n = 100
    dates = make_dates(n)
    close = [None] * 50 + [30.0 + i for i in range(50)]
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=20)
    assert result["dates"] == dates[80:100]  # last 20 of this ticker's own (post-dropna) 50 trading days


def test_build_all_charts_only_includes_requested_tickers_present_in_cache():
    # build_all_charts trusts its `eligible_tickers` argument (main() derives
    # that from compute_eligible_set beforehand) -- its own job is only to
    # skip tickers that aren't requested at all, or have no cache entry.
    dates = make_dates(60)
    close = [10.0 + i for i in range(60)]
    cache = {
        "dates": dates,
        "tickers": {
            "AAA": {"close": close, "high": close, "low": close, "open": close},
            "NOT_REQUESTED": {"close": close, "high": close, "low": close, "open": close},
        },
    }
    charts = bc.build_all_charts({"AAA", "MISSING_FROM_CACHE"}, cache)
    assert set(charts.keys()) == {"AAA"}


def test_ema_sma_conventions_match_the_rest_of_the_dashboard():
    # Same call shape as build_market_features.calc_ticker_features
    # (close.ewm(span=n).mean() / close.rolling(n).mean()) -- spot-check
    # against a hand-computable flat series where EMA/SMA both converge to
    # the flat value everywhere it's defined.
    n = 260
    dates = make_dates(n)
    close = make_flat_series(n, 42.0)
    result = bc.compute_ticker_chart(dates, close, close, close, close, window_days=60)
    assert all(v == 42.0 for v in result["ema10"])
    assert all(v == 42.0 for v in result["sma50"])
    assert all(v == 42.0 for v in result["sma200"])
