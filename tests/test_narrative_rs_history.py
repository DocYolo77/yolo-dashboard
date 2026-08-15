"""
Tests for scripts/build_narratives.compute_narrative_rs_history: the
basket-vs-benchmark relative-strength time series behind the dashboard's
"Benchmark" comparison chart. V6: the benchmark defaults to RSP (S&P 500
Equal Weight ETF), replacing the old SPY default — explicitly NOT SPY, NOT
QQQ (spec point 29B/29C). All tests use synthetic pandas data — no
network / no MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import compute_narrative_rs_history  # noqa: E402


def make_daily_ret(days, columns_pct):
    """days: list of date strings (asc). columns_pct: {symbol: [daily % returns]}
    (first day is NaN, as pct_change() would produce)."""
    df = pd.DataFrame(index=days, columns=list(columns_pct.keys()), dtype=float)
    for sym, rets in columns_pct.items():
        for i, day in enumerate(days):
            df.at[day, sym] = rets[i]
    return df


def make_days(n):
    return [f"2026-01-{d:02d}" if d <= 31 else f"2026-02-{d-31:02d}" for d in range(1, n + 1)]


def test_missing_rsp_returns_none():
    days = make_days(20)
    daily_ret = make_daily_ret(days, {"AAPL": [None] + [0.1] * 19})
    narratives = [{"id": "n1", "tickers": ["AAPL"]}]
    assert compute_narrative_rs_history(narratives, daily_ret, days) is None


def test_insufficient_rsp_history_returns_none():
    days = make_days(20)
    # Only 5 non-NaN RSP values -> below the 10-day minimum.
    rsp = [None] * 15 + [0.1] * 5
    daily_ret = make_daily_ret(days, {"RSP": rsp, "AAPL": [0.1] * 20})
    narratives = [{"id": "n1", "tickers": ["AAPL"]}]
    assert compute_narrative_rs_history(narratives, daily_ret, days) is None


def test_default_benchmark_ticker_is_rsp_not_spy():
    # V6: RSP replaces SPY as the default benchmark — a frame that only has
    # a SPY column (no RSP) must be treated as "benchmark missing", not
    # silently fall back to SPY.
    days = make_days(20)
    daily_ret = make_daily_ret(days, {"SPY": [None] + [0.1] * 19, "AAPL": [None] + [0.1] * 19})
    narratives = [{"id": "n1", "tickers": ["AAPL"]}]
    assert compute_narrative_rs_history(narratives, daily_ret, days) is None


def test_basket_outperforming_rsp_yields_positive_relative_strength():
    days = make_days(20)
    rsp = [None] + [0.1] * 19       # RSP: +0.1%/day
    aapl = [None] + [0.5] * 19      # basket member: +0.5%/day, outperforms
    daily_ret = make_daily_ret(days, {"RSP": rsp, "AAPL": aapl})
    narratives = [{"id": "outperformer", "tickers": ["AAPL"]}]

    result = compute_narrative_rs_history(narratives, daily_ret, days, lookback_days=20)
    assert result is not None
    assert result["benchmark"] == "RSP"
    assert result["lookback_trading_days"] == 20
    assert len(result["dates"]) == 20

    series = result["narratives"]["outperformer"]
    assert len(series) == 20
    # First day: basket_ret and rsp_ret both NaN -> fillna(0) -> 0% relative.
    assert series[0] == 0.0
    # Basket compounds faster than RSP every day -> relative strength should
    # be positive and strictly increasing.
    assert series[-1] > 0
    assert all(b >= a - 1e-9 for a, b in zip(series, series[1:]))


def test_basket_matching_rsp_yields_zero_relative_strength():
    days = make_days(15)
    rsp = [None] + [0.2] * 14
    daily_ret = make_daily_ret(days, {"RSP": rsp, "MSFT": rsp})
    narratives = [{"id": "tracker", "tickers": ["MSFT"]}]

    result = compute_narrative_rs_history(narratives, daily_ret, days, lookback_days=15)
    series = result["narratives"]["tracker"]
    assert all(abs(v) < 1e-6 for v in series)


def test_narrative_without_member_data_is_skipped_not_crashed():
    days = make_days(15)
    rsp = [None] + [0.1] * 14
    daily_ret = make_daily_ret(days, {"RSP": rsp, "AAPL": [0.1] * 15})
    narratives = [
        {"id": "has_data", "tickers": ["AAPL"]},
        {"id": "no_data", "tickers": ["GHOST_TICKER"]},
    ]
    result = compute_narrative_rs_history(narratives, daily_ret, days, lookback_days=15)
    assert "has_data" in result["narratives"]
    assert "no_data" not in result["narratives"]


def test_lookback_trims_to_requested_window():
    days = make_days(40)
    rsp = [None] + [0.05] * 39
    daily_ret = make_daily_ret(days, {"RSP": rsp, "AAPL": [0.1] * 40})
    narratives = [{"id": "n1", "tickers": ["AAPL"]}]

    result = compute_narrative_rs_history(narratives, daily_ret, days, lookback_days=10)
    assert result["lookback_trading_days"] == 10
    assert len(result["dates"]) == 10
    assert len(result["narratives"]["n1"]) == 10


def test_benchmark_ticker_is_configurable_but_defaults_to_rsp():
    # The function stays generically parameterized (benchmark_ticker=...)
    # rather than hardcoding RSP internally, so an explicit override still
    # works exactly like the SPY-hardcoded version used to.
    days = make_days(15)
    qqq = [None] + [0.1] * 14
    daily_ret = make_daily_ret(days, {"QQQ": qqq, "AAPL": [0.1] * 15})
    narratives = [{"id": "n1", "tickers": ["AAPL"]}]
    result = compute_narrative_rs_history(narratives, daily_ret, days, lookback_days=15, benchmark_ticker="QQQ")
    assert result is not None
    assert result["benchmark"] == "QQQ"
