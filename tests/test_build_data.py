"""
Tests for scripts/build_data.py: calc_atr_extension (QQQ/SPY price
structure feeding QQQ Health, V1 rebuild point 11). Synthetic data only —
no network / no yfinance download required for the function under test.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_data import calc_atr_extension  # noqa: E402
from build_market_features import calc_true_range  # noqa: E402  (independently-tested reference formula)


def make_hist(closes, high_low_range_pct=1.0):
    """Synthetic OHLC DataFrame: High = close*(1+range/100), Low = close
    (mirrors the same construction style used in test_market_features.py's
    ADR/ATR tests, so True Range = High - Low = close*range_pct/100 exactly)."""
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="B")
    closes = pd.Series(closes, index=idx, dtype=float)
    low = closes
    high = closes * (1 + high_low_range_pct / 100.0)
    return pd.DataFrame({"Close": closes, "High": high, "Low": low})


def test_insufficient_history_returns_all_none():
    hist = make_hist([100.0] * 30)  # < 51 rows required
    atr14, atr_pct, atr_extension = calc_atr_extension(hist)
    assert atr14 is None and atr_pct is None and atr_extension is None


def test_atr_extension_matches_manual_formula_above_sma50():
    # 80 flat-ish days at 100, then a run-up so close ends well above SMA50.
    closes = [100.0] * 60 + [100 + i * 2 for i in range(1, 21)]  # ends at 140
    hist = make_hist(closes, high_low_range_pct=2.0)
    atr14, atr_pct, atr_extension = calc_atr_extension(hist)

    # Independent recomputation using the canonical True Range formula
    # (calc_true_range is separately tested in test_market_features.py) —
    # this test's purpose is to verify calc_atr_extension's ASSEMBLY of
    # last/SMA50/ATR into %ATR and Extension, not True Range itself.
    close_series = hist["Close"]
    last = float(close_series.iloc[-1])
    sma50 = float(close_series.rolling(50).mean().iloc[-1])
    tr = calc_true_range(hist["High"], hist["Low"], close_series)
    expected_atr14 = float(tr.dropna().iloc[-14:].mean())
    expected_atr_pct = expected_atr14 / last * 100.0
    expected_gain_pct = (last - sma50) / sma50 * 100.0
    expected_extension = expected_gain_pct / expected_atr_pct

    assert atr14 == pytest.approx(expected_atr14, abs=0.01)
    assert atr_pct == pytest.approx(expected_atr_pct, abs=0.01)
    assert atr_extension == pytest.approx(expected_extension, abs=0.05)
    assert atr_extension > 0  # close is above SMA50 here


def test_atr_extension_negative_when_below_sma50():
    closes = [100.0] * 60 + [100 - i * 1.5 for i in range(1, 21)]  # trending down
    hist = make_hist(closes, high_low_range_pct=1.5)
    atr14, atr_pct, atr_extension = calc_atr_extension(hist)
    assert atr14 is not None
    assert atr_extension < 0
