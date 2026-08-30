"""
Tests for scripts/build_data.py: calc_atr_extension (QQQ/SPY price
structure feeding QQQ Health, V1 rebuild point 11) and calc_metrics (index/
crypto/commodity table rows, V6 point 29A NaN-safety fix). Synthetic data
only — no network / no yfinance download required for the functions under
test.
Run with: pytest tests/ -v
"""

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_data import calc_atr_extension, calc_metrics, calc_moving_averages  # noqa: E402
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


# ── V6 point 29A: calc_metrics NaN-safety (root cause of the empty QQQ
# breadth charts in production — a bare NaN close for ONE unrelated global-
# index ticker made json.dump() emit an invalid `NaN` token, which broke
# JSON.parse() for the ENTIRE snapshot.json in the browser, not just that
# ticker/field) ──

def make_close_hist(values, freq="B"):
    idx = pd.date_range("2026-01-01", periods=len(values), freq=freq)
    return pd.DataFrame({"Close": pd.Series(values, index=idx, dtype=float)})


def test_calc_metrics_none_hist_returns_none():
    assert calc_metrics(None) is None


def test_calc_metrics_insufficient_history_returns_none():
    assert calc_metrics(make_close_hist([100.0])) is None


def test_calc_metrics_nan_latest_close_uses_last_valid_close_not_nan():
    # Reproduces the exact production bug: yfinance returned NaN as the
    # LATEST close (observed for 000300.SS) despite otherwise-valid history.
    # The dropna-first fix removes the bad trailing row and uses the last
    # actually-valid close (102.0) instead of either crashing, returning
    # None for a ticker that clearly has usable data, or -- the pre-fix
    # production bug -- silently computing NaN and breaking the whole
    # snapshot.json's JSON validity.
    hist = make_close_hist([100.0, 101.0, 102.0, float('nan')])
    result = calc_metrics(hist)
    assert result is not None
    assert result["price"] == pytest.approx(102.0)
    assert result["price"] == result["price"]  # not NaN


def test_calc_metrics_all_nan_returns_none():
    hist = make_close_hist([float('nan')] * 5)
    assert calc_metrics(hist) is None


def test_calc_metrics_nan_interior_close_is_dropped_not_propagated():
    # A NaN row NOT at the end must also never leak into d1_pct/w1_pct/
    # hi52w_pct/hist_5d via .iloc[-N] landing on it or .max()/.mean() -- the
    # dropna-first fix removes it from the series entirely before any
    # metric is computed.
    hist = make_close_hist([100.0, float('nan'), 102.0, 103.0, 104.0, 105.0])
    result = calc_metrics(hist)
    assert result is not None
    assert all(v is None or (isinstance(v, (int, float)) and v == v) for v in
               [result["price"], result["d1_pct"], result["w1_pct"], result["hi52w_pct"], result["ytd_pct"]])
    assert all(x == x for x in result["hist_5d"])  # x == x is False for NaN


def test_calc_metrics_result_is_actually_json_serializable_without_nan_tokens():
    # The real-world failure mode: json.dump()'s default allow_nan=True
    # would happily emit a bare `NaN` token for a leftover NaN value, which
    # is NOT valid JSON and breaks JSON.parse() for the WHOLE file in the
    # browser. allow_nan=False turns any surviving NaN into a loud
    # ValueError here instead of a silent invalid-JSON byte in production.
    hist = make_close_hist([100.0, float('nan'), 102.0, 103.0, 104.0, 105.0])
    result = calc_metrics(hist)
    json.dumps(result, allow_nan=False)  # must not raise


def test_calc_metrics_normal_data_unaffected_by_the_dropna_fix():
    closes = [100.0 + i for i in range(10)]
    hist = make_close_hist(closes)
    result = calc_metrics(hist)
    assert result is not None
    assert result["price"] == pytest.approx(109.0)
    assert result["hist_5d"] == [pytest.approx(v) for v in closes[-5:]]


def test_calc_metrics_ytd_uses_dropna_close_not_raw_hist():
    # Regression for the ytd_pct fix: year_start now derives from the
    # dropna'd `close` series, not the raw `hist` DataFrame -- a NaN row
    # sitting exactly at the first trading day of the year must not leak
    # a NaN ytd_start.
    dates = pd.date_range("2025-12-29", periods=6, freq="B")
    values = [50.0, float('nan'), 60.0, 61.0, 62.0, 63.0]  # first 2026 row (idx 2) would be NaN pre-fix
    hist = pd.DataFrame({"Close": pd.Series(values, index=dates)})
    result = calc_metrics(hist)
    assert result is not None
    assert result["ytd_pct"] == result["ytd_pct"]  # not NaN


# ── 2026-08-30 incident: the same NaN-close class of bug as V6 point 29A,
# but in the SPY/QQQ regime path (calc_moving_averages/calc_atr_extension),
# which calc_metrics's dropna fix never covered. A trailing NaN QQQ close
# from yfinance made price_structure.close (and everything derived from it:
# dist_ema10_pct, atr_pct, atr_extension) a bare NaN, which blanked Market
# Regime/QQQ Health/Opportunities all at once via a JSON.parse() failure on
# the whole dashboard_state.json, not just the QQQ Health card. ──

def test_calc_moving_averages_nan_trailing_close_does_not_propagate():
    closes = [100.0 + i * 0.1 for i in range(60)] + [float('nan')]
    hist = make_close_hist(closes)
    result = calc_moving_averages(hist)
    assert result is not None
    assert all(v == v for v in result.values())  # none are NaN


def test_calc_moving_averages_nan_drop_below_threshold_returns_none():
    hist = make_close_hist([100.0] * 40 + [float('nan')] * 10)  # 40 valid < 50
    assert calc_moving_averages(hist) is None


def test_calc_atr_extension_nan_trailing_close_does_not_propagate():
    closes = [100.0] * 60 + [100 + i * 2 for i in range(1, 21)] + [float('nan')]
    hist = make_hist(closes, high_low_range_pct=2.0)
    atr14, atr_pct, atr_extension = calc_atr_extension(hist)
    assert atr14 is not None and atr14 == atr14
    assert atr_pct is not None and atr_pct == atr_pct
    assert atr_extension is not None and atr_extension == atr_extension
    # Must match the value computed from the last VALID (non-NaN) close,
    # i.e. dropping the trailing NaN row must not shift which close is "last".
    atr14_clean, atr_pct_clean, atr_extension_clean = calc_atr_extension(hist.dropna(subset=["Close"]))
    assert atr14 == pytest.approx(atr14_clean)
    assert atr_pct == pytest.approx(atr_pct_clean)
    assert atr_extension == pytest.approx(atr_extension_clean)


def test_calc_atr_extension_result_is_json_serializable_without_nan_tokens():
    closes = [100.0] * 60 + [100 + i * 2 for i in range(1, 21)] + [float('nan')]
    hist = make_hist(closes, high_low_range_pct=2.0)
    atr14, atr_pct, atr_extension = calc_atr_extension(hist)
    json.dumps({"atr14": atr14, "atr_pct": atr_pct, "atr_extension": atr_extension}, allow_nan=False)
