"""
Tests for scripts/build_screeners.py — the V1 predefined-Screener engine
(spec point 19-27). Covers the generic rule-engine ops, the Weekly Strength
pilot preset's exact test matrix A-F from the spec, sort order, and the
TradingView MIC-mapping/export-safety rules. All synthetic data — no
network / no MASSIVE_API_KEY required.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_screeners import (  # noqa: E402
    eval_rule, eval_preset, mic_to_tradingview_symbol,
    build_screener_ticker_row, weekly_strength_sort_key, build_screener,
)


WEEKLY_STRENGTH_CFG = {
    "name": "Weekly Strength",
    "enabled": True,
    "all": [
        {"field": "gain_from_sma50_pct", "op": ">", "value": 0},
        {"field": "rs_percentile_1w", "op": ">=", "value": 85},
    ],
    "any": [
        {"field": "ema10_distance_pct", "op": "between", "min": -5, "max": 5},
        {"field": "ema21_distance_pct", "op": "between", "min": -5, "max": 5},
    ],
}

MIC_MAP = {"XNAS": "NASDAQ", "XNYS": "NYSE", "XASE": "AMEX", "ARCX": "AMEX"}


# ── generic rule-engine ops ──────────────────────────────────────

def test_eval_rule_gt_gte_lt_lte_eq():
    t = {"x": 5}
    assert eval_rule({"field": "x", "op": ">", "value": 4}, t) is True
    assert eval_rule({"field": "x", "op": ">", "value": 5}, t) is False
    assert eval_rule({"field": "x", "op": ">=", "value": 5}, t) is True
    assert eval_rule({"field": "x", "op": "<", "value": 6}, t) is True
    assert eval_rule({"field": "x", "op": "<=", "value": 5}, t) is True
    assert eval_rule({"field": "x", "op": "==", "value": 5}, t) is True
    assert eval_rule({"field": "x", "op": "==", "value": 6}, t) is False


def test_eval_rule_between_inclusive():
    assert eval_rule({"field": "x", "op": "between", "min": -5, "max": 5}, {"x": 5.0}) is True
    assert eval_rule({"field": "x", "op": "between", "min": -5, "max": 5}, {"x": -5.0}) is True
    assert eval_rule({"field": "x", "op": "between", "min": -5, "max": 5}, {"x": 5.01}) is False


def test_eval_rule_missing_field_never_passes():
    assert eval_rule({"field": "x", "op": ">", "value": 0}, {}) is False
    assert eval_rule({"field": "x", "op": ">", "value": 0}, {"x": None}) is False


def test_eval_rule_unknown_op_raises():
    with pytest.raises(ValueError):
        eval_rule({"field": "x", "op": "!=", "value": 0}, {"x": 1})


def test_eval_preset_all_and_any():
    preset = {"all": [{"field": "a", "op": ">", "value": 0}],
              "any": [{"field": "b", "op": ">", "value": 0}, {"field": "c", "op": ">", "value": 0}]}
    assert eval_preset(preset, {"a": 1, "b": 1, "c": -1}) is True
    assert eval_preset(preset, {"a": 1, "b": -1, "c": -1}) is False
    assert eval_preset(preset, {"a": -1, "b": 1, "c": 1}) is False


def test_eval_preset_empty_any_imposes_no_extra_condition():
    preset = {"all": [{"field": "a", "op": ">", "value": 0}], "any": []}
    assert eval_preset(preset, {"a": 1}) is True


# ── Weekly Strength pilot: exact test matrix A-F from the spec ──

def make_ticker(eligible=True, gain_from_sma50_pct=1.0, rs_percentile_1w=90.0,
                 ema10_distance_pct=0.0, ema21_distance_pct=0.0, healthcare_biotech_excluded=False):
    return {
        "eligible": eligible and not healthcare_biotech_excluded,
        "gain_from_sma50_pct": gain_from_sma50_pct,
        "rs_percentile_1w": rs_percentile_1w,
        "ema10_distance_pct": ema10_distance_pct,
        "ema21_distance_pct": ema21_distance_pct,
        "healthcare_biotech_excluded": healthcare_biotech_excluded,
    }


def test_case_a_above_sma50_rs90_ema10_plus2_ema21_plus7_match():
    t = make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=90, ema10_distance_pct=2, ema21_distance_pct=7)
    assert eval_preset(WEEKLY_STRENGTH_CFG, t) is True


def test_case_b_above_sma50_rs90_ema10_plus8_ema21_minus3_match():
    t = make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=90, ema10_distance_pct=8, ema21_distance_pct=-3)
    assert eval_preset(WEEKLY_STRENGTH_CFG, t) is True


def test_case_c_rs84_below_threshold_no_match():
    t = make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=84, ema10_distance_pct=0, ema21_distance_pct=0)
    assert eval_preset(WEEKLY_STRENGTH_CFG, t) is False


def test_case_d_below_sma50_rs95_no_match():
    t = make_ticker(gain_from_sma50_pct=-1.0, rs_percentile_1w=95, ema10_distance_pct=0, ema21_distance_pct=0)
    assert eval_preset(WEEKLY_STRENGTH_CFG, t) is False


def test_case_e_above_sma50_rs95_both_emas_outside_band_no_match():
    t = make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=95, ema10_distance_pct=6, ema21_distance_pct=6)
    assert eval_preset(WEEKLY_STRENGTH_CFG, t) is False


def test_case_f_healthcare_biotech_excluded_no_match():
    t = make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=95, ema10_distance_pct=2, ema21_distance_pct=2,
                     healthcare_biotech_excluded=True)
    # eligible is already False upstream once healthcare_biotech_excluded — the
    # preset rules never even need to run; build_screener's own eligible gate
    # is what excludes it (this test proves the ticker itself models that).
    assert t["eligible"] is False


def test_case_f_via_build_screener_excludes_healthcare_biotech():
    tickers = {
        "PHARMA": make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=95, ema10_distance_pct=2,
                               ema21_distance_pct=2, healthcare_biotech_excluded=True),
        "TECH": make_ticker(gain_from_sma50_pct=2.0, rs_percentile_1w=95, ema10_distance_pct=2, ema21_distance_pct=2),
    }
    result, _ = build_screener("weekly_strength", WEEKLY_STRENGTH_CFG, tickers, {}, MIC_MAP)
    symbols = {row["symbol"] for row in result["tickers"]}
    assert "PHARMA" not in symbols
    assert "TECH" in symbols


# ── base eligible gate is never re-derived, only consumed ──────────

def test_build_screener_skips_ineligible_even_if_rules_would_match():
    tickers = {"X": make_ticker(eligible=False, gain_from_sma50_pct=5, rs_percentile_1w=99,
                                 ema10_distance_pct=0, ema21_distance_pct=0)}
    result, _ = build_screener("weekly_strength", WEEKLY_STRENGTH_CFG, tickers, {}, MIC_MAP)
    assert result["count"] == 0


# ── sort order (spec point 24): rs_percentile_1w desc, structural_rs desc, symbol asc ──

def test_weekly_strength_sort_order():
    rows = [
        {"symbol": "BBB", "rs_percentile_1w": 90, "structural_rs": 50},
        {"symbol": "AAA", "rs_percentile_1w": 90, "structural_rs": 60},
        {"symbol": "CCC", "rs_percentile_1w": 95, "structural_rs": 10},
    ]
    rows.sort(key=weekly_strength_sort_key)
    assert [r["symbol"] for r in rows] == ["CCC", "AAA", "BBB"]


def test_weekly_strength_sort_handles_none_gracefully():
    rows = [
        {"symbol": "AAA", "rs_percentile_1w": None, "structural_rs": None},
        {"symbol": "BBB", "rs_percentile_1w": 90, "structural_rs": 50},
    ]
    rows.sort(key=weekly_strength_sort_key)
    assert [r["symbol"] for r in rows] == ["BBB", "AAA"]  # None never outranks a real value


def test_build_screener_applies_weekly_strength_sort_end_to_end():
    tickers = {
        "BBB": make_ticker(gain_from_sma50_pct=1, rs_percentile_1w=90),
        "AAA": make_ticker(gain_from_sma50_pct=1, rs_percentile_1w=95),
    }
    tickers["BBB"]["structural_rs"] = 50
    tickers["AAA"]["structural_rs"] = 50
    result, _ = build_screener("weekly_strength", WEEKLY_STRENGTH_CFG, tickers, {}, MIC_MAP)
    assert [r["symbol"] for r in result["tickers"]] == ["AAA", "BBB"]


# ── TradingView MIC mapping / export safety (spec point 26) ────────

def test_known_mic_maps_to_correct_prefix():
    symbol, unknown = mic_to_tradingview_symbol("AAPL", "XNAS", MIC_MAP)
    assert symbol == "NASDAQ:AAPL"
    assert unknown is False

    symbol, unknown = mic_to_tradingview_symbol("DELL", "XNYS", MIC_MAP)
    assert symbol == "NYSE:DELL"
    assert unknown is False


def test_unknown_mic_never_silently_mis_prefixed():
    symbol, unknown = mic_to_tradingview_symbol("XYZ", "XLON", MIC_MAP)  # not in the audited table
    assert symbol is None
    assert unknown is True


def test_missing_primary_exchange_is_treated_as_unknown():
    symbol, unknown = mic_to_tradingview_symbol("XYZ", None, MIC_MAP)
    assert symbol is None
    assert unknown is True


def test_build_screener_ticker_row_carries_tradingview_symbol_and_no_api_key_shaped_fields():
    ticker = make_ticker()
    ticker.update({"close": 100.0, "adr20": 5.0, "market_cap": 2e9, "structural_rs": 70.0,
                    "sma50_distance_pct": 2.0})
    row, unknown_mic = build_screener_ticker_row("AAPL", ticker, "XNAS", MIC_MAP)
    assert row["symbol"] == "AAPL"
    assert row["tradingview_symbol"] == "NASDAQ:AAPL"
    assert unknown_mic is False
    # No field anywhere in the output row is API-key-shaped -- the screener
    # output is pure market-data fields only (point 20's "API Key niemals
    # ins Frontend" enforced structurally: this script never even imports
    # os.environ / MASSIVE_API_KEY, see the module import list).
    assert not any("key" in k.lower() or "token" in k.lower() or "secret" in k.lower() for k in row)


def test_build_screener_tracks_unknown_mic_tickers_for_audit():
    tickers = {"AAPL": make_ticker(gain_from_sma50_pct=1, rs_percentile_1w=90)}
    types_ref = {"AAPL": {"primary_exchange": "XLON"}}  # not in the audited MIC table
    result, unknown_mic_tickers = build_screener("weekly_strength", WEEKLY_STRENGTH_CFG, tickers, types_ref, MIC_MAP)
    assert result["tickers"][0]["tradingview_symbol"] is None
    assert unknown_mic_tickers == [{"symbol": "AAPL", "primary_exchange": "XLON"}]


def test_build_screener_ticker_row_only_carries_documented_fields():
    ticker = make_ticker()
    ticker.update({"close": 1, "adr20": 1, "market_cap": 1, "structural_rs": 1, "sma50_distance_pct": 1})
    row, _ = build_screener_ticker_row("AAPL", ticker, "XNAS", MIC_MAP)
    expected_keys = {"symbol", "close", "adr20", "market_cap", "rs_percentile_1w", "structural_rs",
                      "sma50_distance_pct", "ema10_distance_pct", "ema21_distance_pct",
                      "primary_exchange", "tradingview_symbol"}
    assert set(row.keys()) == expected_keys  # doesn't dump every Market Features field (spec point 23)


def test_build_screener_uses_massive_api_key_nowhere():
    import inspect
    import build_screeners
    source = inspect.getsource(build_screeners)
    assert "MASSIVE_API_KEY" not in source
    assert "massive_get" not in source
    assert "requests" not in source  # no HTTP client imported at all -- zero network capability
