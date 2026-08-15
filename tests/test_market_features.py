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
import build_market_features  # noqa: E402  (module import, for monkeypatching massive_get)
from build_market_features import (  # noqa: E402
    calc_ticker_features, calc_true_range, compute_eligibility, compute_eligible_universe,
    eligible_percentile_ranks, type_eligible_universe,
    calc_sma50_trend_fields, renormalized_weighted_sum, clamp_0_100,
    compute_recent_leader_bootstrap, load_price_cache, save_price_cache,
    PRICE_CACHE_SCHEMA_VERSION,
    classify_healthcare_biotech, compute_healthcare_excluded_universe,
    fetch_ticker_range_backfill, fetch_grouped_history_full_market_cached,
    calc_return_n_sessions_ago,
)
from build_narratives import percentile_ranks, compute_narrative_thrust_rsp  # noqa: E402


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


def test_healthcare_excluded_stock_is_never_eligible_even_if_adr_and_cap_pass(universe_cfg):
    # V6 point 5: Healthcare/Biotech-Exclusion runs BEFORE eligible=true --
    # a stock that would otherwise pass ADR/Market-Cap must still be
    # ineligible once healthcare_excluded=True.
    assert compute_eligibility(adr20=5.0, market_cap=2_000_000_000, universe_cfg=universe_cfg,
                                healthcare_excluded=True) is False
    assert compute_eligibility(adr20=5.0, market_cap=2_000_000_000, universe_cfg=universe_cfg,
                                healthcare_excluded=False) is True


def test_compute_eligibility_healthcare_excluded_defaults_false_for_legacy_callers(universe_cfg):
    # Pre-V6 callers (other scripts, other tests) that don't pass the new
    # 4th arg must keep working exactly as before.
    assert compute_eligibility(adr20=5.0, market_cap=2_000_000_000, universe_cfg=universe_cfg) is True


# ── V6 point 9: RSP benchmark folded into the price cache, then popped
# back out of `features` -- must never reach eligibility/RS/output ──

def test_rsp_is_computed_transiently_but_never_reaches_final_features():
    # Mirrors main()'s exact pattern: fetch_grouped_history_full_market_cached
    # is called with type_universe | {RSP_TICKER}, so close_df/high_df/low_df
    # transiently contain an RSP column: calc_ticker_features() will compute
    # RSP like any other ticker, but main() immediately pops it back out
    # before ADR-candidate filtering / enrichment / eligibility ever runs.
    close, high, low = make_ohlc(n_days=75, drift_pct=0.3)
    close_df = pd.DataFrame({"AAPL": close, "RSP": close})
    high_df = pd.DataFrame({"AAPL": high, "RSP": high})
    low_df = pd.DataFrame({"AAPL": low, "RSP": low})
    features = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)
    assert "RSP" in features  # computed transiently, just like any ticker
    features.pop("RSP", None)  # exact pop-back-out pattern used in main()
    assert "RSP" not in features
    assert "AAPL" in features


# ── V6 point 5-6: deterministic Healthcare/Biotech universe exclusion ──

@pytest.fixture
def hc_filter_cfg():
    return json.loads(Path(__file__).parent.parent.joinpath(
        "config/narrative_engine.json").read_text())["universe"]["healthcare_biotech_filter"]


def test_pharma_sic_prefix_is_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("2834", "PHARMACEUTICAL PREPARATIONS", None, hc_filter_cfg)
    assert excluded is True
    assert reason is not None


def test_biological_products_sic_prefix_is_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("2836", "BIOLOGICAL PRODUCTS, NO DIAGNOSTIC SUBSTANCES", None, hc_filter_cfg)
    assert excluded is True


def test_medical_surgical_instrument_sic_prefix_is_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("3841", "SURGICAL & MEDICAL INSTRUMENTS & APPARATUS", None, hc_filter_cfg)
    assert excluded is True


def test_hospital_health_service_sic_prefix_is_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("8062", "GENERAL MEDICAL & SURGICAL HOSPITALS", None, hc_filter_cfg)
    assert excluded is True


def test_hospital_medical_service_plan_exact_sic_is_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("6324", "HOSPITAL & MEDICAL SERVICE PLANS", None, hc_filter_cfg)
    assert excluded is True


def test_semiconductor_sic_is_not_excluded(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech("3674", "SEMICONDUCTORS & RELATED DEVICES", None, hc_filter_cfg)
    assert excluded is False
    assert reason is None


def test_software_company_with_healthcare_customer_mention_is_not_falsely_excluded(hc_filter_cfg):
    # V6.1 point 18: company_description is now ALSO consulted when a SIC is
    # present but unmapped (see the tests below) -- but only against a
    # deliberately narrow, company-self-describing phrase list. A pure
    # customer-/application-vertical mention ("used by hospitals ... other
    # healthcare providers") must still never trigger exclusion, regardless
    # of which branch reaches company_description (point 6/18's explicit
    # "keine normalen Tech-Unternehmen ausschliessen, nur weil deren Kunden
    # teilweise aus Healthcare kommen").
    excluded, reason = classify_healthcare_biotech(
        "7372", "SERVICES-PREPACKAGED SOFTWARE",
        "We build workflow software used by hospitals, clinics and other healthcare providers.",
        hc_filter_cfg)
    assert excluded is False
    assert reason is None


def test_sic_present_but_unmapped_still_excluded_via_high_precision_company_description(hc_filter_cfg):
    # V6.1 point 18 fix: V6 skipped company_description ENTIRELY whenever a
    # SIC existed, even if it didn't map to anything -- this is exactly the
    # bug that let a real Biotech company with a generic/unmapped SIC keep
    # counting as eligible (an active "Biotech" narrative surviving on the
    # dashboard). A generic SIC (7372, software) that does NOT itself imply
    # Healthcare/Biotech, combined with a strong company-self-description,
    # must now exclude.
    excluded, reason = classify_healthcare_biotech(
        "7372", "SERVICES-PREPACKAGED SOFTWARE",
        "We are a clinical-stage biopharmaceutical company developing novel cancer therapeutics.",
        hc_filter_cfg)
    assert excluded is True
    assert "7372" in reason


def test_sic_present_but_unmapped_and_generic_description_stays_eligible(hc_filter_cfg):
    # Same generic/unmapped SIC, but a company_description with no
    # high-precision Healthcare/Biotech self-description phrase -> stays
    # eligible (proves the new branch doesn't over-trigger on any SIC that
    # merely fails to match).
    excluded, reason = classify_healthcare_biotech(
        "7372", "SERVICES-PREPACKAGED SOFTWARE",
        "We build project management software for construction companies.", hc_filter_cfg)
    assert excluded is False
    assert reason is None


def test_sic_mapped_directly_never_reaches_company_description_branch(hc_filter_cfg):
    # A SIC that already maps (prefix "283") must exclude via that branch
    # alone -- company_description isn't even relevant here, but pin down
    # that a clearly-irrelevant description doesn't somehow prevent the
    # SIC-based exclusion either.
    excluded, reason = classify_healthcare_biotech(
        "2834", "PHARMACEUTICAL PREPARATIONS", "Irrelevant unrelated text.", hc_filter_cfg)
    assert excluded is True
    assert "SIC 2834" in reason
    assert "company_description" not in reason


def test_sic_code_none_uses_company_description_fallback_deterministically(hc_filter_cfg):
    excluded, reason = classify_healthcare_biotech(
        None, None, "A clinical-stage biopharmaceutical company developing drug candidates.", hc_filter_cfg)
    assert excluded is True
    assert reason is not None

    excluded2, reason2 = classify_healthcare_biotech(None, None, "A logistics and freight company.", hc_filter_cfg)
    assert excluded2 is False
    assert reason2 is None


def test_sic_description_keyword_backstop_fires_for_unlisted_sic_code(hc_filter_cfg):
    # SIC code itself not in sic_codes/sic_prefixes, but sic_description
    # contains a backstop keyword -> still excluded, company_description
    # never consulted in this branch.
    excluded, reason = classify_healthcare_biotech("9999", "OFFICES OF PHYSICIANS", "irrelevant text", hc_filter_cfg)
    assert excluded is True


def test_compute_healthcare_excluded_universe_kill_switch_returns_all_false(hc_filter_cfg):
    features = {"AAA": {}, "BBB": {}}
    sic_by_symbol = {"AAA": {"sic_code": "2834", "sic_description": "PHARMACEUTICAL PREPARATIONS"}, "BBB": {}}
    result = compute_healthcare_excluded_universe(features, sic_by_symbol, {"exclude_healthcare_biotech": False})
    assert result == {"AAA": (False, None), "BBB": (False, None)}


def test_compute_healthcare_excluded_universe_end_to_end(hc_filter_cfg):
    features = {"PHARMA_CO": {}, "TECH_CO": {}, "NO_SIC_DATA": {}}
    sic_by_symbol = {
        "PHARMA_CO": {"sic_code": "2834", "sic_description": "PHARMACEUTICAL PREPARATIONS"},
        "TECH_CO": {"sic_code": "7372", "sic_description": "SERVICES-PREPACKAGED SOFTWARE"},
    }
    universe_cfg = {"exclude_healthcare_biotech": True, "healthcare_biotech_filter": hc_filter_cfg}
    result = compute_healthcare_excluded_universe(features, sic_by_symbol, universe_cfg)
    assert result["PHARMA_CO"][0] is True
    assert result["TECH_CO"][0] is False
    assert result["NO_SIC_DATA"][0] is False  # missing SIC data entirely, no description keyword hit either


def test_healthcare_exclusion_reduces_eligible_universe_end_to_end(hc_filter_cfg):
    u_cfg = {"adr_minimum_pct": 4.0, "market_cap_minimum_usd": 1_000_000_000,
              "exclude_healthcare_biotech": True, "healthcare_biotech_filter": hc_filter_cfg}
    features = {"PHARMA_CO": {"adr20": 6.0}, "TECH_CO": {"adr20": 6.0}}
    market_caps = {"PHARMA_CO": 5_000_000_000, "TECH_CO": 5_000_000_000}
    sic_by_symbol = {
        "PHARMA_CO": {"sic_code": "2834", "sic_description": "PHARMACEUTICAL PREPARATIONS"},
        "TECH_CO": {"sic_code": "7372", "sic_description": "SERVICES-PREPACKAGED SOFTWARE"},
    }
    healthcare_excluded = compute_healthcare_excluded_universe(features, sic_by_symbol, u_cfg)
    eligible = compute_eligible_universe(
        features, market_caps, u_cfg,
        healthcare_excluded_by_symbol={sym: v[0] for sym, v in healthcare_excluded.items()})
    assert eligible == {"PHARMA_CO": False, "TECH_CO": True}


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


# ── V6 point 21: EMA21 is purely ADDITIVE, EMA20 stays canonical/unchanged ──

def test_ema21_is_additive_and_ema20_unchanged():
    close, high, low = make_ohlc(n_days=75, drift_pct=0.5)  # uptrend
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]
    # New field present and populated...
    assert out["ema21"] is not None
    assert out["ema21_distance_pct"] is not None
    # ...but the pre-existing EMA20 formula/value is byte-identical to a
    # standalone ewm(span=20) computation -- EMA21's presence must not have
    # perturbed it.
    expected_ema20 = close.ewm(span=20).mean().iloc[-1]
    assert out["ema20"] == pytest.approx(round(float(expected_ema20), 2))
    # In a steady uptrend, price also sits above the new EMA21.
    assert out["ema21_distance_pct"] > 0


def test_ema21_matches_standalone_formula():
    close, high, low = make_ohlc(n_days=75, drift_pct=0.5)
    close_df = pd.DataFrame({"TEST": close})
    high_df = pd.DataFrame({"TEST": high})
    low_df = pd.DataFrame({"TEST": low})
    out = calc_ticker_features(close_df, high_df, low_df, adr_lookback=20)["TEST"]
    expected_ema21 = close.ewm(span=21).mean().iloc[-1]
    last = close.iloc[-1]
    expected_dist = round(float((last - expected_ema21) / expected_ema21 * 100), 2)
    assert out["ema21"] == pytest.approx(round(float(expected_ema21), 2))
    assert out["ema21_distance_pct"] == pytest.approx(expected_dist)


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
    per_ticker_open = {"AAA": {"2026-01-01": 9.8, "2026-01-02": 10.6}}

    assert load_price_cache(cache_path) is None  # nothing written yet

    save_price_cache(cache_path, trading_days, per_ticker_close, per_ticker_high, per_ticker_low,
                      per_ticker_open, {"AAA"})
    loaded = load_price_cache(cache_path)

    assert loaded["schema_version"] == PRICE_CACHE_SCHEMA_VERSION
    assert loaded["dates"] == trading_days
    assert loaded["tickers"]["AAA"]["close"] == [10.0, 11.0]
    assert loaded["tickers"]["AAA"]["open"] == [9.8, 10.6]


def test_price_cache_skips_tickers_with_no_observed_data(tmp_path):
    cache_path = tmp_path / "market_history.json"
    trading_days = ["2026-01-01"]
    save_price_cache(cache_path, trading_days, {}, {}, {}, {}, {"NEVER_SEEN"})
    loaded = load_price_cache(cache_path)
    assert "NEVER_SEEN" not in loaded["tickers"]  # all-null row -> not persisted


def test_price_cache_discards_mismatched_schema_version(tmp_path):
    cache_path = tmp_path / "market_history.json"
    cache_path.write_text(json.dumps({"schema_version": PRICE_CACHE_SCHEMA_VERSION + 1, "dates": [], "tickers": {}}))
    assert load_price_cache(cache_path) is None  # stale schema -> treated as cold cache


# ── V6 point 9 follow-up: RSP one-time single-ticker backfill ──
# A ticker that is genuinely new to the shared price cache (a real new IPO,
# or a stock newly passing the type/ADR filter) correctly has no earlier
# history to backfill and is left to accumulate one day at a time -- that
# is intentional, unchanged behaviour (see the tests above for
# compute_eligible_universe etc.). RSP is different: it already has decades
# of real trading history available from Massive, it's just new to THIS
# cache because this feature is the first thing that ever asked for it.
# Without the backfill below, RSP would be stranded at 1-2 days of history
# for ~260 daily runs before any Strength/Thrust window could ever populate
# — a real gap the user caught in production, not the intended behaviour.

from datetime import datetime as _dt, timezone as _tz  # noqa: E402


def test_fetch_ticker_range_backfill_converts_timestamps_and_extracts_ohlc(monkeypatch):
    d1 = _dt(2026, 1, 1, tzinfo=_tz.utc)
    d2 = _dt(2026, 1, 2, tzinfo=_tz.utc)
    calls = []

    def fake_massive_get(path, params=None, retries=3):
        calls.append(path)
        return {"results": [
            {"t": int(d1.timestamp() * 1000), "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5},
            {"t": int(d2.timestamp() * 1000), "o": 100.5, "h": 102.0, "l": 100.0, "c": 101.5},
        ]}

    monkeypatch.setattr(build_market_features, "massive_get", fake_massive_get)
    out = fetch_ticker_range_backfill("RSP", "2026-01-01", "2026-01-03")
    assert calls == ["/v2/aggs/ticker/RSP/range/1/day/2026-01-01/2026-01-03"]
    assert out["2026-01-01"] == {"o": 100.0, "h": 101.0, "l": 99.0, "c": 100.5}
    assert out["2026-01-02"]["c"] == 101.5
    assert len(out) == 2


def test_fetch_ticker_range_backfill_handles_missing_or_empty_response(monkeypatch):
    monkeypatch.setattr(build_market_features, "massive_get", lambda *a, **k: None)
    assert fetch_ticker_range_backfill("RSP", "2026-01-01", "2026-01-03") == {}
    monkeypatch.setattr(build_market_features, "massive_get", lambda *a, **k: {"results": []})
    assert fetch_ticker_range_backfill("RSP", "2026-01-01", "2026-01-03") == {}


def test_fetch_grouped_history_backfills_a_new_benchmark_ticker_to_match_existing_cache_depth(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_history.json"
    today = _dt.now(_tz.utc).date()
    trading_days = [(today - pd.Timedelta(days=i)).isoformat() for i in range(4, -1, -1)]  # 5 days ending today

    # Pre-existing cache: only AAPL has history, RSP has never been fetched.
    save_price_cache(
        cache_path, trading_days,
        {"AAPL": {d: 150.0 + i for i, d in enumerate(trading_days)}},
        {"AAPL": {d: 151.0 for d in trading_days}},
        {"AAPL": {d: 149.0 for d in trading_days}},
        {"AAPL": {d: 150.0 for d in trading_days}},
        {"AAPL"},
    )

    range_calls = []

    def fake_massive_get(path, params=None, retries=3):
        if path.startswith("/v2/aggs/ticker/"):
            range_calls.append(path)
            return {"results": [
                {"t": int(_dt.fromisoformat(d).replace(tzinfo=_tz.utc).timestamp() * 1000),
                 "o": 200.0, "h": 201.0, "l": 199.0, "c": 200.0 + i}
                for i, d in enumerate(trading_days)
            ]}
        # Grouped-daily endpoint should never be reached: newest_cached == today,
        # so the incremental walk-forward loop must break on its first iteration.
        raise AssertionError(f"unexpected grouped-daily call: {path}")

    monkeypatch.setattr(build_market_features, "massive_get", fake_massive_get)

    close_df, high_df, low_df, out_days = fetch_grouped_history_full_market_cached(
        {"AAPL", "RSP"}, cache_path, target_days=260, backfill_tickers={"RSP"})

    assert range_calls == [f"/v2/aggs/ticker/RSP/range/1/day/{trading_days[0]}/{trading_days[-1]}"]
    assert list(close_df["RSP"].dropna().index) == trading_days
    assert close_df["RSP"].iloc[-1] == pytest.approx(200.0 + len(trading_days) - 1)
    assert close_df["AAPL"].notna().all()  # existing ticker's history untouched


def test_fetch_grouped_history_skips_backfill_when_ticker_already_in_cache(tmp_path, monkeypatch):
    cache_path = tmp_path / "market_history.json"
    today = _dt.now(_tz.utc).date()
    trading_days = [(today - pd.Timedelta(days=i)).isoformat() for i in range(2, -1, -1)]

    # RSP is ALREADY in the cache from a previous run -> must never re-trigger the backfill call.
    save_price_cache(
        cache_path, trading_days,
        {"RSP": {d: 220.0 for d in trading_days}},
        {"RSP": {d: 221.0 for d in trading_days}},
        {"RSP": {d: 219.0 for d in trading_days}},
        {"RSP": {d: 220.0 for d in trading_days}},
        {"RSP"},
    )

    def fake_massive_get(path, params=None, retries=3):
        if path.startswith("/v2/aggs/ticker/"):
            raise AssertionError("backfill must not be called when the ticker is already cached")
        raise AssertionError(f"unexpected grouped-daily call: {path}")

    monkeypatch.setattr(build_market_features, "massive_get", fake_massive_get)

    close_df, _, _, _ = fetch_grouped_history_full_market_cached(
        {"RSP"}, cache_path, target_days=260, backfill_tickers={"RSP"})
    assert close_df["RSP"].notna().all()


def test_fetch_grouped_history_backfills_a_ticker_already_present_but_stranded_sparse(tmp_path, monkeypatch):
    """Regression test for the real production bug: a ticker can already be a
    key in cached_tickers (added to per_ticker_close by a run that predates
    this backfill feature) while its "close" series is almost entirely None
    -- e.g. RSP had exactly 2 real days out of a 260-day-deep cache after its
    first-ever run, because it was simply new to the shared price cache and
    the old incremental walk only ever appends one day forward per run.
    `ticker in cached_tickers` alone must NOT be treated as "already fully
    backfilled" -- coverage must be compared against the cache's own date
    depth, or a previously-stranded ticker silently stays stranded forever
    even with backfill_tickers set (this is exactly what shipped and was
    caught against real production data)."""
    cache_path = tmp_path / "market_history.json"
    today = _dt.now(_tz.utc).date()
    trading_days = [(today - pd.Timedelta(days=i)).isoformat() for i in range(4, -1, -1)]  # 5 days ending today

    # RSP is a key in cached_tickers, but only the last 2 of 5 days are real
    # -- exactly the shape a pre-backfill run would have left behind.
    sparse_close = {d: None for d in trading_days}
    sparse_close[trading_days[-2]] = 222.73
    sparse_close[trading_days[-1]] = 222.77
    save_price_cache(
        cache_path, trading_days,
        {"AAPL": {d: 150.0 + i for i, d in enumerate(trading_days)}, "RSP": sparse_close},
        {"AAPL": {d: 151.0 for d in trading_days}, "RSP": sparse_close},
        {"AAPL": {d: 149.0 for d in trading_days}, "RSP": sparse_close},
        {"AAPL": {d: 150.0 for d in trading_days}, "RSP": sparse_close},
        {"AAPL", "RSP"},
    )

    range_calls = []

    def fake_massive_get(path, params=None, retries=3):
        if path.startswith("/v2/aggs/ticker/"):
            range_calls.append(path)
            return {"results": [
                {"t": int(_dt.fromisoformat(d).replace(tzinfo=_tz.utc).timestamp() * 1000),
                 "o": 200.0, "h": 201.0, "l": 199.0, "c": 200.0 + i}
                for i, d in enumerate(trading_days)
            ]}
        raise AssertionError(f"unexpected grouped-daily call: {path}")

    monkeypatch.setattr(build_market_features, "massive_get", fake_massive_get)

    close_df, _, _, _ = fetch_grouped_history_full_market_cached(
        {"AAPL", "RSP"}, cache_path, target_days=260, backfill_tickers={"RSP"})

    assert range_calls == [f"/v2/aggs/ticker/RSP/range/1/day/{trading_days[0]}/{trading_days[-1]}"]
    assert close_df["RSP"].notna().all()
    assert close_df["RSP"].iloc[-1] == pytest.approx(200.0 + len(trading_days) - 1)


# ── V6.1 point 16: Stock Thrust (stock_thrust_rs) building block ──

def test_calc_return_n_sessions_ago_worked_example():
    # 10 sessions of close prices, strictly increasing by 1 each day (100..109).
    days = [f"2026-01-{d:02d}" for d in range(1, 11)]
    close_df = pd.DataFrame({"AAPL": [100.0 + i for i in range(10)]}, index=days)
    # today (sessions_ago=0), 5-session window: current=109 (idx9), base=104 (idx4) -> +4.8077%
    ret_today = calc_return_n_sessions_ago(close_df, "AAPL", window=5, sessions_ago=0)
    assert ret_today == pytest.approx((109.0 - 104.0) / 104.0 * 100, abs=1e-6)
    # 3 sessions ago: current=106 (idx6), base=101 (idx1) -> +4.9505%
    ret_3_ago = calc_return_n_sessions_ago(close_df, "AAPL", window=5, sessions_ago=3)
    assert ret_3_ago == pytest.approx((106.0 - 101.0) / 101.0 * 100, abs=1e-6)


def test_calc_return_n_sessions_ago_none_when_insufficient_history():
    days = [f"2026-01-{d:02d}" for d in range(1, 4)]  # only 3 sessions
    close_df = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=days)
    assert calc_return_n_sessions_ago(close_df, "AAPL", window=5, sessions_ago=0) is None


def test_calc_return_n_sessions_ago_none_when_ticker_absent():
    days = [f"2026-01-{d:02d}" for d in range(1, 11)]
    close_df = pd.DataFrame({"AAPL": [100.0 + i for i in range(10)]}, index=days)
    assert calc_return_n_sessions_ago(close_df, "MSFT", window=5, sessions_ago=0) is None


def test_stock_thrust_rs_matches_thrust_formula_end_to_end():
    # Reproduces the spec's worked Thrust example (0.60*90+0.40*80+0.10*(90-70)=88)
    # through the ACTUAL cross-sectional plumbing: build a tiny eligible
    # universe whose 5-session returns rank AAPL at RS1W=90-equivalent today
    # and RS1W=70-equivalent 3 sessions ago, then verify stock_thrust_rs (via
    # the same compute_narrative_thrust_rsp reuse build_market_features.py
    # wires up) matches manual arithmetic on the resulting percentiles.
    weights = {"strength_1w": 0.60, "strength_1m": 0.40, "delta_1w_acceleration": 0.10}
    # rs_1w_today=90, rs_1m_today=80, rs_1w_3_ago=60 (spec's own thrust worked example numbers)
    thrust = compute_narrative_thrust_rsp(90, 80, 60, weights)
    assert thrust == pytest.approx(0.60 * 90 + 0.40 * 80 + 0.10 * (90 - 60))
    assert thrust == pytest.approx(89.0)  # 54 + 32 + 3
