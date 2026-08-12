"""
Tests for scripts/build_dashboard_states.py: Market Regime Score, QQQ Health
Score + Modifiers, Narrative Lifecycle, Opportunity Engine (Leader/Fresh
Leader/Near EMAs/Extended/Constructive Reset/Laggard), Change Detection.
All synthetic data — no network / no file I/O required for the functions
under test (only main()'s I/O wrapper touches disk, not tested here).
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_dashboard_states import (  # noqa: E402
    apply_confirm_days,
    calc_constructive_reset_narratives,
    calc_extended_with_hysteresis,
    calc_fresh_leader_label,
    calc_high_low_score,
    calc_laggard_state,
    calc_leader_entry_condition,
    calc_leader_exit_condition,
    calc_market_breadth_score,
    calc_market_momentum_score,
    calc_market_regime_score,
    calc_mco_score,
    calc_mcsi_score,
    calc_narrative_environment_score,
    calc_narrative_lifecycle_conditions,
    calc_narrative_momentum_scores,
    calc_near_emas,
    calc_qqq_breadth_score,
    calc_qqq_health_score,
    calc_qqq_modifiers,
    calc_qqq_trend_score,
    calc_stock_leadership_score,
    calc_stock_quality_base_state,
    clamp_0_100,
    detect_market_regime_changes,
    detect_narrative_changes,
    detect_qqq_health_changes,
    detect_stock_changes,
    narrative_bottom_pct_members,
    normal_cdf,
    percentile_of_last_value,
    renormalized_weighted_sum,
    select_lifecycle_state,
    series_delta,
    series_mean_last_n,
    state_from_thresholds,
)


# ── Generic primitives ─────────────────────────────────────────

def test_renormalized_weighted_sum_full_data():
    values = {"a": 80.0, "b": 20.0}
    weights = {"a": 0.5, "b": 0.5}
    assert renormalized_weighted_sum(values, weights) == 50.0


def test_renormalized_weighted_sum_missing_component_renormalizes():
    # point 48: a missing subscore must not silently count as 0.
    values = {"a": 80.0, "b": None}
    weights = {"a": 0.4, "b": 0.6}
    assert renormalized_weighted_sum(values, weights) == 80.0  # 100% weight on 'a'


def test_renormalized_weighted_sum_all_missing_returns_none():
    assert renormalized_weighted_sum({"a": None, "b": None}, {"a": 0.5, "b": 0.5}) is None


def test_clamp_0_100():
    assert clamp_0_100(150) == 100.0
    assert clamp_0_100(-10) == 0.0
    assert clamp_0_100(55.5) == 55.5
    assert clamp_0_100(None) is None


def test_normal_cdf():
    assert normal_cdf(0) == pytest.approx(0.5, abs=1e-6)
    assert normal_cdf(3) > 0.99
    assert normal_cdf(-3) < 0.01


def test_series_delta():
    assert series_delta([1, 2, 3, 4, 5, 6], 5) == 5  # 6 - 1
    assert series_delta([1, 2, 3], 5) is None  # not enough history
    assert series_delta([], 5) is None


def test_series_mean_last_n():
    assert series_mean_last_n([1, 2, 3, 4, 5], 3) == 4.0  # mean(3,4,5)
    assert series_mean_last_n([1, 2], 3) is None


def test_percentile_of_last_value():
    vals = list(range(1, 21))  # 1..20, last value 20 is the max
    assert percentile_of_last_value(vals) == 100.0
    assert percentile_of_last_value([1, 2, 3]) is None  # sample too thin (<10)


# ── Market Regime Score (point 6-9) ────────────────────────────

def test_market_breadth_score_weighted_combination():
    features = [
        {"ema10_distance_pct": 1.0, "ema20_distance_pct": 1.0, "gain_from_sma50_pct": 1.0},
        {"ema10_distance_pct": -1.0, "ema20_distance_pct": -1.0, "gain_from_sma50_pct": -1.0},
    ]
    weights = {"pct_above_ema10": 0.40, "pct_above_ema20": 0.35, "pct_above_sma50": 0.25}
    result = calc_market_breadth_score(features, weights)
    assert result["pct_above_ema10"] == 50.0
    assert result["pct_above_ema20"] == 50.0
    assert result["pct_above_sma50"] == 50.0
    assert result["score"] == 50.0


def test_market_breadth_score_missing_component_renormalizes():
    features = [{"ema10_distance_pct": 5.0, "ema20_distance_pct": None, "gain_from_sma50_pct": None}]
    weights = {"pct_above_ema10": 0.40, "pct_above_ema20": 0.35, "pct_above_sma50": 0.25}
    result = calc_market_breadth_score(features, weights)
    assert result["score"] == 100.0  # only ema10 available, all positive -> 100%


def test_market_momentum_score_is_arithmetic_mean():
    features = [
        {"w1_pct": 1.0, "m1_pct": 1.0, "thrust_1w": 1.0},   # all positive
        {"w1_pct": -1.0, "m1_pct": -1.0, "thrust_1w": -1.0},  # all negative
    ]
    result = calc_market_momentum_score(features)
    assert result["pct_positive_1w"] == 50.0
    assert result["pct_positive_1m"] == 50.0
    assert result["pct_positive_thrust_1w"] == 50.0
    assert result["score"] == 50.0  # arithmetic mean of three 50s


def test_narrative_environment_score_arithmetic_mean():
    narratives = [
        {"scores": {"1w": {"strength": 5.0, "thrust": 1.0, "leadership": 60.0, "breadth": {"pct_positive": 70.0}}}},
        {"scores": {"1w": {"strength": -5.0, "thrust": -1.0, "leadership": 20.0, "breadth": {"pct_positive": 30.0}}}},
    ]
    result = calc_narrative_environment_score(narratives)
    assert result["pct_narratives_positive_strength"] == 50.0
    assert result["pct_narratives_positive_thrust"] == 50.0
    assert result["avg_breadth_pct_positive"] == 50.0
    assert result["avg_leadership"] == 40.0
    assert result["score"] == pytest.approx(47.5, abs=0.01)


def test_market_regime_score_weighted_and_clamped():
    score = calc_market_regime_score(80.0, 60.0, 40.0, {"breadth": 0.40, "momentum": 0.30, "narrative_environment": 0.30})
    # 0.4*80 + 0.3*60 + 0.3*40 = 32+18+12 = 62
    assert score == 62.0


def test_market_regime_score_missing_subscore_renormalizes_not_zero():
    # If narrative_environment is unavailable, it must NOT drag the score
    # down as if it were 0 (point 48).
    full = calc_market_regime_score(80.0, 80.0, 0.0, {"breadth": 0.40, "momentum": 0.30, "narrative_environment": 0.30})
    degraded = calc_market_regime_score(80.0, 80.0, None, {"breadth": 0.40, "momentum": 0.30, "narrative_environment": 0.30})
    assert degraded == 80.0
    assert degraded > full


def test_market_regime_state_thresholds():
    thresholds = [
        {"min": 80, "state": "STRONG OFFENSIVE"}, {"min": 65, "state": "OFFENSIVE"},
        {"min": 45, "state": "SELECTIVE"}, {"min": 30, "state": "DEFENSIVE"}, {"min": 0, "state": "RISK OFF"},
    ]
    assert state_from_thresholds(80.0, thresholds) == "STRONG OFFENSIVE"
    assert state_from_thresholds(79.99, thresholds) == "OFFENSIVE"
    assert state_from_thresholds(45.0, thresholds) == "SELECTIVE"
    assert state_from_thresholds(29.99, thresholds) == "RISK OFF"
    assert state_from_thresholds(None, thresholds) == "UNAVAILABLE"


# ── QQQ Health Score (point 10-15) ─────────────────────────────

TREND_POINTS = {"close_above_ema10": 25, "close_above_ema20": 25, "ema10_above_ema20": 20,
                 "close_above_sma50": 15, "close_above_sma200": 15}


def test_qqq_trend_score_all_conditions_true():
    regime = {"price": 100, "mas": {"ema10": 95, "ema20": 90, "sma50": 85, "sma200": 80}}
    assert calc_qqq_trend_score(regime, TREND_POINTS) == 100


def test_qqq_trend_score_no_conditions_true():
    # ema10 < ema20 too, otherwise "EMA10 > EMA20" (worth 20pts) is true
    # independently of price and this wouldn't be a true "0 conditions" case.
    regime = {"price": 50, "mas": {"ema10": 80, "ema20": 85, "sma50": 90, "sma200": 95}}
    assert calc_qqq_trend_score(regime, TREND_POINTS) == 0


def test_qqq_trend_score_partial():
    # Above EMA10/EMA20 (25+25=50), EMA10 > EMA20 (20), below SMA50/SMA200 (0)
    regime = {"price": 92, "mas": {"ema10": 90, "ema20": 88, "sma50": 95, "sma200": 100}}
    assert calc_qqq_trend_score(regime, TREND_POINTS) == 70


def test_qqq_breadth_score_weighted():
    kma_now = {"ema10": 100.0, "ema21": 0.0, "sma50": 0.0, "sma200": 0.0}
    weights = {"ema10": 0.25, "ema21": 0.35, "sma50": 0.25, "sma200": 0.15}
    assert calc_qqq_breadth_score(kma_now, weights) == 25.0


def test_mco_score_maps_via_normal_cdf():
    assert calc_mco_score(0) == 50.0
    assert calc_mco_score(2) > 90
    assert calc_mco_score(None) is None


def test_mcsi_score_trend_component():
    weights = {"level": 0.70, "trend": 0.30}
    above_sma10 = calc_mcsi_score(1.0, 0.5, weights)  # z >= sma10 -> trend=100
    below_sma10 = calc_mcsi_score(1.0, 1.5, weights)  # z < sma10 -> trend=0
    assert above_sma10 > below_sma10
    assert calc_mcsi_score(None, 0.5, weights) is None


def test_high_low_score_percentile_of_last_value():
    hl_history = list(range(-10, 110))  # 120 values, last (109) is the max
    score = calc_high_low_score(hl_history, n_components=100, lookback_days=120)
    assert score == 100.0


def test_high_low_score_thin_history_unavailable():
    assert calc_high_low_score([1, 2, 3], n_components=100, lookback_days=120) is None


def test_qqq_health_final_score_weighted():
    weights = {"trend": 0.25, "breadth": 0.30, "mco": 0.20, "mcsi": 0.15, "high_low": 0.10}
    score = calc_qqq_health_score(100, 100, 100, 100, 100, weights)
    assert score == 100.0
    score_zero = calc_qqq_health_score(0, 0, 0, 0, 0, weights)
    assert score_zero == 0.0


# ── QQQ Health Modifiers (point 14) ────────────────────────────

def test_extended_hysteresis_enter_exit_and_band():
    assert calc_extended_with_hysteresis(6.0, False, 5.0, 4.5) is True   # enter
    assert calc_extended_with_hysteresis(4.0, True, 5.0, 4.5) is False   # exit
    assert calc_extended_with_hysteresis(4.7, True, 5.0, 4.5) is True    # band -> keep previous (True)
    assert calc_extended_with_hysteresis(4.7, False, 5.0, 4.5) is False  # band -> keep previous (False)
    assert calc_extended_with_hysteresis(None, True, 5.0, 4.5) is True   # missing data -> keep previous


MODIFIER_CFG = {
    "modifiers": {
        "extended_atr_extension_enter": 5.0, "extended_atr_extension_exit": 4.5,
        "delta_lookback_days_short": 3, "delta_lookback_days_medium": 5,
        "narrowing_min_conditions": 2, "narrowing_breadth_delta_pp": -5.0,
        "deteriorating_min_conditions": 2, "repairing_min_conditions": 3, "repairing_breadth_delta_pp": 5.0,
    }
}


def _flat_history(n, last_val=None):
    h = [0.0] * n
    if last_val is not None:
        h[-1] = last_val
    return h


def test_qqq_modifiers_narrowing_when_above_ema20_and_two_negative_conditions():
    qqq_breadth = {
        "mco_zscore_history": [0, 0, 0, 0, 0, -1.0],  # delta5d = -1 - 0 = -1 < 0
        "summation_zscore_history": [0] * 6,
        "kma_history": {"ema21": [0, 0, 0, 0, 0, -6.0], "sma50": [0] * 6},  # delta5d = -6 <= -5
        "hl_history": [0] * 6,
    }
    mods, extended = calc_qqq_modifiers(qqq_breadth, close_above_ema20=True, base_state="HEALTHY",
                                          prev_extended=False, atr_extension=None, cfg=MODIFIER_CFG)
    assert "NARROWING" in mods
    assert "DETERIORATING" not in mods


def test_qqq_modifiers_deteriorating_when_below_ema20():
    qqq_breadth = {
        "mco_zscore_history": [0, 0, 0, 0, 0, -1.0],
        "summation_zscore_history": [0] * 6,
        "kma_history": {"ema21": [0, 0, 0, 0, 0, -6.0], "sma50": [0] * 6},
        "hl_history": [0] * 6,
    }
    mods, _ = calc_qqq_modifiers(qqq_breadth, close_above_ema20=False, base_state="FRAGILE",
                                   prev_extended=False, atr_extension=None, cfg=MODIFIER_CFG)
    assert "DETERIORATING" in mods
    assert "NARROWING" not in mods


def test_qqq_modifiers_repairing_only_when_not_healthy():
    qqq_breadth = {
        "mco_zscore_history": [0, 0, 0, 1.0],  # delta3d = 1 > 0
        "summation_zscore_history": [0, 0, 0, 0, 0, 1.0],  # delta5d = 1 > 0
        "kma_history": {"ema21": [0, 0, 0, 0, 0, 6.0], "sma50": [0] * 6},  # delta5d = 6 >= 5
        "hl_history": [0] * 6,
    }
    mods_not_healthy, _ = calc_qqq_modifiers(qqq_breadth, close_above_ema20=True, base_state="MIXED",
                                               prev_extended=False, atr_extension=None, cfg=MODIFIER_CFG)
    assert "REPAIRING" in mods_not_healthy

    mods_healthy, _ = calc_qqq_modifiers(qqq_breadth, close_above_ema20=True, base_state="HEALTHY",
                                           prev_extended=False, atr_extension=None, cfg=MODIFIER_CFG)
    assert "REPAIRING" not in mods_healthy  # gated: only when base state != HEALTHY


def test_qqq_modifiers_conflict_suppresses_both():
    # Constructed so BOTH the negative-group (>=2: mco_delta5<0, ema21/sma50
    # delta5<=-5) and positive-group (>=3: mco_delta3>0, mcsi_delta5>0,
    # hl_mean5>0) fire simultaneously -> spec says: no artificial modifier.
    qqq_breadth = {
        "mco_zscore_history": [1.0, 0, -1.0, 0, 0, 0.0],  # delta5d = 0-1=-1<0; delta3d = 0-(-1)=1>0
        "summation_zscore_history": [0, 0, 0, 0, 0, 1.0],  # delta5d = 1 > 0
        "kma_history": {"ema21": [0, 0, 0, 0, 0, -6.0], "sma50": [0, 0, 0, 0, 0, -6.0]},  # delta5d = -6 <= -5
        "hl_history": [0, 0, 0, 0, 0, 5],  # mean(last 5) = mean(0,0,0,0,5) = 1.0 > 0
    }
    mods, _ = calc_qqq_modifiers(qqq_breadth, close_above_ema20=True, base_state="MIXED",
                                   prev_extended=False, atr_extension=None, cfg=MODIFIER_CFG)
    assert "NARROWING" not in mods
    assert "DETERIORATING" not in mods
    assert "REPAIRING" not in mods


def test_qqq_modifiers_extended_stacks_independently():
    qqq_breadth = {"mco_zscore_history": [0] * 6, "summation_zscore_history": [0] * 6,
                    "kma_history": {"ema21": [0] * 6, "sma50": [0] * 6}, "hl_history": [0] * 6}
    mods, extended = calc_qqq_modifiers(qqq_breadth, close_above_ema20=True, base_state="HEALTHY",
                                          prev_extended=False, atr_extension=6.0, cfg=MODIFIER_CFG)
    assert extended is True
    assert mods == ["EXTENDED"]


# ── Narrative Momentum Score + Lifecycle (point 17-18) ─────────

def test_narrative_momentum_scores_percentile_ranked_narrative_vs_narrative():
    narratives = [
        {"id": "n1", "scores": {"1w": {"strength": 10.0, "thrust": 1.0, "leadership": 80.0, "breadth": {"pct_positive": 90.0}}}},
        {"id": "n2", "scores": {"1w": {"strength": 1.0, "thrust": -1.0, "leadership": 20.0, "breadth": {"pct_positive": 10.0}}}},
    ]
    weights = {"strength_percentile": 0.30, "thrust_percentile": 0.30, "leadership": 0.25, "breadth_pct_positive": 0.15}
    result = calc_narrative_momentum_scores(narratives, weights)
    assert result["n1"]["strength_percentile"] == 100.0
    assert result["n2"]["strength_percentile"] == 50.0
    assert result["n1"]["score"] > result["n2"]["score"]


LIFECYCLE_CFG = {
    "emerging": {"thrust_percentile_1w_min": 75, "leadership_1w_min": 40, "breadth_pct_positive_1w_min": 60, "leadership_delta5d_positive_required": True},
    "active": {"strength_1w_min": 0, "thrust_1w_min": 0, "leadership_1w_min": 40, "breadth_pct_positive_1w_min": 55},
    "mature": {"strength_1m_min": 0, "leadership_1m_min": 40, "thrust_1w_max": 0, "breadth_pct_positive_1w_min": 50},
    "fading": {"thrust_1w_max": 0, "breadth_pct_positive_1w_max": 50, "leadership_delta5d_max": -10, "confirm_days": 2},
    "dormant": {"strength_1m_max": 0, "leadership_1m_max": 20, "breadth_pct_positive_1m_max": 40, "confirm_days": 5},
    "priority": ["DORMANT", "EMERGING", "FADING", "MATURE", "ACTIVE", "NEUTRAL"],
}


def make_narrative(s1w=None, t1w=None, l1w=None, b1w=None, s1m=None, l1m=None, b1m=None):
    return {"scores": {
        "1w": {"strength": s1w, "thrust": t1w, "leadership": l1w, "breadth": {"pct_positive": b1w}},
        "1m": {"strength": s1m, "leadership": l1m, "breadth": {"pct_positive": b1m}},
    }}


def test_lifecycle_emerging_conditions():
    n = make_narrative(l1w=50, b1w=70)
    conds = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=80, leadership_delta5d=1.0, cfg=LIFECYCLE_CFG)
    assert conds["emerging"] is True
    # Without a positive leadership delta, EMERGING must not fire.
    conds_no_delta = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=80, leadership_delta5d=-1.0, cfg=LIFECYCLE_CFG)
    assert conds_no_delta["emerging"] is False
    # Missing delta history (first run) -> can't confirm -> not emerging.
    conds_no_history = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=80, leadership_delta5d=None, cfg=LIFECYCLE_CFG)
    assert conds_no_history["emerging"] is False


def test_lifecycle_active_conditions():
    n = make_narrative(s1w=1.0, t1w=0.5, l1w=50, b1w=60)
    conds = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=50, leadership_delta5d=None, cfg=LIFECYCLE_CFG)
    assert conds["active"] is True


def test_lifecycle_mature_conditions():
    n = make_narrative(t1w=-0.5, b1w=55, s1m=5.0, l1m=45)
    conds = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=10, leadership_delta5d=None, cfg=LIFECYCLE_CFG)
    assert conds["mature"] is True


def test_lifecycle_fading_raw_either_condition():
    # thrust<0 AND breadth<50 -> fading
    n1 = make_narrative(t1w=-0.5, b1w=40)
    conds1 = calc_narrative_lifecycle_conditions(n1, thrust_percentile_1w=10, leadership_delta5d=None, cfg=LIFECYCLE_CFG)
    assert conds1["fading_raw"] is True
    # thrust<0 AND leadership_delta5d<=-10 -> fading, even with healthy breadth
    n2 = make_narrative(t1w=-0.5, b1w=90)
    conds2 = calc_narrative_lifecycle_conditions(n2, thrust_percentile_1w=10, leadership_delta5d=-15, cfg=LIFECYCLE_CFG)
    assert conds2["fading_raw"] is True
    # thrust >= 0 -> never fading regardless of breadth/delta
    n3 = make_narrative(t1w=0.5, b1w=10)
    conds3 = calc_narrative_lifecycle_conditions(n3, thrust_percentile_1w=10, leadership_delta5d=-99, cfg=LIFECYCLE_CFG)
    assert conds3["fading_raw"] is False


def test_lifecycle_dormant_raw_uses_1m_breadth_not_1w():
    n = make_narrative(b1w=90, s1m=-1.0, l1m=10, b1m=30)  # 1W breadth healthy, 1M breadth is what matters
    conds = calc_narrative_lifecycle_conditions(n, thrust_percentile_1w=10, leadership_delta5d=None, cfg=LIFECYCLE_CFG)
    assert conds["dormant_raw"] is True


def test_apply_confirm_days_streak_and_confirmation():
    confirmed, streak = apply_confirm_days(True, prev_streak=0, confirm_days=2)
    assert (confirmed, streak) == (False, 1)  # day 1: not confirmed yet
    confirmed2, streak2 = apply_confirm_days(True, prev_streak=1, confirm_days=2)
    assert (confirmed2, streak2) == (True, 2)  # day 2: confirmed
    confirmed3, streak3 = apply_confirm_days(False, prev_streak=5, confirm_days=2)
    assert (confirmed3, streak3) == (False, 0)  # condition drops -> streak resets


def test_select_lifecycle_state_priority_order():
    priority = LIFECYCLE_CFG["priority"]
    # DORMANT beats EMERGING even if both are (implausibly) true.
    conditions = {"dormant_confirmed": True, "emerging": True, "fading_confirmed": True, "mature": True, "active": True}
    assert select_lifecycle_state(conditions, priority) == "DORMANT"
    conditions2 = {"dormant_confirmed": False, "emerging": True, "fading_confirmed": True, "mature": True, "active": True}
    assert select_lifecycle_state(conditions2, priority) == "EMERGING"
    conditions3 = {"dormant_confirmed": False, "emerging": False, "fading_confirmed": False, "mature": False, "active": False}
    assert select_lifecycle_state(conditions3, priority) == "NEUTRAL"


# ── Opportunity Engine (point 20-28) ───────────────────────────

OPPORTUNITY_CFG = {
    "leadership_weights": {"rs_1w": 0.60, "rs_1m": 0.40},
    "leader_entry": {"leadership_score_min": 85, "rs_1w_min": 80},
    "leader_exit": {"leadership_score_max": 80, "rs_1w_max": 75, "confirm_days": 2},
    "fresh_leader": {"entry_window_days": 3, "rs_1w_delta3d_min": 8, "thrust_percentile_1d_min": 85, "thrust_percentile_1w_min": 80},
    "near_emas": {"distance_pct": 4.0, "max_below_ema20_pct": -2.0},
    "extended": {"enter": 5.0, "exit": 4.5},
    "constructive_reset": {"max_below_ema20_pct": -2.0, "qualifying_lifecycles": ["EMERGING", "ACTIVE", "MATURE"]},
    "laggard": {"leadership_score_max": 60, "bottom_pct_of_narrative": 40, "thrust_percentile_1d_max": 80,
                "thrust_percentile_1w_max": 80, "exit_leadership_score_min": 65},
}


def test_stock_leadership_score_weighted_60_40():
    assert calc_stock_leadership_score(100.0, 0.0, OPPORTUNITY_CFG["leadership_weights"]) == 60.0
    assert calc_stock_leadership_score(0.0, 100.0, OPPORTUNITY_CFG["leadership_weights"]) == 40.0


def test_leader_entry_condition():
    assert calc_leader_entry_condition(90, 85, OPPORTUNITY_CFG) is True
    assert calc_leader_entry_condition(84, 85, OPPORTUNITY_CFG) is False  # leadership just below min
    assert calc_leader_entry_condition(90, 79, OPPORTUNITY_CFG) is False  # rs1w just below min


def test_leader_exit_condition():
    assert calc_leader_exit_condition(79, 90, OPPORTUNITY_CFG) is True   # score below max
    assert calc_leader_exit_condition(90, 74, OPPORTUNITY_CFG) is True   # rs1w below max
    assert calc_leader_exit_condition(90, 90, OPPORTUNITY_CFG) is False  # neither


def test_leader_exit_requires_two_consecutive_confirmed_days():
    # Day 1: was leader, exit condition true -> stays leader (grace period)
    state1, age1, streak1 = calc_stock_quality_base_state(
        entry_condition=False, exit_condition=True, prev_quality_state="leader", prev_exit_streak=0, cfg=OPPORTUNITY_CFG)
    assert state1 == "leader"
    assert streak1 == 1
    # Day 2: still exit condition true -> NOW exits to neutral
    state2, age2, streak2 = calc_stock_quality_base_state(
        entry_condition=False, exit_condition=True, prev_quality_state="leader", prev_exit_streak=1, cfg=OPPORTUNITY_CFG)
    assert state2 == "neutral"
    assert age2 == 0


def test_leader_exit_streak_resets_if_condition_clears():
    state, age, streak = calc_stock_quality_base_state(
        entry_condition=False, exit_condition=False, prev_quality_state="leader", prev_exit_streak=1, cfg=OPPORTUNITY_CFG)
    assert state == "leader"
    assert streak == 0


def test_leader_entry_from_neutral():
    state, age, streak = calc_stock_quality_base_state(
        entry_condition=True, exit_condition=False, prev_quality_state="neutral", prev_exit_streak=0, cfg=OPPORTUNITY_CFG)
    assert (state, age, streak) == ("leader", 1, 0)


def test_fresh_leader_within_window_with_trigger():
    label = calc_fresh_leader_label("leader", leader_age_days=1, rs_1w_delta3d=10, thrust_pct_1d=None, thrust_pct_1w=None, cfg=OPPORTUNITY_CFG)
    assert label == "fresh_leader"


def test_fresh_leader_expires_after_3_days():
    # Day 4 of being a leader -> no longer "fresh" even with a strong trigger.
    label = calc_fresh_leader_label("leader", leader_age_days=4, rs_1w_delta3d=50, thrust_pct_1d=99, thrust_pct_1w=99, cfg=OPPORTUNITY_CFG)
    assert label == "leader"


def test_fresh_leader_no_trigger_stays_plain_leader():
    label = calc_fresh_leader_label("leader", leader_age_days=1, rs_1w_delta3d=1, thrust_pct_1d=10, thrust_pct_1w=10, cfg=OPPORTUNITY_CFG)
    assert label == "leader"


def test_near_emas_either_side_plus_max_below_ema20():
    n = OPPORTUNITY_CFG
    assert calc_near_emas(3.0, 10.0, n) is True   # near EMA10 only
    assert calc_near_emas(10.0, 3.0, n) is True   # near EMA20 only
    assert calc_near_emas(10.0, 10.0, n) is False  # neither near
    # Exactly at the -2% floor is fine; below it is not, even if "near".
    assert calc_near_emas(1.0, -2.0, n) is True
    assert calc_near_emas(1.0, -2.01, n) is False


def test_qqq_and_stock_extended_share_same_hysteresis_function():
    # Point 14 + 26 explicitly use the same rule; verifying via the shared
    # calc_extended_with_hysteresis (already covered above) plus a direct
    # sanity check on the stock-facing threshold values.
    e = OPPORTUNITY_CFG["extended"]
    assert calc_extended_with_hysteresis(6.0, False, e["enter"], e["exit"]) is True
    assert calc_extended_with_hysteresis(4.0, True, e["enter"], e["exit"]) is False


def test_constructive_reset_requires_leader_near_emas_ema_order_not_extended():
    lifecycles = {"n1": "ACTIVE", "n2": "FADING"}
    # All conditions satisfied for n1 (qualifying), n2 is FADING (not qualifying).
    result = calc_constructive_reset_narratives(
        quality_state="leader", near_emas=True, extended=False, ema10=110, ema20=100,
        narrative_memberships=["n1", "n2"], narrative_lifecycles=lifecycles, cfg=OPPORTUNITY_CFG)
    assert result == ["n1"]


def test_constructive_reset_blocked_by_not_leader():
    result = calc_constructive_reset_narratives(
        "neutral", True, False, 110, 100, ["n1"], {"n1": "ACTIVE"}, OPPORTUNITY_CFG)
    assert result == []


def test_constructive_reset_blocked_by_extended():
    result = calc_constructive_reset_narratives(
        "leader", True, True, 110, 100, ["n1"], {"n1": "ACTIVE"}, OPPORTUNITY_CFG)
    assert result == []


def test_constructive_reset_blocked_by_ema_order():
    result = calc_constructive_reset_narratives(
        "leader", True, False, 90, 100, ["n1"], {"n1": "ACTIVE"}, OPPORTUNITY_CFG)  # ema10 < ema20
    assert result == []


def test_constructive_reset_requires_qualifying_narrative():
    # Stock is a leader/near_emas/etc., but its ONLY narrative is DORMANT -> no qualifying membership.
    result = calc_constructive_reset_narratives(
        "leader", True, False, 110, 100, ["n1"], {"n1": "DORMANT"}, OPPORTUNITY_CFG)
    assert result == []


def test_narrative_bottom_pct_members():
    narrative = {"members": [
        {"symbol": "A", "w1_pct": -10}, {"symbol": "B", "w1_pct": -5},
        {"symbol": "C", "w1_pct": 0}, {"symbol": "D", "w1_pct": 5}, {"symbol": "E", "w1_pct": 10},
    ]}
    bottom = narrative_bottom_pct_members(narrative, bottom_pct=40)
    assert bottom == {"A", "B"}  # worst 2 of 5 (40%)


def test_laggard_narrative_specific_multi_membership():
    # Same stock: Laggard in narrative A (EMERGING, in bottom), NOT laggard in
    # narrative B (its lifecycle isn't EMERGING/ACTIVE) — point 27/28 example.
    laggard_a = calc_laggard_state(was_laggard=False, leadership_score=40, thrust_pct_1d=50, thrust_pct_1w=50,
                                     lifecycle_ok=True, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    laggard_b = calc_laggard_state(was_laggard=False, leadership_score=40, thrust_pct_1d=50, thrust_pct_1w=50,
                                     lifecycle_ok=False, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    assert laggard_a is True
    assert laggard_b is False


def test_laggard_entry_requires_all_conditions_positively_confirmed():
    # Missing thrust data must NOT silently satisfy the "< 80" requirement.
    laggard = calc_laggard_state(False, leadership_score=40, thrust_pct_1d=None, thrust_pct_1w=50,
                                   lifecycle_ok=True, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    assert laggard is False


def test_laggard_exit_hysteresis_higher_than_entry():
    # Was laggard; leadership climbs to 62 (above entry's <60 but below exit's >=65) -> STAYS laggard.
    still_laggard = calc_laggard_state(True, leadership_score=62, thrust_pct_1d=50, thrust_pct_1w=50,
                                         lifecycle_ok=True, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    assert still_laggard is True
    # Leadership reaches 65 -> exits.
    exits = calc_laggard_state(True, leadership_score=65, thrust_pct_1d=50, thrust_pct_1w=50,
                                 lifecycle_ok=True, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    assert exits is False


def test_laggard_exits_immediately_when_narrative_leaves_emerging_active():
    exits = calc_laggard_state(True, leadership_score=30, thrust_pct_1d=10, thrust_pct_1w=10,
                                 lifecycle_ok=False, in_bottom_pct=True, cfg=OPPORTUNITY_CFG)
    assert exits is False


def test_laggard_exits_when_leaving_bottom_pct():
    exits = calc_laggard_state(True, leadership_score=30, thrust_pct_1d=10, thrust_pct_1w=10,
                                 lifecycle_ok=True, in_bottom_pct=False, cfg=OPPORTUNITY_CFG)
    assert exits is False


# ── Change Detection (point 33-37) ─────────────────────────────

def test_market_regime_change_detection():
    today = {"state": "OFFENSIVE", "score": 70}
    prev = {"state": "SELECTIVE", "score": 60}
    events = detect_market_regime_changes(today, prev, notable_delta=8)
    types = [e["type"] for e in events]
    assert "state_change" in types
    assert "score_change" in types  # delta=10 >= 8


def test_market_regime_no_events_below_notable_delta():
    today = {"state": "OFFENSIVE", "score": 66}
    prev = {"state": "OFFENSIVE", "score": 60}  # delta=6 < 8, same state
    events = detect_market_regime_changes(today, prev, notable_delta=8)
    assert events == []


def test_market_regime_first_run_no_events():
    events = detect_market_regime_changes({"state": "OFFENSIVE", "score": 70}, None, notable_delta=8)
    assert events == []


def test_qqq_health_change_detection_modifier_gained_and_lost():
    today = {"base_state": "HEALTHY", "score": 75, "modifiers": ["EXTENDED"]}
    prev = {"base_state": "HEALTHY", "score": 75, "modifiers": ["NARROWING"]}
    events = detect_qqq_health_changes(today, prev, notable_delta=8)
    types_and_mods = [(e["type"], e.get("modifier")) for e in events]
    assert ("modifier_gained", "EXTENDED") in types_and_mods
    assert ("modifier_lost", "NARROWING") in types_and_mods


def test_narrative_change_detection_new_emerging_rank_and_score():
    today = {"lifecycle_state": "EMERGING", "rank": 2, "momentum_score": 80}
    prev = {"lifecycle_state": "NEUTRAL", "rank": 6, "momentum_score": 65}
    events = detect_narrative_changes("n1", "AI Infra", today, prev,
                                        {"narrative_rank_improve_min": 3, "narrative_score_delta_min": 10})
    types = [e["type"] for e in events]
    assert "new_emerging" in types
    assert "rank_improved" in types
    assert "score_surge" in types


def test_stock_change_detection_new_leader_and_constructive_reset():
    today = {"quality_state": "leader", "constructive_reset_narratives": ["n1"], "extended": False}
    prev = {"quality_state": "neutral", "constructive_reset_narratives": [], "extended": False}
    events = detect_stock_changes("MU", today, prev)
    types = [e["type"] for e in events]
    assert "new_leader" in types
    assert "new_constructive_reset" in types


def test_stock_change_detection_leadership_lost_and_extension_ended():
    today = {"quality_state": "neutral", "constructive_reset_narratives": [], "extended": False}
    prev = {"quality_state": "leader", "constructive_reset_narratives": [], "extended": True}
    events = detect_stock_changes("MU", today, prev)
    types = [e["type"] for e in events]
    assert "leadership_lost" in types
    assert "extension_ended" in types


def test_stock_change_detection_first_run_no_events():
    events = detect_stock_changes("MU", {"quality_state": "leader"}, None)
    assert events == []
