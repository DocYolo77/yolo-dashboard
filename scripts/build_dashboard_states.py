#!/usr/bin/env python3
"""
YOLO Dashboard — V1 Top-down Momentum Operating System State Builder
Single Source of Truth for every score/state/lifecycle/opportunity
computation in the "Market -> Environment -> QQQ Health -> Narratives ->
Stocks -> Location" decision chain. The frontend renders these results; it
does not recompute any formula (point 39 of the V1 spec).

Reads:
  data/snapshot.json          (SPY/QQQ regime + ATR, QQQ breadth history)
  data/market_features.json   (eligible YOLO universe, Full-Market RS/Thrust)
  data/narratives.json        (Narrative baskets, Strength/Thrust/Leadership/Breadth)
  config/narrative_engine.json (market_regime_v1, qqq_health_v1,
                                 narrative_lifecycle_v1, opportunity_v1,
                                 change_detection, history)
  data/history/daily/*.json   (previous days' compact state, for hysteresis/
                                deltas/change-detection — see below)

Writes:
  data/dashboard_state.json          (today's full state, for the frontend)
  data/history/daily/{as_of_date}.json (today's compact snapshot, for
                                         tomorrow's hysteresis/deltas)

All V1 score weights and state thresholds are V1-startparameters (see
config's "_v1_note") — implemented exactly as specified, kept in
config/narrative_engine.json so a later Calibration Engine can replace them
without an architecture change.

── History / hysteresis design ─────────────────────────────────────────
Several V1 rules need day-over-day memory: Leader exit after 2 confirmed
days, Fresh Leader's 3-trading-day window, Extended enter/exit hysteresis,
Narrative FADING after 2 confirmed days, DORMANT after 5, EMERGING's
"Leadership rose vs. 5 trading days ago", rank/score deltas, and Change
Detection generally. Rather than duplicating OHLC history, each trading day
gets ONE compact JSON file under data/history/daily/ (point 34) holding
only what tomorrow's run needs: final states, streak counters, and a
handful of scalar scores/ranks. "N trading days ago" is resolved by walking
back N *files* in that directory (glob + sort), which is automatically
correct across weekends/holidays since only real trading days ever get a
file — no calendar-date arithmetic anywhere in this module.
On first run (no history files yet) every delta/streak-dependent condition
degrades to "not confirmed" rather than inventing a fake value (point 37) —
see FIRST-RUN handling in main().
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from build_narratives import percentile_ranks  # noqa: E402  (reuse: narrative-vs-narrative ranking)


# ─────────────────────────────────────────────
# GENERIC SCORE PRIMITIVES
# ─────────────────────────────────────────────

def renormalized_weighted_sum(values, weights):
    """values/weights: dicts keyed the same way. Only keys present in BOTH
    (value is not None) contribute; their weights are renormalized to sum
    to 1 first. Point 48: a missing subscore must not silently count as 0
    (which would misrepresent/punish the final score) — either renormalize
    over what's available, or (if nothing is available) return None so the
    caller can mark the whole score unavailable."""
    usable = {k: v for k, v in values.items() if v is not None and k in weights}
    if not usable:
        return None
    total_w = sum(weights[k] for k in usable)
    if total_w <= 0:
        return None
    return sum(values[k] * weights[k] for k in usable) / total_w


def clamp_0_100(v):
    if v is None:
        return None
    return round(max(0.0, min(100.0, v)), 1)


def normal_cdf(z):
    """Standard normal CDF Phi(z), via the error function (no scipy dep)."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def series_delta(history, days_back):
    """history[-1] - history[-1-days_back]. None if not enough history —
    the condition using this delta simply can't be confirmed yet (point 48)."""
    if not history or len(history) <= days_back:
        return None
    return round(history[-1] - history[-1 - days_back], 4)


def series_mean_last_n(history, n):
    if not history or len(history) < n:
        return None
    window = history[-n:]
    return sum(window) / len(window)


def percentile_of_last_value(values, min_sample=10):
    """Percentile rank (0-100, percentile_ranks()-style: (count<=today)/n*100)
    of the LAST element within its own list. None if the sample is too thin
    to make a percentile meaningful."""
    if not values or len(values) < min_sample:
        return None
    today = values[-1]
    rank = sum(1 for v in values if v <= today)
    return round(rank / len(values) * 100, 1)


# ─────────────────────────────────────────────
# MARKET REGIME SCORE (point 6-9)
# ─────────────────────────────────────────────

def calc_market_breadth_score(eligible_features, weights):
    def pct_positive(field):
        vals = [f[field] for f in eligible_features if f.get(field) is not None]
        if not vals:
            return None
        return round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)

    components = {
        "pct_above_ema10": pct_positive("ema10_distance_pct"),
        "pct_above_ema20": pct_positive("ema20_distance_pct"),
        "pct_above_sma50": pct_positive("gain_from_sma50_pct"),
    }
    score = renormalized_weighted_sum(components, weights)
    return {"score": clamp_0_100(score), **components}


def calc_market_momentum_score(eligible_features):
    def pct_positive(field):
        vals = [f[field] for f in eligible_features if f.get(field) is not None]
        if not vals:
            return None
        return round(sum(1 for v in vals if v > 0) / len(vals) * 100, 1)

    components = {
        "pct_positive_1w": pct_positive("w1_pct"),
        "pct_positive_1m": pct_positive("m1_pct"),
        "pct_positive_thrust_1w": pct_positive("thrust_1w"),
    }
    vals = [v for v in components.values() if v is not None]
    score = round(sum(vals) / len(vals), 1) if vals else None
    return {"score": score, **components}


def calc_narrative_environment_score(narratives):
    strength_pos, thrust_pos, breadth_vals, leadership_vals = [], [], [], []
    for n in narratives:
        s1w = (n.get("scores") or {}).get("1w") or {}
        if s1w.get("strength") is not None:
            strength_pos.append(1 if s1w["strength"] > 0 else 0)
        if s1w.get("thrust") is not None:
            thrust_pos.append(1 if s1w["thrust"] > 0 else 0)
        bp = (s1w.get("breadth") or {}).get("pct_positive")
        if bp is not None:
            breadth_vals.append(bp)
        if s1w.get("leadership") is not None:
            leadership_vals.append(s1w["leadership"])

    components = {
        "pct_narratives_positive_strength": round(sum(strength_pos) / len(strength_pos) * 100, 1) if strength_pos else None,
        "pct_narratives_positive_thrust": round(sum(thrust_pos) / len(thrust_pos) * 100, 1) if thrust_pos else None,
        "avg_breadth_pct_positive": round(sum(breadth_vals) / len(breadth_vals), 1) if breadth_vals else None,
        "avg_leadership": round(sum(leadership_vals) / len(leadership_vals), 1) if leadership_vals else None,
    }
    vals = [v for v in components.values() if v is not None]
    score = round(sum(vals) / len(vals), 1) if vals else None
    return {"score": score, **components}


def calc_narrative_environment_score_v1_1(narratives):
    """Narrative Environment subscore — V1.1 point 25: reworked to be
    STRUCTURAL, no 1W Thrust component (the old calc_narrative_environment_score
    above is left untouched as reference/fallback, per the V1.1 config-
    versioning discipline). Same simple-average-of-available-%-components
    style as the function it replaces, just fed structural inputs (each
    narrative's narrative_structural_score / Trend Participation / Structural
    Leadership % / 1M breadth from build_narratives.py) instead of 1W
    Strength/Thrust/Breadth/Leadership."""
    structural_vals, trend_participation_vals, structural_leadership_vals, breadth_1m_vals = [], [], [], []
    for n in narratives:
        if n.get("narrative_structural_score") is not None:
            structural_vals.append(n["narrative_structural_score"])
        tp = (n.get("trend_participation") or {}).get("pct_above_rising_sma50")
        if tp is not None:
            trend_participation_vals.append(tp)
        if n.get("structural_leadership_pct") is not None:
            structural_leadership_vals.append(n["structural_leadership_pct"])
        bp = ((n.get("scores") or {}).get("1m") or {}).get("breadth", {}).get("pct_positive")
        if bp is not None:
            breadth_1m_vals.append(bp)

    def avg(vals):
        return round(sum(vals) / len(vals), 1) if vals else None

    components = {
        "avg_structural_score": avg(structural_vals),
        "avg_trend_participation": avg(trend_participation_vals),
        "avg_structural_leadership_pct": avg(structural_leadership_vals),
        "avg_breadth_pct_positive_1m": avg(breadth_1m_vals),
    }
    vals = [v for v in components.values() if v is not None]
    score = round(sum(vals) / len(vals), 1) if vals else None
    return {"score": score, **components}


def calc_market_regime_score(breadth_score, momentum_score, narrative_env_score, final_weights):
    values = {"breadth": breadth_score, "momentum": momentum_score, "narrative_environment": narrative_env_score}
    return clamp_0_100(renormalized_weighted_sum(values, final_weights))


def state_from_thresholds(score, thresholds):
    """thresholds: [{"min":80,"state":"..."}, ...] sorted descending by min."""
    if score is None:
        return "UNAVAILABLE"
    for t in thresholds:
        if score >= t["min"]:
            return t["state"]
    return thresholds[-1]["state"]


# ─────────────────────────────────────────────
# QQQ HEALTH SCORE (point 10-15)
# ─────────────────────────────────────────────

def calc_qqq_trend_score(qqq_regime, trend_points):
    mas = (qqq_regime or {}).get("mas") or {}
    price = (qqq_regime or {}).get("price")
    if price is None or not mas:
        return None
    ema10, ema20, sma50, sma200 = mas.get("ema10"), mas.get("ema20"), mas.get("sma50"), mas.get("sma200")
    pts = 0
    if ema10 is not None and price > ema10:
        pts += trend_points["close_above_ema10"]
    if ema20 is not None and price > ema20:
        pts += trend_points["close_above_ema20"]
    if ema10 is not None and ema20 is not None and ema10 > ema20:
        pts += trend_points["ema10_above_ema20"]
    if sma50 is not None and price > sma50:
        pts += trend_points["close_above_sma50"]
    if sma200 is not None and price > sma200:
        pts += trend_points["close_above_sma200"]
    return min(pts, 100)


def calc_qqq_breadth_score(kma_now, weights):
    values = {
        "ema10": (kma_now or {}).get("ema10"),
        "ema21": (kma_now or {}).get("ema21"),
        "sma50": (kma_now or {}).get("sma50"),
        "sma200": (kma_now or {}).get("sma200"),
    }
    return clamp_0_100(renormalized_weighted_sum(values, weights))


def calc_mco_score(mco_zscore):
    if mco_zscore is None:
        return None
    return round(normal_cdf(mco_zscore) * 100, 1)


def calc_mcsi_score(mcsi_zscore, mcsi_zscore_sma10, weights):
    if mcsi_zscore is None:
        return None
    level = normal_cdf(mcsi_zscore) * 100
    trend = 100.0 if (mcsi_zscore_sma10 is not None and mcsi_zscore >= mcsi_zscore_sma10) else 0.0
    return clamp_0_100(renormalized_weighted_sum({"level": level, "trend": trend}, weights))


def calc_high_low_score(hl_history, n_components, lookback_days):
    if not hl_history or not n_components:
        return None
    window = [v / n_components for v in hl_history[-lookback_days:]]
    return percentile_of_last_value(window)


def calc_qqq_health_score(trend, breadth, mco, mcsi, high_low, weights):
    values = {"trend": trend, "breadth": breadth, "mco": mco, "mcsi": mcsi, "high_low": high_low}
    return clamp_0_100(renormalized_weighted_sum(values, weights))


# ── QQQ Health modifiers (point 14) ──

def calc_extended_with_hysteresis(atr_extension, prev_extended, enter_threshold, exit_threshold):
    """Shared enter/exit hysteresis, used for BOTH QQQ's own Extended
    modifier and each stock's Extended location flag (point 14 + point 26 —
    identical rule, one implementation)."""
    if atr_extension is None:
        return bool(prev_extended)
    if atr_extension > enter_threshold:
        return True
    if atr_extension < exit_threshold:
        return False
    return bool(prev_extended)


def calc_qqq_modifiers(qqq_breadth, close_above_ema20, base_state, prev_extended, atr_extension, cfg):
    """Returns (modifiers: list[str], extended: bool)."""
    m = cfg["modifiers"]
    extended = calc_extended_with_hysteresis(
        atr_extension, prev_extended, m["extended_atr_extension_enter"], m["extended_atr_extension_exit"])

    short_n = m["delta_lookback_days_short"]
    medium_n = m["delta_lookback_days_medium"]
    kma_hist = (qqq_breadth or {}).get("kma_history") or {}

    mco_delta5 = series_delta((qqq_breadth or {}).get("mco_zscore_history"), medium_n)
    mco_delta3 = series_delta((qqq_breadth or {}).get("mco_zscore_history"), short_n)
    mcsi_delta5 = series_delta((qqq_breadth or {}).get("summation_zscore_history"), medium_n)
    ema21_delta5 = series_delta(kma_hist.get("ema21"), medium_n)
    sma50_delta5 = series_delta(kma_hist.get("sma50"), medium_n)
    hl_mean5 = series_mean_last_n((qqq_breadth or {}).get("hl_history"), medium_n)

    negative_conditions = [
        mco_delta5 is not None and mco_delta5 < 0,
        ema21_delta5 is not None and ema21_delta5 <= m["narrowing_breadth_delta_pp"],
        sma50_delta5 is not None and sma50_delta5 <= m["narrowing_breadth_delta_pp"],
        hl_mean5 is not None and hl_mean5 < 0,
    ]
    negative_count = sum(1 for c in negative_conditions if c)
    narrow_or_deteriorate = negative_count >= m["narrowing_min_conditions"]

    positive_conditions = [
        mco_delta3 is not None and mco_delta3 > 0,
        mcsi_delta5 is not None and mcsi_delta5 > 0,
        ema21_delta5 is not None and ema21_delta5 >= m["repairing_breadth_delta_pp"],
        sma50_delta5 is not None and sma50_delta5 >= m["repairing_breadth_delta_pp"],
        hl_mean5 is not None and hl_mean5 > 0,
    ]
    positive_count = sum(1 for c in positive_conditions if c)
    repairing = (base_state != "HEALTHY") and (positive_count >= m["repairing_min_conditions"])

    modifiers = []
    if extended:
        modifiers.append("EXTENDED")
    if narrow_or_deteriorate and repairing:
        pass  # genuine conflict -> no artificial modifier from this group (spec)
    elif narrow_or_deteriorate:
        modifiers.append("NARROWING" if close_above_ema20 else "DETERIORATING")
    elif repairing:
        modifiers.append("REPAIRING")

    return modifiers, extended


# ─────────────────────────────────────────────
# NARRATIVE MOMENTUM SCORE + LIFECYCLE (point 17-18)
# ─────────────────────────────────────────────

def calc_narrative_momentum_scores(narratives, weights):
    """Percentile-ranks Strength_1w/Thrust_1w NARRATIVE-vs-NARRATIVE (reuses
    percentile_ranks — same function, different population than stock RS).
    Returns {narrative_id: {"score":.., "strength_percentile":.., ...}}."""
    strength_pool = {n["id"]: {"v": (n["scores"]["1w"] or {}).get("strength")}
                      for n in narratives if (n["scores"]["1w"] or {}).get("strength") is not None}
    thrust_pool = {n["id"]: {"v": (n["scores"]["1w"] or {}).get("thrust")}
                   for n in narratives if (n["scores"]["1w"] or {}).get("thrust") is not None}
    strength_pct = percentile_ranks(strength_pool, "v")
    thrust_pct = percentile_ranks(thrust_pool, "v")

    out = {}
    for n in narratives:
        nid = n["id"]
        s1w = n["scores"]["1w"] or {}
        values = {
            "strength_percentile": strength_pct.get(nid),
            "thrust_percentile": thrust_pct.get(nid),
            "leadership": s1w.get("leadership"),
            "breadth_pct_positive": (s1w.get("breadth") or {}).get("pct_positive"),
        }
        score = renormalized_weighted_sum(values, weights)
        out[nid] = {"score": round(score, 1) if score is not None else None, **values}
    return out


def calc_narrative_lifecycle_conditions(narrative, thrust_percentile_1w, leadership_delta5d, cfg):
    """Raw (pre-hysteresis) condition booleans for every lifecycle state.
    FADING/DORMANT still need confirm-day gating (apply_confirm_days)."""
    s1w = narrative["scores"].get("1w") or {}
    s1m = narrative["scores"].get("1m") or {}
    strength_1w, thrust_1w = s1w.get("strength"), s1w.get("thrust")
    leadership_1w = s1w.get("leadership")
    breadth_1w = (s1w.get("breadth") or {}).get("pct_positive")
    strength_1m, leadership_1m = s1m.get("strength"), s1m.get("leadership")
    breadth_1m = (s1m.get("breadth") or {}).get("pct_positive")

    e = cfg["emerging"]
    emerging = bool(
        thrust_percentile_1w is not None and thrust_percentile_1w >= e["thrust_percentile_1w_min"] and
        leadership_1w is not None and leadership_1w >= e["leadership_1w_min"] and
        breadth_1w is not None and breadth_1w >= e["breadth_pct_positive_1w_min"] and
        (not e["leadership_delta5d_positive_required"] or (leadership_delta5d is not None and leadership_delta5d > 0))
    )

    a = cfg["active"]
    active = bool(
        strength_1w is not None and strength_1w > a["strength_1w_min"] and
        thrust_1w is not None and thrust_1w >= a["thrust_1w_min"] and
        leadership_1w is not None and leadership_1w >= a["leadership_1w_min"] and
        breadth_1w is not None and breadth_1w >= a["breadth_pct_positive_1w_min"]
    )

    mt = cfg["mature"]
    mature = bool(
        strength_1m is not None and strength_1m > mt["strength_1m_min"] and
        leadership_1m is not None and leadership_1m >= mt["leadership_1m_min"] and
        thrust_1w is not None and thrust_1w < mt["thrust_1w_max"] and
        breadth_1w is not None and breadth_1w >= mt["breadth_pct_positive_1w_min"]
    )

    f = cfg["fading"]
    fading_raw = bool(
        thrust_1w is not None and thrust_1w < f["thrust_1w_max"] and
        ((breadth_1w is not None and breadth_1w < f["breadth_pct_positive_1w_max"]) or
         (leadership_delta5d is not None and leadership_delta5d <= f["leadership_delta5d_max"]))
    )

    d = cfg["dormant"]
    dormant_raw = bool(
        strength_1m is not None and strength_1m <= d["strength_1m_max"] and
        leadership_1m is not None and leadership_1m <= d["leadership_1m_max"] and
        breadth_1m is not None and breadth_1m < d["breadth_pct_positive_1m_max"]
    )

    return {"emerging": emerging, "active": active, "mature": mature, "fading_raw": fading_raw, "dormant_raw": dormant_raw}


def apply_confirm_days(raw_condition, prev_streak, confirm_days):
    """Returns (confirmed: bool, new_streak: int). Streak = consecutive
    trading days (including today) the raw condition has held."""
    new_streak = (prev_streak or 0) + 1 if raw_condition else 0
    confirmed = bool(raw_condition and new_streak >= confirm_days)
    return confirmed, new_streak


def select_lifecycle_state(conditions, priority):
    """conditions must contain 'dormant_confirmed' and 'fading_confirmed' in
    addition to the raw emerging/active/mature keys from
    calc_narrative_lifecycle_conditions."""
    label_map = {
        "DORMANT": conditions["dormant_confirmed"],
        "EMERGING": conditions["emerging"],
        "FADING": conditions["fading_confirmed"],
        "MATURE": conditions["mature"],
        "ACTIVE": conditions["active"],
    }
    for label in priority:
        if label == "NEUTRAL":
            return "NEUTRAL"
        if label_map.get(label):
            return label
    return "NEUTRAL"


# ─────────────────────────────────────────────
# V1.1: STRUCTURAL NARRATIVE LIFECYCLE
# (replaces the 1W-Thrust-heavy V1 lifecycle above as the ACTIVE logic;
# calc_narrative_momentum_scores / calc_narrative_lifecycle_conditions above
# are left untouched as reference, per the V1.1 config-versioning
# discipline — apply_confirm_days and select_lifecycle_state are fully
# generic and reused as-is, unchanged.)
# ─────────────────────────────────────────────

def calc_narrative_structural_deltas(nid, current_structural_score, current_trend_participation,
                                      current_structural_leadership_pct, history_5, history_10):
    """5-/10-trading-day deltas for the structural Lifecycle conditions
    below. Reads the compact per-narrative history record written by THIS
    module (see main()'s history_narratives), same 'walk back N *files*'
    trading-day resolution as every other delta in this file. None (not 0)
    when either side of the delta is unavailable — the condition using it
    simply can't be confirmed yet (point 48's rule, same as series_delta)."""
    def delta(hist, field, current):
        if hist is None or current is None:
            return None
        prev = ((hist.get("narratives") or {}).get(nid) or {}).get(field)
        if prev is None:
            return None
        return round(current - prev, 2)

    return {
        "structural_score_delta5d": delta(history_5, "structural_score", current_structural_score),
        "trend_participation_delta5d": delta(history_5, "trend_participation", current_trend_participation),
        "structural_leadership_delta5d": delta(history_5, "structural_leadership_pct", current_structural_leadership_pct),
        "trend_participation_delta10d": delta(history_10, "trend_participation", current_trend_participation),
        "structural_leadership_delta10d": delta(history_10, "structural_leadership_pct", current_structural_leadership_pct),
    }


def calc_narrative_lifecycle_conditions_v1_1(narrative_structural_score, thrust_percentile_1w, momentum_modifier,
                                              strength_1m, trend_participation, breadth_pct_positive_1m,
                                              deltas, cfg):
    """Raw (pre-hysteresis) condition booleans for the V1.1 structural
    Lifecycle. FADING/DORMANT/MATURE still need confirm-day gating
    (apply_confirm_days, reused unchanged) — EMERGING/ACTIVE are immediate,
    same as in the V1 function this replaces."""
    e = cfg["emerging"]
    emerging = bool(
        narrative_structural_score is not None and
        e["structural_score_min"] <= narrative_structural_score < e["structural_score_max"] and
        thrust_percentile_1w is not None and thrust_percentile_1w >= e["thrust_percentile_1w_min"] and
        deltas["structural_score_delta5d"] is not None and
        deltas["structural_score_delta5d"] >= e["structural_score_delta5d_min"] and
        deltas["trend_participation_delta5d"] is not None and
        deltas["trend_participation_delta5d"] >= e["trend_participation_delta5d_min"] and
        (not e["structural_leadership_delta5d_positive_required"] or
         (deltas["structural_leadership_delta5d"] is not None and deltas["structural_leadership_delta5d"] > 0))
    )

    a = cfg["active"]
    active = bool(
        narrative_structural_score is not None and narrative_structural_score >= a["structural_score_min"] and
        strength_1m is not None and strength_1m >= a["strength_1m_min"] and
        trend_participation is not None and trend_participation >= a["trend_participation_min"] and
        breadth_pct_positive_1m is not None and breadth_pct_positive_1m >= a["breadth_pct_positive_1m_min"]
    )

    # MATURE (V1.1): a structurally strong narrative (still at/above the
    # ACTIVE score bar) whose short-term Thrust has sustainedly cooled —
    # reuses the momentum_modifier already computed in build_narratives.py
    # rather than re-deriving a cooling signal here.
    mature_raw = bool(
        momentum_modifier == "COOLING" and
        narrative_structural_score is not None and narrative_structural_score >= a["structural_score_min"]
    )

    f = cfg["fading"]
    fading_raw = bool(
        narrative_structural_score is not None and narrative_structural_score <= f["structural_score_max"] and
        (
            (deltas["trend_participation_delta10d"] is not None and
             deltas["trend_participation_delta10d"] <= f["trend_participation_delta10d_max"]) or
            (deltas["structural_leadership_delta10d"] is not None and
             deltas["structural_leadership_delta10d"] <= f["structural_leadership_delta10d_max"]) or
            (breadth_pct_positive_1m is not None and breadth_pct_positive_1m <= f["breadth_pct_positive_1m_max"])
        )
    )

    d = cfg["dormant"]
    dormant_raw = bool(
        narrative_structural_score is not None and narrative_structural_score <= d["structural_score_max"] and
        strength_1m is not None and strength_1m <= d["strength_1m_max"] and
        trend_participation is not None and trend_participation <= d["trend_participation_max"]
    )

    return {"emerging": emerging, "active": active, "mature_raw": mature_raw,
            "fading_raw": fading_raw, "dormant_raw": dormant_raw}


# ─────────────────────────────────────────────
# OPPORTUNITY ENGINE (point 20-28)
# ─────────────────────────────────────────────

def calc_stock_leadership_score(rs_1w, rs_1m, weights):
    score = renormalized_weighted_sum({"rs_1w": rs_1w, "rs_1m": rs_1m}, weights)
    return round(score, 1) if score is not None else None


def calc_leader_entry_condition(leadership_score, rs_1w, cfg):
    e = cfg["leader_entry"]
    return bool(
        leadership_score is not None and rs_1w is not None and
        leadership_score >= e["leadership_score_min"] and rs_1w >= e["rs_1w_min"]
    )


def calc_leader_exit_condition(leadership_score, rs_1w, cfg):
    e = cfg["leader_exit"]
    below_score = leadership_score is not None and leadership_score < e["leadership_score_max"]
    below_rs = rs_1w is not None and rs_1w < e["rs_1w_max"]
    return bool(below_score or below_rs)


def calc_stock_quality_base_state(entry_condition, exit_condition, prev_quality_state, prev_exit_streak, cfg):
    """Returns (base_state: 'leader'|'neutral', leader_age_days: int, exit_streak: int).
    leader_age_days counts consecutive trading days (today inclusive) this
    ticker has been a Leader — used by calc_fresh_leader_label for the
    3-day Fresh window, entirely in trading-day counts (no calendar dates,
    so weekends/holidays never need special-casing)."""
    was_leader = prev_quality_state in ("leader", "fresh_leader")
    confirm_days = cfg["leader_exit"]["confirm_days"]

    if not was_leader:
        if entry_condition:
            return "leader", 1, 0
        return "neutral", 0, 0

    if exit_condition:
        new_streak = (prev_exit_streak or 0) + 1
        if new_streak >= confirm_days:
            return "neutral", 0, 0
        return "leader", None, new_streak  # leader_age_days recomputed by caller (carried forward)
    return "leader", None, 0


def calc_fresh_leader_label(base_state, leader_age_days, rs_1w_delta3d, thrust_pct_1d, thrust_pct_1w, cfg):
    if base_state != "leader":
        return base_state
    f = cfg["fresh_leader"]
    if leader_age_days is None or leader_age_days > f["entry_window_days"]:
        return "leader"
    momentum_trigger = bool(
        (rs_1w_delta3d is not None and rs_1w_delta3d >= f["rs_1w_delta3d_min"]) or
        (thrust_pct_1d is not None and thrust_pct_1d >= f["thrust_percentile_1d_min"]) or
        (thrust_pct_1w is not None and thrust_pct_1w >= f["thrust_percentile_1w_min"])
    )
    return "fresh_leader" if momentum_trigger else "leader"


def calc_near_emas(ema10_distance_pct, ema20_distance_pct, cfg):
    n = cfg["near_emas"]
    if ema20_distance_pct is None:
        return False
    near10 = ema10_distance_pct is not None and abs(ema10_distance_pct) <= n["distance_pct"]
    near20 = abs(ema20_distance_pct) <= n["distance_pct"]
    return bool((near10 or near20) and ema20_distance_pct >= n["max_below_ema20_pct"])


def calc_constructive_reset_narratives(quality_state, near_emas, extended, ema10, ema20, narrative_memberships, narrative_lifecycles, cfg):
    if quality_state not in ("leader", "fresh_leader"):
        return []
    if not near_emas or extended:
        return []
    if ema10 is None or ema20 is None or ema10 < ema20:
        return []
    qualifying = set(cfg["constructive_reset"]["qualifying_lifecycles"])
    return [nid for nid in narrative_memberships if narrative_lifecycles.get(nid) in qualifying]


def narrative_bottom_pct_members(narrative, bottom_pct):
    """Set of member symbols in the bottom `bottom_pct`% of THIS narrative
    by 1W performance (worst performers first)."""
    members = [m for m in narrative["members"] if m.get("w1_pct") is not None]
    if not members:
        return set()
    ordered = sorted(members, key=lambda m: m["w1_pct"])
    cutoff = max(1, round(len(ordered) * bottom_pct / 100))
    return {m["symbol"] for m in ordered[:cutoff]}


def calc_laggard_state(was_laggard, leadership_score, thrust_pct_1d, thrust_pct_1w, lifecycle_ok, in_bottom_pct, cfg):
    """Hysteresis: entry needs leadership<60 (+bottom40%+thrust checks, ALL
    positively confirmed, never assumed true on missing data); exit needs
    leadership>=65 OR leaving the bottom 40% OR the narrative leaving
    EMERGING/ACTIVE (that last one is an unconditional, immediate exit —
    a narrative-level fact, not something to hold onto)."""
    l = cfg["laggard"]
    if not lifecycle_ok:
        return False
    if was_laggard:
        if leadership_score is not None and leadership_score >= l["exit_leadership_score_min"]:
            return False
        if not in_bottom_pct:
            return False
        return True
    leadership_ok = leadership_score is not None and leadership_score < l["leadership_score_max"]
    thrust1d_ok = thrust_pct_1d is not None and thrust_pct_1d < l["thrust_percentile_1d_max"]
    thrust1w_ok = thrust_pct_1w is not None and thrust_pct_1w < l["thrust_percentile_1w_max"]
    return bool(leadership_ok and in_bottom_pct and thrust1d_ok and thrust1w_ok)


# ─────────────────────────────────────────────
# V1.1: STRUCTURAL OPPORTUNITY ENGINE
# (replaces calc_stock_leadership_score/calc_leader_entry_condition/
# calc_leader_exit_condition/calc_stock_quality_base_state/
# calc_fresh_leader_label/calc_constructive_reset_narratives/
# calc_laggard_state above as the ACTIVE logic for main(); those V1
# functions are left untouched as reference, per the V1.1 config-
# versioning discipline. calc_near_emas and calc_extended_with_hysteresis
# are unchanged/shared — same shape, just fed opportunity_v1_1 thresholds.)
# ─────────────────────────────────────────────

def calc_leader_entry_condition_v1_1(structural_rs, trend_strength, cfg):
    e = cfg["leader_entry"]
    return bool(
        structural_rs is not None and trend_strength is not None and
        structural_rs >= e["structural_rs_min"] and trend_strength >= e["trend_strength_min"]
    )


def calc_leader_exit_condition_v1_1(structural_rs, trend_strength, sma50_distance_pct, cfg):
    """Any ONE confirmed weakening signal triggers exit (never assumed true
    on missing data) — structural RS falling out of leadership range, trend
    quality deteriorating, or price actually breaking below its SMA50."""
    e = cfg["leader_exit"]
    below_rs = structural_rs is not None and structural_rs < e["structural_rs_max"]
    below_trend = trend_strength is not None and trend_strength < e["trend_strength_max"]
    below_sma50 = sma50_distance_pct is not None and sma50_distance_pct < e["sma50_distance_max_pct"]
    return bool(below_rs or below_trend or below_sma50)


def calc_stock_quality_base_state_v1_1(entry_condition, exit_condition, prev_quality_state, prev_exit_streak, confirm_days):
    """Same hysteresis SHAPE as calc_stock_quality_base_state (V1) above,
    just gated on the V1.1 structural entry/exit conditions instead of
    short-term Leadership Score/RS1W. Returns (base_state: 'leader'|
    'neutral', leader_age_days: int|None, exit_streak: int).

    Deliberately excludes 'recent_leader' from "was previously a leader":
    Recent Leader is a pure DISPLAY overlay applied below (see
    calc_recent_leader_state) when the base state has already resolved to
    'neutral' — it must not feed back into this hysteresis, or a ticker
    would silently revert to full 'leader' the next day just because no
    exit condition re-triggered, without ever re-satisfying entry_condition."""
    was_leader = prev_quality_state in ("leader", "fresh_leader")

    if not was_leader:
        if entry_condition:
            return "leader", 1, 0
        return "neutral", 0, 0

    if exit_condition:
        new_streak = (prev_exit_streak or 0) + 1
        if new_streak >= confirm_days:
            return "neutral", 0, 0
        return "leader", None, new_streak  # leader_age_days recomputed by caller (carried forward)
    return "leader", None, 0


def calc_fresh_leader_label_v1_1(base_state, leader_age_days, rs_1w, thrust_pct_1d, thrust_pct_1w, cfg):
    """Fresh Leader (V1.1 point 9): a NEW leadership entrant (within the
    entry window) additionally showing strong immediate acceleration —
    short-term RS1W or Thrust — on top of already having cleared the
    structural Leader bar. Acceleration is confirmatory here, never the
    entry gate itself (that's calc_leader_entry_condition_v1_1's job)."""
    if base_state != "leader":
        return base_state
    f = cfg["fresh_leader"]
    if leader_age_days is None or leader_age_days > f["entry_window_days"]:
        return "leader"
    momentum_trigger = bool(
        (rs_1w is not None and rs_1w >= f["rs_1w_min"]) or
        (thrust_pct_1d is not None and thrust_pct_1d >= f["thrust_percentile_1d_min"]) or
        (thrust_pct_1w is not None and thrust_pct_1w >= f["thrust_percentile_1w_min"])
    )
    return "fresh_leader" if momentum_trigger else "leader"


def calc_recent_leader_state(quality_state, symbol, history_snapshots, memory_sessions, bootstrap_recent_leader):
    """Recent Leader (V1.1 point 9/10/13): a ticker that ISN'T currently a
    Leader/Fresh Leader but WAS one within the last `memory_sessions`
    trading days — a normal pullback that cost short-term signals shouldn't
    immediately erase Constructive-Reset eligibility (point 9 symptom: "a
    former leader loses Constructive Reset status on a normal pullback").
    Uses REAL daily history once at least `memory_sessions` snapshots exist
    (walking back N *files*, same trading-day resolution as everywhere else
    in this module); before that (cold start / early rollout), falls back
    to the vectorized bootstrap reconstruction build_market_features.py
    computed once from the 260-session price history (bootstrap_recent_leader),
    so the state works from day one instead of waiting ~3 weeks for real
    history to accumulate."""
    if quality_state in ("leader", "fresh_leader"):
        return quality_state
    if len(history_snapshots) >= memory_sessions:
        window = history_snapshots[-memory_sessions:]
        was_leader_recently = any(
            ((snap.get("stocks") or {}).get(symbol) or {}).get("quality_state") in ("leader", "fresh_leader")
            for snap in window)
        return "recent_leader" if was_leader_recently else "neutral"
    return "recent_leader" if bootstrap_recent_leader else "neutral"


def calc_constructive_reset_v1_1(quality_state, ema20_distance_pct, sma50_distance_pct, extended, narrative_memberships, cfg):
    """Constructive Reset (V1.1 point 9/11) — LOOSENED vs V1:
    - Quality gate now includes Recent Leader (not just Leader/Fresh
      Leader): a stock that pulled back out of short-term Leader status a
      few sessions ago is still a legitimate reset candidate.
    - The narrative-lifecycle hard gate is DROPPED entirely — narrative
      membership is returned as CONTEXT only (every current membership,
      unfiltered), not as an eligibility filter (point 9 symptom: leaders
      losing Constructive Reset purely because their narrative's lifecycle
      label changed).
    Location condition: price sits no more than max_below_ema20_pct /
    max_below_sma50_pct BELOW those anchors (a genuine pullback-to-
    structure, not a breakdown) and is not Extended."""
    c = cfg["constructive_reset"]
    if quality_state not in ("leader", "fresh_leader", "recent_leader"):
        return []
    if extended:
        return []
    if ema20_distance_pct is None or sma50_distance_pct is None:
        return []
    if ema20_distance_pct < c["max_below_ema20_pct"] or sma50_distance_pct < c["max_below_sma50_pct"]:
        return []
    return list(narrative_memberships)


def calc_laggard_state_v1_1(was_laggard, structural_rs, thrust_pct_1d, thrust_pct_1w, in_bottom_pct, cfg):
    """Laggard (V1.1 point 15/16): still narrative-relative (bottom X% of
    the narrative by 1W performance — narrative_bottom_pct_members below is
    unchanged) but the entry/exit gate is now structural_rs-based instead
    of the old short-term Leadership Score: a structurally strong stock
    merely resting this week is not a Laggard; a structurally weak stock in
    the bottom of its narrative is. No narrative-lifecycle hard gate in
    V1.1 (dropped, matching Constructive Reset's loosening — see
    opportunity_v1_1.laggard config, which carries no qualifying_lifecycles
    field, unlike V1's)."""
    l = cfg["laggard"]
    if was_laggard:
        if structural_rs is not None and structural_rs >= l["exit_structural_rs_min"]:
            return False
        if not in_bottom_pct:
            return False
        return True
    structural_ok = structural_rs is not None and structural_rs < l["structural_rs_max"]
    thrust1d_ok = thrust_pct_1d is not None and thrust_pct_1d < l["thrust_percentile_1d_max"]
    thrust1w_ok = thrust_pct_1w is not None and thrust_pct_1w < l["thrust_percentile_1w_max"]
    return bool(structural_ok and in_bottom_pct and thrust1d_ok and thrust1w_ok)


# ─────────────────────────────────────────────
# FULL-UNIVERSE: Primary/Secondary Narrative context for Opportunities
# (spec point 11: narrative context on every opportunity row, NEVER a gate
# on quality-state logic — the quality-state functions above take no
# narrative input at all, by construction.)
# ─────────────────────────────────────────────

def build_narrative_membership_index(narratives):
    """{symbol: {"primary_id":, "primary_name":, "secondary_ids": [...],
    "secondary_names": [...], "all_ids": [...]}} from build_narratives.py's
    output (each member dict already carries assignment_priority — see
    build_narratives.py's load_taxonomy/main). A symbol with no resolvable
    assignment_priority on any of its memberships (e.g. legacy/degraded
    data) contributes to all_ids but not primary/secondary — degrades
    gracefully rather than guessing."""
    idx = defaultdict(lambda: {"primary_id": None, "primary_name": None,
                                "secondary_ids": [], "secondary_names": [], "all_ids": []})
    for n in narratives:
        for m in n["members"]:
            entry = idx[m["symbol"]]
            entry["all_ids"].append(n["id"])
            priority = m.get("assignment_priority")
            if priority == "primary":
                entry["primary_id"] = n["id"]
                entry["primary_name"] = n["name"]
            elif priority == "secondary":
                entry["secondary_ids"].append(n["id"])
                entry["secondary_names"].append(n["name"])
    return idx


# ─────────────────────────────────────────────
# CHANGE DETECTION (point 33-37)
# ─────────────────────────────────────────────

def detect_market_regime_changes(today, prev, notable_delta):
    if prev is None:
        return []
    events = []
    if today.get("state") != prev.get("state"):
        events.append({"category": "market_regime", "type": "state_change", "from": prev.get("state"), "to": today.get("state")})
    ts, ps = today.get("score"), prev.get("score")
    if ts is not None and ps is not None and abs(ts - ps) >= notable_delta:
        events.append({"category": "market_regime", "type": "score_change", "from": ps, "to": ts})
    return events


def detect_qqq_health_changes(today, prev, notable_delta):
    if prev is None:
        return []
    events = []
    if today.get("base_state") != prev.get("base_state"):
        events.append({"category": "qqq_health", "type": "base_state_change", "from": prev.get("base_state"), "to": today.get("base_state")})
    today_mods, prev_mods = set(today.get("modifiers", [])), set(prev.get("modifiers", []))
    for gained in sorted(today_mods - prev_mods):
        events.append({"category": "qqq_health", "type": "modifier_gained", "modifier": gained})
    for lost in sorted(prev_mods - today_mods):
        events.append({"category": "qqq_health", "type": "modifier_lost", "modifier": lost})
    ts, ps = today.get("score"), prev.get("score")
    if ts is not None and ps is not None and abs(ts - ps) >= notable_delta:
        events.append({"category": "qqq_health", "type": "score_change", "from": ps, "to": ts})
    return events


def detect_narrative_changes(narrative_id, narrative_name, today, prev, cfg):
    if prev is None:
        return []
    events = []
    lifecycle_new, lifecycle_old = today.get("lifecycle_state"), prev.get("lifecycle_state")
    if lifecycle_new != lifecycle_old and lifecycle_new in ("EMERGING", "ACTIVE", "FADING"):
        events.append({"category": "narrative", "type": f"new_{lifecycle_new.lower()}",
                        "narrative_id": narrative_id, "narrative_name": narrative_name})
    rank_new, rank_old = today.get("rank"), prev.get("rank")
    if rank_new is not None and rank_old is not None and (rank_old - rank_new) >= cfg["narrative_rank_improve_min"]:
        events.append({"category": "narrative", "type": "rank_improved", "narrative_id": narrative_id,
                        "narrative_name": narrative_name, "from_rank": rank_old, "to_rank": rank_new})
    score_new, score_old = today.get("momentum_score"), prev.get("momentum_score")
    if score_new is not None and score_old is not None and (score_new - score_old) >= cfg["narrative_score_delta_min"]:
        events.append({"category": "narrative", "type": "score_surge", "narrative_id": narrative_id,
                        "narrative_name": narrative_name, "from_score": score_old, "to_score": score_new})
    return events


def detect_stock_changes(symbol, today, prev):
    if prev is None:
        return []
    events = []
    today_q, prev_q = today.get("quality_state"), prev.get("quality_state")
    was_leader_like, is_leader_like = prev_q in ("leader", "fresh_leader"), today_q in ("leader", "fresh_leader")
    if is_leader_like and not was_leader_like:
        events.append({"category": "stock", "type": "new_leader" if today_q == "leader" else "new_fresh_leader", "symbol": symbol})
    if was_leader_like and not is_leader_like:
        events.append({"category": "stock", "type": "leadership_lost", "symbol": symbol})
    today_cr, prev_cr = set(today.get("constructive_reset_narratives", [])), set(prev.get("constructive_reset_narratives", []))
    for nid in sorted(today_cr - prev_cr):
        events.append({"category": "stock", "type": "new_constructive_reset", "symbol": symbol, "narrative_id": nid})
    today_ext, prev_ext = bool(today.get("extended")), bool(prev.get("extended"))
    if today_ext and not prev_ext:
        events.append({"category": "stock", "type": "new_extended", "symbol": symbol})
    if prev_ext and not today_ext:
        events.append({"category": "stock", "type": "extension_ended", "symbol": symbol})
    return events


# ─────────────────────────────────────────────
# V1.1: CHANGE DETECTION (new event types, point 30)
# (detect_narrative_changes/detect_stock_changes above left untouched as
# reference; these v1_1 versions are the ones main() calls.)
# ─────────────────────────────────────────────

def detect_narrative_changes_v1_1(narrative_id, narrative_name, today, prev, cfg):
    if prev is None:
        return []
    events = []
    lifecycle_new, lifecycle_old = today.get("lifecycle_state"), prev.get("lifecycle_state")
    if lifecycle_new != lifecycle_old and lifecycle_new in ("EMERGING", "ACTIVE", "FADING", "MATURE"):
        events.append({"category": "narrative", "type": f"new_{lifecycle_new.lower()}",
                        "narrative_id": narrative_id, "narrative_name": narrative_name})
    rank_new, rank_old = today.get("rank"), prev.get("rank")
    if rank_new is not None and rank_old is not None and (rank_old - rank_new) >= cfg["narrative_rank_improve_min"]:
        events.append({"category": "narrative", "type": "rank_improved", "narrative_id": narrative_id,
                        "narrative_name": narrative_name, "from_rank": rank_old, "to_rank": rank_new})
    score_new, score_old = today.get("structural_score"), prev.get("structural_score")
    if score_new is not None and score_old is not None and (score_new - score_old) >= cfg["narrative_score_delta_min"]:
        events.append({"category": "narrative", "type": "structural_score_surge", "narrative_id": narrative_id,
                        "narrative_name": narrative_name, "from_score": score_old, "to_score": score_new})
    mod_new, mod_old = today.get("momentum_modifier"), prev.get("momentum_modifier")
    if mod_new != mod_old:
        if mod_new:
            events.append({"category": "narrative", "type": "modifier_gained", "narrative_id": narrative_id,
                            "narrative_name": narrative_name, "modifier": mod_new})
        if mod_old:
            events.append({"category": "narrative", "type": "modifier_lost", "narrative_id": narrative_id,
                            "narrative_name": narrative_name, "modifier": mod_old})
    return events


def detect_stock_changes_v1_1(symbol, today, prev):
    if prev is None:
        return []
    events = []
    today_q, prev_q = today.get("quality_state"), prev.get("quality_state")
    leader_like = ("leader", "fresh_leader")
    was_leader_like, is_leader_like = prev_q in leader_like, today_q in leader_like
    if is_leader_like and not was_leader_like:
        events.append({"category": "stock", "type": "new_leader" if today_q == "leader" else "new_fresh_leader", "symbol": symbol})
    if was_leader_like and not is_leader_like:
        events.append({"category": "stock", "type": "leadership_lost", "symbol": symbol})
    if today_q == "recent_leader" and prev_q not in ("recent_leader", "leader", "fresh_leader"):
        events.append({"category": "stock", "type": "new_recent_leader", "symbol": symbol})
    if today_q == "neutral" and prev_q == "recent_leader":
        events.append({"category": "stock", "type": "recent_leader_expired", "symbol": symbol})
    today_cr, prev_cr = set(today.get("constructive_reset_narratives", [])), set(prev.get("constructive_reset_narratives", []))
    for nid in sorted(today_cr - prev_cr):
        events.append({"category": "stock", "type": "new_constructive_reset", "symbol": symbol, "narrative_id": nid})
    today_ext, prev_ext = bool(today.get("extended")), bool(prev.get("extended"))
    if today_ext and not prev_ext:
        events.append({"category": "stock", "type": "new_extended", "symbol": symbol})
    if prev_ext and not today_ext:
        events.append({"category": "stock", "type": "extension_ended", "symbol": symbol})
    return events


# ─────────────────────────────────────────────
# V1.1: POST-BUILD SANITY-CHECK REPORT (console only)
# No automatic threshold optimization — this is purely a human-readable
# visual check that Structural Leadership ranks narratives/stocks
# sensibly (e.g. Software/Cybersecurity no longer buried by a 1W-heavy
# score), per the V1.1 spec's explicit requirement.
# ─────────────────────────────────────────────

def print_sanity_check_report(opportunity_items, narrative_report_rows):
    print("\n" + "=" * 60)
    print("🔍 V1.1 SANITY CHECK — Structural Leadership & Narrative Engine")
    print("=" * 60)

    counts = {
        "fresh_leader": sum(1 for it in opportunity_items if it["quality_state"] == "fresh_leader"),
        "leader": sum(1 for it in opportunity_items if it["quality_state"] == "leader"),
        "recent_leader": sum(1 for it in opportunity_items if it["quality_state"] == "recent_leader"),
        "constructive_reset": sum(1 for it in opportunity_items if it["constructive_reset_narratives"]),
        "extended": sum(1 for it in opportunity_items if it["extended"]),
    }
    print(f"\nCounts: Fresh Leader {counts['fresh_leader']} | Leader {counts['leader']} | "
          f"Recent Leader {counts['recent_leader']} | Constructive Reset {counts['constructive_reset']} | "
          f"Extended {counts['extended']}")

    top20_structural_rs = sorted(
        (it for it in opportunity_items if it.get("structural_rs") is not None),
        key=lambda it: -it["structural_rs"])[:20]
    print("\nTop 20 nach Structural RS:")
    for it in top20_structural_rs:
        print(f"  {it['symbol']:<8} Structural RS {it['structural_rs']:>5} | "
              f"Trend Strength {it.get('trend_strength')!s:>5} | Quality {it['quality_state']}")

    top20_trend_strength = sorted(
        (it for it in opportunity_items if it.get("trend_strength") is not None),
        key=lambda it: -it["trend_strength"])[:20]
    print("\nTop 20 nach Trend Strength:")
    for it in top20_trend_strength:
        print(f"  {it['symbol']:<8} Trend Strength {it['trend_strength']:>5} | "
              f"Structural RS {it.get('structural_rs')!s:>5} | Quality {it['quality_state']}")

    print("\nNarrative-Tabelle (Rank | Name | Structural Score | Lifecycle | Modifier | Trend Participation):")
    for row in sorted(narrative_report_rows, key=lambda r: r["rank"] if r["rank"] is not None else 9999):
        print(f"  #{row['rank']:<3} {row['name']:<30} Score {row['structural_score']!s:>6} | "
              f"{row['lifecycle_state']:<10} | {(row['momentum_modifier'] or '-'):<13} | "
              f"Trend Participation {row['trend_participation']}")

    # Full-Universe spec point 27: how well the Full-Universe classification
    # actually reaches the Opportunity Engine's eligible universe.
    n_with_primary = sum(1 for it in opportunity_items if it.get("primary_narrative_id"))
    link_coverage_pct = round(n_with_primary / len(opportunity_items) * 100, 1) if opportunity_items else 100.0
    print(f"\nOPPORTUNITY LINK: {len(opportunity_items)} Opportunity Rows | "
          f"{n_with_primary} mit Primary Narrative | Coverage {link_coverage_pct}%")
    print("=" * 60)


# ─────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────

def load_json(path):
    p = Path(path)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_history_snapshots(history_dir, as_of_date):
    """All previous days' history, oldest -> newest, EXCLUDING as_of_date
    itself (so a same-day rerun never treats itself as 'yesterday')."""
    d = Path(history_dir)
    if not d.exists():
        return []
    files = sorted(p for p in d.glob("*.json") if p.stem != as_of_date)
    out = []
    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                out.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


def prune_old_history(history_dir, retention_days):
    d = Path(history_dir)
    if not d.exists():
        return
    files = sorted(d.glob("*.json"))
    if len(files) <= retention_days:
        return
    for p in files[: len(files) - retention_days]:
        p.unlink(missing_ok=True)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard V1 State Builder")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--snapshot", default="data/snapshot.json")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--narratives", default="data/narratives.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--history-dir", default=None,
                         help="Override config's history.daily_dir (mainly for local dry-runs/tests, "
                              "so they never touch the real data/history/daily/)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🚀 YOLO Dashboard — V1 State Builder")
    print(f"   Zeit: {datetime.now().isoformat()}")
    print("=" * 60)

    cfg = load_json(args.config)
    if cfg is None:
        print("FATAL: config/narrative_engine.json fehlt.", file=sys.stderr)
        sys.exit(1)

    snapshot = load_json(args.snapshot) or {}
    market_features = load_json(args.market_features)
    narratives_data = load_json(args.narratives)

    if market_features is None or narratives_data is None:
        print("FATAL: data/market_features.json oder data/narratives.json fehlt — "
              "State Builder muss nach beiden laufen.", file=sys.stderr)
        sys.exit(1)

    tickers = market_features.get("tickers", {})
    narratives = narratives_data.get("narratives", [])
    as_of_date = (narratives_data.get("meta") or {}).get("date_range", [None, None])[1] \
        or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    history_dir = args.history_dir or cfg["history"]["daily_dir"]
    history_snapshots = load_history_snapshots(history_dir, as_of_date)
    first_run = len(history_snapshots) == 0
    yesterday = history_snapshots[-1] if history_snapshots else None
    history_3 = history_snapshots[-3] if len(history_snapshots) >= 3 else None
    history_5 = history_snapshots[-5] if len(history_snapshots) >= 5 else None
    history_10 = history_snapshots[-10] if len(history_snapshots) >= 10 else None  # V1.1: FADING delta10d
    print(f"\n📅 Stand: {as_of_date} | Historie: {len(history_snapshots)} Tage vorhanden"
          f"{' (FIRST RUN — Baseline)' if first_run else ''}")

    # ── MARKET REGIME ──────────────────────────────────────────────
    mr_cfg = cfg["market_regime_v1"]
    eligible_features = [f for f in tickers.values() if f.get("eligible")]
    breadth_result = calc_market_breadth_score(eligible_features, mr_cfg["breadth_component_weights"])
    momentum_result = calc_market_momentum_score(eligible_features)
    # V1.1 point 25: Narrative Environment subscore is now structural (no 1W
    # Thrust component) — calc_narrative_environment_score (1W-based) above
    # stays defined as reference/fallback but is no longer called here.
    narrative_env_result = calc_narrative_environment_score_v1_1(narratives)
    mr_score = calc_market_regime_score(breadth_result["score"], momentum_result["score"],
                                         narrative_env_result["score"], mr_cfg["final_weights"])
    mr_state = state_from_thresholds(mr_score, mr_cfg["state_thresholds"])
    mr_prev = (yesterday or {}).get("market_regime")
    market_regime = {
        "score": mr_score, "state": mr_state,
        "prev_score": (mr_prev or {}).get("score"),
        "score_delta": round(mr_score - mr_prev["score"], 1) if (mr_score is not None and mr_prev and mr_prev.get("score") is not None) else None,
        "components": {"breadth": breadth_result, "momentum": momentum_result, "narrative_environment": narrative_env_result},
        "eligible_universe_size": len(eligible_features),
        "vix": (snapshot.get("vix") or [None])[0],
    }
    print(f"  ✅ Market Regime: {mr_score} ({mr_state}) | Eligible Universe: {len(eligible_features)}")

    # ── QQQ HEALTH ──────────────────────────────────────────────────
    qh_cfg = cfg["qqq_health_v1"]
    qqq_regime = (snapshot.get("regime") or {}).get("QQQ") or {}
    qqq_breadth = snapshot.get("qqq_breadth") or {}

    trend = calc_qqq_trend_score(qqq_regime, qh_cfg["trend_points"])
    breadth_q = calc_qqq_breadth_score(qqq_breadth.get("kma_now"), qh_cfg["breadth_component_weights"])
    mco = calc_mco_score(qqq_breadth.get("mco_zscore"))
    mcsi = calc_mcsi_score(qqq_breadth.get("summation_zscore"), qqq_breadth.get("summation_zscore_sma10"), qh_cfg["mcsi_weights"])
    high_low = calc_high_low_score(qqq_breadth.get("hl_history"), qqq_breadth.get("n_components"), qh_cfg["high_low_lookback_days"])
    qh_score = calc_qqq_health_score(trend, breadth_q, mco, mcsi, high_low, qh_cfg["final_weights"])
    qh_base_state = state_from_thresholds(qh_score, qh_cfg["base_state_thresholds"])

    close_above_ema20 = bool(qqq_regime.get("price") is not None and (qqq_regime.get("mas") or {}).get("ema20") is not None
                              and qqq_regime["price"] > qqq_regime["mas"]["ema20"])
    prev_qqq = (yesterday or {}).get("qqq_health") or {}
    modifiers, extended = calc_qqq_modifiers(qqq_breadth, close_above_ema20, qh_base_state,
                                              prev_qqq.get("extended"), qqq_regime.get("atr_extension"), qh_cfg)
    mas = qqq_regime.get("mas") or {}
    price = qqq_regime.get("price")
    qqq_health = {
        "score": qh_score, "base_state": qh_base_state, "modifiers": modifiers,
        "prev_score": prev_qqq.get("score"),
        "score_delta": round(qh_score - prev_qqq["score"], 1) if (qh_score is not None and prev_qqq.get("score") is not None) else None,
        "components": {"trend": trend, "breadth": breadth_q, "mco": mco, "mcsi": mcsi, "high_low": high_low},
        "price_structure": {
            "close": price, "ema10": mas.get("ema10"), "ema20": mas.get("ema20"),
            "sma50": mas.get("sma50"), "sma200": mas.get("sma200"),
            "dist_ema10_pct": round((price - mas["ema10"]) / mas["ema10"] * 100, 2) if price and mas.get("ema10") else None,
            "dist_ema20_pct": round((price - mas["ema20"]) / mas["ema20"] * 100, 2) if price and mas.get("ema20") else None,
            "ema10_above_ema20": bool(mas.get("ema10") is not None and mas.get("ema20") is not None and mas["ema10"] > mas["ema20"]),
            "close_above_ema10": bool(price is not None and mas.get("ema10") is not None and price > mas["ema10"]),
            "close_above_ema20": close_above_ema20,
            "close_above_sma50": bool(price is not None and mas.get("sma50") is not None and price > mas["sma50"]),
            "close_above_sma200": bool(price is not None and mas.get("sma200") is not None and price > mas["sma200"]),
            "atr14": qqq_regime.get("atr14"), "atr_pct": qqq_regime.get("atr_pct"),
            "atr_extension": qqq_regime.get("atr_extension"),
        },
    }
    print(f"  ✅ QQQ Health: {qh_score} ({qh_base_state} {' · '.join(modifiers) if modifiers else ''})")

    # ── NARRATIVE STRUCTURAL SCORE + LIFECYCLE (V1.1) ────────────────
    # narrative_structural_score/trend_participation/structural_leadership_pct/
    # momentum_modifier/thrust_percentile_1w are already computed per
    # narrative in build_narratives.py (point 17-22) — this module reads
    # them and adds what needs day-over-day memory: rank, deltas, and
    # confirm-day-gated Lifecycle state. calc_narrative_momentum_scores
    # (V1, 1W-based) above stays defined as reference but is no longer
    # called here.
    nl_cfg = cfg["narrative_lifecycle_v1_1"]

    # Rank by Structural Score (None sorts last), stable tie-break by name.
    ranked = sorted(narratives, key=lambda n: (
        -(n["narrative_structural_score"] if n["narrative_structural_score"] is not None else -1), n["name"]))
    rank_by_id = {n["id"]: i + 1 for i, n in enumerate(ranked)}

    narrative_items = {}
    lifecycle_counts = {"EMERGING": 0, "ACTIVE": 0, "MATURE": 0, "FADING": 0, "DORMANT": 0, "NEUTRAL": 0}
    for n in narratives:
        nid = n["id"]
        structural_score = n["narrative_structural_score"]
        trend_participation = (n.get("trend_participation") or {}).get("pct_above_rising_sma50")
        structural_leadership_pct = n.get("structural_leadership_pct")
        momentum_modifier = n.get("momentum_modifier")
        strength_1m = (n["scores"]["1m"] or {}).get("strength")
        breadth_pct_positive_1m = ((n["scores"]["1m"] or {}).get("breadth") or {}).get("pct_positive")

        prev_n = ((yesterday or {}).get("narratives") or {}).get(nid) or {}
        deltas = calc_narrative_structural_deltas(nid, structural_score, trend_participation,
                                                   structural_leadership_pct, history_5, history_10)

        conditions = calc_narrative_lifecycle_conditions_v1_1(
            structural_score, n.get("thrust_percentile_1w"), momentum_modifier,
            strength_1m, trend_participation, breadth_pct_positive_1m, deltas, nl_cfg)
        fading_confirmed, fading_streak = apply_confirm_days(
            conditions["fading_raw"], prev_n.get("fading_streak"), nl_cfg["fading"]["confirm_days"])
        dormant_confirmed, dormant_streak = apply_confirm_days(
            conditions["dormant_raw"], prev_n.get("dormant_streak"), nl_cfg["dormant"]["confirm_days"])
        mature_confirmed, mature_streak = apply_confirm_days(
            conditions["mature_raw"], prev_n.get("mature_streak"), nl_cfg["mature"]["cooling_confirm_days"])
        final_conditions = {
            "dormant_confirmed": dormant_confirmed, "emerging": conditions["emerging"],
            "fading_confirmed": fading_confirmed, "mature": mature_confirmed, "active": conditions["active"],
        }
        lifecycle_state = select_lifecycle_state(final_conditions, nl_cfg["priority"])
        lifecycle_counts[lifecycle_state] = lifecycle_counts.get(lifecycle_state, 0) + 1

        rank = rank_by_id[nid]
        prev_rank = prev_n.get("rank")
        narrative_items[nid] = {
            "lifecycle_state": lifecycle_state,
            "structural_score": structural_score,
            "structural_score_delta5d": deltas["structural_score_delta5d"],
            "score_delta": round(structural_score - prev_n["structural_score"], 1)
                if (structural_score is not None and prev_n.get("structural_score") is not None) else None,
            "rank": rank, "prev_rank": prev_rank,
            "rank_delta": (prev_rank - rank) if (prev_rank is not None) else None,
            "trend_participation": trend_participation,
            "structural_leadership_pct": structural_leadership_pct,
            "momentum_modifier": momentum_modifier,
            "_fading_streak": fading_streak, "_dormant_streak": dormant_streak,  # persisted to history only
            "_mature_streak": mature_streak,
        }
    print(f"  ✅ Narrative Lifecycle: {lifecycle_counts}")

    # ── OPPORTUNITY ENGINE (V1.1) ─────────────────────────────────────
    # calc_stock_leadership_score/calc_leader_entry_condition/
    # calc_leader_exit_condition/calc_stock_quality_base_state/
    # calc_fresh_leader_label/calc_constructive_reset_narratives/
    # calc_laggard_state (V1, RS1W-gated) stay defined as reference but are
    # no longer called here — see the _v1_1 versions used below.
    op_cfg = cfg["opportunity_v1_1"]
    narrative_by_id = {n["id"]: n for n in narratives}
    memberships = {}  # symbol -> [narrative_id, ...]
    for n in narratives:
        for m in n["members"]:
            memberships.setdefault(m["symbol"], []).append(n["id"])
    narrative_membership_idx = build_narrative_membership_index(narratives)

    bottom_members_by_narrative = {n["id"]: narrative_bottom_pct_members(n, op_cfg["laggard"]["bottom_pct_of_narrative"]) for n in narratives}

    prev_stocks = (yesterday or {}).get("stocks") or {}
    memory_sessions = op_cfg["recent_leader"]["memory_sessions"]

    # V1.1 point 12: iterate the ENTIRE eligible universe from
    # market_features.json (not just narrative members) — a stock with no
    # narrative membership is still a valid Leader/Constructive-Reset
    # candidate; narratives are attached below purely as context tags.
    opportunity_items = []
    counts = {"fresh_leaders": 0, "leaders": 0, "recent_leaders": 0, "constructive_resets": 0, "extended": 0, "laggards": 0}
    for symbol, f in tickers.items():
        if not f.get("eligible"):
            continue  # Opportunities operates on the eligible YOLO universe only
        narrative_ids = memberships.get(symbol, [])

        structural_rs, trend_strength = f.get("structural_rs"), f.get("trend_strength")
        rs_1w, rs_1m = f.get("rs_percentile_1w"), f.get("rs_percentile_1m")
        thrust_pct_1d, thrust_pct_1w = f.get("thrust_percentile_1d"), f.get("thrust_percentile_1w")
        sma50_distance_pct = f.get("sma50_distance_pct")

        prev_stock = prev_stocks.get(symbol) or {}
        entry_cond = calc_leader_entry_condition_v1_1(structural_rs, trend_strength, op_cfg)
        exit_cond = calc_leader_exit_condition_v1_1(structural_rs, trend_strength, sma50_distance_pct, op_cfg)
        base_state, leader_age_days, exit_streak = calc_stock_quality_base_state_v1_1(
            entry_cond, exit_cond, prev_stock.get("quality_state"), prev_stock.get("exit_streak"),
            op_cfg["leader_exit"]["confirm_days"])
        if leader_age_days is None:  # carried forward from yesterday (still-leader, no reset)
            leader_age_days = (prev_stock.get("leader_age_days") or 0) + 1

        quality_state = calc_fresh_leader_label_v1_1(base_state, leader_age_days, rs_1w, thrust_pct_1d, thrust_pct_1w, op_cfg)
        quality_state = calc_recent_leader_state(quality_state, symbol, history_snapshots, memory_sessions,
                                                  bool(f.get("bootstrap_recent_leader")))

        stock_extended = calc_extended_with_hysteresis(f.get("atr_extension"), prev_stock.get("extended"),
                                                         op_cfg["extended"]["enter"], op_cfg["extended"]["exit"])
        near_emas = calc_near_emas(f.get("ema10_distance_pct"), f.get("ema20_distance_pct"), op_cfg)
        cr_narratives = calc_constructive_reset_v1_1(
            quality_state, f.get("ema20_distance_pct"), sma50_distance_pct, stock_extended, narrative_ids, op_cfg)

        prev_laggards = set(prev_stock.get("laggard_narratives", []))
        laggard_narratives = []
        for nid in narrative_ids:
            in_bottom = symbol in bottom_members_by_narrative.get(nid, set())
            was_laggard = nid in prev_laggards
            if calc_laggard_state_v1_1(was_laggard, structural_rs, thrust_pct_1d, thrust_pct_1w, in_bottom, op_cfg):
                laggard_narratives.append(nid)

        narr_ctx = narrative_membership_idx.get(symbol) or {
            "primary_id": None, "primary_name": None, "secondary_ids": [], "secondary_names": [], "all_ids": [],
        }
        item = {
            "symbol": symbol, "narratives": narrative_ids,
            # Full-Universe spec point 11: narrative context on every
            # Opportunity row — purely informational, never a gate on
            # quality_state/near_emas/extended/etc above (all already
            # computed before this point, from market_features + hysteresis
            # only).
            "narrative_ids": narrative_ids,
            "primary_narrative_id": narr_ctx["primary_id"], "primary_narrative_name": narr_ctx["primary_name"],
            "secondary_narrative_ids": narr_ctx["secondary_ids"], "secondary_narrative_names": narr_ctx["secondary_names"],
            "quality_state": quality_state, "near_emas": near_emas, "extended": stock_extended,
            "constructive_reset_narratives": cr_narratives, "laggard_narratives": laggard_narratives,
            "structural_rs": structural_rs, "trend_strength": trend_strength,
            "sma50_distance_pct": sma50_distance_pct, "sma50_slope_20d_pct": f.get("sma50_slope_20d_pct"),
            "pct_sessions_above_sma50_20d": f.get("pct_sessions_above_sma50_20d"),
            # Acceleration metrics — still shown, explicitly non-gating (V1.1 point 6).
            "rs_1w": rs_1w, "rs_1m": rs_1m,
            "thrust_percentile_1d": thrust_pct_1d, "thrust_percentile_1w": thrust_pct_1w,
            "ema10_distance_pct": f.get("ema10_distance_pct"), "ema20_distance_pct": f.get("ema20_distance_pct"),
            "atr_extension": f.get("atr_extension"), "w1_pct": f.get("w1_pct"), "m1_pct": f.get("m1_pct"),
            "price": f.get("close"), "market_cap": f.get("market_cap"),
            "_leader_age_days": leader_age_days, "_exit_streak": exit_streak,  # persisted to history only
        }
        opportunity_items.append(item)

        if quality_state == "fresh_leader":
            counts["fresh_leaders"] += 1
        elif quality_state == "leader":
            counts["leaders"] += 1
        elif quality_state == "recent_leader":
            counts["recent_leaders"] += 1
        if cr_narratives:
            counts["constructive_resets"] += 1
        if stock_extended:
            counts["extended"] += 1
        if laggard_narratives:
            counts["laggards"] += 1

    opportunity_items.sort(key=lambda it: it["symbol"])
    print(f"  ✅ Opportunities: {len(opportunity_items)} Ticker (gesamtes eligible Universe) | {counts}")

    # ── CHANGE DETECTION (V1.1 event types) ──────────────────────────
    # detect_narrative_changes/detect_stock_changes (V1) stay defined as
    # reference but are no longer called here — see the _v1_1 versions.
    cd_cfg = cfg["change_detection"]
    events = []
    if not first_run:
        events += detect_market_regime_changes(market_regime, mr_prev, mr_cfg["notable_score_delta"])
        events += detect_qqq_health_changes(qqq_health, prev_qqq, qh_cfg["notable_score_delta"])
        for nid, item in narrative_items.items():
            prev_n = ((yesterday or {}).get("narratives") or {}).get(nid)
            events += detect_narrative_changes_v1_1(nid, narrative_by_id[nid]["name"], item, prev_n, cd_cfg)
        for item in opportunity_items:
            prev_stock = prev_stocks.get(item["symbol"])
            events += detect_stock_changes_v1_1(item["symbol"], item, prev_stock)

    what_changed = {"first_run": first_run, "events": events}
    print(f"  ✅ Change Detection: {len(events)} Events" + (" (first run, keine Vergleichsbasis)" if first_run else ""))

    # ── ASSEMBLE dashboard_state.json ───────────────────────────────
    narrative_items_public = {nid: {k: v for k, v in item.items() if not k.startswith("_")} for nid, item in narrative_items.items()}
    opportunity_items_public = [{k: v for k, v in item.items() if not k.startswith("_")} for item in opportunity_items]

    dashboard_state = {
        "meta": {
            "as_of_date": as_of_date,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "dashboard_state_version": cfg.get("dashboard_state_version", "v1"),
            "first_run": first_run,
        },
        "market_regime": market_regime,
        "qqq_health": qqq_health,
        "narratives": {"lifecycle_counts": lifecycle_counts, "items": narrative_items_public},
        "opportunities": {"counts": counts, "items": opportunity_items_public},
        "what_changed": what_changed,
    }

    out_path = out_dir / "dashboard_state.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_state, f, indent=2, ensure_ascii=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"\n✅ dashboard_state.json geschrieben → {out_path} ({size_kb:.0f} KB)")

    # V1.1 point 35: post-build sanity check (console only, no automatic
    # threshold optimization) — visually confirm structurally-dominant
    # narratives (e.g. Software/Cybersecurity) now rank sensibly.
    narrative_report_rows = [
        {"rank": item["rank"], "name": narrative_by_id[nid]["name"], "structural_score": item["structural_score"],
         "lifecycle_state": item["lifecycle_state"], "momentum_modifier": item["momentum_modifier"],
         "trend_participation": item["trend_participation"]}
        for nid, item in narrative_items.items()
    ]
    print_sanity_check_report(opportunity_items, narrative_report_rows)

    # ── PERSIST TODAY'S COMPACT HISTORY ─────────────────────────────
    history_stocks = {}
    for item in opportunity_items:
        history_stocks[item["symbol"]] = {
            "quality_state": item["quality_state"], "leader_age_days": item["_leader_age_days"],
            "exit_streak": item["_exit_streak"], "extended": item["extended"],
            "constructive_reset_narratives": item["constructive_reset_narratives"],
            "laggard_narratives": item["laggard_narratives"], "rs_1w": item["rs_1w"],
        }
    history_narratives = {}
    for nid, item in narrative_items.items():
        history_narratives[nid] = {
            "lifecycle_state": item["lifecycle_state"], "structural_score": item["structural_score"],
            "trend_participation": item["trend_participation"],
            "structural_leadership_pct": item["structural_leadership_pct"],
            "momentum_modifier": item["momentum_modifier"],
            "rank": item["rank"],
            "fading_streak": item["_fading_streak"], "dormant_streak": item["_dormant_streak"],
            "mature_streak": item["_mature_streak"],
        }
    today_history = {
        "date": as_of_date,
        "market_regime": {"score": market_regime["score"], "state": market_regime["state"]},
        "qqq_health": {"score": qqq_health["score"], "base_state": qqq_health["base_state"],
                       "modifiers": qqq_health["modifiers"], "extended": extended},
        "narratives": history_narratives,
        "stocks": history_stocks,
    }
    history_dir_path = Path(history_dir)
    history_dir_path.mkdir(parents=True, exist_ok=True)
    with open(history_dir_path / f"{as_of_date}.json", "w", encoding="utf-8") as f:
        json.dump(today_history, f, indent=2, ensure_ascii=False)
    print(f"✅ Tages-Historie geschrieben → {history_dir_path / (as_of_date + '.json')}")

    prune_old_history(history_dir, cfg["history"]["retention_days"])
    print("=" * 60)


if __name__ == "__main__":
    main()
