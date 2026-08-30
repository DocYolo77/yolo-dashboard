"""
Tests for the Calibration-aware Opportunities UI v1 (scripts/build_dashboard_states.py):
the new Structure badges (EMA10/EMA20 Pullback, Resetting, Extended) and the
Quality relabeling (Leader/Neutral/Laggard), all driven by config's
opportunity_calibration_v2 block -- no new formula/score, no threshold
optimization, pure candidate-rule implementation per the given calibration.
All synthetic data — no network / no file I/O required.
Run with: pytest tests/ -v
"""

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_dashboard_states import (  # noqa: E402
    calc_ema_pullback,
    calc_extended_v2,
    calc_resetting,
    calc_extension_peak_v2,
    calc_quality_v2,
    compute_opportunity_v2_prep_fields,
)

CAL = {
    "ema10_pullback_min": -5.0, "ema10_pullback_max": 5.0,
    "ema20_pullback_min": -5.0, "ema20_pullback_max": 5.0,
    "extended_atr_threshold": 8.0,
    "reset_previous_extension_min": 8.0,
    "reset_current_extension_max": 4.0,
    "reset_ema_distance_max": 0.0,
}


# ── calc_ema_pullback: candidate rule v1, boundaries inclusive ──

def test_ema_pullback_true_within_range():
    assert calc_ema_pullback(3.0, -5.0, 5.0) is True
    assert calc_ema_pullback(-3.0, -5.0, 5.0) is True


def test_ema_pullback_boundaries_inclusive():
    assert calc_ema_pullback(5.0, -5.0, 5.0) is True
    assert calc_ema_pullback(-5.0, -5.0, 5.0) is True


def test_ema_pullback_false_just_outside_range():
    assert calc_ema_pullback(5.01, -5.0, 5.0) is False
    assert calc_ema_pullback(-5.01, -5.0, 5.0) is False


def test_ema_pullback_none_distance_is_false_not_error():
    assert calc_ema_pullback(None, -5.0, 5.0) is False


# ── calc_extended_v2: single threshold, no hysteresis ──

def test_extended_v2_true_exactly_at_threshold():
    assert calc_extended_v2(8.0, 8.0) is True


def test_extended_v2_true_above_threshold():
    assert calc_extended_v2(12.5, 8.0) is True


def test_extended_v2_false_below_threshold():
    assert calc_extended_v2(7.99, 8.0) is False


def test_extended_v2_none_is_false():
    assert calc_extended_v2(None, 8.0) is False


# ── calc_resetting: Previous Peak >= 8 AND Current < 4 AND EMA distance <= 0 ──

def test_resetting_ema10_fires_when_all_conditions_met():
    r10, r20 = calc_resetting(previous_extension_peak=8.0, current_atr_extension=3.9,
                               ema10_distance_pct=0.0, ema20_distance_pct=5.0, cfg=CAL)
    assert r10 is True
    assert r20 is False


def test_resetting_ema20_fires_independently_of_ema10():
    r10, r20 = calc_resetting(previous_extension_peak=9.0, current_atr_extension=1.0,
                               ema10_distance_pct=5.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert r10 is False
    assert r20 is True


def test_resetting_both_can_fire_simultaneously():
    r10, r20 = calc_resetting(previous_extension_peak=10.0, current_atr_extension=0.0,
                               ema10_distance_pct=-2.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert r10 is True
    assert r20 is True


def test_resetting_no_trigger_without_historical_extension():
    # Previous peak below the 8.0 minimum -- never was extended enough.
    r10, r20 = calc_resetting(previous_extension_peak=7.9, current_atr_extension=1.0,
                               ema10_distance_pct=-1.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert (r10, r20) == (False, False)


def test_resetting_no_trigger_when_missing_peak_history():
    # None peak (never tracked / no prior day) must never be treated as "extended enough".
    r10, r20 = calc_resetting(previous_extension_peak=None, current_atr_extension=1.0,
                               ema10_distance_pct=-1.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert (r10, r20) == (False, False)


def test_resetting_no_trigger_when_current_extension_missing():
    r10, r20 = calc_resetting(previous_extension_peak=9.0, current_atr_extension=None,
                               ema10_distance_pct=-1.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert (r10, r20) == (False, False)


def test_resetting_no_trigger_when_current_extension_still_too_high():
    r10, r20 = calc_resetting(previous_extension_peak=9.0, current_atr_extension=4.0,  # not < 4.0
                               ema10_distance_pct=-1.0, ema20_distance_pct=-1.0, cfg=CAL)
    assert (r10, r20) == (False, False)


def test_resetting_no_trigger_when_ema_distance_positive():
    r10, r20 = calc_resetting(previous_extension_peak=9.0, current_atr_extension=1.0,
                               ema10_distance_pct=0.01, ema20_distance_pct=0.01, cfg=CAL)
    assert (r10, r20) == (False, False)


def test_resetting_ema_distance_zero_is_inclusive():
    r10, r20 = calc_resetting(previous_extension_peak=9.0, current_atr_extension=1.0,
                               ema10_distance_pct=0.0, ema20_distance_pct=0.0, cfg=CAL)
    assert (r10, r20) == (True, True)


# ── calc_extension_peak_v2: day-over-day memory feeding tomorrow's "previous peak" ──

def test_extension_peak_starts_at_first_observed_value():
    assert calc_extension_peak_v2(prev_peak=None, current_atr_extension=6.0, resetting=False) == 6.0


def test_extension_peak_tracks_running_max():
    assert calc_extension_peak_v2(prev_peak=6.0, current_atr_extension=9.0, resetting=False) == 9.0
    assert calc_extension_peak_v2(prev_peak=9.0, current_atr_extension=7.0, resetting=False) == 9.0  # doesn't drop


def test_extension_peak_clears_once_resetting_fires():
    # Once Resetting fires, the episode is closed -- peak restarts from
    # today's (now-low) reading instead of persisting the old high forever.
    assert calc_extension_peak_v2(prev_peak=9.0, current_atr_extension=2.0, resetting=True) == 2.0


def test_extension_peak_keeps_memory_when_current_is_missing():
    assert calc_extension_peak_v2(prev_peak=9.0, current_atr_extension=None, resetting=False) == 9.0


# ── calc_quality_v2: pure relabeling, no new threshold ──

def test_quality_v2_leader_bucket_covers_all_leader_gradations():
    for qs in ("fresh_leader", "leader", "recent_leader"):
        assert calc_quality_v2(qs, laggard_narratives=[]) == "leader"


def test_quality_v2_laggard_when_flagged_in_any_narrative():
    assert calc_quality_v2("neutral", laggard_narratives=["memory"]) == "laggard"


def test_quality_v2_neutral_otherwise():
    assert calc_quality_v2("neutral", laggard_narratives=[]) == "neutral"


def test_quality_v2_leader_status_takes_precedence_over_stale_laggard_list():
    # A stock can't be both -- a currently-leading stock's quality_state is
    # never 'neutral' while laggard_narratives is non-empty in practice, but
    # the bucketing itself must still resolve deterministically to leader.
    assert calc_quality_v2("leader", laggard_narratives=["memory"]) == "leader"


# ── compute_opportunity_v2_prep_fields: full integration ──

def test_prep_fields_above_ema_derived_from_existing_distance_fields():
    result = compute_opportunity_v2_prep_fields(2.0, -1.0, 3.0, "neutral", [], None, CAL)
    assert result["above_ema10"] is True
    assert result["above_ema20"] is False


def test_prep_fields_above_ema_boundary_zero_is_not_above():
    result = compute_opportunity_v2_prep_fields(0.0, 0.0, 3.0, "neutral", [], None, CAL)
    assert result["above_ema10"] is False
    assert result["above_ema20"] is False


def test_prep_fields_above_ema_none_when_distance_missing():
    result = compute_opportunity_v2_prep_fields(None, None, None, "neutral", [], None, CAL)
    assert result["above_ema10"] is None
    assert result["above_ema20"] is None


def test_prep_fields_state_v2_always_none_badges_are_not_a_composite_score():
    result = compute_opportunity_v2_prep_fields(0.0, 0.0, 12.0, "leader", [], 9.0, CAL)
    assert result["state_v2"] is None


def test_prep_fields_both_pullback_badges_can_be_true_simultaneously():
    result = compute_opportunity_v2_prep_fields(1.0, -2.0, 1.0, "neutral", [], None, CAL)
    assert result["ema10_pullback"] is True
    assert result["ema20_pullback"] is True


def test_prep_fields_extended_true_at_exactly_8():
    result = compute_opportunity_v2_prep_fields(10.0, 10.0, 8.0, "neutral", [], None, CAL)
    assert result["extended_v2"] is True


def test_prep_fields_extended_false_under_8():
    result = compute_opportunity_v2_prep_fields(10.0, 10.0, 7.99, "neutral", [], None, CAL)
    assert result["extended_v2"] is False


def test_prep_fields_resetting_and_pullback_can_coexist():
    # A stock resetting into an EMA10 pullback zone -- both markers true at once.
    result = compute_opportunity_v2_prep_fields(
        ema10_distance_pct=-1.0, ema20_distance_pct=8.0, atr_extension=1.0,
        quality_state="leader", laggard_narratives=[], prev_extension_peak=9.0, cfg=CAL)
    assert result["resetting"] is True
    assert result["ema10_pullback"] is True
    assert result["ema20_pullback"] is False  # 8.0% is outside +-5%
    assert result["repeat_offender_history_v2"] == "ema10"


def test_prep_fields_resetting_trigger_records_both_when_both_fire():
    result = compute_opportunity_v2_prep_fields(
        ema10_distance_pct=-1.0, ema20_distance_pct=-1.0, atr_extension=1.0,
        quality_state="neutral", laggard_narratives=[], prev_extension_peak=9.0, cfg=CAL)
    assert result["resetting"] is True
    assert result["repeat_offender_history_v2"] == "both"


def test_prep_fields_no_false_badges_when_all_inputs_missing():
    result = compute_opportunity_v2_prep_fields(None, None, None, "neutral", [], None, CAL)
    assert result["ema10_pullback"] is False
    assert result["ema20_pullback"] is False
    assert result["extended_v2"] is False
    assert result["resetting"] is False
    assert result["repeat_offender_history_v2"] is None


def test_prep_fields_extension_peak_persists_for_next_day():
    result = compute_opportunity_v2_prep_fields(0.0, 0.0, 9.5, "neutral", [], 6.0, CAL)
    assert result["extension_peak_history_v2"] == 9.5  # running max(6.0, 9.5)


def test_prep_fields_quality_v2_leader_extended_can_coexist():
    # Explicitly required by the spec: an Extended stock can still be a Leader.
    result = compute_opportunity_v2_prep_fields(10.0, 10.0, 12.0, "leader", [], None, CAL)
    assert result["quality_v2"] == "leader"
    assert result["extended_v2"] is True


def test_prep_fields_signature_takes_the_documented_inputs_only():
    params = list(inspect.signature(compute_opportunity_v2_prep_fields).parameters)
    assert params == ["ema10_distance_pct", "ema20_distance_pct", "atr_extension",
                       "quality_state", "laggard_narratives", "prev_extension_peak", "cfg"]
