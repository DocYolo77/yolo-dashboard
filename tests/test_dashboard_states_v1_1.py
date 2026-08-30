"""
Tests for scripts/build_dashboard_states.py's V1.1 Structural Leadership &
Narrative Engine additions: structural Market-Regime narrative-environment
subscore, structural Narrative Lifecycle, the V1.1 Opportunity Engine
quality-state machine (Leader/Fresh Leader/Recent Leader), Constructive
Reset, Laggard, and the new Change Detection event types. All synthetic
data — no network / no file I/O required for the functions under test.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_dashboard_states import (  # noqa: E402
    calc_narrative_environment_score_v1_1,
    calc_narrative_structural_deltas,
    calc_narrative_lifecycle_conditions_v1_1,
    calc_leader_entry_condition_v1_1,
    calc_leader_exit_condition_v1_1,
    calc_stock_quality_base_state_v1_1,
    calc_fresh_leader_label_v1_1,
    calc_recent_leader_state,
    calc_constructive_reset_v1_1,
    calc_laggard_state_v1_1,
    detect_narrative_changes_v1_1,
    detect_stock_changes_v1_1,
    apply_confirm_days,
    select_lifecycle_state,
    build_narrative_membership_index,
)


# ── Market Regime: structural Narrative Environment subscore ────

def test_narrative_environment_score_v1_1_averages_structural_components():
    narratives = [
        {"narrative_structural_score": 80.0, "trend_participation": {"pct_above_rising_sma50": 60.0},
         "structural_leadership_pct": 50.0, "scores": {"1m": {"breadth": {"pct_positive": 70.0}}}},
        {"narrative_structural_score": 40.0, "trend_participation": {"pct_above_rising_sma50": 20.0},
         "structural_leadership_pct": 10.0, "scores": {"1m": {"breadth": {"pct_positive": 30.0}}}},
    ]
    result = calc_narrative_environment_score_v1_1(narratives)
    assert result["avg_structural_score"] == 60.0
    assert result["avg_trend_participation"] == 40.0
    assert result["avg_structural_leadership_pct"] == 30.0
    assert result["avg_breadth_pct_positive_1m"] == 50.0
    assert result["score"] == pytest.approx(45.0)


def test_narrative_environment_score_v1_1_no_data_returns_none_score():
    result = calc_narrative_environment_score_v1_1([{"narrative_structural_score": None,
                                                       "trend_participation": {}, "scores": {}}])
    assert result["score"] is None


# ── Narrative structural deltas ──────────────────────────────────

def test_narrative_structural_deltas_computed_from_history():
    history_5 = {"narratives": {"n1": {"structural_score": 50.0, "trend_participation": 40.0,
                                        "structural_leadership_pct": 30.0}}}
    history_10 = {"narratives": {"n1": {"structural_score": 45.0, "trend_participation": 35.0,
                                         "structural_leadership_pct": 20.0}}}
    deltas = calc_narrative_structural_deltas("n1", 60.0, 55.0, 45.0, history_5, history_10)
    assert deltas["structural_score_delta5d"] == 10.0
    assert deltas["trend_participation_delta5d"] == 15.0
    assert deltas["structural_leadership_delta5d"] == 15.0
    assert deltas["trend_participation_delta10d"] == 20.0
    assert deltas["structural_leadership_delta10d"] == 25.0


def test_narrative_structural_deltas_none_when_history_missing():
    deltas = calc_narrative_structural_deltas("n1", 60.0, 55.0, 45.0, None, None)
    assert all(v is None for v in deltas.values())


def test_narrative_structural_deltas_none_when_narrative_new_to_history():
    history_5 = {"narratives": {}}  # narrative not present 5 days ago (new)
    deltas = calc_narrative_structural_deltas("n1", 60.0, 55.0, 45.0, history_5, None)
    assert deltas["structural_score_delta5d"] is None


# ── Narrative Lifecycle conditions (V1.1) ────────────────────────

LC_CFG = {
    "emerging": {
        "structural_score_min": 40, "structural_score_max": 65,
        "thrust_percentile_1w_min": 75,
        "structural_score_delta5d_min": 5,
        "trend_participation_delta5d_min": 5,
        "structural_leadership_delta5d_positive_required": True,
    },
    "active": {"structural_score_min": 65, "strength_1m_min": 0, "trend_participation_min": 55,
               "breadth_pct_positive_1m_min": 60},
    "fading": {"structural_score_max": 55, "trend_participation_delta10d_max": -10,
               "structural_leadership_delta10d_max": -10, "breadth_pct_positive_1m_max": 50, "confirm_days": 2},
    "dormant": {"structural_score_max": 35, "strength_1m_max": 0, "trend_participation_max": 40, "confirm_days": 5},
}


def test_lifecycle_v1_1_emerging_all_conditions():
    deltas = {"structural_score_delta5d": 6, "trend_participation_delta5d": 6, "structural_leadership_delta5d": 1,
              "trend_participation_delta10d": None, "structural_leadership_delta10d": None}
    result = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=50, thrust_percentile_1w=80, momentum_modifier=None,
        strength_1m=1.0, trend_participation=50, breadth_pct_positive_1m=55, deltas=deltas, cfg=LC_CFG)
    assert result["emerging"] is True


def test_lifecycle_v1_1_emerging_fails_without_leadership_delta_positive():
    deltas = {"structural_score_delta5d": 6, "trend_participation_delta5d": 6, "structural_leadership_delta5d": -1,
              "trend_participation_delta10d": None, "structural_leadership_delta10d": None}
    result = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=50, thrust_percentile_1w=80, momentum_modifier=None,
        strength_1m=1.0, trend_participation=50, breadth_pct_positive_1m=55, deltas=deltas, cfg=LC_CFG)
    assert result["emerging"] is False


def test_lifecycle_v1_1_active_conditions():
    deltas = {"structural_score_delta5d": None, "trend_participation_delta5d": None,
              "structural_leadership_delta5d": None, "trend_participation_delta10d": None,
              "structural_leadership_delta10d": None}
    result = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=70, thrust_percentile_1w=None, momentum_modifier=None,
        strength_1m=2.0, trend_participation=60, breadth_pct_positive_1m=65, deltas=deltas, cfg=LC_CFG)
    assert result["active"] is True


def test_lifecycle_v1_1_mature_requires_cooling_modifier_and_active_score():
    deltas = {"structural_score_delta5d": None, "trend_participation_delta5d": None,
              "structural_leadership_delta5d": None, "trend_participation_delta10d": None,
              "structural_leadership_delta10d": None}
    cooling = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=70, thrust_percentile_1w=None, momentum_modifier="COOLING",
        strength_1m=2.0, trend_participation=60, breadth_pct_positive_1m=65, deltas=deltas, cfg=LC_CFG)
    assert cooling["mature_raw"] is True

    not_cooling = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=70, thrust_percentile_1w=None, momentum_modifier="ACCELERATING",
        strength_1m=2.0, trend_participation=60, breadth_pct_positive_1m=65, deltas=deltas, cfg=LC_CFG)
    assert not_cooling["mature_raw"] is False

    weak_but_cooling = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=30, thrust_percentile_1w=None, momentum_modifier="COOLING",
        strength_1m=2.0, trend_participation=60, breadth_pct_positive_1m=65, deltas=deltas, cfg=LC_CFG)
    assert weak_but_cooling["mature_raw"] is False  # below the ACTIVE score bar -> not "mature", just weak


def test_lifecycle_v1_1_fading_any_weakening_signal_with_low_score():
    deltas_leadership_drop = {"structural_score_delta5d": None, "trend_participation_delta5d": None,
                               "structural_leadership_delta5d": None,
                               "trend_participation_delta10d": None, "structural_leadership_delta10d": -15}
    result = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=50, thrust_percentile_1w=None, momentum_modifier=None,
        strength_1m=0.0, trend_participation=50, breadth_pct_positive_1m=55,
        deltas=deltas_leadership_drop, cfg=LC_CFG)
    assert result["fading_raw"] is True


def test_lifecycle_v1_1_dormant_requires_all_three_low():
    deltas = {"structural_score_delta5d": None, "trend_participation_delta5d": None,
              "structural_leadership_delta5d": None, "trend_participation_delta10d": None,
              "structural_leadership_delta10d": None}
    dormant = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=20, thrust_percentile_1w=None, momentum_modifier=None,
        strength_1m=-2.0, trend_participation=10, breadth_pct_positive_1m=20, deltas=deltas, cfg=LC_CFG)
    assert dormant["dormant_raw"] is True

    not_dormant = calc_narrative_lifecycle_conditions_v1_1(
        narrative_structural_score=20, thrust_percentile_1w=None, momentum_modifier=None,
        strength_1m=5.0, trend_participation=10, breadth_pct_positive_1m=20, deltas=deltas, cfg=LC_CFG)  # strength positive
    assert not_dormant["dormant_raw"] is False


def test_lifecycle_v1_1_fading_beats_emerging_in_priority():
    # V1.1 point 24: priority order changed vs V1 -- FADING now outranks
    # EMERGING (was the reverse-adjacent case in V1's DORMANT>EMERGING>...).
    priority = ["DORMANT", "FADING", "EMERGING", "MATURE", "ACTIVE", "NEUTRAL"]
    conditions = {"dormant_confirmed": False, "emerging": True, "fading_confirmed": True,
                  "mature": False, "active": False}
    assert select_lifecycle_state(conditions, priority) == "FADING"


# ── Opportunity Engine V1.1: Leader entry/exit ───────────────────

LEADER_CFG = {
    "leader_entry": {"structural_rs_min": 85, "trend_strength_min": 70},
    "leader_exit": {"structural_rs_max": 80, "trend_strength_max": 60, "sma50_distance_max_pct": -2.0, "confirm_days": 2},
    "fresh_leader": {"entry_window_days": 3, "rs_1w_min": 85, "thrust_percentile_1d_min": 85, "thrust_percentile_1w_min": 80},
}


def test_leader_entry_condition_v1_1_requires_both_thresholds():
    assert calc_leader_entry_condition_v1_1(90, 75, LEADER_CFG) is True
    assert calc_leader_entry_condition_v1_1(90, 65, LEADER_CFG) is False  # trend_strength too low
    assert calc_leader_entry_condition_v1_1(80, 75, LEADER_CFG) is False  # structural_rs too low
    assert calc_leader_entry_condition_v1_1(None, 75, LEADER_CFG) is False


def test_leader_exit_condition_v1_1_any_one_signal_triggers():
    assert calc_leader_exit_condition_v1_1(structural_rs=75, trend_strength=75, sma50_distance_pct=5, cfg=LEADER_CFG) is True
    assert calc_leader_exit_condition_v1_1(structural_rs=90, trend_strength=50, sma50_distance_pct=5, cfg=LEADER_CFG) is True
    assert calc_leader_exit_condition_v1_1(structural_rs=90, trend_strength=75, sma50_distance_pct=-5, cfg=LEADER_CFG) is True
    assert calc_leader_exit_condition_v1_1(structural_rs=90, trend_strength=75, sma50_distance_pct=5, cfg=LEADER_CFG) is False
    # missing data never assumed to trigger exit
    assert calc_leader_exit_condition_v1_1(structural_rs=None, trend_strength=None, sma50_distance_pct=None, cfg=LEADER_CFG) is False


def test_stock_quality_base_state_v1_1_entry_from_neutral():
    state, age, streak = calc_stock_quality_base_state_v1_1(True, False, "neutral", 0, confirm_days=2)
    assert (state, age, streak) == ("leader", 1, 0)


def test_stock_quality_base_state_v1_1_exit_requires_confirm_days():
    # first exit-condition day: not yet confirmed, streak 1
    state, age, streak = calc_stock_quality_base_state_v1_1(False, True, "leader", 0, confirm_days=2)
    assert state == "leader" and streak == 1
    # second consecutive exit-condition day: confirmed -> neutral
    state2, age2, streak2 = calc_stock_quality_base_state_v1_1(False, True, "leader", 1, confirm_days=2)
    assert (state2, age2, streak2) == ("neutral", 0, 0)


def test_stock_quality_base_state_v1_1_recent_leader_does_not_count_as_was_leader():
    # Regression test for the bug caught during manual end-to-end smoke
    # testing: a 'recent_leader' (display overlay only) must NOT be treated
    # as "was previously a leader" -- otherwise a ticker whose entry
    # condition still isn't met would silently get promoted back to full
    # 'leader' status just because no exit condition re-triggered.
    state, age, streak = calc_stock_quality_base_state_v1_1(
        entry_condition=False, exit_condition=False, prev_quality_state="recent_leader",
        prev_exit_streak=0, confirm_days=2)
    assert state == "neutral"


def test_fresh_leader_label_v1_1_within_window_with_trigger():
    label = calc_fresh_leader_label_v1_1("leader", leader_age_days=2, rs_1w=90, thrust_pct_1d=50, thrust_pct_1w=50, cfg=LEADER_CFG)
    assert label == "fresh_leader"


def test_fresh_leader_label_v1_1_expires_after_window():
    label = calc_fresh_leader_label_v1_1("leader", leader_age_days=5, rs_1w=90, thrust_pct_1d=50, thrust_pct_1w=50, cfg=LEADER_CFG)
    assert label == "leader"


def test_fresh_leader_label_v1_1_no_trigger_stays_plain_leader():
    label = calc_fresh_leader_label_v1_1("leader", leader_age_days=1, rs_1w=50, thrust_pct_1d=50, thrust_pct_1w=50, cfg=LEADER_CFG)
    assert label == "leader"


# ── Recent Leader (real history + bootstrap) ─────────────────────

def test_recent_leader_state_passthrough_when_already_leader_like():
    assert calc_recent_leader_state("leader", "AAPL", [], 15, False) == "leader"
    assert calc_recent_leader_state("fresh_leader", "AAPL", [], 15, False) == "fresh_leader"


def test_recent_leader_state_uses_real_history_when_enough_snapshots():
    history = [{"stocks": {"AAPL": {"quality_state": "neutral"}}}] * 10 + \
              [{"stocks": {"AAPL": {"quality_state": "leader"}}}] * 5
    assert calc_recent_leader_state("neutral", "AAPL", history, 15, bootstrap_recent_leader=False) == "recent_leader"


def test_recent_leader_state_real_history_neutral_when_never_leader_in_window():
    history = [{"stocks": {"AAPL": {"quality_state": "neutral"}}}] * 15
    assert calc_recent_leader_state("neutral", "AAPL", history, 15, bootstrap_recent_leader=True) == "neutral"


def test_recent_leader_state_falls_back_to_bootstrap_when_history_too_short():
    history = [{"stocks": {"AAPL": {"quality_state": "neutral"}}}] * 3  # < memory_sessions
    assert calc_recent_leader_state("neutral", "AAPL", history, 15, bootstrap_recent_leader=True) == "recent_leader"
    assert calc_recent_leader_state("neutral", "AAPL", history, 15, bootstrap_recent_leader=False) == "neutral"


# ── Constructive Reset V1.1 (loosened) ───────────────────────────

CR_CFG = {"constructive_reset": {"max_below_ema20_pct": -2.0, "max_below_sma50_pct": -2.0}}


def test_constructive_reset_v1_1_accepts_recent_leader():
    result = calc_constructive_reset_v1_1("recent_leader", ema20_distance_pct=-1.0, sma50_distance_pct=-1.0,
                                           extended=False, narrative_memberships=["n1", "n2"], cfg=CR_CFG)
    assert result == ["n1", "n2"]  # narrative kept as unfiltered context, no lifecycle gate


def test_constructive_reset_v1_1_blocked_when_extended():
    result = calc_constructive_reset_v1_1("leader", ema20_distance_pct=-1.0, sma50_distance_pct=-1.0,
                                           extended=True, narrative_memberships=["n1"], cfg=CR_CFG)
    assert result == []


def test_constructive_reset_v1_1_blocked_when_too_far_below_anchors():
    result = calc_constructive_reset_v1_1("leader", ema20_distance_pct=-5.0, sma50_distance_pct=-1.0,
                                           extended=False, narrative_memberships=["n1"], cfg=CR_CFG)
    assert result == []


def test_constructive_reset_v1_1_blocked_when_not_leader_like():
    result = calc_constructive_reset_v1_1("neutral", ema20_distance_pct=-1.0, sma50_distance_pct=-1.0,
                                           extended=False, narrative_memberships=["n1"], cfg=CR_CFG)
    assert result == []


def test_constructive_reset_v1_1_no_narrative_membership_returns_empty_list_not_error():
    result = calc_constructive_reset_v1_1("leader", ema20_distance_pct=-1.0, sma50_distance_pct=-1.0,
                                           extended=False, narrative_memberships=[], cfg=CR_CFG)
    assert result == []


# ── Laggard V1.1 (structural_rs-gated) ────────────────────────────

LAGGARD_CFG = {"laggard": {"structural_rs_max": 60, "thrust_percentile_1d_max": 80,
                            "thrust_percentile_1w_max": 80, "exit_structural_rs_min": 65}}


def test_laggard_v1_1_entry_requires_all_conditions():
    assert calc_laggard_state_v1_1(False, structural_rs=50, thrust_pct_1d=70, thrust_pct_1w=70,
                                    in_bottom_pct=True, cfg=LAGGARD_CFG) is True
    assert calc_laggard_state_v1_1(False, structural_rs=70, thrust_pct_1d=70, thrust_pct_1w=70,
                                    in_bottom_pct=True, cfg=LAGGARD_CFG) is False  # structurally too strong


def test_laggard_v1_1_exit_on_structural_rs_recovery():
    assert calc_laggard_state_v1_1(True, structural_rs=70, thrust_pct_1d=70, thrust_pct_1w=70,
                                    in_bottom_pct=True, cfg=LAGGARD_CFG) is False


def test_laggard_v1_1_exit_on_leaving_bottom_pct():
    assert calc_laggard_state_v1_1(True, structural_rs=40, thrust_pct_1d=70, thrust_pct_1w=70,
                                    in_bottom_pct=False, cfg=LAGGARD_CFG) is False


def test_laggard_v1_1_stays_laggard_while_conditions_hold():
    assert calc_laggard_state_v1_1(True, structural_rs=40, thrust_pct_1d=70, thrust_pct_1w=70,
                                    in_bottom_pct=True, cfg=LAGGARD_CFG) is True


# ── Change Detection V1.1 ─────────────────────────────────────────

CD_CFG = {"narrative_rank_improve_min": 3, "narrative_score_delta_min": 10}


def test_detect_narrative_changes_v1_1_structural_score_surge_and_modifier_events():
    today = {"lifecycle_state": "ACTIVE", "rank": 2, "structural_score": 80.0, "momentum_modifier": "ACCELERATING"}
    prev = {"lifecycle_state": "EMERGING", "rank": 6, "structural_score": 60.0, "momentum_modifier": None}
    events = detect_narrative_changes_v1_1("n1", "Software", today, prev, CD_CFG)
    types = {e["type"] for e in events}
    assert "new_active" in types
    assert "rank_improved" in types
    assert "structural_score_surge" in types
    assert "modifier_gained" in types


def test_detect_narrative_changes_v1_1_modifier_lost():
    today = {"lifecycle_state": "ACTIVE", "rank": 2, "structural_score": 80.0, "momentum_modifier": None}
    prev = {"lifecycle_state": "ACTIVE", "rank": 2, "structural_score": 80.0, "momentum_modifier": "COOLING"}
    events = detect_narrative_changes_v1_1("n1", "Software", today, prev, CD_CFG)
    assert any(e["type"] == "modifier_lost" for e in events)


def test_detect_stock_changes_v1_1_new_recent_leader_and_expiry():
    events = detect_stock_changes_v1_1("AAPL", {"quality_state": "recent_leader"}, {"quality_state": "neutral"})
    assert any(e["type"] == "new_recent_leader" for e in events)

    events2 = detect_stock_changes_v1_1("AAPL", {"quality_state": "neutral"}, {"quality_state": "recent_leader"})
    assert any(e["type"] == "recent_leader_expired" for e in events2)


def test_detect_stock_changes_v1_1_leader_to_recent_leader_is_not_new_recent_leader():
    # Going from full Leader straight to Recent Leader is a leadership-lost
    # event, not a "new" Recent Leader (it was already leader-like).
    events = detect_stock_changes_v1_1("AAPL", {"quality_state": "recent_leader"}, {"quality_state": "leader"})
    types = {e["type"] for e in events}
    assert "leadership_lost" in types
    assert "new_recent_leader" not in types


# ── Full-Universe: Primary/Secondary Narrative context (point 11/N) ──

def test_build_narrative_membership_index_resolves_primary_and_secondary():
    narratives = [
        {"id": "n1", "name": "Narrative One",
         "members": [{"symbol": "AAA", "assignment_priority": "primary"},
                     {"symbol": "BBB", "assignment_priority": "secondary"}]},
        {"id": "n2", "name": "Narrative Two",
         "members": [{"symbol": "AAA", "assignment_priority": "secondary"}]},
    ]
    idx = build_narrative_membership_index(narratives)
    assert idx["AAA"]["primary_id"] == "n1"
    assert idx["AAA"]["primary_name"] == "Narrative One"
    assert idx["AAA"]["secondary_ids"] == ["n2"]
    assert idx["AAA"]["secondary_names"] == ["Narrative Two"]
    assert idx["AAA"]["all_ids"] == ["n1", "n2"]

    assert idx["BBB"]["primary_id"] is None  # BBB is only ever secondary
    assert idx["BBB"]["secondary_ids"] == ["n1"]


def test_build_narrative_membership_index_degrades_gracefully_without_assignment_priority():
    narratives = [{"id": "n1", "name": "Narrative One", "members": [{"symbol": "AAA"}]}]
    idx = build_narrative_membership_index(narratives)
    assert idx["AAA"]["primary_id"] is None
    assert idx["AAA"]["secondary_ids"] == []
    assert idx["AAA"]["all_ids"] == ["n1"]  # still tracked as a membership, just unresolved priority


def test_build_narrative_membership_index_unknown_symbol_returns_default_shape():
    idx = build_narrative_membership_index([])
    entry = idx["NOT_PRESENT"]
    assert entry == {"primary_id": None, "primary_name": None, "secondary_ids": [], "secondary_names": [], "all_ids": []}


# ── V6 point 16-18: Opportunity V2 architecture preparation is now
# populated for real by the Calibration-aware Opportunities UI v1 (see
# tests/test_opportunity_calibration_v2.py) — the "stays null" placeholder
# tests that used to live here are superseded, not just removed.
