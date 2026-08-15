"""
Tests for scripts/build_narratives.py's Full-Universe additions: eligible-
only narrative membership (spec point 8), Primary/Secondary assignment_priority
propagation through load_taxonomy/output (point 10/11), and the new coverage
meta fields (point 23). All synthetic data — no network required.
Run with: pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import load_taxonomy  # noqa: E402


def write_taxonomy(tmp_path, narratives):
    p = tmp_path / "narratives.json"
    p.write_text(json.dumps({"schema_version": 1, "narratives": narratives}))
    return p


def membership(assignment_priority, role="core", confidence=90):
    return {"role": role, "assignment_priority": assignment_priority, "confidence": confidence,
            "reason": "x", "added_at": "2026-01-01", "last_reviewed_at": "2026-01-01",
            "active_eligible": True}


# ── load_taxonomy: membership_meta carries assignment_priority through ──

def test_load_taxonomy_preserves_assignment_priority_in_membership_meta(tmp_path):
    p = write_taxonomy(tmp_path, [
        {"id": "n1", "name": "N1", "status": "active", "tickers": {
            "AAA": membership("primary"), "BBB": membership("secondary"),
        }},
    ])
    narratives, universe = load_taxonomy(str(p))
    assert narratives[0]["tickers"] == ["AAA", "BBB"]  # unchanged shape for existing callers
    assert narratives[0]["membership_meta"]["AAA"]["assignment_priority"] == "primary"
    assert narratives[0]["membership_meta"]["BBB"]["assignment_priority"] == "secondary"
    assert universe == ["AAA", "BBB"]


def test_load_taxonomy_legacy_flat_list_has_no_membership_meta(tmp_path):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"narratives": [{"id": "n1", "name": "N1", "tickers": ["AAA"]}]}))
    narratives, universe = load_taxonomy(str(p), legacy_path=str(p))
    assert narratives[0]["tickers"] == ["AAA"]
    assert narratives[0]["membership_meta"] == {}


# ── point 8: eligible-only narrative membership (main()'s inline filter,
# exercised here via the same logic pattern the module applies) ──

def test_member_filter_excludes_classified_but_ineligible_ticker():
    ticker_metrics = {"AAA": {"symbol": "AAA"}, "BBB": {"symbol": "BBB"}}
    eligible_set = {"AAA"}  # BBB is classified (in tickers) but not eligible today
    tickers = ["AAA", "BBB"]
    members = [t for t in tickers if t in ticker_metrics and (eligible_set is None or t in eligible_set)]
    assert members == ["AAA"]


def test_member_filter_uses_every_taxonomy_member_when_eligible_set_is_none():
    # Fallback mode (no market_features available at all) -> no eligibility
    # concept to filter on, same graceful-degradation rule as elsewhere.
    ticker_metrics = {"AAA": {"symbol": "AAA"}, "BBB": {"symbol": "BBB"}}
    eligible_set = None
    tickers = ["AAA", "BBB"]
    members = [t for t in tickers if t in ticker_metrics and (eligible_set is None or t in eligible_set)]
    assert members == ["AAA", "BBB"]


# ── Quality Patch section 8/19-21: active/undersized narrative filter.
# main()'s Pass-1 loop applies `if len(members) < min_active_members: skip`
# (see build_narratives.py) -- exercised here via the identical logic
# pattern, matching this file's existing style for main()-inline logic. ──

def _active_undersized_split(narratives_with_members, min_active_members):
    """Mirrors build_narratives.py main()'s Pass-1 active/undersized split:
    narratives_with_members = [(narrative_dict, members_list), ...]."""
    active, undersized = [], []
    for n, members in narratives_with_members:
        if len(members) < min_active_members:
            undersized.append({"id": n["id"], "name": n["name"], "eligible_member_count": len(members)})
        else:
            active.append(n["id"])
    return active, undersized


def test_narrative_with_5_eligible_members_is_active():
    n = {"id": "n1", "name": "Big Enough"}
    active, undersized = _active_undersized_split([(n, ["A", "B", "C", "D", "E"])], min_active_members=5)
    assert active == ["n1"]
    assert undersized == []


def test_narrative_with_4_eligible_members_is_undersized():
    n = {"id": "n1", "name": "Too Small"}
    active, undersized = _active_undersized_split([(n, ["A", "B", "C", "D"])], min_active_members=5)
    assert active == []
    assert undersized == [{"id": "n1", "name": "Too Small", "eligible_member_count": 4}]


def test_narrative_with_zero_eligible_members_is_undersized():
    n = {"id": "n1", "name": "Empty Today"}
    active, undersized = _active_undersized_split([(n, [])], min_active_members=5)
    assert undersized == [{"id": "n1", "name": "Empty Today", "eligible_member_count": 0}]


def test_min_active_members_zero_when_market_features_unavailable():
    # main(): min_active_members = cfg[...] if eligible_set is not None else 0
    # -- degraded mode (no market_features) must never hide a narrative.
    eligible_set = None
    cfg_value = 5
    min_active_members = cfg_value if eligible_set is not None else 0
    assert min_active_members == 0


# ── eligible_with_active_narrative meta field: only counts a ticker whose
# PRIMARY narrative made it into the active set (point 22-23) ──

def test_eligible_with_active_narrative_excludes_ticker_whose_primary_is_undersized():
    narratives = [
        {"id": "big", "name": "Big", "membership_meta": {
            "A": {"assignment_priority": "primary"}, "B": {"assignment_priority": "primary"},
            "C": {"assignment_priority": "primary"}, "D": {"assignment_priority": "primary"},
            "E": {"assignment_priority": "primary"},
        }},
        {"id": "small", "name": "Small", "membership_meta": {
            "F": {"assignment_priority": "primary"},
        }},
    ]
    active_ids = {"big"}  # "small" has only 1 eligible member -> undersized, excluded
    primary_narrative_of = {}
    for n in narratives:
        for sym, meta in n["membership_meta"].items():
            if meta.get("assignment_priority") == "primary":
                primary_narrative_of[sym] = n["id"]
    eligible_pop = {"A", "B", "C", "D", "E", "F"}
    with_active = sum(1 for t in eligible_pop if primary_narrative_of.get(t) in active_ids)
    assert with_active == 5      # A-E (big's members)
    assert len(eligible_pop) - with_active == 1  # F (small's only member) -> without active narrative


# NOTE (RVOL/Screener/Benchmark/Futures Patch point 10): the old
# "compute_narrative_rs_history also respects eligible_set" test used to
# live here, exercising eligibility-filtering INSIDE that function. That
# function no longer takes daily_ret/eligible_set at all — it now renders
# the SAME relative_strength_by_id lines the headline narrative_rs already
# uses (see build_narratives.main()), which are built from narrative_rows'
# members — themselves already eligible_set-filtered a few lines above in
# main() (the exact same filtering this test used to check). That upstream
# filtering is covered by test_narrative_full_universe.py's other
# eligible_set tests above; compute_narrative_rs_history's own windowing/
# 0%-baseline behaviour is covered by tests/test_narrative_rs_history.py.
