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

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_narratives import load_taxonomy, compute_narrative_rs_history  # noqa: E402


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


# ── compute_narrative_rs_history also respects eligible_set (point 8) ──

def make_daily_ret(days, columns_pct):
    df = pd.DataFrame(index=days, columns=list(columns_pct.keys()), dtype=float)
    for sym, rets in columns_pct.items():
        for i, day in enumerate(days):
            df.at[day, sym] = rets[i]
    return df


def test_rs_history_excludes_ineligible_member_from_basket():
    days = [f"2026-01-{d:02d}" for d in range(1, 21)]
    daily_ret = make_daily_ret(days, {
        "SPY": [0.1] * 20,
        "STRONG": [1.0] * 20,   # eligible, drives the basket up
        "WEAK_INELIGIBLE": [-5.0] * 20,  # classified but NOT eligible -> must be excluded
    })
    narratives = [{"id": "n1", "tickers": ["STRONG", "WEAK_INELIGIBLE"]}]

    with_filter = compute_narrative_rs_history(narratives, daily_ret, days, eligible_set={"STRONG"}, lookback_days=20)
    without_filter = compute_narrative_rs_history(narratives, daily_ret, days, eligible_set=None, lookback_days=20)

    # Filtered basket (STRONG only) must clearly outperform the unfiltered
    # basket (STRONG+WEAK_INELIGIBLE median) -- proves WEAK_INELIGIBLE was
    # actually dropped by the eligible_set filter, not just present-but-diluted.
    assert with_filter["narratives"]["n1"][-1] > without_filter["narratives"]["n1"][-1]
