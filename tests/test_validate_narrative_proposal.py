"""
Tests for scripts/validate_narrative_proposal.py: Proposal-Schema,
CREATE/ADD/REMOVE-Validation, Overlap-Warning, Multi-Membership.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from validate_narrative_proposal import validate_proposal  # noqa: E402


@pytest.fixture
def config():
    return {
        "universe": {
            "stocks_only": True,
            "excluded_types": ["ETF", "ETN", "FUND", "PFD", "WARRANT", "RIGHT", "UNIT", "INDEX"],
        },
        "create": {
            "minimum_members": 4,
            "minimum_core_members": 3,
            "minimum_share_rs80_pct": 60,
            "minimum_count_rs90": 2,
            "minimum_breadth_pct": 60,
            "positive_thrust_required": True,
            "overlap_warning_threshold_pct": 70,
        },
        "membership": {
            "core_confidence_minimum": 85,
            "secondary_confidence_minimum": 70,
            "remove_confirmation_reviews": 2,
        },
    }


@pytest.fixture
def taxonomy():
    return {
        "narratives": [
            {
                "id": "semiconductors",
                "name": "Semiconductors",
                "status": "active",
                "tickers": {
                    "NVDA": {"role": "core"}, "AMD": {"role": "core"},
                    "MU": {"role": "secondary"}, "INTC": {"role": "secondary"},
                },
            },
            {
                "id": "ai_infrastructure",
                "name": "AI Infrastructure",
                "status": "active",
                "tickers": {"NVDA": {"role": "core"}, "VRT": {"role": "core"}, "SMCI": {"role": "secondary"}},
            },
        ]
    }


def base_proposal(changes):
    return {"schema_version": 1, "created_at": "2026-08-17", "review_type": "weekly", "changes": changes}


# ── Proposal schema ─────────────────────────────────────────────

def test_schema_version_required(config, taxonomy):
    result = validate_proposal({"schema_version": 2, "changes": []}, taxonomy, config)
    assert not result.ok
    assert any("schema_version" in e for e in result.errors)


def test_empty_change_set_is_valid(config, taxonomy):
    result = validate_proposal(base_proposal([]), taxonomy, config)
    assert result.ok


def test_unstructured_entry_rejected(config, taxonomy):
    result = validate_proposal(base_proposal(["just a string, not an object"]), taxonomy, config)
    assert not result.ok


def test_invalid_action_rejected(config, taxonomy):
    change = {"action": "DELETE_EVERYTHING", "semantic_evidence": "x", "reason": "x"}
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok


def test_missing_semantic_evidence_rejected(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "secondary", "confidence": 75, "reason": "fits",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok
    assert any("semantic_evidence" in e for e in result.errors)


# ── Ticker existence / asset type (Universe / ETF exclusion) ───────

def test_invented_ticker_rejected_when_known_tickers_supplied(config, taxonomy):
    known = {"NVDA": {"type": "CS"}}
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "FAKE123",
        "role": "secondary", "confidence": 75, "semantic_evidence": "x", "reason": "x",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config, known_tickers=known)
    assert not result.ok
    assert any("existiert nicht" in e for e in result.errors)


def test_etf_type_rejected_stocks_only(config, taxonomy):
    known = {"SMH": {"type": "ETF"}}
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "SMH",
        "role": "secondary", "confidence": 75, "semantic_evidence": "x", "reason": "x",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config, known_tickers=known)
    assert not result.ok
    assert any("Asset-Type" in e for e in result.errors)


def test_common_stock_type_accepted(config, taxonomy):
    known = {"TXN": {"type": "CS"}}
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "secondary", "confidence": 75, "semantic_evidence": "x", "reason": "x",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config, known_tickers=known)
    assert result.ok


# ── ADD validation ──────────────────────────────────────────────

def test_add_core_below_confidence_minimum_rejected(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "core", "confidence": 80, "semantic_evidence": "core driver", "reason": "core fit",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok
    assert any("confidence 80" in e for e in result.errors)


def test_add_core_at_confidence_minimum_accepted(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "core", "confidence": 85, "semantic_evidence": "core driver", "reason": "core fit",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert result.ok


def test_add_secondary_below_confidence_minimum_rejected(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "secondary", "confidence": 60, "semantic_evidence": "some relevance", "reason": "secondary fit",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok


def test_add_invalid_role_speculative_rejected(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "semiconductors", "ticker": "TXN",
        "role": "speculative", "confidence": 90, "semantic_evidence": "x", "reason": "x",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok


# ── CREATE validation ────────────────────────────────────────────

def _create_change(members, breadth=65, thrust=0.5, narrative_name="Nuclear Fuel Cycle"):
    return {
        "action": "CREATE", "narrative_name": narrative_name,
        "semantic_evidence": "shared driver: nuclear fuel cycle capacity buildout",
        "reason": "new momentum theme",
        "quantitative_evidence": {"members": members, "breadth_pct": breadth, "thrust": thrust},
    }


def test_create_below_minimum_members_rejected(config, taxonomy):
    members = [{"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
               {"ticker": "B", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92}]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert not result.ok
    assert any("Mitglieder" in e for e in result.errors)


def test_create_below_core_minimum_rejected(config, taxonomy):
    members = [
        {"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "B", "role": "secondary", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "C", "role": "secondary", "rs_percentile_1w": 91, "rs_percentile_1m": 91},
        {"ticker": "D", "role": "secondary", "rs_percentile_1w": 88, "rs_percentile_1m": 88},
    ]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert not result.ok
    assert any("Core-Mitglieder" in e for e in result.errors)


def test_create_meeting_all_thresholds_accepted(config, taxonomy):
    members = [
        {"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "B", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "C", "role": "core", "rs_percentile_1w": 85, "rs_percentile_1m": 85},
        {"ticker": "D", "role": "secondary", "rs_percentile_1w": 60, "rs_percentile_1m": 60},
    ]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert result.ok, result.errors


def test_create_insufficient_rs90_count_rejected(config, taxonomy):
    members = [
        {"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "B", "role": "core", "rs_percentile_1w": 82, "rs_percentile_1m": 82},
        {"ticker": "C", "role": "core", "rs_percentile_1w": 81, "rs_percentile_1m": 81},
        {"ticker": "D", "role": "secondary", "rs_percentile_1w": 80, "rs_percentile_1m": 80},
    ]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert not result.ok
    assert any("RS>=90" in e for e in result.errors)


def test_create_negative_thrust_rejected(config, taxonomy):
    members = [
        {"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "B", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "C", "role": "core", "rs_percentile_1w": 85, "rs_percentile_1m": 85},
        {"ticker": "D", "role": "secondary", "rs_percentile_1w": 60, "rs_percentile_1m": 60},
    ]
    result = validate_proposal(base_proposal([_create_change(members, thrust=-0.1)]), taxonomy, config)
    assert not result.ok
    assert any("Thrust" in e for e in result.errors)


def test_create_low_breadth_rejected(config, taxonomy):
    members = [
        {"ticker": "A", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "B", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "C", "role": "core", "rs_percentile_1w": 85, "rs_percentile_1m": 85},
        {"ticker": "D", "role": "secondary", "rs_percentile_1w": 60, "rs_percentile_1m": 60},
    ]
    result = validate_proposal(base_proposal([_create_change(members, breadth=40)]), taxonomy, config)
    assert not result.ok
    assert any("Breadth" in e for e in result.errors)


# ── Overlap warning (warning, not a blocker) ────────────────────

def test_create_high_overlap_is_warning_not_error(config, taxonomy):
    # 3/4 proposed members already in "semiconductors" -> 75% overlap >= 70% threshold
    members = [
        {"ticker": "NVDA", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "AMD", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "MU", "role": "core", "rs_percentile_1w": 85, "rs_percentile_1m": 85},
        {"ticker": "NEWTICK", "role": "secondary", "rs_percentile_1w": 60, "rs_percentile_1m": 60},
    ]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert result.ok  # high overlap alone must not block CREATE
    assert any("Overlap" in w for w in result.warnings)
    assert result.annotated[0]["overlap_warning"] is True
    assert result.annotated[0]["overlap_pct"] >= 70


def test_create_low_overlap_no_warning(config, taxonomy):
    members = [
        {"ticker": "X1", "role": "core", "rs_percentile_1w": 95, "rs_percentile_1m": 95},
        {"ticker": "X2", "role": "core", "rs_percentile_1w": 92, "rs_percentile_1m": 92},
        {"ticker": "X3", "role": "core", "rs_percentile_1w": 85, "rs_percentile_1m": 85},
        {"ticker": "X4", "role": "secondary", "rs_percentile_1w": 60, "rs_percentile_1m": 60},
    ]
    result = validate_proposal(base_proposal([_create_change(members)]), taxonomy, config)
    assert result.ok
    assert result.annotated[0]["overlap_warning"] is False


# ── REMOVE validation + two-review rule ─────────────────────────

def test_remove_weak_performance_reason_rejected(config, taxonomy):
    change = {
        "action": "REMOVE", "narrative": "semiconductors", "ticker": "MU",
        "remove_reason_code": "weak_performance",  # not in the allowed enum
        "semantic_evidence": "underperforming", "reason": "weak price action",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok
    assert any("remove_reason_code" in e for e in result.errors)


def test_remove_valid_reason_but_first_time_not_actionable(config, taxonomy):
    change = {
        "action": "REMOVE", "narrative": "semiconductors", "ticker": "MU",
        "remove_reason_code": "business_model_change",
        "semantic_evidence": "MU pivoted away from memory into unrelated services",
        "reason": "no longer fits Semiconductors",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config, proposal_history=[])
    assert result.ok  # valid, but...
    assert result.annotated[0]["actionable"] is False  # ...not actionable without a 2nd review
    assert any("zweite" in w for w in result.warnings)


def test_remove_confirmed_by_prior_consecutive_review_is_actionable(config, taxonomy):
    prior_proposal = {
        "created_at": "2026-08-10", "changes": [
            {"action": "REMOVE", "narrative": "semiconductors", "ticker": "MU",
             "remove_reason_code": "business_model_change"},
        ],
    }
    change = {
        "action": "REMOVE", "narrative": "semiconductors", "ticker": "MU",
        "remove_reason_code": "business_model_change",
        "semantic_evidence": "confirmed again this week", "reason": "second consecutive review",
    }
    result = validate_proposal(
        base_proposal([change]), taxonomy, config,
        proposal_history=[("2026-08-10", prior_proposal)],
    )
    assert result.ok
    assert result.annotated[0]["actionable"] is True


def test_remove_emergency_delisting_bypasses_two_review_rule(config, taxonomy):
    change = {
        "action": "REMOVE", "narrative": "semiconductors", "ticker": "MU",
        "remove_reason_code": "delisting", "emergency": True,
        "semantic_evidence": "delisted from exchange", "reason": "delisting",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config, proposal_history=[])
    assert result.ok
    assert result.annotated[0]["actionable"] is True


def test_remove_nonmember_ticker_rejected(config, taxonomy):
    change = {
        "action": "REMOVE", "narrative": "semiconductors", "ticker": "AAPL",
        "remove_reason_code": "misclassification",
        "semantic_evidence": "never should have been added", "reason": "misclassified",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert not result.ok


# ── Multi-Membership ─────────────────────────────────────────────

def test_multi_membership_supported_across_narratives(taxonomy):
    # NVDA is a core member of BOTH semiconductors and ai_infrastructure in the fixture.
    memberships = [n["id"] for n in taxonomy["narratives"] if "NVDA" in n["tickers"]]
    assert set(memberships) == {"semiconductors", "ai_infrastructure"}


def test_add_second_membership_for_existing_ticker_is_valid(config, taxonomy):
    change = {
        "action": "ADD", "narrative": "ai_infrastructure", "ticker": "AMD",
        "role": "secondary", "confidence": 72,
        "semantic_evidence": "AMD MI-series accelerators feed AI infra buildout",
        "reason": "secondary AI infra exposure alongside existing Semiconductors core membership",
    }
    result = validate_proposal(base_proposal([change]), taxonomy, config)
    assert result.ok  # AMD already core in semiconductors; adding to a second narrative must not conflict
