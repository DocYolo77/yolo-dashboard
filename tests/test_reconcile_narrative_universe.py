"""
Tests for scripts/reconcile_narrative_universe.py — the Full-Universe
semantic classification/reconciliation engine (points A-K of the
Full-Universe spec's test list). All synthetic data — no network / no
ANTHROPIC_API_KEY required (LLM calls are mocked).
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import reconcile_narrative_universe as r  # noqa: E402
import llm_provider  # noqa: E402


MEMBERSHIP_CFG = {"core_confidence_minimum": 85, "secondary_confidence_minimum": 70}


def make_taxonomy(narratives=None):
    return {"_comment": "x", "schema_version": 1, "narratives": narratives or []}


def make_membership(assignment_priority="primary", role="core", confidence=90, active_eligible=True):
    return {
        "role": role, "assignment_priority": assignment_priority, "confidence": confidence,
        "reason": "x", "added_at": "2026-01-01", "last_reviewed_at": "2026-01-01",
        "active_eligible": active_eligible, "classification_source": "manual_migration",
        "classification_version": 1,
    }


# ── A/B/C: coverage + primary/secondary invariants ───────────────

def test_validate_full_coverage_passes_when_every_eligible_ticker_has_one_primary():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {"AAA": make_membership("primary")}},
        {"id": "n2", "name": "N2", "tickers": {"BBB": make_membership("primary")}},
    ])
    errors = r.validate_full_coverage(taxonomy, {"AAA", "BBB"}, max_secondary=2)
    assert errors == []


def test_validate_full_coverage_fails_when_eligible_ticker_has_no_primary():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {}}])
    errors = r.validate_full_coverage(taxonomy, {"AAA"}, max_secondary=2)
    assert any("kein Primary" in e for e in errors)


def test_validate_full_coverage_fails_when_two_primaries_for_same_ticker():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {"AAA": make_membership("primary")}},
        {"id": "n2", "name": "N2", "tickers": {"AAA": make_membership("primary")}},
    ])
    errors = r.validate_full_coverage(taxonomy, {"AAA"}, max_secondary=2)
    assert any("2 Primary" in e for e in errors)


def test_validate_full_coverage_fails_with_more_than_max_secondary():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {"AAA": make_membership("primary")}},
        {"id": "n2", "name": "N2", "tickers": {"AAA": make_membership("secondary")}},
        {"id": "n3", "name": "N3", "tickers": {"AAA": make_membership("secondary")}},
        {"id": "n4", "name": "N4", "tickers": {"AAA": make_membership("secondary")}},
    ])
    errors = r.validate_full_coverage(taxonomy, {"AAA"}, max_secondary=2)
    assert any("Secondary Narratives" in e for e in errors)


# ── D: non-eligible ticker stays classified, excluded from active set ──

def test_mark_active_eligible_flips_false_without_removing_membership():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {"AAA": make_membership(active_eligible=True)}}])
    r.mark_active_eligible(taxonomy, eligible_now=set())  # AAA no longer eligible today
    assert taxonomy["narratives"][0]["tickers"]["AAA"]["active_eligible"] is False
    assert "AAA" in taxonomy["narratives"][0]["tickers"]  # membership itself untouched (point 7)


# ── E/F: universe changes — reentry reuses, first-time entry needs classification ──

def test_compute_universe_changes_categorizes_correctly():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {
            "STAYED_TICKER": make_membership(active_eligible=True),
            "EXITED_TICKER": make_membership(active_eligible=True),
            "REENTERED_TICKER": make_membership(active_eligible=False),
        }},
    ])
    eligible_now = {"STAYED_TICKER", "REENTERED_TICKER", "NEW_TICKER"}
    changes = r.compute_universe_changes(taxonomy, eligible_now)
    assert changes["stayed"] == ["STAYED_TICKER"]
    assert changes["reentered"] == ["REENTERED_TICKER"]
    assert changes["entered"] == ["NEW_TICKER"]
    assert changes["exited"] == ["EXITED_TICKER"]


def test_reentered_ticker_excluded_from_incremental_classification_target():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {"AAA": make_membership(active_eligible=False)}}])
    classified = r.compute_classified_tickers(taxonomy)
    changes = r.compute_universe_changes(taxonomy, eligible_now={"AAA"})
    # Daily-incremental target = entered tickers not already classified.
    # AAA is a REENTRY (already classified) -> must NOT need reclassification.
    to_classify = [t for t in changes["entered"] if t not in classified]
    assert to_classify == []
    assert changes["reentered"] == ["AAA"]


def test_true_new_entry_is_targeted_for_classification():
    taxonomy = make_taxonomy([])
    changes = r.compute_universe_changes(taxonomy, eligible_now={"BRAND_NEW"})
    classified = r.compute_classified_tickers(taxonomy)
    to_classify = [t for t in changes["entered"] if t not in classified]
    assert to_classify == ["BRAND_NEW"]


# ── G: classification context excludes ALL momentum/RS/thrust/price fields ──

def test_classification_context_excludes_momentum_fields():
    types_ref = {"tickers": {"AAA": {"name": "AAA Corp"}}}
    reference_cache = {"tickers": {"AAA": {"sic_code": "1234", "sic_description": "WIDGETS",
                                            "description": "Makes widgets."}}}
    ctx = r.build_classification_context(["AAA"], types_ref, reference_cache)
    assert len(ctx) == 1
    row = ctx[0]
    forbidden_substrings = ["rs_", "thrust", "structural_rs", "trend_strength", "momentum", "price", "return_"]
    keys_joined = " ".join(row.keys()).lower()
    for bad in forbidden_substrings:
        assert bad not in keys_joined, f"momentum-related field leaked into classification context: {bad!r}"
    assert set(row.keys()) == {"ticker", "name", "sic_code", "sic_description", "description"}


def test_system_prompt_forbids_momentum_reasoning():
    forbidden = ["momentum", "RS", "Thrust", "Performance"]
    # Every forbidden concept must be explicitly named as OFF-LIMITS in the
    # system prompt text (not merely absent) -- weak assertion on absence
    # alone wouldn't catch a prompt that silently invites the model to infer
    # momentum from company descriptions.
    assert "KEINE Kurs-, RS-, Thrust- oder Performance-Daten" in r.CLASSIFICATION_SYSTEM_PROMPT


# ── H/I: new narrative creation + dedup ───────────────────────────

def test_resolve_narrative_reference_creates_new_when_nothing_fits():
    catalog_by_id, catalog_by_norm = {}, {}
    resolved_id, created, err = r.resolve_narrative_reference(
        "regional_banks", True, "Regional Banks", "Community/regional banks", catalog_by_id, catalog_by_norm)
    assert err is None
    assert created is True
    assert resolved_id in catalog_by_id
    assert catalog_by_id[resolved_id]["name"] == "Regional Banks"


def test_resolve_narrative_reference_dedups_near_identical_names():
    catalog_by_id = {"regional_banks": {"name": "Regional Banks", "classification_hint": "x"}}
    catalog_by_norm = {"regionalbanks": "regional_banks"}
    # LLM proposes a "new" narrative with different casing/spacing -- must
    # resolve to the EXISTING id, not create a duplicate.
    resolved_id, created, err = r.resolve_narrative_reference(
        "regional_bankshares", True, "regional  banks", "dup", catalog_by_id, catalog_by_norm)
    assert err is None
    assert created is False
    assert resolved_id == "regional_banks"
    assert len(catalog_by_id) == 1  # no duplicate entry created


def test_resolve_narrative_reference_existing_id_reused_without_new_flag():
    catalog_by_id = {"semiconductors": {"name": "Semiconductors", "classification_hint": "x"}}
    catalog_by_norm = {"semiconductors": "semiconductors"}
    resolved_id, created, err = r.resolve_narrative_reference(
        "semiconductors", False, None, None, catalog_by_id, catalog_by_norm)
    assert (resolved_id, created, err) == ("semiconductors", False, None)


def test_resolve_narrative_reference_errors_on_unknown_id_without_new_flag():
    resolved_id, created, err = r.resolve_narrative_reference("ghost_narrative", False, None, None, {}, {})
    assert resolved_id is None
    assert err is not None


def test_find_duplicate_narrative_names_detects_near_identical_clusters():
    taxonomy = make_taxonomy([
        {"id": "a", "name": "Regional Banks", "tickers": {}},
        {"id": "b", "name": "regional-banks", "tickers": {}},
        {"id": "c", "name": "Semiconductors", "tickers": {}},
    ])
    dupes = r.find_duplicate_narrative_names(taxonomy)
    assert len(dupes) == 1
    (cluster,) = dupes.values()
    assert set(cluster) == {"a", "b"}


# ── J: malformed LLM output doesn't crash / doesn't silently misclassify ──

def test_parse_classification_result_rejects_unknown_ticker():
    result, err = r.parse_classification_result({"ticker": "GHOST", "primary_narrative_id": "n1",
                                                   "primary_confidence": 90}, valid_tickers={"AAA"})
    assert result is None
    assert err is not None


def test_parse_classification_result_rejects_non_numeric_confidence():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": "very high"},
        valid_tickers={"AAA"})
    assert result is None
    assert "confidence" in err


def test_parse_classification_result_tolerates_stringified_number():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": "90", "reasoning": "x"},
        valid_tickers={"AAA"})
    assert err is None
    assert result["primary_confidence"] == 90.0


def test_parse_classification_result_rejects_new_without_name():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "x", "primary_is_new": True, "primary_confidence": 90},
        valid_tickers={"AAA"})
    assert result is None
    assert "primary_new_name" in err


def test_parse_classification_result_skips_malformed_secondary_without_failing_whole_row():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": 90, "reasoning": "x",
         "secondary_narratives": [{"narrative_id": "n2", "confidence": "not a number"},
                                   {"narrative_id": "n3", "confidence": 75}]},
        valid_tickers={"AAA"})
    assert err is None
    assert len(result["secondary_narratives"]) == 1
    assert result["secondary_narratives"][0]["narrative_id"] == "n3"


def test_apply_classification_result_reports_error_without_partial_write_on_bad_primary():
    taxonomy_by_id = {}
    catalog_by_id, catalog_by_norm = {}, {}
    result = {"ticker": "AAA", "primary_narrative_id": "ghost", "primary_is_new": False,
              "primary_new_name": None, "primary_new_definition": None, "primary_confidence": 90,
              "secondary_narratives": [], "reasoning": "x"}
    errors = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm,
                                            result, MEMBERSHIP_CFG, max_secondary=2,
                                            today="2026-08-12", source="daily_reconciliation")
    assert errors
    assert taxonomy_by_id == {}  # nothing written for this ticker


# ── K: a failed classification run must not produce a passing coverage check ──

def test_llm_failure_leaves_ticker_unclassified_and_fails_coverage(tmp_path):
    def failing_generate(system_prompt, user_prompt, model=None):
        raise llm_provider.LLMError("simulated failure")

    taxonomy_by_id = {}
    catalog_by_id, catalog_by_norm = {}, {}
    cfg = {"membership": MEMBERSHIP_CFG, "classification": {"batch_size": 10, "max_secondary_narratives": 2}}
    types_ref = {"tickers": {"AAA": {"name": "AAA Corp"}}}
    reference_cache = {"tickers": {"AAA": {"sic_code": "1", "sic_description": "X", "description": "y"}}}
    checkpoint = tmp_path / "checkpoint.jsonl"

    with patch.object(llm_provider, "generate_ticker_classifications", side_effect=failing_generate):
        errors, n_classified = r.classify_tickers(
            ["AAA"], taxonomy_by_id, catalog_by_id, catalog_by_norm,
            types_ref, reference_cache, cfg, str(checkpoint), "daily_reconciliation")

    assert n_classified == 0
    assert errors
    taxonomy = make_taxonomy(list(taxonomy_by_id.values()))
    coverage_errors = r.validate_full_coverage(taxonomy, {"AAA"}, max_secondary=2)
    assert coverage_errors  # AAA has no Primary Narrative -> a real run would exit(1), never write


# ── Migration: legacy schema backfill (point 22, exercised alongside A-K) ──

def test_migrate_membership_schema_is_idempotent():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        "AAA": {"role": "core", "confidence": 85, "reason": "x", "added_at": "2026-01-01", "last_reviewed_at": "2026-01-01"},
    }}])
    migrated_first = r.migrate_membership_schema(taxonomy, "2026-08-12")
    assert migrated_first == 1
    assert taxonomy["narratives"][0]["tickers"]["AAA"]["assignment_priority"] == "primary"

    migrated_second = r.migrate_membership_schema(taxonomy, "2026-08-13")
    assert migrated_second == 0  # already migrated -> no-op


def test_migrate_membership_schema_backfills_single_membership_as_primary():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        "SOLO": {"role": "secondary", "confidence": 70, "reason": "x", "added_at": "2026-01-01", "last_reviewed_at": "2026-01-01"},
    }}])
    r.migrate_membership_schema(taxonomy, "2026-08-12")
    assert taxonomy["narratives"][0]["tickers"]["SOLO"]["assignment_priority"] == "primary"
