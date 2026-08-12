"""
Tests for scripts/reconcile_narrative_universe.py — Full-Universe semantic
classification/reconciliation PLUS the Narrative Taxonomy Quality Patch
(confidence gates, no-more-aggressive-healing reference resolution, Level-1/
Level-2 dedup, active/undersized stats, one-time cleanup merge). All
synthetic data — no network / no ANTHROPIC_API_KEY required (LLM calls are
mocked).
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


def make_result(ticker="AAA", primary_narrative_id="n1", primary_is_new=False,
                 primary_new_name=None, primary_new_definition=None, primary_confidence=90,
                 secondary_narratives=None, reasoning="x"):
    return {
        "ticker": ticker, "primary_narrative_id": primary_narrative_id, "primary_is_new": primary_is_new,
        "primary_new_name": primary_new_name, "primary_new_definition": primary_new_definition,
        "primary_confidence": primary_confidence, "secondary_narratives": secondary_narratives or [],
        "reasoning": reasoning,
    }


# ── structural invariants: no longer a coverage requirement (Quality Patch) ──

def test_validate_structural_invariants_passes_with_zero_primaries_now():
    # 100% Narrative Coverage is explicitly NOT required anymore -- a ticker
    # with no Primary at all must NOT fail validation.
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {}}])
    errors = r.validate_structural_invariants(taxonomy, max_secondary=2)
    assert errors == []


def test_validate_structural_invariants_fails_when_two_primaries_for_same_ticker():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {"AAA": make_membership("primary")}},
        {"id": "n2", "name": "N2", "tickers": {"AAA": make_membership("primary")}},
    ])
    errors = r.validate_structural_invariants(taxonomy, max_secondary=2)
    assert any("2 Primary" in e for e in errors)


def test_validate_structural_invariants_fails_with_more_than_max_secondary():
    taxonomy = make_taxonomy([
        {"id": "n1", "name": "N1", "tickers": {"AAA": make_membership("primary")}},
        {"id": "n2", "name": "N2", "tickers": {"AAA": make_membership("secondary")}},
        {"id": "n3", "name": "N3", "tickers": {"AAA": make_membership("secondary")}},
        {"id": "n4", "name": "N4", "tickers": {"AAA": make_membership("secondary")}},
    ])
    errors = r.validate_structural_invariants(taxonomy, max_secondary=2)
    assert any("Secondary Narratives" in e for e in errors)


# ── non-eligible ticker stays classified, excluded from active set ──

def test_mark_active_eligible_flips_false_without_removing_membership():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {"AAA": make_membership(active_eligible=True)}}])
    r.mark_active_eligible(taxonomy, eligible_now=set())  # AAA no longer eligible today
    assert taxonomy["narratives"][0]["tickers"]["AAA"]["active_eligible"] is False
    assert "AAA" in taxonomy["narratives"][0]["tickers"]  # membership itself untouched


# ── universe changes — reentry reuses, first-time entry needs classification ──

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
    to_classify = [t for t in changes["entered"] if t not in classified]
    assert to_classify == []
    assert changes["reentered"] == ["AAA"]


def test_true_new_entry_is_targeted_for_classification():
    taxonomy = make_taxonomy([])
    changes = r.compute_universe_changes(taxonomy, eligible_now={"BRAND_NEW"})
    classified = r.compute_classified_tickers(taxonomy)
    to_classify = [t for t in changes["entered"] if t not in classified]
    assert to_classify == ["BRAND_NEW"]


# ── classification context excludes ALL momentum/RS/thrust/price fields ──

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
    assert "KEINE Kurs-, RS-, Thrust- oder Performance-Daten" in r.CLASSIFICATION_SYSTEM_PROMPT


# ═══════════════════════════════════════════════════════════
# CONFIDENCE GATES (Quality Patch section 3 / spec test list points 1-5)
# ═══════════════════════════════════════════════════════════

def _existing_taxonomy_by_id(*ids):
    """Test helper: taxonomy_by_id + matching catalog_by_id/catalog_by_norm_name
    for narratives that already exist BEFORE classification (the realistic
    setup — in production these three dicts are always built from the same
    taxonomy, see main())."""
    taxonomy_by_id = {nid: {"id": nid, "name": nid.upper(), "tickers": {}} for nid in ids}
    catalog_by_id = {nid: {"name": nid.upper(), "classification_hint": "x"} for nid in ids}
    catalog_by_norm = {r.normalize_narrative_name(nid.upper()): nid for nid in ids}
    return taxonomy_by_id, catalog_by_id, catalog_by_norm


def test_primary_confidence_90_accepted_as_core():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1")
    result = make_result(primary_confidence=90)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "applied"
    assert taxonomy_by_id["n1"]["tickers"]["AAA"]["role"] == "core"
    assert taxonomy_by_id["n1"]["tickers"]["AAA"]["assignment_priority"] == "primary"


def test_primary_confidence_75_accepted_as_secondary_role_but_primary_assignment():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1")
    result = make_result(primary_confidence=75)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "applied"
    membership = taxonomy_by_id["n1"]["tickers"]["AAA"]
    assert membership["role"] == "secondary"          # below core_confidence_minimum (85)
    assert membership["assignment_priority"] == "primary"  # role != assignment_priority (deliberately distinct)


def test_primary_confidence_69_not_accepted_triggers_retry():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1")
    result = make_result(primary_confidence=69)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "retry"
    assert outcome["reason"] == "low_confidence_primary"
    assert taxonomy_by_id["n1"]["tickers"] == {}  # nothing written -- no forced low-confidence membership


def test_secondary_confidence_69_dropped_silently_primary_still_applied():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1", "n2")
    result = make_result(primary_confidence=90,
                          secondary_narratives=[{"narrative_id": "n2", "is_new": False, "new_name": None,
                                                  "new_definition": None, "confidence": 69}])
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "applied"
    assert "AAA" not in taxonomy_by_id["n2"]["tickers"]  # secondary below 70 -> dropped
    assert "AAA" in taxonomy_by_id["n1"]["tickers"]      # primary unaffected


def test_secondary_confidence_70_accepted():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1", "n2")
    result = make_result(primary_confidence=90,
                          secondary_narratives=[{"narrative_id": "n2", "is_new": False, "new_name": None,
                                                  "new_definition": None, "confidence": 70}])
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "applied"
    assert "AAA" in taxonomy_by_id["n2"]["tickers"]


# ═══════════════════════════════════════════════════════════
# UNKNOWN NARRATIVE REFERENCES — no more aggressive healing
# (Quality Patch section 5 / spec test list points 6-9)
# ═══════════════════════════════════════════════════════════

def test_unknown_id_without_is_new_never_auto_creates():
    resolved_id, created, err = r.resolve_narrative_reference(
        "totally_unknown_thing", False, None, None, {}, {})
    assert created is False
    assert resolved_id is None
    assert err == "unresolvable:no_match"


def test_unknown_id_with_single_plausible_existing_match_resolves_to_it():
    # Real production case: LLM referenced 'healthcare_services' while the
    # catalog only has 'healthcare_services_facilities' -- a single, clearly
    # plausible existing match must be used, NOT retried, NOT auto-created.
    catalog_by_id = {"healthcare_services_facilities": {"name": "Healthcare Services & Facilities", "classification_hint": "x"}}
    catalog_by_norm = {r.normalize_narrative_name("Healthcare Services & Facilities"): "healthcare_services_facilities"}
    resolved_id, created, err = r.resolve_narrative_reference(
        "healthcare_services", False, None, None, catalog_by_id, catalog_by_norm)
    assert err is None
    assert created is False
    assert resolved_id == "healthcare_services_facilities"


def test_unknown_id_with_multiple_plausible_matches_is_ambiguous_not_created():
    catalog_by_id = {
        "a": {"name": "Regional Banks East", "classification_hint": "x"},
        "b": {"name": "Regional Banks West", "classification_hint": "x"},
    }
    catalog_by_norm = {r.normalize_narrative_name(v["name"]): k for k, v in catalog_by_id.items()}
    resolved_id, created, err = r.resolve_narrative_reference(
        "regional_banks", False, None, None, catalog_by_id, catalog_by_norm)
    assert resolved_id is None
    assert created is False
    assert err == "unresolvable:ambiguous"
    assert len(catalog_by_id) == 2  # nothing created


def test_unknown_id_with_no_match_is_retried_not_created():
    catalog_by_id = {"semiconductors": {"name": "Semiconductors", "classification_hint": "x"}}
    catalog_by_norm = {"semiconductors": "semiconductors"}
    resolved_id, created, err = r.resolve_narrative_reference(
        "quantum_computing_hardware", False, None, None, catalog_by_id, catalog_by_norm)
    assert resolved_id is None
    assert created is False
    assert err == "unresolvable:no_match"
    assert len(catalog_by_id) == 1  # nothing created


def test_apply_classification_result_retries_instead_of_healing_unresolvable_primary():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = {}, {}, {}
    result = make_result(primary_narrative_id="ghost", primary_is_new=False, primary_confidence=90)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "retry"
    assert outcome["reason"] == "unresolvable:no_match"
    assert taxonomy_by_id == {}   # NOTHING written -- no more implicit-CREATE-from-id fallback
    assert catalog_by_id == {}    # no phantom narrative created either


def test_apply_classification_result_drops_unresolvable_secondary_silently():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = _existing_taxonomy_by_id("n1")
    result = make_result(primary_confidence=90,
                          secondary_narratives=[{"narrative_id": "ghost_secondary", "is_new": False,
                                                  "new_name": None, "new_definition": None, "confidence": 90}])
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "applied"   # primary succeeds regardless
    assert "AAA" in taxonomy_by_id["n1"]["tickers"]
    assert len(taxonomy_by_id) == 1  # no phantom narrative created for the dropped secondary


# ═══════════════════════════════════════════════════════════
# NEW NARRATIVE CREATION GATES (Quality Patch section 6 / test list 10-13)
# ═══════════════════════════════════════════════════════════

def test_new_narrative_with_name_and_definition_and_confidence_creates():
    catalog_by_id, catalog_by_norm = {}, {}
    resolved_id, created, err = r.resolve_narrative_reference(
        "regional_banks", True, "Regional Banks", "Community/regional banks", catalog_by_id, catalog_by_norm)
    assert err is None
    assert created is True
    assert catalog_by_id[resolved_id]["name"] == "Regional Banks"


def test_is_new_true_missing_name_retries_not_creates():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = {}, {}, {}
    result = make_result(primary_narrative_id="new_thing", primary_is_new=True,
                          primary_new_name=None, primary_new_definition="A definition", primary_confidence=90)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "retry"
    assert outcome["reason"] == "unresolvable:missing_new_fields"
    assert catalog_by_id == {}


def test_is_new_true_missing_definition_retries_not_creates():
    taxonomy_by_id, catalog_by_id, catalog_by_norm = {}, {}, {}
    result = make_result(primary_narrative_id="new_thing", primary_is_new=True,
                          primary_new_name="New Thing", primary_new_definition=None, primary_confidence=90)
    outcome = r.apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm, result,
                                             MEMBERSHIP_CFG, 2, "2026-08-12", "daily_reconciliation")
    assert outcome["status"] == "retry"
    assert outcome["reason"] == "unresolvable:missing_new_fields"
    assert catalog_by_id == {}


def test_is_new_true_near_duplicate_of_existing_reuses_existing_no_duplicate_create():
    catalog_by_id = {"healthcare_services_facilities": {"name": "Healthcare Services & Facilities", "classification_hint": "x"}}
    catalog_by_norm = {r.normalize_narrative_name("Healthcare Services & Facilities"): "healthcare_services_facilities"}
    resolved_id, created, err = r.resolve_narrative_reference(
        "healthcare_services", True, "Healthcare Services", "def", catalog_by_id, catalog_by_norm)
    assert err is None
    assert created is False
    assert resolved_id == "healthcare_services_facilities"
    assert len(catalog_by_id) == 1  # no duplicate narrative created


# ═══════════════════════════════════════════════════════════
# DEDUPLICATION (Quality Patch section 7 / test list 14-17)
# ═══════════════════════════════════════════════════════════

def test_airline_vs_airlines_detected_as_exact_normalized_duplicate():
    taxonomy = make_taxonomy([
        {"id": "a", "name": "Airline", "tickers": {}},
        {"id": "b", "name": "Airlines", "tickers": {}},
    ])
    dupes = r.find_duplicate_narrative_names(taxonomy)
    assert len(dupes) == 1
    (cluster,) = dupes.values()
    assert set(cluster) == {"a", "b"}


def test_ai_infrastructure_hyphen_vs_space_detected_as_duplicate():
    taxonomy = make_taxonomy([
        {"id": "a", "name": "AI-Infrastructure", "tickers": {}},
        {"id": "b", "name": "AI Infrastructure", "tickers": {}},
    ])
    dupes = r.find_duplicate_narrative_names(taxonomy)
    assert len(dupes) == 1


def test_healthcare_services_vs_facilities_is_near_duplicate_candidate_not_exact():
    taxonomy = make_taxonomy([
        {"id": "a", "name": "Healthcare Services", "tickers": {}},
        {"id": "b", "name": "Healthcare Services & Facilities", "tickers": {}},
    ])
    assert r.find_duplicate_narrative_names(taxonomy) == {}  # NOT an exact/normalized duplicate
    candidates = r.find_near_duplicate_candidates(taxonomy, threshold=0.5)
    assert len(candidates) == 1
    assert {candidates[0]["narrative_a"], candidates[0]["narrative_b"]} == {"a", "b"}


def test_semiconductors_vs_semiconductor_equipment_not_auto_equated():
    catalog_by_id = {"semiconductor_equipment": {"name": "Semiconductor Equipment", "classification_hint": "x"}}
    catalog_by_norm = {r.normalize_narrative_name("Semiconductor Equipment"): "semiconductor_equipment"}
    resolved_id, created, err = r.resolve_narrative_reference(
        "semiconductors", False, None, None, catalog_by_id, catalog_by_norm)
    # similarity (0.5) is below the auto-resolve bar -> retried, never silently equated
    assert resolved_id is None
    assert err == "unresolvable:no_match"


# ═══════════════════════════════════════════════════════════
# ACTIVE / UNDERSIZED NARRATIVE STATS (section 8/19-21, test list 18-20)
# ═══════════════════════════════════════════════════════════

def test_active_undersized_stats_narrative_with_5_eligible_is_active():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        t: make_membership("primary") for t in ["A", "B", "C", "D", "E"]
    }}])
    stats = r.compute_active_undersized_stats(taxonomy, {"A", "B", "C", "D", "E"}, minimum_active_members=5)
    assert stats["active_narrative_count"] == 1
    assert stats["undersized_narrative_count"] == 0


def test_active_undersized_stats_narrative_with_4_eligible_is_undersized():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        t: make_membership("primary") for t in ["A", "B", "C", "D"]
    }}])
    stats = r.compute_active_undersized_stats(taxonomy, {"A", "B", "C", "D"}, minimum_active_members=5)
    assert stats["active_narrative_count"] == 0
    assert stats["undersized_narrative_count"] == 1
    assert stats["undersized_narratives"][0]["eligible_member_count"] == 4


def test_active_undersized_stats_ineligible_members_dont_count_toward_active_size():
    # 4 eligible + 3 historically-classified-but-ineligible = still 4 eligible -> UNDERSIZED,
    # ineligible members must NEVER pad the active-size count.
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        **{t: make_membership("primary") for t in ["A", "B", "C", "D"]},
        **{t: make_membership("primary", active_eligible=False) for t in ["X", "Y", "Z"]},
    }}])
    stats = r.compute_active_undersized_stats(taxonomy, {"A", "B", "C", "D"}, minimum_active_members=5)
    assert stats["undersized_narrative_count"] == 1
    assert stats["undersized_narratives"][0]["eligible_member_count"] == 4


def test_eligible_with_active_narrative_reflects_primary_only():
    active_narrative = {"id": "big", "name": "Big", "tickers": {
        t: make_membership("primary") for t in ["A", "B", "C", "D", "E"]
    }}
    undersized_narrative = {"id": "small", "name": "Small", "tickers": {
        "F": make_membership("primary"),
    }}
    taxonomy = make_taxonomy([active_narrative, undersized_narrative])
    eligible_now = {"A", "B", "C", "D", "E", "F"}
    stats = r.compute_active_undersized_stats(taxonomy, eligible_now, minimum_active_members=5)
    assert stats["eligible_with_active_narrative"] == 5   # A-E, via the active narrative
    assert stats["eligible_without_active_narrative"] == 1  # F's only narrative is undersized


# ═══════════════════════════════════════════════════════════
# MALFORMED LLM OUTPUT
# ═══════════════════════════════════════════════════════════

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


def test_parse_classification_result_tolerates_is_new_without_name():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "x", "primary_is_new": True,
         "primary_confidence": 90, "reasoning": "y"},
        valid_tickers={"AAA"})
    assert err is None
    assert result["primary_new_name"] is None
    assert result["primary_is_new"] is True


def test_parse_classification_result_skips_malformed_secondary_without_failing_whole_row():
    result, err = r.parse_classification_result(
        {"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": 90, "reasoning": "x",
         "secondary_narratives": [{"narrative_id": "n2", "confidence": "not a number"},
                                   {"narrative_id": "n3", "confidence": 75}]},
        valid_tickers={"AAA"})
    assert err is None
    assert len(result["secondary_narratives"]) == 1
    assert result["secondary_narratives"][0]["narrative_id"] == "n3"


# ═══════════════════════════════════════════════════════════
# CLASSIFY_TICKERS: retries + no-forced-coverage outcome
# ═══════════════════════════════════════════════════════════

def test_llm_failure_leaves_ticker_unresolved_but_run_still_succeeds(tmp_path):
    """A total LLM-call failure is logged (hard_errors) and the ticker ends
    up unresolved (no active narrative) -- it must NOT be forced into any
    narrative, and (Quality Patch) it must NOT fail validate_structural_invariants
    either, since 'no narrative' is now an accepted outcome."""
    def failing_generate(system_prompt, user_prompt, model=None):
        raise llm_provider.LLMError("simulated failure")

    taxonomy_by_id = {}
    catalog_by_id, catalog_by_norm = {}, {}
    cfg = {"membership": MEMBERSHIP_CFG,
           "classification": {"batch_size": 10, "max_secondary_narratives": 2, "max_retries_per_ticker": 1}}
    types_ref = {"tickers": {"AAA": {"name": "AAA Corp"}}}
    reference_cache = {"tickers": {"AAA": {"sic_code": "1", "sic_description": "X", "description": "y"}}}
    checkpoint = tmp_path / "checkpoint.jsonl"

    with patch.object(llm_provider, "generate_ticker_classifications", side_effect=failing_generate):
        applied_count, unresolved, hard_errors = r.classify_tickers(
            ["AAA"], taxonomy_by_id, catalog_by_id, catalog_by_norm,
            types_ref, reference_cache, cfg, str(checkpoint), "daily_reconciliation")

    assert applied_count == 0
    assert hard_errors
    assert any(x["ticker"] == "AAA" for x in unresolved)
    taxonomy = make_taxonomy(list(taxonomy_by_id.values()))
    assert r.validate_structural_invariants(taxonomy, max_secondary=2) == []  # NOT a failure anymore


def test_classify_tickers_retries_low_confidence_then_succeeds(tmp_path):
    """First attempt: confidence 60 (rejected, retry). Retry attempt:
    confidence 90 (accepted). Proves the retry loop actually re-queries and
    applies a subsequent successful attempt."""
    calls = {"n": 0}

    def sequenced_generate(system_prompt, user_prompt, model=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return [{"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": 60, "reasoning": "unsure"}]
        return [{"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": 90, "reasoning": "confident"}]

    taxonomy_by_id = {"n1": {"id": "n1", "name": "N1", "tickers": {}}}
    catalog_by_id, catalog_by_norm = r.build_catalog_index(make_taxonomy(list(taxonomy_by_id.values())))
    cfg = {"membership": MEMBERSHIP_CFG,
           "classification": {"batch_size": 10, "max_secondary_narratives": 2, "max_retries_per_ticker": 2}}
    types_ref = {"tickers": {"AAA": {"name": "AAA Corp"}}}
    reference_cache = {"tickers": {"AAA": {"sic_code": "1", "sic_description": "X", "description": "y"}}}
    checkpoint = tmp_path / "checkpoint.jsonl"

    with patch.object(llm_provider, "generate_ticker_classifications", side_effect=sequenced_generate):
        applied_count, unresolved, hard_errors = r.classify_tickers(
            ["AAA"], taxonomy_by_id, catalog_by_id, catalog_by_norm,
            types_ref, reference_cache, cfg, str(checkpoint), "daily_reconciliation")

    assert applied_count == 1
    assert unresolved == []
    assert "AAA" in taxonomy_by_id["n1"]["tickers"]
    assert calls["n"] == 2  # original batch + exactly one retry round


def test_classify_tickers_gives_up_after_max_retries_without_forcing_membership(tmp_path):
    def always_low_confidence(system_prompt, user_prompt, model=None):
        return [{"ticker": "AAA", "primary_narrative_id": "n1", "primary_confidence": 50, "reasoning": "unsure"}]

    taxonomy_by_id = {"n1": {"id": "n1", "name": "N1", "tickers": {}}}
    catalog_by_id, catalog_by_norm = r.build_catalog_index(make_taxonomy(list(taxonomy_by_id.values())))
    cfg = {"membership": MEMBERSHIP_CFG,
           "classification": {"batch_size": 10, "max_secondary_narratives": 2, "max_retries_per_ticker": 2}}
    types_ref = {"tickers": {"AAA": {"name": "AAA Corp"}}}
    reference_cache = {"tickers": {"AAA": {"sic_code": "1", "sic_description": "X", "description": "y"}}}
    checkpoint = tmp_path / "checkpoint.jsonl"

    with patch.object(llm_provider, "generate_ticker_classifications", side_effect=always_low_confidence):
        applied_count, unresolved, hard_errors = r.classify_tickers(
            ["AAA"], taxonomy_by_id, catalog_by_id, catalog_by_norm,
            types_ref, reference_cache, cfg, str(checkpoint), "daily_reconciliation")

    assert applied_count == 0
    assert [x["ticker"] for x in unresolved] == ["AAA"]
    assert "AAA" not in taxonomy_by_id["n1"]["tickers"]  # no forced low-confidence membership


# ═══════════════════════════════════════════════════════════
# CLEANUP: deterministic merge (test list 32-33)
# ═══════════════════════════════════════════════════════════

def test_merge_duplicate_narratives_transfers_membership_metadata():
    taxonomy = make_taxonomy([
        {"id": "airline", "name": "Airline", "created_at": "2026-01-01", "tickers": {
            "AAA": make_membership("primary", role="core", confidence=91),
        }},
        {"id": "airlines", "name": "Airlines", "created_at": "2026-02-01", "tickers": {
            "BBB": make_membership("secondary", role="secondary", confidence=75),
        }},
    ])
    merged, merge_log = r.merge_duplicate_narratives(taxonomy)
    assert len(merged["narratives"]) == 1
    survivor = merged["narratives"][0]
    assert survivor["id"] == "airline"  # more tickers pre-merge tie-break -> deterministic (both have 1; earlier created_at wins)
    assert "AAA" in survivor["tickers"] and "BBB" in survivor["tickers"]
    assert survivor["tickers"]["AAA"]["confidence"] == 91  # metadata preserved, not overwritten
    assert survivor["tickers"]["BBB"]["confidence"] == 75
    assert len(merge_log) == 1
    assert merge_log[0]["into"] == "airline"
    assert merge_log[0]["merged_from"] == ["airlines"]


def test_merge_duplicate_narratives_resolves_conflicting_membership_by_priority():
    # Same ticker present in BOTH duplicates: Primary/higher-confidence wins
    # (point 18's priority order), never silently duplicated.
    taxonomy = make_taxonomy([
        {"id": "airline", "name": "Airline", "created_at": "2026-01-01", "tickers": {
            "AAA": make_membership("secondary", role="secondary", confidence=72),
        }},
        {"id": "airlines", "name": "Airlines", "created_at": "2026-02-01", "tickers": {
            "AAA": make_membership("primary", role="core", confidence=91),
        }},
    ])
    merged, merge_log = r.merge_duplicate_narratives(taxonomy)
    survivor = merged["narratives"][0]
    assert len(survivor["tickers"]) == 1  # no duplicate row for AAA
    assert survivor["tickers"]["AAA"]["assignment_priority"] == "primary"  # primary beats secondary
    assert survivor["tickers"]["AAA"]["confidence"] == 91


def test_merge_duplicate_narratives_is_noop_without_duplicates():
    taxonomy = make_taxonomy([{"id": "n1", "name": "Unique Name", "tickers": {}}])
    merged, merge_log = r.merge_duplicate_narratives(taxonomy)
    assert merge_log == []
    assert len(merged["narratives"]) == 1


# ── run_cleanup: end-to-end orchestration (deterministic parts only —
# LLM reclassification itself is exercised via classify_tickers tests above) ──

def test_run_cleanup_removes_secondary_below_minimum_without_reclassification(tmp_path):
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        "AAA": make_membership("primary", confidence=90),
        "BBB": make_membership("secondary", confidence=65),
    }}])
    cfg = {"membership": MEMBERSHIP_CFG,
           "classification": {"batch_size": 10, "max_secondary_narratives": 2, "max_retries_per_ticker": 0,
                               "near_duplicate_similarity_threshold": 0.5, "minimum_active_narrative_members": 5}}
    market_features = {"tickers": {"AAA": {"eligible": True}, "BBB": {"eligible": True}}}
    types_ref, reference_cache = {"tickers": {}}, {"tickers": {}}
    taxonomy, proposal, active_stats, before_ids = r.run_cleanup(
        taxonomy, market_features, types_ref, reference_cache, cfg, str(tmp_path / "cp.jsonl"), "2026-08-12")
    assert "BBB" not in taxonomy["narratives"][0]["tickers"]  # secondary below min removed
    assert "AAA" in taxonomy["narratives"][0]["tickers"]      # primary untouched
    assert proposal["low_confidence_memberships"]["secondary_below_minimum_removed"][0]["ticker"] == "BBB"


def test_run_cleanup_prunes_fully_empty_narrative_after_stripping(tmp_path):
    # A narrative whose ONLY member was a low-confidence primary ends up
    # with zero tickers after stripping; if reclassification can't be
    # attempted (no LLM available / max_retries_per_ticker=0 short-circuits
    # immediately), it must not linger as a dead, empty taxonomy entry.
    taxonomy = make_taxonomy([{"id": "dead", "name": "Dead Narrative", "tickers": {
        "AAA": make_membership("primary", confidence=50),
    }}])
    cfg = {"membership": MEMBERSHIP_CFG,
           "classification": {"batch_size": 10, "max_secondary_narratives": 2, "max_retries_per_ticker": 0,
                               "near_duplicate_similarity_threshold": 0.5, "minimum_active_narrative_members": 5}}
    market_features = {"tickers": {"AAA": {"eligible": True}}}
    types_ref, reference_cache = {"tickers": {"AAA": {"name": "AAA Corp"}}}, {"tickers": {"AAA": {"sic_code": "1", "sic_description": "x", "description": "y"}}}

    def failing_generate(system_prompt, user_prompt, model=None):
        raise llm_provider.LLMError("no key")

    with patch.object(llm_provider, "generate_ticker_classifications", side_effect=failing_generate):
        taxonomy, proposal, active_stats, before_ids = r.run_cleanup(
            taxonomy, market_features, types_ref, reference_cache, cfg, str(tmp_path / "cp.jsonl"), "2026-08-12")

    assert taxonomy["narratives"] == []  # the now-empty "dead" narrative was pruned
    assert proposal["pruned_empty_narratives"] == ["dead"]
    assert before_ids == {"dead"}


# ── low-confidence membership audit (test list, section 16) ──

def test_find_low_confidence_memberships_buckets_correctly():
    taxonomy = make_taxonomy([{"id": "n1", "name": "N1", "tickers": {
        "LOW_PRIMARY": make_membership("primary", confidence=60),
        "MID_PRIMARY": make_membership("primary", confidence=80),
        "HIGH_PRIMARY": make_membership("primary", confidence=95),
        "LOW_SECONDARY": make_membership("secondary", confidence=65),
    }}])
    buckets = r.find_low_confidence_memberships(taxonomy, MEMBERSHIP_CFG)
    assert [x["ticker"] for x in buckets["primary_below_minimum"]] == ["LOW_PRIMARY"]
    assert [x["ticker"] for x in buckets["primary_70_to_84"]] == ["MID_PRIMARY"]
    assert [x["ticker"] for x in buckets["primary_85_plus"]] == ["HIGH_PRIMARY"]
    assert [x["ticker"] for x in buckets["secondary_below_minimum"]] == ["LOW_SECONDARY"]


# ── Migration: legacy schema backfill (exercised alongside the above) ──

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
