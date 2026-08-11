#!/usr/bin/env python3
"""
YOLO Dashboard — Deterministic Narrative Proposal Validator
Validates a weekly-review Change Set (see data/taxonomy/proposals/*.json)
against the rules in config/narrative_engine.json BEFORE it is allowed onto
a branch / into a pull request. The LLM's output is never trusted directly —
every proposal must pass through `validate_proposal()` first.

Proposal schema (schema_version 1):
{
  "schema_version": 1,
  "created_at": "<ISO date>",
  "review_type": "weekly",
  "changes": [
    {
      "action": "CREATE"|"ADD"|"REMOVE"|"STATUS_CHANGE"|"MERGE_PROPOSAL"|"SPLIT_PROPOSAL",
      "narrative": "<narrative id>",
      "narrative_name": "<display name, required for CREATE>",
      "ticker": "<symbol, for single-ticker ADD/REMOVE>",
      "tickers": ["<symbol>", ...],   # for CREATE (basket members) / SPLIT
      "role": "core"|"secondary",     # required for ADD, per-member for CREATE
      "confidence": 0-100,
      "quantitative_evidence": {...}, # required, non-empty, action-specific shape (see below)
      "semantic_evidence": "<economic reasoning, required non-empty string>",
      "previous_state": {...}|null,
      "proposed_state": {...},
      "reason": "<free text>",
      "remove_reason_code": "misclassification"|"business_model_change"|
                             "narrative_exposure_ended"|"m_and_a"|"delisting"|
                             "ticker_not_found"|"clerical_error"|"other_fundamental",
      "emergency": true|false        # REMOVE only: bypasses the two-review rule
    }
  ]
}

CREATE quantitative_evidence shape:
  {"members": [{"ticker": "X", "role": "core"|"secondary",
                "rs_percentile_1w": 0-100, "rs_percentile_1m": 0-100}, ...],
   "breadth_pct": 0-100, "thrust": <float>}

No unstructured free-text changes are accepted anywhere in `changes`: every
entry must match one of the action shapes below, or the whole proposal is
rejected (fail-closed, per point 16/18 of the spec — a schema or validation
error must stop the workflow, never fall back to publishing anyway).
"""

import argparse
import json
import sys
from pathlib import Path

VALID_ACTIONS = {"CREATE", "ADD", "REMOVE", "STATUS_CHANGE", "MERGE_PROPOSAL", "SPLIT_PROPOSAL"}
VALID_ROLES = {"core", "secondary"}
VALID_STATUSES = {"emerging", "active", "mature", "fading", "dormant"}
VALID_REMOVE_REASONS = {
    "misclassification", "business_model_change", "narrative_exposure_ended",
    "m_and_a", "delisting", "ticker_not_found", "clerical_error", "other_fundamental",
}
EMERGENCY_REMOVE_REASONS = {"delisting", "ticker_not_found", "clerical_error"}
HUMAN_REVIEW_ALWAYS = {"CREATE", "REMOVE", "MERGE_PROPOSAL", "SPLIT_PROPOSAL"}


class ValidationResult:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.annotated = []

    @property
    def ok(self):
        return len(self.errors) == 0

    def err(self, idx, msg):
        self.errors.append(f"changes[{idx}]: {msg}")

    def warn(self, idx, msg):
        self.warnings.append(f"changes[{idx}]: {msg}")


def _narrative_by_id(taxonomy, nid):
    for n in taxonomy.get("narratives", []):
        if n["id"] == nid:
            return n
    return None


def _as_number(v):
    """LLM tool-call output sometimes stringifies numbers (e.g. thrust: "0.42"
    instead of 0.42) even though the tool schema declares them as numbers —
    observed in production on the first real weekly-review run. Treat a
    non-numeric value the same as missing rather than crashing the whole
    validation run on a raw TypeError (fail-closed on the specific field,
    not on the entire proposal)."""
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _overlap_ratio(proposed_tickers, existing_tickers):
    proposed = set(proposed_tickers)
    if not proposed:
        return 0.0, None
    best_ratio, best_id = 0.0, None
    for nid, members in existing_tickers.items():
        inter = proposed & members
        ratio = len(inter) / len(proposed) * 100
        if ratio > best_ratio:
            best_ratio, best_id = ratio, nid
    return round(best_ratio, 1), best_id


def _count_consecutive_prior_removes(narrative_id, ticker, proposal_history, current_date):
    """proposal_history: list of (date_str, proposal_dict) sorted ascending,
    all with date < current_date. Returns True if the immediately preceding
    weekly proposal also contained a REMOVE for the same (narrative, ticker)."""
    prior = [p for d, p in proposal_history if d < current_date]
    if not prior:
        return False
    last = prior[-1]
    for c in last.get("changes", []):
        if c.get("action") == "REMOVE" and c.get("narrative") == narrative_id and \
           (c.get("ticker") == ticker or ticker in (c.get("tickers") or [])):
            return True
    return False


def validate_proposal(proposal, taxonomy, config, known_tickers=None, proposal_history=None):
    """Pure function — no file I/O. `known_tickers` is an optional
    {symbol: {"type": ..., "eligible": ...}} map (from market_features.json)
    used for ticker-existence/asset-type checks; if None, those checks are
    skipped (documented limitation, e.g. for unit tests / offline runs).
    `proposal_history` is an optional list of (date_str, proposal_dict) used
    for the REMOVE two-review rule.
    """
    result = ValidationResult()
    proposal_history = proposal_history or []
    current_date = proposal.get("created_at", "")

    if proposal.get("schema_version") != 1:
        result.errors.append("schema_version muss 1 sein")
        return result
    if not isinstance(proposal.get("changes"), list):
        result.errors.append("changes muss eine Liste sein")
        return result
    if not proposal["changes"]:
        return result  # leerer Change Set ist gültig ("diese Woche nichts vorzuschlagen")

    existing_ticker_sets = {
        n["id"]: set(n.get("tickers", {}).keys()) for n in taxonomy.get("narratives", [])
    }

    create_cfg = config["create"]
    membership_cfg = config["membership"]

    for idx, change in enumerate(proposal["changes"]):
        if not isinstance(change, dict):
            result.err(idx, "Eintrag ist kein Objekt (unstrukturierte Freitextänderung nicht erlaubt)")
            continue

        action = change.get("action")
        if action not in VALID_ACTIONS:
            result.err(idx, f"ungültige action '{action}'")
            continue

        if not change.get("semantic_evidence"):
            result.err(idx, "semantic_evidence fehlt oder ist leer")
        if not change.get("reason"):
            result.err(idx, "reason fehlt oder ist leer")

        annotated = dict(change)
        annotated["human_review_required"] = action in HUMAN_REVIEW_ALWAYS

        # ── Ticker existence / asset type ─────────────────────────────
        tickers_in_change = set(change.get("tickers") or [])
        if change.get("ticker"):
            tickers_in_change.add(change["ticker"])
        if known_tickers is not None:
            for t in tickers_in_change:
                meta = known_tickers.get(t)
                if meta is None:
                    result.err(idx, f"Ticker '{t}' existiert nicht im bekannten Universum (erfundenes Symbol?)")
                elif config["universe"].get("stocks_only") and meta.get("type") in config["universe"]["excluded_types"]:
                    result.err(idx, f"Ticker '{t}' hat Asset-Type '{meta.get('type')}' — kein reguläres Stock")

        # ── Role / confidence ──────────────────────────────────────────
        if change.get("role") is not None and change["role"] not in VALID_ROLES:
            result.err(idx, f"ungültige role '{change['role']}' (nur core/secondary)")

        # ── Action-specific rules ──────────────────────────────────────
        if action == "CREATE":
            if not change.get("narrative_name"):
                result.err(idx, "CREATE benötigt narrative_name")
            members = (change.get("quantitative_evidence") or {}).get("members") or []
            if len(members) < create_cfg["minimum_members"]:
                result.err(idx, f"CREATE: nur {len(members)} Mitglieder, benötigt >= {create_cfg['minimum_members']}")
            core_members = [m for m in members if m.get("role") == "core"]
            if len(core_members) < create_cfg["minimum_core_members"]:
                result.err(idx, f"CREATE: nur {len(core_members)} Core-Mitglieder, benötigt >= {create_cfg['minimum_core_members']}")
            rs1w = [_as_number(m.get("rs_percentile_1w")) for m in members]
            rs1m = [_as_number(m.get("rs_percentile_1m")) for m in members]
            rs_hits = [i for i in range(len(members)) if (rs1w[i] or 0) >= 80 or (rs1m[i] or 0) >= 80]
            share_rs80 = (len(rs_hits) / len(members) * 100) if members else 0
            if share_rs80 < create_cfg["minimum_share_rs80_pct"]:
                result.err(idx, f"CREATE: nur {share_rs80:.0f}% der Mitglieder mit RS>=80, benötigt >= {create_cfg['minimum_share_rs80_pct']}%")
            rs90_count = sum(1 for i in range(len(members)) if (rs1w[i] or 0) >= 90 or (rs1m[i] or 0) >= 90)
            if rs90_count < create_cfg["minimum_count_rs90"]:
                result.err(idx, f"CREATE: nur {rs90_count} Mitglieder mit RS>=90, benötigt >= {create_cfg['minimum_count_rs90']}")
            evidence = change.get("quantitative_evidence") or {}
            breadth = _as_number(evidence.get("breadth_pct"))
            if breadth is None or breadth < create_cfg["minimum_breadth_pct"]:
                result.err(idx, f"CREATE: Breadth {evidence.get('breadth_pct')!r} < benötigt {create_cfg['minimum_breadth_pct']}%")
            thrust = _as_number(evidence.get("thrust"))
            if create_cfg["positive_thrust_required"] and (thrust is None or thrust <= 0):
                result.err(idx, f"CREATE: Thrust {evidence.get('thrust')!r} ist nicht positiv oder keine gültige Zahl")

            proposed_tickers = {m["ticker"] for m in members if m.get("ticker")}
            ratio, overlap_with = _overlap_ratio(proposed_tickers, existing_ticker_sets)
            annotated["overlap_pct"] = ratio
            annotated["overlap_with"] = overlap_with
            if ratio >= create_cfg["overlap_warning_threshold_pct"]:
                result.warn(idx, f"CREATE: Overlap {ratio}% mit bestehendem Narrative '{overlap_with}' >= "
                                  f"{create_cfg['overlap_warning_threshold_pct']}% Warnschwelle (kein automatisches Verbot)")
                annotated["overlap_warning"] = True
            else:
                annotated["overlap_warning"] = False

        elif action == "ADD":
            if not change.get("narrative"):
                result.err(idx, "ADD benötigt narrative (Ziel-Narrative-ID)")
            if not change.get("ticker"):
                result.err(idx, "ADD benötigt ticker")
            role = change.get("role")
            confidence = _as_number(change.get("confidence"))
            if role == "core" and (confidence is None or confidence < membership_cfg["core_confidence_minimum"]):
                result.err(idx, f"ADD core: confidence {change.get('confidence')!r} < {membership_cfg['core_confidence_minimum']}")
            elif role == "secondary" and (confidence is None or confidence < membership_cfg["secondary_confidence_minimum"]):
                result.err(idx, f"ADD secondary: confidence {change.get('confidence')!r} < {membership_cfg['secondary_confidence_minimum']}")
            elif role not in VALID_ROLES:
                result.err(idx, "ADD benötigt eine gültige role (core/secondary)")

        elif action == "REMOVE":
            if not change.get("narrative"):
                result.err(idx, "REMOVE benötigt narrative")
            ticker = change.get("ticker")
            if not ticker:
                result.err(idx, "REMOVE benötigt ticker")
            reason_code = change.get("remove_reason_code")
            if reason_code not in VALID_REMOVE_REASONS:
                result.err(idx, f"REMOVE: ungültiger remove_reason_code '{reason_code}' (schwache Performance ist kein gültiger Grund)")
            narrative = _narrative_by_id(taxonomy, change.get("narrative"))
            if narrative is not None and ticker not in narrative.get("tickers", {}):
                result.err(idx, f"REMOVE: Ticker '{ticker}' ist aktuell kein Mitglied von '{change.get('narrative')}'")

            is_emergency = bool(change.get("emergency")) and reason_code in EMERGENCY_REMOVE_REASONS
            confirmed_by_prior_review = _count_consecutive_prior_removes(
                change.get("narrative"), ticker, proposal_history, current_date
            )
            annotated["actionable"] = bool(is_emergency or confirmed_by_prior_review)
            if not annotated["actionable"]:
                result.warn(idx, "REMOVE: erste Meldung, noch keine zweite aufeinanderfolgende Weekly-Review-Bestätigung "
                                  "und kein Notfall-Grund — wird als 'pending second review' markiert, nicht sofort anwendbar")

        elif action == "STATUS_CHANGE":
            new_status = (change.get("proposed_state") or {}).get("status")
            if new_status not in VALID_STATUSES:
                result.err(idx, f"STATUS_CHANGE: ungültiger Status '{new_status}'")
            if not change.get("quantitative_evidence"):
                result.err(idx, "STATUS_CHANGE benötigt quantitative_evidence (Statusänderungen müssen aus Scores ableitbar sein)")

        elif action in ("MERGE_PROPOSAL", "SPLIT_PROPOSAL"):
            # Never auto-applied — validated only for schema completeness.
            if action == "MERGE_PROPOSAL":
                targets = change.get("tickers") or change.get("merge_narratives")
                if not change.get("narrative") or not targets:
                    result.err(idx, "MERGE_PROPOSAL benötigt narrative + merge_narratives (Liste der Ziel-IDs)")
            else:
                proposed_split = (change.get("proposed_state") or {}).get("split_groups")
                if not change.get("narrative") or not proposed_split:
                    result.err(idx, "SPLIT_PROPOSAL benötigt narrative + proposed_state.split_groups")

        result.annotated.append(annotated)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate a narrative proposal change set")
    parser.add_argument("proposal_file")
    parser.add_argument("--taxonomy", default="data/taxonomy/narratives.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--market-features", default="data/market_features.json",
                         help="Optional; used for ticker-existence/asset-type checks")
    parser.add_argument("--proposals-dir", default="data/taxonomy/proposals")
    args = parser.parse_args()

    with open(args.proposal_file, "r", encoding="utf-8") as f:
        proposal = json.load(f)
    with open(args.taxonomy, "r", encoding="utf-8") as f:
        taxonomy = json.load(f)
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    known_tickers = None
    mf_path = Path(args.market_features)
    if mf_path.exists():
        with open(mf_path, "r", encoding="utf-8") as f:
            known_tickers = json.load(f).get("tickers", {})

    proposal_history = []
    proposals_dir = Path(args.proposals_dir)
    this_file = Path(args.proposal_file).resolve()
    if proposals_dir.exists():
        for p in sorted(proposals_dir.glob("*.json")):
            if p.resolve() == this_file:
                continue
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            proposal_history.append((data.get("created_at", p.stem), data))
    proposal_history.sort(key=lambda x: x[0])

    result = validate_proposal(proposal, taxonomy, config, known_tickers, proposal_history)

    for w in result.warnings:
        print(f"⚠ WARNUNG: {w}")
    for e in result.errors:
        print(f"❌ FEHLER: {e}")

    if not result.ok:
        print(f"\n❌ Validierung fehlgeschlagen: {len(result.errors)} Fehler. Workflow wird gestoppt.")
        sys.exit(1)

    print(f"\n✅ Validierung erfolgreich ({len(result.annotated)} Änderungen, {len(result.warnings)} Warnungen).")
    out_path = Path(args.proposal_file)
    proposal["changes"] = result.annotated
    proposal["validation"] = {"errors": [], "warnings": result.warnings}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(proposal, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
