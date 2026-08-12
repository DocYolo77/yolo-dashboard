#!/usr/bin/env python3
"""
YOLO Dashboard — Weekly AI Narrative Review
Orchestrates the QUALITY half of the Narrative Engine — coverage/scale is
NOT its job anymore (Full-Universe spec point 17). Every eligible ticker
already gets a Primary Narrative daily via scripts/reconcile_narrative_universe.py
(semantic classification from ticker/name/SIC/description only, batched,
100%-coverage-enforced). This script runs on top of that, weekly, and asks
narrower questions a same-day incremental classifier can't:

  - Did a ticker get semantically MISCLASSIFIED (wrong Primary/Secondary)?
  - Did a company's business model change, get acquired, or delist, such
    that an existing membership is now factually wrong (REMOVE, with a
    reason code — weak price action is explicitly NEVER a valid reason)?
  - Are any Secondary memberships still materially justified?
  - Are two narratives redundant enough to MERGE, or one broad enough to
    SPLIT? (never auto-applied — always proposal-only, point 5/17)
  - Are there genuinely new economic clusters the daily classifier's
    ticket-by-ticket view wouldn't surface (CREATE)?

Because the taxonomy is now Full-Universe-sized (~1000+ tickers, not the
old ~150-ticker curated set), the maintenance pool below is capped and
staleness-prioritized (MAX_MAINTENANCE_MEMBERS_IN_PROMPT) rather than sent
in full — a full taxonomy dump would blow the LLM context AND defeats the
point of a fast, focused prompt. The taxonomy is large enough that no
single weekly run can meaningfully re-examine everything at once; picking
the least-recently-reviewed slice each week means the whole taxonomy
rotates through review over several weeks, rather than always re-checking
the same head of the list. The old Discovery pool (momentum-qualifying
tickers not yet in the taxonomy) still exists structurally, but in normal
operation it stays near-empty: daily reconciliation classifies real new
entries same-day, so this pool now mainly catches unusual momentum
clusters worth a second semantic look, not the primary coverage mechanism
it used to be (point 17's explicit point).

Orchestration, otherwise unchanged from before:
  1. Build the (now capped/rotating) Discovery pool + Maintenance pool.
  2. Ask the LLM (via scripts/llm_provider.py) for a structured Change Set
     ONLY — never a full taxonomy replacement.
  3. Run the change set through scripts/validate_narrative_proposal.py.
  4. On success: apply CREATE/ADD/actionable-REMOVE/STATUS_CHANGE to a
     working copy of data/taxonomy/narratives.json, write a dated history
     snapshot, and write the (now validator-annotated) proposal file.
     MERGE_PROPOSAL/SPLIT_PROPOSAL are NEVER applied automatically — they
     stay proposal-only for a human to act on by hand (point 15).
  5. On failure: exit non-zero with a readable error report. The calling
     GitHub Actions workflow stops there — nothing is committed, nothing
     is pushed, main is untouched (points 16/18).

What the LLM is explicitly told it may NOT do (point 11) is enforced partly
by prompt instructions and partly structurally: it never receives write
access to any file or to GitHub, it cannot invent indicator values (none
are requested from it — all quantitative fields come from
data/market_features.json), and its only valid output shape is the
`propose_narrative_changes` tool call (see llm_provider.py), which cannot
express "replace the whole taxonomy".
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from validate_narrative_proposal import validate_proposal  # noqa: E402
from reconcile_narrative_universe import (  # noqa: E402
    find_duplicate_narrative_names, find_near_duplicate_candidates,
    find_low_confidence_memberships, compute_eligible_set, compute_active_undersized_stats,
)
import llm_provider  # noqa: E402

MAX_DISCOVERY_CANDIDATES_IN_PROMPT = 150
MAX_MAINTENANCE_MEMBERS_IN_PROMPT = 150

SYSTEM_PROMPT = """Du bist der wöchentliche Narrative-Review-Assistent eines Trading-Dashboards.

Wichtig: fast jeder eligible Ticker hat bereits taeglich eine automatische semantische
Erstklassifikation (Primary + ggf. Secondary Narrative) erhalten -- 100% Narrative-Coverage ist
seit dem Narrative Taxonomy Quality Patch AUSDRUECKLICH KEIN Ziel mehr (ein Ticker ganz ohne
aktives Narrative ist ein gueltiges Ergebnis). Deine Aufgabe ist NICHT, Coverage aufzubauen,
sondern die QUALITAET bestehender Klassifikationen zu pruefen: echte Fehlzuordnungen,
Geschaeftsmodelländerungen/M&A/Delisting, fragwuerdige Secondary-Memberships, redundante oder zu
breite Narrative (Merge/Split-Vorschlaege), sowie echte neue wirtschaftliche Cluster, die eine
Ticker-fuer-Ticker-Klassifikation nicht erkennen wuerde.

Der Prompt-Kontext enthaelt zusaetzlich near_duplicate_candidates (Narrative-Paare mit aehnlichem
Namen, NUR ein Hinweis -- nie automatisch gemergt, du kannst aber MERGE_PROPOSAL vorschlagen),
undersized_narratives (Narrative mit < minimum_active_narrative_members aktuell eligible
Mitgliedern -- werden im Dashboard nicht als aktive Baskets angezeigt; pruefe, ob sie wirtschaftlich
in ein bestehendes groesseres Narrative gehoeren, bevor du sie so belaesst) und
low_confidence_memberships (Primary/Secondary mit Confidence knapp ueber/unter den Schwellen --
das haben reconcile_narrative_universe.py's Confidence-Gates bereits weitgehend bereinigt, aber
pruefe verbleibende Grenzfaelle auf echte Fehlzuordnung).

Deine einzige Aufgabe: wirtschaftliche/semantische Einschätzungen zu Aktien-Narrativen liefern,
als strukturierter Change Set über das Tool `propose_narrative_changes`.

Du DARFST:
- wirtschaftliche Gemeinsamkeiten zwischen Kandidaten erkennen
- neue Narrative vorschlagen (CREATE)
- neue Memberships vorschlagen (ADD, Rolle core oder secondary)
- sachlich falsche Memberships zur Entfernung vorschlagen (REMOVE, mit Grund-Code)
- Redundanz zwischen Narratives erkennen und MERGE_PROPOSAL/SPLIT_PROPOSAL vorschlagen
- Status-Änderungen vorschlagen (STATUS_CHANGE: emerging/active/mature/fading/dormant),
  aber NUR wenn du sie aus den mitgelieferten quantitativen Scores ableitest, nicht frei erfunden

Du DARFST NICHT:
- technische Indikatoren, RS oder Basket-Scores selbst berechnen oder schätzen — nutze
  ausschließlich die dir mitgelieferten Zahlen
- Ticker erfinden, die dir nicht in den Daten mitgeteilt wurden
- direkt Dateien oder GitHub verändern — du lieferst nur den Change Set zurück
- eigenständig mergen oder die bestehende Taxonomie vollständig ersetzen
- schwache Kursperformance als Grund für ein REMOVE angeben — die einzig gültigen
  REMOVE-Gründe sind: misclassification, business_model_change, narrative_exposure_ended,
  m_and_a, delisting, ticker_not_found, clerical_error, other_fundamental (mit Begründung)
- role "speculative" vergeben — nur "core" oder "secondary" sind zulässig

Für core-Membership ist confidence >= 85 und ein klarer, wesentlicher Bezug zum Narrative
erforderlich. Für secondary genügt confidence >= 70 bei relevantem, aber nicht dominantem Bezug.
Für JEDE Änderung ist semantic_evidence (wirtschaftliche Begründung) und reason (Kurzfassung)
PFLICHT. CREATE-Vorschläge müssen quantitative_evidence.members (Liste mit ticker/role/
rs_percentile_1w/rs_percentile_1m), quantitative_evidence.breadth_pct und
quantitative_evidence.thrust enthalten — nutze dafür exakt die mitgelieferten Zahlen.

WICHTIG zu Zahlenfeldern (confidence, quantitative_evidence.thrust,
quantitative_evidence.breadth_pct, rs_percentile_1w, rs_percentile_1m): Diese Felder
MÜSSEN reine Zahlen sein (z.B. 0.42, 65, 90) — NIEMALS Beschreibungen, Sätze oder Text
wie "positive across all members". Wenn du für ein Zahlenfeld keinen konkreten Wert aus
den mitgelieferten Daten ableiten kannst, lass die betroffene Änderung lieber komplett weg,
statt einen Text anstelle einer Zahl einzusetzen — ein einziges falsches Zahlenfeld lässt
den kompletten Change Set fehlschlagen und blockiert auch alle anderen, an sich gültigen
Änderungen in diesem Review.
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_discovery_pool(market_features, taxonomy_tickers):
    tickers = market_features["tickers"]
    candidates = [
        {
            "ticker": sym,
            "rs_percentile_1w": t.get("rs_percentile_1w"),
            "rs_percentile_1m": t.get("rs_percentile_1m"),
            "thrust_1d": t.get("thrust_1d"),
            "thrust_1w": t.get("thrust_1w"),
            "sic_description": t.get("sic_description"),
        }
        for sym, t in tickers.items()
        if t.get("discovery_candidate") and sym not in taxonomy_tickers
    ]
    candidates.sort(key=lambda c: (c["rs_percentile_1m"] or 0) + (c["thrust_1w"] or 0), reverse=True)
    return candidates[:MAX_DISCOVERY_CANDIDATES_IN_PROMPT]


def build_maintenance_pool(market_features, taxonomy, max_members=MAX_MAINTENANCE_MEMBERS_IN_PROMPT):
    """Full-Universe spec point 17: the taxonomy is now ~1000+ tickers, so
    'all current members' can no longer mean literally every membership in
    one prompt (context limit, and it would swamp the signal). Sorted by
    last_reviewed_at ascending (missing = treated as oldest, i.e. reviewed
    first) and capped — the whole taxonomy rotates through weekly review
    over several runs instead of a single unbounded dump."""
    tickers = market_features["tickers"]
    rows = []
    for n in taxonomy["narratives"]:
        for sym, meta in n["tickers"].items():
            t = tickers.get(sym, {})
            rows.append({
                "narrative": n["id"],
                "ticker": sym,
                "role": meta.get("role"),
                "assignment_priority": meta.get("assignment_priority"),
                "last_reviewed_at": meta.get("last_reviewed_at"),
                "eligible": t.get("eligible"),
                "rs_percentile_1w": t.get("rs_percentile_1w"),
                "rs_percentile_1m": t.get("rs_percentile_1m"),
                "sic_description": t.get("sic_description"),
            })
    rows.sort(key=lambda r: r["last_reviewed_at"] or "")
    return rows[:max_members]


def build_user_prompt(taxonomy, discovery_pool, maintenance_pool, config,
                       near_duplicate_candidates=None, undersized_narratives=None,
                       low_confidence_memberships=None):
    taxonomy_summary = [
        {"id": n["id"], "name": n["name"], "status": n["status"], "member_count": len(n["tickers"]),
         "sample_members": sorted(n["tickers"].keys())[:10]}
        for n in taxonomy["narratives"]
    ]
    payload = {
        "existing_taxonomy": taxonomy_summary,
        "discovery_candidates": discovery_pool,
        "maintenance_members": maintenance_pool,
        "near_duplicate_candidates": near_duplicate_candidates or [],
        "undersized_narratives": undersized_narratives or [],
        "low_confidence_memberships": low_confidence_memberships or {},
        "config": config,
    }
    return (
        "Hier ist der aktuelle Stand für den wöchentlichen Review. "
        "existing_taxonomy = aktuelle Narrative + Mitglieder. "
        "discovery_candidates = Ticker, die momentan Momentum-Kriterien erfüllen und noch "
        "keiner Narrative angehören — prüfe, ob sie wirtschaftlich zu einem bestehenden oder "
        "neuen Narrative gehören. "
        "maintenance_members = alle aktuellen Mitglieder aller Narratives (auch schwache) — "
        "prüfe NUR auf sachliche Fehlzuordnung, Geschäftsmodelländerung, M&A, Delisting; "
        "ignoriere aktuelle Kursschwäche vollständig. "
        "near_duplicate_candidates/undersized_narratives/low_confidence_memberships = deterministisch "
        "vorberechnete Kandidaten fuer Merge/Split-Vorschlaege bzw. Qualitätspruefung (siehe System-Prompt). "
        "config = die geltenden Schwellenwerte, gegen die dein Vorschlag später deterministisch "
        "geprüft wird.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def apply_changes(taxonomy, annotated_changes, today):
    """Applies CREATE/ADD/actionable-REMOVE/STATUS_CHANGE in place.
    MERGE_PROPOSAL/SPLIT_PROPOSAL are intentionally never applied here."""
    by_id = {n["id"]: n for n in taxonomy["narratives"]}
    applied, skipped = [], []

    for c in annotated_changes:
        action = c["action"]
        if action == "CREATE":
            nid = c.get("narrative") or c["narrative_name"].lower().replace(" ", "_").replace("/", "_")
            if nid in by_id:
                skipped.append((c, "narrative id existiert bereits"))
                continue
            members = (c.get("quantitative_evidence") or {}).get("members", [])
            new_narrative = {
                "id": nid,
                "name": c["narrative_name"],
                "status": "emerging",
                "classification_hint": c.get("semantic_evidence"),
                "tickers": {
                    m["ticker"]: {
                        "role": m.get("role", "secondary"),
                        "confidence": m.get("confidence", c.get("confidence", 70)),
                        "reason": c.get("reason"),
                        "added_at": today,
                        "last_reviewed_at": today,
                    } for m in members if m.get("ticker")
                },
            }
            taxonomy["narratives"].append(new_narrative)
            by_id[nid] = new_narrative
            applied.append(c)

        elif action == "ADD":
            n = by_id.get(c.get("narrative"))
            if n is None:
                skipped.append((c, "Ziel-Narrative existiert nicht"))
                continue
            n["tickers"][c["ticker"]] = {
                "role": c["role"],
                "confidence": c["confidence"],
                "reason": c.get("reason"),
                "added_at": today,
                "last_reviewed_at": today,
            }
            applied.append(c)

        elif action == "REMOVE":
            n = by_id.get(c.get("narrative"))
            if n is None or not c.get("actionable"):
                skipped.append((c, "nicht anwendbar (Ziel fehlt oder zweite Review-Bestätigung steht aus)"))
                continue
            n["tickers"].pop(c["ticker"], None)
            applied.append(c)

        elif action == "STATUS_CHANGE":
            n = by_id.get(c.get("narrative"))
            if n is None:
                skipped.append((c, "Ziel-Narrative existiert nicht"))
                continue
            n["status"] = (c.get("proposed_state") or {}).get("status")
            applied.append(c)

        else:  # MERGE_PROPOSAL / SPLIT_PROPOSAL
            skipped.append((c, "Merge/Split werden in V1 nie automatisch angewendet — nur vorgeschlagen"))

    return taxonomy, applied, skipped


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Weekly AI Narrative Review")
    parser.add_argument("--taxonomy", default="data/taxonomy/narratives.json")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--proposals-dir", default="data/taxonomy/proposals")
    parser.add_argument("--history-dir", default="data/taxonomy/history")
    parser.add_argument("--dry-run", action="store_true", help="Call the LLM + validate, but do not write/apply anything")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"🤖 YOLO Dashboard — Weekly AI Narrative Review ({today})")
    print("=" * 60)

    taxonomy = load_json(args.taxonomy)
    market_features = load_json(args.market_features)
    config = load_json(args.config)

    taxonomy_tickers = {sym for n in taxonomy["narratives"] for sym in n["tickers"]}
    discovery_pool = build_discovery_pool(market_features, taxonomy_tickers)
    maintenance_pool = build_maintenance_pool(market_features, taxonomy)
    print(f"  → Discovery-Pool: {len(discovery_pool)} Kandidaten | Maintenance-Pool: {len(maintenance_pool)} Mitglieder")

    near_dup_threshold = config["classification"].get("near_duplicate_similarity_threshold", 0.5)
    near_duplicate_candidates = find_near_duplicate_candidates(taxonomy, near_dup_threshold)
    eligible_now = compute_eligible_set(market_features)
    min_active = config["classification"].get("minimum_active_narrative_members", 5)
    active_stats = compute_active_undersized_stats(taxonomy, eligible_now, min_active)
    low_confidence_memberships = find_low_confidence_memberships(taxonomy, config["membership"])
    print(f"  → Near-Duplicate Candidates: {len(near_duplicate_candidates)} | "
          f"Undersized Narratives: {active_stats['undersized_narrative_count']} | "
          f"Low-Confidence Memberships: primary<min={len(low_confidence_memberships['primary_below_minimum'])} "
          f"secondary<min={len(low_confidence_memberships['secondary_below_minimum'])}")

    user_prompt = build_user_prompt(taxonomy, discovery_pool, maintenance_pool, config,
                                     near_duplicate_candidates, active_stats["undersized_narratives"],
                                     low_confidence_memberships)

    try:
        raw_changes = llm_provider.generate_proposal_changes(SYSTEM_PROMPT, user_prompt)
    except llm_provider.LLMError as e:
        print(f"\n❌ LLM-Aufruf fehlgeschlagen: {e}", file=sys.stderr)
        print("   Workflow gestoppt — Taxonomie bleibt unverändert.", file=sys.stderr)
        sys.exit(1)

    print(f"  → LLM lieferte {len(raw_changes)} vorgeschlagene Änderungen")

    proposal = {
        "schema_version": 1,
        "created_at": today,
        "review_type": "weekly",
        "changes": raw_changes,
    }

    known_tickers = market_features["tickers"]
    proposal_history = []
    proposals_dir = Path(args.proposals_dir)
    if proposals_dir.exists():
        for p in sorted(proposals_dir.glob("*.json")):
            proposal_history.append((load_json(p).get("created_at", p.stem), load_json(p)))
    proposal_history.sort(key=lambda x: x[0])

    result = validate_proposal(proposal, taxonomy, config, known_tickers, proposal_history)
    for w in result.warnings:
        print(f"  ⚠ {w}")
    for e in result.errors:
        print(f"  ❌ {e}")

    proposal["changes"] = result.annotated
    proposal["validation"] = {"errors": result.errors, "warnings": result.warnings}

    proposals_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposals_dir / f"{today}.json"
    if not args.dry_run:
        with open(proposal_path, "w", encoding="utf-8") as f:
            json.dump(proposal, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Proposal geschrieben → {proposal_path}")

    if not result.ok:
        print(f"\n❌ Validierung fehlgeschlagen ({len(result.errors)} Fehler). "
              "Taxonomie wird NICHT verändert, kein PR wird erstellt.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("\n(dry-run: keine Taxonomie-Änderungen angewendet)")
        return

    updated_taxonomy, applied, skipped = apply_changes(taxonomy, result.annotated, today)
    print(f"\n  → {len(applied)} Änderungen angewendet, {len(skipped)} zurückgestellt "
          "(Merge/Split-Vorschläge, unbestätigte REMOVEs, o.ä.)")

    with open(args.taxonomy, "w", encoding="utf-8") as f:
        json.dump(updated_taxonomy, f, indent=2, ensure_ascii=False)
    print(f"✅ Taxonomie aktualisiert → {args.taxonomy}")

    history_dir = Path(args.history_dir)
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"{today}.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(updated_taxonomy, f, indent=2, ensure_ascii=False)
    print(f"✅ History-Snapshot geschrieben → {history_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
