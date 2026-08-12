#!/usr/bin/env python3
"""
YOLO Dashboard — Narrative Universe Reconciliation
Full-Universe Narrative Engine, Full-Universe-Spec (see project brief).

Keeps the persistent semantic taxonomy (data/taxonomy/narratives.json) in
sync with the current eligible YOLO universe (data/market_features.json),
so that EVERY currently eligible ticker carries exactly one Primary Narrative
and up to `classification.max_secondary_narratives` Secondary Narratives —
not just the historically curated Discovery subset.

Two modes, one script (spec point 3/6/20):

  --full-universe   One-time (or manually re-triggered) bulk classification
                     of every currently eligible ticker that has NEVER been
                     classified before. Batches of `classification.batch_size`
                     tickers per LLM call, growing catalog (a narrative
                     created in batch 3 is visible to batch 4+, which is the
                     main defense against near-duplicate narrative creation
                     across batches — see module docstring further down for
                     why this replaces the spec's suggested separate
                     "Pass B consolidation" LLM call with a simpler,
                     equally-robust alternative).

  (default)          Daily incremental reconciliation: classifies only
                     tickers that are eligible today and have NEVER been
                     classified before (true first-time entries). Re-entries
                     (previously classified, temporarily ineligible, eligible
                     again) reuse their stored classification — zero LLM
                     calls. Exits flip `active_eligible=false` on their
                     memberships without touching the classification itself
                     (point 7: exiting the eligible universe is not a
                     semantic REMOVE).

Semantic classification NEVER sees momentum/RS/Thrust/price-performance data
(point 16) — only ticker, company name, SIC code/description, and company
description (already available in the existing Massive reference caches,
zero extra API calls), plus the existing narrative catalog.

Coverage is a hard requirement (point 15): if even one eligible ticker ends
the run without exactly one Primary Narrative, NOTHING is written to
data/taxonomy/narratives.json — the job exits non-zero. A per-batch
checkpoint under .cache/ (mirroring the price-history cache's git-ignored,
actions/cache-backed pattern) means a failed run doesn't have to re-pay for
already-successful LLM batches on retry (point 21).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import llm_provider  # noqa: E402

DESCRIPTION_MAX_CHARS = 320
CHECKPOINT_PATH = ".cache/narrative_classification_checkpoint.jsonl"


# ─────────────────────────────────────────────
# I/O HELPERS
# ─────────────────────────────────────────────

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path, data):
    """Write via a temp file + rename so a crash mid-write never leaves a
    truncated/corrupt taxonomy on disk — filesystem-level belt-and-braces on
    top of the in-memory coverage gate (point 15)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(p)


def load_checkpoint(path):
    """{ticker: raw_classification_dict}, or {} if no checkpoint exists yet."""
    p = Path(path)
    if not p.exists():
        return {}
    out = {}
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = row.get("ticker")
            if t:
                out[t] = row
    return out


def append_checkpoint(path, results):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def clear_checkpoint(path):
    Path(path).unlink(missing_ok=True)


# ─────────────────────────────────────────────
# PURE: universe / taxonomy bookkeeping
# ─────────────────────────────────────────────

def _as_number(v):
    """Same tolerant-parsing rule as validate_narrative_proposal.py's
    _as_number: an LLM tool call occasionally stringifies a number even
    though the schema declares it numeric — treat non-numeric as missing
    rather than crashing the whole batch on one bad field."""
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


def compute_eligible_set(market_features):
    return {sym for sym, t in market_features.get("tickers", {}).items() if t.get("eligible")}


def compute_classified_tickers(taxonomy):
    """Every ticker that appears in ANY narrative's tickers dict, regardless
    of current eligibility — the taxonomy is a persistent semantic memory
    (point 6), classification survives eligibility exits."""
    return {sym for n in taxonomy.get("narratives", []) for sym in n.get("tickers", {})}


def ticker_memberships(taxonomy):
    """{ticker: [(narrative_dict, membership_dict), ...]} — a ticker can
    appear in multiple narratives (1 primary + up to N secondary)."""
    out = defaultdict(list)
    for n in taxonomy.get("narratives", []):
        for sym, meta in n.get("tickers", {}).items():
            out[sym].append((n, meta))
    return out


def ticker_was_active(memberships_for_ticker):
    """A ticker's active_eligible is a fact about the TICKER, not about any
    one narrative — all of its memberships are kept in sync, so reading any
    one of them (first, arbitrarily) is representative. None (not yet
    migrated / brand new) counts as not-previously-active."""
    if not memberships_for_ticker:
        return False
    return bool(memberships_for_ticker[0][1].get("active_eligible"))


def compute_universe_changes(taxonomy, eligible_now):
    """Returns {"stayed": [...], "entered": [...], "reentered": [...],
    "exited": [...]} (point 6), each a sorted list of ticker symbols.
    Computed from each membership's CURRENTLY STORED active_eligible flag
    (i.e. "as of the previous run") before that flag gets updated for today
    — callers must call this BEFORE mark_active_eligible()."""
    classified = compute_classified_tickers(taxonomy)
    memberships = ticker_memberships(taxonomy)
    previously_active = {t for t in classified if ticker_was_active(memberships[t])}

    stayed, entered, reentered, exited = [], [], [], []
    for t in sorted(eligible_now):
        if t in previously_active:
            stayed.append(t)
        elif t in classified:
            reentered.append(t)
        else:
            entered.append(t)
    for t in sorted(previously_active - eligible_now):
        exited.append(t)

    return {"stayed": stayed, "entered": entered, "reentered": reentered, "exited": exited}


def mark_active_eligible(taxonomy, eligible_now):
    """Mechanical, LLM-free bookkeeping (point 7): every membership's
    active_eligible flag is set to whether ITS ticker is in eligible_now
    today. Never touches which narratives a ticker belongs to — exiting the
    eligible universe is never a semantic REMOVE."""
    for n in taxonomy.get("narratives", []):
        for sym, meta in n.get("tickers", {}).items():
            meta["active_eligible"] = sym in eligible_now


def migrate_membership_schema(taxonomy, today):
    """Idempotent, additive schema migration (point 22) — adds
    assignment_priority/classification_source/classification_version to
    legacy memberships that predate the Full-Universe engine, without
    touching role/confidence/reason/added_at/last_reviewed_at. Entries that
    already carry assignment_priority are left completely untouched (so
    re-running this on an already-migrated taxonomy is a no-op). Returns the
    number of memberships migrated.

    Primary/secondary backfill for a legacy ticker with multiple
    memberships: the membership with role=='core' wins (ties broken by
    higher confidence, then alphabetically-first narrative id for
    determinism); everything else becomes secondary. Any legacy ticker with
    more than 1 + max_secondary_narratives memberships would need manual
    review post-migration (not observed in the current taxonomy, but
    checked for and reported rather than silently truncated)."""
    memberships = ticker_memberships(taxonomy)
    migrated = 0
    overflow = []

    for sym, entries in memberships.items():
        unmigrated = [(n, m) for n, m in entries if "assignment_priority" not in m]
        if not unmigrated:
            continue

        already_primary = [(n, m) for n, m in entries if m.get("assignment_priority") == "primary"]
        if already_primary:
            primary_pick = already_primary[0]
        else:
            ranked = sorted(
                entries,
                key=lambda nm: (0 if nm[1].get("role") == "core" else 1, -(nm[1].get("confidence") or 0), nm[0]["id"]),
            )
            primary_pick = ranked[0]

        for n, m in unmigrated:
            m["classification_source"] = m.get("classification_source", "manual_migration")
            m["classification_version"] = m.get("classification_version", 1)
            if (n, m) is primary_pick or (n["id"] == primary_pick[0]["id"] and m is primary_pick[1]):
                m["assignment_priority"] = "primary"
            else:
                m["assignment_priority"] = "secondary"
            migrated += 1

        secondary_count = sum(1 for n, m in entries if m.get("assignment_priority") == "secondary")
        if secondary_count > 2:
            overflow.append((sym, secondary_count))

    if overflow:
        print(f"  ⚠ Migration: {len(overflow)} Ticker mit >2 Secondary-Memberships nach Backfill "
              f"(manuelle Nachpruefung empfohlen): {overflow}", file=sys.stderr)
    return migrated


# ─────────────────────────────────────────────
# PURE: narrative catalog / dedup
# ─────────────────────────────────────────────

def normalize_narrative_name(name):
    """Lowercase, alnum-only comparison key — catches 'AI Infrastructure' vs
    'AI-Infrastructure' vs 'ai infrastructure' as the same narrative without
    a second LLM call (point 4: 'keine nahezu identischen Narrative')."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def build_catalog_index(taxonomy):
    """{narrative_id: {"name":..., "classification_hint":...}} plus a
    {normalized_name: narrative_id} reverse index for dedup matching. Both
    are mutated in place as new narratives are created within a run, so a
    growing catalog is visible to later batches in the same run."""
    by_id = {n["id"]: {"name": n["name"], "classification_hint": n.get("classification_hint")}
              for n in taxonomy.get("narratives", [])}
    by_norm_name = {normalize_narrative_name(v["name"]): nid for nid, v in by_id.items()}
    return by_id, by_norm_name


def slugify_narrative_id(name):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "narrative"


def unique_narrative_id(base_slug, taken_ids):
    if base_slug not in taken_ids:
        return base_slug
    i = 2
    while f"{base_slug}_{i}" in taken_ids:
        i += 1
    return f"{base_slug}_{i}"


def resolve_narrative_reference(narrative_id, is_new, new_name, new_definition, catalog_by_id, catalog_by_norm_name):
    """Resolves one LLM-proposed narrative reference (primary or secondary)
    against the (possibly already-grown-this-run) catalog. Returns
    (resolved_id, created_new: bool, error: str|None).

    Dedup happens BEFORE trusting is_new — even if the LLM claims a
    narrative is new, a near-identical name already in the catalog wins
    (case/punctuation-insensitive match), so two batches proposing
    "Regional Banks" and "regional-banks" land on the same id.

    An unresolvable reference (narrative_id not in the catalog, is_new not
    set) does NOT error out — observed in the first real Full-Universe run:
    the LLM occasionally echoes a catalog id back slightly mangled/
    abbreviated (e.g. 'healthcare_services' for the catalog's
    'healthcare_services_facilities') without flagging it as new. That is a
    confident classification with an id-matching slip, not a missing
    classification (point 15's fail-closed rule is about NO classification
    being produced, not about exact id fidelity) — so it's healed as an
    implicit new narrative using the reference itself as a fallback name.
    Any resulting near-duplicate is still caught by
    find_duplicate_narrative_names for human follow-up in the weekly
    review, same safety net as a genuine LLM-proposed new narrative."""
    if narrative_id and narrative_id in catalog_by_id and not is_new:
        return narrative_id, False, None

    name_for_match = new_name if is_new else narrative_id
    norm = normalize_narrative_name(name_for_match)
    existing = catalog_by_norm_name.get(norm)
    if existing:
        return existing, False, None

    if narrative_id in catalog_by_id:
        return narrative_id, False, None

    fallback_name = new_name or (narrative_id.replace("_", " ").replace("-", " ").strip().title() if narrative_id else None)
    if not fallback_name:
        return None, False, "weder narrative_id noch new_name vorhanden — keine Klassifikation ableitbar"

    new_id = unique_narrative_id(slugify_narrative_id(fallback_name), set(catalog_by_id.keys()))
    catalog_by_id[new_id] = {"name": fallback_name, "classification_hint": new_definition}
    catalog_by_norm_name[normalize_narrative_name(fallback_name)] = new_id
    return new_id, True, None


# ─────────────────────────────────────────────
# PURE: prompt construction
# ─────────────────────────────────────────────

CLASSIFICATION_SYSTEM_PROMPT = """Du bist der semantische Klassifikations-Assistent eines Trading-Dashboards.

Deine einzige Aufgabe: jedem gegebenen Ticker genau EIN Primary Narrative und optional bis zu
zwei Secondary Narratives zuordnen — rein wirtschaftlich/geschaeftsmodellseitig.

Du bekommst NUR: Ticker, Company Name, SIC Code, SIC Description, Company Description, sowie den
bestehenden Narrative-Katalog. Du bekommst KEINE Kurs-, RS-, Thrust- oder Performance-Daten und
darfst solche Informationen auch nicht "erraten" oder in deine Begruendung einfliessen lassen.

WICHTIG:
- Primary Narrative = das wirtschaftlich wichtigste/praeziseste Narrative des Unternehmens.
- Secondary Narrative NUR bei echtem, materiellem zusaetzlichem wirtschaftlichem Exposure
  (nicht: "AI" kommt irgendwo in der Beschreibung vor).
- Bevorzuge IMMER ein bestehendes Narrative aus dem Katalog, wenn es wirtschaftlich passt.
- Erzeuge ein neues Narrative NUR wenn wirklich kein bestehendes passt. Pruefe vorher, ob es nicht
  bereits ein nahezu identisches Narrative im Katalog gibt (auch mit leicht anderem Namen) -- nutze
  in dem Fall das bestehende statt ein Duplikat zu erzeugen.
- Neue Narrative brauchen einen kurzen, klaren wirtschaftlichen Definitionssatz.
- Keine riesige "Other"/Sonstiges-Restkategorie -- wenn ein Unternehmen wirklich in keine bestehende
  Kategorie passt, ist ein praezises neues Narrative besser als eine Sammelkategorie.
- confidence-Felder muessen reine Zahlen 0-100 sein, niemals Text.
"""


def build_classification_context(tickers, types_ref, reference_cache, desc_max_chars=DESCRIPTION_MAX_CHARS):
    """Per-ticker fundamental/semantic context — deliberately excludes every
    momentum/RS/Thrust/price field (point 16). types_ref/reference_cache are
    the existing Massive reference caches (build_market_reference.py);
    zero additional API calls needed."""
    types_tickers = types_ref.get("tickers", {})
    ref_tickers = reference_cache.get("tickers", {})
    out = []
    for t in tickers:
        type_meta = types_tickers.get(t, {})
        ref_meta = ref_tickers.get(t, {})
        desc = (ref_meta.get("description") or "")[:desc_max_chars]
        out.append({
            "ticker": t,
            "name": type_meta.get("name"),
            "sic_code": ref_meta.get("sic_code"),
            "sic_description": ref_meta.get("sic_description"),
            "description": desc,
        })
    return out


def build_classification_user_prompt(batch_context, catalog_by_id, max_secondary):
    catalog_summary = [{"id": nid, "name": v["name"], "definition": v.get("classification_hint")}
                        for nid, v in sorted(catalog_by_id.items())]
    payload = {
        "existing_narrative_catalog": catalog_summary,
        "max_secondary_narratives": max_secondary,
        "tickers_to_classify": batch_context,
    }
    return (
        "Klassifiziere jeden Ticker unten in existing_narrative_catalog (bevorzugt) oder ein neues "
        "Narrative, falls wirklich nichts passt. max_secondary_narratives begrenzt, wie viele "
        "Secondary-Narratives du je Ticker vorschlagen darfst (0 ist erlaubt und oft richtig).\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ─────────────────────────────────────────────
# Batch classification (parsing is pure; the LLM call itself is not)
# ─────────────────────────────────────────────

def parse_classification_result(raw, valid_tickers):
    """Validates/cleans one raw classification dict from the LLM tool call.
    Returns (cleaned_dict, error: str|None) — never raises; a malformed
    single result must not crash the whole batch (point 15's fail-closed
    rule applies at the coverage-check level, not by exploding mid-parse)."""
    ticker = raw.get("ticker")
    if ticker not in valid_tickers:
        return None, f"unbekannter oder fehlender ticker in Klassifikationsantwort: {raw.get('ticker')!r}"
    primary_id = raw.get("primary_narrative_id")
    if not primary_id:
        return None, f"{ticker}: primary_narrative_id fehlt"
    primary_confidence = _as_number(raw.get("primary_confidence"))
    if primary_confidence is None:
        return None, f"{ticker}: primary_confidence ist keine gueltige Zahl"
    # NOTE: primary_is_new=true with a missing primary_new_name is
    # deliberately NOT rejected here — resolve_narrative_reference derives a
    # fallback name from primary_narrative_id itself (point 15 is about a
    # ticker ending up with NO classification, not about the LLM's is_new/
    # new_name bookkeeping being perfectly filled in every field).

    secondaries = []
    for s in (raw.get("secondary_narratives") or []):
        conf = _as_number(s.get("confidence"))
        nid = s.get("narrative_id")
        if not nid or conf is None:
            continue  # skip malformed secondary entries individually, don't fail the whole ticker
        secondaries.append({
            "narrative_id": nid, "is_new": bool(s.get("is_new")),
            "new_name": s.get("new_name"), "new_definition": s.get("new_definition"),
            "confidence": conf,
        })

    return {
        "ticker": ticker,
        "primary_narrative_id": primary_id,
        "primary_is_new": bool(raw.get("primary_is_new")),
        "primary_new_name": raw.get("primary_new_name"),
        "primary_new_definition": raw.get("primary_new_definition"),
        "primary_confidence": primary_confidence,
        "secondary_narratives": secondaries,
        "reasoning": raw.get("reasoning", ""),
    }, None


def apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm_name, result,
                                 membership_cfg, max_secondary, today, source):
    """Materializes one resolved classification into the taxonomy's
    narrative->tickers dicts (creating new narrative entries as needed).
    Returns a list of error strings (empty on full success)."""
    errors = []
    ticker = result["ticker"]

    primary_id, created, err = resolve_narrative_reference(
        result["primary_narrative_id"], result["primary_is_new"],
        result["primary_new_name"], result["primary_new_definition"],
        catalog_by_id, catalog_by_norm_name)
    if err:
        errors.append(f"{ticker}: {err}")
        return errors
    if created:
        taxonomy_by_id[primary_id] = {
            "id": primary_id, "name": catalog_by_id[primary_id]["name"], "status": "emerging",
            "classification_hint": result["primary_new_definition"],
            "created_at": today, "created_reason": f"Full-Universe classification ({source})",
            "tickers": {},
        }

    role = "core" if result["primary_confidence"] >= membership_cfg["core_confidence_minimum"] else "secondary"
    taxonomy_by_id[primary_id]["tickers"][ticker] = {
        "role": role, "assignment_priority": "primary",
        "confidence": result["primary_confidence"], "reason": result["reasoning"],
        "added_at": today, "last_reviewed_at": today, "active_eligible": True,
        "classification_source": source, "classification_version": 1,
    }

    accepted_secondary = 0
    for s in result["secondary_narratives"]:
        if accepted_secondary >= max_secondary:
            break
        sec_id, sec_created, sec_err = resolve_narrative_reference(
            s["narrative_id"], s["is_new"], s["new_name"], s["new_definition"],
            catalog_by_id, catalog_by_norm_name)
        if sec_err:
            errors.append(f"{ticker} (secondary): {sec_err}")
            continue
        if sec_id == primary_id:
            continue  # a narrative can't be both primary and secondary for the same ticker
        if sec_created:
            taxonomy_by_id[sec_id] = {
                "id": sec_id, "name": catalog_by_id[sec_id]["name"], "status": "emerging",
                "classification_hint": s["new_definition"],
                "created_at": today, "created_reason": f"Full-Universe classification ({source})",
                "tickers": {},
            }
        sec_role = "core" if s["confidence"] >= membership_cfg["core_confidence_minimum"] else "secondary"
        taxonomy_by_id[sec_id]["tickers"][ticker] = {
            "role": sec_role, "assignment_priority": "secondary",
            "confidence": s["confidence"], "reason": result["reasoning"],
            "added_at": today, "last_reviewed_at": today, "active_eligible": True,
            "classification_source": source, "classification_version": 1,
        }
        accepted_secondary += 1

    return errors


# ─────────────────────────────────────────────
# Coverage validation (the fail-closed gate)
# ─────────────────────────────────────────────

def validate_full_coverage(taxonomy, eligible_now, max_secondary):
    """Hard requirement (point 15): every ticker in eligible_now must have
    EXACTLY one membership with assignment_priority=='primary' somewhere in
    the taxonomy, and at most max_secondary 'secondary' memberships. Returns
    a list of human-readable error strings; empty = coverage OK."""
    memberships = ticker_memberships(taxonomy)
    errors = []
    for t in sorted(eligible_now):
        entries = memberships.get(t, [])
        primaries = [m for n, m in entries if m.get("assignment_priority") == "primary"]
        secondaries = [m for n, m in entries if m.get("assignment_priority") == "secondary"]
        if len(primaries) == 0:
            errors.append(f"{t}: kein Primary Narrative")
        elif len(primaries) > 1:
            errors.append(f"{t}: {len(primaries)} Primary Narratives (muss genau 1 sein)")
        if len(secondaries) > max_secondary:
            errors.append(f"{t}: {len(secondaries)} Secondary Narratives (max {max_secondary})")
    return errors


def find_duplicate_narrative_names(taxonomy):
    """Post-hoc detection (not auto-merge) of near-identical narrative names
    — surfaced in the audit report/sanity check for human review, per
    point 4's 'keine nahezu identischen Narrative' combined with the
    existing MERGE_PROPOSAL-stays-human-only rule (point 5/17)."""
    by_norm = defaultdict(list)
    for n in taxonomy.get("narratives", []):
        by_norm[normalize_narrative_name(n["name"])].append(n["id"])
    return {norm: ids for norm, ids in by_norm.items() if len(ids) > 1}


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def chunked(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def classify_tickers(tickers_to_classify, taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
                      types_ref, reference_cache, cfg, checkpoint_path, source, model=None):
    """Runs the batched LLM classification loop. Returns (all_errors: list,
    n_classified: int). Uses the checkpoint to skip tickers already
    successfully classified in a prior attempt within the same run/retry
    (point 21) — checkpointed raw results still go through the same
    parse/resolve/apply path as a fresh LLM response."""
    membership_cfg = cfg["membership"]
    max_secondary = cfg["classification"]["max_secondary_narratives"]
    batch_size = cfg["classification"]["batch_size"]

    checkpoint = load_checkpoint(checkpoint_path)
    still_needed = [t for t in tickers_to_classify if t not in checkpoint]
    n_reused = len(set(checkpoint) & set(tickers_to_classify))
    if n_reused:
        print(f"  → Checkpoint gefunden: {n_reused} bereits klassifizierte Ticker werden wiederverwendet, "
              "kein erneuter LLM-Call")

    all_errors = []
    n_classified = 0
    batches = list(chunked(still_needed, batch_size))
    n_batches = len(batches)

    # Replay checkpointed results first (already-successful earlier batches).
    checkpointed_for_this_run = [checkpoint[t] for t in tickers_to_classify if t in checkpoint]
    for raw in checkpointed_for_this_run:
        result, err = parse_classification_result(raw, set(tickers_to_classify))
        if err:
            all_errors.append(f"[checkpoint] {err}")
            continue
        errs = apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
                                            result, membership_cfg, max_secondary,
                                            datetime.now(timezone.utc).strftime("%Y-%m-%d"), source)
        all_errors.extend(errs)
        n_classified += 1

    for i, batch in enumerate(batches, 1):
        print(f"  → Batch {i}/{n_batches} ({len(batch)} Ticker)...")
        context = build_classification_context(batch, types_ref, reference_cache)
        user_prompt = build_classification_user_prompt(context, catalog_by_id, max_secondary)

        raw_results = None
        last_err = None
        for attempt in range(2):
            try:
                raw_results = llm_provider.generate_ticker_classifications(
                    CLASSIFICATION_SYSTEM_PROMPT, user_prompt, model=model)
                break
            except llm_provider.LLMError as e:
                last_err = e
                print(f"    ⚠ Batch {i} Versuch {attempt + 1} fehlgeschlagen: {e}", file=sys.stderr)
        if raw_results is None:
            all_errors.append(f"Batch {i} ({batch[0]}..{batch[-1]}): LLM-Aufruf endgueltig fehlgeschlagen: {last_err}")
            continue

        append_checkpoint(checkpoint_path, raw_results)

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for raw in raw_results:
            result, err = parse_classification_result(raw, set(batch))
            if err:
                all_errors.append(err)
                continue
            errs = apply_classification_result(taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
                                                result, membership_cfg, max_secondary, today, source)
            all_errors.extend(errs)
            n_classified += 1

    return all_errors, n_classified


def build_audit_state(as_of_date, eligible_now, classified_eligible_count, changes,
                       narratives_before, narratives_after, classification_errors):
    coverage_pct = round(classified_eligible_count / len(eligible_now) * 100, 1) if eligible_now else 100.0
    return {
        "as_of_date": as_of_date,
        "eligible_count": len(eligible_now),
        "classified_eligible_count": classified_eligible_count,
        "coverage_pct": coverage_pct,
        "entered": changes["entered"],
        "reentered": changes["reentered"],
        "exited": changes["exited"],
        "newly_classified": sorted(set(changes["entered"])),
        "new_narratives_created": sorted(set(narratives_after) - set(narratives_before)),
        "classification_errors": classification_errors,
    }


def print_sanity_report(eligible_now, changes, taxonomy_before_ids, taxonomy_after,
                          classification_errors, sample_tickers):
    memberships = ticker_memberships(taxonomy_after)
    classified_now = compute_classified_tickers(taxonomy_after)

    print("\n" + "=" * 60)
    print("🔍 FULL-UNIVERSE NARRATIVE ENGINE — SANITY REPORT")
    print("=" * 60)

    print(f"\nELIGIBLE UNIVERSE: {len(eligible_now)}")

    classified_eligible = eligible_now & classified_now
    coverage_pct = round(len(classified_eligible) / len(eligible_now) * 100, 1) if eligible_now else 100.0
    primary_n = sum(1 for t in eligible_now if any(m.get("assignment_priority") == "primary" for _, m in memberships[t]))
    secondary_n = sum(len([m for _, m in memberships[t] if m.get("assignment_priority") == "secondary"]) for t in eligible_now)
    print(f"CLASSIFICATION: classified={len(classified_eligible)} | coverage={coverage_pct}% | "
          f"primary_assignments={primary_n} | secondary_assignments={secondary_n}")

    print(f"UNIVERSE CHANGES: stayed={len(changes['stayed'])} | entered={len(changes['entered'])} | "
          f"reentered={len(changes['reentered'])} | exited={len(changes['exited'])}")

    n_ids_after = {n["id"] for n in taxonomy_after.get("narratives", [])}
    created = sorted(n_ids_after - taxonomy_before_ids)
    print(f"NARRATIVES: before={len(taxonomy_before_ids)} | after={len(n_ids_after)} | neu erzeugt={len(created)}")
    if created:
        print(f"  neue Narratives: {created}")

    ranked = sorted(taxonomy_after.get("narratives", []),
                     key=lambda n: -sum(1 for sym in n["tickers"] if sym in eligible_now))
    print("  Top 20 nach eligible Members:")
    for n in ranked[:20]:
        n_elig = sum(1 for sym in n["tickers"] if sym in eligible_now)
        print(f"    {n['name']:<35} {n_elig}")

    dupes = find_duplicate_narrative_names(taxonomy_after)
    no_primary = [t for t in eligible_now if not any(m.get("assignment_priority") == "primary" for _, m in memberships[t])]
    too_many_secondary = [t for t in eligible_now
                           if len([m for _, m in memberships[t] if m.get("assignment_priority") == "secondary"]) > 2]
    empty_narratives = [n["id"] for n in taxonomy_after.get("narratives", []) if not n["tickers"]]
    print(f"\nQUALITY CHECK: ohne Primary={len(no_primary)} | >2 Secondary={len(too_many_secondary)} | "
          f"leere Narratives={len(empty_narratives)} | Duplicate-Name-Cluster={len(dupes)} | "
          f"Klassifikationsfehler={len(classification_errors)}")
    if dupes:
        print(f"  Duplicate-Name-Cluster (Merge pruefen): {dupes}")

    print("\nSTICHPROBEN:")
    for t in sample_tickers:
        entries = memberships.get(t)
        if not entries:
            print(f"  {t}: NICHT KLASSIFIZIERT")
            continue
        primary = next((n["name"] for n, m in entries if m.get("assignment_priority") == "primary"), "?")
        secondary = [n["name"] for n, m in entries if m.get("assignment_priority") == "secondary"]
        reason = next((m.get("reason") for n, m in entries if m.get("assignment_priority") == "primary"), "")
        print(f"  {t}: Primary={primary} | Secondary={secondary or '-'}")
        if reason:
            print(f"       Begruendung: {reason[:180]}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="YOLO Dashboard Narrative Universe Reconciliation")
    parser.add_argument("--market-features", default="data/market_features.json")
    parser.add_argument("--taxonomy", default="data/taxonomy/narratives.json")
    parser.add_argument("--config", default="config/narrative_engine.json")
    parser.add_argument("--types-ref", default="data/taxonomy/market_reference_types.json")
    parser.add_argument("--reference-cache", default="data/taxonomy/market_reference_cache.json")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--full-universe", action="store_true",
                         help="Classify EVERY currently-unclassified eligible ticker, not just today's new entries")
    parser.add_argument("--batch-size", type=int, default=None, help="Override config.classification.batch_size")
    parser.add_argument("--dry-run", action="store_true", help="Compute + report, never write the taxonomy")
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print("=" * 60)
    print(f"🧭 YOLO Dashboard — Narrative Universe Reconciliation ({today})"
          f"{' [FULL-UNIVERSE]' if args.full_universe else ' [daily incremental]'}")
    print("=" * 60)

    market_features = load_json(args.market_features)
    taxonomy = load_json(args.taxonomy)
    cfg = load_json(args.config)
    types_ref = load_json(args.types_ref)
    reference_cache = load_json(args.reference_cache)

    if args.batch_size:
        cfg["classification"]["batch_size"] = args.batch_size

    migrated = migrate_membership_schema(taxonomy, today)
    if migrated:
        print(f"  → Schema-Migration: {migrated} Legacy-Memberships mit assignment_priority nachgeruestet")

    eligible_now = compute_eligible_set(market_features)
    changes = compute_universe_changes(taxonomy, eligible_now)
    print(f"  → Eligible: {len(eligible_now)} | Stayed: {len(changes['stayed'])} | "
          f"Entered: {len(changes['entered'])} | Reentered: {len(changes['reentered'])} | Exited: {len(changes['exited'])}")

    classified = compute_classified_tickers(taxonomy)
    if args.full_universe:
        tickers_to_classify = sorted(eligible_now - classified)
        print(f"  → Full-Universe-Modus: {len(tickers_to_classify)} noch unklassifizierte eligible Ticker")
    else:
        tickers_to_classify = sorted(t for t in changes["entered"] if t not in classified)
        print(f"  → Inkrementeller Modus: {len(tickers_to_classify)} echte Neuzugaenge zu klassifizieren")

    narratives_before_ids = {n["id"] for n in taxonomy["narratives"]}
    catalog_by_id, catalog_by_norm_name = build_catalog_index(taxonomy)
    taxonomy_by_id = {n["id"]: n for n in taxonomy["narratives"]}

    classification_errors = []
    if tickers_to_classify:
        source = "full_universe_run" if args.full_universe else "daily_reconciliation"
        classification_errors, n_classified = classify_tickers(
            tickers_to_classify, taxonomy_by_id, catalog_by_id, catalog_by_norm_name,
            types_ref, reference_cache, cfg, args.checkpoint, source)
        print(f"  → {n_classified}/{len(tickers_to_classify)} Ticker klassifiziert, "
              f"{len(classification_errors)} Fehler")
        taxonomy["narratives"] = list(taxonomy_by_id.values())

    mark_active_eligible(taxonomy, eligible_now)

    coverage_errors = validate_full_coverage(taxonomy, eligible_now, cfg["classification"]["max_secondary_narratives"])
    all_errors = classification_errors + coverage_errors

    sample_tickers = [t for t in ["NVDA", "DELL", "MU", "SNDK", "CRWD", "FTNT"] if t in eligible_now]
    sample_tickers += sorted((eligible_now - set(sample_tickers)))[:3]

    if all_errors:
        print(f"\n❌ Coverage/Klassifikations-Fehler ({len(all_errors)}) — Taxonomie wird NICHT geschrieben:", file=sys.stderr)
        for e in all_errors[:50]:
            print(f"   - {e}", file=sys.stderr)
        if len(all_errors) > 50:
            print(f"   ... und {len(all_errors) - 50} weitere", file=sys.stderr)
        print_sanity_report(eligible_now, changes, narratives_before_ids, taxonomy, all_errors, sample_tickers)
        sys.exit(1)

    print_sanity_report(eligible_now, changes, narratives_before_ids, taxonomy, all_errors, sample_tickers)

    if args.dry_run:
        print("\n(dry-run: Taxonomie nicht geschrieben)")
        return

    save_json_atomic(args.taxonomy, taxonomy)
    print(f"\n✅ Taxonomie geschrieben → {args.taxonomy}")

    audit = build_audit_state(today, eligible_now, len(eligible_now & compute_classified_tickers(taxonomy)),
                               changes, narratives_before_ids, {n["id"] for n in taxonomy["narratives"]},
                               all_errors)
    audit_path = Path(args.out_dir) / "narrative_universe_state.json"
    save_json_atomic(audit_path, audit)
    print(f"✅ Audit-Report geschrieben → {audit_path}")

    clear_checkpoint(args.checkpoint)
    print("=" * 60)


if __name__ == "__main__":
    main()
